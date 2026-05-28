"""Triangular Fixed-Point Refinement (§2.2 of the paper).

Implements the Gauss-Seidel cycle M -> O -> L -> M that jointly refines
the Observable, Latent, and Memory state vectors before decoding:

    o^(k+1) = (1 - α_o) o^(k) + α_o · CA_{O<-M}(o^(k), m^(k))    (Eq. 1)
    l^(k+1) = (1 - α_l) l^(k) + α_l · CA_{L<-O}(l^(k), o^(k+1))  (Eq. 2)
    m^(k+1) = (1 - α_m) m^(k) + α_m · CA_{M<-L}(m^(k), l^(k+1))  (Eq. 3)

Each edge is a pairwise cross-attention (one query head -> KV bank).
We apply spectral normalization to the linear maps and clamp pre-softmax
logits so the resulting operator stays in the contraction regime of
Theorem 1 (Assumption 1(i)-(ii)). The Gauss-Seidel ordering ensures each
update consumes the most recent partner state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BridgeConfig


def _maybe_spectral(linear: nn.Linear, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(linear) if enabled else linear


class PairwiseCrossAttention(nn.Module):
    """CA_{A<-B}(a, b): query from space A, key/value from space B.

    Operates on single vectors per sample (a: [B, d_a], b: [B, d_b]) by
    treating both as length-1 sequences with `n_heads` heads. This is
    cheap (O(d^2) per step) and matches the latency budget cited in the
    paper (~6 ms/step on A100, constant in context length).
    """

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        n_heads: int,
        spectral: bool,
        logit_clamp: float,
    ):
        super().__init__()
        assert d_query % n_heads == 0, "n_heads must divide d_query"
        self.d_query = d_query
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.head_dim = d_query // n_heads
        self.logit_clamp = logit_clamp

        self.q_proj = _maybe_spectral(nn.Linear(d_query, d_query, bias=False), spectral)
        self.k_proj = _maybe_spectral(nn.Linear(d_kv, d_query, bias=False), spectral)
        self.v_proj = _maybe_spectral(nn.Linear(d_kv, d_query, bias=False), spectral)
        self.o_proj = _maybe_spectral(nn.Linear(d_query, d_query, bias=False), spectral)
        self.norm = nn.LayerNorm(d_query)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # query: [B, d_q], kv: [B, d_kv] -> output: [B, d_q]
        b = query.shape[0]
        q = self.q_proj(query).view(b, self.n_heads, 1, self.head_dim)
        k = self.k_proj(kv).view(b, self.n_heads, 1, self.head_dim)
        v = self.v_proj(kv).view(b, self.n_heads, 1, self.head_dim)
        scale = self.head_dim ** -0.5
        logits = (q @ k.transpose(-1, -2)) * scale
        # Bounded-logit operation (Assumption 1(ii)) keeps softmax Lipschitz.
        logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (attn @ v).reshape(b, self.d_query)
        return self.norm(self.o_proj(out))


@dataclass
class RefinementTrace:
    """Diagnostics emitted by the refinement loop (Theorem 1 audit)."""
    final_residual: torch.Tensor          # ||z^(K) - z^(K-1)||
    residual_trajectory: torch.Tensor     # [K] residuals across iterations
    contraction_proxy: torch.Tensor       # alpha-tilde (paper, Appendix)
    o: torch.Tensor
    l: torch.Tensor
    m: torch.Tensor


class TriangularFixedPointRefinement(nn.Module):
    """Iterative O-L-M refinement with K Gauss-Seidel steps."""

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        spec = config.spectral_norm_attention
        clamp = config.logit_clamp
        h = config.cross_attention_heads

        self.ca_o_from_m = PairwiseCrossAttention(config.d_o, config.d_m, h, spec, clamp)
        self.ca_l_from_o = PairwiseCrossAttention(config.d_l, config.d_o, h, spec, clamp)
        self.ca_m_from_l = PairwiseCrossAttention(config.d_m, config.d_l, h, spec, clamp)

    def forward(
        self,
        o0: torch.Tensor,
        l0: torch.Tensor,
        m0: torch.Tensor,
        K: int | None = None,
    ) -> RefinementTrace:
        cfg = self.config
        K = K if K is not None else cfg.K_refinement
        a_o, a_l, a_m = cfg.alpha_o, cfg.alpha_l, cfg.alpha_m

        o, l, m = o0, l0, m0
        residuals = []
        prev = torch.cat([o, l, m], dim=-1)

        for _ in range(K):
            o_new = (1 - a_o) * o + a_o * self.ca_o_from_m(o, m)         # Eq. 1
            l_new = (1 - a_l) * l + a_l * self.ca_l_from_o(l, o_new)     # Eq. 2
            m_new = (1 - a_m) * m + a_m * self.ca_m_from_l(m, l_new)     # Eq. 3

            curr = torch.cat([o_new, l_new, m_new], dim=-1)
            residuals.append((curr - prev).norm(dim=-1))  # [B]
            prev = curr
            o, l, m = o_new, l_new, m_new

        residual_traj = torch.stack(residuals, dim=-1)                  # [B, K]
        final_res = residual_traj[..., -1]
        # Contraction proxy alpha-tilde ~ ratio between consecutive residuals.
        if K >= 2:
            ratios = residual_traj[..., 1:] / residual_traj[..., :-1].clamp_min(1e-8)
            proxy = ratios.mean(dim=-1)
        else:
            proxy = torch.full_like(final_res, float("nan"))

        return RefinementTrace(
            final_residual=final_res,
            residual_trajectory=residual_traj,
            contraction_proxy=proxy,
            o=o, l=l, m=m,
        )
