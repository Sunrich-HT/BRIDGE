"""Dual-System Processing (§2.3 of the paper).

Produces the initial latent vector l^(0) consumed by triangular refinement,
plus a turn-level steering signal c_t that is injected into the frozen
backbone hidden states (Eq. 9).

  Seed latent     : l~_t = P_l ( Pool(H_t) )                         (Eq. 6)
  Fast (System 1) : h^(1) = Attn_fast( l~_t, K_habit, V_habit )
  Slow (System 2) : h^(2) = Attn_slow^(D) ∘ ... ∘ Attn_slow^(1)( l~_t )
  Axial fusion    : l^(0) = W_fuse [ h^(1); h^(2) ] + b_fuse          (Eq. 7)
  Multiplicative
  context gate    : g_t   = σ( W_g [l~; l^(0); l~ ⊙ l^(0)] + b_g )
                    c_t   = g_t ⊙ h^(1) + (1 - g_t) ⊙ h^(2)           (Eq. 8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BridgeConfig


def _maybe_spectral(linear: nn.Linear, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(linear) if enabled else linear


class HabitBank(nn.Module):
    """Learned (K, V) bank of low-latency persona habits queried by System-1."""

    def __init__(self, d: int, n_slots: int, spectral: bool):
        super().__init__()
        self.K = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.V = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.q_proj = _maybe_spectral(nn.Linear(d, d, bias=False), spectral)
        self.scale = d ** -0.5

    def forward(self, x: torch.Tensor, logit_clamp: float) -> torch.Tensor:
        q = self.q_proj(x)                                # [B, d]
        logits = (q @ self.K.t()) * self.scale            # [B, n_slots]
        logits = logits.clamp(-logit_clamp, logit_clamp)
        attn = torch.softmax(logits, dim=-1)
        return attn @ self.V                              # [B, d]


class _SlowLayer(nn.Module):
    def __init__(self, d: int, spectral: bool, logit_clamp: float, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q = _maybe_spectral(nn.Linear(d, d, bias=False), spectral)
        self.k = _maybe_spectral(nn.Linear(d, d, bias=False), spectral)
        self.v = _maybe_spectral(nn.Linear(d, d, bias=False), spectral)
        self.o = _maybe_spectral(nn.Linear(d, d, bias=False), spectral)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            _maybe_spectral(nn.Linear(d, d * 2, bias=False), spectral),
            nn.GELU(),
            _maybe_spectral(nn.Linear(d * 2, d, bias=False), spectral),
        )
        self.logit_clamp = logit_clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        b = h.shape[0]
        q = self.q(h).view(b, self.n_heads, 1, self.head_dim)
        k = self.k(h).view(b, self.n_heads, 1, self.head_dim)
        v = self.v(h).view(b, self.n_heads, 1, self.head_dim)
        logits = (q @ k.transpose(-1, -2)) * self.head_dim ** -0.5
        logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        attn = torch.softmax(logits, dim=-1)
        out = (attn @ v).reshape(b, -1)
        x = x + self.o(out)
        x = x + self.ff(self.norm2(x))
        return x


@dataclass
class DualSystemOutput:
    l_seed: torch.Tensor          # l~_t      [B, d_l]
    l0: torch.Tensor              # l^(0)_t   [B, d_l]
    h_fast: torch.Tensor          # h^(1)_t   [B, d_l]
    h_slow: torch.Tensor          # h^(2)_t   [B, d_l]
    c_t: torch.Tensor             # gated mix [B, d_l]
    gate: torch.Tensor            # g_t       [B, d_l]


class DualSystemProcessor(nn.Module):
    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        d_h = config.backbone_hidden_size
        d_l = config.d_l
        spec = config.spectral_norm_attention

        # Seed latent projection P_l : R^{d_h} -> R^{d_l}    (Eq. 6)
        self.P_l = _maybe_spectral(nn.Linear(d_h, d_l, bias=False), spec)

        # Fast pathway: queries a learned (K, V) habit bank.
        self.habit_bank = HabitBank(d_l, config.habit_bank_size, spec)

        # Slow pathway: D-layer deliberative stack.
        self.slow_stack = nn.ModuleList([
            _SlowLayer(d_l, spec, config.logit_clamp)
            for _ in range(config.slow_depth)
        ])

        # Axial fusion (Eq. 7): [h^(1); h^(2)] -> l^(0).
        self.W_fuse = _maybe_spectral(nn.Linear(2 * d_l, d_l, bias=True), spec)

        # Multiplicative context-aware gate (Eq. 8):
        # input is [l~; l^(0); l~ ⊙ l^(0)] -> g ∈ [0, 1]^{d_l}.
        self.W_g = _maybe_spectral(nn.Linear(3 * d_l, d_l, bias=True), spec)

    @staticmethod
    def _pool(H_t: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return H_t.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(H_t.dtype)
        return (H_t * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        H_t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DualSystemOutput:
        cfg = self.config
        pooled = self._pool(H_t, attention_mask)               # [B, d_h]
        l_seed = self.P_l(pooled)                              # [B, d_l]      Eq. 6

        h_fast = self.habit_bank(l_seed, cfg.logit_clamp)      # System 1
        h_slow = l_seed
        for layer in self.slow_stack:
            h_slow = layer(h_slow)                             # System 2

        l0 = self.W_fuse(torch.cat([h_fast, h_slow], dim=-1))  # Eq. 7
        gate_in = torch.cat([l_seed, l0, l_seed * l0], dim=-1)
        gate = torch.sigmoid(self.W_g(gate_in))                # Eq. 8 (gate)
        c_t = gate * h_fast + (1.0 - gate) * h_slow            # Eq. 8 (mix)

        return DualSystemOutput(l_seed=l_seed, l0=l0,
                                h_fast=h_fast, h_slow=h_slow,
                                c_t=c_t, gate=gate)


class HiddenStateInjector(nn.Module):
    """Projects [c_t; C_t] into the backbone hidden space and broadcasts
    it along the sequence dimension. Implements Eq. 9:

        H_hat_t = H_t + Φ([c_t; C_t])
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        d_h = config.backbone_hidden_size
        in_dim = config.d_l + config.d_o + config.d_l + config.d_m
        # c_t: d_l ;  C_t: [o; l; m] = d_o + d_l + d_m
        self.proj = nn.Linear(in_dim, d_h, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, H_t: torch.Tensor, c_t: torch.Tensor, C_t: torch.Tensor) -> torch.Tensor:
        delta = self.proj(torch.cat([c_t, C_t], dim=-1))           # [B, d_h]
        return H_t + delta.unsqueeze(1)                            # broadcast over T
