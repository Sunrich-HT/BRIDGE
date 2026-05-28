"""Hierarchical Memory Evolution (§2.4 of the paper).

Three tiers with distinct timescales and clip bounds:

  episodic     m^(e)   η_e = 0.20    δ_e = 0.5     τ_e ≈ 3-5  turns
  affective    m^(a)   η_a = 0.05    δ_a = 0.2     τ_a ≈ 10-20 turns
  personality  m^(p)   η_p = 0.0035  δ_p = 0.05    τ_p ≈ 150-250 turns

Each tier follows the anchored clipped update (Eq. 10):

  m_{t+1}^{(i)} = (1 - η_i) m_t^{(i)} + η_i ( m_0^{(i)}
                  + clip( Δm_t^{(i)}, -δ_i, +δ_i ) )

This anchors evolution to the initial-persona encoding m_0^{(i)} so that
the Lyapunov function

  V(m_t) = Σ_i γ_i ||m_t^{(i)} - m_0^{(i)}||^2

is uniformly bounded (Theorem 2):

  V(m_t) ≤ Σ_i γ_i · max(||m_init - m_0||, δ_i √d_i)^2

For deployment, working memory is periodically snapshotted (every T=10
turns by default) into a persistent store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import BridgeConfig


@dataclass
class MemorySnapshot:
    """Periodic external snapshot of the working memory tiers."""
    step: int
    episodic: torch.Tensor
    affective: torch.Tensor
    personality: torch.Tensor


class _PersonaEncoder(nn.Module):
    """Encodes the initial persona description (a vector of pooled tokens)
    into three tier anchors m_0^{(e)}, m_0^{(a)}, m_0^{(p)}."""

    def __init__(self, config: BridgeConfig):
        super().__init__()
        d_h = config.backbone_hidden_size
        d_tier = config.d_m_tier
        self.head_e = nn.Linear(d_h, d_tier, bias=True)
        self.head_a = nn.Linear(d_h, d_tier, bias=True)
        self.head_p = nn.Linear(d_h, d_tier, bias=True)

    def forward(self, persona_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head_e(persona_repr), self.head_a(persona_repr), self.head_p(persona_repr)


class HierarchicalMemory(nn.Module):
    """Per-dialogue working memory with three-tier evolution dynamics."""

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        self.persona_encoder = _PersonaEncoder(config)

        # Projection from refined m^(K) (concatenated, d_m) back to per-tier
        # deltas. The refinement operator works on the concatenated state;
        # we split it back into tiers for the per-tier update.
        # Anchors m_0^{(i)} and current m_t^{(i)} are kept off-graph (state).

    # -------- State management --------

    def initialize(
        self,
        persona_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute tier anchors m_0^{(i)} and set m_init = m_0."""
        m0_e, m0_a, m0_p = self.persona_encoder(persona_repr)
        state = {
            "m0_e": m0_e, "m0_a": m0_a, "m0_p": m0_p,
            "m_e": m0_e.clone(), "m_a": m0_a.clone(), "m_p": m0_p.clone(),
            "snapshots": [],
            "step": 0,
        }
        return state

    @staticmethod
    def concat_working(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concat the three tiers to feed the triangular refinement operator."""
        return torch.cat([state["m_e"], state["m_a"], state["m_p"]], dim=-1)

    def split_working(self, m_concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.config.d_m_tier
        return m_concat[..., :d], m_concat[..., d:2*d], m_concat[..., 2*d:]

    # -------- Per-turn update (Eq. 10) --------

    def update(
        self,
        state: Dict[str, torch.Tensor],
        m_refined_concat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        m_e_new, m_a_new, m_p_new = self.split_working(m_refined_concat)

        def _step(m_t, m_refined, m_0, eta, delta):
            delta_m = m_refined - m_t                                    # Δm^(i)_t
            clipped = delta_m.clamp(-delta, delta)
            return (1.0 - eta) * m_t + eta * (m_0 + clipped)

        state["m_e"] = _step(state["m_e"], m_e_new, state["m0_e"], cfg.eta_e, cfg.delta_e)
        state["m_a"] = _step(state["m_a"], m_a_new, state["m0_a"], cfg.eta_a, cfg.delta_a)
        state["m_p"] = _step(state["m_p"], m_p_new, state["m0_p"], cfg.eta_p, cfg.delta_p)
        state["step"] += 1

        if state["step"] % cfg.snapshot_every == 0:
            state["snapshots"].append(MemorySnapshot(
                step=state["step"],
                episodic=state["m_e"].detach().clone(),
                affective=state["m_a"].detach().clone(),
                personality=state["m_p"].detach().clone(),
            ))
        return state

    # -------- Diagnostics --------

    @staticmethod
    def lyapunov_energy(state: Dict[str, torch.Tensor], gammas: Tuple[float, float, float]) -> torch.Tensor:
        """V(m_t) = Σ_i γ_i ||m_t^{(i)} - m_0^{(i)}||^2  (Theorem 2)."""
        g_e, g_a, g_p = gammas
        return (
            g_e * (state["m_e"] - state["m0_e"]).pow(2).sum(dim=-1)
            + g_a * (state["m_a"] - state["m0_a"]).pow(2).sum(dim=-1)
            + g_p * (state["m_p"] - state["m0_p"]).pow(2).sum(dim=-1)
        )

    def lyapunov_bound(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Theoretical V_max = Σ_i γ_i · (δ_i √d_i)^2  when m_init = m_0."""
        cfg = self.config
        d = cfg.d_m_tier
        return torch.tensor(
            cfg.gamma_e * (cfg.delta_e ** 2) * d
            + cfg.gamma_a * (cfg.delta_a ** 2) * d
            + cfg.gamma_p * (cfg.delta_p ** 2) * d
        )

    @staticmethod
    def personality_drift(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-turn diagnostic: ||m^(p)_t - m^(p)_0||."""
        return (state["m_p"] - state["m0_p"]).norm(dim=-1)
