"""BRIDGE configuration.

Mirrors the hyperparameters reported in the paper (Appendix Table 6,
"Training hyperparameters for BRIDGE"). Defaults match the
Qwen2.5-32B-Instruct backbone used for all main-table results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BridgeConfig:
    # -------- Backbone (frozen) --------
    base_model: str = "Qwen/Qwen2.5-32B-Instruct"
    backbone_hidden_size: int = 5120           # Qwen2.5-32B hidden dim (d_h)
    backbone_vocab_size: int = 152_064
    backbone_dtype: str = "bfloat16"
    freeze_backbone: bool = True

    # -------- State dimensions (Appendix B "Module configuration") --------
    d_o: int = 1024                            # Observable
    d_l: int = 1024                            # Latent
    d_m_tier: int = 1024                       # Per-tier memory dim
    cross_attention_heads: int = 16            # per pairwise CA module

    # Concatenated working memory dim d_m = 3 * d_m_tier
    @property
    def d_m(self) -> int:
        return 3 * self.d_m_tier

    # -------- Triangular Fixed-Point Refinement (§2.2) --------
    K_refinement: int = 3                      # default refinement depth
    alpha_o: float = 0.25
    alpha_l: float = 0.25
    alpha_m: float = 0.25
    logit_clamp: float = 8.0                   # pre-softmax bound (Assumption 1)
    spectral_norm_attention: bool = True
    contraction_proxy_tol: float = 1.0         # alpha-tilde threshold for diagnostics

    # -------- Dual-System Processing (§2.3) --------
    habit_bank_size: int = 256                 # number of (K, V) habit memories
    slow_depth: int = 4                        # D-layer deliberative stack

    # -------- Hierarchical Memory Evolution (§2.4 + Table 6) --------
    eta_e: float = 0.20                        # episodic update rate
    eta_a: float = 0.05                        # affective
    eta_p: float = 0.0035                      # personality (slow)
    delta_e: float = 0.5                       # episodic clip bound
    delta_a: float = 0.2                       # affective
    delta_p: float = 0.05                      # personality
    snapshot_every: int = 10                   # turns between persistent snapshots

    # Lyapunov weights γ_i (used by V(m_t))
    gamma_e: float = 1.0
    gamma_a: float = 1.0
    gamma_p: float = 1.0

    # -------- Training Objective (Eq. 11, Table 6) --------
    lambda_cycle: float = 0.10
    lambda_persona: float = 0.50

    # -------- Optimization (Table 6) --------
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 32
    total_steps: int = 50_000
    warmup_steps: int = 1_000

    # -------- Persona-consistency classifier --------
    persona_clf_hidden: int = 512
    response_encoder_dim: int = 1024           # output dim of Enc(r_gt)

    # -------- Markers / formatting --------
    persona_marker_open: str = "<|persona|>"
    persona_marker_close: str = "<|/persona|>"
    user_marker_open: str = "<|user|>"
    user_marker_close: str = "<|/user|>"

    def __post_init__(self) -> None:
        assert 0 < self.alpha_o < 1 and 0 < self.alpha_l < 1 and 0 < self.alpha_m < 1
        assert self.eta_e > self.eta_a > self.eta_p, "timescale separation η_e > η_a > η_p"
        assert self.delta_e > self.delta_a > self.delta_p, "clip-bound hierarchy"
