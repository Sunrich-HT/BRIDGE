"""BRIDGE training objective (§2.5 / Eq. 11).

  L_total = L_LM + λ1 · L_cycle + λ2 · L_persona

  * L_LM      : next-token cross-entropy through the frozen backbone.
  * L_cycle   : cyclic coherence loss (Eq. 5) — penalizes the post-refinement
                residuals across the three pairwise edges.
  * L_persona : BCE from a lightweight persona-consistency classifier whose
                input is [o^(K); l^(K); Enc(r_gt)] (Appendix B.4).

Default weights from validation: λ1 = 0.1, λ2 = 0.5.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BridgeConfig


def cyclic_coherence_loss(
    o_K: torch.Tensor,
    l_K: torch.Tensor,
    m_K: torch.Tensor,
    f_O: nn.Module,
    f_L: nn.Module,
    f_M: nn.Module,
) -> torch.Tensor:
    """Eq. 5:
        L_cycle = ||o^K - f_O(m^K)||^2 + ||l^K - f_L(o^K)||^2 + ||m^K - f_M(l^K)||^2
    """
    return (
        (o_K - f_O(m_K)).pow(2).sum(dim=-1)
        + (l_K - f_L(o_K)).pow(2).sum(dim=-1)
        + (m_K - f_M(l_K)).pow(2).sum(dim=-1)
    ).mean()


class PersonaConsistencyClassifier(nn.Module):
    """Predicts P(response consistent with persona | o^K, l^K, Enc(r_gt)).

    Trained with BCE under teacher forcing: positives are (persona, gt_response)
    pairs from the corpus; negatives are (persona, response-from-other-persona).
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        in_dim = config.d_o + config.d_l + config.response_encoder_dim
        h = config.persona_clf_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        o_K: torch.Tensor,
        l_K: torch.Tensor,
        response_repr: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([o_K, l_K, response_repr], dim=-1)).squeeze(-1)


def persona_consistency_loss(
    classifier: PersonaConsistencyClassifier,
    o_K: torch.Tensor,
    l_K: torch.Tensor,
    response_repr: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """target: 1 for consistent, 0 for inconsistent (mined negatives)."""
    logits = classifier(o_K, l_K, response_repr)
    return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))


def bridge_total_loss(
    lm_loss: torch.Tensor,
    cycle_loss: torch.Tensor,
    persona_loss: torch.Tensor,
    config: BridgeConfig,
) -> Dict[str, torch.Tensor]:
    total = lm_loss + config.lambda_cycle * cycle_loss + config.lambda_persona * persona_loss
    return {
        "total": total,
        "lm": lm_loss,
        "cycle": cycle_loss,
        "persona": persona_loss,
    }
