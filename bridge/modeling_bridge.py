"""BRIDGE model assembly (Algorithm 1 in the paper).

A frozen LLM backbone (Qwen2.5-32B-Instruct by default) provides per-token
hidden states H_t. BRIDGE wraps this backbone with:

  - DualSystemProcessor          (Stage I, §2.3)
  - TriangularFixedPointRefinement (Stage II, §2.2)
  - HiddenStateInjector + LM head (Stage III)
  - HierarchicalMemory            (Stage IV, §2.4)

Only BRIDGE-specific modules are trainable; the backbone is frozen. The
paper reports 277M trainable params (~0.85% of a 32B backbone).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BridgeConfig
from .dual_system import DualSystemProcessor, HiddenStateInjector
from .triangular import TriangularFixedPointRefinement, RefinementTrace
from .memory import HierarchicalMemory
from .losses import PersonaConsistencyClassifier


@dataclass
class BridgeStepOutput:
    """Per-turn output of a BridgeForCausalLM forward pass."""
    logits: torch.Tensor
    h_hat: torch.Tensor                       # H_hat_t, conditioned token states
    o_K: torch.Tensor
    l_K: torch.Tensor
    m_K_concat: torch.Tensor
    refinement: RefinementTrace
    dual_system: object
    memory_state: Dict[str, torch.Tensor]
    loss: Optional[torch.Tensor] = None


class BridgeModel(nn.Module):
    """Frozen backbone + trainable BRIDGE modules. No LM head here."""

    def __init__(self, config: BridgeConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.backbone = backbone                # optional; can be attached later

        self.dual_system = DualSystemProcessor(config)
        self.refinement = TriangularFixedPointRefinement(config)
        self.memory = HierarchicalMemory(config)
        self.injector = HiddenStateInjector(config)

        # Cyclic-coherence projections (Eq. 5 in losses.py).
        self.f_O = nn.Linear(config.d_m, config.d_o, bias=False)
        self.f_L = nn.Linear(config.d_o, config.d_l, bias=False)
        self.f_M = nn.Linear(config.d_l, config.d_m, bias=False)

        # P_o : pooled backbone -> initial observable (Algorithm 1, line 7).
        self.P_o = nn.Linear(config.backbone_hidden_size, config.d_o, bias=False)

        if config.freeze_backbone and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    # ----------------- Backbone IO -----------------

    def attach_backbone(self, backbone: nn.Module) -> None:
        self.backbone = backbone
        if self.config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @torch.no_grad()
    def _encode_persona(self, persona_input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pooled backbone encoding of the initial persona description."""
        assert self.backbone is not None, "Backbone not attached"
        out = self.backbone(persona_input_ids, output_hidden_states=True)
        h = out.hidden_states[-1]              # [B, n_p, d_h]
        return h.mean(dim=1)                   # [B, d_h]

    def _backbone_hidden(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.backbone is None:
            # Test/dev surrogate: random embedding-like states.
            b, t = input_ids.shape
            return torch.zeros(b, t, self.config.backbone_hidden_size, device=input_ids.device)
        out = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-1]

    # ----------------- Per-turn forward (Algorithm 1) -----------------

    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        persona_repr: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
    ) -> BridgeStepOutput:
        H_t = self._backbone_hidden(input_ids, attention_mask)               # [B, n, d_h]

        # ----- Stage IV bootstrap: initialize memory if absent -----
        if memory_state is None:
            assert persona_repr is not None, "Need persona_repr or memory_state"
            memory_state = self.memory.initialize(persona_repr)
        m0 = self.memory.concat_working(memory_state)                        # m^(0) ← m_{t-1}

        # ----- Stage I: state initialization (Algorithm 1, lines 3-7) -----
        ds = self.dual_system(H_t, attention_mask)                           # l_seed, l^(0), c_t

        pooled = self.dual_system._pool(H_t, attention_mask)                 # [B, d_h]
        o0 = self.P_o(pooled)                                                # [B, d_o]
        l0 = ds.l0                                                           # [B, d_l]

        # ----- Stage II: triangular fixed-point refinement -----
        trace = self.refinement(o0, l0, m0, K=K)
        o_K, l_K, m_K = trace.o, trace.l, trace.m

        # ----- Stage III: control signal and hidden-state injection -----
        C_t = torch.cat([o_K, l_K, m_K], dim=-1)                             # Eq. 4
        h_hat = self.injector(H_t, ds.c_t, C_t)                              # Eq. 9

        # ----- Stage IV: hierarchical memory update -----
        memory_state = self.memory.update(memory_state, m_K)

        return BridgeStepOutput(
            logits=None,                       # set by BridgeForCausalLM
            h_hat=h_hat,
            o_K=o_K, l_K=l_K, m_K_concat=m_K,
            refinement=trace,
            dual_system=ds,
            memory_state=memory_state,
        )


class BridgeForCausalLM(nn.Module):
    """BridgeModel + LM head pulled from the frozen backbone for decoding."""

    def __init__(self, config: BridgeConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.model = BridgeModel(config, backbone=backbone)
        self.persona_classifier = PersonaConsistencyClassifier(config)
        # Response encoder for the persona BCE loss: a small MLP over pooled
        # backbone states of the ground-truth response. Cheap and on-graph.
        self.response_encoder = nn.Sequential(
            nn.Linear(config.backbone_hidden_size, config.response_encoder_dim),
            nn.GELU(),
            nn.Linear(config.response_encoder_dim, config.response_encoder_dim),
        )

    @property
    def backbone(self) -> Optional[nn.Module]:
        return self.model.backbone

    def attach_backbone(self, backbone: nn.Module) -> None:
        self.model.attach_backbone(backbone)

    def _lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Re-use the frozen LM head from the backbone.
        if self.backbone is None:
            return torch.zeros(*hidden_states.shape[:-1], self.config.backbone_vocab_size,
                               device=hidden_states.device)
        if hasattr(self.backbone, "lm_head"):
            return self.backbone.lm_head(hidden_states)
        # Fall back to the tied embedding projection used by many HF models.
        emb = self.backbone.get_input_embeddings().weight
        return F.linear(hidden_states, emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        persona_input_ids: Optional[torch.Tensor] = None,
        persona_repr: Optional[torch.Tensor] = None,
        response_input_ids: Optional[torch.Tensor] = None,
        persona_target: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
    ) -> BridgeStepOutput:
        # Encode persona description if memory state is absent.
        if memory_state is None and persona_repr is None and persona_input_ids is not None:
            persona_repr = self.model._encode_persona(persona_input_ids)

        out = self.model.step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            persona_repr=persona_repr,
            K=K,
        )
        out.logits = self._lm_logits(out.h_hat)

        if labels is not None:
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out.loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    @torch.no_grad()
    def encode_response(
        self,
        response_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-pooled backbone encoding of the ground-truth response,
        then projected through the response encoder for L_persona."""
        h = self.model._backbone_hidden(response_input_ids, attention_mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = h.mean(dim=1)
        return self.response_encoder(pooled)

    # ----- Parameter accounting -----

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_hf_backbone(
        cls,
        config: BridgeConfig,
        hf_model_name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "BridgeForCausalLM":
        """Convenience factory that loads a HuggingFace backbone and freezes it."""
        hf_model_name = hf_model_name or config.base_model
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError("transformers is required for from_hf_backbone()") from e
        dtype = dtype or {"bfloat16": torch.bfloat16, "float16": torch.float16,
                          "float32": torch.float32}[config.backbone_dtype]
        backbone = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=dtype)
        # Sync sizes from the actual backbone to avoid drift.
        config.backbone_hidden_size = backbone.config.hidden_size
        config.backbone_vocab_size = backbone.config.vocab_size
        return cls(config, backbone=backbone)
