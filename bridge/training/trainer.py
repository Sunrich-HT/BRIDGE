"""BRIDGE trainer.

A single-phase SFT trainer that:
  - freezes the backbone (handled by BridgeModel);
  - optimizes the BRIDGE-specific modules only;
  - computes L_total = L_LM + λ1·L_cycle + λ2·L_persona at every step.

Defaults from Appendix Table 6: AdamW, lr=1e-5, batch=32, 50K steps,
weight decay 0.01, grad clip 1.0.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import BridgeConfig
from ..losses import (
    bridge_total_loss,
    cyclic_coherence_loss,
    persona_consistency_loss,
)


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in n.lower() or "bias" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def _warmup_cosine(optimizer, total_steps: int, warmup: int):
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class BridgeTrainer:
    def __init__(
        self,
        model,
        config: BridgeConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        log_fn: Callable[[Dict], None] = print,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.log_fn = log_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def _compute_aux_losses(self, out, batch) -> Dict[str, torch.Tensor]:
        # Cyclic coherence (Eq. 5).
        cycle = cyclic_coherence_loss(
            out.o_K, out.l_K, out.m_K_concat,
            f_O=self.model.model.f_O,
            f_L=self.model.model.f_L,
            f_M=self.model.model.f_M,
        )

        # Persona BCE (Appendix B.4): one positive (gt) + one mined negative.
        response_repr = self.model.encode_response(
            batch["response_ids"],
            batch.get("response_attention_mask"),
        )
        pos = persona_consistency_loss(
            self.model.persona_classifier, out.o_K, out.l_K, response_repr,
            target=batch["persona_target"].to(self.device),
        )
        if "negative_response_ids" in batch:
            neg_repr = self.model.encode_response(
                batch["negative_response_ids"],
                batch.get("negative_response_attention_mask"),
            )
            neg_target = torch.zeros_like(batch["persona_target"]).to(self.device)
            neg = persona_consistency_loss(
                self.model.persona_classifier, out.o_K, out.l_K, neg_repr, target=neg_target,
            )
            persona = 0.5 * (pos + neg)
        else:
            persona = pos

        return {"cycle": cycle, "persona": persona}

    def train(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        optimizer = _make_optimizer(self.model, self.config.learning_rate, self.config.weight_decay)
        scheduler = _warmup_cosine(optimizer, self.config.total_steps, self.config.warmup_steps)

        self.model.train()
        # Keep the frozen backbone in eval mode even though the wrapper is in train mode.
        if self.model.backbone is not None:
            self.model.backbone.eval()

        global_step = 0
        optimizer.zero_grad()
        while global_step < self.config.total_steps:
            for batch in self.train_loader:
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                    persona_input_ids=batch.get("persona_ids"),
                )
                aux = self._compute_aux_losses(out, batch)
                losses = bridge_total_loss(out.loss, aux["cycle"], aux["persona"], self.config)
                (losses["total"] / self.gradient_accumulation_steps).backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.grad_clip,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                if global_step % 20 == 0:
                    self.log_fn({
                        "step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "loss/total": float(losses["total"].detach()),
                        "loss/lm": float(losses["lm"].detach()),
                        "loss/cycle": float(losses["cycle"].detach()),
                        "loss/persona": float(losses["persona"].detach()),
                        "refine/final_residual": float(out.refinement.final_residual.mean().detach()),
                        "refine/contraction_proxy": float(out.refinement.contraction_proxy.mean().detach()),
                    })

                if global_step >= self.config.total_steps:
                    break

        ckpt = {
            "config": self.config.__dict__,
            "model": {n: p.detach().cpu() for n, p in self.model.state_dict().items()
                      if not n.startswith("model.backbone.")},
        }
        torch.save(ckpt, os.path.join(output_dir, "bridge_final.pt"))
        self.log_fn({"event": "checkpoint", "path": os.path.join(output_dir, "bridge_final.pt")})
