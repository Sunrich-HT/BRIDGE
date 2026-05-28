"""CLI entry point for BRIDGE training.

Usage:
    python -m bridge.training.train \\
        --train_jsonl data/role_dialogues.jsonl \\
        --base_model Qwen/Qwen2.5-32B-Instruct \\
        --output_dir runs/bridge_32b
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from ..config import BridgeConfig
from ..data.dataset import PersonaDialogueDataset, collate_persona_dialogue
from ..modeling_bridge import BridgeForCausalLM
from .trainer import BridgeTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--output_dir", default="runs/bridge_32b")
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--config_overrides", default=None)
    args = p.parse_args()

    config = BridgeConfig()
    if args.config_overrides:
        for k, v in json.loads(args.config_overrides).items():
            if hasattr(config, k):
                setattr(config, k, v)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [
        config.persona_marker_open, config.persona_marker_close,
        config.user_marker_open, config.user_marker_close,
    ]})

    dataset = PersonaDialogueDataset(args.train_jsonl, tokenizer, config)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_persona_dialogue(b, pad_id=pad_id),
        num_workers=2,
        drop_last=True,
    )

    print(f"Loading frozen backbone: {args.base_model}")
    model = BridgeForCausalLM.from_hf_backbone(config, args.base_model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable / 1e6:.1f}M "
          f"({100 * n_trainable / n_total:.2f}% of total {n_total / 1e9:.2f}B)")

    trainer = BridgeTrainer(
        model=model,
        config=config,
        train_loader=loader,
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer.train(args.output_dir)


if __name__ == "__main__":
    main()
