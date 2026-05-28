"""Persona-grounded dialogue dataset for BRIDGE training.

Reads JSONL with one example per line:

    {
        "persona": "<initial persona description P_0>",
        "history": [
            {"speaker": "user",      "text": "..."},
            {"speaker": "assistant", "text": "..."},
            ...
        ],
        "response": "<gold reply r_t>",
        "negative_response": "<optional mined negative for L_persona>"
    }

We follow the training data construction described in Appendix C of the
paper (RoleMRC + OpenCharacter; 10K held out per source for validation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..config import BridgeConfig


class PersonaDialogueDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        config: BridgeConfig,
        max_length: int = 2048,
        persona_max_length: int = 256,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.persona_max_length = persona_max_length

        with self.path.open("r", encoding="utf-8") as f:
            self.rows: List[Dict] = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.rows)

    def _format_context(self, persona: str, history: List[Dict]) -> str:
        cfg = self.config
        parts = [cfg.persona_marker_open + persona + cfg.persona_marker_close]
        for turn in history:
            if turn["speaker"] == "user":
                parts.append(cfg.user_marker_open + turn["text"] + cfg.user_marker_close)
            else:
                parts.append(turn["text"])
        return "\n".join(parts) + "\n"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        persona = row.get("persona", "")
        history = row.get("history", [])
        response = row["response"]
        negative = row.get("negative_response")

        context = self._format_context(persona, history)
        prompt_ids = self.tokenizer.encode(context, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        full_ids = (prompt_ids + response_ids)[: self.max_length]
        prompt_n = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_n + full_ids[prompt_n:]
        labels = labels[: len(full_ids)]

        persona_ids = self.tokenizer.encode(persona, add_special_tokens=False)
        persona_ids = persona_ids[: self.persona_max_length] or [self.tokenizer.eos_token_id or 0]

        response_only_ids = response_ids[: self.max_length // 4]

        item = {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "persona_ids": torch.tensor(persona_ids, dtype=torch.long),
            "response_ids": torch.tensor(response_only_ids, dtype=torch.long),
            "persona_target": torch.tensor(1.0, dtype=torch.float32),
        }
        if negative:
            neg_ids = self.tokenizer.encode(negative, add_special_tokens=False)[: self.max_length // 4]
            item["negative_response_ids"] = torch.tensor(neg_ids, dtype=torch.long)
        return item


def _pad(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(int(s.size(0)) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_persona_dialogue(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ("input_ids", "labels", "persona_ids", "response_ids"):
        pad_val = pad_id if key in ("input_ids", "persona_ids", "response_ids") else -100
        out[key] = _pad([b[key] for b in batch], pad_val)
    out["attention_mask"] = (out["input_ids"] != pad_id).long()
    out["persona_attention_mask"] = (out["persona_ids"] != pad_id).long()
    out["response_attention_mask"] = (out["response_ids"] != pad_id).long()
    out["persona_target"] = torch.stack([b["persona_target"] for b in batch], dim=0)

    if "negative_response_ids" in batch[0]:
        out["negative_response_ids"] = _pad([b["negative_response_ids"] for b in batch], pad_id)
        out["negative_response_attention_mask"] = (out["negative_response_ids"] != pad_id).long()
    return out
