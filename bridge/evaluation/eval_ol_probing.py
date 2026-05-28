"""Observable / Latent specialization probes (paper §3.2, Figure 5).

Validates that o and l do not collapse to redundant representations by:
  1) Reporting cos(o, l) before and after training (target: ~0.85 -> ~0.52).
  2) Training linear probes for behavioral vs. cognitive attributes and
     showing the diagonal-dominance pattern:
        O predicts behavioral better (78.3% vs. 58.9%)
        L predicts cognitive    better (74.6% vs. 61.2%)

Inputs:
  --probe_jsonl  : JSONL with fields {persona, prompt, behavioral_label,
                                       cognitive_label}.
  --checkpoint   : BRIDGE checkpoint (loads frozen backbone, runs the
                   refinement, dumps o^(K), l^(K) features).
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..config import BridgeConfig
from ..modeling_bridge import BridgeForCausalLM


@torch.no_grad()
def extract_features(
    model: BridgeForCausalLM,
    tokenizer,
    rows: List[Dict],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = model.config
    o_feats, l_feats, beh, cog = [], [], [], []
    for row in rows:
        persona_ids = tokenizer.encode(row["persona"], add_special_tokens=False, return_tensors="pt").to(device)
        prompt_text = (
            cfg.persona_marker_open + row["persona"] + cfg.persona_marker_close + "\n"
            + cfg.user_marker_open + row["prompt"] + cfg.user_marker_close + "\n"
        )
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
        out = model(input_ids=input_ids, persona_input_ids=persona_ids)
        o_feats.append(out.o_K.cpu())
        l_feats.append(out.l_K.cpu())
        beh.append(int(row["behavioral_label"]))
        cog.append(int(row["cognitive_label"]))
    return (
        torch.cat(o_feats, dim=0),
        torch.cat(l_feats, dim=0),
        torch.tensor(beh, dtype=torch.long),
        torch.tensor(cog, dtype=torch.long),
    )


def _train_linear(features: torch.Tensor, labels: torch.Tensor, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    n = features.size(0)
    perm = torch.randperm(n, generator=g)
    split = int(n * 0.8)
    tr_x, tr_y = features[perm[:split]], labels[perm[:split]]
    te_x, te_y = features[perm[split:]], labels[perm[split:]]
    n_classes = int(labels.max().item() + 1)
    probe = torch.nn.Linear(features.size(-1), n_classes)
    optim = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    for _ in range(50):
        loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True, generator=g)
        for xb, yb in loader:
            optim.zero_grad()
            F.cross_entropy(probe(xb), yb).backward()
            optim.step()
    with torch.no_grad():
        return float((probe(te_x).argmax(-1) == te_y).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe_jsonl", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    config = BridgeConfig()
    model = BridgeForCausalLM.from_hf_backbone(config, args.base_model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    with open(args.probe_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    o, l, beh, cog = extract_features(model, tokenizer, rows, args.device)
    cos_ol = F.cosine_similarity(o, l, dim=-1).mean().item()

    seeds = (42, 2023, 12345)
    results = {
        "cos(o, l)": cos_ol,
        "O -> behavioral": sum(_train_linear(o, beh, s) for s in seeds) / len(seeds),
        "L -> behavioral": sum(_train_linear(l, beh, s) for s in seeds) / len(seeds),
        "O -> cognitive":  sum(_train_linear(o, cog, s) for s in seeds) / len(seeds),
        "L -> cognitive":  sum(_train_linear(l, cog, s) for s in seeds) / len(seeds),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
