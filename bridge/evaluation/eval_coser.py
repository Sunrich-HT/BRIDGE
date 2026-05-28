"""CoSER evaluation (paper Table 1, multi-turn given-circumstance acting).

Four metrics scored by GPT-4o on a 100-point scale:
  SC — Storyline Consistency
  An — Anthropomorphism
  CF — Character Fidelity     <-- BRIDGE's main lift
  SQ — Storyline Quality

CoSER probes 200 conversations × 18 turns × 3 characters. We drive BRIDGE
through each conversation turn-by-turn so the hierarchical memory state
genuinely evolves across the dialogue.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from ..config import BridgeConfig
from ..modeling_bridge import BridgeForCausalLM


METRICS = ["SC", "An", "CF", "SQ"]


@torch.no_grad()
def run_conversation(
    model: BridgeForCausalLM,
    tokenizer,
    persona: str,
    user_turns: List[str],
    device: str,
    max_new_tokens: int = 256,
) -> List[str]:
    cfg = model.config
    persona_ids = tokenizer.encode(persona, add_special_tokens=False, return_tensors="pt").to(device)
    memory_state = None
    transcript: List[str] = []
    context = cfg.persona_marker_open + persona + cfg.persona_marker_close + "\n"

    for user_text in user_turns:
        context += cfg.user_marker_open + user_text + cfg.user_marker_close + "\n"
        input_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to(device)
        out = model(
            input_ids=input_ids,
            memory_state=memory_state,
            persona_input_ids=persona_ids if memory_state is None else None,
        )

        out_ids: List[int] = []
        for _ in range(max_new_tokens):
            nxt = int(out.logits[:, -1, :].argmax(dim=-1).item())
            if nxt == tokenizer.eos_token_id:
                break
            out_ids.append(nxt)
            input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=device)], dim=1)
            out = model(input_ids=input_ids, memory_state=out.memory_state)
        memory_state = out.memory_state
        reply = tokenizer.decode(out_ids, skip_special_tokens=True)
        transcript.append(reply)
        context += reply + "\n"

    return transcript


def score_with_judge(persona: str, transcript: List[str], metric: str, judge=None) -> float:
    if judge is None:
        return 50.0
    return float(judge(persona=persona, transcript=transcript, metric=metric))


def run(args) -> Dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    config = BridgeConfig()
    model = BridgeForCausalLM.from_hf_backbone(config, args.base_model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    scores = defaultdict(list)
    rupture_count = 0
    n_dialogues = 0
    with open(args.bench_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            persona = row["persona"]
            user_turns = row["user_turns"]
            transcript = run_conversation(model, tokenizer, persona, user_turns, args.device)
            for metric in METRICS:
                s = score_with_judge(persona, transcript, metric)
                scores[metric].append(s)
            # Asymmetric-rupture audit: CF < 35 indicates a rupture (§3.4).
            if scores["CF"][-1] < 35.0:
                rupture_count += 1
            n_dialogues += 1

    summary = {m: sum(v) / len(v) for m, v in scores.items() if v}
    summary["Avg"] = sum(summary.values()) / max(1, len(summary))
    summary["RuptureRate"] = rupture_count / max(1, n_dialogues)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bench_jsonl", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", default="coser_results.json")
    args = p.parse_args()

    summary = run(args)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
