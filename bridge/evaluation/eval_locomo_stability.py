"""LoCoMo stability diagnostics (paper §3.3 + Appendix).

Runs an auto-regressive 500-turn simulation on LoCoMo conversations and
records:
  - V(m_t) / V_max          Lyapunov-energy trajectory (Theorem 2)
  - Personality drift       ||m^(p)_t - m^(p)_0||
  - Per-tier drifts          for episodic/affective/personality
  - Contraction proxy α̃    per turn

Used to validate Figure 4 in the paper: V(m_t)/V_max stays below 1 with a
peak around 31.6%.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from ..config import BridgeConfig
from ..memory import HierarchicalMemory
from ..modeling_bridge import BridgeForCausalLM


@torch.no_grad()
def simulate(
    model: BridgeForCausalLM,
    tokenizer,
    persona: str,
    user_turns: List[str],
    device: str,
    K: int = 3,
) -> Dict[str, List[float]]:
    cfg = model.config
    persona_ids = tokenizer.encode(persona, add_special_tokens=False, return_tensors="pt").to(device)
    memory_state = None
    context = cfg.persona_marker_open + persona + cfg.persona_marker_close + "\n"

    bound = float(model.model.memory.lyapunov_bound(state={}))
    trace = {
        "V": [], "V_over_Vmax": [], "drift_p": [],
        "drift_e": [], "drift_a": [],
        "alpha_tilde": [],
    }

    for user_text in user_turns:
        context += cfg.user_marker_open + user_text + cfg.user_marker_close + "\n"
        input_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to(device)
        out = model(
            input_ids=input_ids,
            memory_state=memory_state,
            persona_input_ids=persona_ids if memory_state is None else None,
            K=K,
        )
        memory_state = out.memory_state

        gammas = (cfg.gamma_e, cfg.gamma_a, cfg.gamma_p)
        V = HierarchicalMemory.lyapunov_energy(memory_state, gammas).item()
        trace["V"].append(V)
        trace["V_over_Vmax"].append(V / max(bound, 1e-8))
        trace["drift_p"].append((memory_state["m_p"] - memory_state["m0_p"]).norm().item())
        trace["drift_e"].append((memory_state["m_e"] - memory_state["m0_e"]).norm().item())
        trace["drift_a"].append((memory_state["m_a"] - memory_state["m0_a"]).norm().item())
        trace["alpha_tilde"].append(float(out.refinement.contraction_proxy.mean()))

        context += "[assistant continues]\n"  # we don't need to decode to study memory dynamics
    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--locomo_jsonl", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_turns", type=int, default=500)
    p.add_argument("--output_json", default="locomo_stability.json")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    config = BridgeConfig()
    model = BridgeForCausalLM.from_hf_backbone(config, args.base_model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    aggregated: Dict[str, List[float]] = {}
    n = 0
    with open(args.locomo_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            turns = row["user_turns"][: args.max_turns]
            trace = simulate(model, tokenizer, row["persona"], turns, args.device)
            for k, v in trace.items():
                aggregated.setdefault(k, []).append(v)
            n += 1
            if n >= 10:
                break

    # Summary statistics across simulations.
    summary = {
        "n": n,
        "peak_V_over_Vmax": max(max(v) for v in aggregated["V_over_Vmax"]) if aggregated.get("V_over_Vmax") else 0.0,
        "mean_alpha_tilde": sum(sum(v) / len(v) for v in aggregated["alpha_tilde"]) / max(1, n),
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "traces": aggregated}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
