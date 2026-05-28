#!/usr/bin/env bash
# Long-horizon stability diagnostics on LoCoMo (Figure 4 in the paper).
set -euo pipefail
CKPT="${CKPT:-runs/bridge_32b/bridge_final.pt}"
BENCH="${BENCH:-data/locomo/test.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

python -m bridge.evaluation.eval_locomo_stability \
    --checkpoint "${CKPT}" \
    --locomo_jsonl "${BENCH}" \
    --base_model "${BASE_MODEL}" \
    --max_turns 500 \
    --output_json results/locomo_stability.json
