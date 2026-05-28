#!/usr/bin/env bash
set -euo pipefail
CKPT="${CKPT:-runs/bridge_32b/bridge_final.pt}"
BENCH="${BENCH:-data/coser/test.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

python -m bridge.evaluation.eval_coser \
    --checkpoint "${CKPT}" \
    --bench_jsonl "${BENCH}" \
    --base_model "${BASE_MODEL}" \
    --output_json results/coser.json
