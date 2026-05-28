#!/usr/bin/env bash
# Train BRIDGE end-to-end on a frozen Qwen2.5-32B-Instruct backbone.
set -euo pipefail

TRAIN_JSONL="${TRAIN_JSONL:-data/persona_dialogues.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/bridge_32b}"

python -m bridge.training.train \
    --train_jsonl "${TRAIN_JSONL}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 32
