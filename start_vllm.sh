#!/bin/bash
#
# start_vllm.sh — serve the chat/SQL LLM (Qwen3-14B-AWQ) on :8001 for the RAG lanes
# (and available to other local projects, e.g. the PPT generator).
#
# WHY 14B AND NOT 32B — measured on this box (48 GB A6000), do not "upgrade" casually:
#   Qwen3-32B-AWQ takes ~24 GB. With the OCR VLMs (~16.4 GB) resident that leaves tens of
#   MiB, and OCR's batched band read then fails with CUDA OOM. `_extract_table_crop` falls
#   back to the unconstrained VLM, which silently LOSES the column-shift guarantee and
#   produces ZERO structured rows for the knowledge base — while still returning HTTP 200
#   with a table, so it looks fine. Tuning does not fix it: util 0.55 and 0.50 both OOM,
#   and 0.45 won't start (needs >=1 GiB KV after ~19 GB of weights).
#   Qwen3-14B-AWQ (~9 GB weights, ~13 GB with KV) leaves ~15 GB free and scored the SAME
#   15/15 on the SQL eval. Verified: OCR keeps producing 144 cells alongside it.
#
# VLLM_USE_FLASHINFER_SAMPLER=0 is REQUIRED: FlashInfer's Ninja JIT breaks on the SPACES
# in this project's path ("ninja: error: '/home/ongc/Desktop/backup(ocr)/AVG'").
# HF_HUB_OFFLINE=1 avoids the stale ~/.cache/huggingface/token (401 on hub calls).

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-14B-AWQ}"
SERVED_NAME="${VLLM_SERVED_NAME:-qwen3-14b-awq}"
PORT="${VLLM_PORT:-8001}"
GPU_FRAC="${VLLM_GPU_FRACTION:-0.28}"   # ~13 GB — leaves ~15 GB for the OCR models
MAX_LEN="${VLLM_MAX_LEN:-8192}"

if [ ! -x "$PROJECT_DIR/.venv-vllm/bin/vllm" ]; then
    echo "❌ vLLM venv missing: $PROJECT_DIR/.venv-vllm"; exit 1
fi

# Clear only this port's listener (never a broad pkill — it matches this script itself).
for p in $(lsof -ti ":$PORT" -sTCP:LISTEN 2>/dev/null); do
    echo "  clearing :$PORT (pid $p)"; kill -9 "$p" 2>/dev/null
done
sleep 2

echo "🔥 Serving $MODEL on :$PORT (gpu fraction $GPU_FRAC, max-len $MAX_LEN)"
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$PROJECT_DIR/.venv-vllm/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_FRAC" \
    --max-model-len "$MAX_LEN" \
    --served-model-name "$SERVED_NAME"
