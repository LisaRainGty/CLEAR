#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLAIMARC_REMOTE_ROOT:-/root/CLEAR_run}"
PY="${CLAIMARC_PYTHON:-/root/claimarc_venv/bin/python}"
STATUS="$ROOT/results/fair_rerun/status/claimarc_canonical_s0.json"

export CLAIMARC_PYTHON="$PY"
export CLAIMARC_BATCH_SIZE="${CLAIMARC_BATCH_SIZE:-12}"
export CLAIMARC_EFFECTIVE_BATCH="${CLAIMARC_EFFECTIVE_BATCH:-36}"
export CLAIMARC_BGE_PATH="${CLAIMARC_BGE_PATH:-/root/modelscope_cache/AI-ModelScope/bge-large-zh-v1___5}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/modelscope_cache}"
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CLAIMARC_GRADIENT_CHECKPOINTING="${CLAIMARC_GRADIENT_CHECKPOINTING:-0}"

# The first canonical seed is deliberately run as a gate before the multi-day
# queue.  This wrapper can be started while that gate is still running.
while [[ ! -f "$STATUS" ]]; do
  sleep 30
done
"$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["returncode"] == 0 else 1)' "$STATUS"

cd "$ROOT"
exec "$PY" scripts/run_paper_suite.py \
  --stages all --execute --continue-on-error --quiet
