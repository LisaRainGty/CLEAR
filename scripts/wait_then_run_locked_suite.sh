#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?usage: wait_then_run_locked_suite.sh WAIT_PID CONFIG CODE_COMMIT [EXPECTED_TOKEN] [STAGES]}"
config="${2:?usage: wait_then_run_locked_suite.sh WAIT_PID CONFIG CODE_COMMIT [EXPECTED_TOKEN] [STAGES]}"
code_commit="${3:?usage: wait_then_run_locked_suite.sh WAIT_PID CONFIG CODE_COMMIT [EXPECTED_TOKEN] [STAGES]}"
expected_token="${4:-llm_sft_v2_tuning.json}"
stages="${5:-all}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

while kill -0 "$wait_pid" 2>/dev/null; do
  cmdline="$(tr '\0' ' ' < "/proc/${wait_pid}/cmdline" 2>/dev/null || true)"
  if [[ "$cmdline" != *"$expected_token"* ]]; then
    echo "PID ${wait_pid} no longer matches ${expected_token}; refusing to wait on a reused PID" >&2
    exit 2
  fi
  sleep 30
done

exec env \
  CLAIMARC_BGE_PATH=/root/modelscope_cache/AI-ModelScope/bge-large-zh-v1___5 \
  CLAIMARC_PYTHON=/root/claimarc_venv/bin/python \
  CLAIMARC_GRADIENT_CHECKPOINTING=0 \
  CLAIMARC_BATCH_SIZE=12 \
  CLAIMARC_EFFECTIVE_BATCH=36 \
  CLAIMARC_CODE_COMMIT="$code_commit" \
  /root/claimarc_venv/bin/python scripts/run_paper_suite.py \
    --config "$config" \
    --stages "$stages" \
    --execute \
    --continue-on-error \
    --quiet
