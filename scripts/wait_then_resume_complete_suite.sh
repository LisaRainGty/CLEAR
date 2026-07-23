#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?usage: wait_then_resume_complete_suite.sh WAIT_PID EXPECTED_TOKEN CODE_COMMIT}"
expected_token="${2:?usage: wait_then_resume_complete_suite.sh WAIT_PID EXPECTED_TOKEN CODE_COMMIT}"
code_commit="${3:?usage: wait_then_resume_complete_suite.sh WAIT_PID EXPECTED_TOKEN CODE_COMMIT}"

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
    --config configs/paper_fair.json \
    --stages all \
    --execute \
    --continue-on-error \
    --quiet
