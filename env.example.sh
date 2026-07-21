#!/usr/bin/env bash

# Copy this file to env.sh locally and fill in private values there.
# env.sh is intentionally ignored by git.

export MATPOOL_BASE_URL="${MATPOOL_BASE_URL:-https://token.matpool.com/v1}"
export MATPOOL_API_KEY="${MATPOOL_API_KEY:-}"

export CLAIMARC_PYTHONPATH="${CLAIMARC_PYTHONPATH:-src}"

# Make `models.*` / `common.*` / `config` importable.
export CLAIMARC_ROOT="${CLAIMARC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
export PYTHONPATH="$CLAIMARC_ROOT/src:${PYTHONPATH:-}"

# Optional: load BAAI/bge-large-zh-v1.5 from a local directory (offline / faster).
# If unset, the encoder is fetched from ModelScope/HuggingFace on first run.
export CLAIMARC_BGE_PATH="${CLAIMARC_BGE_PATH:-}"
