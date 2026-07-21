#!/usr/bin/env bash
# Deploy CLEAR/main to a Matpool GPU box and launch sources_only paper runs.
# Usage (from your laptop, AFTER SSH works without Clash TUN blocking port 22/mapped ports):
#
#   export REMOTE='root@HOST'
#   export REMOTE_PORT=29378
#   export REMOTE_DIR=/root/CLEAR
#   bash scripts/deploy_remote.sh
#
# Do NOT put passwords in this file. Prefer: ssh-copy-id, or `ssh claimarc-gpu`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE:?set REMOTE=user@host}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/root/CLEAR}"
SSH=(ssh -p "$REMOTE_PORT" -o StrictHostKeyChecking=accept-new "$REMOTE")
RSYNC=(rsync -azP -e "ssh -p $REMOTE_PORT -o StrictHostKeyChecking=accept-new")

echo "[deploy] sync $ROOT -> $REMOTE:$REMOTE_DIR"
"${SSH[@]}" "mkdir -p '$REMOTE_DIR'"
"${RSYNC[@]}" \
  --exclude '.git' --exclude 'embeddings' --exclude 'results/legacy_pre_sources_only' \
  --exclude '__pycache__' --exclude '*.pt' \
  "$ROOT/" "$REMOTE:$REMOTE_DIR/"

echo "[deploy] launch paper runs in tmux"
"${SSH[@]}" bash -s <<EOF
set -euo pipefail
cd '$REMOTE_DIR'
cp -n env.example.sh env.sh || true
# Prefer local encoders if present
export CLAIMARC_ROOT='$REMOTE_DIR'
export PYTHONPATH='$REMOTE_DIR/src'
export CLAIMARC_BGE_PATH="\${CLAIMARC_BGE_PATH:-/root/models/bge-large-zh-v1.5}"
export CLAIMARC_BERT_PATH="\${CLAIMARC_BERT_PATH:-/root/models/bert-base-chinese}"
export CLAIMARC_ROBERTA_PATH="\${CLAIMARC_ROBERTA_PATH:-/root/models/chinese-roberta-wwm-ext}"
export CLAIMARC_PYTHON="\${CLAIMARC_PYTHON:-/root/miniconda3/envs/clm/bin/python}"
tmux has-session -t clear_paper 2>/dev/null && tmux kill-session -t clear_paper || true
tmux new-session -d -s clear_paper "cd '$REMOTE_DIR' && source env.sh 2>/dev/null; export CLAIMARC_ROOT='$REMOTE_DIR' PYTHONPATH='$REMOTE_DIR/src' CLAIMARC_BGE_PATH='\$CLAIMARC_BGE_PATH' CLAIMARC_PYTHON='\$CLAIMARC_PYTHON'; bash scripts/run_paper_all.sh 2>&1 | tee results/run_paper_all.log"
echo "[remote] tmux session clear_paper started; attach with: tmux a -t clear_paper"
EOF
echo "[deploy] done"
