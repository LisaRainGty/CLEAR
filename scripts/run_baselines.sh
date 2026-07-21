#!/usr/bin/env bash
# Table 3 baselines under sources_only (same splits / same evidence as CLAIMARC).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CLAIMARC_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT/src"

DS="$ROOT/data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl"
OUT="$ROOT/results/baselines_results.jsonl"
mkdir -p "$ROOT/results"
: > "$OUT.tmp" || true

run_ft () {
  local kind="$1" seed="$2"
  echo "[baseline] $kind seed=$seed"
  python -m models.baselines_ft --dataset "$DS" --kind "$kind" --seed "$seed" --loss bce \
    | tee /dev/stderr | awk '/^RESULT /{print substr($0,8)}' >> "$OUT"
}

# Neural from-scratch + NLI (3 seeds)
for seed in 0 1 2; do
  for kind in textcnn bilstm dam; do
    echo "[baseline] neural $kind seed=$seed"
    python -m models.baselines_neural --dataset "$DS" --kind "$kind" --seed "$seed" \
      | tee /dev/stderr | awk '/^RESULT /{print substr($0,8)}' >> "$OUT" || true
  done
done

# Fine-tuned encoders (3 seeds)
for seed in 0 1 2; do
  for kind in esim bert_nli bert_cls roberta_cls; do
    run_ft "$kind" "$seed"
  done
done

# Frozen probes (single run)
echo "[baseline] frozen probes"
python -m models.baselines --dataset "$DS" \
  | tee /dev/stderr | awk '/^RESULT /{print substr($0,8)}' >> "$OUT" || true
python -m models.baselines_frozen --dataset "$DS" --save_dir "$ROOT/results/frozen_preds" \
  | tee /dev/stderr | awk '/^RESULT /{print substr($0,8)}' >> "$OUT" || true

echo "[baseline] wrote $OUT"
