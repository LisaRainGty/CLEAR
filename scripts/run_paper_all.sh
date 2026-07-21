#!/usr/bin/env bash
# Reproduce all paper experiments under sources_only (3-source evidence only).
# Run from repo root after: source env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CLAIMARC_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT/src"

DS="$ROOT/data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl"
echo "[paper] CLAIMARC_ROOT=$ROOT"
echo "[paper] evidence_policy=sources_only"
echo "[paper] dataset=$DS"

echo "======== RQ1/RQ3/RQ4 campaign6 ========"
python -m models.run_campaign6

echo "======== RQ4 campaign7 ========"
python -m models.run_campaign7

echo "======== RQ4 campaign8 ========"
python -m models.run_campaign8

echo "======== RQ1 in-domain baselines ========"
bash "$ROOT/scripts/run_baselines.sh"

echo "======== RQ2 geometry ========"
python -m models.run_geom_campaign --dataset "$DS" --outdir "$ROOT/embeddings/geom" --seeds 0 1 2
python -m models.geom_probe2 --emb_dir "$ROOT/embeddings/geom" --seeds 0 1 2 --out "$ROOT/results/artifacts/geom2.json"

echo "======== figures ========"
python -m models.metrics_rich || true
python -m models.make_figs || true
python -m models.make_geom_figs --emb_dir "$ROOT/embeddings/geom" --geom_json "$ROOT/results/artifacts/geom2.json" --outdir "$ROOT/paper/figs" || true
python -m models.make_inject_fig || true
python -m models.make_selective_fig || true

echo "######## ALL PAPER EXPERIMENTS COMPLETE ########"
