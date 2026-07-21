"""CLAIMARC 数据流水线编排器。

按依赖顺序运行各阶段。支持只跑某些阶段、限定品类、pilot 小样验证。

阶段顺序：
  A0 -> A1 -> A2 -> A3 -> B0 -> B1 -> B2B3 -> B4B5 -> C1 -> C2 -> C3 -> C4 -> C5 -> labels -> final
（B 与 C 在 A 之后可并行，这里顺序执行以简化；各阶段内部已并发/可断点续跑。）

用法：
  source env.sh
  python -m run_pipeline --all                       # 全量
  python -m run_pipeline --pilot                     # food_and_beverages 小样
  python -m run_pipeline --stage A0 A1 --category food_and_beverages
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import config

STAGES = [
    ("A0", "stage_a.a0_build_cas"),
    ("A1", "stage_a.a1_extract_aspects"),
    ("A2", "stage_a.a2_aggregate_free"),
    ("A3", "stage_a.a3_resolve_labels"),
    ("B0", "stage_b.b0_acmt"),
    ("B1", "stage_b.b1_claim_extract"),
    ("B2B3", "stage_b.b2_b3_passage"),
    ("B4B5", "stage_b.b4_b5_align"),
    ("C1", "stage_c.c1_image_triage"),
    ("C2", "stage_c.c2_params"),
    ("C3", "stage_c.c3_ocr"),
    ("C4", "stage_c.c4_vlm"),
    ("C5", "stage_c.c5_fact_records"),
    ("labels", "labels.build_labels"),
    ("final", "final.join_split"),
]
STAGE_MAP = dict(STAGES)
# 接受 --category 的阶段（B2B3/B0/A3/C5/labels/final 为全量聚合，不按品类切）
CATEGORY_AWARE = {"A0", "A1", "A2", "A3", "B1", "B4B5", "C1", "C2", "C3", "C4"}


def run_stage(name: str, module: str, category: str | None, limit: int | None) -> float:
    cmd = [sys.executable, "-m", module]
    if category and name in CATEGORY_AWARE:
        cmd += ["--category", category]
    if limit and name in {"A1", "B1", "B4B5", "C1", "C2", "C3", "C4"}:
        cmd += ["--limit", str(limit)]
    print(f"\n{'=' * 60}\n[run] {name}: {' '.join(cmd)}\n{'=' * 60}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[run] stage {name} FAILED (exit {r.returncode})")
    print(f"[run] {name} done in {dt:.1f}s", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="food_and_beverages 全量小样")
    ap.add_argument("--stage", nargs="+", default=None, help="只跑指定阶段，如 A0 A1")
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not config.MATPOOL_API_KEY:
        print("[warn] MATPOOL_API_KEY 未设置（source env.sh）。需要 LLM 的阶段会失败。")

    category = args.category
    if args.pilot:
        category = category or "food_and_beverages"

    if args.stage:
        names = [s for s in args.stage]
    else:
        names = [n for n, _ in STAGES]

    total = 0.0
    for name in names:
        if name not in STAGE_MAP:
            print(f"[warn] 未知阶段 {name}，跳过。")
            continue
        total += run_stage(name, STAGE_MAP[name], category, args.limit)
    print(f"\n[run] ALL DONE in {total:.1f}s")


if __name__ == "__main__":
    main()
