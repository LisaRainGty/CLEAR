"""Stage C5 — 三源证据并列输出（无 LLM）。

对每个 (p, a) ∈ A_cmt(p)，把 C2/C3/C4 三源 evidence list 原样并列到一条 fact_record，
不做裁决/归一化/合并。附 evidence_count 与 coverage、confidence 三档。
产出 data/processed/stageC/fact_records.jsonl

用法：python -m stage_c.c5_fact_records
     python -m stage_c.c5_fact_records --acmt data/processed/stageB_product_v2/acmt_product_v2.json --out data/processed/stageB_product_v2/fact_records_product_v2.jsonl
"""
from __future__ import annotations

import argparse

import config
from common import product_index as pidx
from common.io_utils import read_json, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--evidence_params", default=str(config.STAGE_C / "evidence_params.json"))
    ap.add_argument("--evidence_ocr", default=str(config.STAGE_C / "evidence_ocr.json"))
    ap.add_argument("--evidence_vlm", default=str(config.STAGE_C / "evidence_vlm.json"))
    ap.add_argument("--out", default=str(config.STAGE_C / "fact_records.jsonl"))
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    ev_params = read_json(args.evidence_params, default={})
    ev_ocr = read_json(args.evidence_ocr, default={})
    ev_vlm = read_json(args.evidence_vlm, default={})
    bundles = pidx.build_bundles()

    rows = []
    for pid, attrs in acmt.items():
        cat = bundles[pid].category if pid in bundles else ""
        for aid in attrs:
            p = ev_params.get(pid, {}).get(aid, []) or []
            o = ev_ocr.get(pid, {}).get(aid, []) or []
            v = ev_vlm.get(pid, {}).get(aid, []) or []
            count = {"params": len(p), "ocr": len(o), "vlm": len(v)}
            coverage = sum(1 for n in count.values() if n > 0)
            rows.append({
                "fact_id": f"f{pid}__{aid}",
                "product_id": pid,
                "category": cat,
                "attribute_id": aid,
                "evidence_params": p,
                "evidence_ocr": o,
                "evidence_vlm": v,
                "evidence_count": count,
                "coverage": coverage,
                "confidence": config.CONFIDENCE_BY_COVERAGE.get(coverage, "absent"),
            })
    write_jsonl(args.out, rows)
    from collections import Counter
    cov = Counter(r["coverage"] for r in rows)
    print(f"[C5] fact_records={len(rows)} coverage_dist={dict(sorted(cov.items()))} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
