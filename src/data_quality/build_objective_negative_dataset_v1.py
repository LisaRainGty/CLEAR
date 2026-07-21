"""Build objective-negative records (y=0) from "claim-without-comment" pairs.

These pairs have a streamer claim but NO aligned consumer comment, so the §2
perception weight collapses to 0 (f_sat=0). We instead assign an EVIDENCE-DRIVEN
weight: a claim that is corroborated by more product-fact sources (higher Stage-C
coverage) is a more confident "truthful / no perceived risk" negative. A global
PU discount kappa accounts for the "absence-of-complaint != truthful" noise.

Output schema is concatenation-compatible with the FULLPOOL dual-label dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import config
from common import product_index as pidx

CONFIDENCE_BY_COVERAGE = {3: "high", 2: "medium", 1: "low", 0: "absent"}
# evidence-confidence base by coverage, then PU discount
CONF_BASE = {3: 1.0, 2: 0.8, 1: 0.55, 0: 0.30}
KAPPA = 0.6           # presumed-negative (PU) discount
C_FLOOR = config.C_FLOOR


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def remap_image_path(p: str) -> str:
    if not p:
        return p
    parts = p.replace("\\", "/").split("/")
    if "data" in parts:
        return "/".join(parts[parts.index("data"):])
    return p


def remap_items(items):
    out = []
    for it in items or []:
        if isinstance(it, dict):
            it = dict(it)
            if isinstance(it.get("image_path"), str):
                it["image_path"] = remap_image_path(it["image_path"])
        out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/stageB_fullschema_gap/claim_no_comment_pairs_v1.jsonl")
    ap.add_argument("--fact_records", default="data/processed/stageC_neg/fact_records_neg.jsonl")
    ap.add_argument("--ref_dataset", default="data/final/repaired_v1/dataset_planbaseline_duallabel_FULLPOOL_all_20260614_stagec.jsonl")
    ap.add_argument("--out", default="data/final/repaired_v1/dataset_objective_negatives_v1_20260615.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/dataset_objective_negatives_v1_20260615.report.json")
    args = ap.parse_args()

    # fact_records for new pairs
    fr = {}
    for r in load_jsonl(args.fact_records):
        fr[(str(r["product_id"]), r["attribute_id"])] = r

    # room_id -> split from existing dataset (grouped, leakage-safe); also existing pair set
    room_split = {}
    existing_pairs = set()
    for r in load_jsonl(args.ref_dataset):
        rid = r.get("room_id")
        sp = r.get("split")
        if rid and sp and rid not in room_split:
            room_split[rid] = sp
        existing_pairs.add((str(r["product_id"]), r["attribute_id"]))

    import hashlib

    def split_for(rid: str) -> str:
        if rid in room_split:
            return room_split[rid]
        h = int(hashlib.md5(str(rid).encode()).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        if h < 80:
            return "val"
        return "test"

    bundles = pidx.build_bundles()
    rows = []
    stats = defaultdict(int)
    cov_dist = defaultdict(int)
    for p in load_jsonl(args.pairs):
        pid = str(p["product_id"])
        aid = p["attribute_id"]
        key = (pid, aid)
        if key in existing_pairs:
            stats["skip_overlap_existing"] += 1
            continue
        b = bundles.get(pid)
        rid = b.room_id if b else "UNKNOWN"
        fc = fr.get(key)
        ep = remap_items(fc.get("evidence_params", [])) if fc else []
        eo = remap_items(fc.get("evidence_ocr", [])) if fc else []
        ev = remap_items(fc.get("evidence_vlm", [])) if fc else []
        count = {"params": len(ep), "ocr": len(eo), "vlm": len(ev)}
        coverage = sum(1 for v in count.values() if v > 0)
        cov_dist[coverage] += 1
        c_obj = max(C_FLOOR, round(KAPPA * CONF_BASE[coverage], 4))
        rows.append({
            "pair_id": p["pair_id"],
            "product_id": pid,
            "room_id": rid,
            "category": p["category"],
            "subcategory": b.subcategory if b else "",
            "attribute_id": aid,
            "attribute_name": p["attribute_name"],
            "product_title": p.get("product_title", b.title if b else ""),
            "y": 0,
            "y_perception": 0,
            "label_observed": True,
            "c": c_obj,
            "c_reliability": c_obj,
            "sample_role": "objective_negative",
            "contrastive_mask": coverage >= 1,
            "contrastive_mask_reason": "evidence_backed_objective_negative" if coverage >= 1 else "no_evidence_objective_negative",
            "confidence": CONFIDENCE_BY_COVERAGE[coverage],
            "claim": p["claim"],
            "evidence_params": ep,
            "evidence_ocr": eo,
            "evidence_vlm": ev,
            "coverage": coverage,
            "evidence_count": count,
            "proposal_label_audit": {
                "basis": "claim_without_comment",
                "rule": "no aligned consumer comment -> no perceived false-advertising risk -> y=0",
                "weight_rule": f"c = max(C_FLOOR, KAPPA*CONF_BASE[coverage]); KAPPA={KAPPA}",
                "coverage": coverage,
            },
            "label_audit": {"objective_negative": True, "n_aligned_comments": 0},
            "_llm_review": {},
            "_aligned_consumer_mentions": [],
            "_consumer_mentions_total": 0,
            "_consumer_mentions_neg": 0,
            "_attribute_scope": "fullschema",
            "split": split_for(rid),
            "y_recon": 0,
            "c_recon": c_obj,
            "y_plan": 0,
            "c_plan": c_obj,
            "label_audit_plan": {
                "note": "objective negative; §2 perception weight=0 (no aligned comment); evidence-driven weight used",
                "coverage": coverage,
            },
            "_stagec_source": "factrecords_neg" if fc else "no_factrecord",
            "_origin": "fullschema_claim_no_comment",
            "n_claim_segments": p.get("n_claim_segments", 0),
        })
        stats["written"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "input_pairs": stats["written"] + stats["skip_overlap_existing"],
        "written": stats["written"],
        "skip_overlap_existing": stats["skip_overlap_existing"],
        "coverage_dist": dict(sorted(cov_dist.items())),
        "split_dist": _count(rows, "split"),
        "c_by_coverage": {cov: round(max(C_FLOOR, KAPPA * CONF_BASE[cov]), 4) for cov in (0, 1, 2, 3)},
        "out": args.out,
    }
    json.dump(report, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


def _count(rows, key):
    from collections import Counter
    return dict(Counter(r[key] for r in rows))


if __name__ == "__main__":
    main()
