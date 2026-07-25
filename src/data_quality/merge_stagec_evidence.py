"""Merge frozen Stage-C PARAM/OCR/VLM facts into reconstructed pair records.

The historical FULLPOOL repair pipeline kept reconstruction-time evidence for
audit, then replaced the model-facing source blocks with the final Stage-C fact
record for the same (product_id, attribute_id).  This script restores that
previously missing deterministic step and preserves the exact field order used
by the frozen 2026-06-14 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_image_path(value: str) -> str:
    if not value:
        return value
    parts = value.replace("\\", "/").split("/")
    return "/".join(parts[parts.index("data"):]) if "data" in parts else value


def remap_items(items):
    out = []
    for item in items or []:
        item = dict(item)
        if isinstance(item.get("image_path"), str):
            item["image_path"] = remap_image_path(item["image_path"])
        out.append(item)
    return out


def pair_key(row):
    return str(row.get("product_id", "")), str(row.get("attribute_id", ""))


def merge_record(row, fact):
    rec = dict(row)
    # Append audit copies in the same order as the frozen artifact.
    rec["_evidence_recon_params"] = rec.get("evidence_params", []) or []
    rec["_evidence_recon_ocr"] = rec.get("evidence_ocr", []) or []
    rec["_evidence_recon_vlm"] = rec.get("evidence_vlm", []) or []
    rec["_coverage_recon"] = rec.get("coverage", 0)
    rec["_confidence_recon"] = rec.get("confidence", "absent")
    # Assigning existing keys retains their original JSON insertion position.
    rec["evidence_params"] = remap_items(fact.get("evidence_params", []))
    rec["evidence_ocr"] = remap_items(fact.get("evidence_ocr", []))
    rec["evidence_vlm"] = remap_items(fact.get("evidence_vlm", []))
    rec["evidence_count"] = fact.get("evidence_count", {
        "params": len(rec["evidence_params"]),
        "ocr": len(rec["evidence_ocr"]),
        "vlm": len(rec["evidence_vlm"]),
    })
    rec["coverage"] = int(fact.get("coverage", sum(
        bool(rec["evidence_count"].get(name, 0)) for name in ("params", "ocr", "vlm")
    )))
    rec["confidence"] = fact.get("confidence", {0: "absent", 1: "low", 2: "medium", 3: "high"}[
        rec["coverage"]
    ])
    rec["_stagec_source"] = "factrecords"
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fact-records",
        default="data/processed/stageC/fact_records_plus_gap_20260615.jsonl",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    fact_path = Path(args.fact_records)
    input_path = Path(args.input)
    output_path = Path(args.output)
    facts = {}
    for fact in read_jsonl(fact_path):
        key = pair_key(fact)
        if key in facts:
            raise ValueError(f"duplicate Stage-C fact record: {key}")
        facts[key] = fact

    rows = []
    before, after = Counter(), Counter()
    missing = []
    for row in read_jsonl(input_path):
        before[int(row.get("coverage", 0) or 0)] += 1
        fact = facts.get(pair_key(row))
        if fact is None:
            missing.append(pair_key(row))
            continue
        merged = merge_record(row, fact)
        after[int(merged["coverage"])] += 1
        rows.append(merged)
    if missing:
        raise ValueError(f"{len(missing)} input pairs have no Stage-C fact record; first={missing[:3]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "fact_records": len(facts),
        "rows": len(rows),
        "missing": 0,
        "coverage_before": {str(k): before[k] for k in sorted(before)},
        "coverage_after": {str(k): after[k] for k in sorted(after)},
        "input": str(input_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
