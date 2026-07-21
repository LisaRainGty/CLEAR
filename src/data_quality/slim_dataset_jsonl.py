"""Remove unused fields from a CLAIMARC dataset.jsonl.

Keeps only fields read by training / evaluation code (models.*, xdom) and
minimal nested evidence / mention payloads required for tokenization, c
recompute, and cross-domain splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEEP_TOP = {
    "pair_id",
    "product_id",
    "room_id",
    "category",
    "subcategory",
    "attribute_id",
    "attribute_name",
    "product_title",
    "split",
    "y",
    "c",
    "confidence",
    "sample_role",
    "contrastive_mask",
    "claim",
    "evidence_params",
    "evidence_ocr",
    "evidence_vlm",
    "evidence_count",
    "coverage",
    "proposal_label_audit",
    "_aligned_consumer_mentions",
    "_consumer_mentions_total",
}

KEEP_PROPOSAL_AUDIT = {"aligned_comment_count", "old_c"}

KEEP_CLAIM = {"has_claim_srt", "passage", "segments"}
KEEP_SEGMENT = {"text"}
KEEP_PARAM = {"param_key", "raw_text"}
KEEP_OCR = {"raw_text"}
KEEP_VLM = {"raw_quote"}
KEEP_MENTION = {
    "review_id",
    "evidence_span",
    "polarity",
    "review_polarity",
    "mention_strength",
    "explicit_fact_hit",
    "review_time",
    "_judgment",
}
KEEP_JUDGMENT = {"relation", "aligned_to_claim", "reason"}


def _pick(obj: dict, keys: set[str]) -> dict:
    return {k: obj[k] for k in keys if k in obj}


def slim_claim(claim: dict | None) -> dict:
    claim = claim or {}
    segs = [_pick(s, KEEP_SEGMENT) for s in (claim.get("segments") or [])]
    out = _pick(claim, KEEP_CLAIM)
    out["segments"] = segs
    return out


def slim_evidence_params(items: list | None) -> list[dict]:
    return [_pick(it, KEEP_PARAM) for it in (items or [])]


def slim_evidence_ocr(items: list | None) -> list[dict]:
    return [_pick(it, KEEP_OCR) for it in (items or [])]


def slim_evidence_vlm(items: list | None) -> list[dict]:
    return [_pick(it, KEEP_VLM) for it in (items or [])]


def slim_mention(m: dict) -> dict:
    out = _pick(m, KEEP_MENTION)
    j = m.get("_judgment") or {}
    if j:
        out["_judgment"] = _pick(j, KEEP_JUDGMENT)
    return out


def slim_proposal_audit(audit: dict | None) -> dict:
    audit = audit or {}
    return _pick(audit, KEEP_PROPOSAL_AUDIT)


def slim_record(rec: dict) -> dict:
    out: dict = {}
    for k in KEEP_TOP:
        if k not in rec:
            continue
        v = rec[k]
        if k == "claim":
            out[k] = slim_claim(v)
        elif k == "evidence_params":
            out[k] = slim_evidence_params(v)
        elif k == "evidence_ocr":
            out[k] = slim_evidence_ocr(v)
        elif k == "evidence_vlm":
            out[k] = slim_evidence_vlm(v)
        elif k == "proposal_label_audit":
            slimmed = slim_proposal_audit(v)
            if slimmed:
                out[k] = slimmed
        elif k == "_aligned_consumer_mentions":
            out[k] = [slim_mention(m) for m in (v or [])]
        else:
            out[k] = v
    return out


def slim_file(src: Path, dst: Path | None = None) -> dict:
    dst = dst or src
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    n = 0
    before_bytes = src.stat().st_size
    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fout.write(json.dumps(slim_record(rec), ensure_ascii=False) + "\n")
            n += 1
    after_bytes = tmp.stat().st_size
    tmp.replace(dst)
    return {
        "rows": n,
        "before_mb": round(before_bytes / (1024 * 1024), 2),
        "after_mb": round(after_bytes / (1024 * 1024), 2),
        "saved_mb": round((before_bytes - after_bytes) / (1024 * 1024), 2),
        "path": str(dst),
    }


def default_output_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_slim{src.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="source dataset.jsonl")
    ap.add_argument(
        "--output",
        default="",
        help="output path (default: <input_stem>_slim.jsonl, never overwrites input)",
    )
    args = ap.parse_args()
    src = Path(args.input)
    dst = Path(args.output) if args.output else default_output_path(src)
    if dst.resolve() == src.resolve():
        raise SystemExit("[slim] output must differ from input; pass --output explicitly")
    stats = slim_file(src, dst)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
