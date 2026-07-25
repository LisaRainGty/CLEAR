"""Build and validate the exact frozen paper training JSONL without arguments."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_SHA256 = "1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76"
EXPECTED_ROWS = 4883


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> dict:
    rows = 0
    pair_ids = set()
    source_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    argument_records = 0
    split_rows = {"train": 0, "val": 0, "test": 0}
    split_positive = {"train": 0, "val": 0, "test": 0}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", ""))
            if not pair_id or pair_id in pair_ids:
                raise ValueError(f"{path}:{number}: missing or duplicate pair_id={pair_id!r}")
            pair_ids.add(pair_id)
            split = str(row.get("split", ""))
            if split not in split_rows:
                raise ValueError(f"{path}:{number}: invalid split={split!r}")
            split_rows[split] += 1
            split_positive[split] += int(row.get("y", 0))
            source_count = sum(bool(row.get(key)) for key in (
                "evidence_params", "evidence_ocr", "evidence_vlm"))
            source_counts[source_count] += 1
            arguments = row.get("arguments", {}) or {}
            argument_records += int(any(str(arguments.get(key, "") or "").strip() for key in (
                "supporting_argument", "refuting_argument", "evidence_gap")))
            rows += 1
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "rows": rows,
        "split_rows": split_rows,
        "split_positive": split_positive,
        "source_availability_0_1_2_3": source_counts,
        "argument_records": argument_records,
    }
    if rows != EXPECTED_ROWS or result["sha256"] != EXPECTED_SHA256:
        raise ValueError(f"frozen training-set contract failed: {result}")
    if argument_records:
        raise ValueError("arguments are forbidden in the paper training set")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perception",
        default="data/final/repaired_v1/"
                "dataset_planbaseline_duallabel_FULLPOOL_supervised_20260614_stagec.jsonl",
    )
    parser.add_argument(
        "--objective-negative",
        default="data/final/repaired_v1/dataset_objective_negatives_v1_20260615.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="results/audit/paper_training_set_build.json")
    args = parser.parse_args()

    inputs = [Path(args.perception), Path(args.objective_negative)]
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        report = {"action": "validated_existing", **validate(output)}
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as destination:
            for source in inputs:
                with source.open("rb") as handle:
                    shutil.copyfileobj(handle, destination)
        temporary.replace(output)
        report = {"action": "built", "inputs": [str(path) for path in inputs],
                  **validate(output)}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
