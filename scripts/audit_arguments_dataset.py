#!/usr/bin/env python3
"""Audit exact lineage and leakage controls for the arguments-only dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from generate_arguments_dataset import (
    ARGUMENT_FIELDS,
    FORBIDDEN_INPUT_FIELDS,
    NO_SOURCE_ARGUMENTS,
    VALIDATOR_REVISION,
    allowed_payload,
    canonical_hash,
    has_source,
    read_jsonl,
    sha256_file,
    validate_arguments,
    validate_grounding,
)


ALLOWED_PAYLOAD_KEYS = {"attribute_name", "claim", "PARAM", "OCR", "VLM"}
DERIVED_KEYS = {"arguments", "_argument_generation"}


def stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = list(read_jsonl(args.source))
    dataset = list(read_jsonl(args.dataset))
    cache = list(read_jsonl(args.cache))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("status") != "complete":
        errors.append("generation manifest is not complete")
    generator_path = Path(__file__).resolve().with_name("generate_arguments_dataset.py")
    if manifest.get("generator", {}).get("script_sha256") != sha256_file(generator_path):
        errors.append("generator script SHA-256 does not match manifest")
    if manifest.get("selection", {}).get("mode") != "all":
        errors.append("final dataset was not generated from the full source")
    if manifest.get("validator_revision") != VALIDATOR_REVISION:
        errors.append("validator revision does not match the audited implementation")
    if sha256_file(args.source) != manifest.get("source_sha256"):
        errors.append("source SHA-256 does not match generation manifest")
    if sha256_file(args.dataset) != manifest.get("output_sha256"):
        errors.append("output SHA-256 does not match generation manifest")
    if len(source) != len(dataset) or len(dataset) != manifest.get("rows"):
        errors.append("source/dataset/manifest row counts differ")

    valid_cache: dict[tuple[str, str, str, str], dict] = {}
    cache_status = Counter()
    for record in cache:
        cache_status[str(record.get("status", "missing"))] += 1
        request_prompt = record.get("request_prompt")
        request_prompt_sha = str(record.get("request_prompt_sha256", ""))
        if not isinstance(request_prompt, str):
            errors.append(f"{record.get('pair_id')}: raw-cache request prompt missing")
        elif hashlib.sha256(request_prompt.encode("utf-8")).hexdigest() != request_prompt_sha:
            errors.append(f"{record.get('pair_id')}: raw-cache request prompt hash mismatch")
        if record.get("mode") == "model" and not request_prompt:
            errors.append(f"{record.get('pair_id')}: model request prompt is empty")
        api_response = record.get("api_response")
        if api_response is not None:
            if canonical_hash(api_response) != record.get("api_response_sha256"):
                errors.append(f"{record.get('pair_id')}: API response hash mismatch")
            try:
                response_text = api_response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                errors.append(f"{record.get('pair_id')}: malformed retained API response")
            else:
                if response_text != record.get("raw_text"):
                    errors.append(f"{record.get('pair_id')}: API response/raw text mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict) or set(payload) != ALLOWED_PAYLOAD_KEYS:
            errors.append(f"{record.get('pair_id')}: cache payload key contract failed")
            continue
        forbidden = set(payload).intersection(FORBIDDEN_INPUT_FIELDS)
        if forbidden:
            errors.append(f"{record.get('pair_id')}: forbidden cache inputs {sorted(forbidden)}")
        if canonical_hash(payload) != record.get("input_sha256"):
            errors.append(f"{record.get('pair_id')}: cache input hash mismatch")
        if record.get("status") == "ok":
            try:
                normalized = validate_grounding(record.get("arguments"), payload)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{record.get('pair_id')}: invalid successful cache row: {exc}")
                continue
            item = dict(record)
            item["arguments"] = normalized
            key = (str(record.get("pair_id", "")), str(record.get("input_sha256", "")),
                   str(record.get("prompt_sha256", "")),
                   str(record.get("generation_sha256", "")))
            valid_cache[key] = item

    modes = Counter()
    fields = Counter()
    for index, (before, after) in enumerate(zip(source, dataset), 1):
        pair_id = str(before.get("pair_id", ""))
        if pair_id != str(after.get("pair_id", "")):
            errors.append(f"row {index}: pair_id/order changed")
            continue
        stripped = {key: value for key, value in after.items() if key not in DERIVED_KEYS}
        if stable(before) != stable(stripped):
            errors.append(f"{pair_id}: a frozen source field changed")
        payload = allowed_payload(before)
        generation = after.get("_argument_generation", {}) or {}
        input_sha = canonical_hash(payload)
        prompt_sha = str(generation.get("prompt_sha256", ""))
        generation_sha = str(generation.get("generation_sha256", ""))
        if generation.get("backend") != manifest.get("model", {}).get("backend"):
            errors.append(f"{pair_id}: final backend mismatch")
        if input_sha != generation.get("input_sha256"):
            errors.append(f"{pair_id}: final input hash mismatch")
        if prompt_sha != manifest.get("prompt_sha256"):
            errors.append(f"{pair_id}: final prompt hash mismatch")
        if generation_sha != manifest.get("model", {}).get("generation_sha256"):
            errors.append(f"{pair_id}: final generation hash mismatch")
        if generation.get("label_blind") is not True:
            errors.append(f"{pair_id}: label_blind flag is not true")
        cache_row = valid_cache.get((pair_id, input_sha, prompt_sha, generation_sha))
        if cache_row is None:
            errors.append(f"{pair_id}: no matching successful raw-cache row")
            continue
        try:
            final_arguments = validate_grounding(after.get("arguments"), payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pair_id}: invalid final arguments: {exc}")
            continue
        if final_arguments != cache_row["arguments"]:
            errors.append(f"{pair_id}: final arguments differ from raw cache")
        mode = str(generation.get("mode", ""))
        modes[mode] += 1
        for field in ARGUMENT_FIELDS:
            fields[field] += int(bool(final_arguments[field]))
        if has_source(payload):
            if mode != "model":
                errors.append(f"{pair_id}: source-bearing row was not model generated")
        else:
            if mode != "deterministic_no_source":
                errors.append(f"{pair_id}: source-free row did not use deterministic gap")
            if final_arguments != NO_SOURCE_ARGUMENTS:
                errors.append(f"{pair_id}: source-free argument text changed")

    expected_counts = manifest.get("counts", {})
    if modes["model"] != expected_counts.get("model_generated"):
        errors.append("model-generated count differs from manifest")
    if modes["deterministic_no_source"] != expected_counts.get("deterministic_no_source"):
        errors.append("deterministic no-source count differs from manifest")
    if cache_status["invalid"]:
        warnings.append(
            f"append-only cache retains {cache_status['invalid']} rejected attempts; "
            "these are provenance only and never enter the final dataset"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "source": str(args.source),
        "dataset": str(args.dataset),
        "source_sha256": sha256_file(args.source),
        "dataset_sha256": sha256_file(args.dataset),
        "rows": len(dataset),
        "cache_records": len(cache),
        "cache_status": dict(cache_status),
        "generation_modes": dict(modes),
        "argument_field_coverage": dict(fields),
        "checks": {
            "labels_and_all_frozen_fields_unchanged": not any(
                "frozen source field" in error for error in errors),
            "raw_cache_payload_is_label_blind": not any(
                "cache payload" in error or "forbidden cache" in error for error in errors),
            "every_final_argument_has_raw_lineage": not any(
                "matching successful raw-cache" in error for error in errors),
            "source_free_rows_use_fixed_gap": not any(
                "source-free" in error for error in errors),
        },
        "errors": errors,
        "warnings": warnings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(dataset),
                      "errors": len(errors), "warnings": len(warnings)},
                     ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
