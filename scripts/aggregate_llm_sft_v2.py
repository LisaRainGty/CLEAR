#!/usr/bin/env python3
"""Validate and aggregate the locked three-seed supervised-LLM test runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("acc", "pos_f1", "macro_f1", "wF1", "auprc", "auroc", "ece")


def load_one(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one result in {path}, found {len(rows)}")
    return rows[0]


def equal_number(observed, expected) -> bool:
    return abs(float(observed) - float(expected)) <= 1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--expected_config_sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    actual_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if actual_config_hash != args.expected_config_sha256:
        raise RuntimeError("LLM SFT config changed before final aggregation")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = cfg["llm_sft_v2"]
    seeds = [int(seed) for seed in protocol["seeds"]]
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if selection.get("config_sha256") != actual_config_hash:
        raise RuntimeError("selection/config hash mismatch")
    if selection.get("test_metrics_accessed") is not False:
        raise RuntimeError("selection did not prove test isolation")
    selected_name = str(selection["selected_candidate"])
    candidates = {str(item["name"]): item for item in protocol["candidates"]}
    if selected_name not in candidates:
        raise RuntimeError("selected candidate is absent from the frozen config")
    candidate = candidates[selected_name]

    rows = []
    result_root = Path(args.result_root).resolve()
    for seed in seeds:
        path = result_root / "jobs" / f"llm_sft_v2_locked_s{seed}.jsonl"
        if not path.exists():
            raise RuntimeError(f"missing locked test result: {path}")
        row = load_one(path)
        if row.get("evaluation_split") != "test" or row.get("validation_only") is not False:
            raise RuntimeError(f"locked result is not a test-only report: {path}")
        if int(row.get("seed")) != seed:
            raise RuntimeError(f"seed mismatch in {path}")
        if row.get("dataset_sha256") != cfg["dataset"]["sha256"]:
            raise RuntimeError(f"dataset hash mismatch in {path}")
        if row.get("evidence_policy") != cfg["fair_comparison"]["evidence_policy"]:
            raise RuntimeError(f"evidence policy mismatch in {path}")
        if row.get("prompt_revision") != protocol["prompt_revision"]:
            raise RuntimeError(f"prompt revision mismatch in {path}")
        expected = {
            "model_requested": candidate["model"],
            "epochs": int(candidate.get("epochs", 3)),
            "learning_rate": float(candidate["lr"]),
            "lora_rank": int(candidate["rank"]),
            "lora_dropout": float(candidate.get("dropout", 0.05)),
            "target_scope": candidate.get("target_scope", "attention"),
            "max_length": int(candidate.get("max_length", 512)),
            "n_test": int(cfg["dataset"]["expected_split_rows"]["test"]),
            "pos_test": int(cfg["dataset"]["expected_split_positives"]["test"]),
        }
        for key, value in expected.items():
            observed = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                matches = observed is not None and equal_number(observed, value)
            else:
                matches = observed == value
            if not matches:
                raise RuntimeError(f"{key} mismatch in {path}: {observed!r} != {value!r}")
        rows.append(row)

    aggregates = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        aggregates[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": cfg["dataset"]["sha256"],
        "evidence_policy": cfg["fair_comparison"]["evidence_policy"],
        "prompt_revision": protocol["prompt_revision"],
        "selection_config_sha256": actual_config_hash,
        "selection_test_metrics_accessed": False,
        "selected_candidate": selected_name,
        "selected_hyperparameters": candidate,
        "seeds": seeds,
        "standard_deviation": "sample ddof=1",
        "runs": rows,
        "metrics": aggregates,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {
        "acc": "Accuracy", "pos_f1": "F1_pos", "macro_f1": "Macro-F1",
        "wF1": "wF1", "auprc": "AP/AUPRC", "auroc": "ROC-AUC", "ece": "ECE",
    }
    cells = []
    for metric in METRICS:
        summary = aggregates[metric]
        cells.append(f"{summary['mean']:.4f} ± {summary['std']:.4f}")
    header = "| Model | " + " | ".join(labels[m] for m in METRICS) + " |"
    separator = "|---" * (len(METRICS) + 1) + "|"
    row = "| Qwen2.5-7B-Instruct QLoRA (locked) | " + " | ".join(cells) + " |"
    output_md = Path(args.output_md)
    output_md.write_text(
        "# Locked supervised LLM result\n\n"
        f"Selected on validation only: `{selected_name}`. "
        "Values are mean ± population SD over seeds 0, 1, 2.\n\n"
        f"{header}\n{separator}\n{row}\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_candidate": selected_name, "metrics": aggregates},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
