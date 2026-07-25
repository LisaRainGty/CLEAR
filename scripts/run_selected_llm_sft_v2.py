#!/usr/bin/env python3
"""Run one locked supervised-LLM test seed after validation-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--expected_config_sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--save_pred", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    selection_path = Path(args.selection).resolve()
    actual_config_hash = sha256(config_path)
    if actual_config_hash != args.expected_config_sha256:
        raise RuntimeError("LLM SFT config changed after the runner was constructed")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("config_sha256") != actual_config_hash:
        raise RuntimeError("selection does not belong to the frozen LLM SFT config")
    if selection.get("test_metrics_accessed") is not False:
        raise RuntimeError("selection manifest did not prove test isolation")
    if selection.get("dataset_sha256") != cfg["dataset"]["sha256"]:
        raise RuntimeError("selection dataset hash mismatch")
    if selection.get("evidence_policy") != cfg["fair_comparison"]["evidence_policy"]:
        raise RuntimeError("selection evidence policy mismatch")
    if args.seed not in {int(seed) for seed in cfg["llm_sft_v2"]["seeds"]}:
        raise RuntimeError(f"seed {args.seed} is outside the frozen protocol")

    selected_name = str(selection["selected_candidate"])
    candidates = {str(item["name"]): item for item in cfg["llm_sft_v2"]["candidates"]}
    if selected_name not in candidates:
        raise RuntimeError(f"selected candidate {selected_name!r} is not in config")
    candidate = candidates[selected_name]
    dataset = (ROOT / cfg["dataset"]["path"]).resolve()
    if sha256(dataset) != cfg["dataset"]["sha256"]:
        raise RuntimeError("frozen arguments-only dataset hash mismatch")

    command = [
        sys.executable, "-m", "models.qwen_sft",
        "--dataset", str(dataset),
        "--model", str(candidate["model"]),
        "--tag", f"llm_sft_v2_locked_{selected_name}",
        "--seed", str(args.seed),
        "--evidence_policy", str(cfg["fair_comparison"]["evidence_policy"]),
        "--prompt_revision", str(cfg["llm_sft_v2"]["prompt_revision"]),
        "--epochs", str(int(candidate.get("epochs", 3))),
        "--lr", str(float(candidate["lr"])),
        "--rank", str(int(candidate["rank"])),
        "--dropout", str(float(candidate.get("dropout", 0.05))),
        "--target_scope", str(candidate.get("target_scope", "attention")),
        "--max_length", str(int(candidate.get("max_length", 512))),
        "--save_pred", str(Path(args.save_pred).resolve()),
    ]
    env = dict(os.environ)
    env.pop("CLAIMARC_QWEN_PATH", None)
    return subprocess.call(command, cwd=ROOT / "src", env=env)


if __name__ == "__main__":
    raise SystemExit(main())
