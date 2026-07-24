#!/usr/bin/env python3
"""Generate the final paper config from the audited four-combination selector."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_config(
    base: dict,
    selection: dict,
    *,
    selection_path: str,
    selection_sha256: str,
) -> dict:
    if selection.get("test_metrics_accessed") is not False:
        raise RuntimeError("combo selector accessed test metrics")
    if selection.get("racl_mandatory") is not True:
        raise RuntimeError("combo selector did not keep RACL mandatory")
    if selection.get("dataset_sha256") != base["dataset"]["sha256"]:
        raise RuntimeError("combo selector dataset mismatch")
    if selection.get("evidence_policy") != base["fair_comparison"]["evidence_policy"]:
        raise RuntimeError("combo selector evidence-policy mismatch")
    selected = str(selection["selected_candidate"])
    spec = dict(selection["selected_spec"])
    if spec.get("racl_mandatory") is not True:
        raise RuntimeError("selected combo is not RACL-enabled")
    if bool(spec.get("attribute_blocked", True)):
        raise RuntimeError("selected combo is not global RACL")

    cfg = copy.deepcopy(base)
    cfg["status"] = "final_validation_selected_fusion_racl_combo_20260724"
    cfg["paper_suite"]["figure_namespace"] = (
        f"arguments_only_best_racl_{selected}"
    )
    cfg["paper_suite"]["architecture_selection_manifest"] = selection_path
    cfg["paper_suite"]["architecture_selection_manifest_sha256"] = (
        selection_sha256
    )
    claimarc = cfg["claimarc"]
    fusion_reference = copy.deepcopy(claimarc.get("locked_fusion", {}))
    no_fusion = bool(spec["no_fusion"])
    if no_fusion:
        claimarc.pop("locked_fusion", None)
        claimarc["no_fusion_main"] = True
        claimarc["with_fusion_reference"] = {
            "role": "Table 8 fusion ablation only",
            **fusion_reference,
        }
    else:
        claimarc["no_fusion_main"] = False
        claimarc.pop("with_fusion_reference", None)
    claimarc["attribute_blocked_contrast"] = False
    claimarc["class_balanced_contrast"] = bool(spec["class_balanced"])
    for cfg_key, spec_key in (
        ("warmup_epochs", "warmup_epochs"),
        ("contrastive_epochs", "contrastive_epochs"),
        ("lambda_cl", "lambda_cl"),
        ("tau", "tau"),
        ("kp", "kp"),
        ("kn", "kn"),
    ):
        claimarc[cfg_key] = spec[spec_key]
    claimarc["locked_racl"] = {
        "revision": "best_fusion_racl_combo_locked_20260724",
        "selection_manifest": selection_path,
        "selection_manifest_sha256": selection_sha256,
        "selected_candidate": selected,
        "selection_primary_metric": "validation auprc mean over seeds 0,1,2",
        "selection_tie_breaker": "validation positive-F1 mean",
        "test_metrics_accessed_during_selection": False,
        "warmup_epochs": int(spec["warmup_epochs"]),
        "contrastive_epochs": int(spec["contrastive_epochs"]),
        "lambda_cl": float(spec["lambda_cl"]),
        "tau": float(spec["tau"]),
        "kp": int(spec["kp"]),
        "kn": int(spec["kn"]),
        "exclude_self": bool(spec.get("exclude_self", False)),
        "hard_positive": bool(spec["hard_positive"]),
        "attribute_blocked": False,
        "class_balanced": bool(spec["class_balanced"]),
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base.resolve()
    selection_path = args.selection.resolve()
    output_path = args.output.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    try:
        selection_rel = str(selection_path.relative_to(ROOT))
    except ValueError as exc:
        raise RuntimeError("selection manifest must live inside the repository") from exc
    digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    cfg = build_config(
        base,
        selection,
        selection_path=selection_rel,
        selection_sha256=digest,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "selected_candidate": selection["selected_candidate"],
        "selection_manifest_sha256": digest,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
