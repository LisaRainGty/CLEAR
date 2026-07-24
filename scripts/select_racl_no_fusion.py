#!/usr/bin/env python3
"""Select RACL using validation-only matched-seed runs on a frozen architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("acc", "pos_f1", "macro_f1", "auprc", "auroc", "wF1", "ece")


def load_one(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one result in {path}, found {len(rows)}")
    return rows[0]


def select_candidate(aggregates: dict[str, dict], selection: dict) -> tuple[str, list[str], dict]:
    """Select the best mandatory RACL candidate; no-RACL is diagnostic only."""
    control_name = str(selection["diagnostic_control"])
    if control_name not in aggregates:
        raise RuntimeError(f"missing diagnostic control: {control_name}")
    control = aggregates[control_name]

    diagnostics = {}
    for name, summary in aggregates.items():
        if name == control_name:
            diagnostics[name] = {
                "selectable": False,
                "role": "diagnostic_control",
                "matched_seed_ap_wins": 0,
                "mean_ap_delta": 0.0,
                "mean_pos_f1_delta": 0.0,
            }
            continue
        ap_delta = summary["auprc"]["mean"] - control["auprc"]["mean"]
        f1_delta = summary["pos_f1"]["mean"] - control["pos_f1"]["mean"]
        wins = sum(
            candidate_ap > control_ap
            for candidate_ap, control_ap in zip(
                summary["auprc"]["values"], control["auprc"]["values"]
            )
        )
        diagnostics[name] = {
            "selectable": True,
            "role": "racl_candidate",
            "matched_seed_ap_wins": int(wins),
            "mean_ap_delta": float(ap_delta),
            "mean_pos_f1_delta": float(f1_delta),
        }

    primary = str(selection["primary_metric"])
    tie_breaker = str(selection["tie_breaker"])
    ranked_racl = sorted(
        (name for name in aggregates if name != control_name),
        key=lambda name: (
            -aggregates[name][primary]["mean"],
            -aggregates[name][tie_breaker]["mean"],
            name,
        ),
    )
    if not ranked_racl:
        raise RuntimeError("no RACL-enabled candidate is available")
    return ranked_racl[0], ranked_racl, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = cfg.get("racl_tuning", cfg.get("racl_no_fusion"))
    if not protocol:
        raise RuntimeError("missing racl_tuning protocol")
    architecture = str(protocol.get("architecture", "no_fusion"))
    job_prefix = str(protocol.get("job_prefix", "racl_nf"))
    fixed_fusion = protocol.get("fixed_fusion")
    seeds = [int(seed) for seed in protocol["seeds"]]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"frozen RACL seeds changed: {seeds}")
    expected_hash = cfg["dataset"]["sha256"]
    expected_policy = cfg["fair_comparison"]["evidence_policy"]
    expected_n_val = int(cfg["dataset"]["expected_split_rows"]["val"])
    expected_pos_val = int(cfg["dataset"]["expected_split_positives"]["val"])
    result_root = Path(args.result_root).resolve()
    parent_phase = protocol.get("parent_phase")
    parent_summary = None
    if parent_phase:
        parent_path = (config_path.parents[1] / str(
            parent_phase["selection_manifest"]
        )).resolve()
        parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if parent_hash != parent_phase["selection_manifest_sha256"]:
            raise RuntimeError("parent RACL selection manifest hash mismatch")
        parent_manifest = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent_manifest.get("test_metrics_accessed") is not False:
            raise RuntimeError("parent RACL phase did not prove test isolation")
        if parent_manifest.get("dataset_sha256") != expected_hash:
            raise RuntimeError("parent RACL dataset hash mismatch")
        if parent_manifest.get("evidence_policy") != expected_policy:
            raise RuntimeError("parent RACL evidence policy mismatch")
        expected_parent_architecture = parent_phase.get("architecture")
        if (expected_parent_architecture is not None
                and parent_manifest.get("architecture") not in (
                    None, expected_parent_architecture)):
            raise RuntimeError("parent RACL architecture mismatch")
        source_config_path = parent_phase.get("source_config")
        if source_config_path:
            source_path = (config_path.parents[1] / str(source_config_path)).resolve()
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if source_hash != parent_phase["source_config_sha256"]:
                raise RuntimeError("parent source config hash mismatch")
        parent_name = str(parent_phase.get(
            "source_candidate", parent_manifest["selected_candidate"]
        ))
        if parent_manifest.get("selected_candidate") != parent_name:
            raise RuntimeError("parent source candidate was not validation-selected")
        if parent_name not in parent_manifest["aggregates"]:
            raise RuntimeError("parent selected candidate has no aggregate")
        parent_alias = str(parent_phase.get("candidate_alias", parent_name))
        parent_summary = {
            "phase": str(parent_phase["phase"]),
            "candidate": parent_alias,
            "source_candidate": parent_name,
            "spec": parent_phase.get("spec", {}),
            "selection_manifest": str(parent_path),
            "selection_manifest_sha256": parent_hash,
            "aggregate": parent_manifest["aggregates"][parent_name],
        }

    aggregates: dict[str, dict] = {}
    candidate_specs = {
        str(candidate["name"]): candidate for candidate in protocol["candidates"]
    }
    for name, candidate in candidate_specs.items():
        rows = []
        for seed in seeds:
            path = result_root / "jobs" / f"{job_prefix}_{name}_s{seed}.jsonl"
            if not path.exists():
                raise RuntimeError(f"missing candidate result: {path}")
            row = load_one(path)
            if row.get("evaluation_split") != "validation" or row.get("validation_only") is not True:
                raise RuntimeError(f"non-validation result forbidden in tuning: {path}")
            forbidden = {"n_test", "pos_test", "auprc_rkc", "auroc_rkc", "acc_rkc"}
            if forbidden.intersection(row):
                raise RuntimeError(f"test/RKC metric leaked into tuning result: {path}")
            if row.get("dataset_sha256") != expected_hash:
                raise RuntimeError(f"dataset hash mismatch in {path}")
            if row.get("evidence_policy") != expected_policy:
                raise RuntimeError(f"evidence-policy mismatch in {path}")
            if int(row.get("seed")) != seed:
                raise RuntimeError(f"seed mismatch in {path}")
            if int(row.get("n_val")) != expected_n_val or int(row.get("pos_val")) != expected_pos_val:
                raise RuntimeError(f"validation split cardinality mismatch in {path}")
            if architecture == "no_fusion":
                if row.get("no_fusion") is not True:
                    raise RuntimeError(f"fusion was not disabled in {path}")
            elif architecture == "locked_fusion_global":
                if row.get("no_fusion") is not False:
                    raise RuntimeError(f"fusion was not enabled in {path}")
                if not fixed_fusion:
                    raise RuntimeError("locked-fusion RACL tuning lacks fixed_fusion")
                for key, row_key in (
                    ("n_fusion", "n_fusion"),
                    ("heads", "heads"),
                    ("fusion_dropout", "fusion_dropout"),
                    ("lr_fusion", "lr_fusion"),
                ):
                    if abs(float(row.get(row_key)) - float(fixed_fusion[key])) > 1e-12:
                        raise RuntimeError(f"{row_key} mismatch in {path}")
            else:
                raise RuntimeError(f"unsupported RACL architecture: {architecture}")
            enabled = bool(candidate["racl_enabled"])
            if bool(row.get("no_cl")) == enabled:
                raise RuntimeError(f"RACL enablement mismatch in {path}")
            expected_lambda = float(candidate["lambda_cl"])
            if abs(float(row.get("lambda_cl")) - expected_lambda) > 1e-12:
                raise RuntimeError(f"lambda_cl mismatch in {path}")
            if abs(float(row.get("tau")) - float(candidate["tau"])) > 1e-12:
                raise RuntimeError(f"tau mismatch in {path}")
            if int(row.get("Kp")) != int(candidate["kp"]) or int(row.get("Kn")) != int(candidate["kn"]):
                raise RuntimeError(f"Kp/Kn mismatch in {path}")
            if bool(row.get("cl_exclude_self")) != bool(candidate.get("exclude_self", False)):
                raise RuntimeError(f"self-exclusion mismatch in {path}")
            if bool(row.get("cl_hard_pos")) != bool(candidate.get("hard_positive", False)):
                raise RuntimeError(f"hard-positive mismatch in {path}")
            if bool(row.get("cl_attribute_blocked")) != bool(
                    candidate.get("attribute_blocked", False)):
                raise RuntimeError(f"attribute-blocking mismatch in {path}")
            if abs(float(row.get("cl_c_min")) - float(candidate.get("cl_c_min", 0.0))) > 1e-12:
                raise RuntimeError(f"cl_c_min mismatch in {path}")
            if abs(float(row.get("cl_neg_c_min")) - float(candidate.get("cl_neg_c_min", 0.0))) > 1e-12:
                raise RuntimeError(f"cl_neg_c_min mismatch in {path}")
            expected_warmup = int(candidate.get(
                "warmup_epochs", cfg.get("claimarc", {}).get("warmup_epochs", 3)
            ))
            expected_cl_epochs = int(candidate.get(
                "contrastive_epochs",
                cfg.get("claimarc", {}).get("contrastive_epochs", 6),
            ))
            if int(row.get("warmup_epochs")) != expected_warmup:
                raise RuntimeError(f"warmup epoch mismatch in {path}")
            if int(row.get("contrastive_epochs")) != expected_cl_epochs:
                raise RuntimeError(f"contrastive epoch mismatch in {path}")
            rows.append(row)

        summary = {"spec": candidate, "seeds": seeds, "runs": rows}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "values": values,
            }
        aggregates[name] = summary

    selection = protocol["selection"]
    primary = str(selection["primary_metric"])
    phase_selected, ranked_racl, diagnostics = select_candidate(
        aggregates, selection
    )
    selected = phase_selected
    selected_source = "current_phase"
    cross_phase_ranked = [f"current_phase:{name}" for name in ranked_racl]
    if parent_summary is not None:
        primary = str(selection["primary_metric"])
        tie_breaker = str(selection["tie_breaker"])
        candidates = {
            **{f"current_phase:{name}": summary
               for name, summary in aggregates.items()
               if diagnostics[name]["selectable"]},
            f"{parent_summary['phase']}:{parent_summary['candidate']}":
                parent_summary["aggregate"],
        }
        cross_phase_ranked = sorted(
            candidates,
            key=lambda name: (
                -candidates[name][primary]["mean"],
                -candidates[name][tie_breaker]["mean"],
                name,
            ),
        )
        winner_source, selected = cross_phase_ranked[0].split(":", 1)
        selected_source = (
            "current_phase" if winner_source == "current_phase" else winner_source
        )

    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_sha256": expected_hash,
        "evidence_policy": expected_policy,
        "architecture": architecture,
        "test_metrics_accessed": False,
        "selection": selection,
        "decision_provenance": protocol.get("decision_provenance"),
        "racl_mandatory": True,
        "phase_selected_candidate": phase_selected,
        "selected_candidate": selected,
        "selected_source": selected_source,
        "ranked_racl_candidates": ranked_racl,
        "cross_phase_ranked_candidates": cross_phase_ranked,
        "parent_phase_summary": parent_summary,
        "diagnostics_vs_no_racl": diagnostics,
        "aggregates": aggregates,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_aggregate = (
        aggregates[selected]
        if selected_source == "current_phase"
        else parent_summary["aggregate"]
    )
    print(json.dumps({
        "racl_mandatory": True,
        "selected_candidate": selected,
        "selected_source": selected_source,
        "primary_metric": primary,
        "validation_mean": selected_aggregate[primary]["mean"],
        "test_metrics_accessed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
