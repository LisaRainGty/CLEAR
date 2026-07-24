#!/usr/bin/env python3
"""Audit and select the preregistered dual-space RACL using validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


METRICS = ("acc", "pos_f1", "macro_f1", "wF1", "auprc", "auroc", "ece")
FORBIDDEN_RESULT_KEYS = {
    "n_test", "pos_test", "auprc_rkc", "auroc_rkc", "acc_rkc",
    "pos_f1_rkc", "macro_f1_rkc", "wF1_rkc", "alpha_rkc",
}


def _load_one(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one validation result in {path}, found {len(rows)}")
    return rows[0]


def _prediction_bundle(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"validation prediction hash mismatch: {path}")
    data = np.load(path, allow_pickle=False)
    required = {"p", "y", "c", "pair_id"}
    if set(data.files) != required:
        raise RuntimeError(f"unexpected validation prediction fields in {path}")
    return {key: np.asarray(data[key]) for key in required}


def _paired_bootstrap(candidate: list[dict[str, np.ndarray]],
                      control: list[dict[str, np.ndarray]],
                      repetitions: int, seed: int) -> dict:
    if len(candidate) != len(control):
        raise RuntimeError("paired bootstrap seed count mismatch")
    n = len(candidate[0]["y"])
    reference_y = candidate[0]["y"]
    reference_ids = candidate[0]["pair_id"]
    for bundle in candidate + control:
        if not np.array_equal(bundle["y"], reference_y):
            raise RuntimeError("validation labels differ between paired runs")
        if not np.array_equal(bundle["pair_id"], reference_ids):
            raise RuntimeError("validation pair order differs between paired runs")
    rng = np.random.RandomState(seed)
    values = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        index = rng.randint(0, n, size=n)
        y = reference_y[index]
        candidate_ap = np.mean([
            average_precision_score(y, bundle["p"][index])
            for bundle in candidate
        ])
        control_ap = np.mean([
            average_precision_score(y, bundle["p"][index])
            for bundle in control
        ])
        values[repetition] = candidate_ap - control_ap
    return {
        "repetitions": repetitions,
        "seed": seed,
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = cfg["racl_tuning"]
    selection = protocol["selection"]
    seeds = [int(seed) for seed in protocol["seeds"]]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"frozen seeds changed: {seeds}")
    if protocol.get("test_access_forbidden") is not True:
        raise RuntimeError("dual-space tuning must explicitly forbid test access")
    expected_hash = str(cfg["dataset"]["sha256"])
    expected_policy = str(cfg["fair_comparison"]["evidence_policy"])
    expected_n_val = int(cfg["dataset"]["expected_split_rows"]["val"])
    expected_pos_val = int(cfg["dataset"]["expected_split_positives"]["val"])
    result_root = Path(args.result_root).resolve()
    semantic_cache = (project_root / str(protocol["semantic_cache"])).resolve()
    if not semantic_cache.exists():
        raise RuntimeError(f"missing frozen semantic cache: {semantic_cache}")
    cached = np.load(semantic_cache, allow_pickle=False)
    semantic_cache_sha256 = hashlib.sha256(semantic_cache.read_bytes()).hexdigest()
    if semantic_cache_sha256 != protocol["semantic_cache_sha256"]:
        raise RuntimeError("semantic cache SHA-256 mismatch")
    if protocol.get("semantic_cache_splits") != ["train", "val"]:
        raise RuntimeError("semantic cache split declaration changed")
    if any(key.startswith("test_") for key in cached.files):
        raise RuntimeError("validation-only semantic cache contains forbidden test data")
    if str(cached["dataset_sha256"].item()) != expected_hash:
        raise RuntimeError("semantic cache dataset hash mismatch")
    if str(cached["revision"].item()) != protocol["semantic_revision"]:
        raise RuntimeError("semantic cache revision mismatch")
    candidates = {
        str(candidate["name"]): candidate
        for candidate in protocol["candidates"]
    }
    control_name = str(selection["diagnostic_control"])
    if control_name not in candidates or candidates[control_name]["racl_enabled"]:
        raise RuntimeError("invalid matched no-RACL control")

    aggregates = {}
    prediction_bundles = {}
    for name, spec in candidates.items():
        rows = []
        prediction_bundles[name] = []
        for seed in seeds:
            result_path = (
                result_root / "jobs"
                / f"{protocol['job_prefix']}_{name}_s{seed}.jsonl"
            )
            if not result_path.exists():
                raise RuntimeError(f"missing candidate result: {result_path}")
            row = _load_one(result_path)
            if row.get("evaluation_split") != "validation":
                raise RuntimeError(f"non-validation evaluation in {result_path}")
            if row.get("validation_only") is not True:
                raise RuntimeError(f"validation_only flag missing in {result_path}")
            leaked = FORBIDDEN_RESULT_KEYS.intersection(row)
            if leaked:
                raise RuntimeError(f"test/RKC keys leaked in {result_path}: {sorted(leaked)}")
            if row.get("dataset_sha256") != expected_hash:
                raise RuntimeError(f"dataset hash mismatch in {result_path}")
            if row.get("evidence_policy") != expected_policy:
                raise RuntimeError(f"evidence policy mismatch in {result_path}")
            if int(row.get("seed")) != seed:
                raise RuntimeError(f"seed mismatch in {result_path}")
            if int(row.get("n_val")) != expected_n_val:
                raise RuntimeError(f"validation row count mismatch in {result_path}")
            if int(row.get("pos_val")) != expected_pos_val:
                raise RuntimeError(f"validation positive count mismatch in {result_path}")
            if row.get("no_fusion") is not False:
                raise RuntimeError(f"locked fusion disabled in {result_path}")
            fixed_fusion = protocol["fixed_fusion"]
            for key in ("n_fusion", "heads", "fusion_dropout", "lr_fusion"):
                if abs(float(row.get(key)) - float(fixed_fusion[key])) > 1e-12:
                    raise RuntimeError(f"{key} mismatch in {result_path}")
            enabled = bool(spec["racl_enabled"])
            if bool(row.get("no_cl")) == enabled:
                raise RuntimeError(f"RACL enablement mismatch in {result_path}")
            checks = {
                "racl_dual_space": bool(spec["dual_space"]),
                "racl_memory_context": bool(spec["memory_context"]),
                "racl_all_samples": bool(spec["dual_space"]),
                "racl_local_margin": bool(spec["dual_space"]),
                "cl_exclude_self": bool(spec["exclude_self"]),
                "cl_hard_pos": bool(spec["hard_positive"]),
                "cl_set_nce": bool(spec["set_nce"]),
                "cl_attribute_blocked": bool(spec["attribute_blocked"]),
                "cl_class_balanced": bool(spec["class_balanced"]),
            }
            for key, expected in checks.items():
                if bool(row.get(key)) != expected:
                    raise RuntimeError(f"{key} mismatch in {result_path}")
            numeric = {
                "lambda_cl": spec["lambda_cl"],
                "tau": spec["tau"],
                "Kp": spec["kp"],
                "Kn": spec["kn"],
                "racl_logit_alpha": spec["racl_logit_alpha"],
                "racl_memory_head_alpha": spec["racl_memory_head_alpha"],
                "racl_margin": spec["margin"],
                "racl_geom_weight": spec["geom_weight"],
                "racl_rank_weight": spec["rank_weight"],
                "racl_memory_k": spec["memory_k"],
            }
            for key, expected in numeric.items():
                if abs(float(row.get(key)) - float(expected)) > 1e-12:
                    raise RuntimeError(f"{key} mismatch in {result_path}")
            if row.get("racl_semantic_revision") != protocol["semantic_revision"]:
                raise RuntimeError(f"semantic revision mismatch in {result_path}")
            if row.get("racl_semantic_cache_sha256") != protocol["semantic_cache_sha256"]:
                raise RuntimeError(f"semantic cache SHA-256 mismatch in {result_path}")
            prediction_path = Path(str(row["validation_prediction_file"]))
            if not prediction_path.is_absolute():
                prediction_path = (project_root / prediction_path).resolve()
            prediction_bundles[name].append(_prediction_bundle(
                prediction_path, str(row["validation_prediction_sha256"])
            ))
            rows.append(row)
        summary = {"spec": spec, "seeds": seeds, "runs": rows}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }
        for metric in ("val_g_mean_cosine", "val_g_effective_rank"):
            values = [float(row[metric]) for row in rows]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }
        aggregates[name] = summary

    control = aggregates[control_name]
    diagnostics = {}
    selectable = []
    repetitions = int(selection["paired_bootstrap_repetitions"])
    bootstrap_seed = int(selection["paired_bootstrap_seed"])
    for name, summary in aggregates.items():
        if name == control_name:
            diagnostics[name] = {
                "role": "matched_control",
                "selectable": False,
                "mean_auprc_delta": 0.0,
                "mean_pos_f1_delta": 0.0,
                "gate_passed": False,
            }
            continue
        ap_delta = summary["auprc"]["mean"] - control["auprc"]["mean"]
        f1_delta = summary["pos_f1"]["mean"] - control["pos_f1"]["mean"]
        bootstrap = _paired_bootstrap(
            prediction_bundles[name], prediction_bundles[control_name],
            repetitions, bootstrap_seed,
        )
        gate_checks = {
            "mean_auprc_delta": (
                ap_delta >= float(selection["minimum_mean_auprc_delta"])
            ),
            "mean_pos_f1_delta": (
                f1_delta >= float(selection["minimum_mean_pos_f1_delta"])
            ),
            "auprc_ci_lower_above_zero": (
                bootstrap["ci95_lower"] > 0
                if selection["require_auprc_ci_lower_above_zero"] else True
            ),
        }
        passed = all(gate_checks.values())
        diagnostics[name] = {
            "role": "racl_candidate",
            "selectable": True,
            "mean_auprc_delta": float(ap_delta),
            "mean_pos_f1_delta": float(f1_delta),
            "matched_seed_auprc_wins": int(sum(
                candidate > baseline
                for candidate, baseline in zip(
                    summary["auprc"]["values"], control["auprc"]["values"]
                )
            )),
            "paired_sample_bootstrap": bootstrap,
            "gate_checks": gate_checks,
            "gate_passed": passed,
        }
        if passed:
            selectable.append(name)

    primary = str(selection["primary_metric"])
    tie_breaker = str(selection["tie_breaker"])
    ranked = sorted(
        selectable,
        key=lambda name: (
            -aggregates[name][primary]["mean"],
            -aggregates[name][tie_breaker]["mean"],
            name,
        ),
    )
    selected = ranked[0] if ranked else None
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_sha256": expected_hash,
        "evidence_policy": expected_policy,
        "architecture": "locked_fusion_global",
        "semantic_revision": protocol["semantic_revision"],
        "test_metrics_accessed": False,
        "rkc_metrics_accessed": False,
        "racl_mandatory": selected is not None,
        "selection": selection,
        "selected_candidate": selected,
        "selected_spec": candidates[selected] if selected else None,
        "ranked_passing_candidates": ranked,
        "status": (
            "validation_gate_passed"
            if selected else "no_racl_candidate_passed_preregistered_gate"
        ),
        "requires_new_authorization": selected is None,
        "diagnostics_vs_no_racl": diagnostics,
        "aggregates": aggregates,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": output["status"],
        "selected_candidate": selected,
        "test_metrics_accessed": False,
        "rkc_metrics_accessed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
