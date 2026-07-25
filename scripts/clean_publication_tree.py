#!/usr/bin/env python3
"""Move non-paper artifacts out of ``main`` while preserving recovery.

The default is a dry run.  ``--execute`` moves candidates to a sibling archive
with their repository-relative paths intact; it never deletes them.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs/dataset_provenance/STATEFUL_PROPOSAL_DATASET_V2_FULLPOOL_20260614.md"


def review_inputs() -> list[Path]:
    text = PROVENANCE.read_text(encoding="utf-8")
    match = re.search(r"^- reviews: `(\[.*\])`$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"cannot read review manifest from {PROVENANCE}")
    return [ROOT / path for path in ast.literal_eval(match.group(1))]


def repaired_keep_names() -> set[str]:
    names = {
        "REPAIRED_DATASETS_V1.md",
        "proposal_quality_audit_all_v1_20260613.jsonl",
        "proposal_quality_audit_all_v1_20260613_report.json",
        "full_pair_reconstruction_queue_v1_20260614.jsonl",
        "full_pair_reconstruction_queue_v1_20260614.report.json",
        "stateful_proposal_dataset_v2_FULLPOOL_all_20260614.jsonl",
        "stateful_proposal_dataset_v2_FULLPOOL_supervised_20260614.jsonl",
        "stateful_proposal_dataset_v2_FULLPOOL_contrastive_20260614.jsonl",
        "stateful_proposal_dataset_v2_FULLPOOL_repair_20260614.jsonl",
        "stateful_proposal_dataset_v2_FULLPOOL_20260614.report.json",
        "dataset_planbaseline_duallabel_FULLPOOL_all_20260614.jsonl",
        "dataset_planbaseline_duallabel_FULLPOOL_supervised_20260614.jsonl",
        "dataset_planbaseline_duallabel_FULLPOOL_20260614.report.json",
        "dataset_planbaseline_duallabel_FULLPOOL_all_20260614_stagec.jsonl",
        "dataset_planbaseline_duallabel_FULLPOOL_supervised_20260614_stagec.jsonl",
        "merge_stagec_factrecords_plusgap_20260615.report.json",
        "dataset_objective_negatives_v1_20260615.jsonl",
        "dataset_objective_negatives_v1_20260615.report.json",
    }
    for path in review_inputs():
        names.add(path.name)
        stem = path.name.removesuffix(".jsonl")
        names.add(stem + ".report.json")
        names.add(stem + "_report.json")
    return names


def add_tree(candidates: set[Path], path: Path) -> None:
    if path.exists() or path.is_symlink():
        candidates.add(path)


def candidates() -> list[Path]:
    out: set[Path] = set()

    # Root data views not used by any paper experiment.
    add_tree(out, ROOT / "data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_all_20260615.jsonl")
    add_tree(out, ROOT / "data/samples.json")

    # Exploratory processed variants; keep official A/B/C, product_v2,
    # repaired Stage-A, full-schema gaps, and positive/negative Stage-C facts.
    for name in ("_prev_llm", "_sweep", "stageB_product_v2_candidate",
                 "stageB_product_v2_candidate_20260613"):
        add_tree(out, ROOT / "data/processed" / name)

    # final/ must contain data artifacts, not historical experiments.
    final = ROOT / "data/final"
    for child in final.iterdir():
        if child.name not in {"dataset.jsonl", "repaired_v1"}:
            out.add(child)

    repaired = final / "repaired_v1"
    keep = repaired_keep_names()
    for child in repaired.iterdir():
        if child.name not in keep:
            out.add(child)

    # Only the FULLPOOL provenance drives the released training set.
    provenance_dir = ROOT / "docs/dataset_provenance"
    for child in provenance_dir.iterdir():
        if child.name != PROVENANCE.name:
            out.add(child)

    # Historical results/embeddings are not evidence for the fresh campaign.
    for path in (
        ROOT / "results/artifacts",
        ROOT / "results/alpha_sweep.json",
        ROOT / "results/campaign6_results.jsonl",
        ROOT / "results/campaign7_results.jsonl",
        ROOT / "results/campaign8_results.jsonl",
    ):
        add_tree(out, path)
    for child in (ROOT / "embeddings").iterdir():
        if child.name not in {"README.md", "fair_rerun"}:
            out.add(child)

    # Old inventories become invalid after this cleanup and are regenerated.
    for name in ("CHECKSUMS.sha256", "MANIFEST.tsv"):
        add_tree(out, ROOT / name)

    # Finder/AppleDouble and bytecode are never scientific artifacts.
    for path in ROOT.rglob("._*"):
        out.add(path)
    for path in ROOT.rglob("__pycache__"):
        out.add(path)
    for path in ROOT.rglob("*.pyc"):
        out.add(path)

    # Keep only top-most candidate paths so a directory is not moved twice.
    ordered = sorted(out, key=lambda path: (len(path.parts), str(path)))
    top = []
    for path in ordered:
        if not any(parent == selected for selected in top for parent in path.parents):
            top.append(path)
    return top


def logical_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT.parent / "claimarc_main_nonpublication_archive_20260721",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "results/audit/publication_cleanup.json")
    args = parser.parse_args()

    selected = candidates()
    items = []
    for source in selected:
        relative = source.relative_to(ROOT)
        items.append({"path": str(relative), "bytes": logical_size(source)})
        if args.execute:
            destination = args.archive / relative
            if destination.exists():
                raise FileExistsError(f"archive target already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

    report = {
        "mode": "executed" if args.execute else "dry_run",
        "archive": str(args.archive),
        "recoverable": True,
        "candidate_paths": len(items),
        "candidate_bytes": sum(item["bytes"] for item in items),
        "items": items,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "items"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
