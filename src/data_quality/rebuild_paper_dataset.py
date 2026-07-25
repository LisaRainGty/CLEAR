"""Rebuild and byte-verify the frozen paper training data.

The default mode rebuilds the deterministic post-review lineage in a temporary
directory, leaving the publication snapshot untouched.  ``--in-place`` writes
the same lineage to its documented paths.  External LLM/VLM review calls are
not replayed: their 44 immutable JSONL outputs are declared in the provenance
record and are treated as frozen data-construction inputs.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator

try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal hosts
    zstd = None


PROVENANCE = Path(
    "docs/dataset_provenance/STATEFUL_PROPOSAL_DATASET_V2_FULLPOOL_20260614.md"
)
QUEUE = Path("data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
FACTS = Path("data/processed/stageC/fact_records_plus_gap_20260615.jsonl")
PROCESSED_ARCHIVE = Path("data/archives/processed.tar.zst")
FACTS_ARCHIVE_MEMBER = "processed/stageC/fact_records_plus_gap_20260615.jsonl"
FACTS_SHA256 = "89e6cfce78f3dc92f063c6811f3d84ab9def056d106d84062b7e2711d06bb2e0"
NEG_FACTS = Path("data/processed/stageC_neg/fact_records_neg.jsonl")
NEG_FACTS_ARCHIVE_MEMBER = "processed/stageC_neg/fact_records_neg.jsonl"
NEG_FACTS_SHA256 = "70c0d8851ed7a0f491b976e08b29175e9c65025a67a3e811bd430e019cd6ba0a"
NEG_PAIRS = Path("data/processed/stageB_fullschema_gap/claim_no_comment_pairs_v1.jsonl")
NEG_PAIRS_ARCHIVE_MEMBER = (
    "processed/stageB_fullschema_gap/claim_no_comment_pairs_v1.jsonl"
)
NEG_PAIRS_SHA256 = "706aa312382ea2a939964ad3957b6c1247ee27747a466625cfbde4795702c8dd"

EXPECTED = {
    "stateful_all": (8908, "06943103e87d6ccce83f86db3392bca1fb3646e0aa01a91d38700a19736c471e"),
    "stateful_supervised": (2278, "92f04c2e50f64f28b13d4a59ffeb5b248a8fcd698f070c587154460a29f728c2"),
    "stateful_contrastive": (1046, "b87e439fb60044e45a954333628446d5e06c0208b4389705b833a1676d95a93f"),
    "stateful_repair": (6630, "adcd05f8a5ed65d0cabb09d537a36691b10043fa65247c3622d875b4c40eeaf9"),
    "plan_all": (8908, "5d64148eaa207efd8c7518eecd602b6e6488198219e5efa53832922b9998044a"),
    "plan_supervised": (2278, "5bdd16a2c784f412a8e5a84228659580cc77b739f4c1c7bb13324badc5f6dfd5"),
    "plan_all_stagec": (8908, "3edc9c98c043bf88b120c4f1084ed9883eebf399f3080853b14befa94707b4cd"),
    "plan_supervised_stagec": (2278, "0cb580f5df791aebfa963c9e84a701c96f7aa4c3d85d337dfb65839f11b60492"),
    "objective_negative": (2605, "7df3f9ac18c6691410489322c129600d3ba528190c0f3c9757351a57b6b151c9"),
    "paper_training": (4883, "1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


@contextmanager
def decompressed_reader(path: Path) -> Iterator[BinaryIO]:
    if zstd is not None:
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as stream:
                yield stream
        return
    executable = shutil.which("zstd")
    if not executable:
        raise RuntimeError("install zstandard>=0.23 or provide a zstd executable")
    process = subprocess.Popen(
        [executable, "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        return_code = process.wait()
        # Streaming a single early archive member closes the pipe intentionally.
        if return_code not in (0, -13, 141):
            raise subprocess.CalledProcessError(return_code, process.args)


def materialize_processed_file(
    source: Path,
    member_name: str,
    expected_sha256: str,
    scratch: Path,
) -> Path:
    """Return one processed input, streaming it from the compact archive if needed."""
    if source.is_file():
        if sha256(source) != expected_sha256:
            raise ValueError(f"processed-input checksum mismatch: {source}")
        return source
    if not PROCESSED_ARCHIVE.is_file():
        raise FileNotFoundError(f"missing {source} and {PROCESSED_ARCHIVE}")

    target = scratch / source.name
    found = False
    with decompressed_reader(PROCESSED_ARCHIVE) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if member.name != member_name:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"missing payload for {member_name}")
                with target.open("wb") as output:
                    shutil.copyfileobj(handle, output, length=1024 * 1024)
                found = True
                break
    if not found:
        raise FileNotFoundError(f"{member_name} not found in {PROCESSED_ARCHIVE}")
    if sha256(target) != expected_sha256:
        raise ValueError(f"archived processed-input checksum mismatch: {target}")
    return target


def review_paths() -> list[str]:
    text = PROVENANCE.read_text(encoding="utf-8")
    match = re.search(r"^- reviews: `(\[.*\])`$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"review manifest not found in {PROVENANCE}")
    paths = ast.literal_eval(match.group(1))
    if not isinstance(paths, list) or len(paths) != 44:
        raise ValueError(f"expected 44 frozen review inputs, found {len(paths)}")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen review inputs: {missing[:3]}")
    return paths


def run_module(module: str, *args: str) -> None:
    env = dict(os.environ)
    src = str(Path("src").resolve())
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, "-m", module, *args], check=True, env=env)


def build_paths(root: Path, in_place: bool) -> dict[str, Path]:
    if in_place:
        repaired = Path("data/final/repaired_v1")
        return {
            "stateful_all": repaired / "stateful_proposal_dataset_v2_FULLPOOL_all_20260614.jsonl",
            "stateful_supervised": repaired / "stateful_proposal_dataset_v2_FULLPOOL_supervised_20260614.jsonl",
            "stateful_contrastive": repaired / "stateful_proposal_dataset_v2_FULLPOOL_contrastive_20260614.jsonl",
            "stateful_repair": repaired / "stateful_proposal_dataset_v2_FULLPOOL_repair_20260614.jsonl",
            "plan_all": repaired / "dataset_planbaseline_duallabel_FULLPOOL_all_20260614.jsonl",
            "plan_supervised": repaired / "dataset_planbaseline_duallabel_FULLPOOL_supervised_20260614.jsonl",
            "plan_all_stagec": repaired / "dataset_planbaseline_duallabel_FULLPOOL_all_20260614_stagec.jsonl",
            "plan_supervised_stagec": repaired / "dataset_planbaseline_duallabel_FULLPOOL_supervised_20260614_stagec.jsonl",
            "objective_negative": repaired / "dataset_objective_negatives_v1_20260615.jsonl",
            "paper_training": Path("data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl"),
        }
    return {name: root / f"{name}.jsonl" for name in EXPECTED}


def rebuild(
    root: Path,
    in_place: bool,
    fact_records: Path,
    negative_fact_records: Path,
    negative_pairs: Path,
) -> dict[str, Path]:
    paths = build_paths(root, in_place)
    report_dir = Path("data/final/repaired_v1") if in_place else root
    provenance_md = PROVENANCE if in_place else root / "stateful_provenance.md"
    reviews = review_paths()

    run_module(
        "data_quality.build_stateful_proposal_dataset_v2",
        "--queue", str(QUEUE), "--reviews", *reviews,
        "--out_all", str(paths["stateful_all"]),
        "--out_supervised", str(paths["stateful_supervised"]),
        "--out_contrastive", str(paths["stateful_contrastive"]),
        "--out_repair", str(paths["stateful_repair"]),
        "--report", str(report_dir / "stateful_proposal_dataset_v2_FULLPOOL_20260614.report.json"),
        "--markdown", str(provenance_md),
    )
    run_module(
        "data_quality.build_plan_label_weights_v1",
        "--all_rows", str(paths["stateful_all"]),
        "--out_supervised", str(paths["plan_supervised"]),
        "--out_all", str(paths["plan_all"]),
        "--report", str(report_dir / "dataset_planbaseline_duallabel_FULLPOOL_20260614.report.json"),
    )
    for source, target, report in (
        ("plan_all", "plan_all_stagec", "merge_stagec_fullpool_all.report.json"),
        ("plan_supervised", "plan_supervised_stagec", "merge_stagec_fullpool_supervised.report.json"),
    ):
        run_module(
            "data_quality.merge_stagec_evidence",
            "--fact-records", str(fact_records),
            "--input", str(paths[source]),
            "--output", str(paths[target]),
            "--report", str(report_dir / report),
        )
    run_module(
        "data_quality.build_objective_negative_dataset_v1",
        "--pairs", str(negative_pairs),
        "--fact_records", str(negative_fact_records),
        "--ref_dataset", str(paths["plan_all_stagec"]),
        "--out", str(paths["objective_negative"]),
        "--report", str(report_dir / "dataset_objective_negatives_v1_20260615.report.json"),
    )
    run_module(
        "data_quality.build_paper_training_set",
        "--perception", str(paths["plan_supervised_stagec"]),
        "--objective-negative", str(paths["objective_negative"]),
        "--output", str(paths["paper_training"]),
        "--overwrite",
        "--report", str(report_dir / "paper_training_set_build.json"),
    )
    return paths


def verify(paths: dict[str, Path]) -> dict:
    report = {"status": "PASS", "artifacts": {}}
    for name, path in paths.items():
        actual = {"path": str(path), "rows": rows(path), "sha256": sha256(path)}
        expected_rows, expected_hash = EXPECTED[name]
        actual["expected_rows"] = expected_rows
        actual["expected_sha256"] = expected_hash
        actual["match"] = actual["rows"] == expected_rows and actual["sha256"] == expected_hash
        if not actual["match"]:
            report["status"] = "FAIL"
        report["artifacts"][name] = actual
    if report["status"] != "PASS":
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="write the documented publication paths")
    parser.add_argument("--report", default="results/audit/paper_data_lineage_rebuild.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="claimarc-paper-data-") as tmp:
        scratch = Path(tmp)
        fact_records = materialize_processed_file(
            FACTS, FACTS_ARCHIVE_MEMBER, FACTS_SHA256, scratch
        )
        negative_fact_records = materialize_processed_file(
            NEG_FACTS, NEG_FACTS_ARCHIVE_MEMBER, NEG_FACTS_SHA256, scratch
        )
        negative_pairs = materialize_processed_file(
            NEG_PAIRS, NEG_PAIRS_ARCHIVE_MEMBER, NEG_PAIRS_SHA256, scratch
        )
        if args.in_place:
            paths = rebuild(
                Path("."), True, fact_records, negative_fact_records, negative_pairs
            )
        else:
            paths = rebuild(
                scratch, False, fact_records, negative_fact_records, negative_pairs
            )
        report = verify(paths)
    report.update({
        "mode": "in_place" if args.in_place else "temporary_verification",
        "frozen_review_inputs": 44,
        "arguments_used": False,
    })
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
