#!/usr/bin/env python3
"""Create a deterministic file inventory and checksums for reviewer handoff."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF_OUTPUTS = {"MANIFEST.tsv", "CHECKSUMS.sha256"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-all", action="store_true",
                        help="Hash every file; default hashes reproducibility-critical files only.")
    args = parser.parse_args()
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.name not in SELF_OUTPUTS
        and not path.name.startswith("._") and path.name != ".DS_Store"
        and "__pycache__" not in path.parts and ".git" not in path.parts
        and path.name != "env.sh" and "data/cache" not in path.as_posix()
    )
    critical_roots = {"configs", "scripts", "src", "results", "paper"}
    critical_names = {
        "dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl",
        "dataset.jsonl", "requirements.txt", "requirements-train.txt",
        "environment.yml", "README.md", "env.example.sh",
    }
    manifest = ["path\tbytes\tsha256"]
    checksums = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        should_hash = args.hash_all or rel.split("/", 1)[0] in critical_roots or path.name in critical_names
        sha = digest(path) if should_hash else "NOT_HASHED"
        manifest.append(f"{rel}\t{path.stat().st_size}\t{sha}")
        if sha != "NOT_HASHED":
            checksums.append(f"{sha}  {rel}")
    (ROOT / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (ROOT / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"files={len(files)} hashed={len(checksums)}")


if __name__ == "__main__":
    main()
