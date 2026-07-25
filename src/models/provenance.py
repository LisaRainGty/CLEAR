"""Small, dependency-free provenance helpers shared by paper experiments."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path_or_name: str) -> str:
    path = Path(path_or_name)
    return str(path.resolve()) if path.exists() else path_or_name


def attach_run_provenance(result: dict, args, model_path: str = "") -> dict:
    """Add fields required to determine whether two RESULT rows are comparable."""
    dataset = str(getattr(args, "dataset", ""))
    result["dataset"] = dataset
    result["dataset_sha256"] = sha256_file(dataset) if dataset and Path(dataset).is_file() else ""
    result["label_field"] = "y"
    result["split_field"] = "split"
    result["split_group"] = "room_id"
    result["evidence_policy"] = getattr(args, "evidence_policy", "record") or "record"
    if model_path:
        result["resolved_model"] = model_identity(model_path)
    return result
