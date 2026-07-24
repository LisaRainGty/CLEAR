#!/usr/bin/env python3
"""Seed a fresh paper namespace with audited architecture-independent baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import run_paper_suite as suite
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import run_paper_suite as suite


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODULES = {
    "models.baselines_ft",
    "models.baselines_neural",
    "models.baselines_frozen",
}


def module_name(command: tuple[str, ...]) -> str:
    command = tuple(command)
    return command[command.index("-m") + 1] if "-m" in command else ""


def normalized_command(command: tuple[str, ...], namespace: str) -> tuple[str, ...]:
    marker = f"/{namespace}/"
    return tuple(
        value.replace(marker, "/<ARTIFACT_NAMESPACE>/")
        for value in command
    )


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_output(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def collect_jobs(config: Path) -> tuple[dict[str, suite.Job], str]:
    cfg = suite.configure(config)
    namespace = str(cfg["paper_suite"]["artifact_namespace"])
    jobs = {
        job.name: job
        for job in suite.build_jobs({"table3"})
        if module_name(job.command) in ALLOWED_MODULES
    }
    return jobs, namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_config = args.source_config.resolve()
    target_config = args.target_config.resolve()
    source_jobs, source_namespace = collect_jobs(source_config)
    target_jobs, target_namespace = collect_jobs(target_config)
    if source_namespace == target_namespace:
        raise RuntimeError("source and target namespaces must differ")
    source_root = ROOT / "results" / source_namespace
    target_root = ROOT / "results" / target_namespace
    reused: list[dict] = []
    skipped: list[dict] = []
    for name, target_job in sorted(target_jobs.items()):
        source_job = source_jobs.get(name)
        reason = None
        if source_job is None:
            reason = "missing_source_job"
        elif normalized_command(source_job.command, source_namespace) != \
                normalized_command(target_job.command, target_namespace):
            reason = "command_mismatch"
        source_status_path = source_root / "status" / f"{name}.json"
        source_status = None
        if reason is None:
            if not source_status_path.exists():
                reason = "missing_source_status"
            else:
                source_status = json.loads(
                    source_status_path.read_text(encoding="utf-8")
                )
                if source_status.get("returncode") != 0:
                    reason = "source_failed"
                elif source_status.get("signature") != source_job.signature:
                    reason = "source_signature_mismatch"
                elif not all(path.exists() for path in source_job.outputs):
                    reason = "missing_source_output"
        if reason is not None:
            skipped.append({"job": name, "reason": reason})
            continue
        reused.append({
            "job": name,
            "module": module_name(target_job.command),
            "source_signature": source_job.signature,
            "target_signature": target_job.signature,
        })
        if args.dry_run:
            continue
        if len(source_job.outputs) != len(target_job.outputs):
            raise RuntimeError(f"{name}: source/target output count mismatch")
        for source_output, target_output in zip(
                source_job.outputs, target_job.outputs
        ):
            copy_output(source_output, target_output)
        for subdir, suffix in (("jobs", ".jsonl"), ("logs", ".log")):
            source_path = source_root / subdir / f"{name}{suffix}"
            if source_path.exists():
                target_path = target_root / subdir / f"{name}{suffix}"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
        target_status = dict(source_status)
        target_status.update({
            "signature": target_job.signature,
            "command": list(target_job.command),
            "command_shell": shlex.join(target_job.command),
            "log": f"results/{target_namespace}/logs/{name}.log",
            "result_file": f"results/{target_namespace}/jobs/{name}.jsonl",
            "outputs": [
                str(path.relative_to(ROOT)) for path in target_job.outputs
            ],
            "reused_from_namespace": source_namespace,
            "reused_source_signature": source_job.signature,
            "reuse_reason": "architecture-independent Table 3 baseline",
        })
        target_status_path = target_root / "status" / f"{name}.json"
        target_status_path.parent.mkdir(parents=True, exist_ok=True)
        target_status_path.write_text(
            json.dumps(target_status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_namespace": source_namespace,
        "target_namespace": target_namespace,
        "source_config": str(source_config.relative_to(ROOT)),
        "source_config_sha256": config_sha256(source_config),
        "target_config": str(target_config.relative_to(ROOT)),
        "target_config_sha256": config_sha256(target_config),
        "allowed_modules": sorted(ALLOWED_MODULES),
        "reused_count": len(reused),
        "reused": reused,
        "skipped": skipped,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        target_root.mkdir(parents=True, exist_ok=True)
        (target_root / "reuse_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
