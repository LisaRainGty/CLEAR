#!/usr/bin/env python3
"""Resumable, reviewer-facing runner for every experiment in the paper.

Every train/evaluation command is frozen to the same ``sources_only`` view:
PARAM + OCR + VLM, in that order.  Generated arguments and single-source views
are intentionally absent from the matrix.  Each job has an independent log,
result file and completion record, so an interrupted GPU rental can resume
without repeating successful work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATASET = ROOT / "data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl"
CONFIG = ROOT / "configs/paper_fair.json"
OUT = ROOT / "results/fair_rerun"
LOGS = OUT / "logs"
STATUS = OUT / "status"
JOB_RESULTS = OUT / "jobs"
PRED = ROOT / "embeddings/fair_rerun/baseline_predictions"
GEOM = ROOT / "embeddings/fair_rerun/emb_geom"
ABL = ROOT / "embeddings/fair_rerun/ablations"
HP = ROOT / "embeddings/fair_rerun/hparams_lora"
XDOM = ROOT / "embeddings/fair_rerun/xdom"
XDOM_LLM = ROOT / "embeddings/fair_rerun/xdom_llm"
POLICY = "sources_only"
SEEDS = (0, 1, 2)
CATEGORIES = (
    "apparel_and_underwear", "general", "baby_kids_and_pets", "shoes_and_bags",
    "food_and_beverages", "smart_home", "digital_and_electronics",
    "sports_and_outdoor", "beauty_and_personal_care", "jewelry_and_collectibles",
)


@dataclass(frozen=True)
class Job:
    name: str
    stage: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def signature(self) -> str:
        blob = json.dumps({"name": self.name, "command": self.command}, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False).stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def claimarc_command(py: str, tag: str, seed: int, extra=(), *, lora=False,
                     bundle: Path | None = None) -> tuple[str, ...]:
    batch = int(os.environ.get("CLAIMARC_BATCH_SIZE", "12"))
    effective = int(os.environ.get("CLAIMARC_EFFECTIVE_BATCH", "36"))
    accum = max(1, math.ceil(effective / batch))
    train_mode = ("--enc_train", "lora", "--lr", "2e-5") if lora else (
        "--enc_train", "full", "--lr", "1e-5")
    command = (
        py, "-m", "models.train", "--dataset", str(DATASET), "--tag", tag,
        "--seed", str(seed), "--warmup", "3", "--cl_epochs", "6",
        "--bs", str(batch), "--accum", str(accum), "--loss", "bce",
        "--lambda_cl", "0.5", "--tau", "0.07", "--Kp", "3", "--Kn", "5",
        "--encoder_name", os.environ.get("CLAIMARC_BGE_PATH", "BAAI/bge-large-zh-v1.5"),
        *train_mode, "--cl_no_attr_block", "--cl_class_balanced",
        "--evidence_policy", POLICY, *tuple(extra),
    )
    if bundle is not None:
        command += ("--save_emb", str(bundle))
    return command


def add_claimarc(jobs: list[Job], name: str, stage: str, extra=(), *, lora=False,
                 seeds=SEEDS, bundle_dir: Path = ABL, bundle_prefix: str | None = None):
    for seed in seeds:
        stem = bundle_prefix or name
        bundle = bundle_dir / f"{stem}_s{seed}.pt"
        jobs.append(Job(
            f"{name}_s{seed}", stage,
            claimarc_command(sys.executable, name, seed, extra, lora=lora, bundle=bundle),
            (bundle,),
        ))


def build_jobs(stages: set[str]) -> list[Job]:
    py = os.environ.get("CLAIMARC_PYTHON", sys.executable)
    jobs: list[Job] = []

    if "audit" in stages:
        audit_out = ROOT / "results/audit/reproducibility_report.json"
        jobs.append(Job("dataset_audit", "audit", (
            py, str(ROOT / "scripts/audit_reproducibility.py"),
        ), (audit_out,)))

    if "table3" in stages:
        add_claimarc(jobs, "claimarc_canonical", "table3", seeds=SEEDS,
                     bundle_dir=GEOM, bundle_prefix="emb_geom_racl")
        for kind in ("bert_cls", "roberta_cls", "bert_nli", "esim"):
            for seed in SEEDS:
                pred = PRED / f"{kind}_s{seed}.pt"
                jobs.append(Job(f"{kind}_s{seed}", "table3", (
                    py, "-m", "models.baselines_ft", "--dataset", str(DATASET),
                    "--kind", kind, "--seed", str(seed), "--loss", "bce",
                    "--evidence_policy", POLICY, "--save_pred", str(pred),
                ), (pred,)))
        for kind in ("textcnn", "bilstm", "dam"):
            for seed in SEEDS:
                pred = PRED / f"{kind}_s{seed}.pt"
                jobs.append(Job(f"{kind}_s{seed}", "table3", (
                    py, "-m", "models.baselines_neural", "--dataset", str(DATASET),
                    "--kind", kind, "--seed", str(seed), "--loss", "bce",
                    "--evidence_policy", POLICY, "--save_pred", str(pred),
                ), (pred,)))
        # Frozen probes do not update the encoder; the paper specifies one run.
        # A fixed seed is retained only for the weighted MLP initialization.
        for seed in (0,):
            frozen_dir = PRED / f"frozen_s{seed}"
            jobs.append(Job(f"frozen_probes_s{seed}", "table3", (
                py, "-m", "models.baselines_frozen", "--dataset", str(DATASET),
                "--seed", str(seed), "--evidence_policy", POLICY,
                "--save_dir", str(frozen_dir), "--paper_only",
            ), (frozen_dir / "BGEfz_LR_4tuple.pt", frozen_dir / "BGEfz_SVM_4tuple.pt",
                frozen_dir / "BGEfz_MLP_4tuple.pt", frozen_dir / "BGEfz_kNN_attr_k15.pt")))
        for seed in SEEDS:
            pred = PRED / f"qwen2p5_7b_qlora_s{seed}.pt"
            jobs.append(Job(f"qwen2p5_7b_qlora_s{seed}", "table3", (
                py, "-m", "models.qwen_sft", "--dataset", str(DATASET),
                "--seed", str(seed), "--evidence_policy", POLICY,
                "--save_pred", str(pred),
            ), (pred,)))

    if "ablation" in stages:
        # Table 6 and Table 7 share the exact same no-RACL run.
        add_claimarc(jobs, "no_racl", "ablation", ("--no_cl",),
                     bundle_dir=GEOM, bundle_prefix="emb_geom_none")
        add_claimarc(jobs, "supcon", "ablation", ("--cl_mode", "supcon"),
                     bundle_dir=GEOM, bundle_prefix="emb_geom_supcon")
        variants = {
            # Table 7
            "no_reliability": ("--no_weight",),
            "no_class_balance": (),  # canonical flags are sanitized below
            "no_four_tuple": ("--head_concat_only",),
            "bert_backbone": ("--backbone", "bert"),
            # Table 8
            "no_fusion": ("--no_fusion",),
            "claim_only": ("--no_fusion", "--stream_mode", "claim"),
            "evidence_only": ("--no_fusion", "--stream_mode", "evidence"),
            # Table 9
            "hard_positive": ("--cl_hard_pos",),
            "same_attribute_negative": (),  # canonical no-attr flag sanitized below
            "same_evidence_type_negative": ("--cl_neg_filter", "same_evtype"),
            "kp1": ("--Kp", "1"), "kp5": ("--Kp", "5"),
            "kn1": ("--Kn", "1"), "kn10": ("--Kn", "10"),
        }
        for name, extra in variants.items():
            for seed in SEEDS:
                bundle = ABL / f"{name}_s{seed}.pt"
                command = list(claimarc_command(py, name, seed, extra, bundle=bundle))
                if name == "no_class_balance":
                    command.remove("--cl_class_balanced")
                if name == "same_attribute_negative":
                    command.remove("--cl_no_attr_block")
                jobs.append(Job(f"{name}_s{seed}", "ablation", tuple(command), (bundle,)))
        # Uniform weighting is exactly the Table 7 no-reliability ablation;
        # reuse that three-seed run in Table 10 instead of training it twice.
        for transform in ("inverse", "permute", "binary", "count", "sqrt"):
            add_claimarc(jobs, f"weight_{transform}", "ablation",
                         ("--c_transform", transform))
        c_specs = {
            "c_k1p5": "k=1.5,lambda=0.3,rho=0.4,phi=1.2",
            "c_k6": "k=6,lambda=0.3,rho=0.4,phi=1.2",
            "c_lambda0p1": "k=3,lambda=0.1,rho=0.4,phi=1.2",
            "c_lambda0p6": "k=3,lambda=0.6,rho=0.4,phi=1.2",
            "c_rho0p2": "k=3,lambda=0.3,rho=0.2,phi=1.2",
            "c_rho0p6": "k=3,lambda=0.3,rho=0.6,phi=1.2",
            "c_phi1p0": "k=3,lambda=0.3,rho=0.4,phi=1.0",
            "c_phi1p5": "k=3,lambda=0.3,rho=0.4,phi=1.5",
        }
        for name, spec in c_specs.items():
            # Paper Table 13 is a matched-seed sensitivity scan: compare every
            # perturbation with canonical seed 0, never with a three-seed mean.
            add_claimarc(jobs, name, "ablation", ("--c_recompute", spec), seeds=(0,))

    if "hparams" in stages:
        variants = {
            "lora_canonical": (),
            "fusion1": ("--n_fusion", "1"), "fusion3": ("--n_fusion", "3"),
            "fusion4": ("--n_fusion", "4"),
            "heads4": ("--heads", "4"), "heads16": ("--heads", "16"),
            "rank8": ("--lora_rank", "8"), "rank32": ("--lora_rank", "32"),
            "lambda0p1": ("--lambda_cl", "0.1"),
            "lambda0p3": ("--lambda_cl", "0.3"),
            "lambda1p0": ("--lambda_cl", "1.0"),
            "tau0p05": ("--tau", "0.05"), "tau0p10": ("--tau", "0.10"),
            "tau0p20": ("--tau", "0.20"),
            "k1_1": ("--Kp", "1", "--Kn", "1"),
            "k5_10": ("--Kp", "5", "--Kn", "10"),
            "loss_asl": ("--loss", "asl"), "loss_focal": ("--loss", "focal"),
            "ffn_gelu": ("--ffn", "gelu"),
            "xattn_c2e": ("--xattn_dir", "c2e"),
            "xattn_e2c": ("--xattn_dir", "e2c"),
            "independent_projection": ("--indep_proj",),
        }
        for name, extra in variants.items():
            # Table 12 is explicitly the paper's single-seed LoRA sensitivity
            # analysis; the LoRA canonical in this same block is its comparator.
            add_claimarc(jobs, f"hp_{name}", "hparams", extra, lora=True,
                         bundle_dir=HP, seeds=(0,))

    if "xdom" in stages:
        common = (
            "--dataset", str(DATASET), "--outdir", str(XDOM),
            "--warmup", "3", "--cl_epochs", "6", "--enc_train", "full",
            "--lr", "1e-5", "--cl_no_attr_block", "--cl_class_balanced",
            "--evidence_policy", POLICY,
        )
        for category in CATEGORIES:
            label = category[:24]
            for model in ("clarc", "bert_cls", "roberta_cls", "esim"):
                output = XDOM / f"{model}_category_{label}_s0.pt"
                jobs.append(Job(f"xdom_category_{label}_{model}", "xdom", (
                    py, "-m", "models.xdom_fold", *common, "--mode", "category",
                    "--holdout", category, "--seed", "0", "--models", model,
                ), (output,)))
        for seed in SEEDS:
            for model in ("clarc", "bert_cls", "roberta_cls", "esim"):
                output = XDOM / f"{model}_rooms_rooms_s{seed}.pt"
                jobs.append(Job(f"xdom_rooms_s{seed}_{model}", "xdom", (
                    py, "-m", "models.xdom_fold", *common, "--mode", "rooms",
                    "--seed", str(seed), "--models", model,
                ), (output,)))
        cat_out = OUT / "table4_xdom_category.json"
        room_out = OUT / "table4_xdom_rooms.json"
        inject_rooms = OUT / "table5_injection_rooms.json"
        jobs.extend([
            Job("aggregate_xdom_category", "xdom", (
                py, "-m", "models.xdom_agg", "--indir", str(XDOM),
                "--mode", "category", "--out", str(cat_out)), (cat_out,)),
            Job("aggregate_xdom_rooms", "xdom", (
                py, "-m", "models.xdom_agg", "--indir", str(XDOM),
                "--mode", "rooms", "--out", str(room_out)), (room_out,)),
            Job("injection_rooms", "xdom", (
                py, "-m", "models.xdom_inject", "--bundle_dir", str(XDOM),
                "--mode", "rooms", "--out", str(inject_rooms)), (inject_rooms,)),
        ])

    if "llm" in stages:
        # API runs may incur cost, so this stage is explicit rather than part of --stages all.
        # Hosted reasoning models need enough completion budget to finish the
        # same compact JSON schema.  These caps were fixed using longest-prompt
        # parse probes, not downstream labels or metrics.
        output_budgets = {
            "Qwen-Flash": 320,
            "GPT-5.4": 320,
            "Gemini-3.5-Flash": 1024,
            "Kimi-K2.6": 4096,
        }
        for model, short in (("Qwen-Flash", "qwen_flash"), ("GPT-5.4", "gpt54"),
                             ("Gemini-3.5-Flash", "gemini35"), ("Kimi-K2.6", "kimi")):
            for mode in ("zero", "fewshot"):
                tag = f"{short}_{'fs5' if mode == 'fewshot' else 'zero'}"
                out = OUT / f"llm_{tag}.json"
                jobs.append(Job(f"llm_{tag}", "llm", (
                    py, "-m", "models.run_llm_baselines", "--dataset", str(DATASET),
                    "--model", model, "--mode", mode, "--shots", "5", "--tag", tag,
                    "--max_tokens", str(output_budgets[model]),
                    "--seed", "0", "--evidence_policy", POLICY, "--eval_out", str(out),
                ), (out,)))
        # Table 4 uses Qwen-Flash as the representative zero/five-shot LLM.
        # Each bundle contains both modes; all few-shot examples come from the
        # source-domain train split and all target examples remain read-only.
        for category in CATEGORIES:
            label = category[:24]
            output = XDOM_LLM / f"llm_Qwen-Flash_category_{label}_s0.pt"
            jobs.append(Job(f"xdom_llm_category_{label}", "llm", (
                py, "-m", "models.xdom_llm", "--dataset", str(DATASET),
                "--mode", "category", "--holdout", category, "--model", "Qwen-Flash",
                "--outdir", str(XDOM_LLM), "--seed", "0",
                "--evidence_policy", POLICY,
            ), (output,)))
        for seed in SEEDS:
            output = XDOM_LLM / f"llm_Qwen-Flash_rooms_rooms_s{seed}.pt"
            jobs.append(Job(f"xdom_llm_rooms_s{seed}", "llm", (
                py, "-m", "models.xdom_llm", "--dataset", str(DATASET),
                "--mode", "rooms", "--model", "Qwen-Flash",
                "--outdir", str(XDOM_LLM), "--seed", str(seed),
                "--evidence_policy", POLICY,
            ), (output,)))
        for mode in ("category", "rooms"):
            output = OUT / f"table4_xdom_{mode}_all.json"
            jobs.append(Job(f"aggregate_xdom_{mode}_with_llm", "llm", (
                py, "-m", "models.xdom_agg", "--indir", str(XDOM),
                "--llm_indir", str(XDOM_LLM), "--mode", mode, "--out", str(output),
            ), (output,)))

    if "analysis" in stages:
        geom_out = OUT / "table6_geometry.json"
        jobs.append(Job("geometry_probe", "analysis", (
            py, "-m", "models.geom_probe2", "--emb_dir", str(GEOM),
            "--variants", "none", "supcon", "racl", "--seeds", "0", "1", "2",
            "--prefix", "emb_geom_", "--out", str(geom_out),
        ), (geom_out,)))
        jobs.append(Job("geometry_figures", "analysis", (
            py, "-m", "models.make_geom_figs", "--emb_dir", str(GEOM),
            "--geom_json", str(geom_out), "--outdir", str(ROOT / "paper/figs"),
            "--seed", "0",
        ), (ROOT / "paper/figs/fig_umap_label.png",)))
        bootstrap = OUT / "table3_paired_bootstrap.json"
        jobs.append(Job("paired_bootstrap", "analysis", (
            py, str(ROOT / "scripts/paired_bootstrap.py"), "--root", str(ROOT),
            "--repetitions", "2000", "--output", str(bootstrap),
        ), (bootstrap,)))
        jobs.append(Job("metrics_and_pr_roc", "analysis", (
            py, "-m", "models.metrics_rich",
        ), (OUT / "metrics_rich.json", ROOT / "paper/figs/fig_pr_roc.png")))
        jobs.append(Job("injection_figure", "analysis", (
            py, "-m", "models.make_inject_fig", "--input",
            str(OUT / "table5_injection_rooms.json"), "--outdir", str(ROOT / "paper/figs"),
        ), (ROOT / "paper/figs/fig_inject.png",)))
        selective = OUT / "selective_canon.json"
        jobs.append(Job("selective_prediction", "analysis", (
            py, str(ROOT / "scripts/selective_prediction.py"), "--emb-dir", str(GEOM),
            "--output", str(selective), "--random-repetitions", "200",
        ), (selective,)))
        jobs.append(Job("selective_figure", "analysis", (
            py, "-m", "models.make_selective_fig",
        ), (ROOT / "paper/figs/fig_selective.png",)))
        jobs.append(Job("calibration_and_hparam_figures", "analysis", (
            py, "-m", "models.make_figs",
        ), (ROOT / "paper/figs/fig_calibration.png", ROOT / "paper/figs/fig_hparam.png")))
        error_out = OUT / "error_analysis.json"
        jobs.append(Job("error_analysis", "analysis", (
            py, str(ROOT / "scripts/error_analysis.py"), "--dataset", str(DATASET),
            "--emb-dir", str(GEOM), "--output", str(error_out),
        ), (error_out,)))
        tables = OUT / "paper_tables.md"
        jobs.append(Job("aggregate_paper_tables", "analysis", (
            py, str(ROOT / "scripts/aggregate_paper_results.py"), "--root", str(ROOT),
            "--output", str(tables),
        ), (tables,)))
    return jobs


def verify_frozen_contract(jobs: list[Job]) -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = cfg["dataset"]["sha256"]
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"dataset SHA-256 mismatch: {digest} != {expected}")
    model_modules = {
        "models.train", "models.baselines_ft", "models.baselines_neural",
        "models.baselines_frozen", "models.qwen_sft", "models.xdom_fold",
        "models.run_llm_baselines", "models.xdom_llm",
    }
    for job in jobs:
        cmd = list(job.command)
        module = cmd[cmd.index("-m") + 1] if "-m" in cmd else ""
        if module not in model_modules:
            continue
        if "--evidence_policy" not in cmd:
            raise RuntimeError(f"{job.name}: missing explicit evidence policy")
        value = cmd[cmd.index("--evidence_policy") + 1]
        if value != POLICY:
            raise RuntimeError(f"{job.name}: forbidden evidence policy {value}")
        forbidden = {"args_only", "args_first", "params_only", "ocr_only", "vlm_only"}
        if forbidden.intersection(cmd):
            raise RuntimeError(f"{job.name}: arguments/single-source view leaked into command")


def successful(job: Job) -> bool:
    path = STATUS / f"{job.name}.json"
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (record.get("returncode") == 0 and record.get("signature") == job.signature
            and all(path.exists() for path in job.outputs))


def invocation_label(stages: set[str], only: str) -> str:
    """Return a stable filename label for one resumable suite invocation."""
    non_api = {"audit", "table3", "ablation", "hparams", "xdom", "analysis"}
    if stages == non_api:
        label = "non_api"
    elif stages == {"llm"}:
        label = "llm"
    else:
        label = "_".join(sorted(stages))
    if only:
        label += "__only_" + hashlib.sha256(only.encode("utf-8")).hexdigest()[:8]
    return label


def run_job(job: Job, env: dict[str, str], quiet: bool) -> int:
    for output in job.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{job.name}.log"
    result_path = JOB_RESULTS / f"{job.name}.jsonl"
    started = now()
    result_lines: list[dict] = []
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(job.command, cwd=SRC, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if not quiet:
                print(line, end="", flush=True)
            if line.startswith("RESULT "):
                try:
                    result_lines.append(json.loads(line[7:]))
                except json.JSONDecodeError:
                    pass
        returncode = proc.wait()
    with result_path.open("w", encoding="utf-8") as handle:
        for row in result_lines:
            row["_suite_job"] = job.name
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    record = {
        "job": job.name, "stage": job.stage, "signature": job.signature,
        "command": list(job.command), "command_shell": shlex.join(job.command),
        "started_utc": started, "finished_utc": now(), "returncode": returncode,
        "log": str(log_path.relative_to(ROOT)),
        "result_file": str(result_path.relative_to(ROOT)),
        "outputs": [str(path.relative_to(ROOT)) for path in job.outputs],
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "evidence_policy": POLICY,
    }
    tmp = STATUS / f".{job.name}.json.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS / f"{job.name}.json")
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="audit,table3,ablation,hparams,xdom,analysis",
                        help="audit,table3,ablation,hparams,xdom,analysis,llm or all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--only", default="", help="regex selecting job names")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    stages = {part.strip() for part in args.stages.split(",") if part.strip()}
    if "all" in stages:
        stages = {"audit", "table3", "ablation", "hparams", "xdom", "analysis"}
    jobs = build_jobs(stages)
    if args.only:
        pattern = re.compile(args.only)
        jobs = [job for job in jobs if pattern.search(job.name)]
    verify_frozen_contract(jobs)
    print(f"Frozen dataset: {DATASET}")
    print(f"Evidence policy: {POLICY} (PARAM + OCR + VLM only; no arguments)")
    print(f"Jobs selected: {len(jobs)}")
    for job in jobs:
        marker = "SKIP" if (not args.rerun and successful(job)) else "RUN "
        print(f"[{marker}] {job.stage:9s} {job.name}: {shlex.join(job.command)}")
    if not args.execute:
        print("\nDry run. Add --execute to run; completed jobs resume automatically.")
        return 0

    for directory in (OUT, LOGS, STATUS, JOB_RESULTS, PRED, GEOM, ABL, HP, XDOM, XDOM_LLM):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    invocation = invocation_label(stages, args.only)
    environment = {
        "started_utc": now(), "python": sys.version, "platform": platform.platform(),
        "invocation": invocation,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "evidence_policy": POLICY, "selected_stages": sorted(stages),
        "gradient_checkpointing": os.environ.get("CLAIMARC_GRADIENT_CHECKPOINTING", "0"),
        "selected_jobs": [job.name for job in jobs],
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "model_paths": {key: os.environ.get(key, "") for key in (
            "CLAIMARC_BGE_PATH", "CLAIMARC_BERT_PATH", "CLAIMARC_ROBERTA_PATH",
            "CLAIMARC_NLI_PATH", "CLAIMARC_QWEN_PATH",
        )},
    }
    (OUT / f"suite_environment_{invocation}.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = []
    for index, job in enumerate(jobs, 1):
        if not args.rerun and successful(job):
            print(f"[{index}/{len(jobs)}] skip {job.name}", flush=True)
            continue
        print(f"[{index}/{len(jobs)}] run {job.name}", flush=True)
        rc = run_job(job, env, args.quiet)
        if rc:
            failures.append({"job": job.name, "returncode": rc})
            print(f"FAILED {job.name} (rc={rc}); see {LOGS / (job.name + '.log')}",
                  file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
    (OUT / f"suite_failures_{invocation}.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
