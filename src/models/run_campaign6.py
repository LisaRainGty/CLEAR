"""Paper campaign6 (sources_only fair protocol).

Canonical: BGE full fine-tuning + RACL (no attr-block, class-balanced) +
           evidence_policy=sources_only (params/OCR/VLM only; no LLM arguments).

Waves (resumable):
  W1a  c6_canon x3 seeds + --save_emb
  W1b  xdom leave-one-category (10 folds, s0)  — CLAIMARC + baselines
  W1c  xdom leave-20-streamers (s0,1,2)
  W2   core ablations (3 seeds)
  W3   c-formula hyperparameter sweep (1 seed)

Results -> results/campaign6_results.jsonl
"""
from __future__ import annotations
import json, os, subprocess, sys, time

import config

PY = os.environ.get("CLAIMARC_PYTHON", sys.executable)
ROOT = str(config.ROOT)
SRC = os.path.join(ROOT, "src")
DS = str(config.DATASET_SUPERVISED)
OUT = os.path.join(ROOT, "results", "campaign6_results.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
EMB = os.path.join(ROOT, "embeddings", "c6"); os.makedirs(EMB, exist_ok=True)
XD = os.path.join(ROOT, "results", "xdom_c6"); os.makedirs(XD, exist_ok=True)

ENC = os.environ.get("CLAIMARC_BGE_PATH", "/root/models/bge-large-zh-v1.5")
BERT = os.environ.get("CLAIMARC_BERT_PATH", "/root/models/bert-base-chinese")
ROBERTA = os.environ.get("CLAIMARC_ROBERTA_PATH", "/root/models/chinese-roberta-wwm-ext")

ENV = {**os.environ,
       "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
       "PYTHONPATH": SRC, "TOKENIZERS_PARALLELISM": "false",
       "CLAIMARC_ROOT": ROOT}

FULL = ["--warmup", "3", "--cl_epochs", "6", "--bs", "12", "--accum", "3"]
EV = ["--evidence_policy", "sources_only"]
CANON = ["--tau", "0.07", "--lambda_cl", "0.5", "--Kp", "3", "--Kn", "5",
         "--loss", "bce", "--encoder_name", ENC, "--enc_train", "full", "--lr", "1e-5", *EV]
B = ["--cl_no_attr_block", "--cl_class_balanced"]
S3 = (0, 1, 2)
CATS = ["apparel_and_underwear", "general", "baby_kids_and_pets", "shoes_and_bags",
        "food_and_beverages", "smart_home", "digital_and_electronics",
        "sports_and_outdoor", "beauty_and_personal_care", "jewelry_and_collectibles"]


def tjob(tag, extra, seeds=S3, save=False):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *CANON, *extra]
        if save:
            cmd += ["--save_emb", os.path.join(EMB, f"{tag}_s{s}.pt")]
        jobs.append(("train", f"{tag}__s{s}", cmd))
    return jobs


def xjob(mode, holdout, seed, models="clarc,bert_cls,roberta_cls,esim"):
    label = holdout[:24] if mode == "category" else mode
    bundle = os.path.join(XD, f"clarc_{mode}_{label}_s{seed}.pt")
    cmd = [PY, "-m", "models.xdom_fold", "--dataset", DS, "--mode", mode,
           "--outdir", XD, "--seed", str(seed), "--models", models,
           "--warmup", "3", "--cl_epochs", "6", "--enc_train", "full", "--lr", "1e-5",
           "--cl_no_attr_block", "--cl_class_balanced",
           "--evidence_policy", "sources_only"]
    if mode == "category":
        cmd += ["--holdout", holdout]
    return ("xdom", f"xdom_{mode}_{label}_s{seed}", cmd, bundle)


JOBS = []
JOBS += tjob("c6_canon", B, save=True)
for c in CATS:
    JOBS.append(xjob("category", c, 0))
for s in S3:
    JOBS.append(xjob("rooms", "", s))
JOBS += tjob("c6_no_cl", B + ["--no_cl"])
JOBS += tjob("c6_no_fusion", B + ["--no_fusion"])
JOBS += tjob("c6_no_weight", B + ["--no_weight"])
JOBS += tjob("c6_attrblock", ["--cl_class_balanced"])
JOBS += tjob("c6_lora", ["--cl_no_attr_block", "--cl_class_balanced",
                         "--enc_train", "lora", "--lr", "2e-5"], save=True)
JOBS += tjob("c6_bert", B + ["--backbone", "bert", "--encoder_name", BERT])
for spec, tag in [("k=1.5,lambda=0.3,rho=0.4,phi=1.2", "c6_cf_k1p5"),
                  ("k=6,lambda=0.3,rho=0.4,phi=1.2", "c6_cf_k6"),
                  ("k=3,lambda=0.1,rho=0.4,phi=1.2", "c6_cf_lam0p1"),
                  ("k=3,lambda=0.6,rho=0.4,phi=1.2", "c6_cf_lam0p6"),
                  ("k=3,lambda=0.3,rho=0.2,phi=1.2", "c6_cf_rho0p2"),
                  ("k=3,lambda=0.3,rho=0.6,phi=1.2", "c6_cf_rho0p6"),
                  ("k=3,lambda=0.3,rho=0.4,phi=1.0", "c6_cf_phi1p0"),
                  ("k=3,lambda=0.3,rho=0.4,phi=1.5", "c6_cf_phi1p5")]:
    JOBS += tjob(tag, B + ["--c_recompute", spec], seeds=(0,))


def done_tags():
    if not os.path.exists(OUT):
        return set()
    tags = set()
    for line in open(OUT):
        try:
            tags.add(json.loads(line)["_job"])
        except Exception:
            pass
    return tags


def log_result(job_key, payload):
    payload["_job"] = job_key
    payload["_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_proc(job_key, cmd, want_prefixes=("RESULT ",)):
    print(f"\n{'='*72}\n[RUN] {job_key}\n{'='*72}", flush=True)
    t0 = time.time(); got = False
    try:
        p = subprocess.Popen(cmd, cwd=SRC, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=ENV)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            if line.startswith("RESULT "):
                try:
                    log_result(job_key, json.loads(line[7:])); got = True
                except Exception as e:
                    print(f"[parse-err] {e}", flush=True)
            elif line.startswith("CLARC_RES "):
                got = True
        p.wait()
    except Exception as e:
        print(f"[JOB-ERR] {job_key}: {e}", flush=True)
    print(f"[DONE] {job_key} in {(time.time()-t0)/60:.1f} min (got={got})", flush=True)
    return got


def main():
    done = done_tags()
    print(f"[campaign6] sources_only | {len(JOBS)} jobs, {len(done)} done", flush=True)
    for item in JOBS:
        kind, key = item[0], item[1]
        if kind == "train":
            if key in done:
                print(f"[skip] {key}", flush=True); continue
            got = run_proc(key, item[2])
            if not got:
                log_result(key, {"error": "no_result"})
        else:
            cmd, bundle = item[2], item[3]
            if os.path.exists(bundle):
                print(f"[skip] {key} (bundle exists)", flush=True); continue
            run_proc(key, cmd)
    print("\n######## RUN_CAMPAIGN6 COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
