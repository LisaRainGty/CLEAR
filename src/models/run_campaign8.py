"""Paper campaign8 architecture ablations under sources_only.

  c8_concat  head uses [h_c, h_e] only (drop ESIM 4-tuple)
  c8_joint   joint encode then split streams

Results -> results/campaign8_results.jsonl
"""
from __future__ import annotations
import json, os, subprocess, sys, time

import config

PY = os.environ.get("CLAIMARC_PYTHON", sys.executable)
ROOT = str(config.ROOT)
SRC = os.path.join(ROOT, "src")
DS = str(config.DATASET_SUPERVISED)
OUT = os.path.join(ROOT, "results", "campaign8_results.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ENC = os.environ.get("CLAIMARC_BGE_PATH", "/root/models/bge-large-zh-v1.5")
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


def tjob(tag, extra, seeds=S3):
    jobs = []
    for s in seeds:
        cmd = [PY, "-m", "models.train", "--dataset", DS, "--seed", str(s),
               "--tag", tag, *FULL, *CANON, *extra]
        jobs.append(("train", f"{tag}__s{s}", cmd))
    return jobs


JOBS = []
JOBS += tjob("c8_concat", B + ["--head_concat_only"])
JOBS += tjob("c8_joint",  B + ["--joint_encode"])


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


def run_proc(job_key, cmd):
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
        p.wait()
    except Exception as e:
        print(f"[JOB-ERR] {job_key}: {e}", flush=True)
    print(f"[DONE] {job_key} in {(time.time()-t0)/60:.1f} min (got={got})", flush=True)
    return got


def main():
    done = done_tags()
    print(f"[campaign8] sources_only | {len(JOBS)} jobs, {len(done)} done", flush=True)
    for kind, key, cmd in JOBS:
        if key in done:
            print(f"[skip] {key}", flush=True); continue
        got = run_proc(key, cmd)
        if not got:
            log_result(key, {"error": "no_result"})
    print("\n######## RUN_CAMPAIGN8 COMPLETE ########", flush=True)


if __name__ == "__main__":
    main()
