"""Controlled 3-variant geometry campaign for RQ2 (embedding geometry, §4.6).

Trains, with an identical BGE full-fine-tuning backbone, classifier head and
hyperparameters, three variants that differ ONLY in the contrastive objective:
    none    -- BCE only
    supcon  -- faithful in-batch supervised contrastive (Khosla et al., 2020)
    racl    -- retrieval-augmented contrast (ours)
Each variant is run for the given seeds; the test+train embedding bundles are
dumped so geom_probe2 can measure label-conditional geometry. Resumable: skips
any (variant, seed) whose bundle already exists.

Usage (remote):
  python -m models.run_geom_campaign --dataset <DS> --outdir data/final/emb_geom --seeds 0 1 2
"""
from __future__ import annotations
import argparse, os, subprocess, sys

ENC = "BAAI/bge-large-zh-v1.5"
BASE = ["--encoder_name", ENC, "--enc_train", "full", "--lr", "1e-5",
        "--tau", "0.07", "--lambda_cl", "0.5", "--Kp", "3", "--Kn", "5",
        "--loss", "bce", "--warmup", "3", "--cl_epochs", "6",
        "--evidence_policy", "sources_only"]
VARIANT_FLAGS = {
    "none":   ["--cl_mode", "none"],
    "supcon": ["--cl_mode", "supcon"],
    "racl":   ["--cl_mode", "racl", "--cl_no_attr_block", "--cl_class_balanced"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default="data/final/emb_geom")
    ap.add_argument("--prefix", default="emb_geom_")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="*", default=["none", "supcon", "racl"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    jobs = []
    for v in args.variants:
        for s in args.seeds:
            out = os.path.join(args.outdir, f"{args.prefix}{v}_s{s}.pt")
            jobs.append((v, s, out))

    done = [j for j in jobs if os.path.exists(j[2])]
    print(f"[geom] {len(jobs)} jobs, {len(done)} bundles already present", flush=True)
    for v, s, out in jobs:
        if os.path.exists(out):
            print(f"[geom] skip {v} s{s} (exists)", flush=True)
            continue
        cmd = [sys.executable, "-m", "models.train",
               "--dataset", args.dataset, "--seed", str(s),
               "--save_emb", out, *BASE, *VARIANT_FLAGS[v]]
        print(f"[geom] RUN {v} s{s}\n       {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[geom] FAILED {v} s{s} rc={r.returncode}", flush=True)
            sys.exit(r.returncode)
    print("[geom] all done", flush=True)


if __name__ == "__main__":
    main()
