"""§4.4.2 跨域单折 LLM runner：对一个 holdout，用源域 train 构造 few-shot 示例，
在该折 val（封顶采样，调阈值）+ test（全量）上跑 zero-shot 与 few-shot，存逐项概率。

LLM 不训练，跨域即在 held-out 子集上评估；few-shot 示例严格取自源域 train（天然跨域）。
为控制网关成本，val 仅分层采样至多 --val_cap 条用于选阈值。

用法（已设 MATPOOL_API_KEY）：
  python -m models.xdom_llm --dataset DS --mode category --holdout food_and_beverages \
      --model Qwen-Flash --outdir /tmp/xdom --val_cap 200
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from models.run_llm_baselines import build_fewshot, score_split
from models.xdom_common import build_splits, holdout_rooms


def _arr(res, key):
    return np.array([(x.get(key) if x.get(key) is not None
                      else (0.5 if key == "risk_score" else 0)) for x in res], dtype=float)


def cap_val(val, cap, seed):
    if len(val) <= cap:
        return val
    rng = random.Random(seed)
    pos = [r for r in val if int(r.get("y", 0)) == 1]
    neg = [r for r in val if int(r.get("y", 0)) == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    half = cap // 2
    pick = pos[:half] + neg[:cap - half]
    rng.shuffle(pick)
    return pick


def run_mode(model, mode, shots, seed, train, val, test, conc, max_tokens, ns_pre):
    fewshot = build_fewshot(train, shots, seed) if mode == "fewshot" else ""
    rv = score_split(val, model, fewshot, f"{ns_pre}_{mode}_val", conc, max_tokens)
    rt = score_split(test, model, fewshot, f"{ns_pre}_{mode}_test", conc, max_tokens)
    return {
        "p_val": _arr(rv, "risk_score").tolist(),
        "y_val": [int(r["y"]) for r in val],
        "p": _arr(rt, "risk_score").tolist(),
        "y": [int(r["y"]) for r in test],
        "c": [float(r.get("c", 0.05)) for r in test],
        "attr": [r.get("attribute_id", "") for r in test],
        "n_err": int(sum(1 for x in rt if x.get("__error__"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", default="category", choices=["category", "rooms", "time"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--model", default="Qwen-Flash")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--val_cap", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=320)
    ap.add_argument("--modes", default="zero,fewshot")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == "rooms" and not args.holdout:
        args.holdout = ",".join(holdout_rooms(args.dataset, 20))
    label = (args.holdout[:24] if args.mode == "category" else args.mode)
    splits = build_splits(args.dataset, args.mode, args.holdout, seed=args.seed)
    train, val, test = splits["train"], cap_val(splits["val"], args.val_cap, args.seed), splits["test"]
    n_pos = sum(int(r["y"]) for r in test)
    print("FOLD_META", json.dumps({"mode": args.mode, "holdout": label,
          "n_test": len(test), "pos_test": n_pos, "n_val_capped": len(val)},
          ensure_ascii=False), flush=True)
    if n_pos < 3:
        print("SKIP too_few_pos", flush=True)
        return

    ns_pre = f"xdomllm_{args.model}_{args.mode}_{label}_s{args.seed}"
    out = {"model": args.model, "mode": args.mode, "holdout": label}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        out[mode] = run_mode(args.model, mode, args.shots, args.seed,
                             train, val, test, args.concurrency, args.max_tokens, ns_pre)
        print(f"LLM_{mode.upper()}_DONE", label, "n_err=", out[mode]["n_err"], flush=True)
    path = os.path.join(args.outdir, f"llm_{args.model}_{args.mode}_{label}_s{args.seed}.pt")
    torch.save(out, path)
    print("LLM_SAVED", path, flush=True)


if __name__ == "__main__":
    main()
