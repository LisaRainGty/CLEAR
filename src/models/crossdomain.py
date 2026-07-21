"""§4.4.2 跨域适应：留一品类，仅靠检索库注入完成适应（编码器训练后冻结）。

协议：留出品类 X 不进入训练；训练后冻结，把 X 拆成 support/query；
向检索库注入 m∈{0,1,3,5,10} 条 support，RKC 在 query 上投票，记录 Macro-F1(m)。
对照：forward classifier（参数化，无适应）。

用法：python -m models.crossdomain --holdout food_and_beverages --warmup 2 --cl_epochs 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from models.data import ClaimDataset, make_collate, load_split, build_tokenizer, resolve_bge_path
from models.train import train, predict, macro_f1, best_threshold_macroF1, rkc_predict, set_seed


def rkc_with_library(lib_g, lib_y, lib_c, qg, k=10):
    return rkc_predict(lib_g, lib_y, lib_c, qg, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DEFAULT_TRAIN_DATASET))
    ap.add_argument("--holdout", required=True,
                    help="category 模式：品类名；rooms 模式：逗号分隔 room_id 集合")
    ap.add_argument("--holdout_mode", default="category", choices=["category", "rooms"],
                    help="留一品类 (§4.4.2 协议1) 或 留一主播集合 (§4.4.2 协议2)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cl_epochs", type=int, default=3)
    ap.add_argument("--lambda_cl", type=float, default=0.5)
    ap.add_argument("--pos_weight", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # 透传 train() 需要的默认旋钮（与 canonical 一致：LoRA16 / τ0.07 / λ0.5 / ASL γ4 / 非SWA）
    for k, v in dict(no_cl=False, no_fusion=False, n_fusion=2, fusion_dropout=0.2,
                     no_lora=False, no_weight=False, lora_rank=16, heads=8, tau=0.07,
                     Kp=3, Kn=5, global_neg=False, backbone="bge", xattn_dir="both",
                     indep_proj=False, ffn="swiglu", save_emb="", swa=False,
                     loss="asl", gamma_neg=4.0, gamma_pos=0.0,
                     encoder_name="BAAI/bge-large-zh-v1.5",
                     cl_c_min=0.0, cl_neg_c_min=0.0, cl_teacher_conf_min=0.0,
                     cl_teacher_mode="off", view_ce_weight=0.0,
                     tag=f"xdom_{args.holdout_mode}_{args.holdout[:16]}").items():
        setattr(args, k, v)

    set_seed(args.seed)
    full = load_split(args.dataset)
    allrecs = full["train"] + full["val"] + full["test"]
    if args.holdout_mode == "rooms":
        held_set = {x.strip() for x in args.holdout.split(",") if x.strip()}
        held = [r for r in allrecs if r.get("room_id") in held_set]
        rest = [r for r in allrecs if r.get("room_id") not in held_set]
    else:
        held = [r for r in allrecs if r.get("category") == args.holdout]
        rest = [r for r in allrecs if r.get("category") != args.holdout]
    rng = np.random.RandomState(args.seed); rng.shuffle(rest)
    nval = max(50, len(rest) // 10)
    splits = {"train": rest[nval:], "val": rest[:nval], "test": held}
    print(f"[xdom {args.holdout}] train={len(splits['train'])} held={len(held)} "
          f"pos_held={sum(r['y'] for r in held)}", flush=True)
    if sum(r["y"] for r in held) < 3:
        print("RESULT_XDOM", json.dumps({"holdout": args.holdout, "holdout_mode": args.holdout_mode,
              "skip": "too_few_pos"}), flush=True)
        return

    model, loaders, device, train_pack, res = train(args, splits=splits, return_model=True)
    lib_g, lib_y, lib_c = train_pack

    tok = build_tokenizer(resolve_bge_path())
    collate = make_collate(tok.pad_token_id)
    held_loader = DataLoader(ClaimDataset(held, tok), batch_size=args.bs,
                             shuffle=False, collate_fn=collate, num_workers=6, pin_memory=True)
    p_fwd, hg, hy, hc, hattr = predict(model, held_loader, device)

    # support/query 拆分（分层保正例）
    idx = np.arange(len(hy)); rng.shuffle(idx)
    pos_idx = [i for i in idx if hy[i] == 1]
    sup_pool = pos_idx[: max(10, len(pos_idx) // 2)] + [i for i in idx if hy[i] == 0][:50]
    sup_pool = np.array(sup_pool)
    query = np.array([i for i in idx if i not in set(sup_pool.tolist())])
    if len(query) < 10 or len(set(hy[query])) < 2:
        query = idx; sup_pool = idx

    thr = res["thr"]
    hold_label = ("rooms_" + str(len(held_set))) if args.holdout_mode == "rooms" else args.holdout
    out = {"holdout": hold_label, "holdout_mode": args.holdout_mode,
           "n_held": int(len(held)), "pos_held": int(sum(hy)),
           "forward_macro_f1": round(macro_f1(hy[query],
            (p_fwd[query] >= thr).astype(int)), 4), "rkc": {}}
    for m in (0, 1, 3, 5, 10):
        if m == 0:
            g2, y2, c2 = lib_g, lib_y, lib_c
        else:
            take = sup_pool[:m]
            g2 = torch.cat([lib_g, hg[take]], 0)
            y2 = np.concatenate([lib_y, hy[take]])
            c2 = np.concatenate([lib_c, hc[take]])
        prkc = rkc_with_library(g2, y2, c2, hg[query])
        out["rkc"][str(m)] = round(macro_f1(hy[query], (prkc >= 0.5).astype(int)), 4)
    print("RESULT_XDOM", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
