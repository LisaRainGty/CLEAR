"""§4.4.2 跨域单折 runner：对一个 holdout 训练 CLAIMARC + BERT/RoBERTa/ESIM，
存下各模型在该折 val/test 上的预测，供 xdom_agg.py 统一计算 Acc/P/R/F1/MacroF1/AUPRC/AUROC。

CLAIMARC 走 train(--save_emb)，bundle 含 train/val/test 的 (g,p,y,c,attr)；
基线走 baselines_ft.run(--save_pred)，bundle 含 val/test 的 (p,y)。

用法：
  python -m models.xdom_fold --dataset DS --mode category --holdout food_and_beverages \
      --outdir /tmp/xdom --models clarc,bert_cls,roberta_cls,esim
"""
from __future__ import annotations

import argparse
import json
import os
import time
from argparse import Namespace

from models import baselines_ft
from models.train import train
from models.xdom_common import build_splits, holdout_rooms


# CLAIMARC canonical 旋钮（对齐 in-domain claimarc_v2：bce / λ0.5 / τ0.07 / Kp3 Kn5）
# enc_train/lr 可被 CLI 覆盖，以支持 full-FT 新 canonical（--enc_train full --lr 1e-5）。
def clarc_args(dataset, seed, save_emb, tag, warmup, cl_epochs,
               enc_train="lora", lr=2e-5, cl_no_attr_block=False,
               cl_class_balanced=False, cl_hard_pos=False,
               rel_soft=False, rel_aux_weight=0.0, rel_pow=1.0, c_transform="none",
               evidence_policy="sources_only"):
    return Namespace(
        dataset=dataset, seed=seed, save_emb=save_emb, tag=tag,
        bs=12, accum=3, lr=lr, lr_head=1e-4,
        warmup=warmup, cl_epochs=cl_epochs, lambda_cl=0.5, pos_weight=-1.0,
        loss="bce", gamma_neg=4.0, gamma_pos=0.0,
        no_cl=False, swa=False, no_fusion=False, n_fusion=2, fusion_dropout=0.2,
        no_lora=False, no_weight=False, lora_rank=16, heads=8, tau=0.07, Kp=3, Kn=5,
        cl_no_attr_block=cl_no_attr_block, cl_class_balanced=cl_class_balanced,
        cl_hard_pos=cl_hard_pos, rel_soft=rel_soft, rel_aux_weight=rel_aux_weight,
        rel_pow=rel_pow, c_transform=c_transform,
        global_neg=False, cl_c_min=0.0, cl_neg_c_min=0.0, cl_teacher_mode="off",
        cl_teacher_conf_min=0.0, cl_neg_filter="none", cl_neg_bonus=0.0,
        cl_neg_bonus_filter="none", source0_ce_scale=1.0, source0_cl_scale=1.0,
        source_rich_ce_scale=1.0, source_rich_cl_scale=1.0,
        distill_bge_weight=0.0, distill_bge_folds=5, distill_teacher_seed=0,
        distill_temp=1.0, distill_conf_min=0.0, distill_c_min=0.0, distill_mode="all",
        backbone="bge", encoder_name=os.environ.get("CLAIMARC_BGE_PATH", "/root/models/bge-large-zh-v1.5"),
        enc_train=enc_train, unfreeze_top=0, no_ret_disc=False, xattn_dir="both",
        indep_proj=False, ffn="swiglu", evidence_policy=evidence_policy, evidence_policy_mix="",
        view_consistency_mix="", view_ce_weight=0.0, view_logit_weight=0.0,
        view_embed_weight=0.0, view_consistency_in_warmup=False,
        source_aux_combo_weight=0.0, source_aux_conf_weight=0.0,
        source_aux_count_weight=0.0, source_aux_in_warmup=False,
        proto_aux_weight=0.0, proto_aux_group="source_bin", proto_aux_mode="ce",
        proto_aux_margin=0.15, proto_aux_tau=0.10, proto_aux_min_class=3,
        proto_aux_c_min=0.10, proto_aux_in_warmup=False,
        save_ckpt="", load_ckpt="", eval_only=False, infer_jsonl="",
        cl_mode="racl", stream_mode="dual", head_concat_only=False, joint_encode=False,
        c_recompute="",
    )


def baseline_args(dataset, seed, kind, save_pred):
    return Namespace(
        dataset=dataset, kind=kind, bs=16, lr=2e-5, epochs=4, seed=seed,
        save_pred=save_pred, infer_jsonl="",
        model_path={"bert_cls": "/root/models/bert-base-chinese",
                    "roberta_cls": "/root/models/chinese-roberta-wwm-ext"}.get(kind, ""),
        loss="bce", gamma_neg=4.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", default="category", choices=["category", "rooms", "time"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="clarc,bert_cls,roberta_cls,esim")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cl_epochs", type=int, default=4)
    ap.add_argument("--enc_train", default="lora", choices=["lora", "topk", "full"])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cl_no_attr_block", action="store_true")
    ap.add_argument("--cl_class_balanced", action="store_true")
    ap.add_argument("--evidence_policy", default="sources_only")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == "rooms" and not args.holdout:
        args.holdout = ",".join(holdout_rooms(args.dataset, 20))
    label = (args.holdout[:24] if args.mode == "category" else args.mode)
    splits = build_splits(args.dataset, args.mode, args.holdout, seed=args.seed)
    # Force fair three-source evidence on all records in this fold.
    for rows in splits.values():
        for r in rows:
            r["_evidence_policy"] = args.evidence_policy
    n_pos = sum(int(r["y"]) for r in splits["test"])
    meta = {"mode": args.mode, "holdout": label, "seed": args.seed,
            "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "n_test": len(splits["test"]), "pos_test": n_pos,
            "evidence_policy": args.evidence_policy}
    print("FOLD_META", json.dumps(meta, ensure_ascii=False), flush=True)
    if n_pos < 3:
        print("SKIP too_few_pos", flush=True)
        return

    want = [m.strip() for m in args.models.split(",") if m.strip()]
    pre = f"{args.mode}_{label}_s{args.seed}"

    if "clarc" in want:
        t0 = time.time()
        emb = os.path.join(args.outdir, f"clarc_{pre}.pt")
        res = train(clarc_args(args.dataset, args.seed, emb, f"xdom_{pre}",
                               args.warmup, args.cl_epochs,
                               enc_train=args.enc_train, lr=args.lr,
                               cl_no_attr_block=args.cl_no_attr_block,
                               cl_class_balanced=args.cl_class_balanced,
                               evidence_policy=args.evidence_policy), splits=splits)
        print("CLARC_RES", json.dumps({"holdout": label, **{k: res.get(k) for k in
              ("auprc", "auroc", "macro_f1", "pos_f1", "auprc_rkc", "auroc_rkc",
               "macro_f1_rkc", "alpha_rkc", "thr")}}, ensure_ascii=False),
              f"[{time.time()-t0:.0f}s]", flush=True)

    for kind in ("bert_cls", "roberta_cls", "esim"):
        if kind not in want:
            continue
        t0 = time.time()
        sp = os.path.join(args.outdir, f"{kind}_{pre}.pt")
        res = baselines_ft.run(baseline_args(args.dataset, args.seed, kind, sp), splits=splits)
        print(f"{kind.upper()}_RES", json.dumps({"holdout": label, **res}, ensure_ascii=False),
              f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
