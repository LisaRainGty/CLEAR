"""基线模型（与 CLAIMARC 同监督对照）。

- frozen BGE 句向量 + LogisticRegression（带 sample weight c），分别用
  claim-only / evidence-only / concat 三种特征，作为"无显式交互"对照。
所有基线共用 §3 的文本（claim passage / 三源证据拼接），用 val 调阈值，test 报 F1/AUC/AP。

用法：python -m models.baselines --dataset data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl
"""
from __future__ import annotations

import argparse
import json

import config

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

from models.data import load_split, resolve_bge_path


def claim_text(r):
    c = r.get("claim", {}) or {}
    segs = c.get("segments", []) or []
    return (r.get("attribute_name", "") + " " + " ".join(s.get("text", "") for s in segs)).strip()


def evidence_text(r, policy: str | None = None):
    """Build evidence string. Default / canonical: three sources only (no arguments)."""
    parts = [r.get("attribute_name", "")]

    def source_parts():
        for it in r.get("evidence_params", []) or []:
            yield it.get("raw_text", "")
        for it in r.get("evidence_ocr", []) or []:
            yield it.get("raw_text", "")
        for it in r.get("evidence_vlm", []) or []:
            yield it.get("raw_quote", "")

    # Paper-fair canonical: params + OCR + VLM only (ignore LLM arguments).
    pol = policy or r.get("_evidence_policy") or r.get("evidence_policy") or getattr(
        config, "EVIDENCE_POLICY_CANONICAL", "sources_only"
    )
    if pol in ("", "record", "sources_only", "source_only", "no_args", "args_first",
               "source_first", "args_only"):
        # All paper runs force three-source evidence; argument views are retired.
        parts.extend(source_parts())
    else:
        parts.extend(source_parts())
    return " ".join(p for p in parts if p).strip()


def embed_all(texts, bge_path, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(bge_path, device=device)
    return np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False))


def metrics(y, p, thr, c=None):
    pred = (p >= thr).astype(int)
    w = np.clip(c, 0.05, None) if c is not None else None
    return {
        "acc": round(float((pred == y).mean()), 4),
        "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
        "wF1": round(f1_score(y, pred, average="macro", sample_weight=w, zero_division=0), 4)
        if w is not None else None,
        "auprc": round(average_precision_score(y, p), 4) if len(set(y)) > 1 else None,
        "auroc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
    }


def best_thr(y, p):
    return max(np.linspace(0.05, 0.95, 19),
               key=lambda t: f1_score(y, (p >= t).astype(int), average="macro", zero_division=0))


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import os as _os
    _local = "/root/models/bge-large-zh-v1.5"
    bge = _local if _os.path.isdir(_local) else resolve_bge_path()
    sp = load_split(args.dataset)
    out = []
    for name, feat in (("claim_only", claim_text), ("evidence_only", evidence_text),
                       ("concat", None)):
        def build(recs):
            if feat is None:
                ec = embed_all([claim_text(r) for r in recs], bge, device)
                ee = embed_all([evidence_text(r) for r in recs], bge, device)
                return np.concatenate([ec, ee], axis=1)
            return embed_all([feat(r) for r in recs], bge, device)
        Xtr, Xv, Xte = build(sp["train"]), build(sp["val"]), build(sp["test"])
        ytr = np.array([r["y"] for r in sp["train"]]); ctr = np.array([r["c"] for r in sp["train"]])
        yv = np.array([r["y"] for r in sp["val"]]); yte = np.array([r["y"] for r in sp["test"]])
        cte = np.array([r["c"] for r in sp["test"]])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, ytr, sample_weight=np.clip(ctr, 0.05, None))
        pv = clf.predict_proba(Xv)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
        thr = float(best_thr(yv, pv))
        res = {"tag": f"BGE_frozen_LR_{name}", "thr": round(thr, 3),
               **metrics(yte, pte, thr, c=cte),
               "n_test": int(len(yte)), "pos_test": int(yte.sum())}
        print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DEFAULT_TRAIN_DATASET))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
