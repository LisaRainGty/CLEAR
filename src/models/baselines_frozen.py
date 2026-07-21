"""Frozen-representation & classical baselines (same split / supervision / metrics as CLAIMARC).

One process emits many RESULT lines, one per sub-baseline, each with a distinct `tag`.
All use the reliability weight c as sample weight, pick the threshold on val by Macro-F1,
and report Acc / pos-F1 / Macro-F1 / wF1 / AUPRC / AUROC / ECE on test.

Families covered (methodologically aligned to a retrieval + contrastive framework):
  (B-classical)  TF-IDF(word) + {LR, LinearSVM, ComplementNB};  char n-gram + LR (fastText-style)
  (B-frozen)     BGE-large frozen [c,e,c-e,c*e] + {LR, LinearSVM, MLP}
  (B-retrieval)  BGE frozen + reliability-weighted kNN {global, attribute-blocked}
  (A-noninter)   BGE frozen dual-encoder cosine (no cross interaction)
  (probe)        RoBERTa-wwm frozen mean-pool + LR (different encoder)

Usage (remote):
  python -m models.baselines_frozen --dataset .../dataset.jsonl --save_dir .../frozen_preds
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from models.data import load_split
from models.baselines import claim_text, evidence_text
from models.train import macro_f1, best_threshold_macroF1, ece

BGE = "/root/models/bge-large-zh-v1.5"
ROBERTA = "/root/models/chinese-roberta-wwm-ext"


def _metrics(tag, yv, pv, y, p, c, seed=0):
    thr = best_threshold_macroF1(yv, pv)
    pred = (p >= thr).astype(int)
    return {
        "tag": tag, "seed": seed, "thr": round(float(thr), 3),
        "acc": round(float((pred == y).mean()), 4),
        "macro_f1": round(macro_f1(y, pred), 4),
        "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
        "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
        "auprc": round(average_precision_score(y, p), 4) if len(set(y)) > 1 else None,
        "auroc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
        "ece": round(ece(y, p), 4),
        "n_test": int(len(y)), "pos_test": int(y.sum()),
    }


def sbert_embed(path, texts, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path, device=device)
    return np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=64,
                               show_progress_bar=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--save_dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    sp = load_split(args.dataset)
    Y = {s: np.array([int(r.get("y", 0)) for r in sp[s]]) for s in sp}
    C = {s: np.array([float(r.get("c", 0.05)) for r in sp[s]]) for s in sp}
    A = {s: np.array([r.get("attribute_id", "") for r in sp[s]]) for s in sp}
    results = []
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    def emit(res, pv=None, p=None):
        print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
        results.append(res)
        if args.save_dir and pv is not None:
            torch.save({"thr": res["thr"], "val": {"p": pv, "y": Y["val"]},
                        "test": {"p": p, "y": Y["test"], "c": C["test"], "attr": A["test"]}},
                       os.path.join(args.save_dir, res["tag"] + ".pt"))

    # ---------- classical TF-IDF family ----------
    def text_concat(r):
        return (claim_text(r) + " [SEP] " + evidence_text(r)).strip()
    Tc = {s: [text_concat(r) for r in sp[s]] for s in sp}

    def tfidf_run(tag, vec, clf, calibrate=False):
        Xtr = vec.fit_transform(Tc["train"]); Xv = vec.transform(Tc["val"]); Xte = vec.transform(Tc["test"])
        model = CalibratedClassifierCV(clf, method="sigmoid", cv=3) if calibrate else clf
        model.fit(Xtr, Y["train"], sample_weight=np.clip(C["train"], 0.05, None))
        pv = model.predict_proba(Xv)[:, 1]; p = model.predict_proba(Xte)[:, 1]
        emit(_metrics(tag, Y["val"], pv, Y["test"], p, C["test"], args.seed), pv, p)

    tfidf_run("TFIDF_word_LR",
              TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=2, max_features=50000),
              LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0))
    tfidf_run("TFIDF_word_SVM",
              TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=2, max_features=50000),
              LinearSVC(class_weight="balanced", C=1.0), calibrate=True)
    tfidf_run("TFIDF_word_NB",
              TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), min_df=2, max_features=50000),
              ComplementNB())
    tfidf_run("CharNgram_LR",
              TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=80000),
              LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0))

    # ---------- frozen encoder feature families ----------
    def feat4(ce, ee):
        return np.concatenate([ce, ee, ce - ee, ce * ee], axis=1)

    def encode_pair(path):
        out = {}
        for s in ("train", "val", "test"):
            ce = sbert_embed(path, [claim_text(r) for r in sp[s]], device)
            ee = sbert_embed(path, [evidence_text(r) for r in sp[s]], device)
            out[s] = (ce, ee)
        return out

    feats = encode_pair(BGE)

    X4 = {s: feat4(*feats[s]) for s in feats}
    # BGE frozen + LR / SVM / MLP
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    clf.fit(X4["train"], Y["train"], sample_weight=np.clip(C["train"], 0.05, None))
    emit(_metrics("BGEfz_LR_4tuple", Y["val"], clf.predict_proba(X4["val"])[:, 1],
                  Y["test"], clf.predict_proba(X4["test"])[:, 1], C["test"], args.seed),
         clf.predict_proba(X4["val"])[:, 1], clf.predict_proba(X4["test"])[:, 1])
    svm = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), method="sigmoid", cv=3)
    svm.fit(X4["train"], Y["train"], sample_weight=np.clip(C["train"], 0.05, None))
    emit(_metrics("BGEfz_SVM_4tuple", Y["val"], svm.predict_proba(X4["val"])[:, 1],
                  Y["test"], svm.predict_proba(X4["test"])[:, 1], C["test"], args.seed),
         svm.predict_proba(X4["val"])[:, 1], svm.predict_proba(X4["test"])[:, 1])
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400, early_stopping=True,
                        random_state=args.seed)
    mlp.fit(X4["train"], Y["train"])
    emit(_metrics("BGEfz_MLP_4tuple", Y["val"], mlp.predict_proba(X4["val"])[:, 1],
                  Y["test"], mlp.predict_proba(X4["test"])[:, 1], C["test"], args.seed),
         mlp.predict_proba(X4["val"])[:, 1], mlp.predict_proba(X4["test"])[:, 1])

    # concat[c,e] for retrieval-style kNN (matches frozen retrieval probe)
    Xcat = {s: np.concatenate(feats[s], axis=1) for s in feats}
    for s in Xcat:
        Xcat[s] = Xcat[s] / (np.linalg.norm(Xcat[s], axis=1, keepdims=True) + 1e-8)

    def knn_probs(split, k, attr_block):
        Xtr = Xcat["train"]; ytr = Y["train"].astype(float); ctr = np.clip(C["train"], 0.05, None)
        atr = A["train"]; Xs = Xcat[split]; ats = A[split]
        sims = Xs @ Xtr.T
        pr = np.zeros(len(Xs))
        for i in range(len(Xs)):
            if attr_block:
                idx = np.where(atr == ats[i])[0]
                if len(idx) < 3:
                    idx = np.arange(len(atr))
            else:
                idx = np.arange(len(atr))
            s = sims[i, idx]
            kk = min(k, len(idx))
            top = idx[np.argpartition(-s, kk - 1)[:kk]]
            w = ctr[top] * np.clip(sims[i, top], 0, None)
            pr[i] = (w * ytr[top]).sum() / (w.sum() + 1e-8)
        return pr
    for k in (15,):
        emit(_metrics(f"BGEfz_kNN_global_k{k}", Y["val"], knn_probs("val", k, False),
                      Y["test"], knn_probs("test", k, False), C["test"], args.seed))
        emit(_metrics(f"BGEfz_kNN_attr_k{k}", Y["val"], knn_probs("val", k, True),
                      Y["test"], knn_probs("test", k, True), C["test"], args.seed))

    # dual-encoder cosine (no interaction) -> [0,1]
    def cos_probs(split):
        ce, ee = feats[split]
        return ((ce * ee).sum(1) + 1.0) / 2.0
    emit(_metrics("BGEfz_dual_cosine", Y["val"], cos_probs("val"),
                  Y["test"], cos_probs("test"), C["test"], args.seed))

    # ---------- RoBERTa-wwm frozen mean-pool probe ----------
    try:
        rfeats = encode_pair(ROBERTA)
        Xr = {s: feat4(*rfeats[s]) for s in rfeats}
        rclf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        rclf.fit(Xr["train"], Y["train"], sample_weight=np.clip(C["train"], 0.05, None))
        emit(_metrics("RoBERTafz_LR_4tuple", Y["val"], rclf.predict_proba(Xr["val"])[:, 1],
                      Y["test"], rclf.predict_proba(Xr["test"])[:, 1], C["test"], args.seed),
             rclf.predict_proba(Xr["val"])[:, 1], rclf.predict_proba(Xr["test"])[:, 1])
    except Exception as e:
        print(f"[roberta-probe skipped] {repr(e)[:150]}", flush=True)

    print(f"[frozen] done {len(results)} sub-baselines", flush=True)


if __name__ == "__main__":
    main()
