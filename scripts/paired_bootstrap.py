#!/usr/bin/env python3
"""Paired example-level bootstrap for the fair Table 3 prediction bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def load_bundle(path: Path):
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "val": raw["validation"],
            "test": raw["test"],
            "provenance": {key: raw.get(key) for key in (
                "dataset", "dataset_sha256", "evidence_policy", "resolved_model",
                "seed", "tag",
            )},
        }
    return torch.load(path, map_location="cpu", weights_only=False)


def arrays(bundle, split):
    block = bundle[split]
    return (np.asarray(block["p"], dtype=float), np.asarray(block["y"], dtype=int),
            list(block.get("pair_id", [])))


def ensemble(paths):
    bundles = [load_bundle(path) for path in paths]
    if not bundles:
        raise ValueError("empty bundle list")
    out = {}
    for split in ("val", "test"):
        triples = [arrays(bundle, split) for bundle in bundles]
        y0, id0 = triples[0][1], triples[0][2]
        for _, y, ids in triples[1:]:
            if not np.array_equal(y0, y) or ids != id0:
                raise ValueError(f"seed bundles disagree on {split} order")
        out[split] = {"p": np.mean([t[0] for t in triples], axis=0), "y": y0,
                      "pair_id": id0}
    return out


def threshold(y, p):
    return float(max(np.linspace(0.01, 0.99, 99), key=lambda t:
                     f1_score(y, p >= t, average="macro", zero_division=0)))


def scores(y, p, t):
    pred = p >= t
    return {
        "ap": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "acc": float(np.mean(pred == y)),
        "f1_pos": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def paired(canonical, baseline, repetitions, seed):
    for split in ("val", "test"):
        if canonical[split]["pair_id"] != baseline[split]["pair_id"]:
            raise ValueError(f"pair_id mismatch on {split}")
        if not np.array_equal(canonical[split]["y"], baseline[split]["y"]):
            raise ValueError(f"label mismatch on {split}")
    tc = threshold(canonical["val"]["y"], canonical["val"]["p"])
    tb = threshold(baseline["val"]["y"], baseline["val"]["p"])
    y = canonical["test"]["y"]
    pc, pb = canonical["test"]["p"], baseline["test"]["p"]
    observed_c, observed_b = scores(y, pc, tc), scores(y, pb, tb)
    rng = np.random.RandomState(seed)
    deltas = {key: [] for key in observed_c}
    accepted = 0
    while accepted < repetitions:
        index = rng.randint(0, len(y), len(y))
        yy = y[index]
        if len(np.unique(yy)) < 2:
            continue
        sc, sb = scores(yy, pc[index], tc), scores(yy, pb[index], tb)
        for key in deltas:
            deltas[key].append(sc[key] - sb[key])
        accepted += 1
    summary = {}
    for key, values in deltas.items():
        values = np.asarray(values)
        p = min(1.0, 2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
        summary[key] = {
            "canonical": observed_c[key], "baseline": observed_b[key],
            "delta": observed_c[key] - observed_b[key],
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "p_two_sided": p,
        }
    return {"canonical_threshold": tc, "baseline_threshold": tb, "metrics": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    canonical_paths = [root / f"embeddings/fair_rerun/emb_geom/emb_geom_racl_s{s}.pt"
                       for s in range(3)]
    if not all(path.exists() for path in canonical_paths):
        raise FileNotFoundError("canonical three-seed bundles are incomplete")
    canonical = ensemble(canonical_paths)
    base = root / "embeddings/fair_rerun/baseline_predictions"
    patterns = {
        "BERT-CLS": [base / f"bert_cls_s{s}.pt" for s in range(3)],
        "RoBERTa-CLS": [base / f"roberta_cls_s{s}.pt" for s in range(3)],
        "BERT-NLI": [base / f"bert_nli_s{s}.pt" for s in range(3)],
        "ESIM": [base / f"esim_s{s}.pt" for s in range(3)],
        "TextCNN": [base / f"textcnn_s{s}.pt" for s in range(3)],
        "BiLSTM": [base / f"bilstm_s{s}.pt" for s in range(3)],
        "DAM": [base / f"dam_s{s}.pt" for s in range(3)],
        "Qwen2.5-7B QLoRA": [base / f"qwen2p5_7b_qlora_s{s}.pt" for s in range(3)],
        # The paper defines frozen probes and prompted LLMs as one fixed run.
        "BGE frozen + LR": [base / "frozen_s0/BGEfz_LR_4tuple.pt"],
        "BGE frozen + SVM": [base / "frozen_s0/BGEfz_SVM_4tuple.pt"],
        "BGE frozen + MLP": [base / "frozen_s0/BGEfz_MLP_4tuple.pt"],
        "BGE frozen + kNN": [base / "frozen_s0/BGEfz_kNN_attr_k15.pt"],
        "Qwen-Flash zero-shot": [root / "results/fair_rerun/llm_qwen_flash_zero.json"],
        "Qwen-Flash five-shot": [root / "results/fair_rerun/llm_qwen_flash_fs5.json"],
        "GPT-5.4 zero-shot": [root / "results/fair_rerun/llm_gpt54_zero.json"],
        "GPT-5.4 five-shot": [root / "results/fair_rerun/llm_gpt54_fs5.json"],
        "Gemini-3.5-Flash zero-shot": [root / "results/fair_rerun/llm_gemini35_zero.json"],
        "Gemini-3.5-Flash five-shot": [root / "results/fair_rerun/llm_gemini35_fs5.json"],
        "Kimi-K2.6 zero-shot": [root / "results/fair_rerun/llm_kimi_zero.json"],
        "Kimi-K2.6 five-shot": [root / "results/fair_rerun/llm_kimi_fs5.json"],
    }
    result = {"repetitions": args.repetitions, "seed": args.seed, "comparisons": {},
              "missing": {}}
    for name, paths in patterns.items():
        missing = [str(path.relative_to(root)) for path in paths if not path.exists()]
        if missing:
            result["missing"][name] = missing
            continue
        result["comparisons"][name] = paired(
            canonical, ensemble(paths), args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[written] {args.output}")
    if result["missing"]:
        print(f"[warning] incomplete comparisons: {sorted(result['missing'])}")


if __name__ == "__main__":
    main()
