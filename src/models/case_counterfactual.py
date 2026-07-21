#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""受控反事实探针（§4.4.7 机制案例，对标 Mei et al. 2024 RGCL 的 confounder 构造）。

对每一对构造样本：固定主播话术，仅微调商品事实使“被消费者感知为误导”的标签翻转
(mislead y=1  vs  ok y=0)。把同一对喂给三种判别方式：
  - CLAIMARC（完整 RACL + RKC 检索投票）
  - CLAIMARC w/o RACL（同架构去对比学习，作为机制对照）
  - BERT 单流微调 / LLM 网关
度量：
  1) 两成员在检索表示 g 上的余弦相似度（full vs no_cl）——验证“只有 RACL 把它们拉开”；
  2) 各方法对两成员的判定是否同时正确——验证“只有 CLAIMARC 通过 RKC 同时判对”。
输出 JSON + 一张相似度对比图。
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from models.train import rkc_attr_predict, best_threshold_macroF1, macro_f1


def load_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def gnorm(g):
    return F.normalize(torch.as_tensor(g).float(), dim=-1)


def claimarc_decision(bundle):
    """复现 evaluate：在 val 上搜 (alpha, thr) 最大化 Macro-F1，应用到 test。
    返回 test 的 RKC 融合分数、阈值、alpha、纯 cls 分。"""
    tr, va, te = bundle["train"], bundle["val"], bundle["test"]
    tg = torch.as_tensor(tr["g"]).float()
    ty = np.asarray(tr["y"]); tc = np.asarray(tr["c"]); ta = tr["attr"]
    pva = np.asarray(va["p"]); yva = np.asarray(va["y"])
    pte = np.asarray(te["p"])
    prkc_va = rkc_attr_predict(tg, ty, tc, ta, torch.as_tensor(va["g"]).float(), va["attr"])
    prkc_te = rkc_attr_predict(tg, ty, tc, ta, torch.as_tensor(te["g"]).float(), te["attr"])
    best = (-1.0, 1.0, 0.5)
    for a in np.linspace(0.0, 1.0, 21):
        cv = a * pva + (1 - a) * prkc_va
        thr = best_threshold_macroF1(yva, cv)
        f = macro_f1(yva, (cv >= thr).astype(int))
        if f > best[0]:
            best = (f, a, thr)
    _, alpha, thr = best
    comb = alpha * pte + (1 - alpha) * prkc_te
    return comb, float(thr), float(alpha), pte, prkc_te


def index_by_pair(bundle):
    return {p: i for i, p in enumerate(bundle["test"]["pair_id"])}


def disp_claim(r):
    segs = (r.get("claim", {}) or {}).get("segments", []) or []
    t = " ".join(s.get("text", "").strip() for s in segs if s.get("text"))
    return t.strip() or (r.get("claim", {}) or {}).get("passage", "") or r.get("attribute_name", "")


def disp_evidence(r):
    parts = []
    for it in r.get("evidence_params", []) or []:
        if it.get("raw_text"):
            parts.append("[参数] " + it["raw_text"])
    for it in r.get("evidence_ocr", []) or []:
        if it.get("raw_text"):
            parts.append("[详情页] " + it["raw_text"])
    for it in r.get("evidence_vlm", []) or []:
        q = it.get("raw_quote", it.get("raw_text", ""))
        if q:
            parts.append("[图片] " + q)
    return "  ".join(parts).strip() or "（无可核验商品事实）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--nocl", required=True)
    ap.add_argument("--bert", default="")
    ap.add_argument("--llm", default="", help="llm cf 评分 jsonl（pair_id, risk_score, decision）")
    ap.add_argument("--cf", required=True, help="cf_probe.jsonl 原文")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default="")
    args = ap.parse_args()

    bf = load_bundle(args.full)
    bn = load_bundle(args.nocl)
    recs = {r["pair_id"]: r for r in (json.loads(l) for l in open(args.cf, encoding="utf-8") if l.strip())}

    comb_f, thr_f, alpha_f, pcls_f, prkc_f = claimarc_decision(bf)
    idx_f = index_by_pair(bf)
    idx_n = index_by_pair(bn)
    gf = gnorm(bf["test"]["g"]); gn = gnorm(bn["test"]["g"])

    bert = None
    if args.bert:
        bb = load_bundle(args.bert)
        bidx = {p: i for i, p in enumerate(bb["test"].get("pair_id", []))}
        bert = (bb, bidx)

    llm = {}
    if args.llm:
        for l in open(args.llm, encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            if d.get("risk_score") is not None:
                llm[d["pair_id"]] = d

    # 配对（cf{N}_mislead / cf{N}_ok）
    pair_ids = sorted({p.rsplit("_", 1)[0] for p in idx_f})
    rows = []
    sims_full, sims_nocl = [], []
    for base in pair_ids:
        pm, po = f"{base}_mislead", f"{base}_ok"
        if pm not in idx_f or po not in idx_f:
            continue
        im, io = idx_f[pm], idx_f[po]
        cos_full = float((gf[im] * gf[io]).sum())
        nm, no = idx_n[pm], idx_n[po]
        cos_nocl = float((gn[nm] * gn[no]).sum())
        sims_full.append(cos_full); sims_nocl.append(cos_nocl)

        def cla_pred(i):  # CLAIMARC RKC 决策
            return int(comb_f[i] >= thr_f)

        def bert_pred(pid):
            if not bert:
                return None
            bb, bidx = bert
            if pid not in bidx:
                return None
            p = float(np.asarray(bb["test"]["p"])[bidx[pid]])
            return int(p >= float(bb.get("thr", 0.5)))

        def llm_pred(pid):
            d = llm.get(pid)
            if not d:
                return None
            return int(d.get("decision", int(d["risk_score"] >= 0.5)))

        attr = recs[pm].get("attribute_name", "")
        row = {
            "pair": base, "attribute": attr,
            "claim": disp_claim(recs[pm]),
            "evidence_mislead": disp_evidence(recs[pm]),
            "evidence_ok": disp_evidence(recs[po]),
            "cos_full": round(cos_full, 4), "cos_nocl": round(cos_nocl, 4),
            "claimarc": {"mislead": cla_pred(im), "ok": cla_pred(io),
                          "p_cls_m": round(float(pcls_f[im]), 3), "p_cls_o": round(float(pcls_f[io]), 3),
                          "p_rkc_m": round(float(prkc_f[im]), 3), "p_rkc_o": round(float(prkc_f[io]), 3)},
            "bert": {"mislead": bert_pred(pm), "ok": bert_pred(po)},
            "llm": {"mislead": llm_pred(pm), "ok": llm_pred(po)},
        }
        # 正确性：mislead 应判 1，ok 应判 0
        def ok2(d):
            if d is None or d.get("mislead") is None or d.get("ok") is None:
                return None
            return (d["mislead"] == 1) and (d["ok"] == 0)
        row["claimarc_correct_both"] = ok2(row["claimarc"])
        row["bert_correct_both"] = ok2(row["bert"])
        row["llm_correct_both"] = ok2(row["llm"])
        rows.append(row)

    summary = {
        "thr_claimarc": round(thr_f, 3), "alpha_claimarc": round(alpha_f, 3),
        "mean_cos_full": round(float(np.mean(sims_full)), 4),
        "mean_cos_nocl": round(float(np.mean(sims_nocl)), 4),
        "n_pairs": len(rows),
        "claimarc_both_correct": int(sum(1 for r in rows if r["claimarc_correct_both"])),
        "bert_both_correct": int(sum(1 for r in rows if r["bert_correct_both"])),
        "llm_both_correct": int(sum(1 for r in rows if r["llm_correct_both"])),
        # 仅 CLAIMARC 同时判对、BERT 或 LLM 至少错一个的“黄金对”
        "claimarc_only_pairs": [r["pair"] for r in rows
                                 if r["claimarc_correct_both"]
                                 and (r["bert_correct_both"] is False or r["llm_correct_both"] is False)],
    }
    out = {"summary": summary, "rows": rows}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("CF_RESULT", json.dumps(summary, ensure_ascii=False), flush=True)
    for r in rows:
        print(f"[{r['pair']:5s} {r['attribute']:8s}] cos full={r['cos_full']:+.3f} nocl={r['cos_nocl']:+.3f} "
              f"| CLA m/o={r['claimarc']['mislead']}/{r['claimarc']['ok']} "
              f"BERT={r['bert']['mislead']}/{r['bert']['ok']} LLM={r['llm']['mislead']}/{r['llm']['ok']} "
              f"| CLA_ok={r['claimarc_correct_both']} BERT_ok={r['bert_correct_both']} LLM_ok={r['llm_correct_both']}",
              flush=True)

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r["pair"] for r in rows]
        x = np.arange(len(labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(labels)), 3.4))
        ax.bar(x - w / 2, [r["cos_nocl"] for r in rows], w, label="w/o RACL", color="#c0c4cc")
        ax.bar(x + w / 2, [r["cos_full"] for r in rows], w, label="CLAIMARC (RACL)", color="#c0392b")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("cos(claim-fixed, fact-flipped)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_title("Counterfactual pair similarity: only RACL pulls flipped pairs apart")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=200); fig.savefig(args.fig.replace(".png", ".pdf"))
        print(f"[fig] -> {args.fig}", flush=True)


if __name__ == "__main__":
    main()
