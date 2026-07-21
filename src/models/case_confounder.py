#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""§4.4.7 重设计：混淆样本(confounder)案例与几何分析（对标 Mei et al. 2024 RGCL §5/Table 8）。

两类直观翻转场景，附完整主播话术与商品证据原文，并对比逐对基线判定：
  场景 A（事实相近、话术不同 → 感知翻转）：同属性、商品事实(evidence) BGE cosine≥0.85、
       但消费者标签相反的 anchor(y=1)–confounder(y=0) 对。
  场景 B（话术相近、事实不同 → 感知翻转）：主播话术(claim) BGE cosine≥0.80、标签相反、
       但事实相似度较低的对。
对每对：要求 CLAIMARC 两端判别均正确，且 BERT 或 Qwen2.5-7B(SFT) 至少一端判错——
即“直观可见 CLAIMARC 对、强基线错”的展示样本。

几何机制（§4.4.4）：RACL 前后 anchor–confounder 余弦相似度塌缩（72 对汇总 + 配对散点图）。

产出：
  case_confounder.json        —— 几何汇总 + 场景 A/B 展示样本（完整原文 + 各模型判定）
  fig_confounder_sim.pdf/png  —— anchor–confounder 余弦相似度分布 + 配对散点
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

from models.data import load_split, resolve_bge_path
from models.baselines import evidence_text, claim_text

D = os.path.expanduser("~/claimarc/data/final")
OUT = os.path.expanduser("~/claimarc/figs")
os.makedirs(OUT, exist_ok=True)


def disp_claim(r):
    """完整主播话术（拼接所有口播片段）。"""
    segs = (r.get("claim", {}) or {}).get("segments", []) or []
    t = " / ".join(s.get("text", "").strip() for s in segs if s.get("text"))
    return t.strip() or r.get("attribute_name", "")


def disp_evidence(r):
    """带来源标签的完整商品证据原文。"""
    parts = []
    for it in r.get("evidence_params", []) or []:
        if it.get("raw_text"):
            parts.append("[参数] " + it["raw_text"].strip())
    for it in r.get("evidence_ocr", []) or []:
        if it.get("raw_text"):
            parts.append("[详情图] " + it["raw_text"].strip())
    for it in r.get("evidence_vlm", []) or []:
        if it.get("raw_quote"):
            parts.append("[视觉] " + it["raw_quote"].strip())
    args = r.get("arguments", {}) or {}
    if args.get("supporting_argument"):
        parts.append("[佐证] " + args["supporting_argument"].strip())
    if args.get("refuting_argument"):
        parts.append("[反证] " + args["refuting_argument"].strip())
    return "  ".join(parts).strip() or "（无可核验商品事实）"


def bge_embed(texts, dev):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(resolve_bge_path(), device=dev)
    return torch.tensor(m.encode(texts, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=False))


def gsim_matrix(g):
    g = F.normalize(torch.as_tensor(np.asarray(g), dtype=torch.float32), dim=-1)
    return (g @ g.T).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--full", default=os.path.join(D, "emb_clarc_v2_s0.pt"))
    ap.add_argument("--nocl", default=os.path.join(D, "emb_nocl_s0.pt"))
    ap.add_argument("--bert", default=os.path.join(D, "pred_bert_s0.pt"))
    ap.add_argument("--qwen", default=os.path.join(D, "pred_qwen7b_lora_s0.pt"))
    ap.add_argument("--ev_thr", type=float, default=0.85)
    ap.add_argument("--cl_thr", type=float, default=0.80)
    ap.add_argument("--out", default=os.path.join(D, "case_confounder.json"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    test = load_split(args.dataset)["test"]
    full = torch.load(args.full, map_location="cpu", weights_only=False)
    nocl = torch.load(args.nocl, map_location="cpu", weights_only=False)
    bert = torch.load(args.bert, map_location="cpu", weights_only=False)
    qwen = torch.load(args.qwen, map_location="cpu", weights_only=False) \
        if os.path.exists(args.qwen) else None

    def pid(r):
        return r.get("pair_id") or f"p{r.get('product_id')}__{r.get('attribute_id')}"
    test_pids = [pid(r) for r in test]
    n = len(test)

    def index(bundle):
        t = bundle["test"]
        if "pair_id" in t:
            pos = {p: i for i, p in enumerate(np.asarray(t["pair_id"]))}
            return [pos.get(p, -1) for p in test_pids]
        assert len(t["p"]) == n, "test 长度不一致，无法按位置对齐"
        return list(range(n))
    fi, ni, bi = index(full), index(nocl), index(bert)
    qi = index(qwen) if qwen else None
    assert min(fi) >= 0 and min(ni) >= 0 and min(bi) >= 0, "对齐失败"

    attr = np.array([r.get("attribute_id", "") for r in test])
    y = np.array([int(r.get("y", 0)) for r in test])

    # 客观证据 / 主播话术 的冻结 BGE 相似度（独立于学习表征）
    ev = bge_embed([evidence_text(r) for r in test], dev)
    cl = bge_embed([claim_text(r) for r in test], dev)
    esim = (ev @ ev.T).numpy()
    csim = (cl @ cl.T).numpy()

    gf = np.asarray(full["test"]["g"])[fi]
    gn = np.asarray(nocl["test"]["g"])[ni]
    simf = gsim_matrix(gf)
    simn = gsim_matrix(gn)

    pf = np.asarray(full["test"]["p"])[fi]; thrf = full.get("thr", 0.5)
    pn = np.asarray(nocl["test"]["p"])[ni]; thrn = nocl.get("thr", 0.5)
    pb = np.asarray(bert["test"]["p"])[bi]; thrb = bert.get("thr", 0.5)
    if qwen:
        pq = np.asarray(qwen["test"]["p"])[qi]; thrq = qwen.get("thr", 0.5)

    # ---------- 几何汇总（场景 A 全体 72 对）----------
    geom_pairs = []
    for i in range(n):
        if y[i] != 1:
            continue
        for j in np.where((attr == attr[i]) & (y == 0) & (esim[i] >= args.ev_thr))[0]:
            geom_pairs.append((i, j))
    sim_full = np.array([simf[i, j] for i, j in geom_pairs])
    sim_nocl = np.array([simn[i, j] for i, j in geom_pairs])
    summary = {
        "n_confounder_pairs": len(geom_pairs),
        "evidence_sim_thr": args.ev_thr,
        "mean_anchor_conf_sim_nocl": round(float(sim_nocl.mean()), 4) if geom_pairs else None,
        "mean_anchor_conf_sim_full": round(float(sim_full.mean()), 4) if geom_pairs else None,
        "sim_reduction": round(float(sim_nocl.mean() - sim_full.mean()), 4) if geom_pairs else None,
        "rel_reduction_pct": round(float(100 * (sim_nocl.mean() - sim_full.mean()) / abs(sim_nocl.mean())), 1) if geom_pairs else None,
        "frac_neg_nocl": round(float((sim_nocl < 0).mean()), 3) if geom_pairs else None,
        "frac_neg_full": round(float((sim_full < 0).mean()), 3) if geom_pairs else None,
        "frac_pushed_down": round(float((sim_full < sim_nocl).mean()), 3) if geom_pairs else None,
    }

    # ---------- 展示样本：CLAIMARC 两端全对、强基线至少一端错 ----------
    def ok(p, thr, lab):
        return int(p >= thr) == lab

    def member(i, lab):
        m = {"pair_id": test_pids[i], "attribute": test[i].get("attribute_name", attr[i]),
             "y": int(lab), "claim": disp_claim(test[i]), "evidence": disp_evidence(test[i]),
             "claimarc_p": round(float(pf[i]), 3), "claimarc_ok": ok(pf[i], thrf, lab),
             "bert_p": round(float(pb[i]), 3), "bert_ok": ok(pb[i], thrb, lab)}
        if qwen:
            m["qwen7b_p"] = round(float(pq[i]), 3); m["qwen7b_ok"] = ok(pq[i], thrq, lab)
        return m

    def build_cases(cand_pairs, sim_key):
        scored = []
        for (i, j) in cand_pairs:
            both_full = ok(pf[i], thrf, 1) and ok(pf[j], thrf, 0)
            if not both_full:
                continue
            bert_bad = (not ok(pb[i], thrb, 1)) + (not ok(pb[j], thrb, 0))
            qwen_bad = ((not ok(pq[i], thrq, 1)) + (not ok(pq[j], thrq, 0))) if qwen else 0
            if bert_bad + qwen_bad == 0:
                continue
            # 优先：基线错得越多、CLAIMARC 越自信、证据/话术原文越完整
            conf = abs(pf[i] - 0.5) + abs(pf[j] - 0.5)
            txt = min(len(disp_claim(test[i])), len(disp_claim(test[j])))
            score = 3 * (bert_bad + qwen_bad) + conf + min(txt, 40) / 40.0
            scored.append((score, i, j, bert_bad, qwen_bad))
        scored.sort(reverse=True)
        cases, seen = [], set()
        for score, i, j, bb, qb in scored:
            key = attr[i]
            if key in seen:
                continue
            seen.add(key)
            cases.append({
                "attribute": test[i].get("attribute_name", attr[i]),
                "evidence_sim": round(float(esim[i, j]), 3),
                "claim_sim": round(float(csim[i, j]), 3),
                "anchor_conf_sim_nocl": round(float(simn[i, j]), 3),
                "anchor_conf_sim_full": round(float(simf[i, j]), 3),
                "bert_wrong_members": int(bb), "qwen_wrong_members": int(qb),
                "anchor": member(i, 1), "confounder": member(j, 0),
            })
            if len(cases) >= 6:
                break
        return cases

    # 场景 A：事实相近、话术不同
    poolA = geom_pairs
    casesA = build_cases(poolA, "evidence")
    # 场景 B：话术相近、事实不同（标签相反、claim 高相似、evidence 较低）
    poolB = []
    for i in range(n):
        if y[i] != 1:
            continue
        for j in np.where((y == 0) & (csim[i] >= args.cl_thr) & (esim[i] < args.ev_thr))[0]:
            poolB.append((i, j))
    casesB = build_cases(poolB, "claim")

    out = {"summary": summary, "thr": {"full": thrf, "nocl": thrn, "bert": thrb,
                                       **({"qwen7b": thrq} if qwen else {})},
           "scenario_A_facts_similar_claims_differ": casesA,
           "scenario_B_claims_similar_facts_differ": casesB}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    print(f"[scenarioA] {len(casesA)} cases | [scenarioB] {len(poolB)} cand -> {len(casesB)} cases")
    for tag, cs in (("A", casesA), ("B", casesB)):
        for c in cs[:3]:
            print(f"  [{tag}] {c['attribute']} | ev_sim={c['evidence_sim']} cl_sim={c['claim_sim']}"
                  f" | bert_wrong={c['bert_wrong_members']} qwen_wrong={c['qwen_wrong_members']}"
                  f" | sim {c['anchor_conf_sim_nocl']}->{c['anchor_conf_sim_full']}")
    print("[save] ->", args.out)

    # ---------- 图：几何塌缩 ----------
    if geom_pairs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams.update({"font.family": "serif", "font.size": 11,
                         "savefig.bbox": "tight", "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linestyle": "--"})
        C_NEG, C_POS = "#d1495b", "#2e86ab"
        fig, (axh, axs) = plt.subplots(1, 2, figsize=(10.6, 4.3))
        bins = np.linspace(-1, 1, 33)
        axh.hist(sim_nocl, bins=bins, alpha=0.6, color=C_NEG,
                 label=f"w/o RACL (mean {sim_nocl.mean():.2f})", density=True)
        axh.hist(sim_full, bins=bins, alpha=0.6, color=C_POS,
                 label=f"CLAIMARC (mean {sim_full.mean():.2f})", density=True)
        axh.axvline(sim_nocl.mean(), color=C_NEG, ls="--", lw=1.5)
        axh.axvline(sim_full.mean(), color=C_POS, ls="--", lw=1.5)
        axh.axvline(0, color="grey", lw=1)
        axh.set_xlabel("anchor–confounder cosine similarity")
        axh.set_ylabel("density"); axh.set_title("(a) Similarity distribution")
        axh.legend(framealpha=0.9, fontsize=9)
        axs.scatter(sim_nocl, sim_full, s=26, alpha=0.7, color=C_POS,
                    edgecolors="white", linewidths=0.5, zorder=3)
        lo, hi = -0.8, 1.0
        axs.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1.2, zorder=2, label="no change")
        axs.axhline(0, color="grey", lw=0.8, zorder=1)
        axs.fill_between([lo, hi], [lo, hi], lo, color=C_POS, alpha=0.06, zorder=0)
        axs.set_xlim(lo, hi); axs.set_ylim(lo, hi)
        axs.set_xlabel("similarity w/o RACL"); axs.set_ylabel("similarity with CLAIMARC")
        below = float((sim_full < sim_nocl).mean()) * 100
        axs.set_title(f"(b) {below:.0f}% of pairs pushed apart")
        axs.legend(framealpha=0.9, fontsize=9, loc="upper left")
        fig.suptitle(f"RACL separates same-attribute opposite-label confounders "
                     f"({len(geom_pairs)} pairs)", y=1.0, fontsize=12)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"fig_confounder_sim.{ext}"))
        print("SAVED fig_confounder_sim")


if __name__ == "__main__":
    main()
