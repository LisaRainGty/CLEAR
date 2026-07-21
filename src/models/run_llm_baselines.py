"""§4.2(C) LLM 基线：零样本 / 五样本思维链，固定划分 + 同指标（与 CLAIMARC 对齐）。

与 llm_risk_baseline.py（CV 协议）不同，本脚本严格使用 §4.1 的 room 分组 train/val/test：
  - 仅对 val + test 调用 LLM（val 调阈值，test 报指标），省成本；
  - 指标与 CLAIMARC 完全一致：Macro-F1(主)、pos-F1、可靠性加权 F1、AUPRC、AUROC、ECE；
  - 不使用消费者评论 / 弱标签 / 可靠性权重，只给 (属性, 主播话术, 商品事实) —— 与基线设定一致。

用法（远端，已 source env.sh 且设好 MATPOOL_API_KEY）：
  python -m models.run_llm_baselines --model Qwen-Flash --mode zero --tag qwen_flash_zero
  python -m models.run_llm_baselines --model Qwen-Flash --mode fewshot --shots 5 --tag qwen_flash_fs5
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from common.llm import chat_json, run_many
from models.data import load_split
from models.train import macro_f1, best_threshold_macroF1, ece


def trim(t, n):
    t = (t or "").strip()
    return t if len(t) <= n else t[:n] + "..."


def claim_text(r):
    c = r.get("claim", {}) or {}
    p = (c.get("passage") or "").strip()
    if p:
        return p
    return "\n".join((s.get("text", "") or "").strip()
                     for s in (c.get("segments", []) or []) if s.get("text"))


def evidence_text(r):
    parts = []
    for label, key, field in (("参数", "evidence_params", "raw_text"),
                              ("详情图OCR", "evidence_ocr", "raw_text"),
                              ("主图/详情图视觉", "evidence_vlm", "raw_quote")):
        for it in r.get(key, []) or []:
            t = trim(str(it.get(field, "") or ""), 300)
            if t:
                parts.append(f"[{label}] {t}")
    return "\n".join(parts)


SYSTEM = "你是严谨的中文直播电商宣传风险核验助手，只输出 JSON。"
SCHEMA = ("""请输出严格 JSON（不要 Markdown）：
{"analysis": "一句话对照分析", "decision": 0或1, "risk_score": 0到1之间小数, "confidence": 0到1之间小数}""")
TASK = ("""判断该「商品-属性」pair 是否存在“主播宣传与可核验商品事实不一致、关键宣传无证据覆盖、"""
        """或消费者据此可能感到被误导”的用户感知虚假宣传风险。只依据给出的主播话术与商品事实证据，"""
        """不得使用消费者评论或外部知识。risk_score 越高风险越高；decision=1 表示有风险。""")


def render_case(r):
    claim = trim(claim_text(r), 1200) or "(无明确主播话术)"
    ev = trim(evidence_text(r), 2200) or "(无商品参数/OCR/视觉证据)"
    return (f"商品类目：{r.get('category','')}\n属性：{r.get('attribute_name','')}\n\n"
            f"主播话术：\n{claim}\n\n商品事实证据：\n{ev}")


def exemplar_analysis(r):
    y = int(r.get("y", 0))
    has_ev = bool(evidence_text(r).strip())
    if y == 1:
        return ("主播话术对该属性作出强化或承诺，但商品事实证据"
                + ("未能覆盖/与之存在落差" if has_ev else "缺失，关键宣传无法核验")
                + "，消费者易形成被误导感知。")
    return ("主播话术与可核验商品事实"
            + ("基本一致" if has_ev else "无明显冲突，且属一般性表述")
            + "，未见诱发感知落差的依据。")


def build_fewshot(train, shots, seed):
    rng = random.Random(seed)
    pos = [r for r in train if int(r.get("y", 0)) == 1 and float(r.get("c", 0)) >= 0.5]
    neg = [r for r in train if int(r.get("y", 0)) == 0 and float(r.get("c", 0)) >= 0.5]
    rng.shuffle(pos); rng.shuffle(neg)
    npos = max(1, shots // 2)
    pick = pos[:npos] + neg[:shots - npos]
    rng.shuffle(pick)
    blocks = []
    for r in pick:
        out = {"analysis": exemplar_analysis(r), "decision": int(r.get("y", 0)),
               "risk_score": 0.85 if int(r.get("y", 0)) == 1 else 0.15, "confidence": 0.8}
        blocks.append(f"### 示例\n{render_case(r)}\n输出：{json.dumps(out, ensure_ascii=False)}")
    return "\n\n".join(blocks)


def make_prompt(r, fewshot_block):
    head = TASK + "\n\n"
    if fewshot_block:
        head += ("下面是若干已判定示例（先对照分析再给二值判定），请据此判定最后一个待判 pair：\n\n"
                 + fewshot_block + "\n\n### 待判定\n")
    else:
        head += "### 待判定\n"
    return head + render_case(r) + "\n\n" + SCHEMA


def clamp01(x, d=0.5):
    try:
        v = float(x)
        return min(1.0, max(0.0, v)) if v == v else d
    except Exception:
        return d


def score_split(recs, model, fewshot_block, namespace, concurrency, max_tokens):
    def fn(r):
        try:
            obj = chat_json(make_prompt(r, fewshot_block), system=SYSTEM, model=model,
                            temperature=0.0, namespace=namespace, max_tokens=max_tokens)
            rs = clamp01(obj.get("risk_score"))
            dec = obj.get("decision")
            dec = int(dec) if dec in (0, 1, "0", "1") else (1 if rs >= 0.5 else 0)
            return {"risk_score": rs, "decision": dec}
        except Exception as e:  # noqa: BLE001
            return {"risk_score": None, "decision": None, "__error__": repr(e)[:200]}
    res = run_many(recs, fn, concurrency=concurrency, desc=f"{model}:{namespace}")
    return res


def metrics_block(y, p, c, thr):
    pred = (p >= thr).astype(int)
    return {
        "thr": round(float(thr), 3),
        "macro_f1": round(macro_f1(y, pred), 4),
        "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
        "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
        "auprc": round(average_precision_score(y, p), 4) if len(set(y)) > 1 else None,
        "auroc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
        "ece": round(ece(y, p), 4),
        "n_test": int(len(y)), "pos_test": int(y.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["zero", "fewshot"], default="zero")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=320)
    ap.add_argument("--eval_out", default="")
    args = ap.parse_args()

    sp = load_split(args.dataset)
    val, test, train = sp["val"], sp["test"], sp["train"]
    ns = f"llmbase_{args.tag}"
    fewshot_block = build_fewshot(train, args.shots, args.seed) if args.mode == "fewshot" else ""

    rv = score_split(val, args.model, fewshot_block, ns + "_val", args.concurrency, args.max_tokens)
    rt = score_split(test, args.model, fewshot_block, ns + "_test", args.concurrency, args.max_tokens)

    def arr(recs, res, key):
        return np.array([(x.get(key) if x.get(key) is not None else (0.5 if key == "risk_score" else 0))
                         for x in res], dtype=float)
    pv = arr(val, rv, "risk_score"); yv = np.array([int(r["y"]) for r in val], float)
    pt = arr(test, rt, "risk_score"); yt = np.array([int(r["y"]) for r in test], float)
    ct = np.array([float(r.get("c", 0.05)) for r in test], float)
    n_err = sum(1 for x in rt if x.get("__error__"))

    thr = best_threshold_macroF1(yv, pv)
    res = {"tag": args.tag, "model": args.model, "mode": args.mode, "shots": args.shots,
           "n_err_test": int(n_err), **metrics_block(yt, pt, ct, thr),
           # 同时报固定 0.5 阈值下的 decision 指标，便于核对模型自带判定
           "macro_f1_dec05": round(macro_f1(yt, arr(test, rt, "decision").astype(int)), 4)}
    print("RESULT_LLM", json.dumps(res, ensure_ascii=False), flush=True)
    if args.eval_out:
        Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.eval_out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
