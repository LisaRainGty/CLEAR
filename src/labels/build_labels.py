"""§2 弱监督标签与样本权重。

输入 Stage B 的 pair_records.jsonl（含 per-review polarity/strength/explicit_fact_hit/
y_supportability + N_total/N_aligned/N_aligned_neg）。

硬标签：aligned 评论中存在 neg → y=1，否则 y=0。
样本权重 c：见 Methodology_Data §2。输出全部 label_audit 中间量。
产出 data/processed/labels.jsonl

用法：python -m labels.build_labels
     python -m labels.build_labels --pair_records data/processed/stageB_product_v2/pair_records_product_v2.jsonl --out data/processed/stageB_product_v2/labels_product_v2.jsonl
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime

import config
from common.io_utils import bigram_jaccard, read_jsonl, write_jsonl


def _parse_time(s: str):
    s = (s or "").strip()
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _review_evidence_score(c: dict) -> float:
    s = config.STRENGTH_MULT.get(c.get("mention_strength", "weak"), 0.7)
    gamma = config.GAMMA if c.get("explicit_fact_hit") else 0.0
    return (1.0 + gamma) * s


def _suspected_fake(aligned: list[dict], n_total: int) -> bool:
    if not aligned:
        return False
    # (a) N_total<10 且全部 pos
    pols = [c["polarity"] for c in aligned]
    if n_total < config.FAKE_LOWN and all(p == "pos" for p in pols):
        return True
    # (b) aligned 评论 bigram Jaccard 多样性低（措辞高度同质）
    texts = [c.get("text") or c.get("evidence_span") or "" for c in aligned]
    texts = [t for t in texts if t]
    if len(texts) >= 3:
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(bigram_jaccard(texts[i], texts[j]))
        if sims:
            mean_sim = sum(sims) / len(sims)
            diversity = 1.0 - mean_sim
            if diversity < config.FAKE_JACCARD_DIV:
                return True
    # (c) aligned 评论集中在 <=3 天
    times = [_parse_time(c.get("review_time", "")) for c in aligned]
    times = [t for t in times if t]
    if len(times) >= 3:
        span_days = (max(times) - min(times)).days
        if span_days <= config.FAKE_WINDOW_DAYS:
            return True
    return False


def compute_label(rec: dict) -> dict:
    reviews = rec.get("reviews", [])
    aligned = [c for c in reviews if int(c.get("y_supportability", 0)) == 1]
    n_total = rec.get("stats", {}).get("N_total", len(reviews))
    n_aligned = len(aligned)

    aligned_neg = [c for c in aligned if c["polarity"] == "neg"]
    aligned_pos = [c for c in aligned if c["polarity"] == "pos"]
    y = 1 if aligned_neg else 0

    S_neg = sum(_review_evidence_score(c) for c in aligned_neg)
    S_pos = sum(_review_evidence_score(c) for c in aligned_pos)

    # 因子
    f_sat = 1.0 - math.exp(-n_aligned / config.K_SAT)
    f_cov = (n_aligned / (n_total + 1.0)) ** config.BETA_COV
    c_base = f_sat * f_cov

    audit = {
        "n_aligned": n_aligned, "n_total": n_total,
        "n_neg_aligned": len(aligned_neg), "n_pos_aligned": len(aligned_pos),
        "S_neg": round(S_neg, 4), "S_pos": round(S_pos, 4),
        "f_sat": round(f_sat, 4), "f_cov": round(f_cov, 4),
        "c_base": round(c_base, 4),
    }

    if y == 1:
        denom = S_neg + config.LAMBDA_POS * S_pos
        f_asym = S_neg / denom if denom > 0 else 1.0
        has_strong_neg = any(
            c.get("mention_strength") == "strong" or c.get("explicit_fact_hit")
            for c in aligned_neg
        )
        phi = config.PHI_BONUS if has_strong_neg else 1.0
        c = min(1.0, c_base * f_asym * phi)
        audit.update({"f_asym": round(f_asym, 4), "phi_bonus": phi})
    else:
        susp = _suspected_fake(aligned, n_total)
        f_fake = 1.0 - config.RHO_FAKE * (1 if susp else 0)
        c = c_base * f_fake
        audit.update({"f_fake": round(f_fake, 4), "suspected_fake": susp})

    c = max(c, config.C_FLOOR)
    return {"y": y, "c": round(c, 4), "label_audit": audit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_records", default=str(config.STAGE_B / "pair_records.jsonl"))
    ap.add_argument("--out", default=str(config.LABELS_PATH))
    args = ap.parse_args()

    records = list(read_jsonl(args.pair_records))
    rows = []
    for rec in records:
        lab = compute_label(rec)
        rows.append({
            "pair_id": rec["pair_id"],
            "product_id": rec["product_id"],
            "attribute_id": rec["attribute_id"],
            **lab,
        })
    write_jsonl(args.out, rows)
    pos = sum(1 for r in rows if r["y"] == 1)
    cs = [r["c"] for r in rows]
    print(f"[labels] pairs={len(rows)} y=1: {pos} ({pos / max(1, len(rows)):.1%}) "
          f"| c mean={sum(cs) / max(1, len(cs)):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
