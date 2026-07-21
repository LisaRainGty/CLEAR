"""Compute Methodology_Data.md §2 hard labels and sample weights (y, c).

This is the *plan-baseline* label engine. It reads the stateful reconstruction
`_all_` dataset (which already carries, per pair, the Stage-A aligned consumer
mentions with polarity / mention_strength / explicit_fact_hit and the total
comment count) and produces:

  y_plan  : §2.1 hard label  ->  1 iff there exists an aligned comment with
            polarity == neg, else 0.
  c_plan  : §2.2-2.4 sample weight using f_sat, f_cov, f_asym (y=1) /
            f_fake (y=0), the evidence score w_i and phi_bonus.
  label_audit_plan : all §2 intermediate quantities for ablation.

It does NOT mutate the reconstruction labels. Each output record keeps the
reconstruction view (`y_recon`, `c_recon`) side by side with the plan view so a
direct §2-vs-strict ablation is possible on identical rows and identical
room-grouped splits.

No LLM calls. Pure offline aggregation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# §2.6 default parameters
GAMMA = 2.0          # explicit_fact_hit bonus
K_SAT = 3.0          # evidence saturation rate
LAMBDA = 0.3         # asymmetric pos discount
RHO = 0.4            # fake-review discount
PHI_BONUS = 1.2      # strong neg evidence bonus
S_STRONG = 1.2       # mention_strength multiplier (strong)
S_WEAK = 0.7         # mention_strength multiplier (weak)
C_FLOOR = 0.05       # weight floor


def parse_time(s: str) -> datetime | None:
    s = str(s or "").strip()
    if not s:
        return None
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def bigrams(text: str) -> set[str]:
    t = "".join(str(text or "").split())
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()


def low_diversity(spans: list[str]) -> bool:
    """Approximate §2.3(b): aligned comments highly homogeneous in bigram space."""
    spans = [s for s in spans if s]
    if len(spans) < 3:
        return False
    sims = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = bigrams(spans[i]), bigrams(spans[j])
            if not a or not b:
                continue
            sims.append(len(a & b) / len(a | b))
    if not sims:
        return False
    return (sum(sims) / len(sims)) >= 0.7


def w_i(m: dict[str, Any]) -> float:
    s = S_STRONG if str(m.get("mention_strength")) == "strong" else S_WEAK
    efh = 1.0 if m.get("explicit_fact_hit") else 0.0
    return (1.0 + GAMMA * efh) * s


def aligned_set(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for m in row.get("_aligned_consumer_mentions") or []:
        j = m.get("_judgment") or {}
        if j.get("aligned_to_claim"):
            out.append(m)
    return out


def compute(row: dict[str, Any]) -> dict[str, Any] | None:
    aligned = aligned_set(row)
    n_total = int(row.get("_consumer_mentions_total") or 0)
    n_aligned = len(aligned)
    if n_aligned == 0:
        return None  # no grounded alignment -> not in plan-baseline supervised universe

    neg = [m for m in aligned if str(m.get("polarity")) == "neg"]
    pos = [m for m in aligned if str(m.get("polarity")) == "pos"]
    s_neg = sum(w_i(m) for m in neg)
    s_pos = sum(w_i(m) for m in pos)

    y = 1 if len(neg) > 0 else 0

    f_sat = 1.0 - math.exp(-n_aligned / K_SAT)
    f_cov = n_aligned / (n_total + 1)
    c_base = f_sat * f_cov

    audit = {
        "policy": "methodology_data_section2",
        "n_aligned": n_aligned,
        "n_total": n_total,
        "n_neg_aligned": len(neg),
        "n_pos_aligned": len(pos),
        "S_neg": round(s_neg, 4),
        "S_pos": round(s_pos, 4),
        "f_sat": round(f_sat, 4),
        "f_cov": round(f_cov, 4),
        "c_base": round(c_base, 4),
        "params": {"gamma": GAMMA, "k": K_SAT, "lambda": LAMBDA, "rho": RHO, "phi_bonus": PHI_BONUS},
    }

    if y == 1:
        f_asym = s_neg / (s_neg + LAMBDA * s_pos) if (s_neg + LAMBDA * s_pos) > 0 else 0.0
        phi = PHI_BONUS if any(
            (str(m.get("mention_strength")) == "strong" or m.get("explicit_fact_hit")) for m in neg
        ) else 1.0
        c = min(1.0, c_base * f_asym * phi)
        audit.update({"f_asym": round(f_asym, 4), "phi_bonus_applied": phi})
    else:
        spans = [str(m.get("evidence_span") or "") for m in aligned]
        times = [parse_time(m.get("review_time")) for m in aligned]
        times = [t for t in times if t]
        fake_a = (n_total < 10 and len(neg) == 0 and len(pos) > 0)
        fake_b = low_diversity(spans)
        fake_c = bool(times) and ((max(times) - min(times)).days <= 3) and n_aligned >= 3
        suspected = fake_a or fake_b or fake_c
        f_fake = 1.0 - RHO * (1.0 if suspected else 0.0)
        c = c_base * f_fake
        audit.update({
            "f_fake": round(f_fake, 4),
            "suspected_fake": suspected,
            "fake_rules": {"a_lowN_allpos": fake_a, "b_low_diversity": fake_b, "c_burst_3day": fake_c},
        })

    c = max(c, C_FLOOR)
    audit["c_plan"] = round(c, 4)
    return {"y_plan": y, "c_plan": round(c, 4), "label_audit_plan": audit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_rows", default="data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_bigexpand_all_20260614.jsonl")
    ap.add_argument("--out_supervised", default="data/final/repaired_v1/dataset_planbaseline_duallabel_supervised_20260614.jsonl")
    ap.add_argument("--out_all", default="data/final/repaired_v1/dataset_planbaseline_duallabel_all_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/dataset_planbaseline_duallabel_20260614.report.json")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.all_rows).read_text(encoding="utf-8").splitlines() if ln.strip()]

    out_all: list[dict[str, Any]] = []
    supervised: list[dict[str, Any]] = []
    agree = Counter()
    plan_y = Counter()
    recon_y = Counter()

    for row in rows:
        rec = dict(row)
        rec["y_recon"] = row.get("y")
        rec["c_recon"] = row.get("c")
        res = compute(row)
        if res:
            rec.update(res)
            recon_label = row.get("y")
            agree[(res["y_plan"], recon_label)] += 1
            plan_y[res["y_plan"]] += 1
            if recon_label in (0, 1):
                recon_y[recon_label] += 1
            supervised.append(rec)
        else:
            rec["y_plan"] = None
            rec["c_plan"] = None
            rec["label_audit_plan"] = None
        out_all.append(rec)

    Path(args.out_all).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_all) + "\n", encoding="utf-8")
    Path(args.out_supervised).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in supervised) + "\n", encoding="utf-8")

    c_plan_vals = [r["c_plan"] for r in supervised]
    report = {
        "all_rows_in": len(rows),
        "plan_supervised_rows": len(supervised),
        "plan_y": dict(plan_y),
        "recon_y_on_same_rows": dict(recon_y),
        "split": dict(Counter(str(r.get("split")) for r in supervised)),
        "agreement_plan_vs_recon": {f"plan={k[0]},recon={k[1]}": v for k, v in sorted(agree.items(), key=lambda x: str(x[0]))},
        "c_plan_stats": {
            "min": round(min(c_plan_vals), 4) if c_plan_vals else None,
            "max": round(max(c_plan_vals), 4) if c_plan_vals else None,
            "mean": round(sum(c_plan_vals) / len(c_plan_vals), 4) if c_plan_vals else None,
        },
        "out_supervised": args.out_supervised,
        "out_all": args.out_all,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
