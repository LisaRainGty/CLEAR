"""Stage B4/B5 — 评论×claim 对齐判定 + pair 级聚合记录。

B4：对每个 pair，把其属性下的评论（来自 Stage A resolved_aspects，按 product+attribute 取）
    与 claim passage 一起送 LLM，逐条判 y_supportability∈{0,1}；passage 空时强制 0。
B5：聚合 pair_records.jsonl（passage+segments + 透传评论 polarity/strength/explicit_fact_hit
    + N_total/N_aligned/N_aligned_neg）。

用法：python -m stage_b.b4_b5_align [--category ...] [--limit N]
     python -m stage_b.b4_b5_align --pair_skeleton data/processed/stageB_product_v2/pair_skeleton_product_v2_rerun.jsonl --resolved data/processed/stageB_product_v2/resolved_aspects_product_v2.jsonl --out data/processed/stageB_product_v2/pair_records_product_v2.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import config
from common import llm
from common import product_index as pidx
from common.io_utils import read_jsonl, write_jsonl

B4_PROMPT = """角色：电商直播虚假宣传审查员。
属性：{attr}（同义词：{aliases}）
主播口播（按时间拼接）：
{passage}

任务：对下面每条评论判断它是否针对主播口播中关于该属性的"具体表述"做出了直接回应？

判 y_supportability=1：评论指向 claim 中的具体说法（同向肯定/反向否定均算）。
  例：主播"每盒125毫升" + 评论"125ml这个量也太少了" → 1
      主播"口感丝滑"     + 评论"说丝滑没觉得"        → 1
判 y_supportability=0：评论仅泛泛谈论该属性、未指向具体说法（如"还行""还可以"）；
  或与主播口播无可比较点；当主播口播为空时强制为 0。

输入评论（编号 1..K）：
{reviews}

严格输出 JSON 数组，每条：{{"cid": <编号int>, "y_supportability": 0或1}}
只输出 JSON 数组。"""

NO_CLAIM_MARK = "【本商品无主播相关口播】"


def _align_pair(pair: dict, comments: list[dict]) -> list[int]:
    """返回与 comments 等长的 y_supportability 列表。"""
    if not comments:
        return []
    if not pair.get("has_claim_srt") or not pair.get("passage", "").strip():
        return [0] * len(comments)
    review_lines = "\n".join(
        f"{i + 1}. {c['text'][:120]}" for i, c in enumerate(comments)
    )
    prompt = B4_PROMPT.format(
        attr=pair["attribute_canonical"],
        aliases="、".join(pair.get("aliases", [])[:5]),
        passage=pair["passage"][:2000],
        reviews=review_lines,
    )
    try:
        arr = llm.chat_json(prompt, namespace="b4", max_tokens=1024)
    except Exception:
        return [0] * len(comments)
    ys = [0] * len(comments)
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            cid = item.get("cid")
            try:
                k = int(cid) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(comments):
                ys[k] = 1 if int(item.get("y_supportability", 0)) == 1 else 0
    return ys


def _comments_by_pair(resolved_path: str) -> dict[tuple[str, str], list[dict]]:
    """从 resolved_aspects 聚合每 (product, attribute) 的评论级证据（去重 review_id）。"""
    by_pair: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(resolved_path):
        key = (r["product_id"], r["attribute_id"])
        rid = r["review_id"]
        # 同一评论可能对同一属性多次提及，保留信息最强的一条
        prev = by_pair[key].get(rid)
        cur = {
            "comment_id": rid,
            "text": r.get("review_text") or r.get("evidence_span", "") or "",
            "polarity": r.get("polarity") or r.get("review_polarity", "neu"),
            "review_polarity": r.get("review_polarity", "neu"),
            "mention_strength": r.get("mention_strength", "weak"),
            "explicit_fact_hit": bool(r.get("explicit_fact_hit", False)),
            "evidence_span": r.get("evidence_span", ""),
            "review_time": r.get("review_time", ""),
        }
        if prev is None or (not prev["explicit_fact_hit"] and cur["explicit_fact_hit"]) \
                or (prev["mention_strength"] != "strong" and cur["mention_strength"] == "strong"):
            by_pair[key][rid] = cur
    return {k: list(v.values()) for k, v in by_pair.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pair_skeleton", default=str(config.STAGE_B / "pair_skeleton.jsonl"))
    ap.add_argument("--resolved", default=str(config.STAGE_A / "resolved_aspects.jsonl"))
    ap.add_argument("--out", default=str(config.STAGE_B / "pair_records.jsonl"))
    args = ap.parse_args()

    bundles = pidx.build_bundles()
    pairs = list(read_jsonl(args.pair_skeleton))
    if args.category:
        pairs = [p for p in pairs if bundles.get(p["product_id"]) and bundles[p["product_id"]].category == args.category]
    if args.limit:
        pairs = pairs[:args.limit]
    comments_map = _comments_by_pair(args.resolved)
    print(f"[B4] pairs to align: {len(pairs)}")

    def _job(pair):
        comments = comments_map.get((pair["product_id"], pair["attribute_id"]), [])
        ys = _align_pair(pair, comments)
        for c, y in zip(comments, ys):
            c["y_supportability"] = y
        n_total = len(comments)
        n_aligned = sum(ys)
        n_aligned_neg = sum(1 for c, y in zip(comments, ys) if y == 1 and c["polarity"] == "neg")
        return {
            "pair_id": pair["pair_id"],
            "product_id": pair["product_id"],
            "category": bundles[pair["product_id"]].category if pair["product_id"] in bundles else "",
            "attribute_id": pair["attribute_id"],
            "attribute_canonical": pair["attribute_canonical"],
            "claim": {
                "has_claim_srt": pair["has_claim_srt"],
                "passage": pair["passage"],
                "segments": pair["segments"],
            },
            "reviews": comments,
            "stats": {
                "N_total": n_total,
                "N_aligned": n_aligned,
                "N_aligned_neg": n_aligned_neg,
            },
        }

    records = llm.run_many(pairs, _job, desc="B4/B5")
    records = [r for r in records if isinstance(r, dict) and "__error__" not in r]
    write_jsonl(args.out, records)
    tot_aligned = sum(r["stats"]["N_aligned"] for r in records)
    print(f"[B4/B5] pair_records={len(records)} total aligned reviews={tot_aligned} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
