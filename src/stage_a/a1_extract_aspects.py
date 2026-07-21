"""Stage A1 — CAS 约束下的评论开放抽取。

对每条评论：把其商品品类的 CAS（canonical_name + aliases）作为优先映射表传给 LLM，
要求能映射到 CAS 就用其 attribute_id，否则 FREE::<名词短语>。同时输出
polarity / evidence_span / type / explicit_fact_hit / mention_strength。
过滤 type=personal。

产出 data/processed/stageA/raw_aspects.jsonl（追加式，支持断点续跑：按 product 缓存）。

用法：
  python -m stage_a.a1_extract_aspects [--category food_and_beverages] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re

import config
from common import llm
from common import product_index as pidx
from common.io_utils import read_comment_xls, read_json, write_jsonl

A1_PROMPT = """你是电商评论属性抽取员。下面给出某商品品类的【标准属性表】和【一条评论】。

请抽取评论中提及的每一个商品/服务相关的 aspect，并按属性表映射。

【标准属性表】（attribute_id | 标准名 | 别名）：
{cas}

【评论】：
{review}

== attribute_id 取值规则（关键）==
1) 优先映射：能对上标准属性表里某条 → 直接用它的 attribute_id。
2) 实在没有对应 → 用 FREE::<客观属性名词短语>。FREE 名必须是该商品的一个【客观属性/部件/服务维度】的中性名词短语，例如：保质期、净含量、开关、电池续航、物流速度、客服服务。
   FREE 命名硬约束：
   - 禁止评价性短语：性价比、品质、做工、质量好坏、外观优势、卖家态度好坏 等评价词不能当属性名——它们要归到其客观属性（如 价格 / 质量 / 客服）并用 polarity 表达好坏。
   - 禁止把同一个部件/功能按不同角度拆成多个属性：如"开关设计/开关响应性/开关功能异常"统一为一个"开关"；"杯子旋转/搅拌动力/搅拌性能"统一为一个"搅拌性能"。一个客观属性只用一个名词。
   - 禁止主观感受词、口语整句、营销话术；只用简短客观名词短语（2-6字为宜）。
3) 同一条评论里指向同一属性的多次提及只输出一条。

对评论中提及的每个 aspect 输出一条 JSON，组成数组：
[
 {{
   "attribute_id": "<标准属性表的 attribute_id 或 FREE::<客观属性名词短语>>",
   "polarity": "pos|neg|neu",
   "evidence_span": "<评论原文片段，≤30字>",
   "type": "attribute|service|personal",
   "explicit_fact_hit": true/false,   // 是否含"说是/宣传的/写的/标的/承诺"等对宣传的印证或反驳信号
   "mention_strength": "strong|weak"
 }}
]
type 判定：
- attribute=商品本身的客观属性（成分/规格/产地/口感/外观/材质…），可带好坏评价但必须指向某个具体客观属性；
- service=物流/包装/客服/售后等服务维度；
- personal=纯个人偏好、消费情境、或【没有指向任何具体客观属性的笼统评价/情绪/意愿】，例如："我女儿爱喝""买来送人""性价比高""很满意""会回购""值得推荐""总体不错""划算"。
  这类只表达整体好恶/满意度/回购意愿/购买动机/推荐度而不落到具体属性的，一律判 personal。
personal 一律不要输出。
注意：若评论说"甜度刚好""包装好看""保质期短"——这些落到了具体客观属性，应判 attribute 并用 polarity 表达好坏，不要判 personal。
若评论无任何可抽取 aspect，输出 []。只输出 JSON 数组，不要解释。"""

_FREE_PREFIX_RE = re.compile(r"^FREE(?:\s*::|\s*[：:＿_]\s*)(.+)$", re.I)


def _normalize_attribute_id(value: str) -> str:
    """Normalize common LLM variants such as FREE_保质期 into FREE::保质期."""
    aid = str(value or "").strip()
    m = _FREE_PREFIX_RE.match(aid)
    if m:
        phrase = m.group(1).strip(" ：:_")
        return f"FREE::{phrase}" if phrase else "FREE::"
    return aid


def _cas_block(cas: dict) -> str:
    lines = []
    for a in cas.get("attributes", []):
        al = "、".join(a.get("aliases", [])[:6])
        lines.append(f"- {a['attribute_id']} | {a['canonical_name']} | {al}")
    return "\n".join(lines)


def _extract_one(comment: dict, cas_block: str, product_id: str, category: str) -> list[dict]:
    prompt = A1_PROMPT.format(cas=cas_block, review=comment["text"][:1000])
    try:
        arr = llm.chat_json(prompt, namespace="a1", max_tokens=1024)
    except Exception as e:  # noqa: BLE001
        return [{"__error__": repr(e)[:200], "comment_id": comment["comment_id"]}]
    if not isinstance(arr, list):
        arr = [arr] if isinstance(arr, dict) else []
    out = []
    for m in arr:
        if not isinstance(m, dict) or "attribute_id" not in m:
            continue
        if m.get("type") == "personal":
            continue
        aid = _normalize_attribute_id(m["attribute_id"])
        if aid == "FREE::":
            continue
        out.append({
            "review_id": comment["comment_id"],
            "product_id": product_id,
            "category": category,
            "attribute_id": aid,
            # polarity is attribute-level when the LLM can infer it; review_polarity is kept separately.
            "polarity": m.get("polarity") or comment.get("polarity", "neu"),
            "review_polarity": comment.get("polarity", "neu"),
            "review_text": str(comment.get("text", ""))[:300],
            "evidence_span": str(m.get("evidence_span", ""))[:30],
            "type": m.get("type", "attribute"),
            "explicit_fact_hit": bool(m.get("explicit_fact_hit", False)),
            "mention_strength": m.get("mention_strength", "weak"),
            "review_time": comment.get("time", ""),
        })
    return out


def run_category(category: str, bundles, limit: int | None = None) -> list[dict]:
    cas = read_json(config.STAGE_A / f"CAS_{category}.json", default={"attributes": []})
    cas_block = _cas_block(cas)
    if not cas_block:
        print(f"[A1:{category}] 无 CAS，跳过。请先跑 A0。")
        return []

    # 收集 (comment, product_id)
    jobs = []
    seen_files = set()
    for b in bundles:
        comments = []
        for cf in b.comment_files:
            p = pidx.resolve(cf)
            if str(p) in seen_files:
                continue
            seen_files.add(str(p))
            try:
                comments.extend(read_comment_xls(p))
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] 读评论失败 {p}: {e!r}")
        for c in comments:
            jobs.append((c, b.product_id))
    if limit:
        jobs = jobs[:limit]
    print(f"[A1:{category}] comments to extract: {len(jobs)} (products={len(bundles)})")

    results = llm.run_many(
        jobs, lambda j: _extract_one(j[0], cas_block, j[1], category),
        desc=f"A1:{category}",
    )
    rows = []
    errors = 0
    for r in results:
        if isinstance(r, dict) and "__error__" in r:
            errors += 1
            continue
        for item in r:
            if "__error__" in item:
                errors += 1
            else:
                rows.append(item)
    print(f"[A1:{category}] aspects={len(rows)} errors={errors}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None, help="每品类最多处理多少评论（调试）")
    args = ap.parse_args()

    by_cat = pidx.bundles_by_category()
    cats = [args.category] if args.category else sorted(by_cat)
    all_rows = []
    for cat in cats:
        rows = run_category(cat, by_cat.get(cat, []), limit=args.limit)
        # 每品类单独落盘，便于断点
        write_jsonl(config.STAGE_A / f"raw_aspects_{cat}.jsonl", rows)
        all_rows.extend(rows)
    # 汇总
    write_jsonl(config.STAGE_A / "raw_aspects.jsonl", all_rows)
    print(f"[A1] total aspects: {len(all_rows)} -> raw_aspects.jsonl")


if __name__ == "__main__":
    main()
