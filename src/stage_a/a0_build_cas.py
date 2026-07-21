"""Stage A0 — 品类 CAS（Category Attribute Schema）构建。

输入：每品类下所有商品的 `产品参数` 字典。
步骤：收集 param key → 归一化 → BGE 聚类 → LLM 裁决 canonical_name+aliases
      → 扫 value 标 value_type → CAS_<cat>.json。

用法：
  python -m stage_a.a0_build_cas [--category food_and_beverages] [--no-llm]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import config
from common import embedding, llm
from common import product_index as pidx
from common.io_utils import normalize, write_json

A0_PROMPT = """你是电商商品属性标准化专家。下面是同一品类「{category}」中一组**含义相近**的商品参数字段名（已按语义聚类）。

请判断它们是否指向**同一个商品属性**，并输出标准化结果。

字段名列表：
{keys}

输出严格 JSON：
{{
  "same_attribute": true/false,            // 这组字段是否确为同一属性
  "canonical_name": "<最规范的中文属性名>", // same_attribute=true 时给出；否则给出最主要的那个
  "aliases": ["<同义字段名/别称>", ...]      // 去重后的别名（含原始字段名）
}}
只输出 JSON，不要解释。"""


def _value_type(values: list[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() != "nan"]
    if not vals:
        return "text"
    # numeric:<unit>
    num_units = []
    for v in vals:
        m = re.match(r"^\s*[\d.]+\s*([a-zA-Z\u4e00-\u9fa5%/]+)?\s*$", v)
        if m:
            num_units.append((m.group(1) or "").strip())
        else:
            num_units = None
            break
    if num_units is not None and len(num_units) >= max(2, int(0.7 * len(vals))):
        unit = ""
        units = [u for u in num_units if u]
        if units:
            unit = max(set(units), key=units.count)
        return f"numeric:{unit}" if unit else "numeric"
    # boolean
    uniq = set(vals)
    if uniq <= {"是", "否", "有", "无", "true", "false", "支持", "不支持"}:
        return "boolean"
    # enum（取值少且重复高）
    if len(uniq) <= max(2, int(0.3 * len(vals))) and len(uniq) <= 12:
        return "enum:[" + ",".join(sorted(uniq)[:12]) + "]"
    return "text"


def build_category(category: str, bundles, use_llm: bool = True) -> dict:
    # 收集 key -> values
    key_values: dict[str, list[str]] = defaultdict(list)
    for b in bundles:
        for k, v in (b.params or {}).items():
            key_values[str(k).strip()].append(v)
    keys = [k for k in key_values if k]
    if not keys:
        return {"category": category, "attributes": []}

    print(f"[A0:{category}] unique param keys: {len(keys)}")

    # 聚类（保守阈值）
    if use_llm and len(keys) > 1:
        try:
            labels = embedding.cluster(keys, config.A0_CLUSTER_DISTANCE)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 聚类失败({e!r})，退化为每 key 独立。")
            labels = list(range(len(keys)))
    else:
        labels = list(range(len(keys)))

    clusters: dict[int, list[str]] = defaultdict(list)
    for k, lab in zip(keys, labels):
        clusters[lab].append(k)

    prefix = config.category_prefix(category)
    attributes = []
    seen_ids = set()
    cluster_list = list(clusters.values())

    def _judge(cl_keys):
        if not (use_llm and len(cl_keys) > 1):
            return None
        try:
            return llm.chat_json(
                A0_PROMPT.format(category=category, keys="\n".join(f"- {k}" for k in cl_keys)),
                namespace="a0", max_tokens=512,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] LLM 裁决失败({e!r})，用启发式。")
            return None

    judgments = llm.run_many(cluster_list, _judge, desc=f"A0:{category}") \
        if use_llm else [None] * len(cluster_list)

    for cl_keys, res in zip(cluster_list, judgments):
        canonical = max(cl_keys, key=len)
        aliases = list(dict.fromkeys(cl_keys))
        if isinstance(res, dict):
            canonical = (res.get("canonical_name") or canonical).strip()
            aliases = list(dict.fromkeys(cl_keys + [a for a in res.get("aliases", []) if a]))

        # value_type：合并该 cluster 所有 key 的 value
        all_vals = []
        for k in cl_keys:
            all_vals.extend(key_values[k])
        vtype = _value_type(all_vals)

        attr_id = _make_attr_id(prefix, canonical, seen_ids)
        attributes.append({
            "attribute_id": attr_id,
            "canonical_name": canonical,
            "aliases": aliases,
            "value_type": vtype,
            "source_keys": cl_keys,
            "source": "param",
            "category": category,
        })
    print(f"[A0:{category}] -> {len(attributes)} CAS attributes")
    return {"category": category, "attributes": attributes}


_NON_WORD = re.compile(r"[^0-9A-Za-z\u4e00-\u9fa5]+")


def _make_attr_id(prefix: str, name: str, seen: set) -> str:
    base = _NON_WORD.sub("_", normalize(name)).strip("_").upper()[:24] or "ATTR"
    attr_id = f"{prefix}_{base}"
    i = 2
    while attr_id in seen:
        attr_id = f"{prefix}_{base}_{i}"
        i += 1
    seen.add(attr_id)
    return attr_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None, help="只处理指定一级品类")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM/嵌入，纯启发式（调试用）")
    args = ap.parse_args()

    by_cat = pidx.bundles_by_category()
    cats = [args.category] if args.category else sorted(by_cat)
    for cat in cats:
        bundles = by_cat.get(cat, [])
        cas = build_category(cat, bundles, use_llm=not args.no_llm)
        out = config.STAGE_A / f"CAS_{cat}.json"
        write_json(out, cas)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
