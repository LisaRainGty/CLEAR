"""Stage A2 — 自由生成属性聚合 → CAS+。

只处理 attribute_id 以 FREE:: 开头的条目，按品类独立分桶：
  1) 字符串 Jaccard 粗去重；
  2) BGE 聚类；
  3) 每个 cluster 交 LLM 对照该品类现有 CAS 判定：并入（追加 alias）或新增（source=review）。
产出 CAS+_<cat>.json 与 free_resolution_<cat>.json（FREE 短语 -> 标准 attribute_id 的映射表）。

用法：
  python -m stage_a.a2_aggregate_free [--category food_and_beverages] [--no-llm]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import config
from common import embedding, llm
from common import product_index as pidx
from common.io_utils import char_jaccard, normalize, read_jsonl, read_json, write_json

A2_PROMPT = """你是电商商品属性标准化专家。品类「{category}」已有如下标准属性（CAS）：
{cas}

现在有一组从用户评论中自由生成、含义相近的 aspect 短语（已聚类）：
{phrases}

请判断这组短语对应的属性，输出严格 JSON：
{{
  "decision": "merge" | "new",            // merge=并入某个已有 CAS 属性；new=作为新属性
  "target_attribute_id": "<merge 时填已有 attribute_id；new 时为空字符串>",
  "canonical_name": "<new 时给出规范中文属性名；merge 时可空>",
  "aliases": ["<这些短语去 FREE:: 前缀后的别名>", ...],
  "value_type": "numeric|enum|boolean|text|text:subjective"
}}
判断标准：只有当短语确为某已有属性的同义变体时才 merge。只输出 JSON。"""


_FREE_PREFIX = re.compile(r"^FREE(?:\s*::|\s*[：:＿_]\s*)(.+)$", re.I)


def _free_phrase(attr_id: str) -> str:
    m = _FREE_PREFIX.match(str(attr_id or "").strip())
    return m.group(1).strip(" ：:_") if m else str(attr_id or "").strip()


def _is_free_attr_id(attr_id: str) -> bool:
    return _FREE_PREFIX.match(str(attr_id or "").strip()) is not None


def _cas_block(cas: dict) -> str:
    return "\n".join(
        f"- {a['attribute_id']} | {a['canonical_name']} | {'、'.join(a.get('aliases', [])[:5])}"
        for a in cas.get("attributes", [])
    )


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


def run_category(category: str, use_llm: bool = True):
    cas = read_json(config.STAGE_A / f"CAS_{category}.json", default={"category": category, "attributes": []})
    raw = list(read_jsonl(config.STAGE_A / f"raw_aspects_{category}.jsonl"))
    free_phrases = sorted({_free_phrase(r["attribute_id"]) for r in raw
                           if _is_free_attr_id(str(r.get("attribute_id", "")))})
    print(f"[A2:{category}] FREE phrases: {len(free_phrases)}")

    resolution: dict[str, str] = {}   # FREE 短语 -> 标准 attribute_id
    if not free_phrases:
        write_json(config.STAGE_A / f"CAS+_{category}.json", cas)
        write_json(config.STAGE_A / f"free_resolution_{category}.json", resolution)
        return

    # 1) Jaccard 粗去重（归并到代表短语）
    reps: list[str] = []
    phrase_to_rep: dict[str, str] = {}
    for ph in free_phrases:
        nph = normalize(ph)
        matched = None
        for rep in reps:
            if char_jaccard(nph, normalize(rep)) > config.A2_JACCARD_DEDUP:
                matched = rep
                break
        if matched is None:
            reps.append(ph)
            matched = ph
        phrase_to_rep[ph] = matched

    # 2) 对代表短语聚类
    if use_llm and len(reps) > 1:
        try:
            labels = embedding.cluster(reps, config.A2_CLUSTER_DISTANCE)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 聚类失败({e!r})，每短语独立。")
            labels = list(range(len(reps)))
    else:
        labels = list(range(len(reps)))
    clusters: dict[int, list[str]] = defaultdict(list)
    for rep, lab in zip(reps, labels):
        clusters[lab].append(rep)

    # 3) LLM 裁决（并发计算，串行应用以保证线程安全）
    prefix = config.category_prefix(category)
    seen_ids = {a["attribute_id"] for a in cas["attributes"]}
    cas_block = _cas_block(cas)
    attr_by_id = {a["attribute_id"]: a for a in cas["attributes"]}

    cluster_list = list(clusters.values())

    def _judge(cl_reps):
        if not use_llm:
            return None
        try:
            return llm.chat_json(
                A2_PROMPT.format(
                    category=category, cas=cas_block,
                    phrases="\n".join(f"- {p}" for p in cl_reps)),
                namespace="a2", max_tokens=512,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] LLM 裁决失败({e!r})。")
            return None

    judgments = llm.run_many(cluster_list, _judge, desc=f"A2:{category}") if use_llm \
        else [None] * len(cluster_list)

    for cl_reps, res in zip(cluster_list, judgments):
        decision, target, canonical, aliases, vtype = "new", "", cl_reps[0], list(cl_reps), "text"
        if isinstance(res, dict):
            decision = res.get("decision", "new")
            target = (res.get("target_attribute_id") or "").strip()
            canonical = (res.get("canonical_name") or cl_reps[0]).strip()
            aliases = list(dict.fromkeys(cl_reps + [a for a in res.get("aliases", []) if a]))
            vtype = res.get("value_type", "text")

        if decision == "merge" and target in attr_by_id:
            attr_by_id[target]["aliases"] = list(dict.fromkeys(
                attr_by_id[target].get("aliases", []) + aliases))
            attr_by_id[target]["source"] = "both" if attr_by_id[target].get("source") == "param" else attr_by_id[target].get("source", "review")
            assigned = target
        else:
            new_id = _make_attr_id(prefix, canonical, seen_ids)
            new_attr = {
                "attribute_id": new_id, "canonical_name": canonical,
                "aliases": aliases, "value_type": vtype,
                "source": "review", "category": category,
            }
            cas["attributes"].append(new_attr)
            attr_by_id[new_id] = new_attr
            assigned = new_id

        # 记录该 cluster 内所有原始 FREE 短语 -> assigned
        for rep in cl_reps:
            for ph, r in phrase_to_rep.items():
                if r == rep:
                    resolution[ph] = assigned

    write_json(config.STAGE_A / f"CAS+_{category}.json", cas)
    write_json(config.STAGE_A / f"free_resolution_{category}.json", resolution)
    print(f"[A2:{category}] CAS+ attrs: {len(cas['attributes'])} | resolved FREE: {len(resolution)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    by_cat = pidx.bundles_by_category()
    cats = [args.category] if args.category else sorted(by_cat)
    for cat in cats:
        run_category(cat, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
