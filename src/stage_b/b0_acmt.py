"""Stage B0 — 评论侧候选属性集 A_cmt(p)。

按 product 聚合 Stage A3 的 resolved_aspects，得到每个商品评论侧已命中的
标准属性集合 A_cmt(p)，作为 claim 抽取的强约束 schema 与 pair 候选集。

产出 data/processed/stageB/acmt.json：{product_id: {attribute_id: {canonical_name, aliases}}}
以及把每条 resolved aspect 按 (product, attribute) 归档，供 B5/标签复用。

用法：python -m stage_b.b0_acmt
"""
from __future__ import annotations

from collections import defaultdict

import config
from common import product_index as pidx
from common.io_utils import read_jsonl, read_json, write_json


def build_acmt():
    resolved = list(read_jsonl(config.STAGE_A / "resolved_aspects.jsonl"))
    # 收集每品类 CAS+ 以取 canonical_name/aliases
    cas_cache: dict[str, dict] = {}

    def attr_meta(category: str, attribute_id: str):
        if category not in cas_cache:
            cas_cache[category] = {
                a["attribute_id"]: a
                for a in read_json(config.STAGE_A / f"CAS+_{category}.json", default={"attributes": []}).get("attributes", [])
            }
        a = cas_cache[category].get(attribute_id, {})
        return {
            "canonical_name": a.get("canonical_name", attribute_id),
            "aliases": a.get("aliases", []),
        }

    kws = getattr(config, "EVAL_LEAKAGE_KEYWORDS", [])

    def is_eval_leak(meta: dict) -> bool:
        name = (meta.get("canonical_name") or "").strip("<> ")
        return any(k in name for k in kws)

    acmt: dict[str, dict] = defaultdict(dict)
    n_drop = 0
    for r in resolved:
        pid = r["product_id"]
        aid = r["attribute_id"]
        cat = r["category"]
        if aid not in acmt[pid]:
            meta = attr_meta(cat, aid)
            if is_eval_leak(meta):
                n_drop += 1
                continue
            acmt[pid][aid] = meta
    write_json(config.STAGE_B / "acmt.json", acmt)
    sizes = [len(v) for v in acmt.values()]
    avg = sum(sizes) / len(sizes) if sizes else 0
    print(f"[B0] products with A_cmt: {len(acmt)} | avg |A_cmt(p)|={avg:.1f} "
          f"| total candidate pairs={sum(sizes)} | dropped eval-leak attrs={n_drop}")
    return acmt


if __name__ == "__main__":
    build_acmt()
