"""Stage A3 — 评论标签重写。

遍历 raw_aspects：
- 非 FREE:: 条目原样保留；
- FREE:: 条目按 A2 的 free_resolution 表替换为 CAS+ 标准 attribute_id；
- 找不到的写入 unresolved_pool.jsonl。
产出 resolved_aspects.jsonl（attribute_id 全标准化）。

用法：python -m stage_a.a3_resolve_labels
"""
from __future__ import annotations

import argparse
import re

import config
from common import product_index as pidx
from common.io_utils import read_jsonl, read_json, write_jsonl

_FREE_PREFIX = re.compile(r"^FREE(?:\s*::|\s*[：:＿_]\s*)(.+)$", re.I)


def _free_phrase(attr_id: str) -> str | None:
    m = _FREE_PREFIX.match(str(attr_id or "").strip())
    if not m:
        return None
    return m.group(1).strip(" ：:_")


def run_category(category: str):
    raw = list(read_jsonl(config.STAGE_A / f"raw_aspects_{category}.jsonl"))
    resolution = read_json(config.STAGE_A / f"free_resolution_{category}.json", default={})
    casplus = read_json(config.STAGE_A / f"CAS+_{category}.json", default={"attributes": []})
    valid_ids = {a["attribute_id"] for a in casplus.get("attributes", [])}

    resolved, unresolved = [], []
    for r in raw:
        aid = str(r.get("attribute_id", ""))
        phrase = _free_phrase(aid)
        if phrase is not None:
            new_id = resolution.get(phrase)
            if not new_id:
                unresolved.append(r)
                continue
            r = {**r, "attribute_id": new_id, "_was_free": phrase}
        else:
            if aid not in valid_ids:
                # LLM 自创了 CAS 之外的非 FREE id（少见）→ 入 unresolved
                unresolved.append(r)
                continue
        resolved.append(r)
    return resolved, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    args = ap.parse_args()
    by_cat = pidx.bundles_by_category()
    cats = [args.category] if args.category else sorted(by_cat)

    all_resolved, all_unresolved = [], []
    for cat in cats:
        res, unres = run_category(cat)
        all_resolved.extend(res)
        all_unresolved.extend(unres)
        print(f"[A3:{cat}] resolved={len(res)} unresolved={len(unres)}")

    write_jsonl(config.STAGE_A / "resolved_aspects.jsonl", all_resolved)
    write_jsonl(config.STAGE_A / "unresolved_pool.jsonl", all_unresolved)
    total = len(all_resolved) + len(all_unresolved)
    rate = len(all_unresolved) / total if total else 0
    print(f"[A3] resolved={len(all_resolved)} unresolved={len(all_unresolved)} "
          f"({rate:.2%}) -> resolved_aspects.jsonl")


if __name__ == "__main__":
    main()
