"""最终合并与划分。

在 (product_id, attribute_id) 键上 inner-join：
  Stage B pair_records.jsonl × Stage C fact_records.jsonl × labels.jsonl
→ data/final/dataset.jsonl（schema 见 Methodology_Data §3.2）。

按 room_id 分组 70:10:20（同直播间所有 pair 落同一 split）。
产出 Table 1 分布统计（按品类/属性/主播 + 正类占比）。

用法：python -m final.join_split
     python -m final.join_split --pair_records data/processed/stageB_product_v2/pair_records_product_v2.jsonl --facts data/processed/stageB_product_v2/fact_records_product_v2.jsonl --labels data/processed/stageB_product_v2/labels_product_v2.jsonl --out data/final/repaired_v1/dataset_product_v2.jsonl --table_prefix data/final/repaired_v1/product_v2_table1
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import config
from common import product_index as pidx
from common.io_utils import read_jsonl, write_json, write_jsonl


def grouped_split(group_keys: list[str]) -> dict[str, str]:
    """按 group 分配 split，使各 split 的 pair 数尽量接近目标比例。"""
    groups = sorted(set(group_keys))
    rng = random.Random(config.SPLIT_SEED)
    rng.shuffle(groups)
    counts = Counter(group_keys)
    total = sum(counts.values())
    targets = {k: v * total for k, v in config.SPLIT_RATIO.items()}
    cur = {k: 0 for k in config.SPLIT_RATIO}
    assign: dict[str, str] = {}
    # 先大组后小组，贪心放进"最缺额"的 split
    for g in sorted(groups, key=lambda x: -counts[x]):
        split = max(cur, key=lambda s: targets[s] - cur[s])
        assign[g] = split
        cur[split] += counts[g]
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_records", default=str(config.STAGE_B / "pair_records.jsonl"))
    ap.add_argument("--facts", default=str(config.STAGE_C / "fact_records.jsonl"))
    ap.add_argument("--labels", default=str(config.LABELS_PATH))
    ap.add_argument("--out", default=str(config.DATASET_PATH))
    ap.add_argument("--table_prefix", default=str(config.FINAL / "table1_stats"))
    args = ap.parse_args()

    pair_records = {(r["product_id"], r["attribute_id"]): r
                    for r in read_jsonl(args.pair_records)}
    fact_records = {(r["product_id"], r["attribute_id"]): r
                    for r in read_jsonl(args.facts)}
    labels = {(r["product_id"], r["attribute_id"]): r
              for r in read_jsonl(args.labels)}
    bundles = pidx.build_bundles()

    keys = set(pair_records) & set(labels)   # fact 可能缺（无证据），用空兜底
    print(f"[final] pair={len(pair_records)} fact={len(fact_records)} "
          f"labels={len(labels)} joinable={len(keys)}")

    # 先确定每条记录的 room_id，再按 room_id 分组划分
    key_room = {}
    for k in keys:
        pid = k[0]
        key_room[k] = bundles[pid].room_id if pid in bundles else "UNKNOWN"
    assign = grouped_split([key_room[k] for k in keys])
    room_split = {}  # room_id -> split（用 assign，但 assign 基于 list 顺序；重建 map）
    # grouped_split 返回 group->split
    room_split = grouped_split(list(key_room.values()))

    rows = []
    for k in sorted(keys):
        pid, aid = k
        pr = pair_records[k]
        fr = fact_records.get(k, {})
        lb = labels[k]
        b = bundles.get(pid)
        split = room_split.get(key_room[k], "train")
        rows.append({
            "pair_id": pr["pair_id"],
            "product_id": pid,
            "category": pr.get("category") or (b.category if b else ""),
            "subcategory": b.subcategory if b else "",
            "room_id": key_room[k],
            "attribute_id": aid,
            "attribute_name": pr.get("attribute_canonical", aid),
            "claim": pr.get("claim", {}),
            "evidence_params": fr.get("evidence_params", []),
            "evidence_ocr": fr.get("evidence_ocr", []),
            "evidence_vlm": fr.get("evidence_vlm", []),
            "evidence_count": fr.get("evidence_count", {"params": 0, "ocr": 0, "vlm": 0}),
            "coverage": fr.get("coverage", 0),
            "confidence": fr.get("confidence", "absent"),
            "y": lb["y"],
            "c": lb["c"],
            "label_audit": lb.get("label_audit", {}),
            "split": split,
        })
    write_jsonl(args.out, rows)
    _table1(rows, args.table_prefix, args.out)


def _table1(rows: list[dict], table_prefix: str, out_path: str):
    n = len(rows)
    pos = sum(1 for r in rows if r["y"] == 1)
    split_c = Counter(r["split"] for r in rows)
    split_pos = defaultdict(int)
    for r in rows:
        if r["y"] == 1:
            split_pos[r["split"]] += 1

    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += 1
        by_cat[r["category"]][1] += r["y"]
    by_room = defaultdict(lambda: [0, 0])
    for r in rows:
        by_room[r["room_id"]][0] += 1
        by_room[r["room_id"]][1] += r["y"]

    stats = {
        "n_pairs": n,
        "n_pos": pos,
        "pos_rate": round(pos / max(1, n), 4),
        "n_products": len({r["product_id"] for r in rows}),
        "n_attributes": len({r["attribute_id"] for r in rows}),
        "n_rooms": len({r["room_id"] for r in rows}),
        "split": {s: {"n": split_c[s], "pos": split_pos[s],
                      "pos_rate": round(split_pos[s] / max(1, split_c[s]), 4)}
                  for s in ("train", "val", "test")},
        "by_category": {k: {"n": v[0], "pos": v[1], "pos_rate": round(v[1] / max(1, v[0]), 4)}
                        for k, v in sorted(by_cat.items())},
        "coverage_dist": dict(sorted(Counter(r["coverage"] for r in rows).items())),
    }
    write_json(f"{table_prefix}.json", stats)

    lines = ["# Table 1 — 数据集分布统计\n",
             f"- pair 总数: **{n}**　正类(y=1): **{pos}**（{stats['pos_rate']:.1%}）",
             f"- 商品数: {stats['n_products']}　属性数: {stats['n_attributes']}　直播间数: {stats['n_rooms']}\n",
             "## 划分", "| split | n | y=1 | 正类占比 |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        d = stats["split"][s]
        lines.append(f"| {s} | {d['n']} | {d['pos']} | {d['pos_rate']:.1%} |")
    lines += ["\n## 按一级品类", "| 品类 | n | y=1 | 正类占比 |", "|---|---|---|---|"]
    for k, v in stats["by_category"].items():
        lines.append(f"| {k} | {v['n']} | {v['pos']} | {v['pos_rate']:.1%} |")
    lines += ["\n## 三源证据 coverage 分布", "| coverage | pairs |", "|---|---|"]
    for k, v in stats["coverage_dist"].items():
        lines.append(f"| {k} | {v} |")
    from pathlib import Path
    Path(f"{table_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[final] dataset.jsonl rows={n} pos={pos} ({stats['pos_rate']:.1%})")
    print(f"[final] split={dict(split_c)} -> {out_path} + {table_prefix}.{{json,md}}")


if __name__ == "__main__":
    main()
