"""Stage B2/B3 — (商品,属性) pair 枚举 + claim passage 拼接。

B2：pair 候选 = A_cmt(p) 全集；has_claim_srt 标记主播在 SRT 中是否真有相关 claim。
B3：对每个 pair，取该属性全部 claim → 时序排序 → 相邻 Jaccard≥0.9 去重 → 拼接成 passage
    （token 超限沿时间轴均匀下采样），segments 元数据完整保留。
产出 data/processed/stageB/pair_skeleton.jsonl

用法：python -m stage_b.b2_b3_passage
     python -m stage_b.b2_b3_passage --acmt data/processed/stageB_product_v2/acmt_product_v2.json --claim_dir data/processed/stageB_product_v2/claim_list --out data/processed/stageB_product_v2/pair_skeleton_product_v2_rerun.jsonl
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import config
from common import srt as S
from common.io_utils import bigram_jaccard, char_jaccard, read_jsonl, read_json, write_jsonl


def _approx_tokens(s: str) -> int:
    # 中文近似：字符数即 token 数量级
    return len(s)


def build_passage(claims: list[dict]) -> tuple[str, list[dict]]:
    claims = sorted(claims, key=lambda c: (c.get("srt_file", ""), S.ts_to_seconds(c.get("start_ts", "00:00:00,000"))))
    # 相邻去重
    dedup = []
    for c in claims:
        if dedup and char_jaccard(c["claim_text"], dedup[-1]["claim_text"]) >= config.B3_CLAIM_JACCARD_DEDUP:
            continue
        dedup.append(c)
    # 下采样
    texts = [c["claim_text"] for c in dedup]
    total = _approx_tokens("\n---\n".join(texts))
    if total > config.B3_PASSAGE_MAX_TOKENS and len(dedup) > 1:
        keep_ratio = config.B3_PASSAGE_TARGET_TOKENS / total
        keep_n = max(1, int(len(dedup) * keep_ratio))
        step = len(dedup) / keep_n
        idxs = sorted({min(len(dedup) - 1, int(i * step)) for i in range(keep_n)})
        dedup = [dedup[i] for i in idxs]
    passage = "\n---\n".join(c["claim_text"] for c in dedup)
    segments = [{
        "claim_id": c["claim_id"],
        "clip_id": c.get("srt_file", ""),
        "t_start": S.ts_to_seconds(c.get("start_ts", "00:00:00,000")),
        "t_end": S.ts_to_seconds(c.get("end_ts", "00:00:00,000")),
        "start_ts": c.get("start_ts", ""),
        "end_ts": c.get("end_ts", ""),
        "text": c["claim_text"],
    } for c in dedup]
    return passage, segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--claim_dir", default=str(config.STAGE_B / "claim_list"))
    ap.add_argument("--out", default=str(config.STAGE_B / "pair_skeleton.jsonl"))
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    claim_dir = config.ROOT / args.claim_dir if not str(args.claim_dir).startswith("/") else args.claim_dir

    rows = []
    n_with_claim = 0
    for pid, attrs in acmt.items():
        from pathlib import Path
        claims = list(read_jsonl(Path(claim_dir) / f"{pid}.jsonl"))
        by_attr: dict[str, list[dict]] = defaultdict(list)
        for c in claims:
            by_attr[c["attribute_id"]].append(c)
        for aid, meta in attrs.items():
            ac = by_attr.get(aid, [])
            has = len(ac) > 0
            if has:
                n_with_claim += 1
                passage, segments = build_passage(ac)
            else:
                passage, segments = "", []
            rows.append({
                "pair_id": f"p{pid}__{aid}",
                "product_id": pid,
                "attribute_id": aid,
                "attribute_canonical": meta.get("canonical_name", aid),
                "aliases": meta.get("aliases", []),
                "has_claim_srt": has,
                "passage": passage,
                "segments": segments,
            })
    write_jsonl(args.out, rows)
    print(f"[B2/B3] pairs={len(rows)} (has_claim_srt={n_with_claim}, "
          f"no_claim={len(rows) - n_with_claim}) -> {args.out}")


if __name__ == "__main__":
    main()
