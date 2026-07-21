"""Stage B1 — 主播原子 claim 抽取（LangExtract，schema-guided + source-span grounding）。

严格按 Methodology §1 B1：
  1) 把同一 product_id 名下所有 clip 的 SRT 升序拼接为单一长文本 + cue 边界表；
  2) 用 LangExtract 在 A_cmt(p) 约束下抽取 extraction_class="attribute_claim"，
     extraction_text 为主播原话连续子串（库自动做 source-span grounding，给出 char_interval）；
  3) 后处理四步：char_interval=None 丢弃；attribute_id ∉ A_cmt(p) 丢弃；
     用 char_interval.start_pos/end_pos 在 cue 边界表反查 srt_file/start_ts/end_ts；写记录。

后端：LangExtract 官方后端 Gemini 2.5 Flash 因地理封锁不可达，改用 OpenAI 兼容
provider 指向 matpool 网关 + Qwen-Flash（保留 LangExtract 的 grounding 与 char_interval 机制）。

产出 data/processed/stageB/claim_list/<product_id>.jsonl

用法：python -m stage_b.b1_claim_extract [--category ...] [--limit N]
     python -m stage_b.b1_claim_extract --acmt data/processed/stageB_product_v2/acmt_product_v2.json --out_dir data/processed/stageB_product_v2/claim_list
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache

import config
from common import product_index as pidx
from common import srt as S
from common.io_utils import normalize, read_json, write_jsonl

import langextract as lx

# ---- LangExtract 任务描述与 few-shot（schema 教学；attribute_id 以各商品 A_cmt 为准）----
B1_TASK = (
    "角色：电商直播事实抽取员。从主播口播文本中，抽取所有针对【某个商品属性】的独立事实陈述（claim）。\n"
    "硬约束：\n"
    "1. extraction_text 必须是口播原文中真实存在的连续子串，禁止改写、概括或跨段拼接。\n"
    "2. attributes.attribute_id 必须取自下方候选属性集合 A_cmt(p)；不能归入其中任何一项的，"
    "一律不要输出（包括主播提到但不在候选集合内的属性）。\n"
    "3. 必须做语义精确匹配，禁止把 claim 归到最接近但不相同的宽泛属性。"
    "例如头围/尺码不能归入帽顶款式，弹力/松紧不能归入厚度，适用人群不能归入款式。\n"
    "4. 同一属性的口语复读只抽表达最完整的一次；不同角度（如容量 vs 件数）拆为独立 claim。\n"
    "5. 忽略下单话术 / 链接编号 / 主播八卦 / 与商品属性无关的内容；"
    "若句子只是在推荐链接或颜色、没有明确商品事实，也不要输出。\n"
)

B1_EXAMPLES = [
    lx.data.ExampleData(
        text=("今天这款娟姗牛奶，每100毫升蛋白质4克，远高于普通牛奶。"
              "保质期是21天，冷藏储存。产地内蒙古，绝对的好奶源。"),
        extractions=[
            lx.data.Extraction(
                extraction_class="attribute_claim",
                extraction_text="保质期是21天",
                attributes={"attribute_id": "FOOD_保质期"},
            ),
            lx.data.Extraction(
                extraction_class="attribute_claim",
                extraction_text="产地内蒙古",
                attributes={"attribute_id": "FOOD_产地"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text="这件外套是100%棉面料，尺码从S到XL都有，厚度是加厚款。",
        extractions=[
            lx.data.Extraction(
                extraction_class="attribute_claim",
                extraction_text="100%棉面料",
                attributes={"attribute_id": "APPAREL_面料材质"},
            ),
            lx.data.Extraction(
                extraction_class="attribute_claim",
                extraction_text="尺码从S到XL都有",
                attributes={"attribute_id": "APPAREL_尺码"},
            ),
            lx.data.Extraction(
                extraction_class="attribute_claim",
                extraction_text="厚度是加厚款",
                attributes={"attribute_id": "APPAREL_厚度"},
            ),
        ],
    ),
]


@lru_cache(maxsize=1)
def _lx_model():
    from langextract.providers.openai import OpenAILanguageModel
    return OpenAILanguageModel(
        model_id=config.TEXT_MODEL,
        api_key=config.MATPOOL_API_KEY,
        base_url=config.MATPOOL_BASE_URL,
        format_type=lx.data.FormatType.JSON,
        max_workers=1,
        temperature=0.0,
    )


def _acmt_block(acmt_p: dict) -> str:
    return "\n".join(
        f"- {aid} | {meta.get('canonical_name', aid)}"
        f" | family={meta.get('source_family', '')}"
        f" | value_type={meta.get('value_type', '')}"
        f" | aliases={'、'.join(meta.get('aliases', [])[:8])}"
        for aid, meta in acmt_p.items()
    )


def _chunk_ranges(concat: S.ConcatResult, chunk_chars: int) -> list[tuple[int, int]]:
    if chunk_chars <= 0 or len(concat.text) <= chunk_chars or not concat.spans:
        return [(0, len(concat.text))]
    out: list[tuple[int, int]] = []
    start = concat.spans[0].char_start
    end = start
    for sp in concat.spans:
        if end > start and sp.char_end - start > chunk_chars:
            out.append((start, end))
            start = sp.char_start
        end = sp.char_end
    if end > start:
        out.append((start, end))
    return out


def extract_product(
    product_id: str,
    acmt_p: dict,
    srt_files: list[str],
    seq_start: int = 0,
    max_char_buffer: int = 3000,
    chunk_chars: int = 0,
) -> list[dict]:
    files = [str(pidx.resolve(f)) for f in srt_files]
    files = [f for f in files if os.path.exists(f)]
    if not files or not acmt_p:
        return []
    concat = S.concat_product_srt(files)
    text = concat.text
    if not text.strip():
        return []

    prompt = B1_TASK + "\n本商品候选属性集合 A_cmt(p)（attribute_id | 标准名 | 别名）：\n" + _acmt_block(acmt_p)
    valid_ids = set(acmt_p.keys())
    rows = []
    seq = seq_start
    seen: set[tuple[str, str, str, str]] = set()
    for chunk_start, chunk_end in _chunk_ranges(concat, chunk_chars):
        chunk_text = text[chunk_start:chunk_end]
        if not chunk_text.strip():
            continue
        try:
            doc = lx.extract(
                text_or_documents=chunk_text,
                prompt_description=prompt,
                examples=B1_EXAMPLES,
                model=_lx_model(),
                fence_output=True,
                use_schema_constraints=False,
                max_char_buffer=max_char_buffer,
                extraction_passes=1,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] B1 LangExtract 失败 {product_id}: {e!r}")
            continue

        for ex in (doc.extractions or []):
            if ex.extraction_class != "attribute_claim":
                continue
            ci = ex.char_interval
            # 后处理1：char_interval=None（无法对齐源文本的幻觉）丢弃
            if ci is None or ci.start_pos is None or ci.end_pos is None:
                continue
            aid = str((ex.attributes or {}).get("attribute_id", "")).strip()
            # 后处理2：attribute_id 越界丢弃
            if aid not in valid_ids:
                continue
            # 后处理3：确认 extraction_text 可回到源文本，并反查完整跨 cue 时间戳。
            global_start = chunk_start + ci.start_pos
            global_end = chunk_start + ci.end_pos
            interval_text = text[global_start:global_end]
            if normalize(ex.extraction_text) and normalize(ex.extraction_text) not in normalize(interval_text):
                continue
            spans = concat.lookup_range(global_start, global_end)
            if not spans:
                continue
            first, last = spans[0], spans[-1]
            key = (aid, normalize(ex.extraction_text), first.srt_file, first.start_ts)
            if key in seen:
                continue
            seen.add(key)
            seq += 1
            rows.append({
                "claim_id": f"{product_id}_{seq}",
                "attribute_id": aid,
                "claim_text": ex.extraction_text,
                "srt_file": os.path.basename(first.srt_file),
                "srt_path": first.srt_file,
                "start_ts": first.start_ts,
                "end_ts": last.end_ts,
                "char_start": global_start,
                "char_end": global_end,
                "cue_span_count": len(spans),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--product_id", action="append", default=None,
                    help="Optional product_id filter; may be passed multiple times.")
    ap.add_argument("--max_char_buffer", type=int, default=3000)
    ap.add_argument("--chunk_chars", type=int, default=0,
                    help="Optional cue-aware product text window size. 0 keeps the original whole-product extraction.")
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--out_dir", default=str(config.STAGE_B / "claim_list"))
    ap.add_argument("--force", action="store_true", help="Re-extract even if the output file exists.")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    if not acmt:
        print(f"[B1] 无 acmt: {args.acmt}。请先构建 A_cmt。")
        return
    bundles = pidx.build_bundles()
    out_dir = os.path.abspath(args.out_dir)
    out_dir = os.path.realpath(out_dir)
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pids = [p for p in acmt if p in bundles]
    if args.product_id:
        wanted = {str(x) for x in args.product_id}
        pids = [p for p in pids if p in wanted]
    if args.category:
        pids = [p for p in pids if bundles[p].category == args.category]
    if args.limit:
        pids = pids[:args.limit]
    print(f"[B1] products to extract claims (LangExtract/{config.TEXT_MODEL}): {len(pids)}")

    from common import llm

    def _job(pid):
        out = out_dir / f"{pid}.jsonl"
        if out.exists() and not args.force:
            return ("skip", pid, 0)
        rows = extract_product(
            pid,
            acmt[pid],
            bundles[pid].srt_files,
            max_char_buffer=args.max_char_buffer,
            chunk_chars=args.chunk_chars,
        )
        write_jsonl(out, rows)
        return ("done", pid, len(rows))

    results = llm.run_many(pids, _job, desc="B1")
    done = sum(1 for r in results if isinstance(r, tuple) and r[0] == "done")
    nclaims = sum(r[2] for r in results if isinstance(r, tuple))
    print(f"[B1] done products={done} total claims={nclaims}")


if __name__ == "__main__":
    main()
