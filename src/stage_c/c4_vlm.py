"""Stage C4 — 逐属性 × VLM 视觉证据抽取。

对每个商品，主图 + C1 选出的代表图（≤8）一次 multi-image 调用，按属性给出可在视觉上
直接观察到的客观证据。产出 data/processed/stageC/evidence_vlm.json：
  {product_id: {attribute_id: [{raw_quote, image_path}]}}

用法：python -m stage_c.c4_vlm [--limit N]
     python -m stage_c.c4_vlm --acmt data/processed/stageB_product_v2/acmt_product_v2.json --out data/processed/stageB_product_v2/evidence_vlm_product_v2.json
"""
from __future__ import annotations

import argparse
import os

import config
from common import llm
from common import product_index as pidx
from common.io_utils import read_json, write_json


def resolve_image_path(path: str) -> str:
    """Map stale absolute image paths from earlier remote runs to this workspace."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    parts = os.path.normpath(path).split(os.sep)
    if "data" in parts:
        i = parts.index("data")
        cand = str(config.ROOT.joinpath(*parts[i:]))
        if os.path.exists(cand):
            return cand
    return path

C4_PROMPT = """角色：电商商品多模态视觉取证员。
任务：对下方候选属性列表，逐一审视输入的商品图片，给出所有能在视觉上直接观察到的客观证据。
禁出现"看起来/可能/大概"。
硬约束：
1. 只输出能被视觉证据直接支撑的项；attribute_id 必须取自候选列表。
2. 【只列出有视觉证据命中的 attribute_id；图中观察不到的属性请直接省略，不要输出空 matches 项】。
3. 每条证据报告对应图片的序号（image_index，从1开始，对应输入图片顺序）。
4. raw_quote 是对视觉细节的简短客观描述（如"主图盒身正面标'4g蛋白'""第4图模特展示版型"），
   不复述商品标题或参数文本。
5. 同一 attribute 在多张图、多个细节都可见的，全部列出。

候选属性（attribute_id | 标准名）：
{attrs}

输入图片顺序：
{img_list}

输出严格 JSON 数组（只含有命中的属性）：
[{{"attribute_id": "...", "matches": [{{"raw_quote": "...", "image_index": <int>}}]}}]
只输出 JSON 数组。"""


def extract_product(pid: str, acmt_p: dict, image_paths: list[str]) -> dict:
    res = {aid: [] for aid in acmt_p}
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if not image_paths:
        return res
    data_urls = []
    used_paths = []
    for p in image_paths:
        u = llm.encode_image(p)
        if u:
            data_urls.append(u)
            used_paths.append(p)
    if not data_urls:
        return res
    img_list = "\n".join(f"{i + 1}. {os.path.basename(p)}" for i, p in enumerate(used_paths))
    attrs_block = "\n".join(f"- {a} | {acmt_p[a].get('canonical_name', a)}" for a in acmt_p)
    try:
        arr = llm.chat_json(
            C4_PROMPT.format(attrs=attrs_block, img_list=img_list),
            images=data_urls, model=config.VISION_MODEL,
            namespace="c4", max_tokens=4096,
        )
    except Exception:
        return res
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            aid = item.get("attribute_id")
            if aid not in res:
                continue
            for m in item.get("matches", []) or []:
                rq = str(m.get("raw_quote", "")).strip()
                if not rq:
                    continue
                try:
                    k = int(m.get("image_index", 1)) - 1
                    ip = used_paths[k] if 0 <= k < len(used_paths) else (used_paths[0] if used_paths else "")
                except (TypeError, ValueError):
                    ip = used_paths[0] if used_paths else ""
                res[aid].append({"raw_quote": rq, "image_path": ip})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--image_index", default=str(config.STAGE_C / "image_index.json"))
    ap.add_argument("--out", default=str(config.STAGE_C / "evidence_vlm.json"))
    ap.add_argument("--rerun-empty", action="store_true",
                    help="Reprocess products whose cached VLM evidence is entirely empty.")
    ap.add_argument("--rerun-missing-attrs", action="store_true",
                    help="Reprocess products whose cached VLM evidence lacks requested attribute keys.")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    image_index = read_json(args.image_index, default={})
    bundles = pidx.build_bundles()

    pids = [p for p in acmt if p in bundles]
    if args.category:
        pids = [p for p in pids if bundles[p].category == args.category]
    if args.limit:
        pids = pids[:args.limit]

    out = read_json(args.out, default={})
    def _is_empty(pid: str) -> bool:
        ev = out.get(pid)
        if not isinstance(ev, dict):
            return True
        return not any(bool(v) for v in ev.values())

    def _missing_attrs(pid: str) -> bool:
        ev = out.get(pid)
        if not isinstance(ev, dict):
            return True
        return any(a not in ev for a in acmt[pid])

    todo = [
        p for p in pids
        if p not in out
        or (args.rerun_empty and _is_empty(p))
        or (args.rerun_missing_attrs and _missing_attrs(p))
    ]
    print(f"[C4] products to VLM-extract: {len(todo)}")

    def _job(pid):
        b = bundles[pid]
        imgs = []
        if b.main_image:
            mp = str(pidx.resolve(b.main_image))
            if os.path.exists(mp):
                imgs.append(mp)
        imgs += [resolve_image_path(p) for p in image_index.get(pid, {}).get("rep_images", [])]
        return pid, extract_product(pid, acmt[pid], imgs)

    results = llm.run_many(todo, _job, desc="C4")
    for r in results:
        if isinstance(r, tuple):
            out[r[0]] = r[1]
    write_json(args.out, out)
    hit = sum(1 for p in out.values() for lst in p.values() if lst)
    print(f"[C4] products={len(out)} attr-hits={hit} -> {args.out}")


if __name__ == "__main__":
    main()
