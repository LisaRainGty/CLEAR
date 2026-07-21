"""Stage C1 — 详情图分流（VLM 单图 9 类分类）。

只对在 A_cmt 中有候选属性的商品处理。给每张详情图打 9 类标签之一，决定下游：
  spec_table/certificate/size_chart → C3 OCR；其余 → C4 VLM。
并采样代表图（每个 VLM 类别若干，总数 ≤ C1_REP_IMAGE_MAX）。
产出 data/processed/stageC/image_index.json：{product_id: {images:[{path,category}], ocr_images:[...], rep_images:[...]}}

用法：python -m stage_c.c1_image_triage [--limit N]
     python -m stage_c.c1_image_triage --acmt data/processed/stageB_product_v2/acmt_atomic_productv2_direct_strict_full.json --out data/processed/stageB_product_v2/image_index_atomic_productv2.json
"""
from __future__ import annotations

import argparse
import os

import config
from common import llm
from common import product_index as pidx
from common.io_utils import read_json, write_json

C1_PROMPT = """你是电商详情图分类器。请判断这张图属于以下哪一类（只选一个）：
- spec_table: 文字规格表/参数列表图
- certificate: 认证检测图/资质证书
- size_chart: 尺寸表/码数对照
- material_closeup: 材质特写/纹理细节
- product_photo: 产品实拍/多角度展示
- scene_demo: 场景使用图/模特展示
- packaging: 包装展示
- comparison: 颜色/款式对比
- other: 其他
只输出 JSON：{"category": "<上述之一>"}"""


def classify_image(path: str) -> str:
    data_url = llm.encode_image(path)
    if not data_url:
        return "other"
    try:
        res = llm.chat_json(C1_PROMPT, images=[data_url], model=config.VISION_MODEL,
                            namespace="c1", max_tokens=64)
        cat = str(res.get("category", "other")).strip()
        return cat if cat in config.IMAGE_CATEGORIES else "other"
    except Exception:
        return "other"


def process_product(pid: str, bundle) -> dict:
    images = []
    for dp in bundle.detail_images:
        ap = str(pidx.resolve(dp))
        if not os.path.exists(ap):
            continue
        cat = classify_image(ap)
        images.append({"path": ap, "category": cat})
    ocr_images = [im for im in images if im["category"] in config.OCR_CATEGORIES]
    vlm_pool = [im for im in images if im["category"] not in config.OCR_CATEGORIES]
    # 代表图：优先 material_closeup/product_photo/comparison/packaging，按出现顺序取
    priority = ["material_closeup", "product_photo", "comparison", "packaging", "scene_demo", "other"]
    vlm_pool_sorted = sorted(vlm_pool, key=lambda im: priority.index(im["category"]) if im["category"] in priority else 99)
    rep_images = [im["path"] for im in vlm_pool_sorted[:config.C1_REP_IMAGE_MAX]]
    return {
        "images": images,
        "ocr_images": [im["path"] for im in ocr_images],
        "rep_images": rep_images,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--out", default=str(config.STAGE_C / "image_index.json"))
    ap.add_argument("--rerun-empty", action="store_true",
                    help="Reprocess products whose cached image list is empty.")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    bundles = pidx.build_bundles()
    out_path = args.out
    index = read_json(out_path, default={})

    pids = [p for p in acmt if p in bundles]
    if args.category:
        pids = [p for p in pids if bundles[p].category == args.category]
    if args.limit:
        pids = pids[:args.limit]
    def _is_empty(pid: str) -> bool:
        info = index.get(pid)
        return not isinstance(info, dict) or not info.get("images")

    todo = [p for p in pids if p not in index or (args.rerun_empty and _is_empty(p))]
    print(f"[C1] products to triage: {len(todo)} (already done={len(pids) - len(todo)})")

    def _job(pid):
        return pid, process_product(pid, bundles[pid])

    results = llm.run_many(todo, _job, desc="C1")
    for r in results:
        if isinstance(r, tuple):
            pid, val = r
            index[pid] = val
    write_json(out_path, index)
    n_ocr = sum(len(v["ocr_images"]) for v in index.values())
    print(f"[C1] indexed products={len(index)} ocr_images_total={n_ocr} -> image_index.json")


if __name__ == "__main__":
    main()
