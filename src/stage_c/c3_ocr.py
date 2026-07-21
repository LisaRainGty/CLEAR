"""Stage C3 — 逐属性 × OCR 文本证据抽取。

对每个商品的 spec_table/certificate/size_chart 类图跑 PaddleOCR PP-OCRv4，
把多图 OCR 文本拼接（每段前缀 === <image_path> ===），送 LLM 按属性定向抽取原文片段。
产出 data/processed/stageC/evidence_ocr.json：{product_id: {attribute_id: [{raw_text, image_path}]}}
OCR 原文缓存到 data/processed/stageC/ocr_text/<product_id>.json，避免重复跑。

用法：python -m stage_c.c3_ocr [--limit N]
     python -m stage_c.c3_ocr --acmt data/processed/stageB_product_v2/acmt_product_v2.json --out data/processed/stageB_product_v2/evidence_ocr_product_v2.json
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache

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


def normalize_ocr_cache(cache: dict) -> dict:
    """Normalize stale absolute cache keys so local reruns can reuse OCR text."""
    out = {}
    for k, v in (cache or {}).items():
        nk = resolve_image_path(str(k))
        out[nk] = v
        out.setdefault(str(k), v)
    return out

C3_PROMPT = """角色：电商详情页结构化字段定向抽取员。
任务：对下方候选属性列表中的每一个 attribute_id，扫遍 OCR 文本，找出所有能直接支撑该
属性的连续原文片段（多条全部列出，禁改写）；找不到的返回空。
硬约束：raw_text 必须是 OCR 原文中真实存在的连续字符串；attribute_id 必须取自候选列表；
每条 raw_text 必须报告它所在的 image_path（取 === <path> === 前缀）。

候选属性（attribute_id | 标准名）：
{attrs}

OCR 文本（多图拼接）：
{ocr}

输出严格 JSON 数组：
[{{"attribute_id": "...", "matches": [{{"raw_text": "...", "image_path": "..."}}]}}]
只输出 JSON 数组。"""


@lru_cache(maxsize=1)
def _ocr_engine():
    import os as _os
    _os.environ.setdefault("FLAGS_use_mkldnn", "0")
    import warnings
    warnings.filterwarnings("ignore")
    from paddleocr import PaddleOCR  # type: ignore
    # PaddleOCR 3.x：禁用 oneDNN 规避 CPU PIR 运行期 bug；用 .predict() 接口。
    # 模型档位由 config.OCR_MODEL 控制：server=高召回（慢），mobile=快。
    tier = getattr(config, "OCR_MODEL", "mobile")
    if tier == "server":
        det, rec = "PP-OCRv5_server_det", "PP-OCRv5_server_rec"
    else:
        det, rec = "PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"
    kw = dict(
        lang="ch", enable_mkldnn=False,
        text_detection_model_name=det,
        text_recognition_model_name=rec,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    for attempt in (kw, {"lang": "ch", "enable_mkldnn": False}, {"lang": "ch"}):
        try:
            return PaddleOCR(**attempt)
        except TypeError:
            continue
    return PaddleOCR(lang="ch")


def ocr_image(path: str) -> str:
    try:
        engine = _ocr_engine()
        result = engine.predict(path)
        lines: list[str] = []
        for res in (result or []):
            # 3.x：dict（含 rec_texts）；老接口：嵌套 list
            if isinstance(res, dict):
                for t in (res.get("rec_texts") or []):
                    if t:
                        lines.append(str(t))
            else:
                for item in (res or []):
                    try:
                        txt = item[1][0]
                        if txt:
                            lines.append(str(txt))
                    except Exception:
                        continue
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] OCR 失败 {os.path.basename(path)}: {e!r}")
        return ""


def _ocr_one(path: str) -> tuple[str, str]:
    """进程池 worker：返回 (path, text)。每个进程各自 lru_cache 一个引擎。"""
    return path, ocr_image(path)


def get_product_ocr(pid: str, ocr_images: list[str]) -> dict[str, str]:
    """读取/补全单商品 OCR 缓存。缓存按 path 增量补全（支持后续扩大图集）。"""
    cache_path = config.STAGE_C / "ocr_text" / f"{pid}.json"
    cached = normalize_ocr_cache(read_json(cache_path, default={}) or {})
    missing = [ip for ip in ocr_images if os.path.exists(ip) and ip not in cached]
    for ip in missing:
        cached[ip] = ocr_image(ip)
    if missing:
        write_json(cache_path, cached)
    return {ip: cached.get(ip, "") for ip in ocr_images if os.path.exists(ip)}


def ocr_all_parallel(jobs: list[tuple[str, list[str]]]) -> None:
    """对 (pid, image_paths) 批量并行 OCR，结果写入各 product 的 ocr_text 缓存。
    jobs 里所有缺失的图统一进 ProcessPool 并发处理。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    cache_dir = config.STAGE_C / "ocr_text"
    caches: dict[str, dict] = {}
    todo_imgs: list[str] = []
    img_owner: dict[str, str] = {}
    for pid, imgs in jobs:
        c = normalize_ocr_cache(read_json(cache_dir / f"{pid}.json", default={}) or {})
        caches[pid] = c
        for ip in imgs:
            if os.path.exists(ip) and ip not in c and ip not in img_owner:
                img_owner[ip] = pid
                todo_imgs.append(ip)
    if not todo_imgs:
        return
    print(f"[C3] OCR 并行处理图片：{len(todo_imgs)} 张，workers={config.OCR_WORKERS}")
    done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=config.OCR_WORKERS, mp_context=ctx) as ex:
        for path, text in ex.map(_ocr_one, todo_imgs):
            caches[img_owner[path]][path] = text
            done += 1
            if done % 50 == 0:
                print(f"  [C3] OCR {done}/{len(todo_imgs)}")
    for pid, c in caches.items():
        write_json(cache_dir / f"{pid}.json", c)


def _parse_matches(arr, res: dict, fixed_image: str, ocr_texts: dict[str, str]):
    valid_imgs = set(ocr_texts.keys())
    if not isinstance(arr, list):
        return
    for item in arr:
        if not isinstance(item, dict):
            continue
        aid = item.get("attribute_id")
        if aid not in res:
            continue
        for m in item.get("matches", []) or []:
            rt = str(m.get("raw_text", "")).strip()
            if not rt:
                continue
            ip = fixed_image or m.get("image_path", "")
            if ip in valid_imgs and rt not in ocr_texts.get(ip, ""):
                ip = next((k for k, v in ocr_texts.items() if rt in v), "")
            elif ip not in valid_imgs:
                ip = next((k for k, v in ocr_texts.items() if rt in v), "")
            if not ip:
                continue
            res[aid].append({"raw_text": rt, "image_path": ip})


def extract_product(pid: str, acmt_p: dict, ocr_texts: dict[str, str]) -> dict:
    """按属性从 OCR 文本定向抽取。C3_PER_IMAGE=True 时逐图各发一次（抗稀释，
    适合全图 OCR）；否则把多图拼成一块发一次（成本低）。"""
    res = {aid: [] for aid in acmt_p}
    attrs_block = "\n".join(f"- {a} | {acmt_p[a].get('canonical_name', a)}" for a in acmt_p)
    nonempty = [(ip, txt) for ip, txt in ocr_texts.items() if txt.strip()]
    if not nonempty:
        return res

    if config.C3_PER_IMAGE:
        for ip, txt in nonempty:
            block = f"=== {ip} ===\n{txt}"[:config.OCR_TEXT_CAP]
            try:
                arr = llm.chat_json(
                    C3_PROMPT.format(attrs=attrs_block, ocr=block),
                    namespace="c3", max_tokens=1536,
                )
            except Exception:
                continue
            _parse_matches(arr, res, ip, ocr_texts)
        return res

    ocr_block = "\n".join(f"=== {ip} ===\n{txt}" for ip, txt in nonempty)[:config.OCR_TEXT_CAP]
    try:
        arr = llm.chat_json(
            C3_PROMPT.format(attrs=attrs_block, ocr=ocr_block),
            namespace="c3", max_tokens=2048,
        )
    except Exception:
        return res
    _parse_matches(arr, res, "", ocr_texts)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--image_index", default=str(config.STAGE_C / "image_index.json"))
    ap.add_argument("--out", default=str(config.STAGE_C / "evidence_ocr.json"))
    ap.add_argument("--rerun-empty", action="store_true",
                    help="Reprocess products whose cached OCR evidence is entirely empty.")
    ap.add_argument("--rerun-missing-attrs", action="store_true",
                    help="Reprocess products whose cached OCR evidence lacks requested attribute keys.")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    image_index = read_json(args.image_index, default={})
    bundles = pidx.build_bundles()

    pids = [p for p in acmt if p in image_index and p in bundles]
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
    print(f"[C3] products to OCR-extract: {len(todo)} (OCR_ALL={config.OCR_ALL_IMAGES})")

    def imgs_for(pid: str) -> list[str]:
        info = image_index.get(pid, {})
        if config.OCR_ALL_IMAGES:
            return [resolve_image_path(im["path"]) for im in info.get("images", [])]
        return [resolve_image_path(p) for p in info.get("ocr_images", [])]

    # 阶段1：全部缺失图并行 OCR（CPU 进程池），写入 ocr_text 缓存
    ocr_all_parallel([(pid, imgs_for(pid)) for pid in todo])

    # 阶段2：逐商品 LLM 定向抽取（线程并发）
    def _job(pid):
        imgs = imgs_for(pid)
        if not imgs:
            return pid, {aid: [] for aid in acmt[pid]}
        texts = get_product_ocr(pid, imgs)
        return pid, extract_product(pid, acmt[pid], texts)

    results = llm.run_many(todo, _job, desc="C3-extract")
    for r in results:
        if isinstance(r, tuple):
            out[r[0]] = r[1]
    write_json(args.out, out)
    hit = sum(1 for p in out.values() for lst in p.values() if lst)
    print(f"[C3] products={len(out)} attr-hits={hit} -> {args.out}")


if __name__ == "__main__":
    main()
