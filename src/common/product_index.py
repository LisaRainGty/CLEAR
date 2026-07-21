"""加载 product_index.json，并按 product_id 聚合三类原始数据。

一个 product_id 可出现在多个 clip（多场直播）。本模块负责：
- 解析 clips / products；
- 为每个 product_id 汇总：所属品类、room_id 集合、SRT 文件列表、评论文件列表、产品参数、图片路径；
- 提供路径解析（索引里是相对 ROOT 的路径）。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import config
from common.io_utils import read_json


def resolve(path_str: str) -> Path:
    """索引中的路径以 ROOT 为基准（形如 data/raw/...）；兼容绝对路径。

    一些早期 Stage C 产物是在远端机器生成的，绝对路径前缀可能是
    /root/claimarc。若该绝对路径在当前机器不存在，但能定位到其中的
    data/... 相对段，则重映射到当前 CLAIMARC_ROOT。
    """
    if not path_str:
        return Path()
    p = Path(path_str)
    if p.is_absolute():
        if p.exists():
            return p
        parts = p.parts
        if "data" in parts:
            i = parts.index("data")
            cand = config.ROOT.joinpath(*parts[i:])
            if cand.exists():
                return cand
        return p
    return config.ROOT / p


@dataclass
class ProductBundle:
    product_id: str
    category: str = ""            # 一级品类（直播间一级分类）
    subcategory: str = ""
    rooms: set[str] = field(default_factory=set)      # 直播间名称集合
    title: str = ""
    params: dict = field(default_factory=dict)        # 产品参数
    srt_files: list[str] = field(default_factory=list)
    comment_files: list[str] = field(default_factory=list)
    main_image: str = ""
    detail_images: list[str] = field(default_factory=list)

    @property
    def room_id(self) -> str:
        # 划分用单一 room_id：取字典序最小（绝大多数商品仅属一个直播间）
        return sorted(self.rooms)[0] if self.rooms else "UNKNOWN"


@lru_cache(maxsize=1)
def load_index() -> dict:
    return read_json(config.PRODUCT_INDEX, default={"clips": [], "products": {}})


@lru_cache(maxsize=1)
def build_bundles() -> dict[str, ProductBundle]:
    """构造 product_id -> ProductBundle。仅纳入在 clips 中被引用、且能定位到 products 的商品。"""
    idx = load_index()
    products = idx.get("products", {})
    bundles: dict[str, ProductBundle] = {}

    for clip in idx.get("clips", []):
        pid = str(clip.get("product_id") or "").strip()
        if not pid:
            continue
        b = bundles.get(pid)
        if b is None:
            b = ProductBundle(product_id=pid)
            bundles[pid] = b
        b.category = clip.get("直播间一级分类", b.category) or b.category
        b.subcategory = clip.get("直播间二级分类", b.subcategory) or b.subcategory
        room = clip.get("直播间名称")
        if room:
            b.rooms.add(room)
        srt = clip.get("srt切片存储路径")
        if srt and srt not in b.srt_files:
            b.srt_files.append(srt)
        cmt = clip.get("商品评价存储路径")
        if cmt and cmt not in b.comment_files:
            b.comment_files.append(cmt)

    # 补充 products 字段（参数、图片、标题）
    for pid, b in bundles.items():
        prod = products.get(pid, {})
        b.title = prod.get("title") or prod.get("商品名称") or ""
        b.params = prod.get("产品参数", {}) or {}
        imgs = prod.get("images", {}) or {}
        main = (imgs.get("主图") or {})
        b.main_image = main.get("本地路径", "") if isinstance(main, dict) else ""
        for d in (imgs.get("详情图") or []):
            if isinstance(d, dict) and d.get("本地路径"):
                b.detail_images.append(d["本地路径"])
    return bundles


def bundles_by_category() -> dict[str, list[ProductBundle]]:
    by = defaultdict(list)
    for b in build_bundles().values():
        by[b.category].append(b)
    return by


def all_categories() -> list[str]:
    return sorted({b.category for b in build_bundles().values() if b.category})


if __name__ == "__main__":
    bs = build_bundles()
    print(f"products in clips: {len(bs)}")
    print(f"categories: {all_categories()}")
    multi = sum(1 for b in bs.values() if len(b.srt_files) > 1)
    print(f"products with >1 srt: {multi}")
    sample = next(iter(bs.values()))
    print("sample:", sample.product_id, sample.category, sample.room_id,
          "| srt:", len(sample.srt_files), "| cmt:", len(sample.comment_files),
          "| params:", len(sample.params), "| detail_imgs:", len(sample.detail_images))
