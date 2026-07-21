"""通用 IO：jsonl 读写、评论 .xls 读取、文本归一化、Jaccard。"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# 文本归一化
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s\-_/()（）·．。，,、:：;；'\"“”‘’!！?？]+")


def normalize(s: str) -> str:
    """归一化：全半角统一、小写、去标点空白。用于 key/alias 匹配。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = _PUNCT.sub("", s)
    return s


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def bigram_jaccard(a: str, b: str) -> float:
    def grams(x: str) -> set[str]:
        x = re.sub(r"\s+", "", x)
        return {x[i:i + 2] for i in range(len(x) - 1)} if len(x) >= 2 else {x}
    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


# --------------------------------------------------------------------------
# 评论 .xls 读取（飞瓜导出，OLE2/BIFF）
# --------------------------------------------------------------------------
COMMENT_COLS = ["评论内容", "评论时间", "评论情感", "来自商品", "来源小店", "商品链接"]
POLARITY_MAP = {"好评": "pos", "中评": "neu", "差评": "neg"}


def read_comment_xls(path: str | Path) -> list[dict]:
    """读取一个评论 xls，返回规范化记录列表。

    优先 pandas+xlrd；失败再尝试 openpyxl（.xlsx）/ html 表（部分导出实为 html）。
    每条返回：{comment_id, text, time, sentiment(好评/中评/差评), polarity(pos/neu/neg)}
    """
    path = Path(path)
    rows = _load_table(path)
    out: list[dict] = []
    for i, r in enumerate(rows):
        text = str(r.get("评论内容", "") or "").strip()
        if not text:
            continue
        senti = str(r.get("评论情感", "") or "").strip()
        out.append({
            "comment_id": f"{path.stem}#{i}",
            "text": text,
            "time": str(r.get("评论时间", "") or "").strip(),
            "sentiment": senti,
            "polarity": POLARITY_MAP.get(senti, "neu"),
            "shop": str(r.get("来源小店", "") or "").strip(),
        })
    return out


def _load_table(path: Path) -> list[dict]:
    # 1) pandas
    try:
        import pandas as pd  # type: ignore
        try:
            df = pd.read_excel(path)  # 需要 xlrd(.xls) / openpyxl(.xlsx)
        except Exception:
            # 部分飞瓜 "xls" 实为 html 表
            df = pd.read_html(path)[0]
        df.columns = [str(c).strip() for c in df.columns]
        return df.to_dict("records")
    except Exception:
        pass
    # 2) 纯 xlrd（无 pandas 环境兜底）
    try:
        import xlrd  # type: ignore
        book = xlrd.open_workbook(str(path))
        sh = book.sheet_by_index(0)
        header = [str(sh.cell_value(0, c)).strip() for c in range(sh.ncols)]
        recs = []
        for r in range(1, sh.nrows):
            recs.append({header[c]: sh.cell_value(r, c) for c in range(sh.ncols)})
        return recs
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"无法读取评论文件 {path}: {e!r}")
