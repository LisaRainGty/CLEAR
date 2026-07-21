"""SRT/字幕解析：支持 .srt 与 .txt（同格式）。

提供：
- parse_srt: 解析单文件为 cue 列表（含时间戳）；
- concat_product_srt: 把一个商品名下多个 clip 的 SRT 升序拼接为长文本，
  并返回 cue 边界表（字符偏移 -> srt_file/cue_idx/start_ts/end_ts），
  供 Stage B1 把 LangExtract/LLM 的 char_interval 反查为时间戳。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import config

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_TS_LINE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def ts_to_seconds(ts: str) -> float:
    m = _TS.search(ts)
    if not m:
        return 0.0
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


@dataclass
class Cue:
    idx: int
    start_ts: str
    end_ts: str
    text: str


def parse_srt(path: str | Path) -> list[Cue]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="gb18030", errors="ignore")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: list[Cue] = []
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        # 找时间戳行
        ts_i = next((i for i, ln in enumerate(lines) if _TS_LINE.search(ln)), None)
        if ts_i is None:
            continue
        m = _TS_LINE.search(lines[ts_i])
        text = " ".join(ln.strip() for ln in lines[ts_i + 1:]).strip()
        if not text:
            continue
        cues.append(Cue(idx=len(cues), start_ts=m.group(1), end_ts=m.group(2), text=text))
    return cues


@dataclass
class CueSpan:
    char_start: int
    char_end: int
    srt_file: str
    cue_idx: int
    start_ts: str
    end_ts: str


@dataclass
class ConcatResult:
    text: str
    spans: list[CueSpan] = field(default_factory=list)

    def lookup(self, char_start: int, char_end: int) -> CueSpan | None:
        """给定字符区间，返回与之重叠最多的 cue span。"""
        best, best_ov = None, -1
        for sp in self.spans:
            ov = min(char_end, sp.char_end) - max(char_start, sp.char_start)
            if ov > best_ov:
                best_ov, best = ov, sp
        return best

    def lookup_range(self, char_start: int, char_end: int) -> list[CueSpan]:
        """Return all cue spans overlapped by a source character interval."""
        out = []
        for sp in self.spans:
            ov = min(char_end, sp.char_end) - max(char_start, sp.char_start)
            if ov > 0:
                out.append(sp)
        return out


def concat_product_srt(srt_files: list[str | Path]) -> ConcatResult:
    """把多个 SRT 文件升序拼接为单一长文本 + cue 边界表。

    clip 之间插入 CLIP_BREAK 分隔符；每条 cue 文本之间用换行连接，
    并记录每条 cue 文本在长文本中的字符偏移区间。
    """
    parts: list[str] = []
    spans: list[CueSpan] = []
    pos = 0
    files = sorted(str(f) for f in srt_files)
    for fi, f in enumerate(files):
        if fi > 0 or True:
            brk = config.CLIP_BREAK.format(srt_file=Path(f).name)
            parts.append(brk)
            pos += len(brk)
        for cue in parse_srt(f):
            seg = cue.text + "\n"
            spans.append(CueSpan(
                char_start=pos, char_end=pos + len(cue.text),
                srt_file=f, cue_idx=cue.idx,
                start_ts=cue.start_ts, end_ts=cue.end_ts,
            ))
            parts.append(seg)
            pos += len(seg)
    return ConcatResult(text="".join(parts), spans=spans)
