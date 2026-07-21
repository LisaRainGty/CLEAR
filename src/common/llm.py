"""matpool 网关 LLM/VLM 客户端（OpenAI 兼容）。

特性：
- 文本与多图视觉统一接口；
- 基于 (model, messages, params) 的磁盘缓存，断点续跑零重复花费；
- 指数退避重试；
- 线程池并发批处理 run_many；
- robust JSON 解析（容忍 ```json 包裹与前后噪声）。

仅依赖标准库 + openai（若安装）。未安装 openai 时回退到 urllib 直连。
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import config

_CACHE_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------
def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(key: str, namespace: str) -> Path:
    d = config.CACHE / namespace / key[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _cache_get(key: str, namespace: str):
    p = _cache_path(key, namespace)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_put(key: str, namespace: str, value: Any):
    p = _cache_path(key, namespace)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------
# 低层请求
# --------------------------------------------------------------------------
def _post(body: dict) -> dict:
    if not config.MATPOOL_API_KEY:
        raise RuntimeError("MATPOOL_API_KEY is not set in the process environment.")
    url = config.MATPOOL_BASE_URL.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Authorization": "Bearer " + config.MATPOOL_API_KEY,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_with_retry(body: dict) -> dict:
    last = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            return _post(body)
        except Exception as e:  # noqa: BLE001
            last = e
            wait = min(2 ** attempt, 30) + 0.5 * attempt
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} retries: {last!r}")


# --------------------------------------------------------------------------
# 图片
# --------------------------------------------------------------------------
def encode_image(path: str | Path, max_side: int | None = None) -> str | None:
    """读取图片 → data URL（可选缩放控成本）。失败返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    max_side = max_side or config.VISION_MAX_SIDE
    raw = path.read_bytes()
    mime = "image/jpeg"
    try:
        from PIL import Image  # type: ignore

        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        raw = buf.getvalue()
    except Exception:
        # 没装 PIL 或解码失败 → 原图直传
        if path.suffix.lower() == ".png":
            mime = "image/png"
        elif path.suffix.lower() == ".webp":
            mime = "image/webp"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


# --------------------------------------------------------------------------
# 高层接口
# --------------------------------------------------------------------------
def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    images: list[str] | None = None,
    temperature: float = 0.0,
    namespace: str = "chat",
    use_cache: bool = True,
    max_tokens: int | None = None,
) -> str:
    """单次对话，返回 assistant 文本。images 为 data URL 列表（视觉任务）。"""
    model = model or config.TEXT_MODEL
    content: Any
    if images:
        content = [{"type": "text", "text": prompt}]
        for u in images:
            content.append({"type": "image_url", "image_url": {"url": u}})
    else:
        content = prompt
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    body = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        body["max_tokens"] = max_tokens

    # 缓存键不含 base64 原图（太大），用图片内容哈希代替
    if images:
        img_sig = [hashlib.sha256(u.encode()).hexdigest()[:16] for u in images]
        ck_payload = {"model": model, "system": system, "prompt": prompt,
                      "images": img_sig, "temperature": temperature, "max_tokens": max_tokens}
    else:
        ck_payload = {"model": model, "system": system, "prompt": prompt,
                      "temperature": temperature, "max_tokens": max_tokens}
    key = _cache_key(ck_payload)

    if use_cache:
        cached = _cache_get(key, namespace)
        if cached is not None:
            return cached["text"]

    resp = _call_with_retry(body)
    text = resp["choices"][0]["message"]["content"] or ""
    if use_cache:
        _cache_put(key, namespace, {"text": text, "usage": resp.get("usage")})
    return text


def chat_json(prompt: str, **kw) -> Any:
    """对话并解析 JSON（容错）。失败抛 ValueError。"""
    txt = chat(prompt, **kw)
    return parse_json(txt)


def run_many(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    concurrency: int | None = None,
    desc: str = "",
) -> list[Any]:
    """并发执行 fn(item)。保持输入顺序返回。异常以 {'__error__': ...} 占位。"""
    items = list(items)
    concurrency = concurrency or config.LLM_CONCURRENCY
    results: list[Any] = [None] * len(items)

    def _wrap(idx_item):
        idx, it = idx_item
        try:
            return idx, fn(it)
        except Exception as e:  # noqa: BLE001
            return idx, {"__error__": repr(e)[:300]}

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for idx, res in ex.map(_wrap, enumerate(items)):
            results[idx] = res
            done += 1
            if desc and (done % 50 == 0 or done == len(items)):
                print(f"  [{desc}] {done}/{len(items)}", flush=True)
    return results


# --------------------------------------------------------------------------
# JSON 解析
# --------------------------------------------------------------------------
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """从模型输出中鲁棒地解析 JSON 对象/数组。"""
    if text is None:
        raise ValueError("empty text")
    s = text.strip()
    m = _JSON_FENCE.search(s)
    if m:
        s = m.group(1).strip()
    # 直接尝试
    try:
        return json.loads(s)
    except Exception:
        pass
    # 截取第一个 { 或 [ 到最后一个 } 或 ]
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    ends = [i for i in (s.rfind("}"), s.rfind("]")) if i >= 0]
    if starts and ends:
        cand = s[min(starts): max(ends) + 1]
        try:
            return json.loads(cand)
        except Exception:
            pass
    # 兜底：被 max_tokens 截断的 JSON 数组，逐个抢救已完整的顶层对象
    salvaged = _salvage_truncated_array(s)
    if salvaged is not None:
        return salvaged
    raise ValueError(f"cannot parse JSON from: {text[:200]!r}")


def _salvage_truncated_array(s: str):
    """从形如 '[ {..}, {..}, {截断' 的截断文本里，按括号配平抢救出已完整的对象列表。"""
    lb = s.find("[")
    if lb < 0:
        return None
    out = []
    depth = 0
    in_str = False
    esc = False
    obj_start = -1
    for i in range(lb + 1, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                try:
                    out.append(json.loads(s[obj_start:i + 1]))
                except Exception:
                    pass
                obj_start = -1
    return out if out else None


def ping(model: str | None = None) -> str:
    """连通性自检。"""
    return chat("只回复两个字：在线", model=model, use_cache=False, temperature=0)
