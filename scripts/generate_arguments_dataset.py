#!/usr/bin/env python3
"""Generate one leakage-safe arguments view for every frozen paper record.

The generator never receives labels, split names, sample roles, confidence
weights, consumer comments, or historical arguments.  Its only inputs are the
attribute, livestream claim, and the frozen PARAM/OCR/VLM source texts.  Raw
model responses are append-only and resumable; the final dataset is written
atomically only after every source row has a valid argument object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARGUMENT_FIELDS = ("supporting_argument", "refuting_argument", "evidence_gap")
SOURCE_FIELDS = (
    ("PARAM", "evidence_params", "raw_text"),
    ("OCR", "evidence_ocr", "raw_text"),
    ("VLM", "evidence_vlm", "raw_quote"),
)
FORBIDDEN_INPUT_FIELDS = (
    "y", "y_perception", "split", "sample_role", "c", "c_reliability",
    "confidence", "arguments", "_aligned_consumer_mentions",
    "_consumer_mentions_total", "_consumer_mentions_neg", "label_audit",
    "proposal_label_audit", "_llm_review",
)
FORBIDDEN_OUTPUT_TERMS = (
    "消费者评论", "用户评价", "评论", "评价", "样本标签", "训练标签",
    "y=", "y =",
)
INFERENCE_TERMS = ("可能", "有助于", "推测", "猜测", "意味着")
GAP_ASSERTION_TERMS = ("不符", "矛盾", "已证明", "证明了", "属实", "为假")
NO_SOURCE_ARGUMENTS = {
    "supporting_argument": "",
    "refuting_argument": "",
    "evidence_gap": (
        "未提供PARAM、OCR或VLM商品证据，无法判断直播声称是否获得客观支持"
        "或与商品事实矛盾。"
    ),
}
SYSTEM_PROMPT = (
    "你是商品事实证据审核助手。你只能使用用户提供的属性、直播声称和"
    "PARAM/OCR/VLM证据，不得假设未提供的事实。你不知道也不得推测样本标签、"
    "数据划分、消费者评论或训练元数据。你只做文本对齐，不使用背景常识"
    "解释数值、因果或功效。"
)
PROMPT_TEMPLATE = """\
仅根据下列内容生成简短、可核对的证据论证：
1. supporting_argument：只摘录或紧贴改写证据直接支持声称的部分；没有则为空字符串。
2. refuting_argument：只摘录或紧贴改写证据与声称直接矛盾的部分；没有则为空字符串。
3. evidence_gap：证据无法覆盖的声称、缺少的测试/认证/规格；没有则为空字符串。
生成前必须逐一检查换行或“---”分隔的每个子声称；任何未被支持或反驳
覆盖的子声称，必须在 evidence_gap 中点明。
每个字段最多120个中文字符；只保留与声称直接相关的最小证据片段，
不要复述整段证据。多个未覆盖的子声称应在 evidence_gap 中合并简写。
支持与反驳不得填写完全相同的文本；若同一证据对不同子声称有不同作用，
必须在两个字段中分别简写它与对应子声称的关系；如果无法分别说明，
只保留直接矛盾的 refuting_argument，supporting_argument 置空。
禁止根据常识解释指标含义，禁止把“可能”、“有助于”之类推测写入支持或反驳。
只有声称与证据的数值单位和测量维度都一致时，才能用数值差异作为反驳；
单位不同或测量维度不明时只能写入 evidence_gap，不得换算。
不要输出“真/假”、“正/负样本”或分类结果。不要引用消费者评论。
只输出以下三行，不要输出 Markdown、JSON 或其他解释：
<supporting_argument>...</supporting_argument>
<refuting_argument>...</refuting_argument>
<evidence_gap>...</evidence_gap>

[属性]
{attribute}
[直播声称]
{claim}
{evidence}
"""
RETRY_PROMPT = (
    "\n上一次输出未通过结构检查。请严格输出三个完整闭合标签，"
    "每个字段不超过120个中文字符，不要使用 Markdown、JSON 或附加解释。"
    "支持/反驳必须包含"
    "PARAM/OCR/VLM 中可直接定位的原文，不得复制只出现在直播"
    "声称中的内容。"
)
RETRY_GUIDANCE = {
    "are identical": (
        "上次支持与反驳完全相同。若无法用不同文字明确说明它们"
        "分别对应哪个子声称，supporting_argument 必须置空，只在"
        " refuting_argument 保留直接矛盾证据；绝不能再复制相同内容。"
    ),
    "no direct": (
        "支持/反驳必须原样包含来自 PARAM/OCR/VLM 的可定位连续文本；"
        "不能只改写直播声称。"
    ),
    "numbers absent": "删除所有未在原始证据中出现的数字。",
    "exceeds": "只保留最小证据片段，将每个字段压缩到120个中文字符以内。",
    "no complete": "立即结束长句，优先输出三个完整闭合标签。",
    "speculative": "删除可能、有助于、推测、猜测、意味着等推测词。",
    "asserts a conclusion": (
        "evidence_gap 只写缺少什么证据，不得写不符、矛盾、已证明、"
        "属实或为假等结论。"
    ),
}
VALIDATION_POLICY = "direct_source_anchor_numeric_faithfulness_no_dup_v2"
MAX_ARGUMENT_CHARS = 160


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifacts(path: Path, include_shard_hashes: bool) -> list[dict]:
    """Record enough local model provenance to identify the exact snapshot."""
    names = (
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "model.safetensors.index.json",
    )
    files = [path / name for name in names if (path / name).is_file()]
    files.extend(sorted(path.glob("*.safetensors")))
    out = []
    for item in files:
        record = {"path": item.name, "bytes": item.stat().st_size}
        if item.suffix != ".safetensors" or include_shard_hashes:
            record["sha256"] = sha256_file(item)
        out.append(record)
    return out


def runtime_manifest() -> dict:
    from importlib import metadata

    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "unavailable"

    block = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "modelscope": version("modelscope"),
    }
    try:
        import torch
        block["cuda_available"] = torch.cuda.is_available()
        block["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as exc:  # noqa: BLE001
        block["torch_probe_error"] = repr(exc)
    return block


def canonical_hash(value: object) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc


def claim_text(rec: dict) -> str:
    claim = rec.get("claim", {}) or {}
    passage = str(claim.get("passage", "") or "").strip()
    if passage:
        return passage
    return "\n".join(
        str(item.get("text", "") or "").strip()
        for item in claim.get("segments", []) or []
        if str(item.get("text", "") or "").strip()
    )


def allowed_payload(rec: dict) -> dict:
    payload = {
        "attribute_name": str(rec.get("attribute_name", "") or "").strip(),
        "claim": claim_text(rec),
    }
    for label, key, field in SOURCE_FIELDS:
        payload[label] = [
            str(item.get(field, "") or "").strip()
            for item in rec.get(key, []) or []
            if str(item.get(field, "") or "").strip()
        ]
    return payload


def has_source(payload: dict) -> bool:
    return any(payload[label] for label, _, _ in SOURCE_FIELDS)


def build_prompt(payload: dict, attempt: int = 1, failure_hint: str = "") -> str:
    blocks = []
    for label, _, _ in SOURCE_FIELDS:
        values = payload[label]
        body = "\n".join(f"- {text}" for text in values) if values else "- <无>"
        blocks.append(f"[{label}]\n{body}")
    prompt = PROMPT_TEMPLATE.format(
        attribute=payload["attribute_name"] or "<未命名属性>",
        claim=payload["claim"] or "<无可用直播声称>",
        evidence="\n".join(blocks),
    )
    if attempt > 1:
        prompt += RETRY_PROMPT
        prompt += f"\n当前是第{attempt}次确定性生成尝试。"
        if failure_hint:
            prompt += f"\n上一次校验失败原因：{failure_hint[:180]}"
            for marker, guidance in RETRY_GUIDANCE.items():
                if marker in failure_hint:
                    prompt += f"\n针对该错误的强制修正：{guidance}"
                    break
    return prompt


def validate_arguments(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("parsed response is not an object")
    out = {}
    for key in ARGUMENT_FIELDS:
        value = parsed.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"{key} is not a string")
        value = "" if value.strip() == "..." else value
        out[key] = " ".join(value.strip().split())
        if len(out[key]) > MAX_ARGUMENT_CHARS:
            raise ValueError(f"{key} exceeds {MAX_ARGUMENT_CHARS} characters")
    if not any(out.values()):
        raise ValueError("all argument fields are empty")
    joined = "\n".join(out.values()).lower()
    leaked = [term for term in FORBIDDEN_OUTPUT_TERMS if term.lower() in joined]
    if leaked:
        raise ValueError(f"forbidden output terms: {leaked}")
    return out


def extract_response(text: str) -> dict:
    cleaned = text.strip()
    tagged = {}
    for key in ARGUMENT_FIELDS:
        aliases = (key, "referring_argument") if key == "refuting_argument" else (key,)
        match = None
        for alias in aliases:
            match = re.search(
                rf"<{key}>\s*(.*?)\s*</{alias}>", cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                break
        if match:
            tagged[key] = match.group(1)
    if len(tagged) == len(ARGUMENT_FIELDS):
        return validate_arguments(tagged)

    # Backward-compatible fallback for append-only QA caches from the JSON prompt.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned,
                       flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no complete tagged fields or JSON object found")
        cleaned = cleaned[start:end + 1]
    return validate_arguments(json.loads(cleaned))


def normalized_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff%]+", "", text).lower()


def number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def grounded_in_sources(text: str, source_values: list[str]) -> bool:
    """Require a direct source anchor; claim-only restatement is not evidence."""
    arg = normalized_text(text)
    if not arg:
        return True
    sources = [normalized_text(value) for value in source_values]
    sources = [value for value in sources if value]
    if not sources:
        return False
    for source in sources:
        if len(source) >= 2 and (source in arg or arg in source):
            return True
        if len(source) >= 4 and any(source[index:index + 4] in arg
                                    for index in range(len(source) - 3)):
            return True
    return False


def validate_grounding(arguments: dict, payload: dict) -> dict:
    """Reject claim copying, unsupported numbers, and speculative arguments."""
    normalized = validate_arguments(arguments)
    source_values = [
        value for label, _, _ in SOURCE_FIELDS for value in payload.get(label, [])
    ]
    source_blob = "\n".join(source_values)
    support = normalized_text(normalized["supporting_argument"])
    refute = normalized_text(normalized["refuting_argument"])
    if support and support == refute:
        raise ValueError("supporting_argument and refuting_argument are identical")
    for key in ("supporting_argument", "refuting_argument"):
        text = normalized[key]
        if not text:
            continue
        inferred = [term for term in INFERENCE_TERMS if term in text]
        if inferred:
            raise ValueError(f"{key} contains speculative terms: {inferred}")
        unsupported_numbers = number_tokens(text) - number_tokens(source_blob)
        if unsupported_numbers:
            raise ValueError(
                f"{key} contains numbers absent from sources: {sorted(unsupported_numbers)}")
        if not grounded_in_sources(text, source_values):
            raise ValueError(f"{key} has no direct PARAM/OCR/VLM text anchor")
    gap_numbers = number_tokens(normalized["evidence_gap"])
    available_numbers = number_tokens(payload.get("claim", "") + "\n" + source_blob)
    if gap_numbers - available_numbers:
        raise ValueError("evidence_gap introduces numbers absent from claim and sources")
    asserted = [term for term in GAP_ASSERTION_TERMS
                if term in normalized["evidence_gap"]]
    if asserted:
        raise ValueError(f"evidence_gap asserts a conclusion: {asserted}")
    return normalized


def load_cache(path: Path) -> dict[tuple[str, str, str, str, str, str], dict]:
    cached = {}
    if not path.exists():
        return cached
    for row in read_jsonl(path):
        if row.get("status") == "ok" and isinstance(row.get("arguments"), dict):
            cached[(
                str(row.get("pair_id", "")), str(row.get("input_sha256", "")),
                str(row.get("prompt_sha256", "")), str(row.get("model_id", "")),
                str(row.get("model_revision", "")),
                str(row.get("generation_sha256", "")),
            )] = row
    return cached


def append_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def generate_batch(model, tokenizer, prompts: list[str], max_input_tokens: int,
                   max_new_tokens: int) -> list[str]:
    import torch

    messages = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}]
        for prompt in prompts
    ]
    rendered = [tokenizer.apply_chat_template(
        item, tokenize=False, add_generation_prompt=True) for item in messages]
    inputs = tokenizer(
        rendered, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def matpool_call(prompt: str, *, model_id: str, base_url: str,
                 max_new_tokens: int, timeout: int,
                 request_retries: int) -> dict:
    """Call the OpenAI-compatible gateway without persisting the API key."""
    api_key = os.environ.get("MATPOOL_API_KEY", "")
    if not api_key:
        return {"raw_text": "", "api_response": None,
                "error": "MATPOOL_API_KEY is not set", "fatal": True}
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    for request_attempt in range(1, request_retries + 1):
        request = urllib.request.Request(
            url, data=encoded,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("API response content is not a string")
            return {
                "raw_text": content,
                "api_response": payload,
                "api_response_sha256": canonical_hash(payload),
                "error": "",
                "fatal": False,
                "request_attempts": request_attempt,
            }
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")[:1000]
            error = f"HTTP {exc.code}: {response_body}"
            quota_markers = ("quota", "insufficient", "balance", "余额", "额度")
            fatal = (
                (400 <= exc.code < 500 and exc.code not in {408, 409, 425, 429})
                or any(marker in response_body.lower() for marker in quota_markers)
            )
            if fatal or request_attempt == request_retries:
                return {"raw_text": "", "api_response": None,
                        "error": error, "fatal": fatal,
                        "request_attempts": request_attempt}
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if request_attempt == request_retries:
                return {"raw_text": "", "api_response": None,
                        "error": error, "fatal": False,
                        "request_attempts": request_attempt}
        time.sleep(min(2 ** (request_attempt - 1), 20))
    raise AssertionError("unreachable")


def matpool_batch(prompts: list[str], *, model_id: str, base_url: str,
                  max_new_tokens: int, timeout: int,
                  request_retries: int, concurrency: int) -> list[dict]:
    def worker(prompt: str) -> dict:
        return matpool_call(
            prompt, model_id=model_id, base_url=base_url,
            max_new_tokens=max_new_tokens, timeout=timeout,
            request_retries=request_retries,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(worker, prompts))


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=(
        ROOT / "data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "data/arguments_only/dataset_arguments_only_20260722.jsonl"))
    parser.add_argument("--cache", type=Path, default=(
        ROOT / "data/arguments_only/raw_generations.jsonl"))
    parser.add_argument("--manifest", type=Path, default=(
        ROOT / "data/arguments_only/MANIFEST.json"))
    parser.add_argument("--backend", choices=("local", "matpool"), default="local")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--api-concurrency", type=int, default=4)
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-request-retries", type=int, default=5)
    parser.add_argument("--matpool-base-url", default=os.environ.get(
        "MATPOOL_BASE_URL", "https://token.matpool.com/v1"))
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N rows (smoke testing only).")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Process a deterministic random subset (QA only).")
    parser.add_argument("--sample-seed", type=int, default=20260722)
    parser.add_argument("--hash-model-shards", action="store_true",
                        help="Include SHA-256 for every model shard in the manifest.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load the fixed generator with bitsandbytes NF4 quantization.")
    args = parser.parse_args()

    if args.limit and args.sample_size:
        parser.error("--limit and --sample-size are mutually exclusive")
    if args.backend == "local" and not args.model_path:
        parser.error("--model-path is required for --backend local")
    if args.backend == "matpool" and not os.environ.get("MATPOOL_API_KEY"):
        parser.error("MATPOOL_API_KEY must be set in the process environment")
    if args.api_concurrency < 1:
        parser.error("--api-concurrency must be positive")

    source = args.source.resolve()
    output = args.output.resolve()
    cache_path = args.cache.resolve()
    manifest_path = args.manifest.resolve()
    all_rows = list(read_jsonl(source))
    rows = all_rows
    if args.limit:
        rows = rows[:args.limit]
    elif args.sample_size:
        if args.sample_size > len(rows):
            parser.error("--sample-size exceeds source row count")
        rng = random.Random(args.sample_seed)
        selected = sorted(rng.sample(range(len(rows)), args.sample_size))
        rows = [rows[index] for index in selected]
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    if not all(pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise RuntimeError("pair_id must be non-empty and unique")

    source_sha = sha256_file(source)
    prompt_sha = canonical_hash({
        "system": SYSTEM_PROMPT,
        "template": PROMPT_TEMPLATE,
        "retry": RETRY_PROMPT,
        "retry_guidance": RETRY_GUIDANCE,
        "validation_policy": VALIDATION_POLICY,
        "inference_terms": INFERENCE_TERMS,
        "gap_assertion_terms": GAP_ASSERTION_TERMS,
        "max_argument_chars": MAX_ARGUMENT_CHARS,
    })
    generation_sha = canonical_hash({
        "backend": args.backend,
        "model_id": args.model_id,
        "revision": args.revision,
        "base_url": args.matpool_base_url if args.backend == "matpool" else None,
        "load_in_4bit": args.load_in_4bit if args.backend == "local" else False,
        "quantization": (
            "bitsandbytes_nf4_double_bfloat16" if args.load_in_4bit
            else "bfloat16_or_float32") if args.backend == "local" else None,
        "temperature": 0.0,
        "provider_determinism_not_assumed": args.backend == "matpool",
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
    })
    payloads = {pair_id: allowed_payload(row) for pair_id, row in zip(pair_ids, rows)}
    input_shas = {pair_id: canonical_hash(payloads[pair_id]) for pair_id in pair_ids}
    cached = load_cache(cache_path)
    resolved: dict[str, dict] = {}
    modes: dict[str, str] = {}
    for pair_id in pair_ids:
        item = cached.get((pair_id, input_shas[pair_id], prompt_sha,
                           args.model_id, args.revision, generation_sha))
        if item:
            resolved[pair_id] = {key: str(item["arguments"].get(key, "") or "")
                                 for key in ARGUMENT_FIELDS}
            modes[pair_id] = str(item.get("mode", "model"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_handle:
        for pair_id in pair_ids:
            if pair_id in resolved or has_source(payloads[pair_id]):
                continue
            record = {
                "pair_id": pair_id,
                "input_sha256": input_shas[pair_id],
                "prompt_sha256": prompt_sha,
                "model_id": args.model_id,
                "model_revision": args.revision,
                "generation_sha256": generation_sha,
                "mode": "deterministic_no_source",
                "attempt": 0,
                "status": "ok",
                "payload": payloads[pair_id],
                "request_prompt": "",
                "request_prompt_sha256": text_hash(""),
                "raw_text": "",
                "api_response": None,
                "arguments": dict(NO_SOURCE_ARGUMENTS),
                "generated_utc": utc_now(),
            }
            append_record(cache_handle, record)
            resolved[pair_id] = dict(NO_SOURCE_ARGUMENTS)
            modes[pair_id] = "deterministic_no_source"

        pending = [pair_id for pair_id in pair_ids if pair_id not in resolved]
        if pending:
            model = tokenizer = None
            if args.backend == "local":
                import torch
                from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                          BitsAndBytesConfig)

                random.seed(0)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path, trust_remote_code=True)
                tokenizer.padding_side = "left"
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                device_map = {"": 0} if torch.cuda.is_available() else None
                model_kwargs = {
                    "torch_dtype": dtype,
                    "device_map": device_map,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }
                if args.load_in_4bit:
                    if not torch.cuda.is_available():
                        raise RuntimeError("--load-in-4bit requires a CUDA GPU")
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, **model_kwargs)
                model.eval()
                model.generation_config.do_sample = False
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None
            print(
                f"[load] backend={args.backend} model={args.model_id} "
                f"pending={len(pending)}", flush=True)

            failures: dict[str, str] = {}
            for attempt in range(1, args.max_retries + 1):
                current = [pair_id for pair_id in pending if pair_id not in resolved]
                if not current:
                    break
                for start in range(0, len(current), args.batch_size):
                    batch_ids = current[start:start + args.batch_size]
                    prompts = [build_prompt(
                        payloads[pair_id], attempt, failures.get(pair_id, ""),
                    ) for pair_id in batch_ids]
                    if args.backend == "local":
                        try:
                            generated = generate_batch(
                                model, tokenizer, prompts, args.max_input_tokens,
                                args.max_new_tokens)
                        except RuntimeError as exc:
                            if "out of memory" in str(exc).lower() and args.batch_size > 1:
                                raise RuntimeError(
                                    "CUDA OOM: rerun with a smaller --batch-size; "
                                    "cache is resumable"
                                ) from exc
                            raise
                        raw_results = [
                            {"raw_text": text, "api_response": None,
                             "error": "", "fatal": False,
                             "request_attempts": 1}
                            for text in generated
                        ]
                    else:
                        raw_results = matpool_batch(
                            prompts, model_id=args.model_id,
                            base_url=args.matpool_base_url,
                            max_new_tokens=args.max_new_tokens,
                            timeout=args.api_timeout,
                            request_retries=args.api_request_retries,
                            concurrency=args.api_concurrency,
                        )
                    fatal_errors = []
                    for pair_id, request_prompt, result in zip(
                            batch_ids, prompts, raw_results):
                        raw_text = str(result.get("raw_text", "") or "")
                        status, parsed, error = "ok", None, ""
                        if result.get("error"):
                            status, error = "api_error", str(result["error"])
                            failures[pair_id] = error
                            if result.get("fatal"):
                                fatal_errors.append(f"{pair_id}: {error}")
                        else:
                            try:
                                parsed = validate_grounding(
                                    extract_response(raw_text), payloads[pair_id])
                            except Exception as exc:  # noqa: BLE001
                                status, error = "invalid", repr(exc)
                                failures[pair_id] = error
                        record = {
                            "pair_id": pair_id,
                            "input_sha256": input_shas[pair_id],
                            "prompt_sha256": prompt_sha,
                            "model_id": args.model_id,
                            "model_revision": args.revision,
                            "generation_sha256": generation_sha,
                            "mode": "model",
                            "attempt": attempt,
                            "status": status,
                            "payload": payloads[pair_id],
                            "request_prompt": request_prompt,
                            "request_prompt_sha256": text_hash(request_prompt),
                            "raw_text": raw_text,
                            "api_response": result.get("api_response"),
                            "api_response_sha256": result.get("api_response_sha256"),
                            "request_attempts": result.get("request_attempts", 1),
                            "arguments": parsed,
                            "error": error,
                            "generated_utc": utc_now(),
                        }
                        append_record(cache_handle, record)
                        if parsed is not None:
                            resolved[pair_id] = parsed
                            modes[pair_id] = "model"
                            failures.pop(pair_id, None)
                    if fatal_errors:
                        raise RuntimeError(
                            "fatal MatPool API error; cache is resumable: "
                            + " | ".join(fatal_errors[:3]))
                    done = len(resolved)
                    print(f"[progress] resolved={done}/{len(rows)} attempt={attempt}",
                          flush=True)

    missing = [pair_id for pair_id in pair_ids if pair_id not in resolved]
    matching_cache = [
        record for record in read_jsonl(cache_path)
        if str(record.get("prompt_sha256", "")) == prompt_sha
        and str(record.get("generation_sha256", "")) == generation_sha
        and str(record.get("model_id", "")) == args.model_id
        and str(record.get("model_revision", "")) == args.revision
    ]
    api_usage = Counter()
    provider_models = set()
    provider_response_ids = set()
    for record in matching_cache:
        response = record.get("api_response") or {}
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                api_usage[key] += int(usage.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
        if response.get("model"):
            provider_models.add(str(response["model"]))
        if response.get("id"):
            provider_response_ids.add(str(response["id"]))
    estimated_api_cost = (
        api_usage["prompt_tokens"] * args.input_price_per_million
        + api_usage["completion_tokens"] * args.output_price_per_million
    ) / 1_000_000
    manifest = {
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "created_utc": utc_now(),
        "source": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "source_sha256": source_sha,
        "rows": len(rows),
        "source_rows": len(all_rows),
        "selection": {
            "mode": "first_n" if args.limit else (
                "random_sample" if args.sample_size else "all"),
            "limit": args.limit,
            "sample_size": args.sample_size,
            "sample_seed": args.sample_seed if args.sample_size else None,
        },
        "prompt_sha256": prompt_sha,
        "validation_policy": VALIDATION_POLICY,
        "generator": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
            "runtime": runtime_manifest(),
        },
        "model": {
            "backend": args.backend,
            "id": args.model_id,
            "revision": args.revision,
            "path_at_generation": args.model_path if args.backend == "local" else None,
            "base_url": args.matpool_base_url if args.backend == "matpool" else None,
            "api_key_present": bool(os.environ.get("MATPOOL_API_KEY"))
            if args.backend == "matpool" else None,
            "deterministic_decode": args.backend == "local",
            "provider_determinism_not_assumed": args.backend == "matpool",
            "temperature": 0.0,
            "generation_sha256": generation_sha,
            "load_in_4bit": args.load_in_4bit if args.backend == "local" else False,
            "quantization": (
                "bitsandbytes_nf4_double_bfloat16" if args.load_in_4bit
                else "bfloat16_or_float32") if args.backend == "local" else None,
            "batch_size": args.batch_size,
            "api_concurrency": args.api_concurrency
            if args.backend == "matpool" else None,
            "api_timeout_seconds": args.api_timeout
            if args.backend == "matpool" else None,
            "api_request_retries": args.api_request_retries
            if args.backend == "matpool" else None,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "artifacts": (
                model_artifacts(Path(args.model_path).resolve(),
                                args.hash_model_shards)
                if args.backend == "local" else []),
            "provider_reported_models": sorted(provider_models),
            "provider_response_ids_sha256": canonical_hash(
                sorted(provider_response_ids)),
            "api_usage": dict(api_usage),
            "api_calls": sum(
                record.get("api_response") is not None for record in matching_cache),
            "api_error_records": sum(
                record.get("status") == "api_error" for record in matching_cache),
            "pricing_yuan_per_million_tokens": {
                "input": args.input_price_per_million,
                "output": args.output_price_per_million,
                "source": "https://www.matpool.com/models",
            } if args.backend == "matpool" else None,
            "estimated_cost_yuan_from_usage": round(estimated_api_cost, 6)
            if args.backend == "matpool" else None,
        },
        "input_contract": {
            "allowed": ["attribute_name", "claim", "PARAM", "OCR", "VLM"],
            "forbidden": list(FORBIDDEN_INPUT_FIELDS),
            "historical_arguments_reused": False,
        },
        "counts": {
            "resolved": len(resolved),
            "model_generated": sum(mode == "model" for mode in modes.values()),
            "deterministic_no_source": sum(
                mode == "deterministic_no_source" for mode in modes.values()),
            "missing": len(missing),
        },
        "missing_pair_ids": missing,
        "cache": str(cache_path.relative_to(ROOT)
                     if cache_path.is_relative_to(ROOT) else cache_path),
    }
    if missing:
        write_manifest(manifest_path, manifest)
        print(f"[incomplete] unresolved={len(missing)}; resume with the same command",
              flush=True)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    field_coverage = Counter()
    with tmp.open("w", encoding="utf-8") as handle:
        for pair_id, rec in zip(pair_ids, rows):
            arguments = resolved[pair_id]
            for key in ARGUMENT_FIELDS:
                field_coverage[key] += int(bool(arguments[key]))
            rec["arguments"] = arguments
            rec["_argument_generation"] = {
                "backend": args.backend,
                "model_id": args.model_id,
                "model_revision": args.revision,
                "prompt_sha256": prompt_sha,
                "input_sha256": input_shas[pair_id],
                "generation_sha256": generation_sha,
                "mode": modes[pair_id],
                "label_blind": True,
            }
            handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(output)
    manifest["output"] = str(output.relative_to(ROOT)
                             if output.is_relative_to(ROOT) else output)
    manifest["output_sha256"] = sha256_file(output)
    manifest["argument_field_coverage"] = dict(field_coverage)
    write_manifest(manifest_path, manifest)
    print(json.dumps({"status": "complete", "output": str(output),
                      "sha256": manifest["output_sha256"],
                      "counts": manifest["counts"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
