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
import random
import re
from collections import Counter
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
禁止根据常识解释指标含义，禁止把“可能”、“有助于”之类推测写入支持或反驳。
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


def canonical_hash(value: object) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


def build_prompt(payload: dict, attempt: int = 1) -> str:
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
        prompt += (
            "\n上一次输出未通过结构检查。请严格输出三个完整闭合标签，"
            "不要使用 Markdown、JSON 或附加解释。"
        )
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


def load_cache(path: Path) -> dict[tuple[str, str], dict]:
    cached = {}
    if not path.exists():
        return cached
    for row in read_jsonl(path):
        if row.get("status") == "ok" and isinstance(row.get("arguments"), dict):
            cached[(str(row.get("pair_id", "")), str(row.get("input_sha256", "")))] = row
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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N rows (smoke testing only).")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Process a deterministic random subset (QA only).")
    parser.add_argument("--sample-seed", type=int, default=20260722)
    parser.add_argument("--hash-model-shards", action="store_true",
                        help="Include SHA-256 for every model shard in the manifest.")
    args = parser.parse_args()

    if args.limit and args.sample_size:
        parser.error("--limit and --sample-size are mutually exclusive")

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
    prompt_sha = canonical_hash({"system": SYSTEM_PROMPT, "template": PROMPT_TEMPLATE})
    payloads = {pair_id: allowed_payload(row) for pair_id, row in zip(pair_ids, rows)}
    input_shas = {pair_id: canonical_hash(payloads[pair_id]) for pair_id in pair_ids}
    cached = load_cache(cache_path)
    resolved: dict[str, dict] = {}
    modes: dict[str, str] = {}
    for pair_id in pair_ids:
        item = cached.get((pair_id, input_shas[pair_id]))
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
                "mode": "deterministic_no_source",
                "attempt": 0,
                "status": "ok",
                "payload": payloads[pair_id],
                "raw_text": "",
                "arguments": dict(NO_SOURCE_ARGUMENTS),
                "generated_utc": utc_now(),
            }
            append_record(cache_handle, record)
            resolved[pair_id] = dict(NO_SOURCE_ARGUMENTS)
            modes[pair_id] = "deterministic_no_source"

        pending = [pair_id for pair_id in pair_ids if pair_id not in resolved]
        if pending:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

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
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, device_map=device_map,
                low_cpu_mem_usage=True, trust_remote_code=True,
            )
            model.eval()
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            print(f"[load] model={args.model_id} pending={len(pending)}", flush=True)

            failures: dict[str, str] = {}
            for attempt in range(1, args.max_retries + 1):
                current = [pair_id for pair_id in pending if pair_id not in resolved]
                if not current:
                    break
                for start in range(0, len(current), args.batch_size):
                    batch_ids = current[start:start + args.batch_size]
                    prompts = [build_prompt(payloads[pair_id], attempt) for pair_id in batch_ids]
                    try:
                        raw_outputs = generate_batch(
                            model, tokenizer, prompts, args.max_input_tokens,
                            args.max_new_tokens)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and args.batch_size > 1:
                            raise RuntimeError(
                                "CUDA OOM: rerun with a smaller --batch-size; cache is resumable"
                            ) from exc
                        raise
                    for pair_id, raw_text in zip(batch_ids, raw_outputs):
                        status, parsed, error = "ok", None, ""
                        try:
                            parsed = extract_response(raw_text)
                        except Exception as exc:  # noqa: BLE001
                            status, error = "invalid", repr(exc)
                            failures[pair_id] = error
                        record = {
                            "pair_id": pair_id,
                            "input_sha256": input_shas[pair_id],
                            "prompt_sha256": prompt_sha,
                            "model_id": args.model_id,
                            "model_revision": args.revision,
                            "mode": "model",
                            "attempt": attempt,
                            "status": status,
                            "payload": payloads[pair_id],
                            "raw_text": raw_text,
                            "arguments": parsed,
                            "error": error,
                            "generated_utc": utc_now(),
                        }
                        append_record(cache_handle, record)
                        if parsed is not None:
                            resolved[pair_id] = parsed
                            modes[pair_id] = "model"
                            failures.pop(pair_id, None)
                    done = len(resolved)
                    print(f"[progress] resolved={done}/{len(rows)} attempt={attempt}",
                          flush=True)

    missing = [pair_id for pair_id in pair_ids if pair_id not in resolved]
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
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "path_at_generation": args.model_path,
            "deterministic_decode": True,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "artifacts": model_artifacts(Path(args.model_path).resolve(),
                                         args.hash_model_shards),
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
                "model_id": args.model_id,
                "model_revision": args.revision,
                "prompt_sha256": prompt_sha,
                "input_sha256": input_shas[pair_id],
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
