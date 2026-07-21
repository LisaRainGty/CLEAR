"""CLAIMARC 模型输入：双流 tokenization（§3.3 / §3.4）。

读取 data/dataset_duallabel_*.jsonl，按 §3 把每条 (product, attribute) pair 切成两条 token 流：
  Claim Flow  X^c : [CLS][ATTR]{attr} [CLM]{seg} [SEP_C] ... [SEP]
  Evidence Flow X^e: [CLS][ATTR]{attr}[EVD] [PARAM]{..}[SEP_E].. [OCR]{..} [VLM]{..}
新增 6 类 special token，随机初始化随训练更新。

提供 build_tokenizer()、ClaimDataset、collate、make_loaders（按 room_id grouped split）。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = [
    "[ATTR]", "[CLM]", "[CLM_NULL]", "[EVD]",
    "[PARAM]", "[OCR]", "[VLM]", "[SEP_C]", "[SEP_E]",
    "[ARG_SUP]", "[ARG_REF]", "[ARG_GAP]",
]

L_C = 384
L_E = 384


def resolve_bge_path(name: str = "BAAI/bge-large-zh-v1.5") -> str:
    """优先本地目录 / 环境变量，其次 ModelScope 缓存，否则 HF 名。"""
    import os
    if os.path.isdir(name):
        return name
    env_path = os.environ.get("CLAIMARC_BGE_PATH", "")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from modelscope import snapshot_download
        return snapshot_download(name.replace("BAAI/", "AI-ModelScope/"))
    except Exception:
        return name


def build_tokenizer(bge_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bge_path)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text or "", add_special_tokens=False)


def build_claim_ids(tok, rec: dict) -> list[int]:
    sp = tok.convert_tokens_to_ids
    cls, sep = tok.cls_token_id, tok.sep_token_id
    out = [cls, sp("[ATTR]")] + _ids(tok, rec.get("attribute_name", ""))
    claim = rec.get("claim", {}) or {}
    segs = claim.get("segments", []) or []
    if not claim.get("has_claim_srt") or not segs:
        out += [sp("[CLM_NULL]"), sep]
        return out[:L_C]
    for s in segs:
        out += [sp("[CLM]")] + _ids(tok, s.get("text", "")) + [sp("[SEP_C]")]
        if len(out) > L_C:
            break
    out += [sep]
    return out[:L_C]


def _argument_blocks(rec: dict):
    args = rec.get("arguments", {}) or {}
    for src_tok, key in (
        ("[ARG_SUP]", "supporting_argument"),
        ("[ARG_REF]", "refuting_argument"),
        ("[ARG_GAP]", "evidence_gap"),
    ):
        txt = args.get(key, "")
        if txt:
            yield src_tok, [txt]


def _source_blocks(rec: dict, only: set[str] | None = None):
    for src_tok, key, field, name in (
        ("[PARAM]", "evidence_params", "raw_text", "params"),
        ("[OCR]", "evidence_ocr", "raw_text", "ocr"),
        ("[VLM]", "evidence_vlm", "raw_quote", "vlm"),
    ):
        if only is not None and name not in only:
            continue
        texts = [it.get(field, "") for it in (rec.get(key, []) or []) if it.get(field, "")]
        if texts:
            yield src_tok, texts


def build_evidence_ids(tok, rec: dict, policy_override: str | None = None) -> list[int]:
    sp = tok.convert_tokens_to_ids
    cls, sep = tok.cls_token_id, tok.sep_token_id
    out = [cls, sp("[ATTR]")] + _ids(tok, rec.get("attribute_name", "")) + [sp("[EVD]")]

    if policy_override and policy_override != "record":
        policy = policy_override
    else:
        # Paper-fair default: three product sources only (params/OCR/VLM).
        policy = rec.get("_evidence_policy") or rec.get("evidence_policy") or "sources_only"
    source_blocks = list(_source_blocks(rec))
    argument_blocks = list(_argument_blocks(rec))
    blocks = []
    if policy == "source_first":
        # Retired for paper runs; keep path but prefer sources for fairness.
        blocks = source_blocks
    elif policy in ("no_args", "source_only", "sources_only", "args_first", "", None):
        blocks = source_blocks
    elif policy == "args_only":
        # Kept for legacy; paper campaign no longer schedules this.
        blocks = argument_blocks if argument_blocks else source_blocks
    elif policy == "params_only":
        blocks = list(_source_blocks(rec, {"params"}))
    elif policy == "ocr_only":
        blocks = list(_source_blocks(rec, {"ocr"}))
    elif policy == "vlm_only":
        blocks = list(_source_blocks(rec, {"vlm"}))
    elif policy == "params_args":
        blocks = list(_source_blocks(rec, {"params"}))
    elif policy == "ocr_args":
        blocks = list(_source_blocks(rec, {"ocr"}))
    elif policy == "vlm_args":
        blocks = list(_source_blocks(rec, {"vlm"}))
    else:
        raise ValueError(f"unknown evidence_policy: {policy}")

    for src_tok, texts in blocks:
        out += [sp(src_tok)]
        for txt in texts:
            out += _ids(tok, txt) + [sp("[SEP_E]")]
            if len(out) > L_E:
                break
        if len(out) > L_E:
            break
    out += [sep]
    return out[:L_E]


@dataclass
class Batch:
    c_ids: torch.Tensor
    c_mask: torch.Tensor
    e_ids: torch.Tensor
    e_mask: torch.Tensor
    e_view_ids: torch.Tensor | None
    e_view_mask: torch.Tensor | None
    y: torch.Tensor
    c: torch.Tensor
    teacher_p: torch.Tensor
    contrastive_mask: torch.Tensor
    source_count: torch.Tensor
    source_len: torch.Tensor
    arg_len: torch.Tensor
    evidence_combo: list
    confidence: list
    attr: list
    pair_id: list


def source_count(rec: dict) -> int:
    ev = rec.get("evidence_count", {}) or {}
    if isinstance(ev, dict):
        return (
            int(ev.get("params", 0) or 0)
            + int(ev.get("ocr", 0) or 0)
            + int(ev.get("vlm", 0) or 0)
        )
    try:
        return int(ev)
    except Exception:
        return (
            len(rec.get("evidence_params") or [])
            + len(rec.get("evidence_ocr") or [])
            + len(rec.get("evidence_vlm") or [])
        )


def source_len(rec: dict) -> int:
    total = 0
    for key, field in (
        ("evidence_params", "raw_text"),
        ("evidence_ocr", "raw_text"),
        ("evidence_vlm", "raw_quote"),
    ):
        for item in rec.get(key, []) or []:
            total += len(str(item.get(field, "") or ""))
    return total


def arg_len(rec: dict) -> int:
    args = rec.get("arguments", {}) or {}
    return sum(len(str(args.get(k, "") or "")) for k in
               ("supporting_argument", "refuting_argument", "evidence_gap"))


def _nonempty(value) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def evidence_combo(rec: dict) -> str:
    parts = []
    if _nonempty(rec.get("evidence_params")):
        parts.append("P")
    if _nonempty(rec.get("evidence_ocr")):
        parts.append("O")
    if _nonempty(rec.get("evidence_vlm")):
        parts.append("V")
    return "".join(parts) if parts else "none"


def confidence_bin(rec: dict) -> str:
    conf = str(rec.get("confidence", "") or "").strip()
    if conf:
        return conf
    c = float(rec.get("c", 0.0) or 0.0)
    if c < 0.20:
        return "absent"
    if c < 0.40:
        return "low"
    if c < 0.70:
        return "medium"
    return "high"


class ClaimDataset(Dataset):
    def __init__(self, records: list[dict], tok, evidence_policy_mix: list[str] | None = None,
                 evidence_consistency_mix: list[str] | None = None):
        self.recs = records
        self.tok = tok
        self.evidence_policy_mix = [p for p in (evidence_policy_mix or []) if p]
        self.evidence_consistency_mix = [p for p in (evidence_consistency_mix or []) if p]

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        policy = random.choice(self.evidence_policy_mix) if self.evidence_policy_mix else None
        item = {
            "c_ids": build_claim_ids(self.tok, r),
            "e_ids": build_evidence_ids(self.tok, r, policy_override=policy),
            "y": float(r.get("y", 0)),
            "c": float(r.get("c", 0.05)),
            "teacher_p": float(r.get("_teacher_p", -1.0)),
            "contrastive_mask": float(bool(r.get("contrastive_mask", True))),
            "source_count": float(source_count(r)),
            "source_len": float(source_len(r)),
            "arg_len": float(arg_len(r)),
            "evidence_combo": evidence_combo(r),
            "confidence": confidence_bin(r),
            "attr": r.get("attribute_id", ""),
            "pair_id": r.get("pair_id", ""),
        }
        if self.evidence_consistency_mix:
            base_policy = policy or r.get("_evidence_policy", r.get("evidence_policy", "args_first"))
            choices = [p for p in self.evidence_consistency_mix if p != base_policy]
            aux_policy = random.choice(choices or self.evidence_consistency_mix)
            item["e_view_ids"] = build_evidence_ids(self.tok, r, policy_override=aux_policy)
        return item


def make_collate(pad_id: int, stream_mode: str = "dual"):
    """stream_mode：
      dual     → 默认双流（claim 流 + evidence 流）。
      claim    → 单流消融：evidence 流镜像为 claim 流（仅主播话术进入两个编码器槽）。
      evidence → 单流消融：claim 流镜像为 evidence 流（仅证据进入两个编码器槽）。
    单流变体需配合 --no_fusion 使用，使两次编码完全等价、不存在跨流交互。"""
    def collate(items) -> Batch:
        def pad(key):
            seqs = [it[key] for it in items]
            m = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), m), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[j, :len(s)] = 1
            return ids, mask
        c_ids, c_mask = pad("c_ids")
        e_ids, e_mask = pad("e_ids")
        e_view_ids = e_view_mask = None
        if any("e_view_ids" in it for it in items):
            e_view_ids, e_view_mask = pad("e_view_ids")
        if stream_mode == "claim":
            e_ids, e_mask = c_ids, c_mask
            e_view_ids = e_view_mask = None
        elif stream_mode == "evidence":
            c_ids, c_mask = e_ids, e_mask
            e_view_ids = e_view_mask = None
        elif stream_mode != "dual":
            raise ValueError(f"unknown stream_mode: {stream_mode}")
        return Batch(
            c_ids=c_ids, c_mask=c_mask, e_ids=e_ids, e_mask=e_mask,
            e_view_ids=e_view_ids, e_view_mask=e_view_mask,
            y=torch.tensor([it["y"] for it in items], dtype=torch.float),
            c=torch.tensor([it["c"] for it in items], dtype=torch.float),
            teacher_p=torch.tensor([it["teacher_p"] for it in items], dtype=torch.float),
            contrastive_mask=torch.tensor([it["contrastive_mask"] for it in items], dtype=torch.float),
            source_count=torch.tensor([it["source_count"] for it in items], dtype=torch.float),
            source_len=torch.tensor([it["source_len"] for it in items], dtype=torch.float),
            arg_len=torch.tensor([it["arg_len"] for it in items], dtype=torch.float),
            evidence_combo=[it["evidence_combo"] for it in items],
            confidence=[it["confidence"] for it in items],
            attr=[it["attr"] for it in items],
            pair_id=[it["pair_id"] for it in items],
        )
    return collate


def load_split(dataset_path: str) -> dict[str, list[dict]]:
    by = {"train": [], "val": [], "test": []}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("y") is None:
                continue
            split = r.get("split") or "train"
            by.get(split, by["train"]).append(r)
    return by
