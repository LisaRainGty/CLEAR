"""CLAIMARC 训练与评估（严格对齐 §3.2.7–§3.2.9, §4.3）。

- Attribute-Blocked RACL：内存库 g 向量按 (attribute, y) 分区；正例 Kp=3、同属性反标签
  hard neg Kn=5（长尾 fallback 全局反标签）；InfoNCE τ=0.07；按可靠性权重 c 加权。
- 两阶段：warmup(仅 CE) → +CL(λ=0.3)，每 epoch 末重建 FAISS。
- 差分 LR（编码器 2e-5 / 融合+头 1e-4）、线性 warmup+衰减、梯度累积(有效 batch 32)、bf16。
- 指标（§4.3 / 用户指定）：Macro-F1(主)、AUPRC、AUROC、可靠性加权 F1(c)、ECE；
  val 上按 Macro-F1 选阈值。推理：forward + RKC + 自一致 abstain(δ=0.3)。

用法：python -m models.train --dataset .../dataset.jsonl --seed 0 [--no_cl|--no_fusion|--no_lora|--no_weight]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import config
from models.data import (ClaimDataset, make_collate, load_split, build_tokenizer,
                         resolve_bge_path, SPECIAL_TOKENS,
                         evidence_combo, confidence_bin)
from models.model import CLAIMARC


COMBO_LABELS = ("none", "P", "O", "V", "PO", "PV", "OV", "POV")
COMBO_TO_ID = {v: i for i, v in enumerate(COMBO_LABELS)}
CONF_LABELS = ("absent", "low", "medium", "high")
CONF_TO_ID = {v: i for i, v in enumerate(CONF_LABELS)}
SOURCE_BIN_LABELS = ("src0", "src1", "src2_3", "src4p")


def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def apply_c_transform(splits: dict[str, list[dict]], transform: str, seed: int = 0,
                      k_sat: float = 2.0):
    """可靠性权重设计合理性验证：对 TRAIN 侧的 c 做反事实变换（val/test 的 c 保持不变，
    使 wF1/阈值等评估口径可比）。变换后的 c 同时流入 CE/对比损失权重与 RKC 投票权重。

    none      : 不变（canonical，w=c）
    uniform   : w=1（去可靠性，损失与 RKC 全均匀）
    inverse   : w=(cmin+cmax)-c（翻转方向：低可靠样本反而高权重）
    permute   : 跨训练样本随机打乱 c（同边际分布，破坏样本对应）
    binary    : w=1[c>=median]（粗二值，丢弃分级信息），0.05 下限
    count     : w=1-exp(-N_aligned/k)（仅证据饱和因子，丢弃不对称/真实性/覆盖），0.05 下限
    oldc      : w=old_c（promotion 前版本）
    sqrt      : w=sqrt(c)（弱化分级强度的对照）
    """
    if not transform or transform == "none":
        return
    rows = splits["train"]
    c = np.array([float(r.get("c", 0.05) or 0.05) for r in rows], dtype=float)
    if transform == "uniform":
        new = np.ones_like(c)
    elif transform == "inverse":
        new = (c.min() + c.max()) - c
        new = np.clip(new, 0.05, None)
    elif transform == "permute":
        rng = np.random.RandomState(seed)
        new = c[rng.permutation(len(c))]
    elif transform == "binary":
        med = float(np.median(c))
        new = np.where(c >= med, 1.0, 0.05)
    elif transform == "count":
        na = np.array([float((r.get("proposal_label_audit", {}) or {}).get("aligned_comment_count", 0) or 0)
                       for r in rows], dtype=float)
        new = np.clip(1.0 - np.exp(-na / max(1e-6, k_sat)), 0.05, None)
    elif transform == "oldc":
        new = np.array([float((r.get("proposal_label_audit", {}) or {}).get("old_c", r.get("c", 0.05)) or 0.05)
                        for r in rows], dtype=float)
    elif transform == "sqrt":
        new = np.sqrt(np.clip(c, 0.0, None))
    else:
        raise ValueError(f"unknown c_transform: {transform}")
    for r, v in zip(rows, new):
        r["c"] = float(v)
    print(f"[c_transform={transform}] train c: mean={float(np.mean(new)):.3f} "
          f"std={float(np.std(new)):.3f} (was mean={float(np.mean(c)):.3f})", flush=True)


def recompute_c(splits: dict[str, list[dict]], spec: str):
    """c 公式自身超参敏感性（§3.2.2）：从 _aligned_consumer_mentions 忠实重算 train 侧 c。
    spec 形如 'k=3,lambda=0.3,rho=0.4,phi=1.2,gamma=2'；缺省取 builder 默认。
    仅对存在对齐评论的样本重算（客观负例无对齐评论，保持原 c）。"""
    import math
    if not spec or spec == "none":
        return
    d = {"k": 3.0, "lambda": 0.3, "rho": 0.4, "phi": 1.2, "gamma": 2.0,
         "s_strong": 1.2, "s_weak": 0.7, "floor": 0.05}
    for kv in spec.replace(";", ",").split(","):
        if "=" in kv:
            kk, vv = kv.split("=", 1)
            d[kk.strip()] = float(vv)

    def w_i(m):
        s = d["s_strong"] if str(m.get("mention_strength")) == "strong" else d["s_weak"]
        efh = 1.0 if m.get("explicit_fact_hit") else 0.0
        return (1.0 + d["gamma"] * efh) * s

    def _ptime(s):
        s = str(s or "").strip()
        for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    def _bigrams(t):
        t = "".join(str(t or "").split())
        return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()

    def _low_div(spans):
        spans = [s for s in spans if s]
        if len(spans) < 3:
            return False
        sims = []
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a, b = _bigrams(spans[i]), _bigrams(spans[j])
                if a and b:
                    sims.append(len(a & b) / len(a | b))
        return bool(sims) and (sum(sims) / len(sims)) >= 0.7

    n_changed = 0
    for r in splits["train"]:
        aligned = [m for m in (r.get("_aligned_consumer_mentions") or [])
                   if (m.get("_judgment") or {}).get("aligned_to_claim")]
        if not aligned:
            continue
        n_total = int(r.get("_consumer_mentions_total") or 0)
        n_aligned = len(aligned)
        neg = [m for m in aligned if str(m.get("polarity")) == "neg"]
        pos = [m for m in aligned if str(m.get("polarity")) == "pos"]
        s_neg = sum(w_i(m) for m in neg); s_pos = sum(w_i(m) for m in pos)
        y = 1 if neg else 0
        f_sat = 1.0 - math.exp(-n_aligned / max(1e-6, d["k"]))
        f_cov = n_aligned / (n_total + 1)
        c_base = f_sat * f_cov
        if y == 1:
            f_asym = s_neg / (s_neg + d["lambda"] * s_pos) if (s_neg + d["lambda"] * s_pos) > 0 else 0.0
            phi = d["phi"] if any((str(m.get("mention_strength")) == "strong" or m.get("explicit_fact_hit")) for m in neg) else 1.0
            c = min(1.0, c_base * f_asym * phi)
        else:
            spans = [str(m.get("evidence_span") or "") for m in aligned]
            times = [t for t in (_ptime(m.get("review_time")) for m in aligned) if t]
            fake_a = (n_total < 10 and not neg and len(pos) > 0)
            fake_b = _low_div(spans)
            fake_c = bool(times) and ((max(times) - min(times)).days <= 3) and n_aligned >= 3
            f_fake = 1.0 - d["rho"] * (1.0 if (fake_a or fake_b or fake_c) else 0.0)
            c = c_base * f_fake
        r["c"] = float(max(c, d["floor"]))
        n_changed += 1
    print(f"[c_recompute={spec}] recomputed train c on {n_changed} comment-driven pairs", flush=True)


def apply_evidence_policy(splits: dict[str, list[dict]], policy: str | None):
    """Force a tokenization-time evidence policy while preserving default record policy."""
    if not policy or policy == "record":
        return
    for rows in splits.values():
        for r in rows:
            r["_evidence_policy"] = policy


def parse_evidence_policy_mix(value) -> list[str]:
    """Parse comma/space separated train-time evidence views."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [x.strip() for x in str(value).replace(",", " ").split()]
    return [x for x in parts if x]


def asl_loss(logit, y, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
    """Asymmetric Loss（Ridnik et al. ICCV'21），高 neg:pos 不平衡的 SOTA，逐样本返回。"""
    p = torch.sigmoid(logit)
    pm = (p - clip).clamp(min=0)               # 负样本概率平移，丢弃极易负例
    los_pos = y * torch.log(p.clamp(min=eps)) * (1 - p) ** gamma_pos
    los_neg = (1 - y) * torch.log((1 - pm).clamp(min=eps)) * (pm ** gamma_neg)
    return -(los_pos + los_neg)


def focal_loss(logit, y, gamma=2.0, alpha=0.5, eps=1e-8):
    p = torch.sigmoid(logit)
    los_pos = alpha * y * torch.log(p.clamp(min=eps)) * (1 - p) ** gamma
    los_neg = (1 - alpha) * (1 - y) * torch.log((1 - p).clamp(min=eps)) * p ** gamma
    return -(los_pos + los_neg)


def cls_loss(logit, y, cw, mode, pos_weight, gamma_neg=4.0, gamma_pos=0.0):
    if mode == "asl":
        per = asl_loss(logit.float(), y, gamma_pos=gamma_pos, gamma_neg=gamma_neg)
    elif mode == "focal":
        per = focal_loss(logit.float(), y, gamma=gamma_neg)
    else:
        per = F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none",
                                                 pos_weight=pos_weight)
    return (per * cw).mean()


def source_scaled_weights(base_cw, batch, args, device, kind):
    """Optional source-domain weighting. Defaults keep historical behavior."""
    cw = base_cw
    source0_scale = float(getattr(args, f"source0_{kind}_scale", 1.0))
    rich_scale = float(getattr(args, f"source_rich_{kind}_scale", 1.0))
    if source0_scale == 1.0 and rich_scale == 1.0:
        return cw
    sc = batch.source_count.to(device)
    sl = batch.source_len.to(device)
    source0 = sc <= 0
    source_rich = (sc >= 2) | (sl >= 20)
    scale = torch.ones_like(cw)
    if source0_scale != 1.0:
        scale = torch.where(source0, torch.full_like(scale, source0_scale), scale)
    if rich_scale != 1.0:
        scale = torch.where(source_rich, scale * rich_scale, scale)
    return cw * scale


def source_count_bins(source_count_tensor: torch.Tensor) -> torch.Tensor:
    """Coarse source sufficiency bins: none / one / few / several / many."""
    sc = source_count_tensor.float()
    out = torch.zeros_like(sc, dtype=torch.long)
    out = torch.where(sc == 1, torch.ones_like(out), out)
    out = torch.where((sc >= 2) & (sc <= 3), torch.full_like(out, 2), out)
    out = torch.where((sc >= 4) & (sc <= 8), torch.full_like(out, 3), out)
    out = torch.where(sc >= 9, torch.full_like(out, 4), out)
    return out


def source_bin_from_count_value(value: int | float) -> str:
    sc = int(value or 0)
    if sc <= 0:
        return "src0"
    if sc == 1:
        return "src1"
    if sc <= 3:
        return "src2_3"
    return "src4p"


def source_bin_list(source_count_tensor: torch.Tensor) -> list[str]:
    values = source_count_tensor.detach().cpu().numpy().tolist()
    return [source_bin_from_count_value(v) for v in values]


def make_source_aux_heads(ret_dim: int):
    return nn.ModuleDict({
        "combo": nn.Linear(ret_dim, len(COMBO_LABELS)),
        "conf": nn.Linear(ret_dim, len(CONF_LABELS)),
        "count": nn.Linear(ret_dim, 5),
    })


def source_aux_loss(g, batch, heads, args, device, cw):
    """Metadata-aware regularizer for the retrieval representation."""
    if heads is None:
        return torch.tensor(0.0, device=device)
    x = g.float()
    total = torch.tensor(0.0, device=device)
    cw = cw.float()
    denom = cw.mean().clamp_min(1e-6)
    combo_w = float(getattr(args, "source_aux_combo_weight", 0.0))
    conf_w = float(getattr(args, "source_aux_conf_weight", 0.0))
    count_w = float(getattr(args, "source_aux_count_weight", 0.0))
    if combo_w > 0:
        tgt = torch.tensor([COMBO_TO_ID.get(str(v), 0) for v in batch.evidence_combo],
                           dtype=torch.long, device=device)
        per = F.cross_entropy(heads["combo"](x), tgt, reduction="none")
        total = total + combo_w * (per * cw).mean() / denom
    if conf_w > 0:
        tgt = torch.tensor([CONF_TO_ID.get(str(v), 0) for v in batch.confidence],
                           dtype=torch.long, device=device)
        per = F.cross_entropy(heads["conf"](x), tgt, reduction="none")
        total = total + conf_w * (per * cw).mean() / denom
    if count_w > 0:
        tgt = source_count_bins(batch.source_count.to(device))
        per = F.cross_entropy(heads["count"](x), tgt, reduction="none")
        total = total + count_w * (per * cw).mean() / denom
    return total


# ----------------------------- C: 可靠性建模 -----------------------------
def rel_soft_target(y: torch.Tensor, c: torch.Tensor, base_rate: float,
                    lo: float = 0.18, hi: float = 0.82) -> torch.Tensor:
    """Noise-aware 软标签（弱监督去噪）：可靠性 c 越低，标签越向基率收缩（更不确定）。

    c 是评论锚定的感知可靠性。c=hi → 完全采信观测标签 y；c=lo → 目标≈数据集正类基率。
    这样低可靠样本不再以硬 0/1 强行拉模型，避免噪声标签主导决策边界。
    """
    cn = ((c - lo) / max(1e-6, hi - lo)).clamp(0.0, 1.0)
    return cn * y + (1.0 - cn) * float(base_rate)


def make_rel_head(ret_dim: int) -> nn.Module:
    """可靠性辅助回归头：从检索表征预测样本可靠性 c（sigmoid 到 (0,1)）。"""
    return nn.Sequential(nn.Linear(ret_dim, 128), nn.GELU(), nn.Linear(128, 1))


def rel_aux_loss(g, batch, head, weight, device):
    """让检索表征显式编码感知可靠性：MSE(sigmoid(head(g)), c)。weight<=0 关闭。"""
    if head is None or weight <= 0:
        return torch.tensor(0.0, device=device)
    pred = torch.sigmoid(head(g.float())).squeeze(-1)
    tgt = batch.c.to(device).float()
    return float(weight) * F.mse_loss(pred, tgt)


_BGE_TEACHER_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _teacher_signature(recs: list[dict]) -> str:
    """Hash the actual teacher inputs, not just pair ids; args/no-args datasets share ids."""
    from models.baselines import claim_text, evidence_text
    h = hashlib.sha1()
    for r in recs:
        h.update(str(r.get("pair_id", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(claim_text(r).encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(evidence_text(r).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def attach_bge_oof_teacher(train_recs: list[dict], inner_folds=5, seed=0):
    """Attach fold-local BGE+LR OOF probabilities for distillation.

    The teacher sees only the current outer-train records. It predicts each training
    record from an inner model that did not train on that record, which keeps the
    distillation target from becoming an in-sample memorization signal.
    """
    if not train_recs:
        return
    inner_folds = max(2, int(inner_folds))
    key = (_teacher_signature(train_recs), inner_folds, int(seed))
    if key in _BGE_TEACHER_CACHE:
        probs = _BGE_TEACHER_CACHE[key]
    else:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        from models.baselines import claim_text, evidence_text

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bge = resolve_bge_path()
        enc = SentenceTransformer(bge, device=device)
        c_txt = [claim_text(r) for r in train_recs]
        e_txt = [evidence_text(r) for r in train_recs]
        c_vec = np.asarray(enc.encode(c_txt, normalize_embeddings=True, batch_size=64,
                                      show_progress_bar=False))
        e_vec = np.asarray(enc.encode(e_txt, normalize_embeddings=True, batch_size=64,
                                      show_progress_bar=False))
        X = np.concatenate([c_vec, e_vec, c_vec - e_vec, c_vec * e_vec], axis=1)
        y = np.array([int(r.get("y", 0)) for r in train_recs])
        c = np.array([float(r.get("c", 0.05)) for r in train_recs])
        groups = np.array([r.get("room_id", r.get("product_id", i))
                           for i, r in enumerate(train_recs)])

        min_class = int(np.bincount(y, minlength=2).min())
        n_splits = min(inner_folds, min_class, len(set(groups)))
        probs = np.full(len(train_recs), np.nan, dtype=float)
        if n_splits >= 2:
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                split_iter = splitter.split(np.zeros(len(y)), y, groups)
            except Exception:
                splitter = StratifiedKFold(n_splits=min(inner_folds, min_class),
                                           shuffle=True, random_state=seed)
                split_iter = splitter.split(np.zeros(len(y)), y)
            for tr, te in split_iter:
                clf = LogisticRegression(C=1.0, max_iter=3000)
                clf.fit(X[tr], y[tr], sample_weight=np.clip(c[tr], 0.05, None))
                probs[te] = clf.predict_proba(X[te])[:, 1]
        if np.isnan(probs).any():
            clf = LogisticRegression(C=1.0, max_iter=3000)
            clf.fit(X, y, sample_weight=np.clip(c, 0.05, None))
            miss = np.isnan(probs)
            probs[miss] = clf.predict_proba(X[miss])[:, 1]
        probs = np.clip(probs, 1e-4, 1 - 1e-4)
        _BGE_TEACHER_CACHE[key] = probs
    for r, p in zip(train_recs, probs):
        r["_teacher_p"] = float(p)
    print(f"[distill] attached BGE+LR OOF teacher: n={len(train_recs)} "
          f"mean={float(np.mean(probs)):.4f} std={float(np.std(probs)):.4f}",
          flush=True)


def distill_loss(logit, teacher_p, cw, sample_c=None, weight=0.0, temp=1.0,
                 conf_min=0.0, c_min=0.0, mode="all"):
    if weight <= 0:
        return torch.tensor(0.0, device=logit.device)
    mask = teacher_p >= 0
    if conf_min > 0:
        mask = mask & ((teacher_p - 0.5).abs() >= conf_min)
    if c_min > 0 and sample_c is not None:
        mask = mask & (sample_c >= c_min)
    if mode == "disagree":
        student_label = torch.sigmoid(logit.detach()) >= 0.5
        teacher_label = teacher_p >= 0.5
        mask = mask & (student_label != teacher_label)
    elif mode != "all":
        raise ValueError(f"unknown distill mode: {mode}")
    if not mask.any():
        return torch.tensor(0.0, device=logit.device)
    target = teacher_p[mask].float()
    student = logit.float()[mask]
    if temp != 1.0:
        target = torch.sigmoid(torch.logit(target.clamp(1e-6, 1 - 1e-6)) / temp)
        student = student / temp
    per = F.binary_cross_entropy_with_logits(student, target, reduction="none")
    return weight * (per * cw[mask]).mean() * (temp ** 2)


def view_consistency_loss(logit, g, logit_view, g_view, cw,
                          logit_weight=0.0, embed_weight=0.0):
    """Align an auxiliary evidence view to the main view without changing eval protocol."""
    loss = torch.tensor(0.0, device=logit.device)
    if logit_weight > 0:
        target = torch.sigmoid(logit.float()).detach()
        pred = torch.sigmoid(logit_view.float())
        loss = loss + float(logit_weight) * (((pred - target) ** 2) * cw).mean()
    if embed_weight > 0:
        g_ref = F.normalize(g.float(), dim=-1).detach()
        g_aux = F.normalize(g_view.float(), dim=-1)
        per = 1.0 - (g_ref * g_aux).sum(dim=-1).clamp(-1.0, 1.0)
        loss = loss + float(embed_weight) * (per * cw).mean()
    return loss


# ----------------------------- 内存库 + InfoNCE -----------------------------
class MemoryBank:
    def __init__(self, g: torch.Tensor, attrs: list[str], y: torch.Tensor,
                 c: torch.Tensor | np.ndarray | None = None,
                 teacher_p: torch.Tensor | np.ndarray | None = None,
                 evidence_combo: list[str] | np.ndarray | None = None,
                 confidence: list[str] | np.ndarray | None = None,
                 source_bin: list[str] | np.ndarray | None = None,
                 contrastive_mask: list[bool] | np.ndarray | torch.Tensor | None = None):
        self.g = g
        self.attrs = np.array(attrs)
        self.y = y.cpu().numpy()
        if c is None:
            self.c = np.ones(len(self.y), dtype=float)
        elif isinstance(c, torch.Tensor):
            self.c = c.cpu().numpy()
        else:
            self.c = np.asarray(c, dtype=float)
        if teacher_p is None:
            self.teacher_p = np.full(len(self.y), -1.0, dtype=float)
        elif isinstance(teacher_p, torch.Tensor):
            self.teacher_p = teacher_p.cpu().numpy()
        else:
            self.teacher_p = np.asarray(teacher_p, dtype=float)
        if evidence_combo is None:
            self.evidence_combo = np.asarray([""] * len(self.y), dtype=object)
        else:
            self.evidence_combo = np.asarray(evidence_combo, dtype=object)
        if confidence is None:
            self.confidence = np.asarray([""] * len(self.y), dtype=object)
        else:
            self.confidence = np.asarray(confidence, dtype=object)
        if source_bin is None:
            self.source_bin = np.asarray([""] * len(self.y), dtype=object)
        else:
            self.source_bin = np.asarray(source_bin, dtype=object)
        if contrastive_mask is None:
            self.contrastive_mask = np.ones(len(self.y), dtype=bool)
        elif isinstance(contrastive_mask, torch.Tensor):
            self.contrastive_mask = contrastive_mask.cpu().numpy().astype(bool)
        else:
            self.contrastive_mask = np.asarray(contrastive_mask, dtype=bool)

    def neighbors(self, gq: torch.Tensor, mask: np.ndarray, k: int,
                  bonus: np.ndarray | None = None, hardest: bool = False):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        sims = (self.g[idx] @ gq)
        if bonus is not None:
            sims = sims + torch.as_tensor(bonus[idx], device=sims.device, dtype=sims.dtype)
        kk = min(k, len(idx))
        # hardest=True 选相似度最低的同标签样本（hard positive，避免易正例梯度消失）；
        # 否则选 top-K 最相似（hard negative 的常规口径）。
        top = torch.topk(sims, kk, largest=not hardest).indices.cpu().numpy()
        return idx[top]


def info_nce(g_anchor, g_pos, g_negs, tau=0.07):
    pos = (g_anchor * g_pos).sum() / tau
    if len(g_negs) == 0:
        return torch.tensor(0.0, device=g_anchor.device)
    neg = (g_negs @ g_anchor) / tau
    denom = torch.logsumexp(torch.cat([pos.unsqueeze(0), neg]), 0)
    return -(pos - denom)


def supcon_loss(g, batch, tau=0.07):
    """Faithful supervised contrastive loss (Khosla et al., 2020), L_out form.

    In-batch only: for each anchor, positives are all OTHER same-label samples in
    the batch, negatives are all other-label samples in the batch. No retrieval,
    no hard mining, no reliability/class weighting — the standard 2020 baseline.
    """
    z = F.normalize(g, dim=-1)
    device = z.device
    y = batch.y.to(device).view(-1)
    if hasattr(batch, "contrastive_mask"):
        elig = batch.contrastive_mask.to(device).view(-1) >= 0.5
    else:
        elig = torch.ones(len(y), dtype=torch.bool, device=device)
    idx = torch.nonzero(elig).flatten()
    if idx.numel() < 2:
        return torch.tensor(0.0, device=device)
    zz = z[idx]; yy = y[idx]
    m = zz.shape[0]
    sim = (zz @ zz.T) / tau
    self_mask = torch.eye(m, dtype=torch.bool, device=device)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    denom = exp.sum(dim=1).clamp_min(1e-12)          # A(i): all except self
    log_prob = sim - torch.log(denom).view(-1, 1)
    pos_mask = (yy.view(-1, 1) == yy.view(1, -1)) & (~self_mask)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def contrastive_loss(g, batch, bank: MemoryBank, cw, Kp=3, Kn=5, tau=0.07, global_neg=False,
                     cl_c_min=0.0, cl_neg_c_min=0.0,
                     cl_teacher_mode="off", cl_teacher_conf_min=0.0,
                     cl_neg_filter="none", cl_neg_bonus=0.0,
                     cl_neg_bonus_filter="none",
                     cl_attr_block=True, cl_class_balanced=False, cl_hard_pos=False):
    """检索增强监督对比（RACL）。

    B 优化旋钮（消融证实属性分块作用很小，默认仍保留以兼容 canonical）：
      cl_attr_block=False  → 去属性分块，正/负例都在全局标签池内挑选（类平衡监督对比口径）。
      cl_class_balanced=True → 用 0.5/p_y 类频逆权重缩放每个 anchor，抵消正类(~26%)被多数类淹没。
      cl_hard_pos=True     → 同标签正例取相似度最低的 Kp 个（hard positive，避免易正例梯度消失）。
    """
    losses = []
    g_d = F.normalize(g, dim=-1)
    bank_teacher_ok = np.ones(len(bank.y), dtype=bool)
    if cl_teacher_mode in ("agree", "agree_pos"):
        tp = bank.teacher_p
        bank_teacher_ok = (tp >= 0) & (np.abs(tp - 0.5) >= cl_teacher_conf_min)
        bank_teacher_ok &= ((tp >= 0.5).astype(int) == bank.y)
    elif cl_teacher_mode != "off":
        raise ValueError(f"unknown cl_teacher_mode: {cl_teacher_mode}")
    neg_teacher_ok = bank_teacher_ok if cl_teacher_mode == "agree" else np.ones(len(bank.y), dtype=bool)
    # 类平衡权重（A Tale of Two Classes / ResCom 的类平衡对比）：基于库内合格样本的标签频率
    class_w = {0: 1.0, 1: 1.0}
    if cl_class_balanced:
        elig = bank.contrastive_mask & (bank.c >= cl_c_min)
        n1 = float(((bank.y == 1) & elig).sum())
        n0 = float(((bank.y == 0) & elig).sum())
        p1 = n1 / max(1.0, n1 + n0)
        p1 = min(max(p1, 1e-3), 1 - 1e-3)
        class_w = {0: 0.5 / (1 - p1), 1: 0.5 / p1}
    for i in range(len(batch.y)):
        if hasattr(batch, "contrastive_mask") and float(batch.contrastive_mask[i].item()) < 0.5:
            continue
        if float(cw[i].detach().cpu()) < cl_c_min:
            continue
        a = batch.attr[i]; yi = int(batch.y[i].item())
        if cl_teacher_mode in ("agree", "agree_pos"):
            tpi = float(batch.teacher_p[i].detach().cpu())
            if tpi < 0 or abs(tpi - 0.5) < cl_teacher_conf_min:
                continue
            if int(tpi >= 0.5) != yi:
                continue
        same = (bank.attrs == a) if cl_attr_block else np.ones(len(bank.y), dtype=bool)
        eligible = bank.contrastive_mask
        pos_mask = same & (bank.y == yi) & (bank.c >= cl_c_min) & bank_teacher_ok & eligible
        if pos_mask.sum() == 0 and cl_attr_block:
            pos_mask = (bank.y == yi) & (bank.c >= cl_c_min) & bank_teacher_ok & eligible
        if global_neg or not cl_attr_block:
            base_neg_mask = (bank.y != yi) & (bank.c >= cl_neg_c_min) & neg_teacher_ok & eligible  # 全集合反标签
        else:
            base_neg_mask = same & (bank.y != yi) & (bank.c >= cl_neg_c_min) & neg_teacher_ok & eligible
            if base_neg_mask.sum() == 0:
                base_neg_mask = (bank.y != yi) & (bank.c >= cl_neg_c_min) & neg_teacher_ok & eligible
        neg_mask = base_neg_mask
        neg_bonus = None
        ev_i = str(batch.evidence_combo[i])
        conf_i = str(batch.confidence[i])
        if cl_neg_filter != "none":
            if cl_neg_filter == "same_evtype":
                filt = bank.evidence_combo == ev_i
            elif cl_neg_filter == "same_evtype_conf":
                filt = (bank.evidence_combo == ev_i) & (bank.confidence == conf_i)
            elif cl_neg_filter == "medium_evtype_conf":
                filt = (
                    (bank.evidence_combo == ev_i)
                    & (bank.confidence == conf_i)
                    & (conf_i == "medium")
                )
            else:
                raise ValueError(f"unknown cl_neg_filter: {cl_neg_filter}")
            narrowed = base_neg_mask & filt
            if narrowed.sum() > 0:
                neg_mask = narrowed
        if cl_neg_bonus > 0 and cl_neg_bonus_filter != "none":
            if cl_neg_bonus_filter == "same_evtype":
                bonus_mask = bank.evidence_combo == ev_i
            elif cl_neg_bonus_filter == "same_evtype_conf":
                bonus_mask = (bank.evidence_combo == ev_i) & (bank.confidence == conf_i)
            elif cl_neg_bonus_filter == "medium_evtype_conf":
                bonus_mask = (
                    (bank.evidence_combo == ev_i)
                    & (bank.confidence == conf_i)
                    & (conf_i == "medium")
                )
            else:
                raise ValueError(f"unknown cl_neg_bonus_filter: {cl_neg_bonus_filter}")
            neg_bonus = np.where(bonus_mask, float(cl_neg_bonus), 0.0)
        pos_idx = bank.neighbors(g_d[i].detach(), pos_mask, Kp, hardest=cl_hard_pos)
        neg_idx = bank.neighbors(g_d[i].detach(), neg_mask, Kn, bonus=neg_bonus)
        if len(pos_idx) == 0:
            continue
        g_negs = bank.g[neg_idx] if len(neg_idx) else bank.g[:0]
        li = 0.0
        for pj in pos_idx:
            li = li + info_nce(g_d[i], bank.g[pj], g_negs, tau)
        losses.append(class_w[yi] * cw[i] * li / max(1, len(pos_idx)))
    if not losses:
        return torch.tensor(0.0, device=g.device)
    return torch.stack(losses).mean()


def _bank_group_array(bank: MemoryBank, group: str) -> np.ndarray:
    if group == "global":
        return np.asarray(["all"] * len(bank.y), dtype=object)
    if group == "attr":
        return bank.attrs
    if group == "source_bin":
        return bank.source_bin
    if group == "evidence_combo":
        return bank.evidence_combo
    if group == "confidence":
        return bank.confidence
    if group == "combo_conf":
        return np.asarray(
            [f"{a}:{b}" for a, b in zip(bank.evidence_combo, bank.confidence)],
            dtype=object,
        )
    if group == "source_conf":
        return np.asarray(
            [f"{a}:{b}" for a, b in zip(bank.source_bin, bank.confidence)],
            dtype=object,
        )
    raise ValueError(f"unknown proto_aux_group: {group}")


def _batch_group_value(batch, i: int, group: str) -> str:
    if group == "global":
        return "all"
    if group == "attr":
        return str(batch.attr[i])
    if group == "source_bin":
        return source_bin_from_count_value(float(batch.source_count[i].item()))
    if group == "evidence_combo":
        return str(batch.evidence_combo[i])
    if group == "confidence":
        return str(batch.confidence[i])
    if group == "combo_conf":
        return f"{batch.evidence_combo[i]}:{batch.confidence[i]}"
    if group == "source_conf":
        sb = source_bin_from_count_value(float(batch.source_count[i].item()))
        return f"{sb}:{batch.confidence[i]}"
    raise ValueError(f"unknown proto_aux_group: {group}")


def _weighted_proto(g_bank: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    w = torch.as_tensor(weights, device=g_bank.device, dtype=g_bank.dtype)
    proto = (g_bank * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-8)
    return F.normalize(proto, dim=0)


def prototype_aux_loss(g, batch, bank: MemoryBank | None, cw, args, device):
    """Train-time prototype relation loss over the epoch memory bank.

    Each anchor is classified by its similarity to positive/negative prototypes
    inside a small evidence/source stratum, falling back to global prototypes
    when the stratum is too sparse.  The bank is detached, so this regularizes
    current-batch retrieval embeddings without backpropagating through the
    epoch-level cache.
    """
    weight = float(getattr(args, "proto_aux_weight", 0.0))
    if weight <= 0 or bank is None:
        return torch.tensor(0.0, device=device)
    group = str(getattr(args, "proto_aux_group", "source_bin"))
    min_class = int(getattr(args, "proto_aux_min_class", 3))
    c_min = float(getattr(args, "proto_aux_c_min", 0.0))
    tau = max(float(getattr(args, "proto_aux_tau", 0.10)), 1e-6)
    mode = str(getattr(args, "proto_aux_mode", "ce"))
    margin = float(getattr(args, "proto_aux_margin", 0.15))
    bank_g = F.normalize(bank.g.float().detach(), dim=-1)
    g_norm = F.normalize(g.float(), dim=-1)
    group_arr = _bank_group_array(bank, group)
    eligible = bank.contrastive_mask
    global_pos = (bank.y == 1) & (bank.c >= c_min) & eligible
    global_neg = (bank.y == 0) & (bank.c >= c_min) & eligible
    losses = []
    for i in range(len(batch.y)):
        if hasattr(batch, "contrastive_mask") and float(batch.contrastive_mask[i].item()) < 0.5:
            continue
        if float(cw[i].detach().cpu()) < c_min:
            continue
        key = _batch_group_value(batch, i, group)
        in_group = group_arr == key
        pos_mask = in_group & global_pos
        neg_mask = in_group & global_neg
        if int(pos_mask.sum()) < min_class or int(neg_mask.sum()) < min_class:
            pos_mask = global_pos
            neg_mask = global_neg
        if int(pos_mask.sum()) == 0 or int(neg_mask.sum()) == 0:
            continue
        proto_neg = _weighted_proto(bank_g[neg_mask], bank.c[neg_mask])
        proto_pos = _weighted_proto(bank_g[pos_mask], bank.c[pos_mask])
        sim_neg = (g_norm[i] * proto_neg).sum()
        sim_pos = (g_norm[i] * proto_pos).sum()
        if mode == "ce":
            logits = torch.stack([sim_neg / tau, sim_pos / tau]).unsqueeze(0)
            target = batch.y[i].long().view(1).to(device)
            per = F.cross_entropy(logits, target, reduction="none").squeeze(0)
        elif mode == "margin":
            y_sign = torch.where(
                batch.y[i].to(device).float() > 0.5,
                torch.ones((), device=device, dtype=sim_pos.dtype),
                -torch.ones((), device=device, dtype=sim_pos.dtype),
            )
            signed_gap = y_sign * (sim_pos - sim_neg)
            per = F.softplus((margin - signed_gap) / tau) * tau
        else:
            raise ValueError(f"unknown proto_aux_mode: {mode}")
        losses.append(cw[i].float() * per)
    if not losses:
        return torch.tensor(0.0, device=device)
    denom = cw.float().mean().clamp_min(1e-6)
    return weight * torch.stack(losses).mean() / denom


# ----------------------------- 评估 -----------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, gs, ys, cs, attrs = [], [], [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            lg, g = model(b.c_ids.to(device), b.c_mask.to(device), b.e_ids.to(device), b.e_mask.to(device))
        logits.append(torch.sigmoid(lg.float()).cpu()); gs.append(g.float().cpu())
        ys.append(b.y); cs.append(b.c); attrs += b.attr
    return (torch.cat(logits).numpy(), torch.cat(gs), torch.cat(ys).numpy(),
            torch.cat(cs).numpy(), attrs)


def macro_f1(y, pred, w=None):
    return f1_score(y, pred, average="macro", sample_weight=w, zero_division=0)


def best_threshold_macroF1(y, p):
    best_t, best = 0.5, -1
    for t in np.linspace(0.02, 0.98, 49):
        f = macro_f1(y, (p >= t).astype(int))
        if f > best:
            best, best_t = f, t
    return best_t


def ece(y, p, n_bins=15):
    """标准二分类期望校准误差：每个分箱内 |平均预测概率 − 经验正例率|，按箱占比加权。

    注意：这里 p 是 P(y=1) 的估计；校准对比的是“预测概率 vs 该概率区间的真实正例频率”，
    而非某个固定阈值下的分类准确率。
    """
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        hi = (p < bins[i + 1]) if i < n_bins - 1 else (p <= bins[i + 1])
        m = (p >= bins[i]) & hi
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y[m].mean()
        e += (m.mean()) * abs(conf - acc)
    return float(e)


def rkc_predict(train_g, train_y, train_c, test_g, k=10):
    sims = test_g @ train_g.T
    kk = min(k, train_g.shape[0])
    top = torch.topk(sims, kk, dim=1)
    idx, val = top.indices.numpy(), top.values.numpy()
    tc, ty = train_c[idx], train_y[idx]
    w = tc * np.clip(val, 0, None)
    num = (w * ty).sum(1); den = w.sum(1) + 1e-8
    return num / den


def rkc_attr_predict(train_g, train_y, train_c, train_attr, test_g, test_attr, k=10):
    """属性分块检索增强 kNN（§3.4 RKC）：在同一属性的训练近邻内做可信度加权投票，
    回退到全局检索当同属性近邻不足。"""
    g_tr = F.normalize(train_g, dim=-1)
    g_te = F.normalize(test_g, dim=-1)
    sims = (g_te @ g_tr.T).numpy()
    atr = np.asarray(train_attr); ate = np.asarray(test_attr)
    n = g_te.shape[0]
    out = np.zeros(n)
    for i in range(n):
        idx = np.where(atr == ate[i])[0]
        if len(idx) < 3:
            idx = np.arange(len(atr))
        s = sims[i, idx]
        kk = min(k, len(idx))
        top = np.argpartition(-s, kk - 1)[:kk]
        j = idx[top]
        w = train_c[j] * np.clip(s[top], 0, None)
        out[i] = (w * train_y[j]).sum() / (w.sum() + 1e-8)
    return out


def retrieval_quality(test_g, test_y, test_attr, k=10):
    """Label-match@10 与 Attribute-match mAP@10（§4.3 检索表征质量）。"""
    g = F.normalize(test_g, dim=-1)
    sims = g @ g.T
    n = len(test_y)
    sims.fill_diagonal_(-2)
    kk = min(k, n - 1)
    top = torch.topk(sims, kk, dim=1).indices.numpy()
    ty = np.array(test_y); ta = np.array(test_attr)
    lm = np.mean([(ty[top[i]] == ty[i]).mean() for i in range(n)])
    aps = []
    for i in range(n):
        rel = (ta[top[i]] == ta[i]).astype(int)
        if rel.sum() == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(kk) + 1)
        aps.append((prec * rel).sum() / rel.sum())
    return round(float(lm), 4), round(float(np.mean(aps)), 4)


def evaluate(model, loaders, device, train_pack, tag="", seed=0):
    p_val, g_val, y_val, _, attr_val = predict(model, loaders["val"], device)
    thr = best_threshold_macroF1(y_val, p_val)
    p, g, y, c, attrs = predict(model, loaders["test"], device)
    pred = (p >= thr).astype(int)
    res = {
        "tag": tag, "seed": seed, "thr": round(float(thr), 3),
        "acc": round(float((pred == y).mean()), 4),
        "macro_f1": round(macro_f1(y, pred), 4),
        "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
        "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
        "auprc": round(average_precision_score(y, p), 4) if len(set(y)) > 1 else None,
        "auroc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
        "ece": round(ece(y, p), 4),
        "n_test": int(len(y)), "pos_test": int(y.sum()),
    }
    tg, ty, tc = train_pack
    _, gtr_full, _, _, attr_tr = predict(model, loaders["train_eval"], device)
    # ---- 属性分块检索增强推理（RKC）：在 val 上调融合权重 α，再用于 test，避免泄漏 ----
    prkc_val = rkc_attr_predict(gtr_full, ty, tc, attr_tr, g_val, attr_val)
    best_a, best_s = 1.0, -1.0
    for a in np.linspace(0.0, 1.0, 11):
        s = average_precision_score(y_val, a * p_val + (1 - a) * prkc_val) if len(set(y_val)) > 1 else 0
        if s > best_s:
            best_s, best_a = s, a
    prkc = rkc_attr_predict(gtr_full, ty, tc, attr_tr, g, attrs)
    comb = best_a * p + (1 - best_a) * prkc
    thr_rkc = best_threshold_macroF1(y_val, best_a * p_val + (1 - best_a) * prkc_val)
    res["alpha_rkc"] = round(float(best_a), 2)
    res["auprc_rkc"] = round(average_precision_score(y, comb), 4) if len(set(y)) > 1 else None
    res["auroc_rkc"] = round(roc_auc_score(y, comb), 4) if len(set(y)) > 1 else None
    res["acc_rkc"] = round(float(((comb >= thr_rkc).astype(int) == y).mean()), 4)
    res["pos_f1_rkc"] = round(f1_score(y, (comb >= thr_rkc).astype(int), zero_division=0), 4)
    res["macro_f1_rkc"] = round(macro_f1(y, (comb >= thr_rkc).astype(int)), 4)
    res["wF1_rkc"] = round(macro_f1(y, (comb >= thr_rkc).astype(int), w=np.clip(c, 0.05, None)), 4)
    lm, amap = retrieval_quality(g, y, attrs)
    res["label_match@10"] = lm
    res["attr_mAP@10"] = amap
    return res


# ----------------------------- 训练 -----------------------------
def train(args, splits=None, return_model=False):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.backbone == "bert":
        from models.baselines_ft import resolve as resolve_bert
        bge = resolve_bert("bert-base-chinese")
    elif args.backbone == "roberta":
        from models.baselines_ft import resolve as resolve_bert
        bge = resolve_bert("hfl/chinese-roberta-wwm-ext")
    else:
        bge = resolve_bge_path(args.encoder_name)
    tok = build_tokenizer(bge)
    if splits is None:
        splits = load_split(args.dataset)
    if getattr(args, "infer_jsonl", ""):
        # 反事实/构造样本推理：用外部 jsonl 替换 test 划分（train/val 仍用于 bank/阈值/RKC 索引）
        recs = [json.loads(l) for l in open(args.infer_jsonl, encoding="utf-8") if l.strip()]
        for r in recs:
            r["split"] = "test"
        splits["test"] = recs
        print(f"[infer_jsonl] test split replaced with {len(recs)} constructed records", flush=True)
    apply_evidence_policy(splits, getattr(args, "evidence_policy", ""))
    apply_c_transform(splits, getattr(args, "c_transform", "none"), seed=args.seed)
    recompute_c(splits, getattr(args, "c_recompute", "none"))
    needs_teacher = (
        getattr(args, "distill_bge_weight", 0.0) > 0
        or getattr(args, "cl_teacher_mode", "off") != "off"
    )
    if needs_teacher:
        attach_bge_oof_teacher(
            splits["train"],
            inner_folds=getattr(args, "distill_bge_folds", 5),
            seed=getattr(args, "distill_teacher_seed", 0),
        )
    print({k: len(v) for k, v in splits.items()},
          "pos:", {k: int(sum(r["y"] for r in v)) for k, v in splits.items()}, flush=True)
    collate = make_collate(tok.pad_token_id, getattr(args, "stream_mode", "dual"))
    evidence_policy_mix = parse_evidence_policy_mix(getattr(args, "evidence_policy_mix", ""))
    evidence_consistency_mix = parse_evidence_policy_mix(
        getattr(args, "view_consistency_mix", "")
    )
    if evidence_policy_mix:
        print(f"[evidence_policy_mix] train-only views={evidence_policy_mix}", flush=True)
    if evidence_consistency_mix:
        print(f"[view_consistency_mix] aux train-only views={evidence_consistency_mix} "
              f"ce={getattr(args, 'view_ce_weight', 0.0)} "
              f"logit={getattr(args, 'view_logit_weight', 0.0)} "
              f"embed={getattr(args, 'view_embed_weight', 0.0)}",
              flush=True)
    loaders = {
        s: DataLoader(ClaimDataset(
            splits[s], tok,
            evidence_policy_mix=(evidence_policy_mix if s == "train" else None),
            evidence_consistency_mix=(
                evidence_consistency_mix if s == "train" else None
            ),
        ), batch_size=args.bs,
                      shuffle=(s == "train"), collate_fn=collate, num_workers=6,
                      pin_memory=True, persistent_workers=True)
        for s in ("train", "val", "test")
    }
    loaders["train_eval"] = DataLoader(ClaimDataset(splits["train"], tok), batch_size=args.bs,
                                       shuffle=False, collate_fn=collate, num_workers=6,
                                       pin_memory=True, persistent_workers=True)
    model = CLAIMARC(bge, len(tok), n_special=len(SPECIAL_TOKENS),
                     n_fusion=(0 if args.no_fusion else args.n_fusion),
                     use_lora=not args.no_lora, fusion_dropout=args.fusion_dropout,
                     lora_rank=args.lora_rank, xattn_dir=args.xattn_dir,
                     indep_proj=args.indep_proj, ffn=args.ffn, heads=args.heads,
                     enc_train=getattr(args, "enc_train", "lora"),
                     unfreeze_top=getattr(args, "unfreeze_top", 0),
                     ret_disc=not getattr(args, "no_ret_disc", False),
                     head_4tuple=not getattr(args, "head_concat_only", False),
                     joint_encode=getattr(args, "joint_encode", False)).to(device)
    if getattr(args, "load_ckpt", ""):
        sd = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[load_ckpt] {args.load_ckpt} | missing={len(miss)} unexpected={len(unexp)}", flush=True)
    source_aux_on = (
        getattr(args, "source_aux_combo_weight", 0.0) > 0
        or getattr(args, "source_aux_conf_weight", 0.0) > 0
        or getattr(args, "source_aux_count_weight", 0.0) > 0
    )
    source_aux_heads = make_source_aux_heads(model.ret[-1].out_features).to(device) if source_aux_on else None
    rel_aux_head = (make_rel_head(model.ret[-1].out_features).to(device)
                    if getattr(args, "rel_aux_weight", 0.0) > 0 else None)
    if rel_aux_head is not None:
        print(f"[rel_aux] weight={getattr(args, 'rel_aux_weight', 0.0)}", flush=True)
    if getattr(args, "rel_soft", False):
        print("[rel_soft] noise-aware soft labels ON (low-c labels shrink toward base rate)", flush=True)
    if source_aux_on:
        print("[source_aux] combo="
              f"{getattr(args, 'source_aux_combo_weight', 0.0)} "
              f"conf={getattr(args, 'source_aux_conf_weight', 0.0)} "
              f"count={getattr(args, 'source_aux_count_weight', 0.0)} "
              f"in_warmup={getattr(args, 'source_aux_in_warmup', False)}",
              flush=True)
    if getattr(args, "proto_aux_weight", 0.0) > 0:
        print("[proto_aux] weight="
              f"{getattr(args, 'proto_aux_weight', 0.0)} "
              f"group={getattr(args, 'proto_aux_group', 'source_bin')} "
              f"mode={getattr(args, 'proto_aux_mode', 'ce')} "
              f"margin={getattr(args, 'proto_aux_margin', 0.15)} "
              f"tau={getattr(args, 'proto_aux_tau', 0.10)} "
              f"min_class={getattr(args, 'proto_aux_min_class', 3)} "
              f"c_min={getattr(args, 'proto_aux_c_min', 0.10)} "
              f"in_warmup={getattr(args, 'proto_aux_in_warmup', False)}",
              flush=True)
    param_groups = model.param_groups(args.lr, args.lr_head)
    if source_aux_heads is not None:
        param_groups.append({"params": source_aux_heads.parameters(), "lr": args.lr_head})
    if rel_aux_head is not None:
        param_groups.append({"params": rel_aux_head.parameters(), "lr": args.lr_head})
    opt = torch.optim.AdamW(param_groups,
                            betas=(0.9, 0.999), weight_decay=0.01)
    train_recs = splits["train"]
    n_pos = sum(r["y"] for r in train_recs); n_neg = len(train_recs) - n_pos
    base_rate = n_pos / max(1, len(train_recs))
    pw = args.pos_weight if args.pos_weight > 0 else (n_neg / max(1, n_pos))
    pos_weight = torch.tensor(min(pw, 50.0), device=device)
    print(f"[train] pos_weight={float(pos_weight):.1f} seed={args.seed}", flush=True)

    accum = max(1, args.accum)
    steps_per_epoch = (len(loaders["train"]) + accum - 1) // accum
    total_steps = steps_per_epoch * (args.warmup + args.cl_epochs)
    warm = min(200, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warm:
            return step / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def build_bank():
        _p, g, y, c, attrs = predict(model, loaders["train_eval"], device)
        teacher_p = np.array([float(r.get("_teacher_p", -1.0)) for r in splits["train"]], dtype=float)
        ev_combo = [evidence_combo(r) for r in splits["train"]]
        conf = [confidence_bin(r) for r in splits["train"]]
        src_bin = []
        for r in splits["train"]:
            ev = r.get("evidence_count", {}) or {}
            if isinstance(ev, dict):
                sc = sum(int(ev.get(k, 0) or 0) for k in ("params", "ocr", "vlm"))
            else:
                try:
                    sc = int(ev)
                except Exception:
                    sc = 0
            src_bin.append(source_bin_from_count_value(sc))
        cmask = [bool(r.get("contrastive_mask", True)) for r in splits["train"]]
        return (
            MemoryBank(g.to(device), attrs, torch.tensor(y), c,
                       teacher_p=teacher_p, evidence_combo=ev_combo,
                       confidence=conf, source_bin=src_bin,
                       contrastive_mask=cmask),
            (g, y, c),
        )

    import copy

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    def snap():
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                if p.requires_grad}

    def load_trainable(state):
        msd = model.state_dict()
        for n in state:
            msd[n] = state[n].to(device)
        model.load_state_dict(msd)

    def val_score():
        pv, _, yv, _, _ = predict(model, loaders["val"], device)
        vthr = best_threshold_macroF1(yv, pv)
        vf1 = macro_f1(yv, (pv >= vthr).astype(int))
        vap = average_precision_score(yv, pv) if len(set(yv)) > 1 else 0.0
        return vf1 + 0.5 * vap

    ckpts = []  # (score, trainable_state)
    eval_only = getattr(args, "eval_only", False)
    for epoch in range(0 if not eval_only else args.warmup + args.cl_epochs,
                       args.warmup + args.cl_epochs):
        cl_mode = getattr(args, "cl_mode", "racl")
        cl_on = (epoch >= args.warmup) and (not args.no_cl) and (cl_mode != "none")
        bank = build_bank()[0] if (cl_on and cl_mode == "racl") else None
        model.train()
        tot, ce_tot, cl_tot, td_tot, view_tot, aux_tot, proto_tot = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        opt.zero_grad()
        for bi, b in enumerate(loaders["train"]):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logit, g = model(b.c_ids.to(device), b.c_mask.to(device),
                                 b.e_ids.to(device), b.e_mask.to(device))
                y = b.y.to(device)
                base_cw = b.c.to(device) if not args.no_weight else torch.ones_like(b.c).to(device)
                rp = float(getattr(args, "rel_pow", 1.0))
                if rp != 1.0 and not args.no_weight:
                    # 锐化可靠性加权（c^rp），保持均值不变以免改变各损失项相对量级
                    w = base_cw.clamp(min=1e-4) ** rp
                    base_cw = w * (base_cw.mean() / w.mean().clamp_min(1e-6))
                ce_cw = source_scaled_weights(base_cw, b, args, device, "ce")
                cl_cw = source_scaled_weights(base_cw, b, args, device, "cl")
                if getattr(args, "rel_soft", False):
                    soft_t = rel_soft_target(y, b.c.to(device), base_rate)
                    per = F.binary_cross_entropy_with_logits(
                        logit.float(), soft_t, reduction="none", pos_weight=pos_weight)
                    ce = (per * ce_cw).mean()
                else:
                    ce = cls_loss(logit, y, ce_cw, args.loss, pos_weight,
                                  gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
                rel_loss = rel_aux_loss(g, b, rel_aux_head,
                                        getattr(args, "rel_aux_weight", 0.0), device)
                td = distill_loss(logit, b.teacher_p.to(device), ce_cw,
                                  sample_c=b.c.to(device),
                                  weight=getattr(args, "distill_bge_weight", 0.0),
                                  temp=getattr(args, "distill_temp", 1.0),
                                  conf_min=getattr(args, "distill_conf_min", 0.0),
                                  c_min=getattr(args, "distill_c_min", 0.0),
                                  mode=getattr(args, "distill_mode", "all"))
                aux_loss = torch.tensor(0.0, device=device)
                aux_on = (
                    source_aux_heads is not None
                    and (
                        getattr(args, "source_aux_in_warmup", False)
                        or epoch >= args.warmup
                    )
                )
                if aux_on:
                    aux_loss = source_aux_loss(g, b, source_aux_heads, args, device, ce_cw)
                view_loss = torch.tensor(0.0, device=device)
                view_weight_on = (
                    getattr(args, "view_ce_weight", 0.0) > 0
                    or getattr(args, "view_logit_weight", 0.0) > 0
                    or getattr(args, "view_embed_weight", 0.0) > 0
                )
                view_on = (
                    b.e_view_ids is not None
                    and view_weight_on
                    and (
                        getattr(args, "view_consistency_in_warmup", False)
                        or epoch >= args.warmup
                    )
                )
                if view_on:
                    logit_view, g_view = model(
                        b.c_ids.to(device), b.c_mask.to(device),
                        b.e_view_ids.to(device), b.e_view_mask.to(device),
                    )
                    if getattr(args, "view_ce_weight", 0.0) > 0:
                        view_loss = view_loss + float(args.view_ce_weight) * cls_loss(
                            logit_view, y, ce_cw, args.loss, pos_weight,
                            gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
                        )
                    view_loss = view_loss + view_consistency_loss(
                        logit, g, logit_view, g_view, cl_cw,
                        logit_weight=getattr(args, "view_logit_weight", 0.0),
                        embed_weight=getattr(args, "view_embed_weight", 0.0),
                    )
            cl_val = torch.tensor(0.0, device=device)
            if cl_on and cl_mode == "supcon":
                cl_val = supcon_loss(g.float(), b, tau=args.tau)
            elif cl_on:
                cl_val = contrastive_loss(g.float(), b, bank, cl_cw, Kp=args.Kp, Kn=args.Kn,
                                          tau=args.tau, global_neg=args.global_neg,
                                          cl_c_min=args.cl_c_min,
                                          cl_neg_c_min=args.cl_neg_c_min,
                                          cl_teacher_mode=args.cl_teacher_mode,
                                          cl_teacher_conf_min=args.cl_teacher_conf_min,
                                          cl_neg_filter=getattr(args, "cl_neg_filter", "none"),
                                          cl_neg_bonus=getattr(args, "cl_neg_bonus", 0.0),
                                          cl_neg_bonus_filter=getattr(args, "cl_neg_bonus_filter", "none"),
                                          cl_attr_block=not getattr(args, "cl_no_attr_block", False),
                                          cl_class_balanced=getattr(args, "cl_class_balanced", False),
                                          cl_hard_pos=getattr(args, "cl_hard_pos", False))
            proto_loss = torch.tensor(0.0, device=device)
            proto_on = (
                getattr(args, "proto_aux_weight", 0.0) > 0
                and bank is not None
                and (
                    getattr(args, "proto_aux_in_warmup", False)
                    or epoch >= args.warmup
                )
            )
            if proto_on:
                proto_loss = prototype_aux_loss(g.float(), b, bank, cl_cw, args, device)
            loss = (ce + td + args.lambda_cl * cl_val + view_loss + aux_loss
                    + proto_loss + rel_loss) / accum
            loss.backward()
            if (bi + 1) % accum == 0:
                clip_params = [p for p in model.parameters() if p.requires_grad]
                if source_aux_heads is not None:
                    clip_params += [p for p in source_aux_heads.parameters() if p.requires_grad]
                if rel_aux_head is not None:
                    clip_params += [p for p in rel_aux_head.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            tot += 1; ce_tot += ce.item()
            td_tot += float(td.detach()) if torch.is_tensor(td) else float(td)
            cl_tot += float(cl_val.detach()) if torch.is_tensor(cl_val) else float(cl_val)
            view_tot += float(view_loss.detach()) if torch.is_tensor(view_loss) else float(view_loss)
            aux_tot += float(aux_loss.detach()) if torch.is_tensor(aux_loss) else float(aux_loss)
            proto_tot += float(proto_loss.detach()) if torch.is_tensor(proto_loss) else float(proto_loss)
        # 验证集模型选择（§4.1 用 val；按 Macro-F1 选最优 epoch，避免末轮过拟合）
        pv, _, yv, _, _ = predict(model, loaders["val"], device)
        vthr = best_threshold_macroF1(yv, pv)
        vf1 = macro_f1(yv, (pv >= vthr).astype(int))
        vap = average_precision_score(yv, pv) if len(set(yv)) > 1 else 0.0
        score = vf1 + 0.5 * vap
        print(f"[ep{epoch}] {'CL' if cl_on else 'warmup'} ce={ce_tot/tot:.4f} "
              f"td={td_tot/tot:.4f} cl={cl_tot/tot:.4f} view={view_tot/tot:.4f} "
              f"aux={aux_tot/tot:.4f} proto={proto_tot/tot:.4f} "
              f"val_mF1={vf1:.4f} val_ap={vap:.4f}", flush=True)
        # 仅对 CL 阶段（含最后一个 warmup epoch）的检查点入池，避免欠拟合早期权重污染
        if epoch >= max(0, args.warmup - 1):
            ckpts.append((score, snap()))

    # ---- 检查点选择：默认用最优单 val 检查点（保留高上限）；--swa 可选权重平均（降方差但压上限）。
    # 跨种子方差由 5 种子集成 + RKC 检索增强推理消除，效果优于 SWA（见消融 abl_swa）。
    if ckpts and args.swa:
        keys = list(ckpts[0][1].keys())
        swa = {k: sum(st[k] for _, st in ckpts) / len(ckpts) for k in keys}
        load_trainable(swa)
        print(f"[swa] averaged {len(ckpts)} CL-phase ckpts, val_score={val_score():.4f}", flush=True)
    elif ckpts:
        ckpts.sort(key=lambda x: -x[0])
        load_trainable(ckpts[0][1])
        print(f"[select] best single-val checkpoint score={ckpts[0][0]:.4f}", flush=True)

    if getattr(args, "save_ckpt", "") and not eval_only:
        torch.save(model.state_dict(), args.save_ckpt)
        print(f"[save_ckpt] -> {args.save_ckpt}", flush=True)
    _, train_pack = build_bank()
    res = evaluate(model, loaders, device, train_pack, tag=args.tag, seed=args.seed)
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
    if args.save_emb:
        # 完整导出三划分的 (g, p_cls, y, c, attr, pair_id)，供离线 ARF/集成/标定迭代（§3.4）。
        ptr, gtr_f, ytr, ctr, attr_tr = predict(model, loaders["train_eval"], device)
        pva, gva, yva, cva, attr_va = predict(model, loaders["val"], device)
        pte, gte, yte, cte, attr_te = predict(model, loaders["test"], device)
        pid = lambda s: [r.get("pair_id", "") for r in splits[s]]
        torch.save({
            "thr": res["thr"], "alpha_rkc": res.get("alpha_rkc", 1.0),
            "train": {"g": gtr_f, "p": ptr, "y": ytr, "c": ctr, "attr": attr_tr, "pair_id": pid("train")},
            "val": {"g": gva, "p": pva, "y": yva, "c": cva, "attr": attr_va, "pair_id": pid("val")},
            "test": {"g": gte, "p": pte, "y": yte, "c": cte, "attr": attr_te, "pair_id": pid("test")},
        }, args.save_emb)
        print(f"[save_emb] -> {args.save_emb}", flush=True)
    if return_model:
        return model, loaders, device, train_pack, res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DEFAULT_TRAIN_DATASET))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cl_epochs", type=int, default=6)
    ap.add_argument("--lambda_cl", type=float, default=0.5)  # 调参后的最优 canonical
    ap.add_argument("--pos_weight", type=float, default=-1.0)
    ap.add_argument("--loss", default="asl", choices=["bce", "focal", "asl"])
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    ap.add_argument("--gamma_pos", type=float, default=0.0)
    ap.add_argument("--no_cl", action="store_true")
    ap.add_argument("--cl_mode", default="racl", choices=["racl", "supcon", "none"],
                    help="racl=retrieval-augmented contrast (ours); supcon=faithful "
                         "in-batch supervised contrastive (Khosla et al., 2020); none=no contrast.")
    ap.add_argument("--swa", action="store_true", help="启用 SWA 权重平均（默认关闭；消融用）")
    ap.add_argument("--no_fusion", action="store_true")
    ap.add_argument("--n_fusion", type=int, default=2)
    ap.add_argument("--stream_mode", default="dual", choices=["dual", "claim", "evidence"],
                    help="单流消融：claim/evidence 把另一条流镜像为所选流（需配合 --no_fusion），"
                         "用于论证双流对照结构的必要性")
    ap.add_argument("--head_concat_only", action="store_true",
                    help="4-tuple 消融：分类/检索头仅用 [h_c,h_e] 拼接，去掉 ESIM 差/积交互特征")
    ap.add_argument("--joint_encode", action="store_true",
                    help="编码顺序消融：先把 claim+evidence 拼为单序列统一编码再拆回两流，"
                         "对照默认的独立编码再融合（融合/头不变）")
    ap.add_argument("--fusion_dropout", type=float, default=0.2)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--no_weight", action="store_true", help="退化为未加权 BCE")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.07)  # paper canonical
    ap.add_argument("--Kp", type=int, default=3)
    ap.add_argument("--Kn", type=int, default=5)
    ap.add_argument("--global_neg", action="store_true", help="全集合随机反标签（消融）")
    # ---- B 优化：去属性分块 + 类平衡监督对比 + 困难正例 ----
    ap.add_argument("--cl_no_attr_block", action="store_true",
                    help="去掉 RACL 的属性分块，正/负例都在全局标签池内挑选（消融证实属性分块作用很小）")
    ap.add_argument("--cl_class_balanced", action="store_true",
                    help="按 0.5/p_y 类频逆权重缩放每个 anchor 的对比损失，抵消正类被多数类淹没")
    ap.add_argument("--cl_hard_pos", action="store_true",
                    help="同标签正例取相似度最低的 Kp 个（hard positive，避免易正例梯度消失）")
    # ---- C 优化：可靠性建模 ----
    ap.add_argument("--rel_soft", action="store_true",
                    help="noise-aware 软标签：低可靠性 c 的标签向数据集基率收缩（弱监督去噪）")
    ap.add_argument("--rel_aux_weight", type=float, default=0.0,
                    help="可靠性辅助回归头权重：从检索表征预测样本可靠性 c 的 MSE；0 关闭")
    ap.add_argument("--rel_pow", type=float, default=1.0,
                    help="可靠性加权锐化指数：样本权重用 c^rel_pow（保均值）；1.0 关闭")
    ap.add_argument("--c_transform", default="none",
                    choices=["none", "uniform", "inverse", "permute", "binary",
                             "count", "oldc", "sqrt"],
                    help="可靠性权重设计合理性验证：对训练侧 c 做反事实变换（none=canonical）")
    ap.add_argument("--c_recompute", default="none",
                    help="c 公式自身超参敏感性：从对齐评论忠实重算 train 侧 c，"
                         "spec 如 'k=3,lambda=0.3,rho=0.4,phi=1.2,gamma=2'；none 关闭")
    ap.add_argument("--cl_c_min", type=float, default=0.0,
                    help="RACL anchor/positive 最低样本可信度；0 保持原始行为")
    ap.add_argument("--cl_neg_c_min", type=float, default=0.0,
                    help="RACL negative 最低样本可信度；0 保持原始行为")
    ap.add_argument("--cl_teacher_mode", default="off", choices=["off", "agree", "agree_pos"],
                    help="teacher-guided RACL：agree 过滤 anchor/positive/negative；agree_pos 仅过滤 anchor/positive")
    ap.add_argument("--cl_teacher_conf_min", type=float, default=0.0,
                    help="teacher-guided RACL 的 |p_teacher-0.5| 最低置信阈值")
    ap.add_argument("--cl_neg_filter", default="none",
                    choices=["none", "same_evtype", "same_evtype_conf", "medium_evtype_conf"],
                    help="RACL hard negative 过滤：优先选同证据组合/同置信层反标签近邻；none 保持原始行为")
    ap.add_argument("--cl_neg_bonus", type=float, default=0.0,
                    help="RACL hard negative 检索排序 bonus；>0 时让符合 cl_neg_bonus_filter 的候选更容易进 top-K")
    ap.add_argument("--cl_neg_bonus_filter", default="none",
                    choices=["none", "same_evtype", "same_evtype_conf", "medium_evtype_conf"],
                    help="RACL hard negative bonus 作用范围；不改变候选池，只改变近邻排序")
    ap.add_argument("--source0_ce_scale", type=float, default=1.0,
                    help="source_count=0 样本的 CE/distillation 权重缩放；1 保持原始行为")
    ap.add_argument("--source0_cl_scale", type=float, default=1.0,
                    help="source_count=0 样本的 RACL anchor 权重缩放；1 保持原始行为")
    ap.add_argument("--source_rich_ce_scale", type=float, default=1.0,
                    help="source_count>=2 或 source_len>=20 样本的 CE/distillation 权重缩放")
    ap.add_argument("--source_rich_cl_scale", type=float, default=1.0,
                    help="source_count>=2 或 source_len>=20 样本的 RACL anchor 权重缩放")
    ap.add_argument("--distill_bge_weight", type=float, default=0.0,
                    help="BGE+LR OOF teacher 软标签蒸馏权重；0 关闭")
    ap.add_argument("--distill_bge_folds", type=int, default=5,
                    help="训练折内部生成 BGE teacher OOF 概率的折数")
    ap.add_argument("--distill_teacher_seed", type=int, default=0,
                    help="BGE teacher 内层 OOF 划分随机种子")
    ap.add_argument("--distill_temp", type=float, default=1.0,
                    help="二分类 teacher/student logit 蒸馏温度")
    ap.add_argument("--distill_conf_min", type=float, default=0.0,
                    help="仅蒸馏 |p_teacher-0.5| 达到该阈值的样本")
    ap.add_argument("--distill_c_min", type=float, default=0.0,
                    help="仅蒸馏样本可信度 c 达到该阈值的训练样本")
    ap.add_argument("--distill_mode", default="all", choices=["all", "disagree"],
                    help="all=所有通过置信筛选的样本；disagree=仅 teacher/student 方向分歧样本")
    ap.add_argument("--backbone", default="bge", choices=["bge", "bert", "roberta"])
    ap.add_argument("--encoder_name", default="BAAI/bge-large-zh-v1.5",
                    help="backbone=bge 时的 encoder 名称，例如 BAAI/bge-small-zh-v1.5")
    ap.add_argument("--enc_train", default="lora", choices=["lora", "topk", "full"],
                    help="编码器训练范围：lora(默认)/topk(顶部若干层)/full(全参微调)")
    ap.add_argument("--unfreeze_top", type=int, default=0, help="topk 模式下解冻的顶层数")
    ap.add_argument("--no_ret_disc", action="store_true",
                    help="检索头退回 [h_c,h_e]（默认用 4 元组差/积，消融用）")
    ap.add_argument("--xattn_dir", default="both", choices=["both", "c2e", "e2c"])
    ap.add_argument("--indep_proj", action="store_true")
    ap.add_argument("--ffn", default="swiglu", choices=["swiglu", "gelu"])
    ap.add_argument("--evidence_policy", default=str(getattr(config, "EVIDENCE_POLICY_CANONICAL", "sources_only")),
                    choices=["", "record", "args_first", "source_first", "no_args",
                             "source_only", "sources_only", "args_only", "params_only",
                             "ocr_only", "vlm_only", "params_args", "ocr_args", "vlm_args"],
                    help="证据视图；论文公平口径默认 sources_only（仅三源，不用 arguments）")
    ap.add_argument("--evidence_policy_mix", default="",
                    help="逗号或空格分隔的 train-only evidence views；例如 source_first,no_args,ocr_only,params_only")
    ap.add_argument("--view_consistency_mix", default="",
                    help="逗号或空格分隔的辅助 evidence views；同一样本第二视图用于一致性/多视图正例")
    ap.add_argument("--view_ce_weight", type=float, default=0.0,
                    help="辅助 evidence view 的监督 CE 权重；0 关闭")
    ap.add_argument("--view_logit_weight", type=float, default=0.0,
                    help="辅助 view sigmoid(logit) 对齐主 view 的 MSE 权重；0 关闭")
    ap.add_argument("--view_embed_weight", type=float, default=0.0,
                    help="辅助 view retrieval embedding 对齐主 view 的 cosine loss 权重；0 关闭")
    ap.add_argument("--view_consistency_in_warmup", action="store_true",
                    help="默认只在 CL 阶段启用 view consistency；打开后 warmup 也启用")
    ap.add_argument("--source_aux_combo_weight", type=float, default=0.0,
                    help="检索表示预测 evidence_combo 的辅助损失权重；0 关闭")
    ap.add_argument("--source_aux_conf_weight", type=float, default=0.0,
                    help="检索表示预测 confidence bin 的辅助损失权重；0 关闭")
    ap.add_argument("--source_aux_count_weight", type=float, default=0.0,
                    help="检索表示预测粗 source_count bin 的辅助损失权重；0 关闭")
    ap.add_argument("--source_aux_in_warmup", action="store_true",
                    help="默认只在 CL 阶段启用 source auxiliary；打开后 warmup 也启用")
    ap.add_argument("--proto_aux_weight", type=float, default=0.0,
                    help="检索表示的正/负 prototype relation 辅助损失权重；0 关闭")
    ap.add_argument("--proto_aux_group", default="source_bin",
                    choices=["global", "attr", "source_bin", "evidence_combo",
                             "confidence", "combo_conf", "source_conf"],
                    help="prototype 分组粒度；默认 source_bin 对齐当前 RACL prototype 诊断")
    ap.add_argument("--proto_aux_mode", default="ce", choices=["ce", "margin"],
                    help="prototype auxiliary 目标：ce 为旧版 prototype 分类；margin 直接拉开正确类相似度 gap")
    ap.add_argument("--proto_aux_margin", type=float, default=0.15,
                    help="proto_aux_mode=margin 时要求正确类 prototype gap 达到的余弦相似度边际")
    ap.add_argument("--proto_aux_tau", type=float, default=0.10,
                    help="prototype relation CE 或 margin softplus 的相似度温度")
    ap.add_argument("--proto_aux_min_class", type=int, default=3,
                    help="分组内每类最少样本数；不足时回退 global prototype")
    ap.add_argument("--proto_aux_c_min", type=float, default=0.10,
                    help="构造 prototype 与 anchor 的最低样本可信度")
    ap.add_argument("--proto_aux_in_warmup", action="store_true",
                    help="默认只在 CL 阶段启用 prototype auxiliary；打开后 warmup 也启用")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_emb", default="")
    ap.add_argument("--save_ckpt", default="", help="保存完整 state_dict 供反事实推理复用")
    ap.add_argument("--load_ckpt", default="", help="加载 state_dict（配合 --eval_only / --infer_jsonl）")
    ap.add_argument("--eval_only", action="store_true", help="跳过训练，仅加载权重做评估/推理")
    ap.add_argument("--infer_jsonl", default="", help="用该 jsonl 替换 test 划分做构造样本推理")
    ap.add_argument("--tag", default="claimarc_full")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
