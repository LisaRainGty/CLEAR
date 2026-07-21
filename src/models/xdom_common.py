"""§4.4.2 跨域泛化（多模型 × 多指标）共享工具。

划分协议（与 crossdomain.py / crossdomain_time.py 完全一致）：
  - category : 留一一级品类（10 折，宏平均）；
  - rooms    : 留出一组 room_id（按评论规模分层选 20 个）；
  - time     : 时间增量（评论最早 review_time 作时间代理，cutoff1/cutoff2 切分）。

所有协议都汇总 train/val/test 三个划分后重新分区，忽略记录里预置的 split 字段。
目标域全部作为只读 test，其余源域样本再按 room_id 整组拆成 train/val；
保证 CLAIMARC / BERT / RoBERTa / ESIM / LLM 拿到完全相同且无直播间交叉的源域划分。
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from models.data import load_split
from models.train import best_threshold_macroF1

CATEGORIES = [
    "apparel_and_underwear", "general", "baby_kids_and_pets", "shoes_and_bags",
    "food_and_beverages", "smart_home", "digital_and_electronics",
    "sports_and_outdoor", "beauty_and_personal_care", "jewelry_and_collectibles",
]


def _parse_time(s):
    s = str(s or "").strip()
    if not s:
        return None
    s = re.split(r"\s+", s)[0]
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _pair_time(r):
    ts = []
    for m in (r.get("_aligned_consumer_mentions", []) or []):
        t = _parse_time(m.get("review_time"))
        if t:
            ts.append(t)
    return min(ts) if ts else None


def holdout_rooms(dataset, n=20):
    """按评论规模分层选 n 个 room_id（与 run_paper.holdout_rooms 一致）。"""
    import collections
    import json
    vol = collections.defaultdict(int)
    with open(dataset, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("y") is None:
                continue
            vol[r.get("room_id", "")] += int(r.get("_consumer_mentions_total", 0) or 0) + 1
    rooms = sorted([k for k in vol if k], key=lambda k: vol[k], reverse=True)
    idx = np.linspace(0, len(rooms) - 1, n).round().astype(int)
    return [rooms[i] for i in sorted(set(idx.tolist()))]


def _grouped_source_split(records, seed, val_fraction=0.10):
    """Create a deterministic room-disjoint source train/validation split."""
    by_room = defaultdict(list)
    for row in records:
        room = str(row.get("room_id", ""))
        if not room:
            raise ValueError("cross-domain source record is missing room_id")
        by_room[room].append(row)
    rooms = sorted(by_room)
    rng = np.random.RandomState(seed)
    rng.shuffle(rooms)
    n_val_rooms = max(1, int(round(len(rooms) * val_fraction)))
    val_rooms = rooms[:n_val_rooms]
    val_set = set(val_rooms)
    train = [row for row in records if str(row.get("room_id", "")) not in val_set]
    val = [row for row in records if str(row.get("room_id", "")) in val_set]
    if not train or not val:
        raise ValueError("unable to create non-empty room-disjoint source split")
    if {int(row.get("y", 0)) for row in train} != {0, 1}:
        raise ValueError("source train split does not contain both labels")
    if {int(row.get("y", 0)) for row in val} != {0, 1}:
        raise ValueError("source validation split does not contain both labels")
    return train, val


def build_splits(dataset, mode, holdout="", seed=0,
                 cutoff1="2025-01-01", cutoff2="2025-02-15"):
    full = load_split(dataset)
    allrecs = full["train"] + full["val"] + full["test"]
    rng = np.random.RandomState(seed)
    if mode == "category":
        held = [r for r in allrecs if r.get("category") == holdout]
        rest = [r for r in allrecs if r.get("category") != holdout]
        train, val = _grouped_source_split(rest, seed)
        return {"train": train, "val": val, "test": held}
    if mode == "rooms":
        held_set = {x.strip() for x in holdout.split(",") if x.strip()}
        held = [r for r in allrecs if r.get("room_id") in held_set]
        rest = [r for r in allrecs if r.get("room_id") not in held_set]
        train, val = _grouped_source_split(rest, seed)
        return {"train": train, "val": val, "test": held}
    if mode == "time":
        c1, c2 = _parse_time(cutoff1), _parse_time(cutoff2)
        early, late, bg = [], [], []
        for r in allrecs:
            if r.get("sample_role") == "objective_negative":
                bg.append(r); continue
            t = _pair_time(r)
            if t is None:
                bg.append(r)
            elif t < c1:
                early.append(r)
            elif t < c2:
                bg.append(r)  # [cutoff1,cutoff2) 近期池并入背景训练（多模型设定下不做 RKC 注入）
            else:
                late.append(r)
        train_recs = early + bg
        rng.shuffle(train_recs)
        nval = max(50, len(train_recs) // 10)
        return {"train": train_recs[nval:], "val": train_recs[:nval], "test": late}
    raise ValueError(mode)


def full_metrics(y_val, p_val, y, p):
    """val 上按 Macro-F1 选阈值，test 上算完整指标。p/y 为 numpy。"""
    y_val = np.asarray(y_val); p_val = np.asarray(p_val)
    y = np.asarray(y); p = np.asarray(p)
    thr = best_threshold_macroF1(y_val, p_val) if len(set(y_val.tolist())) > 1 else 0.5
    pred = (p >= thr).astype(int)
    two = len(set(y.tolist())) > 1
    return {
        "thr": float(thr),
        "acc": 100 * accuracy_score(y, pred),
        "prec": 100 * precision_score(y, pred, zero_division=0),
        "rec": 100 * recall_score(y, pred, zero_division=0),
        "f1pos": 100 * f1_score(y, pred, zero_division=0),
        "macro_f1": 100 * f1_score(y, pred, average="macro", zero_division=0),
        "auprc": 100 * average_precision_score(y, p) if two else float("nan"),
        "auroc": 100 * roc_auc_score(y, p) if two else float("nan"),
        "n": int(len(y)), "pos": int(y.sum()),
    }
