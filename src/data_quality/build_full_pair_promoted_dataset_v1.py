"""Promote full-pair LLM/VLM reconstruction reviews into dataset views.

The promotion step is intentionally conservative:

- LLM output is not written directly as a training dataset.
- Every reviewed row is preserved in a stateful audit view.
- The main supervised candidate only includes complete rows where a recovered
  claim, product evidence, and at least one claim-aligned consumer comment are
  available.
- Missing or ambiguous rows become repair/silver states instead of being
  deleted or converted into clean negatives.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import read_jsonl, write_json, write_jsonl
from data_quality.rebuild_repaired_datasets_v1 import assign_room_splits, split_leakage


IDENTITY_VALUE_ATTRS = {"品牌", "型号", "货号", "条码", "条形码", "执行标准"}
PRICE_ATTR_TERMS = {"价格", "售价", "到手价"}
QUANTITY_ATTR_TERMS = {"净含量", "数量", "包数", "袋数", "件数", "重量", "容量", "尺寸"}
PRICE_VALUE_JUDGMENT_TERMS = {"太贵", "偏贵", "不便宜", "小贵", "不值", "物没价廉", "价格合理", "合理性"}
PRICE_OVERCHARGE_CUES = {"多收", "贵了", "贵", "涨价", "不是这个价", "价格不符", "实付", "到手不是", "付款"}
QUANTITY_VALUE_JUDGMENT_TERMS = {"太少", "少的可怜", "量少", "分量少", "不多", "最少也得", "应该", "应为", "不够"}
NUMERIC_CONFLICT_CUES = {"不是", "不符", "少发", "少给", "只有", "收到少", "实付", "到手不是", "降价", "买成"}
COMMERCIAL_PROMISE_ATTRS = {
    "售卖方式",
    "购买渠道",
    "广告宣传",
    "宣传",
    "活动信息",
    "促销活动",
    "活动规则",
    "直播间优惠信息",
    "直播活动信息",
    "优惠券规则",
    "赔偿金额",
    "官方说辞",
    "广告词",
    "广告参与",
    "过度营销",
    "价格欺诈",
    "活动价格",
    "促销价格",
    "赠品溢价程度",
}
GENERIC_ATTRIBUTE_NAMES = {
    "描述",
    "描述相符",
    "商品描述",
    "广告宣传",
    "宣传",
    "产品",
    "商品",
    "商品属性",
    "属性",
    "属性名",
    "客观属性名词短语",
}
SCHEMA_META_ATTRS = {
    "视频内容",
    "直播内容",
    "详情页",
    "图文详情",
    "实物图描述",
    "与事实不符",
    "描述不符",
    "视频描述",
    "宣传内容",
    "直播宣传效果",
    "宣传效果",
    "宣传力度",
    "广告投入",
    "夸大其词",
}
SCHEMA_META_ATTR_TERMS = {"宣传与实物匹配度"}
SUBJECTIVE_EVAL_ATTR_TERMS = {
    "智商税",
    "虚假宣传",
    "商品质量",
    "真实性评价",
    "性价比",
    "体验",
    "感受",
    "评价",
    "推荐",
    "购买意图",
    "消费者心理",
    "商家信誉",
    "是否上当",
    "是否忽悠",
    "廉价感",
    "价值判断",
    "好评真实性",
    "宣传真实性",
    "购买体验",
    "假货",
    "真假",
    "真伪",
    "社交效果",
    "利用价值",
    "价格透明度",
}
COLOR_ATTR_TERMS = {"颜色", "包装颜色", "色"}
EXPECTATION_MISMATCH_CUES = {"以为", "以爲", "没看清", "本来是买", "本来想买", "结果来一看"}
COUNT_UNIT_CUES = {"个", "颗", "件", "瓶", "袋", "双", "盒", "片", "支", "只", "排"}
BATTERY_CAPACITY_UNIT_CUES = {"mah", "毫安", "安时", "ah"}
NON_TEXTILE_PRODUCT_TERMS = {"洗发", "沐浴", "含片", "豆奶", "粉条", "电池", "眼镜", "隔离霜", "脱毛仪"}
EXHAUSTIVE_ENUM_CUES = {
    "两个颜色",
    "两种颜色",
    "2个颜色",
    "2种颜色",
    "只有",
    "只",
    "一共",
    "总共",
    "就这",
}
COLOR_VALUES = {
    "白色": "白",
    "白底": "白",
    "红色": "红",
    "红底": "红",
    "绿色": "绿",
    "绿底": "绿",
    "蓝色": "蓝",
    "蓝底": "蓝",
    "黑色": "黑",
    "黑底": "黑",
    "黄色": "黄",
    "黄底": "黄",
    "橙色": "橙",
    "橙底": "橙",
    "粉色": "粉",
    "粉底": "粉",
    "紫色": "紫",
    "紫底": "紫",
    "灰色": "灰",
    "灰底": "灰",
    "银色": "银",
    "金色": "金",
    "棕色": "棕",
    "咖啡色": "咖啡",
    "米色": "米",
    "透明": "透明",
    "卡其": "卡其",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or f"p{row.get('product_id')}__{row.get('attribute_id')}")


def read_by_pair(path: str | Path) -> dict[str, dict[str, Any]]:
    if not Path(path).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[pair_id(row)] = row
    return out


def aligned_comments(queue_row: dict[str, Any], review: dict[str, Any]) -> list[dict[str, Any]]:
    mentions = queue_row.get("consumer_mentions") or []
    out: list[dict[str, Any]] = []
    for j in review.get("comment_judgments") or []:
        try:
            cid = int(j.get("cid", 0) or 0)
        except Exception:
            cid = 0
        if cid < 1 or cid > len(mentions):
            continue
        m = dict(mentions[cid - 1])
        m["_judgment"] = {
            "cid": cid,
            "aligned_to_claim": bool(j.get("aligned_to_claim")),
            "relation": clean(j.get("relation")),
            "reason": clean(j.get("reason")),
        }
        if m["_judgment"]["aligned_to_claim"]:
            out.append(m)
    return out


def relation_counts(aligned: list[dict[str, Any]]) -> Counter:
    return Counter(clean(m.get("_judgment", {}).get("relation")) for m in aligned)


def compact(text: Any) -> str:
    return "".join(clean(text).split()).lower()


def identity_expected_values(queue_row: dict[str, Any]) -> list[str]:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if attr not in IDENTITY_VALUE_ATTRS:
        return []
    raw_params = queue_row.get("raw_params") or {}
    values: list[str] = []
    if isinstance(raw_params, dict):
        for key, val in raw_params.items():
            key_s = clean(key).strip("<>")
            if attr == key_s or attr in key_s or key_s in attr:
                text = clean(val)
                if 1 < len(text) <= 80:
                    values.append(text)
    return list(dict.fromkeys(values))


def identity_claim_lacks_value(queue_row: dict[str, Any], review: dict[str, Any]) -> bool:
    vals = identity_expected_values(queue_row)
    if not vals or not review.get("claim_found"):
        return False
    claim_norm = compact(review.get("claim_text"))
    return not any(compact(v) and compact(v) in claim_norm for v in vals)


def numeric_value_judgment_refutes(queue_row: dict[str, Any], review: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    is_price = any(t in attr for t in PRICE_ATTR_TERMS)
    is_quantity = any(t in attr for t in QUANTITY_ATTR_TERMS)
    if not is_price and not is_quantity:
        return False
    mentions = queue_row.get("consumer_mentions") or []
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("aligned_to_claim") or clean(item.get("relation")) != "refute":
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        text = clean(mentions[cid - 1].get("evidence_span")) if 1 <= cid <= len(mentions) else ""
        reason = clean(item.get("reason"))
        blob = text + " " + reason
        if any(cue in blob for cue in NUMERIC_CONFLICT_CUES):
            continue
        if is_price and any(term in blob for term in PRICE_VALUE_JUDGMENT_TERMS):
            return True
        if is_quantity and any(term in blob for term in QUANTITY_VALUE_JUDGMENT_TERMS):
            return True
    return False


def extract_price_values(text: Any) -> list[float]:
    blob = clean(text)
    vals: list[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:元|块|¥|￥)", blob):
        vals.append(float(m.group(1)))
    for m in re.finditer(r"(\d+)\s*(?:元|块)\s*(\d)", blob):
        vals.append(float(f"{m.group(1)}.{m.group(2)}"))
    return vals


def price_comment_not_refuting_claim_value(queue_row: dict[str, Any], review: dict[str, Any], rel: Counter) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if not any(t in attr for t in PRICE_ATTR_TERMS) or rel.get("refute", 0) <= 0:
        return False
    claim_prices = extract_price_values(review.get("claim_text"))
    if not claim_prices:
        return False
    claim_price = max(claim_prices)
    mentions = queue_row.get("consumer_mentions") or []
    comment_prices: list[float] = []
    overcharge_text = False
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("aligned_to_claim") or clean(item.get("relation")) != "refute":
            continue
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        span = clean(mentions[cid - 1].get("evidence_span")) if 1 <= cid <= len(mentions) else ""
        blob = span + " " + clean(item.get("reason"))
        comment_prices.extend(extract_price_values(blob))
        if any(cue in blob for cue in PRICE_OVERCHARGE_CUES):
            overcharge_text = True
    if not comment_prices:
        return False
    if max(comment_prices) <= claim_price * 1.05 and not overcharge_text:
        return True
    if max(comment_prices) <= claim_price * 1.05 and min(comment_prices) < claim_price * 0.8:
        return True
    return False


def commercial_promise_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    return attr in COMMERCIAL_PROMISE_ATTRS


def subjective_eval_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    return any(term in attr for term in SUBJECTIVE_EVAL_ATTR_TERMS)


def schema_meta_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if attr in SCHEMA_META_ATTRS or attr in GENERIC_ATTRIBUTE_NAMES:
        return True
    return any(term in attr for term in SCHEMA_META_ATTR_TERMS)


def consumer_expectation_mismatch(queue_row: dict[str, Any], review: dict[str, Any], rel: Counter) -> bool:
    if not (rel.get("support", 0) > 0 and rel.get("refute", 0) > 0):
        return False
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    if not any(term in attr for term in {"数量", "规格", "包装", "产品名称"}):
        return False
    mentions = queue_row.get("consumer_mentions") or []
    refute = 0
    expectation = 0
    for item in review.get("comment_judgments") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("aligned_to_claim") or clean(item.get("relation")) != "refute":
            continue
        refute += 1
        try:
            cid = int(item.get("cid", 0) or 0)
        except Exception:
            cid = 0
        span = clean(mentions[cid - 1].get("evidence_span")) if 1 <= cid <= len(mentions) else ""
        blob = span + " " + clean(item.get("reason"))
        if any(cue in blob for cue in EXPECTATION_MISMATCH_CUES):
            expectation += 1
    return bool(refute and expectation / refute >= 0.5)


def attribute_semantic_drift(queue_row: dict[str, Any], review: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>").lower()
    if "电池容量" not in attr:
        if "面料" in attr:
            title = clean(queue_row.get("product_title"))
            return any(term in title for term in NON_TEXTILE_PRODUCT_TERMS)
        return False
    blob = (clean(review.get("claim_text")) + " " + clean(review.get("evidence_text"))).lower()
    has_count_unit = any(cue in blob for cue in COUNT_UNIT_CUES)
    has_capacity_unit = any(cue in blob for cue in BATTERY_CAPACITY_UNIT_CUES)
    return bool(has_count_unit and not has_capacity_unit)


def conflicting_comment_relation(rel: Counter) -> bool:
    return bool(rel.get("support", 0) > 0 and rel.get("refute", 0) > 0)


def color_attr(queue_row: dict[str, Any]) -> bool:
    attr = clean(queue_row.get("attribute_name")).strip("<>")
    return any(term in attr for term in COLOR_ATTR_TERMS)


def color_values(text: Any) -> set[str]:
    blob = clean(text)
    return {value for token, value in COLOR_VALUES.items() if token in blob}


def exhaustive_enum_claim(text: Any) -> bool:
    blob = clean(text)
    return any(cue in blob for cue in EXHAUSTIVE_ENUM_CUES)


def enumeration_claim_evidence_extra_values(queue_row: dict[str, Any], review: dict[str, Any]) -> bool:
    """Catch enumeration claims whose product evidence exposes extra options.

    These rows are not discarded. They are held in silver because the product
    evidence no longer cleanly supports the exact主播 claim being judged.
    """
    if not color_attr(queue_row):
        return False
    claim_text = clean(review.get("claim_text"))
    evidence_text = clean(review.get("evidence_text"))
    if not exhaustive_enum_claim(claim_text):
        return False
    claim_vals = color_values(claim_text)
    evidence_vals = color_values(evidence_text)
    return bool(claim_vals and evidence_vals and (evidence_vals - claim_vals))


def confidence_score(value: str) -> float:
    value = clean(value).lower()
    if value == "high":
        return 0.08
    if value == "medium":
        return 0.04
    return 0.0


def reliability(queue_row: dict[str, Any], review: dict[str, Any], aligned: list[dict[str, Any]]) -> float:
    score = 0.05
    if review.get("claim_found"):
        score += 0.18
    if review.get("product_evidence_found"):
        score += 0.15
    if aligned:
        score += 0.15
    if any(clean(m.get("_judgment", {}).get("relation")) == "refute" for m in aligned):
        score += 0.05
    if any(clean(m.get("_judgment", {}).get("relation")) == "support" for m in aligned):
        score += 0.04
    score += min(0.06, 0.015 * max(0, len(aligned) - 1))
    score += min(0.06, 0.02 * sum(1 for m in aligned if m.get("explicit_fact_hit")))
    score += confidence_score(review.get("confidence", ""))
    if queue_row.get("old_label_state") == "label_positive_claim_aligned_neg" and int(review.get("new_y", 0) or 0) == 1:
        score += 0.02
    return round(max(0.03, min(0.88, score)), 4)


def promotion_state(queue_row: dict[str, Any], review: dict[str, Any], rel: Counter) -> str:
    if review.get("__error__"):
        return "llm_error"
    claim_found = bool(review.get("claim_found"))
    evidence_found = bool(review.get("product_evidence_found"))
    claim_evidence_relation = clean(review.get("claim_evidence_relation"))
    if not claim_found:
        return "repair_missing_claim"
    if not evidence_found:
        if rel.get("refute", 0) > 0:
            return "silver_refute_missing_product_evidence"
        return "repair_missing_evidence"
    if claim_evidence_relation in {"", "insufficient"}:
        if rel.get("refute", 0) > 0:
            return "silver_refute_insufficient_product_evidence"
        return "repair_insufficient_product_evidence"
    if identity_claim_lacks_value(queue_row, review):
        return "repair_identity_claim_value"
    if numeric_value_judgment_refutes(queue_row, review):
        return "repair_numeric_value_judgment"
    if price_comment_not_refuting_claim_value(queue_row, review, rel):
        return "silver_price_value_not_direct_refute"
    if commercial_promise_attr(queue_row):
        return "silver_commercial_promise_attribute"
    if subjective_eval_attr(queue_row):
        return "silver_subjective_eval_attribute"
    if schema_meta_attr(queue_row):
        return "silver_schema_meta_attribute"
    if consumer_expectation_mismatch(queue_row, review, rel):
        return "silver_consumer_expectation_mismatch"
    if attribute_semantic_drift(queue_row, review):
        return "silver_attribute_semantic_drift"
    if conflicting_comment_relation(rel):
        return "silver_conflicting_comment_relation"
    if enumeration_claim_evidence_extra_values(queue_row, review):
        return "silver_enumeration_evidence_extra_values"
    if rel.get("refute", 0) > 0:
        return "main_positive_refute"
    if rel.get("support", 0) > 0 and rel.get("mixed", 0) == 0:
        return "main_negative_support"
    if rel.get("mixed", 0) > 0:
        return "silver_mixed_comment_relation"
    return "lowinfo_no_aligned_comment"


def evidence_payload(review: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    src_type = clean(review.get("evidence_source_type"))
    text = clean(review.get("evidence_text"))
    source = clean(review.get("evidence_source"))
    if not text:
        return [], [], []
    item = {
        "raw_text": text,
        "source": source,
        "_source_type": src_type,
        "_reconstructed": True,
    }
    if src_type in {"params", "product_title"}:
        item["param_key"] = source or src_type
        return [item], [], []
    if src_type == "detail_image_ocr":
        item["image_path"] = source
        return [], [item], []
    if src_type == "detail_image_vlm":
        item["image_path"] = source
        item["raw_quote"] = text
        return [], [], [item]
    return [item], [], []


def build_row(queue_row: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    aligned = aligned_comments(queue_row, review)
    rel = relation_counts(aligned)
    state = promotion_state(queue_row, review, rel)
    y = 1 if state == "main_positive_refute" else 0
    c = reliability(queue_row, review, aligned)
    ev_params, ev_ocr, ev_vlm = evidence_payload(review)
    claim_text = clean(review.get("claim_text"))
    row = {
        "pair_id": pair_id(queue_row),
        "product_id": queue_row.get("product_id"),
        "room_id": queue_row.get("room_id"),
        "category": queue_row.get("category"),
        "subcategory": queue_row.get("subcategory"),
        "attribute_id": queue_row.get("attribute_id"),
        "attribute_name": queue_row.get("attribute_name"),
        "product_title": queue_row.get("product_title"),
        "y": y,
        "c": c,
        "confidence": clean(review.get("confidence")) or "low",
        "claim": {
            "has_claim_srt": bool(review.get("claim_found")),
            "passage": claim_text,
            "segments": [
                {
                    "claim_id": f"{pair_id(queue_row)}__fullpair_llm_v1",
                    "clip_id": clean(review.get("claim_source")),
                    "start_ts": clean(review.get("claim_timestamp")),
                    "end_ts": "",
                    "text": claim_text,
                    "_reconstructed": True,
                }
            ] if claim_text else [],
        },
        "evidence_params": ev_params,
        "evidence_ocr": ev_ocr,
        "evidence_vlm": ev_vlm,
        "coverage": int(bool(ev_params)) + int(bool(ev_ocr)) + int(bool(ev_vlm)),
        "evidence_count": len(ev_params) + len(ev_ocr) + len(ev_vlm),
        "label_audit": {
            "policy": "full_pair_reconstruction_v1",
            "promotion_state": state,
            "comment_relation_counts": dict(rel),
            "aligned_comment_count": len(aligned),
            "label_basis": clean(review.get("label_basis")),
            "llm_action": clean(review.get("action")),
            "claim_evidence_relation": clean(review.get("claim_evidence_relation")),
            "old_y": queue_row.get("old_y"),
            "old_c": queue_row.get("old_c"),
            "old_label_state": queue_row.get("old_label_state"),
            "old_label_role": "audit_only_not_final",
        },
        "_full_pair_queue_type": queue_row.get("queue_type"),
        "_full_pair_priority": queue_row.get("priority"),
        "_full_pair_reconstruction_flags": queue_row.get("reconstruction_flags"),
        "_llm_review": {
            "model": review.get("model"),
            "claim_found": review.get("claim_found"),
            "product_evidence_found": review.get("product_evidence_found"),
            "raw_new_y": review.get("raw_new_y"),
            "clean_new_y": review.get("new_y"),
        },
        "_aligned_consumer_mentions": aligned,
        "_consumer_mentions_total": queue_row.get("consumer_mentions_total"),
        "_consumer_mentions_neg": queue_row.get("consumer_mentions_neg"),
        "_attribute_scope": "product_attribute",
    }
    return row


def is_main(row: dict[str, Any]) -> bool:
    state = clean((row.get("label_audit") or {}).get("promotion_state"))
    return state in {"main_positive_refute", "main_negative_support"}


def claim_text_norm(row: dict[str, Any]) -> str:
    claim = ((row.get("claim") or {}).get("passage") or "")
    return compact(claim)


def claim_family_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        clean(row.get("product_id")),
        clean(row.get("room_id")),
        claim_text_norm(row),
    )


def claim_family_score(row: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
    attr = clean(row.get("attribute_name")).strip("<>")
    audit = row.get("label_audit") or {}
    aligned = int(audit.get("aligned_comment_count") or 0)
    evidence_count = int(row.get("evidence_count") or 0)
    return (
        0 if attr in GENERIC_ATTRIBUTE_NAMES else 1,
        evidence_count,
        aligned,
        1 if int(row.get("y", 0) or 0) == 1 else 0,
        len(attr),
        clean(row.get("pair_id")),
    )


def same_claim_family(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if clean(a.get("product_id")) != clean(b.get("product_id")):
        return False
    if clean(a.get("room_id")) != clean(b.get("room_id")):
        return False
    ca = claim_text_norm(a)
    cb = claim_text_norm(b)
    if len(ca) < 5 or len(cb) < 5:
        return False
    short, long = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    return short in long


def claim_family_components(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    remaining = set(range(len(rows)))
    out: list[list[dict[str, Any]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            i = stack.pop()
            for j in list(remaining):
                if same_claim_family(rows[i], rows[j]):
                    remaining.remove(j)
                    comp.add(j)
                    stack.append(j)
        out.append([rows[i] for i in sorted(comp)])
    return out


def apply_claim_family_dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    main_candidates = [row for row in rows if is_main(row) and claim_text_norm(row)]
    groups = claim_family_components(main_candidates)
    demoted = 0
    conflicting_groups = 0
    examples: list[dict[str, Any]] = []
    for vals in groups:
        if len(vals) < 2:
            continue
        labels = {int(row.get("y", 0) or 0) for row in vals}
        if len(labels) > 1:
            conflicting_groups += 1
            kept = max(vals, key=claim_family_score)
            for row in vals:
                audit = row.setdefault("label_audit", {})
                before = clean(audit.get("promotion_state"))
                audit["promotion_state_before_dedupe"] = before
                audit["promotion_state"] = "silver_conflicting_claim_family"
                audit["duplicate_claim_family_reference_pair_id"] = kept.get("pair_id")
                row["y"] = 0
                demoted += 1
                if len(examples) < 20:
                    examples.append({
                        "demoted_pair_id": row.get("pair_id"),
                        "kept_pair_id": kept.get("pair_id"),
                        "attribute_name": row.get("attribute_name"),
                        "kept_attribute_name": kept.get("attribute_name"),
                        "state_before": before,
                        "reason": "conflicting_labels",
                    })
            continue
        kept = max(vals, key=claim_family_score)
        for row in vals:
            if row is kept:
                continue
            audit = row.setdefault("label_audit", {})
            before = clean(audit.get("promotion_state"))
            audit["promotion_state_before_dedupe"] = before
            audit["promotion_state"] = "silver_duplicate_claim_family"
            audit["duplicate_claim_family_kept_pair_id"] = kept.get("pair_id")
            audit["duplicate_claim_family_key"] = "|".join(claim_family_key(kept))
            row["y"] = 0
            demoted += 1
            if len(examples) < 20:
                examples.append({
                    "demoted_pair_id": row.get("pair_id"),
                    "kept_pair_id": kept.get("pair_id"),
                    "attribute_name": row.get("attribute_name"),
                    "kept_attribute_name": kept.get("attribute_name"),
                    "state_before": before,
                    "reason": "duplicate_same_label",
                })

    return rows, {
        "duplicate_claim_family_groups": sum(1 for vals in groups if len(vals) > 1),
        "conflicting_claim_family_groups": conflicting_groups,
        "duplicate_claim_family_demoted": demoted,
        "duplicate_claim_family_examples": examples,
    }


def summarize(
    rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    missing_reviews: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "reviewed_rows": len(rows),
        "main_rows": len(main_rows),
        "missing_reviews": missing_reviews,
        "all_labels": dict(Counter(int(r.get("y", 0)) for r in rows)),
        "main_labels": dict(Counter(int(r.get("y", 0)) for r in main_rows)),
        "promotion_state": dict(Counter(clean((r.get("label_audit") or {}).get("promotion_state")) for r in rows)),
        "confidence": dict(Counter(clean(r.get("confidence")) for r in rows)),
        "main_split": dict(Counter(clean(r.get("split")) for r in main_rows)),
        "main_split_leakage": split_leakage(main_rows) if main_rows else {},
        "category": dict(Counter(clean(r.get("category")) for r in main_rows)),
    }
    if extra:
        report.update(extra)
    return report


def write_markdown(path: str | Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Full Pair Promoted Dataset v1",
        "",
        "This is the promotion report for LLM/VLM full-pair reconstruction reviews.",
        "The main candidate is conservative; stateful rows preserve all reviewed hard cases.",
        "",
        "## Inputs",
        "",
        f"- queue: `{args.queue}`",
        f"- reviews: `{args.reviews}`",
        "",
        "## Outputs",
        "",
        f"- stateful reviewed rows: `{args.out_all}`",
        f"- main supervised candidate: `{args.out_main}`",
        f"- repair/silver rows: `{args.out_repair}`",
        f"- report json: `{args.report}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Promotion Rule",
        "",
        "- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.",
        "- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.",
        "- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl")
    ap.add_argument("--reviews", default="data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl")
    ap.add_argument("--out_all", default="data/final/repaired_v1/dataset_full_pair_reconstruction_stateful_v1_20260614.jsonl")
    ap.add_argument("--out_main", default="data/final/repaired_v1/dataset_full_pair_reconstruction_main_v1_20260614.jsonl")
    ap.add_argument("--out_repair", default="data/final/repaired_v1/full_pair_reconstruction_repair_silver_v1_20260614.jsonl")
    ap.add_argument("--report", default="data/final/repaired_v1/full_pair_reconstruction_promotion_v1_20260614.report.json")
    ap.add_argument("--markdown", default="docs/FULL_PAIR_PROMOTION_REPORT_20260614.md")
    args = ap.parse_args()

    queue = read_by_pair(args.queue)
    reviews = read_by_pair(args.reviews)
    rows: list[dict[str, Any]] = []
    missing_reviews = 0
    for pid, qrow in queue.items():
        review = reviews.get(pid)
        if not review:
            missing_reviews += 1
            continue
        rows.append(build_row(qrow, review))

    rows, dedupe_report = apply_claim_family_dedupe(rows) if rows else (rows, {})
    rows = assign_room_splits(rows) if rows else []
    main_rows = assign_room_splits([r for r in rows if is_main(r)]) if rows else []
    repair_rows = [r for r in rows if not is_main(r)]
    write_jsonl(args.out_all, rows)
    write_jsonl(args.out_main, main_rows)
    write_jsonl(args.out_repair, repair_rows)
    report = summarize(rows, main_rows, missing_reviews, dedupe_report)
    report.update({
        "queue": args.queue,
        "reviews": args.reviews,
        "out_all": args.out_all,
        "out_main": args.out_main,
        "out_repair": args.out_repair,
    })
    write_json(args.report, report)
    write_markdown(args.markdown, report, args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
