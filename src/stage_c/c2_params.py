"""Stage C2 — 逐属性 × 产品参数证据抽取。

以 attribute 为索引，对每个 attribute_id 在该商品 params 字典中收集语义对应的 (key,value)：
  1) alias 反查（无 LLM）：CAS+ 中该属性 aliases∪canonical_name 与 params key 归一化匹配；
  2) LLM 兜底（仅 alias 反查仍空的属性）：把漏网属性 + 全部 params 送 LLM 一次匹配。
产出 data/processed/stageC/evidence_params.json：{product_id: {attribute_id: [{param_key, raw_text}]}}

用法：python -m stage_c.c2_params [--no-llm]
     python -m stage_c.c2_params --acmt data/processed/stageB_product_v2/acmt_product_v2.json --stage_a_dir data/processed/stageA_repaired_v1 --out data/processed/stageB_product_v2/evidence_params_product_v2.json
"""
from __future__ import annotations

import argparse
import os

import config
from common import llm
from common import product_index as pidx
from common.io_utils import normalize, read_json, write_json

# BGE 语义参数匹配：默认关闭。审计显示 BGE-large-zh 对中文短属性名相似度普遍偏高
# （品牌/产品≈0.75、产地/保质期≈0.62），正确匹配与错误匹配分数重叠、无法用阈值切分，
# 精度不足以进弱监督证据。保留代码但默认不启用；真正的语义匹配交给 LLM 兜底。
C2_USE_BGE = os.environ.get("CLAIMARC_C2_BGE", "0") == "1"
C2_BGE_THRESHOLD = float(os.environ.get("CLAIMARC_C2_BGE_TAU", "0.80"))

C2_PROMPT = """角色：电商产品参数到属性的语义匹配员。
任务：对下列每个 attribute_id，从 params 字典中挑出语义对应的所有条目（多对一，全部列出）；
没有任何条目能对应的，返回空列表。
硬约束：只允许从下方 params 中取条目，禁止改写或自创取值；attribute_id 必须取自候选列表。
若 "商品标题" 中直接包含颜色、规格、型号、容量、数量等客观信息，也可作为对应属性证据。

候选属性（attribute_id | 标准名）：
{attrs}

params 字典：
{params}

输出严格 JSON 数组：
[{{"attribute_id": "...", "matches": [{{"param_key": "...", "raw_text": "..."}}]}}]
只输出 JSON 数组。"""


def alias_lookup(acmt_p: dict, params: dict, cas_by_id: dict) -> dict:
    norm_params = {k: str(v) for k, v in params.items()}
    out: dict[str, list] = {aid: [] for aid in acmt_p}
    for aid in acmt_p:
        meta = cas_by_id.get(aid, {})
        patterns = set()
        for s in [meta.get("canonical_name", "")] + list(meta.get("aliases", [])):
            n = normalize(s)
            if n:
                patterns.add(n)
        for k, v in norm_params.items():
            nk = normalize(k)
            if not nk:
                continue
            if nk in patterns or any(p and (p in nk or nk in p) for p in patterns):
                out[aid].append({"param_key": k, "raw_text": v})
    return out


def build_bge_index(pids, acmt, bundles, cas_for):
    """一次性嵌入所有属性名与参数键，返回 {string: vec}。失败返回 None。"""
    strings = set()
    for pid in pids:
        for aid in acmt[pid]:
            meta = cas_for(bundles[pid].category).get(aid, {})
            strings.add(meta.get("canonical_name", aid))
        for k in (bundles[pid].params or {}):
            strings.add(str(k))
    strings = [s for s in strings if s and s.strip()]
    if not strings:
        return None
    try:
        from common import embedding
        import numpy as np  # noqa: F401
        vecs = embedding.embed(strings)
        return {s: vecs[i] for i, s in enumerate(strings)}
    except Exception as e:  # noqa: BLE001
        print(f"  [C2] BGE 嵌入不可用({type(e).__name__})，跳过语义匹配。")
        return None


def bge_match(missing: list[str], acmt_p: dict, params: dict, cas_by_id: dict,
              vec: dict, threshold: float) -> dict:
    """对 alias 未命中的属性，用 BGE 余弦把属性名匹配到语义最近的参数键。"""
    import numpy as np
    res = {a: [] for a in missing}
    param_keys = [str(k) for k in params if str(k) in vec]
    if not param_keys:
        return res
    PK = np.stack([vec[k] for k in param_keys])
    for aid in missing:
        name = cas_by_id.get(aid, {}).get("canonical_name", aid)
        if name not in vec:
            continue
        sims = PK @ vec[name]
        best = int(sims.argmax())
        if float(sims[best]) >= threshold:
            pk = param_keys[best]
            res[aid].append({"param_key": pk, "raw_text": str(params[pk]),
                             "match": "bge", "score": round(float(sims[best]), 3)})
    return res


def llm_fallback(missing: list[str], acmt_p: dict, params: dict) -> dict:
    if not missing or not params:
        return {a: [] for a in missing}
    attrs_block = "\n".join(f"- {a} | {acmt_p[a].get('canonical_name', a)}" for a in missing)
    params_block = "\n".join(f"{k}: {v}" for k, v in params.items())
    try:
        arr = llm.chat_json(
            C2_PROMPT.format(attrs=attrs_block, params=params_block),
            namespace="c2", max_tokens=1024,
        )
    except Exception:
        return {a: [] for a in missing}
    res = {a: [] for a in missing}
    if isinstance(arr, list):
        valid_keys = set(params.keys())
        for item in arr:
            if not isinstance(item, dict):
                continue
            aid = item.get("attribute_id")
            if aid not in res:
                continue
            for m in item.get("matches", []) or []:
                pk = m.get("param_key")
                if pk in valid_keys:
                    res[aid].append({"param_key": pk, "raw_text": str(params[pk])})
    return res


def params_with_title(bundle) -> dict:
    params = {str(k): v for k, v in (bundle.params or {}).items()}
    title = str(getattr(bundle, "title", "") or "").strip()
    if title and "商品标题" not in params:
        params["商品标题"] = title
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acmt", default=str(config.STAGE_B / "acmt.json"))
    ap.add_argument("--stage_a_dir", default=str(config.STAGE_A))
    ap.add_argument("--out", default=str(config.STAGE_C / "evidence_params.json"))
    ap.add_argument("--rerun-empty", action="store_true",
                    help="Reprocess products whose cached parameter evidence is entirely empty.")
    ap.add_argument("--rerun-missing-attrs", action="store_true",
                    help="Reprocess products whose cached parameter evidence lacks requested attribute keys.")
    args = ap.parse_args()

    acmt = read_json(args.acmt, default={})
    bundles = pidx.build_bundles()
    cas_cache: dict[str, dict] = {}

    def cas_for(cat: str) -> dict:
        if cat not in cas_cache:
            cas_cache[cat] = {a["attribute_id"]: a for a in read_json(
                os.path.join(args.stage_a_dir, f"CAS+_{cat}.json"), default={"attributes": []}).get("attributes", [])}
        return cas_cache[cat]

    pids = [p for p in acmt if p in bundles]
    if args.category:
        pids = [p for p in pids if bundles[p].category == args.category]
    if args.limit:
        pids = pids[:args.limit]

    vec = build_bge_index(pids, acmt, bundles, cas_for) if (C2_USE_BGE and not args.no_llm) else None

    out = read_json(args.out, default={})

    def _is_empty(pid: str) -> bool:
        ev = out.get(pid)
        if not isinstance(ev, dict):
            return True
        return not any(bool(v) for v in ev.values())

    def _missing_attrs(pid: str) -> bool:
        ev = out.get(pid)
        if not isinstance(ev, dict):
            return True
        return any(a not in ev for a in acmt[pid])

    todo = [
        p for p in pids
        if p not in out
        or (args.rerun_empty and _is_empty(p))
        or (args.rerun_missing_attrs and _missing_attrs(p))
    ]
    print(f"[C2] products to params-extract: {len(todo)} (already done={len(pids) - len(todo)})")

    def _job(pid: str):
        b = bundles[pid]
        cas_by_id = cas_for(b.category)
        params = params_with_title(b)
        ev = alias_lookup(acmt[pid], params, cas_by_id)
        missing = [a for a, lst in ev.items() if not lst]
        n_bge_one = 0
        llm_used = 0
        # 1) BGE 语义匹配补 alias 漏网
        if missing and vec is not None:
            bm = bge_match(missing, acmt[pid], params, cas_by_id, vec, C2_BGE_THRESHOLD)
            for a, lst in bm.items():
                if lst:
                    ev[a] = lst
                    n_bge_one += 1
            missing = [a for a, lst in ev.items() if not lst]
        # 2) LLM 兜底剩余
        if missing and not args.no_llm:
            fb = llm_fallback(missing, acmt[pid], params)
            for a, lst in fb.items():
                if lst:
                    ev[a] = lst
            llm_used = 1
        return pid, ev, n_bge_one, llm_used

    results = llm.run_many(todo, _job, desc="C2-params") if todo else []
    n_llm = 0
    n_bge = 0
    for r in results:
        if isinstance(r, tuple):
            pid, ev, nb, nl = r
            out[pid] = ev
            n_bge += int(nb)
            n_llm += int(nl)
    write_json(args.out, out)
    hit = sum(1 for p in out.values() for lst in p.values() if lst)
    print(f"[C2] products={len(out)} attr-hits={hit} bge_matches={n_bge} "
          f"llm_fallback_products={n_llm} -> {args.out}")


if __name__ == "__main__":
    main()
