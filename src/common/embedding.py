"""BGE-large-zh-v1.5 文本嵌入 + 层次聚类（Stage A0/A2）。

模型优先从 ModelScope 加载（服务器可达），其次 HF 缓存。
聚类用 scikit-learn AgglomerativeClustering（cosine + average linkage）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config


@lru_cache(maxsize=1)
def _load_model():
    name = config.EMBED_MODEL
    # 1) FlagEmbedding
    try:
        from FlagEmbedding import FlagModel  # type: ignore
        return ("flag", FlagModel(name, use_fp16=True))
    except Exception:
        pass
    # 2) sentence-transformers（可配 ModelScope 下载）
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            from modelscope import snapshot_download  # type: ignore
            local = snapshot_download(name.replace("BAAI/", "AI-ModelScope/"))
            return ("st", SentenceTransformer(local))
        except Exception:
            return ("st", SentenceTransformer(name))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"无法加载嵌入模型 {name}: {e!r}. 请 pip 安装 FlagEmbedding 或 sentence-transformers。")


def embed(texts: list[str]) -> np.ndarray:
    # 1) 当前解释器若有 torch/嵌入库，直接用
    try:
        kind, model = _load_model()
        if kind == "flag":
            vecs = model.encode(texts)
        else:
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    except Exception as local_err:  # noqa: BLE001
        # 2) 退到外部带 torch 的 Python（如 GPU 环境 myconda）跑 embed_worker
        if config.EMBED_PYTHON:
            return _embed_via_worker(texts)
        raise local_err


def _embed_via_worker(texts: list[str]) -> np.ndarray:
    import json as _json
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        inf = Path(td) / "in.json"
        outf = Path(td) / "out.npy"
        inf.write_text(_json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        src_root = str(Path(__file__).resolve().parents[1])  # <root>/src
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = src_root + ":" + env.get("PYTHONPATH", "")
        cmd = [config.EMBED_PYTHON, "-m", "common.embed_worker",
               str(inf), str(outf), config.EMBED_MODEL]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not outf.exists():
            raise RuntimeError(f"embed_worker 失败: {r.stderr[-500:]}")
        return np.load(outf)


def cluster(texts: list[str], distance_threshold: float) -> list[int]:
    """对文本做层次聚类，返回每个文本的 cluster 标签。

    优先 BGE 嵌入 + cosine 距离 + average linkage（distance_threshold 越小越保守）。
    若嵌入模型/torch 不可用，自动回退到 LLM 语义分组（达到等价的同义合并目的）。
    """
    if not texts:
        return []
    if len(texts) == 1:
        return [0]
    try:
        vecs = embed(texts)
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
        sim = np.clip(vecs @ vecs.T, -1.0, 1.0)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        model = AgglomerativeClustering(
            n_clusters=None, metric="precomputed",
            linkage="average", distance_threshold=distance_threshold,
        )
        return model.fit_predict(dist).tolist()
    except Exception as e:  # noqa: BLE001
        print(f"  [embedding] 嵌入不可用({type(e).__name__})，回退 LLM 语义分组。")
        return llm_cluster(texts)


_LLM_CLUSTER_PROMPT = """把下列电商属性短语按"是否指向同一个商品属性"分组。

合并规则（应归为一组）：用词不同但所指相同（如"保质期/保质日期/保质时间"）、
同义改写、繁简/单位/口语差异、明显的上下位极近表达（如"储存条件/贮存方式"）。
不要合并：明确不同维度的属性（如"净含量"≠"规格重量"，"产地"≠"品牌"）。

短语（带编号）：
{items}

输出严格 JSON：{{"groups": [[编号,...], ...]}}（每个编号恰好出现一次；独立属性单独成组）。只输出 JSON。"""


def llm_cluster(texts: list[str]) -> list[int]:
    """用 LLM 对短语做语义分组，返回 cluster 标签（与 texts 等长）。"""
    from common import llm  # 延迟导入避免循环
    labels = [-1] * len(texts)
    # 分批，避免一次过多
    BATCH = 80
    next_label = 0
    for off in range(0, len(texts), BATCH):
        chunk = texts[off:off + BATCH]
        items = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk))
        try:
            res = llm.chat_json(_LLM_CLUSTER_PROMPT.format(items=items),
                                namespace="cluster", max_tokens=2048)
            groups = res.get("groups", []) if isinstance(res, dict) else []
        except Exception:
            groups = []
        assigned = set()
        for g in groups:
            if not isinstance(g, list):
                continue
            valid = [i for i in g if isinstance(i, int) and 0 <= i < len(chunk) and i not in assigned]
            if not valid:
                continue
            for i in valid:
                labels[off + i] = next_label
                assigned.add(i)
            next_label += 1
        # 未分到组的各自独立
        for i in range(len(chunk)):
            if i not in assigned:
                labels[off + i] = next_label
                next_label += 1
    return labels
