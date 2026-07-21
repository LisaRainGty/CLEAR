"""BGE 嵌入 worker（由带 torch 的 Python 解释器运行，可与主流水线 Python 隔离）。

用法：<python> -m common.embed_worker <in_json> <out_npy> [model_name]
  in_json : JSON 字符串数组
  out_npy : 输出 float32 向量 (.npy)，已 L2 归一化
模型优先从 ModelScope 下载（国内可达），其次 HF。
"""
from __future__ import annotations

import json
import sys


def main():
    in_json, out_npy = sys.argv[1], sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "BAAI/bge-large-zh-v1.5"
    with open(in_json, "r", encoding="utf-8") as f:
        texts = json.load(f)

    import numpy as np
    model = _load(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    np.save(out_npy, vecs)


def _load(name: str):
    from sentence_transformers import SentenceTransformer
    # 优先 ModelScope
    try:
        from modelscope import snapshot_download
        ms_name = name.replace("BAAI/", "AI-ModelScope/")
        local = snapshot_download(ms_name)
        return SentenceTransformer(local)
    except Exception:
        return SentenceTransformer(name)


if __name__ == "__main__":
    main()
