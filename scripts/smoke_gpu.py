#!/usr/bin/env python3
"""One-batch CUDA forward/backward smoke test for the canonical CLAIMARC model."""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from models.data import (SPECIAL_TOKENS, ClaimDataset, apply_evidence_policy,
                         build_tokenizer, load_split, make_collate, source_len)
from models.model import CLAIMARC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    splits = load_split(args.dataset)
    apply_evidence_policy(splits, "sources_only")
    rows = [r for r in splits["train"] if any(
        r.get(key) for key in ("evidence_params", "evidence_ocr", "evidence_vlm"))]
    rows.sort(key=source_len, reverse=True)
    tokenizer = build_tokenizer(args.encoder)
    dataset = ClaimDataset(rows[:args.batch_size], tokenizer)
    batch = make_collate(tokenizer.pad_token_id)([dataset[i] for i in range(len(dataset))])
    if max(batch.arg_len.tolist(), default=0) != 0:
        raise RuntimeError("arguments unexpectedly entered the smoke-test batch")
    device = torch.device("cuda")
    model = CLAIMARC(
        args.encoder, len(tokenizer), len(SPECIAL_TOKENS), n_fusion=2,
        enc_train="full", use_lora=False, fusion_dropout=0.2, heads=8,
    ).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(1e-5, 1e-4))
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, embeddings = model(
            batch.c_ids.to(device), batch.c_mask.to(device),
            batch.e_ids.to(device), batch.e_mask.to(device),
        )
        loss = F.binary_cross_entropy_with_logits(logits.float(), batch.y.to(device))
    loss.backward()
    optimizer.step()
    result = {
        "status": "PASS", "batch_size": len(rows[:args.batch_size]),
        "claim_shape": list(batch.c_ids.shape), "evidence_shape": list(batch.e_ids.shape),
        "embedding_shape": list(embeddings.shape), "loss": float(loss.item()),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        "evidence_policy": "sources_only", "argument_length_max": 0,
        "gradient_checkpointing": os.environ.get(
            "CLAIMARC_GRADIENT_CHECKPOINTING", "0") == "1",
    }
    print("SMOKE_RESULT", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
