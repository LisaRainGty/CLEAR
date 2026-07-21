"""Qwen2.5-7B QLoRA sequence-classification baseline for Table 3.

The input is deliberately identical to every other fair-comparison model:
claim text plus the concatenation of PARAM/OCR/VLM evidence.  Arguments are
never read when ``--evidence_policy sources_only`` (the required setting).
Threshold and checkpoint selection use validation data only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from models.baselines import claim_text, evidence_text
from models.data import apply_evidence_policy, load_split
from models.provenance import attach_run_provenance
from models.train import best_threshold_macroF1, ece, macro_f1


def resolve_model(name: str) -> str:
    if os.path.isdir(name):
        return name
    env = os.environ.get("CLAIMARC_QWEN_PATH", "")
    if env and os.path.isdir(env):
        return env
    try:
        from modelscope import snapshot_download
        return snapshot_download(name)
    except Exception:
        return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        text = (
            "任务：判断直播商品声称是否与商品证据矛盾。\n"
            f"[声称] {claim_text(row)}\n"
            f"[证据] {evidence_text(row)}"
        )
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "y": torch.tensor(int(row.get("y", 0)), dtype=torch.long),
            "c": torch.tensor(float(row.get("c", 0.05)), dtype=torch.float32),
        }


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    probs, labels, weights = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        labels.append(batch["y"])
        weights.append(batch["c"])
    return (torch.cat(probs).numpy(), torch.cat(labels).numpy(),
            torch.cat(weights).numpy())


def metrics(y, p, c, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "acc": round(float((pred == y).mean()), 4),
        "macro_f1": round(macro_f1(y, pred), 4),
        "pos_f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
        "auprc": round(float(average_precision_score(y, p)), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "ece": round(float(ece(y, p)), 4),
        "n_test": int(len(y)),
        "pos_test": int(y.sum()),
    }


def run(args):
    if args.evidence_policy != "sources_only":
        raise ValueError("The paper fair rerun requires --evidence_policy sources_only")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5-7B QLoRA requires a CUDA GPU")
    set_seed(args.seed)
    device = torch.device("cuda")
    splits = load_split(args.dataset)
    apply_evidence_policy(splits, args.evidence_policy)
    path = resolve_model(args.model)

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig, get_linear_schedule_with_warmup)
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, quantization_config=quant, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=args.dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    # PEFT 0.13 calls ``torch.dtype.itemsize`` here, which is unavailable in
    # the frozen PyTorch 2.0/CUDA 11.7 environment.  The helper is logging-only;
    # compute the same counts without touching quantisation storage metadata.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / max(1, all_params):.6f}"
    )

    def loader(split, shuffle):
        return DataLoader(
            PairDataset(splits[split], tok, args.max_length), batch_size=args.bs,
            shuffle=shuffle, num_workers=args.workers, pin_memory=True,
        )

    train_loader = loader("train", True)
    val_loader = loader("val", False)
    test_loader = loader("test", False)
    n_pos = sum(int(r.get("y", 0)) for r in splits["train"])
    n_neg = len(splits["train"]) - n_pos
    class_weight = torch.tensor([1.0, min(n_neg / max(1, n_pos), 50.0)], device=device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = (len(train_loader) + args.accum - 1) // args.accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_steps * args.warmup_ratio)), total_steps,
    )

    best_score = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["y"].to(device)
            c = batch["c"].to(device).clamp(min=0.05)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits.float()
                per = F.cross_entropy(logits, y, weight=class_weight, reduction="none")
                loss = (per * c).mean() / args.accum
            loss.backward()
            running += float(loss.item()) * args.accum
            if (step + 1) % args.accum == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        pv, yv, cv = infer(model, val_loader, device)
        threshold = best_threshold_macroF1(yv, pv)
        val_metrics = metrics(yv, pv, cv, threshold)
        score = val_metrics["macro_f1"] + 0.5 * val_metrics["auprc"]
        print(f"[qwen ep{epoch}] loss={running/len(train_loader):.4f} "
              f"val_mF1={val_metrics['macro_f1']:.4f} val_ap={val_metrics['auprc']:.4f}",
              flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({
                k: v.detach().cpu() for k, v in model.state_dict().items()
                if v.requires_grad or "lora_" in k or "modules_to_save" in k
            })
    if best_state:
        model.load_state_dict(best_state, strict=False)

    pv, yv, cv = infer(model, val_loader, device)
    threshold = best_threshold_macroF1(yv, pv)
    p, y, c = infer(model, test_loader, device)
    result = {
        "tag": "qwen2p5_7b_qlora_sft", "seed": args.seed,
        "thr": round(float(threshold), 3), **metrics(y, p, c, threshold),
        "lora_rank": args.rank, "max_length": args.max_length,
    }
    attach_run_provenance(result, args, path)
    print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    if args.save_pred:
        torch.save({
            "thr": result["thr"],
            "provenance": {k: result.get(k) for k in (
                "dataset", "dataset_sha256", "evidence_policy", "resolved_model",
                "label_field", "split_field", "split_group", "seed", "tag",
            )},
            "val": {"p": pv, "y": yv, "c": cv,
                    "pair_id": [r.get("pair_id", "") for r in splits["val"]]},
            "test": {"p": p, "y": y, "c": c,
                     "attr": [r.get("attribute_id", "") for r in splits["test"]],
                     "pair_id": [r.get("pair_id", "") for r in splits["test"]]},
        }, args.save_pred)
        print(f"[save_pred] -> {args.save_pred}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=os.environ.get(
        "CLAIMARC_QWEN_PATH", "Qwen/Qwen2.5-7B"))
    parser.add_argument("--evidence_policy", default="sources_only",
                        choices=["sources_only"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save_pred", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
