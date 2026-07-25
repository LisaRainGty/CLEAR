"""§4.2 规定基线（与 CLAIMARC 同划分/同监督/同指标）。

(A) 客观事实核验：
    - ESIM（Chen et al. 2017）：BiLSTM + 软对齐 + 差积增强 + 组合池化。
    - BERT-NLI：中文 NLI 预训练骨干微调（不可达时回落 bert-base-chinese）。
(B) 文本分类（单流拼接 [CLS] X^c [SEP] X^e [SEP]）：
    - bert-base-chinese + [CLS]
    - chinese-roberta-wwm-ext + [CLS]

全部用 pos_weight + 可靠性权重 c，val 上按 Macro-F1 选阈值；
指标对齐 §4.3：Macro-F1(主)、AUPRC、AUROC、可靠性加权 F1、ECE。

用法：python -m models.baselines_ft --kind bert_cls --seed 0
"""
from __future__ import annotations

import argparse
import json

import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.data import apply_evidence_policy, load_split
from models.baselines import claim_text, evidence_text
from models.provenance import attach_run_provenance
from models.train import macro_f1, best_threshold_macroF1, ece, cls_loss
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def resolve(name, allow_fallback=False):
    import os
    _local = {
        "bert-base-chinese": os.environ.get("CLAIMARC_BERT_PATH", ""),
        "hfl/chinese-roberta-wwm-ext": os.environ.get("CLAIMARC_ROBERTA_PATH", ""),
        "nli": os.environ.get("CLAIMARC_NLI_PATH", ""),
    }.get(name)
    if _local and (os.path.isfile(os.path.join(_local, "pytorch_model.bin"))
                   or os.path.isfile(os.path.join(_local, "model.safetensors"))):
        return _local
    try:
        from modelscope import snapshot_download
        ms = {
            "bert-base-chinese": "tiansz/bert-base-chinese",
            "hfl/chinese-roberta-wwm-ext": "dienstag/chinese-roberta-wwm-ext",
            "nli": "Fengshenbang/Erlangshen-Roberta-110M-NLI",
        }.get(name, name)
        return snapshot_download(ms)
    except Exception:
        if name == "nli" and allow_fallback:
            return resolve("bert-base-chinese", allow_fallback=False)
        if name == "nli":
            raise RuntimeError(
                "BERT-NLI backbone is unavailable. Set --model_path to the exact NLI model; "
                "use --allow_backbone_fallback only for a separately labelled diagnostic run."
            )
        return name


# ----------------------------- 单流拼接微调 -----------------------------
class PairDataset(Dataset):
    def __init__(self, recs, tok, maxlen=512):
        self.recs, self.tok, self.maxlen = recs, tok, maxlen

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        enc = self.tok(claim_text(r), evidence_text(r), truncation=True,
                       max_length=self.maxlen, padding="max_length", return_tensors="pt")
        return (enc["input_ids"][0], enc["attention_mask"][0],
                float(r.get("y", 0)), float(r.get("c", 0.05)))


def collate_pair(items):
    ids = torch.stack([it[0] for it in items])
    mask = torch.stack([it[1] for it in items])
    y = torch.tensor([it[2] for it in items])
    c = torch.tensor([it[3] for it in items])
    return ids, mask, y, c


class SingleStream(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(path)
        d = self.enc.config.hidden_size
        self.cls = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.cls(h).squeeze(-1)


# ----------------------------- ESIM -----------------------------
class SeqDataset(Dataset):
    def __init__(self, recs, tok, maxlen=384):
        self.recs, self.tok, self.maxlen = recs, tok, maxlen

    def __len__(self):
        return len(self.recs)

    def _enc(self, t):
        e = self.tok(t, truncation=True, max_length=self.maxlen,
                     padding="max_length", return_tensors="pt")
        return e["input_ids"][0], e["attention_mask"][0]

    def __getitem__(self, i):
        r = self.recs[i]
        ci, cm = self._enc(claim_text(r))
        ei, em = self._enc(evidence_text(r))
        return ci, cm, ei, em, float(r.get("y", 0)), float(r.get("c", 0.05))


def collate_seq(items):
    f = lambda k: torch.stack([it[k] for it in items])
    return (f(0), f(1), f(2), f(3),
            torch.tensor([it[4] for it in items]), torch.tensor([it[5] for it in items]))


class ESIM(nn.Module):
    def __init__(self, path, hidden=300):
        super().__init__()
        from transformers import AutoModel
        emb = AutoModel.from_pretrained(path).get_input_embeddings()
        self.embedding = emb
        d = emb.embedding_dim
        self.enc = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(8 * hidden, hidden)
        self.comp = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.cls = nn.Sequential(
            nn.Linear(8 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def _soft_align(self, a, b, bmask):
        attn = a @ b.transpose(1, 2)
        attn = attn.masked_fill(~bmask[:, None, :].bool(), -1e9)
        return torch.softmax(attn, -1) @ b

    def forward(self, ci, cm, ei, em):
        a = self.enc(self.embedding(ci))[0]
        b = self.enc(self.embedding(ei))[0]
        a_til = self._soft_align(a, b, em)
        b_til = self._soft_align(b, a, cm)
        ma = torch.cat([a, a_til, a - a_til, a * a_til], -1)
        mb = torch.cat([b, b_til, b - b_til, b * b_til], -1)
        va = self.comp(F.relu(self.proj(ma)))[0]
        vb = self.comp(F.relu(self.proj(mb)))[0]

        def pool(v, m):
            mm = m[:, :, None].bool()
            avg = (v * mm).sum(1) / mm.sum(1).clamp(min=1)
            mx = v.masked_fill(~mm, -1e9).max(1).values
            return torch.cat([avg, mx], -1)
        z = torch.cat([pool(va, cm), pool(vb, em)], -1)
        return self.cls(z).squeeze(-1)


# ----------------------------- 通用训练/评估 -----------------------------
def evaluate(model, val_loader, test_loader, device, esim, tag, seed):
    model.eval()

    @torch.no_grad()
    def infer(loader):
        ps, ys, cs = [], [], []
        for batch in loader:
            if esim:
                ci, cm, ei, em, y, c = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                    lg = model(ci, cm, ei, em)
            else:
                ids, mask, y, c = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    lg = model(ids, mask)
            ps.append(torch.sigmoid(lg.float()).cpu()); ys.append(y.cpu()); cs.append(c.cpu())
        return torch.cat(ps).numpy(), torch.cat(ys).numpy(), torch.cat(cs).numpy()

    pv, yv, _ = infer(val_loader)
    thr = best_threshold_macroF1(yv, pv)
    p, y, c = infer(test_loader)
    pred = (p >= thr).astype(int)
    return {
        "tag": tag, "seed": seed, "thr": round(float(thr), 3),
        "acc": round(float((pred == y).mean()), 4),
        "macro_f1": round(macro_f1(y, pred), 4),
        "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
        "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
        "auprc": round(average_precision_score(y, p), 4),
        "auroc": round(roc_auc_score(y, p), 4),
        "ece": round(ece(y, p), 4),
        "n_test": int(len(y)), "pos_test": int(y.sum()),
    }


def run(args, splits=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sp = splits if splits is not None else load_split(args.dataset)
    apply_evidence_policy(sp, getattr(args, "evidence_policy", "sources_only"))
    if getattr(args, "infer_jsonl", ""):
        recs = [json.loads(l) for l in open(args.infer_jsonl, encoding="utf-8") if l.strip()]
        for r in recs:
            r["split"] = "test"
        sp = dict(sp); sp["test"] = recs
        print(f"[infer_jsonl] test split replaced with {len(recs)} constructed records", flush=True)
    esim = args.kind == "esim"
    paths = {
        "bert_cls": "bert-base-chinese",
        "roberta_cls": "hfl/chinese-roberta-wwm-ext",
        "bert_nli": "nli",
        "esim": "bert-base-chinese",
    }
    path = (args.model_path if getattr(args, "model_path", "") else
            resolve(paths[args.kind], getattr(args, "allow_backbone_fallback", False)))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)

    if esim:
        model = ESIM(path).to(device)
        mk = lambda recs, sh: DataLoader(SeqDataset(recs, tok), batch_size=args.bs,
                                         shuffle=sh, collate_fn=collate_seq, num_workers=6, pin_memory=True)
    else:
        model = SingleStream(path).to(device)
        mk = lambda recs, sh: DataLoader(PairDataset(recs, tok), batch_size=args.bs,
                                         shuffle=sh, collate_fn=collate_pair, num_workers=6, pin_memory=True)
    tr, vl, te = mk(sp["train"], True), mk(sp["val"], False), mk(sp["test"], False)

    n_pos = sum(r["y"] for r in sp["train"]); n_neg = len(sp["train"]) - n_pos
    pw = torch.tensor(min(n_neg / max(1, n_pos), 50.0), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * max(0.0, (total - s) / max(1, total - 200)))

    import copy
    best_score, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train(); tl = 0.0; n = 0
        for batch in tr:
            if esim:
                ci, cm, ei, em, y, c = [x.to(device) for x in batch]
                # ESIM 含 LSTM，bf16 下 _thnn_fused_lstm_cell 未实现 → 关闭 autocast 走 fp32
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                    lg = model(ci, cm, ei, em)
            else:
                ids, mask, y, c = [x.to(device) for x in batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    lg = model(ids, mask)
            loss = cls_loss(lg, y, torch.clamp(c, 0.05), args.loss, pw, gamma_neg=args.gamma_neg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tl += loss.item(); n += 1
        # 验证集模型选择（与 CLAIMARC 同协议，保证公平）
        rv = evaluate(model, vl, vl, device, esim, tag=args.kind, seed=args.seed)
        score = rv["macro_f1"] + 0.5 * rv["auprc"]
        print(f"[{args.kind} ep{ep}] loss={tl/n:.4f} val_mF1={rv['macro_f1']:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    res = evaluate(model, vl, te, device, esim, tag=args.kind, seed=args.seed)
    attach_run_provenance(res, args, path)
    res["backbone_fallback_allowed"] = bool(getattr(args, "allow_backbone_fallback", False))
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
    if args.save_pred:
        @torch.no_grad()
        def infer(loader):
            ps, ys, cs = [], [], []
            for batch in loader:
                if esim:
                    ci, cm, ei, em, y, c = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                    lg = model(ci, cm, ei, em)
                else:
                    ids, mask, y, c = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                    lg = model(ids, mask)
                ps.append(torch.sigmoid(lg.float()).cpu()); ys.append(y.cpu()); cs.append(c.cpu())
            return torch.cat(ps).numpy(), torch.cat(ys).numpy(), torch.cat(cs).numpy()
        p, y, c = infer(te)
        pva, yva, _ = infer(vl)
        torch.save({"thr": res["thr"], "val": {
                        "p": pva, "y": yva,
                        "pair_id": [r.get("pair_id", "") for r in sp["val"]]},
                    "provenance": {k: res.get(k) for k in (
                        "dataset", "dataset_sha256", "evidence_policy", "resolved_model",
                        "label_field", "split_field", "split_group", "seed", "tag"
                    )},
                    "test": {"p": p, "y": y, "c": c,
                    "attr": [r.get("attribute_id", "") for r in sp["test"]],
                    "pair_id": [r.get("pair_id", "") for r in sp["test"]]}}, args.save_pred)
        print(f"[save_pred] -> {args.save_pred}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DEFAULT_TRAIN_DATASET))
    ap.add_argument("--kind", required=True,
                    choices=["bert_cls", "roberta_cls", "bert_nli", "esim"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_pred", default="")
    ap.add_argument("--infer_jsonl", default="", help="用该 jsonl 替换 test 划分做构造样本推理")
    ap.add_argument("--model_path", default="", help="直接指定本地骨干路径，跳过 modelscope")
    ap.add_argument("--evidence_policy", default="sources_only")
    ap.add_argument("--allow_backbone_fallback", action="store_true",
                    help="仅用于诊断；正式 BERT-NLI 结果禁止静默退回普通 BERT")
    ap.add_argument("--loss", default="asl", choices=["bce", "focal", "asl"])
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
