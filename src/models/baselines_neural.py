"""From-scratch deep-learning baselines (same split / supervision / metrics as CLAIMARC).

  textcnn : single-stream TextCNN (Kim 2014) over [claim [SEP] evidence]
  bilstm  : single-stream BiLSTM + max/avg pooling
  dam     : Decomposable Attention (Parikh et al. 2016) over (claim, evidence) -- a
            non-recurrent fact-verification baseline complementary to ESIM

Embeddings are initialized from a Chinese vocab (bert-base-chinese tokenizer) and trained.
pos_weight + reliability weight c; val threshold by Macro-F1; reports the standard metric set.

Usage: python -m models.baselines_neural --dataset .../d.jsonl --kind textcnn --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.data import load_split
from models.baselines import claim_text, evidence_text
from models.train import macro_f1, best_threshold_macroF1, ece, cls_loss
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

BERT = "/root/models/bert-base-chinese"
MAXLEN = 256


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BERT)


class SingleDS(Dataset):
    def __init__(self, recs, tok):
        self.recs, self.tok = recs, tok

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        enc = self.tok(claim_text(r), evidence_text(r), truncation=True, max_length=MAXLEN,
                       padding="max_length", return_tensors="pt")
        return (enc["input_ids"][0], enc["attention_mask"][0],
                float(r.get("y", 0)), float(r.get("c", 0.05)))


class PairDS(Dataset):
    def __init__(self, recs, tok):
        self.recs, self.tok = recs, tok

    def __len__(self):
        return len(self.recs)

    def _e(self, t):
        e = self.tok(t, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        return e["input_ids"][0], e["attention_mask"][0]

    def __getitem__(self, i):
        r = self.recs[i]
        ci, cm = self._e(claim_text(r)); ei, em = self._e(evidence_text(r))
        return ci, cm, ei, em, float(r.get("y", 0)), float(r.get("c", 0.05))


def coll_single(items):
    f = lambda k: torch.stack([it[k] for it in items])
    return f(0), f(1), torch.tensor([it[2] for it in items]), torch.tensor([it[3] for it in items])


def coll_pair(items):
    f = lambda k: torch.stack([it[k] for it in items])
    return (f(0), f(1), f(2), f(3),
            torch.tensor([it[4] for it in items]), torch.tensor([it[5] for it in items]))


class TextCNN(nn.Module):
    def __init__(self, vocab, d=256, filters=128, ks=(2, 3, 4)):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, filters, k, padding=k // 2) for k in ks])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(filters * len(ks), 1)

    def forward(self, ids, mask):
        x = self.emb(ids).transpose(1, 2)
        feats = [F.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, vocab, d=256, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.lstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(4 * hidden, 1)

    def forward(self, ids, mask):
        h, _ = self.lstm(self.emb(ids))
        m = mask[:, :, None].bool()
        avg = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(~m, -1e9).max(1).values
        return self.fc(self.drop(torch.cat([avg, mx], -1))).squeeze(-1)


class DecomposableAttention(nn.Module):
    """Parikh et al. (2016): attend / compare / aggregate, no recurrence."""
    def __init__(self, vocab, d=256, hidden=200):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.F = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(hidden, hidden), nn.ReLU())
        self.G = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(hidden, hidden), nn.ReLU())
        self.proj = nn.Linear(d, hidden)
        self.H = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(hidden, 1))

    def forward(self, ci, cm, ei, em):
        a = self.proj(self.emb(ci)); b = self.proj(self.emb(ei))
        fa, fb = self.F(a), self.F(b)
        e = fa @ fb.transpose(1, 2)
        e = e.masked_fill(~em[:, None, :].bool(), -1e9)
        beta = torch.softmax(e, dim=2) @ b
        e2 = e.transpose(1, 2).masked_fill(~cm[:, None, :].bool(), -1e9)
        alpha = torch.softmax(e2, dim=2) @ a
        v1 = self.G(torch.cat([a, beta], -1)) * cm[:, :, None]
        v2 = self.G(torch.cat([b, alpha], -1)) * em[:, :, None]
        v1 = v1.sum(1); v2 = v2.sum(1)
        return self.H(torch.cat([v1, v2], -1)).squeeze(-1)


@torch.no_grad()
def infer(model, loader, device, pair):
    model.eval(); ps, ys, cs = [], [], []
    for batch in loader:
        b = [x.to(device) for x in batch]
        lg = model(b[0], b[1], b[2], b[3]) if pair else model(b[0], b[1])
        ps.append(torch.sigmoid(lg.float()).cpu())
        ys.append((b[4] if pair else b[2]).cpu()); cs.append((b[5] if pair else b[3]).cpu())
    return torch.cat(ps).numpy(), torch.cat(ys).numpy(), torch.cat(cs).numpy()


def evaluate(model, vl, te, device, pair, tag, seed):
    pv, yv, _ = infer(model, vl, device, pair)
    thr = best_threshold_macroF1(yv, pv)
    p, y, c = infer(model, te, device, pair)
    pred = (p >= thr).astype(int)
    return {"tag": tag, "seed": seed, "thr": round(float(thr), 3),
            "acc": round(float((pred == y).mean()), 4),
            "macro_f1": round(macro_f1(y, pred), 4),
            "pos_f1": round(f1_score(y, pred, zero_division=0), 4),
            "wF1": round(macro_f1(y, pred, w=np.clip(c, 0.05, None)), 4),
            "auprc": round(average_precision_score(y, p), 4),
            "auroc": round(roc_auc_score(y, p), 4), "ece": round(ece(y, p), 4),
            "n_test": int(len(y)), "pos_test": int(y.sum())}


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sp = load_split(args.dataset)
    tok = get_tok(); vocab = tok.vocab_size
    pair = args.kind == "dam"
    DS = PairDS if pair else SingleDS
    coll = coll_pair if pair else coll_single
    mk = lambda recs, sh: DataLoader(DS(recs, tok), batch_size=args.bs, shuffle=sh,
                                     collate_fn=coll, num_workers=4, pin_memory=True)
    tr, vl, te = mk(sp["train"], True), mk(sp["val"], False), mk(sp["test"], False)
    if args.kind == "textcnn":
        model = TextCNN(vocab).to(device)
    elif args.kind == "bilstm":
        model = BiLSTM(vocab).to(device)
    elif args.kind == "dam":
        model = DecomposableAttention(vocab).to(device)
    else:
        raise ValueError(args.kind)
    n_pos = sum(r["y"] for r in sp["train"]); n_neg = len(sp["train"]) - n_pos
    pw = torch.tensor(min(n_neg / max(1, n_pos), 50.0), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train(); tl = 0.0; n = 0
        for batch in tr:
            b = [x.to(device) for x in batch]
            lg = model(b[0], b[1], b[2], b[3]) if pair else model(b[0], b[1])
            y = b[4] if pair else b[2]; c = b[5] if pair else b[3]
            loss = cls_loss(lg, y, torch.clamp(c, 0.05), args.loss, pw, gamma_neg=args.gamma_neg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tl += loss.item(); n += 1
        rv = evaluate(model, vl, vl, device, pair, args.kind, args.seed)
        score = rv["macro_f1"] + 0.5 * rv["auprc"]
        print(f"[{args.kind} ep{ep}] loss={tl/n:.4f} val_mF1={rv['macro_f1']:.4f}", flush=True)
        if score > best:
            best = score; best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    res = evaluate(model, vl, te, device, pair, args.kind, args.seed)
    print("RESULT", json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kind", required=True, choices=["textcnn", "bilstm", "dam"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="bce", choices=["bce", "focal", "asl"])
    ap.add_argument("--gamma_neg", type=float, default=4.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
