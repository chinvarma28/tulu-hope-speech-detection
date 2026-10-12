"""
DravidianCodeMix 2026 -- SubTask 2: Multi-Task Hope Speech Detection in Code-Mixed Tulu

Shared encoder, two classification heads, joint loss.
Includes class weighting, macro-F1 checkpoint selection, constrained decoding,
and multi-seed averaging.

Usage:
    python train_multitask.py --model google/muril-base-cased --seeds 1 2 3
    python train_multitask.py --model xlm-roberta-base --lam 0.5

Requires:
    pip install torch transformers pandas scikit-learn
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

COARSE = "Coarse-Grained"
FINE = "Fine-Grained Hope"

# The "Semantic Dependency Between Outputs" table published on the task page.
# WARNING: the released training data contains pairs that this table forbids
# (notably Blended Hope + Hopelessness). Use --mask empirical unless the
# organisers confirm the table is authoritative.
OFFICIAL_PAIRS = {
    "Optimistic Hope": {"Encouraging Hope"},
    "Realistic Hope":  {"Encouraging Hope"},
    "Inspiring Hope":  {"Encouraging Hope", "Blended Hope"},
    "Fading Hope":     {"Blended Hope", "Discouraging Hope"},
    "Hopelessness":    {"Discouraging Hope"},
}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")          # Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- data

class HopeDataset(Dataset):
    def __init__(self, df, tok, c2i, f2i, max_len):
        self.texts = df["text"].tolist()
        self.y_c = [c2i[x] for x in df[COARSE]]
        self.y_f = [f2i[x] for x in df[FINE]]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_c": torch.tensor(self.y_c[i]),
            "y_f": torch.tensor(self.y_f[i]),
        }


def load_split(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("text", COARSE, FINE):
        df[col] = df[col].str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def class_weights(labels, n_classes, device):
    """Inverse-frequency weights, mildly smoothed with sqrt.
    Without this, minority classes get 0.00 F1 -- the exact failure mode that
    sank the published ACL 2026 systems."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n_classes * counts)) ** 0.5
    return torch.tensor(w, dtype=torch.float, device=device)


# -------------------------------------------------------------------------- model

class MultiTaskHope(nn.Module):
    def __init__(self, model_name, n_coarse, n_fine, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_coarse = nn.Linear(h, n_coarse)
        self.head_fine = nn.Linear(h, n_fine)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over real tokens beats [CLS] on short noisy comments.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return self.head_coarse(pooled), self.head_fine(pooled)


# ---------------------------------------------------------------- constrained decode

def build_valid_mask(df, c2i, f2i, mode):
    """A (n_coarse, n_fine) 0/1 matrix of permitted label pairs.

    'empirical' -- pairs actually observed in the training data.
    'official'  -- pairs allowed by the task page's dependency table.

    These are NOT the same. Masking out a pair that occurs in the data
    guarantees errors on every row carrying it."""
    m = np.zeros((len(c2i), len(f2i)))
    if mode == "empirical":
        for c, f in zip(df[COARSE], df[FINE]):
            m[c2i[c], f2i[f]] = 1.0
    elif mode == "official":
        for f, coarses in OFFICIAL_PAIRS.items():
            if f not in f2i:
                continue
            for c in coarses:
                if c in c2i:
                    m[c2i[c], f2i[f]] = 1.0
    return m


def report_violations(df, name):
    """How much of the data disagrees with the published dependency table?"""
    bad = [(c, f) for c, f in zip(df[COARSE], df[FINE])
           if f in OFFICIAL_PAIRS and c not in OFFICIAL_PAIRS[f]]
    n = len(bad)
    print(f"\n{name}: rows violating the official dependency table: "
          f"{n} / {len(df)} ({100*n/max(len(df),1):.1f}%)")
    if n:
        for pair, cnt in pd.Series(bad).value_counts().items():
            print(f"    {pair[0]} + {pair[1]}  ->  {cnt} rows")
        print("  A hard 'official' mask would score 0 on every one of these.")


def decode(logit_c, logit_f, valid_mask=None):
    """Unconstrained: independent argmax.
    Constrained: pick the highest-scoring pair that was actually observed."""
    lp_c = torch.log_softmax(logit_c, dim=-1).cpu().numpy()
    lp_f = torch.log_softmax(logit_f, dim=-1).cpu().numpy()
    if valid_mask is None:
        return lp_c.argmax(1), lp_f.argmax(1)

    joint = lp_c[:, :, None] + lp_f[:, None, :]        # (B, n_coarse, n_fine)
    joint = np.where(valid_mask[None, :, :] > 0, joint, -1e9)
    flat = joint.reshape(joint.shape[0], -1).argmax(1)
    return flat // valid_mask.shape[1], flat % valid_mask.shape[1]


# ------------------------------------------------------------------------ train/eval

@torch.no_grad()
def evaluate(model, loader, device, valid_mask=None):
    model.eval()
    pc, pf, tc, tf = [], [], [], []
    for b in loader:
        lc, lf = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        a, c = decode(lc, lf, valid_mask)
        pc.extend(a); pf.extend(c)
        tc.extend(b["y_c"].numpy()); tf.extend(b["y_f"].numpy())
    return (np.array(tc), np.array(pc)), (np.array(tf), np.array(pf))


def run_one_seed(args, train_df, dev_df, c2i, f2i, seed, device):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = MultiTaskHope(args.model, len(c2i), len(f2i)).to(device)

    tr = DataLoader(HopeDataset(train_df, tok, c2i, f2i, args.max_len),
                    batch_size=args.batch_size, shuffle=True)
    dv = DataLoader(HopeDataset(dev_df, tok, c2i, f2i, args.max_len),
                    batch_size=args.batch_size * 2)

    w_c = class_weights([c2i[x] for x in train_df[COARSE]], len(c2i), device)
    w_f = class_weights([f2i[x] for x in train_df[FINE]], len(f2i), device)
    # Label smoothing guards against the annotation noise in this dataset.
    loss_c = nn.CrossEntropyLoss(weight=w_c, label_smoothing=args.smooth)
    loss_f = nn.CrossEntropyLoss(weight=w_f, label_smoothing=args.smooth)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    valid_mask = (build_valid_mask(train_df, c2i, f2i, args.mask)
                  if args.mask != "none" else None)
    best = {"score": -1.0}

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for b in tr:
            opt.zero_grad()
            lc, lf = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = (args.lam * loss_c(lc, b["y_c"].to(device))
                    + (1 - args.lam) * loss_f(lf, b["y_f"].to(device)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += loss.item()

        (yc, pc), (yf, pf) = evaluate(model, dv, device, valid_mask)
        f1c = f1_score(yc, pc, average="macro", zero_division=0)
        f1f = f1_score(yf, pf, average="macro", zero_division=0)
        avg = (f1c + f1f) / 2
        print(f"  seed {seed} ep {ep}  loss={running/len(tr):.4f}  "
              f"coarse={f1c:.4f}  fine={f1f:.4f}  avg={avg:.4f}")

        # Select on macro F1, NOT on eval loss. The HF default gets this wrong.
        if avg > best["score"]:
            best = {"score": avg, "f1c": f1c, "f1f": f1f, "epoch": ep,
                    "yc": yc, "pc": pc, "yf": yf, "pf": pf}

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_multitask.csv")
    ap.add_argument("--dev", default="dev_multitask.csv")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=64)
    # The official Joint Score weights both macro F1s equally, so 0.5 is the
    # principled default. Sweep it anyway -- the fine task has 5 classes and
    # may need more capacity.
    ap.add_argument("--lam", type=float, default=0.5,
                    help="weight on the coarse loss; (1-lam) goes to fine")
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--mask", choices=["none", "empirical", "official"],
                    default="none",
                    help="restrict predictions to permitted (coarse, fine) pairs")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}   model: {args.model}   mask: {args.mask}")

    train_df, dev_df = load_split(args.train), load_split(args.dev)
    coarse_labels = sorted(train_df[COARSE].unique())
    fine_labels = sorted(train_df[FINE].unique())
    c2i = {l: i for i, l in enumerate(coarse_labels)}
    f2i = {l: i for i, l in enumerate(fine_labels)}
    print(f"train={len(train_df)}  dev={len(dev_df)}")
    print(f"coarse ({len(c2i)}): {coarse_labels}")
    print(f"fine   ({len(f2i)}): {fine_labels}")

    # Any dev label unseen in train can never be predicted -- know about it now.
    for col, mapping in ((COARSE, c2i), (FINE, f2i)):
        unseen = set(dev_df[col]) - set(mapping)
        if unseen:
            print(f"!! WARNING: {col} labels in dev but not train: {unseen}")
            dev_df = dev_df[dev_df[col].isin(mapping)].reset_index(drop=True)

    report_violations(train_df, "train")
    report_violations(dev_df, "dev")

    runs = [run_one_seed(args, train_df, dev_df, c2i, f2i, s, device)
            for s in args.seeds]

    cs = np.array([r["f1c"] for r in runs])
    fs = np.array([r["f1f"] for r in runs])
    print("\n" + "=" * 70)
    print(f"L1 coarse macro F1 : {cs.mean():.4f} +/- {cs.std():.4f}")
    print(f"L2 fine   macro F1 : {fs.mean():.4f} +/- {fs.std():.4f}")
    print(f"JOINT SCORE        : {(cs.mean()+fs.mean())/2:.4f}   <- official ranking metric")
    print("=" * 70)
    print("Report mean +/- std, never a single seed. With ~3k training rows the")
    print("run-to-run spread is often 1-3 points, so a 1-point 'improvement'")
    print("from one seed is noise.\n")

    best = max(runs, key=lambda r: r["score"])
    print("--- per-class report, best seed, COARSE ---")
    print(classification_report(best["yc"], best["pc"], target_names=coarse_labels,
                                zero_division=0, digits=3))
    print("--- per-class report, best seed, FINE ---")
    print(classification_report(best["yf"], best["pf"], target_names=fine_labels,
                                zero_division=0, digits=3))
    print("Any class showing 0.000 means it was never predicted. Fix that before")
    print("reaching for a bigger model.")

    Path("results.json").write_text(json.dumps({
        "model": args.model, "lam": args.lam, "constrained": args.constrain,
        "coarse_mean": float(cs.mean()), "coarse_std": float(cs.std()),
        "fine_mean": float(fs.mean()), "fine_std": float(fs.std()),
    }, indent=2))


if __name__ == "__main__":
    main()
