"""
train_multitask.py -- Phase 3 (M3): a pretrained MuRIL encoder + two heads.

MuRIL is a BERT-style model that Google pretrained on 17 Indian languages,
including romanised (Latin-script) text. Instead of learning language from our
3,365 comments, we start from what MuRIL already knows and fine-tune it.

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/train_multitask.py --task multi
    python scripts/train_multitask.py --task coarse
    python scripts/train_multitask.py --task fine

--task multi   one shared encoder, two heads, both losses   (multi-task)
--task coarse  only the tone head is trained                (single-task control)
--task fine    only the hope-type head is trained           (single-task control)

MuRIL is large (about 240 million weights), so the real runs happen on Kaggle:
see notebooks/m3_muril_kaggle.ipynb.

Each finished seed is saved to checkpoints/ (scores and predictions only; add
--save_weights to also keep the ~950 MB model). Running the same command again
skips seeds that are already finished.

Requires: torch, transformers, pandas, scikit-learn
"""

# --------------------------------------------------------------------------
# IMPORTS  (see train_char.py for what "import" and "from ... import" mean)
# --------------------------------------------------------------------------
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
# transformers = the Hugging Face library that downloads and runs pretrained models.
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

COARSE = "Coarse-Grained"
FINE = "Fine-Grained Hope"

# The "Semantic Dependency Between Outputs" table published on the task page.
# WARNING: the training data contains pairs that this table forbids
# (notably Blended Hope + Hopelessness), so --mask empirical is the safer choice.
OFFICIAL_PAIRS = {
    "Optimistic Hope": {"Encouraging Hope"},
    "Realistic Hope":  {"Encouraging Hope"},
    "Inspiring Hope":  {"Encouraging Hope", "Blended Hope"},
    "Fading Hope":     {"Blended Hope", "Discouraging Hope"},
    "Hopelessness":    {"Discouraging Hope"},
}


# ==========================================================================
# PART 1. TURNING TEXT INTO NUMBERS
# MuRIL comes with its own "tokenizer": it splits text into word pieces
# ("supper" -> "sup", "##per") and gives each piece the id MuRIL learned it under.
# ==========================================================================

class HopeDataset(Dataset):
    """Tokenizes all comments once, up front, and hands out one example at a time."""

    def __init__(self, df, tok, c2i, f2i, max_len):
        enc = tok(df["text"].tolist(), truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="pt")
        # input_ids: the word-piece ids, padded to max_len.
        # attention_mask: 1 for a real word piece, 0 for padding.
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.y_c = torch.tensor([c2i[label] for label in df[COARSE]])
        self.y_f = torch.tensor([f2i[label] for label in df[FINE]])

    def __len__(self):
        return len(self.y_c)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.y_c[i], self.y_f[i]


def load_split(path):
    """Read a CSV and strip stray spaces. The file on disk is never changed."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("text", COARSE, FINE):
        df[col] = df[col].str.strip()
    return df


# ==========================================================================
# PART 2. THE MODEL
# Same shape as M2: an ENCODER (now MuRIL) + two HEADS.
# ==========================================================================

class MultiTaskHope(nn.Module):
    def __init__(self, model_name, n_coarse, n_fine, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)   # downloads MuRIL the first time
        h = self.encoder.config.hidden_size                     # 768 for MuRIL-base
        self.dropout = nn.Dropout(dropout)
        self.head_coarse = nn.Linear(h, n_coarse)
        self.head_fine = nn.Linear(h, n_fine)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # MuRIL gives one 768-number vector per word piece. "Mean pooling" averages
        # them (ignoring padding) into one vector for the whole comment.
        mask = attention_mask.unsqueeze(-1).float()       # unsqueeze adds an axis so shapes line up
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return self.head_coarse(pooled), self.head_fine(pooled)


# ==========================================================================
# PART 3. CONSTRAINED DECODING (optional, --mask)
# Some (tone, type) pairs never occur together, e.g. Discouraging + Optimistic.
# With a mask, the model must pick the best pair that is allowed.
# ==========================================================================

def build_valid_mask(df, c2i, f2i, mode):
    """A (3 x 5) table of 0/1: 1 = this (tone, type) pair is allowed.
    'empirical' = pairs seen in the TRAINING data; 'official' = the task page's table."""
    m = np.zeros((len(c2i), len(f2i)))
    if mode == "empirical":
        for c, f in zip(df[COARSE], df[FINE]):
            m[c2i[c], f2i[f]] = 1.0
    elif mode == "official":
        for f, coarses in OFFICIAL_PAIRS.items():
            for c in coarses:
                m[c2i[c], f2i[f]] = 1.0
    return m


def report_violations(df, name):
    """How many rows break the official dependency table?"""
    bad = [(c, f) for c, f in zip(df[COARSE], df[FINE])
           if f in OFFICIAL_PAIRS and c not in OFFICIAL_PAIRS[f]]
    print(f"{name}: rows that break the official pair table: "
          f"{len(bad)} / {len(df)} ({100 * len(bad) / max(len(df), 1):.1f}%)")


def decode(logit_c, logit_f, valid_mask=None):
    """No mask: pick the top tone and the top type separately.
    With a mask: score every allowed (tone, type) pair and pick the best pair."""
    lp_c = torch.log_softmax(logit_c, dim=-1).cpu().numpy()   # scores -> log-probabilities
    lp_f = torch.log_softmax(logit_f, dim=-1).cpu().numpy()
    if valid_mask is None:
        return lp_c.argmax(1), lp_f.argmax(1)

    # joint[b, c, f] = log P(tone c) + log P(type f) for comment b.
    # "None" in an index adds an axis, so (B,3,1) + (B,1,5) gives (B,3,5).
    joint = lp_c[:, :, None] + lp_f[:, None, :]
    joint = np.where(valid_mask[None, :, :] > 0, joint, -1e9)   # forbidden pairs: very low score
    flat = joint.reshape(joint.shape[0], -1).argmax(1)          # best of the 15 pairs
    return flat // valid_mask.shape[1], flat % valid_mask.shape[1]   # back to (tone, type)


# ==========================================================================
# PART 4. TRAINING AND EVALUATION
# ==========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds the GPU (cuda / mps)


def get_device(choice="auto"):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(labels, n, device):
    """Rare classes get a bigger weight in the loss (softened with sqrt, as in M2)."""
    counts = np.bincount(labels, minlength=n).astype(float)
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n * counts)) ** 0.5
    return torch.tensor(w, dtype=torch.float, device=device)


@torch.no_grad()
def evaluate(model, loader, device, valid_mask=None):
    model.eval()                     # dropout OFF
    tc, pc, tf, pf = [], [], [], []
    for ids, att, y_c, y_f in loader:
        logit_c, logit_f = model(ids.to(device), att.to(device))
        a, b = decode(logit_c, logit_f, valid_mask)
        pc += a.tolist()             # .tolist() turns numpy numbers into plain Python ints
        pf += b.tolist()
        tc += y_c.tolist()
        tf += y_f.tolist()
    f1c = f1_score(tc, pc, average="macro", zero_division=0)
    f1f = f1_score(tf, pf, average="macro", zero_division=0)
    return f1c, f1f, (tc, pc, tf, pf)


def selection_score(task, f1c, f1f):
    """Single-task: only the trained head counts. Multi-task: the Joint Score."""
    if task == "coarse":
        return f1c
    if task == "fine":
        return f1f
    return (f1c + f1f) / 2


def run_one_seed(args, train_df, dev_df, tok, c2i, f2i, seed, device):
    set_seed(seed)
    model = MultiTaskHope(args.model, len(c2i), len(f2i)).to(device)

    train_dl = DataLoader(HopeDataset(train_df, tok, c2i, f2i, args.max_len),
                          batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(HopeDataset(dev_df, tok, c2i, f2i, args.max_len),
                        batch_size=args.batch_size * 2)

    w_c = class_weights([c2i[x] for x in train_df[COARSE]], len(c2i), device)
    w_f = class_weights([f2i[x] for x in train_df[FINE]], len(f2i), device)
    loss_c = nn.CrossEntropyLoss(weight=w_c, label_smoothing=args.smooth)
    loss_f = nn.CrossEntropyLoss(weight=w_f, label_smoothing=args.smooth)
    lam = {"multi": 0.5, "coarse": 1.0, "fine": 0.0}[args.task]

    # AdamW = Adam + weight decay, the standard optimiser for fine-tuning BERT models.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # The learning rate rises from 0 over the first 10% of steps ("warmup"), then
    # falls linearly to 0. This stops the first updates from wrecking MuRIL's knowledge.
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    # The pair mask only makes sense when both heads are trained.
    use_mask = args.mask != "none" and args.task == "multi"
    valid_mask = build_valid_mask(train_df, c2i, f2i, args.mask) if use_mask else None

    best = {"score": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()                # dropout ON
        total = 0.0
        for ids, att, y_c, y_f in train_dl:
            logit_c, logit_f = model(ids.to(device), att.to(device))
            loss = (lam * loss_c(logit_c, y_c.to(device))
                    + (1 - lam) * loss_f(logit_f, y_f.to(device)))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()             # move the learning rate one step along its schedule
            total += loss.item()

        f1c, f1f, preds = evaluate(model, dev_dl, device, valid_mask)
        score = selection_score(args.task, f1c, f1f)
        print(f"  seed {seed}  epoch {epoch}  loss {total / len(train_dl):.4f}  "
              f"tone {f1c:.3f}  type {f1f:.3f}  selection {score:.3f}", flush=True)

        # Pick the best epoch by dev macro F1, never by loss.
        if score > best["score"]:
            best = {"score": score, "f1c": f1c, "f1f": f1f, "epoch": epoch, "preds": preds}
            if args.save_weights:
                best["weights"] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
    return best


# ==========================================================================
# PART 5. CHECKPOINTS  (same idea as train_char.py)
# ==========================================================================

def run_name(args):
    """e.g. 'm3_muril-base-cased_multi', plus '_mask-empirical' if a mask is used."""
    short = args.model.split("/")[-1]             # "google/muril-base-cased" -> "muril-base-cased"
    name = f"m3_{short}_{args.task}"
    if args.mask != "none" and args.task == "multi":
        name += f"_mask-{args.mask}"
    return name


def checkpoint_settings(args):
    ignore = ("seeds", "device", "out_dir", "ckpt_dir", "save_weights")
    return {k: v for k, v in vars(args).items() if k not in ignore}


def run_or_resume_seed(args, train_df, dev_df, tok, c2i, f2i, seed, device):
    path = Path(args.ckpt_dir) / f"{run_name(args)}_seed{seed}.pt"
    if path.exists():
        saved = torch.load(path, weights_only=True)
        if saved["settings"] == checkpoint_settings(args):
            print(f"  seed {seed}: checkpoint found ({path}), skipping training", flush=True)
            return saved["best"]
        print(f"  seed {seed}: checkpoint has different settings, training again", flush=True)

    best = run_one_seed(args, train_df, dev_df, tok, c2i, f2i, seed, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"settings": checkpoint_settings(args), "c2i": c2i, "f2i": f2i,
                "best": best}, path)
    print(f"  seed {seed}: saved checkpoint {path}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multi", "coarse", "fine"], default="multi")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--train", default="data/train_clean.csv")
    ap.add_argument("--dev", default="data/dev_multitask.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)        # 2e-5 = 0.00002, typical for BERT
    ap.add_argument("--max_len", type=int, default=64)       # word pieces; p95 comment is 13 words
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--mask", choices=["none", "empirical", "official"], default="none",
                    help="multi-task only: restrict predictions to allowed (tone, type) pairs")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    # action="store_true" makes this an on/off switch: writing --save_weights turns it on.
    ap.add_argument("--save_weights", action="store_true",
                    help="also store the ~950 MB model weights in each checkpoint")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"model={args.model}  task={args.task}  mask={args.mask}  "
          f"device={device}  seeds={args.seeds}", flush=True)

    train_df, dev_df = load_split(args.train), load_split(args.dev)
    c2i = {lab: i for i, lab in enumerate(sorted(train_df[COARSE].unique()))}
    f2i = {lab: i for i, lab in enumerate(sorted(train_df[FINE].unique()))}
    print(f"train={len(train_df)}  dev={len(dev_df)}")
    report_violations(train_df, "train")
    report_violations(dev_df, "dev")

    tok = AutoTokenizer.from_pretrained(args.model)

    start = time.time()
    runs = [run_or_resume_seed(args, train_df, dev_df, tok, c2i, f2i, s, device)
            for s in args.seeds]

    cs = np.array([r["f1c"] for r in runs])
    fs = np.array([r["f1f"] for r in runs])
    trained_tone = args.task in ("multi", "coarse")
    trained_type = args.task in ("multi", "fine")

    print("\n" + "=" * 64)
    print(f"{run_name(args)}   ({len(runs)} seeds, {time.time() - start:.0f}s)")
    if trained_tone:
        print(f"  tone (L1) macro F1 : {cs.mean():.3f} +/- {cs.std():.3f}")
    if trained_type:
        print(f"  type (L2) macro F1 : {fs.mean():.3f} +/- {fs.std():.3f}")
    if args.task == "multi":
        print(f"  JOINT SCORE        : {(cs.mean() + fs.mean()) / 2:.3f}")
    print("  M1 floor to beat   : tone 0.579   type 0.400   joint 0.490")
    print("=" * 64)

    best = max(runs, key=lambda r: r["score"])
    tc, pc, tf, pf = best["preds"]
    if trained_tone:
        print("\nTone, best seed:")
        print(classification_report(tc, pc, labels=list(range(len(c2i))),
                                    target_names=list(c2i), digits=3, zero_division=0))
    if trained_type:
        print("Hope type, best seed:")
        print(classification_report(tf, pf, labels=list(range(len(f2i))),
                                    target_names=list(f2i), digits=3, zero_division=0))

    # Per-class F1 for every seed, to catch a class stuck at 0.000 in any seed.
    per_class = []
    for r in runs:
        tc, pc, tf, pf = r["preds"]
        seed_result = {}
        if trained_tone:
            scores = f1_score(tc, pc, labels=list(range(len(c2i))), average=None, zero_division=0)
            seed_result["tone"] = dict(zip(c2i, scores.round(3).tolist()))
        if trained_type:
            scores = f1_score(tf, pf, labels=list(range(len(f2i))), average=None, zero_division=0)
            seed_result["type"] = dict(zip(f2i, scores.round(3).tolist()))
        per_class.append(seed_result)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"{run_name(args)}.json"
    out.write_text(json.dumps({
        "model": args.model, "task": args.task, "mask": args.mask, "seeds": args.seeds,
        "tone_mean": float(cs.mean()) if trained_tone else None,
        "tone_std": float(cs.std()) if trained_tone else None,
        "type_mean": float(fs.mean()) if trained_type else None,
        "type_std": float(fs.std()) if trained_type else None,
        "best_epochs": [r["epoch"] for r in runs],
        "per_class_f1_by_seed": per_class,
        "settings": vars(args),
    }, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
