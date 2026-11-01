"""
train_char.py -- Phase 2 (M2): neural models trained FROM SCRATCH on characters.

Two models, each usable single-task or multi-task:
    cnn     character-level CNN
    bilstm  bidirectional LSTM with attention

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/train_char.py --model cnn    --task multi
    python scripts/train_char.py --model cnn    --task coarse
    python scripts/train_char.py --model cnn    --task fine
    python scripts/train_char.py --model bilstm --task multi

--task multi   one shared encoder, two heads, both losses   (multi-task)
--task coarse  only the tone head is trained                (single-task control)
--task fine    only the hope-type head is trained           (single-task control)

Each seed's best model is saved to checkpoints/ (change with --ckpt_dir). Running
the same command again skips seeds that are already finished.

--test data/multitask_test_withoutlabel.csv  also saves class probabilities (dev and the
UNLABELLED test file) at each seed's best dev epoch, in results/probs/ (format: probs_io.py).
The BiLSTM is slow on the Mac; notebooks/m2_bilstm_kaggle.ipynb runs it on a Kaggle GPU.

Requires: torch, pandas, scikit-learn
"""

# --------------------------------------------------------------------------
# IMPORTS
# "import X" loads a library. "from X import Y" loads one piece of it.
# "import numpy as np" loads numpy but lets us type the short name "np".
# --------------------------------------------------------------------------
import argparse          # reads the --model / --task options from the command line
import json              # saves results as a .json file
import random            # Python's built-in random numbers (we fix its seed)
import time              # measures how long training takes
from pathlib import Path # handles file paths cleanly

import numpy as np
import pandas as pd
import torch
import torch.nn as nn    # "nn" holds the neural-network building blocks (layers)
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from probs_io import load_unlabelled, write_probs   # probs_io.py, in the same folder

# Column names in the CSV files. Defined once so a typo can't creep in later.
COARSE = "Coarse-Grained"
FINE = "Fine-Grained Hope"

# Comments are at most this many characters long; longer ones get cut.
# Your data: p95 = 13 words, roughly 80 characters, so 150 is plenty.
MAX_CHARS = 150


# ==========================================================================
# PART 1. TURNING TEXT INTO NUMBERS
# A neural network only understands numbers. We give every character its own
# id number:  'a' -> 5, 'b' -> 6, ... and turn each comment into a list of ids.
# ==========================================================================

PAD, UNK = 0, 1   # id 0 = padding (filler), id 1 = "unknown character"


def build_vocab(texts, min_count=2):
    """Assign an id to every character that appears at least min_count times
    in the TRAINING texts. Rare characters all share the UNK id."""
    counts = {}                              # {} is an empty dictionary: char -> count
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1   # .get(ch, 0) = count so far, or 0 if new
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for ch, n in sorted(counts.items()):
        if n >= min_count:
            vocab[ch] = len(vocab)           # next free id
    return vocab


def encode(text, vocab):
    """'super' -> [ids...], cut to MAX_CHARS. Unseen characters become UNK."""
    return [vocab.get(ch, UNK) for ch in text[:MAX_CHARS]]
    # [ ... for ch in ... ] is a "list comprehension": a one-line for-loop
    # that builds a list.  text[:MAX_CHARS] means "the first MAX_CHARS characters".


class CharDataset(Dataset):
    """Wraps a table so PyTorch can fetch one example at a time.
    A class is a blueprint; __init__ runs when we create one,
    __len__ answers "how many examples?", __getitem__ answers "give me example i"."""

    def __init__(self, df, vocab, c2i, f2i):
        # self.x means "x stored inside this object", so the other methods can use it.
        self.ids = [encode(t, vocab) for t in df["text"]]
        self.y_c = [c2i[label] for label in df[COARSE]]   # tone label  -> number
        self.y_f = [f2i[label] for label in df[FINE]]     # type label  -> number

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.y_c[i], self.y_f[i]


def collate(batch):
    """Comments have different lengths, but a batch must be one rectangle of
    numbers. So we pad every comment in the batch to the longest one with PAD (0).
    'batch' is a list of (ids, y_c, y_f) tuples from __getitem__."""
    lengths = [max(len(ids), 1) for ids, _, _ in batch]    # "_" = a value we ignore
    longest = max(lengths)
    x = torch.full((len(batch), longest), PAD, dtype=torch.long)  # a grid of zeros
    for row, (ids, _, _) in enumerate(batch):   # enumerate gives (position, item)
        if ids:
            x[row, :len(ids)] = torch.tensor(ids)
    y_c = torch.tensor([b[1] for b in batch])
    y_f = torch.tensor([b[2] for b in batch])
    return x, torch.tensor(lengths), y_c, y_f


# ==========================================================================
# PART 2. THE MODELS
# Every model = an ENCODER (turns characters into one summary vector)
#             + two HEADS (turn that vector into tone scores and type scores).
# A PyTorch model is a class that inherits from nn.Module:
#   __init__  creates the layers;   forward  says how data flows through them.
# ==========================================================================

class TwoHeads(nn.Module):
    """The shared part: dropout + a 3-way tone head + a 5-way type head."""

    def __init__(self, in_dim, n_coarse, n_fine, dropout):
        super().__init__()                     # required first line in every nn.Module
        self.dropout = nn.Dropout(dropout)     # randomly zeroes values while training,
                                               # which stops the model memorising
        self.head_c = nn.Linear(in_dim, n_coarse)  # Linear = weights * input + bias
        self.head_f = nn.Linear(in_dim, n_fine)

    def forward(self, v):
        v = self.dropout(v)
        return self.head_c(v), self.head_f(v)  # two sets of raw scores ("logits")


class CharCNN(nn.Module):
    """Character CNN.
    1. Embedding: each character id becomes a learned vector of 64 numbers.
    2. Convolutions: filters slide across the characters. A width-3 filter looks
       at 3 characters at a time ("sup", "upe", "per"), so it learns n-gram
       patterns like TF-IDF does, except the patterns are learned, not fixed.
    3. Max pooling: for each filter keep its strongest match anywhere in the text.
    4. The resulting vector goes to the two heads."""

    def __init__(self, vocab_size, n_coarse, n_fine, emb=64, filters=128,
                 widths=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        # nn.ModuleList holds several layers so PyTorch tracks all their weights.
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, filters, kernel_size=w, padding=w - 1) for w in widths])
        self.heads = TwoHeads(filters * len(widths), n_coarse, n_fine, dropout)

    def forward(self, x, lengths):
        e = self.emb(x)              # shape (batch, chars, 64)
        e = e.transpose(1, 2)        # Conv1d wants (batch, 64, chars), so swap axes 1 and 2
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(e))          # relu(z) = max(0, z): keeps positives only
            pooled.append(h.max(dim=2).values)   # strongest match along the text
        v = torch.cat(pooled, dim=1)         # join the 4 results: 4 x 128 = 512 numbers
        logit_c, logit_f = self.heads(v)
        return logit_c, logit_f, None        # None = this model has no attention weights


class BiLSTMAttention(nn.Module):
    """BiLSTM with attention.
    1. Embedding, as in the CNN.
    2. A bidirectional LSTM reads the characters left-to-right AND right-to-left,
       producing, at every position, a vector that summarises its context.
    3. Attention scores how important each position is, turns the scores into
       weights that sum to 1 (softmax), and takes the weighted average.
       Those weights can be plotted later to see what the model focused on.
    4. The averaged vector goes to the two heads."""

    def __init__(self, vocab_size, n_coarse, n_fine, emb=64, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.att_proj = nn.Linear(2 * hidden, 2 * hidden)     # 2x because two directions
        self.att_score = nn.Linear(2 * hidden, 1, bias=False)
        self.heads = TwoHeads(2 * hidden, n_coarse, n_fine, dropout)

    def forward(self, x, lengths):
        e = self.emb(x)
        # "Packing" tells the LSTM each comment's real length, so the backward
        # direction does not start by reading padding.
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))   # back to (batch, chars, 256)

        scores = self.att_score(torch.tanh(self.att_proj(h))).squeeze(-1)  # (batch, chars)
        mask = x != PAD                                   # True where there is a real character
        scores = scores.masked_fill(~mask, float("-inf")) # padding gets score -infinity...
        weights = torch.softmax(scores, dim=1)            # ...so its weight becomes exactly 0
        v = (weights.unsqueeze(-1) * h).sum(dim=1)        # weighted average -> (batch, 256)
        logit_c, logit_f = self.heads(v)
        return logit_c, logit_f, weights


# ==========================================================================
# PART 3. TRAINING AND EVALUATION
# ==========================================================================

def set_seed(seed):
    """Fix every source of randomness, so the same seed gives the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(choice="auto"):
    """'auto' = use a GPU if there is one (mps on a Mac, cuda on Kaggle), otherwise the CPU.
    Any other choice ('cpu', 'mps', 'cuda') is used as given."""
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(labels, n, device):
    """Rare classes get a bigger weight in the loss, so mistakes on them cost more.
    sqrt softens it: Fading Hope is ~4.6x rarer than Inspiring, its weight ~2.1x."""
    counts = np.bincount(labels, minlength=n).astype(float)   # how many of each class
    counts[counts == 0] = 1.0
    w = (counts.sum() / (n * counts)) ** 0.5                  # ** means "to the power of"
    return torch.tensor(w, dtype=torch.float, device=device)


@torch.no_grad()   # a "decorator": inside this function PyTorch skips gradient bookkeeping
def evaluate(model, loader, device):
    """Run the model on every dev batch and collect true and predicted labels."""
    model.eval()                     # switches dropout OFF for evaluation
    tc, pc, tf, pf = [], [], [], []
    for x, lengths, y_c, y_f in loader:
        logit_c, logit_f, _ = model(x.to(device), lengths)
        pc += logit_c.argmax(dim=1).cpu().tolist()   # argmax = index of the highest score
        pf += logit_f.argmax(dim=1).cpu().tolist()
        tc += y_c.tolist()
        tf += y_f.tolist()
    f1c = f1_score(tc, pc, average="macro", zero_division=0)
    f1f = f1_score(tf, pf, average="macro", zero_division=0)
    return f1c, f1f, (tc, pc, tf, pf)


@torch.no_grad()
def predict_probs(model, loader, device):
    """Class probabilities and predicted classes for every comment in the loader.
    Everything is kept as plain Python lists, which can be stored in a checkpoint."""
    model.eval()
    prob_c, prob_f = [], []
    for x, lengths, _, _ in loader:             # the labels are not needed here
        logit_c, logit_f, _ = model(x.to(device), lengths)
        # softmax turns raw scores into probabilities that add up to 1.
        prob_c += torch.softmax(logit_c, dim=1).cpu().tolist()
        prob_f += torch.softmax(logit_f, dim=1).cpu().tolist()
    return {"prob_c": prob_c, "prob_f": prob_f,
            "pred_c": [int(np.argmax(p)) for p in prob_c],   # highest probability
            "pred_f": [int(np.argmax(p)) for p in prob_f]}


def selection_score(task, f1c, f1f):
    """Which number decides the "best" epoch. For a single-task model we only
    look at the head that was actually trained -- the untrained head is random."""
    if task == "coarse":
        return f1c
    if task == "fine":
        return f1f
    return (f1c + f1f) / 2           # multi-task: the official Joint Score


def run_one_seed(args, train_df, dev_df, test_df, vocab, c2i, f2i, seed, device):
    set_seed(seed)
    train_ds = CharDataset(train_df, vocab, c2i, f2i)
    dev_ds = CharDataset(dev_df, vocab, c2i, f2i)
    # A DataLoader hands out the data in batches; shuffle=True mixes the order every epoch.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_dl = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
    test_dl = None
    if test_df is not None:   # only when --test is given
        test_dl = DataLoader(CharDataset(test_df, vocab, c2i, f2i), batch_size=256,
                             shuffle=False, collate_fn=collate)

    Model = CharCNN if args.model == "cnn" else BiLSTMAttention
    model = Model(len(vocab), len(c2i), len(f2i)).to(device)   # .to(device) moves it to the GPU

    w_c = class_weights(train_ds.y_c, len(c2i), device)
    w_f = class_weights(train_ds.y_f, len(f2i), device)
    loss_c = nn.CrossEntropyLoss(weight=w_c, label_smoothing=args.smooth)
    loss_f = nn.CrossEntropyLoss(weight=w_f, label_smoothing=args.smooth)
    # Adam = the optimiser: the rule for nudging every weight to reduce the loss.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # How much each loss counts, depending on the task.
    lam = {"multi": 0.5, "coarse": 1.0, "fine": 0.0}[args.task]

    best = {"score": -1.0}
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):      # range(1, 21) = 1, 2, ..., 20
        model.train()                            # switches dropout back ON
        total = 0.0
        for x, lengths, y_c, y_f in train_dl:
            x, y_c, y_f = x.to(device), y_c.to(device), y_f.to(device)
            logit_c, logit_f, _ = model(x, lengths)            # 1. forward pass
            loss = lam * loss_c(logit_c, y_c) + (1 - lam) * loss_f(logit_f, y_f)  # 2. loss
            opt.zero_grad()                                    # 3. clear old gradients
            loss.backward()                                    # 4. compute new gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #    cap them (stability)
            opt.step()                                         # 5. update the weights
            total += loss.item()

        f1c, f1f, preds = evaluate(model, dev_dl, device)
        score = selection_score(args.task, f1c, f1f)
        # f"..." strings insert values: {f1c:.3f} prints f1c with 3 decimals.
        # flush=True prints the line immediately, so progress shows up in log files.
        print(f"  seed {seed}  epoch {epoch:2d}  loss {total / len(train_dl):.4f}  "
              f"tone {f1c:.3f}  type {f1f:.3f}  selection {score:.3f}", flush=True)

        if score > best["score"]:
            # Keep a copy of the weights from this best epoch, so we can save them later.
            # {k: v ... for k, v in ...} is a "dict comprehension": a one-line loop that
            # builds a dictionary. .cpu().clone() makes an independent copy in normal memory.
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {"score": score, "f1c": f1c, "f1f": f1f, "epoch": epoch, "preds": preds,
                    "weights": weights}
            if test_dl is not None:
                # Probabilities from this best epoch, for combining models later.
                best["probs"] = {"dev": predict_probs(model, dev_dl, device),
                                 "test": predict_probs(model, test_dl, device)}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:      # early stopping: no improvement lately
                print(f"  stopping early: no improvement for {args.patience} epochs")
                break
    return best


# ==========================================================================
# PART 4. CHECKPOINTS
# After each seed finishes, its best model is saved to a .pt file. If the run
# is interrupted (e.g. a Kaggle session times out), running the same command
# again skips every seed that already has a checkpoint and continues with the rest.
# ==========================================================================

def checkpoint_path(args, seed):
    return Path(args.ckpt_dir) / f"m2_{args.model}_{args.task}_seed{seed}.pt"


def checkpoint_settings(args):
    """The settings that change what gets trained. A saved checkpoint is only
    reused if these match exactly; otherwise that seed is trained again."""
    # These don't change the model, so they are left out.
    ignore = ("seeds", "device", "out_dir", "ckpt_dir", "test", "probs_dir")
    return {k: v for k, v in vars(args).items() if k not in ignore}


def run_or_resume_seed(args, train_df, dev_df, test_df, vocab, c2i, f2i, seed, device):
    path = checkpoint_path(args, seed)
    if path.exists():
        saved = torch.load(path, weights_only=True)
        same = saved["settings"] == checkpoint_settings(args)
        # With --test we also need the probabilities; older checkpoints don't have them.
        has_probs = test_df is None or "probs" in saved["best"]
        if same and has_probs:
            print(f"  seed {seed}: checkpoint found ({path}), skipping training", flush=True)
            return saved["best"]
        reason = "different settings" if not same else "no saved probabilities"
        print(f"  seed {seed}: checkpoint has {reason}, training again", flush=True)

    best = run_one_seed(args, train_df, dev_df, test_df, vocab, c2i, f2i, seed, device)
    path.parent.mkdir(parents=True, exist_ok=True)   # create the folder if it is missing
    torch.save({"settings": checkpoint_settings(args), "vocab": vocab,
                "c2i": c2i, "f2i": f2i, "best": best}, path)
    print(f"  seed {seed}: saved checkpoint {path}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "bilstm"], default="cnn")
    ap.add_argument("--task", choices=["multi", "coarse", "fine"], default="multi")
    ap.add_argument("--train", default="data/train_clean.csv")
    ap.add_argument("--dev", default="data/dev_multitask.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])  # nargs="+" = one or more
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)       # 1e-3 means 0.001
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--out_dir", default="results")        # where the .json results go
    ap.add_argument("--ckpt_dir", default="checkpoints")   # where the .pt model files go
    ap.add_argument("--test", default=None,
                    help="UNLABELLED test file; if given, save dev + test probabilities")
    ap.add_argument("--probs_dir", default="results/probs")  # where probability files go
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"model={args.model}  task={args.task}  device={device}  seeds={args.seeds}")

    train_df = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    dev_df = pd.read_csv(args.dev, dtype=str, keep_default_na=False)
    for df in (train_df, dev_df):
        # Lowercase, so "SUPER" and "super" become the same characters.
        df["text"] = df["text"].str.strip().str.lower()

    vocab = build_vocab(train_df["text"])        # built from TRAIN only, never dev
    c2i = {lab: i for i, lab in enumerate(sorted(train_df[COARSE].unique()))}
    f2i = {lab: i for i, lab in enumerate(sorted(train_df[FINE].unique()))}
    print(f"train={len(train_df)}  dev={len(dev_df)}  characters in vocabulary={len(vocab)}")

    test_df = None
    if args.test:
        # Placeholder labels (the first tone and type), never used: the file has no labels.
        test_df = load_unlabelled(args.test, list(c2i)[0], list(f2i)[0])
        test_df["text"] = test_df["text"].str.lower()      # same lowercasing as train/dev
        print(f"test (unlabelled)={len(test_df)}: probabilities will be saved")

    start = time.time()
    runs = [run_or_resume_seed(args, train_df, dev_df, test_df, vocab, c2i, f2i, s, device)
            for s in args.seeds]

    cs = np.array([r["f1c"] for r in runs])
    fs = np.array([r["f1f"] for r in runs])
    print("\n" + "=" * 64)
    print(f"{args.model} / {args.task}   ({len(runs)} seeds, {time.time() - start:.0f}s)")
    if args.task in ("multi", "coarse"):
        print(f"  tone (L1) macro F1 : {cs.mean():.3f} +/- {cs.std():.3f}")
    if args.task in ("multi", "fine"):
        print(f"  type (L2) macro F1 : {fs.mean():.3f} +/- {fs.std():.3f}")
    if args.task == "multi":
        print(f"  JOINT SCORE        : {(cs.mean() + fs.mean()) / 2:.3f}")
    print("  M1 floor to beat   : tone 0.579   type 0.400   joint 0.490")
    print("=" * 64)

    # Per-class report for the best seed, only for the head(s) that were trained.
    best = max(runs, key=lambda r: r["score"])   # lambda = a tiny unnamed function
    tc, pc, tf, pf = best["preds"]
    if args.task in ("multi", "coarse"):
        print("\nTone, best seed:")
        print(classification_report(tc, pc, target_names=list(c2i), digits=3, zero_division=0))
    if args.task in ("multi", "fine"):
        print("Hope type, best seed:")
        print(classification_report(tf, pf, target_names=list(f2i), digits=3, zero_division=0))

    # Per-class F1 for EVERY seed (not just the best one), so we can spot a class
    # that collapses to 0.000 in any seed. average=None gives one F1 per class.
    # A single-task model's other head was never trained, so its score means nothing.
    # We store None (shown as null in the .json) instead of a misleading number.
    trained_tone = args.task in ("multi", "coarse")
    trained_type = args.task in ("multi", "fine")

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
        # zip(a, b) pairs items up: ("Blended Hope", 0.51), ...; dict() turns the pairs into a dictionary.
        per_class.append(seed_result)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"m2_{args.model}_{args.task}.json"
    out.write_text(json.dumps({
        "model": args.model, "task": args.task, "seeds": args.seeds,
        "tone_mean": float(cs.mean()) if trained_tone else None,
        "tone_std": float(cs.std()) if trained_tone else None,
        "type_mean": float(fs.mean()) if trained_type else None,
        "type_std": float(fs.std()) if trained_type else None,
        "best_epochs": [r["epoch"] for r in runs],
        "per_class_f1_by_seed": per_class,
        "settings": vars(args),
    }, indent=2))
    print(f"saved: {out}")

    # One probability file per seed and split, e.g. results/probs/m2_cnn_multi/dev_seed1.csv
    if test_df is not None:
        for seed, r in zip(args.seeds, runs):
            for split, df in (("dev", dev_df), ("test", test_df)):
                p = r["probs"][split]
                path = Path(args.probs_dir) / f"m2_{args.model}_{args.task}" / f"{split}_seed{seed}.csv"
                write_probs(path, df["Id"], p["pred_c"], p["pred_f"], p["prob_c"], p["prob_f"],
                            list(c2i), list(f2i))
        print(f"saved probabilities: {Path(args.probs_dir) / f'm2_{args.model}_{args.task}'}/")


# This line means: run main() only when the file is executed directly,
# not when another script imports it.
if __name__ == "__main__":
    main()
