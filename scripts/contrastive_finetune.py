"""Approach (d): fine-tune the small embedding model on pairs of examples (the SetFit recipe).

Fine-tuning a whole model with a new classification head (approach a) is
known to work badly with a handful of examples per intent. SetFit
(Tunstall et al., 2022) does it differently: it first fine-tunes a
sentence-embedding model to put messages of the same intent close together
and others far apart, using pairs of training examples, and then trains a
linear classifier on the new embeddings. This script runs that recipe on the
same bge-small model and the same nested training sets as the frozen
baseline, at the small sizes where it is meant to help.

Fair-play rules:
- The pair training uses SetFit's published defaults (20 rounds of pairs,
  one epoch, batch 16, learning rate 2e-5, 10% warm-up), not tuned here.
- The classifier's regularisation is picked once on the validation set
  (first draw, 5 and 10 examples per intent), then used for every run.

Runs in the training environment (see requirements-train.txt):
    .venv-train/bin/python scripts/contrastive_finetune.py

Writes results/predictions/setfit/, results/runs_setfit.csv and
results/tuning_setfit.csv. Model weights are never saved.
"""

import math
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from fopstudy.data import contrastive_pairs, learning_curve_subsets, load_split, validation_split
from fopstudy.results import RESULTS_DIR, save_predictions

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SIZES = [5, 10, 25]
SEEDS = [0, 1, 2]
TUNING_SIZES = [5, 10]
C_GRID = [1, 3, 10, 30, 100, 300, 1000]
ITERATIONS = 20
BATCH = 16
LR = 2e-5
WARMUP = 0.1
MAX_LEN = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def embed(model, texts, batch=128):
    """CLS embeddings, L2-normalised, as for the frozen baseline."""
    out = []
    with torch.set_grad_enabled(model.training):
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(DEVICE)
            out.append(F.normalize(model(**enc).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out)


def train_body(texts, labels, seed):
    """Contrastive fine-tuning: cosine similarity of each pair pushed towards 1 or 0."""
    torch.manual_seed(seed)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(DEVICE)
    pairs = contrastive_pairs(labels, ITERATIONS, seed)
    steps = math.ceil(len(pairs) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    schedule = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    model.train()
    for b in range(steps):
        batch = pairs[b * BATCH:(b + 1) * BATCH]
        a = embed(model, [texts[i] for i in batch[:, 0]], batch=BATCH)
        p = embed(model, [texts[i] for i in batch[:, 1]], batch=BATCH)
        target = torch.tensor(batch[:, 2], dtype=torch.float32, device=DEVICE)
        loss = F.mse_loss((a * p).sum(-1), target)
        loss.backward()
        opt.step()
        schedule.step()
        opt.zero_grad()
    model.eval()
    return model, len(pairs), steps


def main():
    train_df, test = load_split("train"), load_split("test")
    fit_df, val_df = validation_split(train_df, fraction=0.1, seed=0)
    runs, store = [], {}
    order = [(n, 0) for n in TUNING_SIZES] + [(n, s) for s in SEEDS for n in SIZES if (n, s) not in
                                                [(t, 0) for t in TUNING_SIZES]]
    best_c = None
    for n, seed in order:
        subset = learning_curve_subsets(fit_df, SIZES, seed=seed)[n]
        start = time.time()
        model, n_pairs, steps = train_body(list(subset["text"]), subset["label"].to_numpy(), seed)
        seconds = time.time() - start
        with torch.no_grad():
            x_train, x_val, x_test = (embed(model, list(df["text"])).cpu().numpy() for df in (subset, val_df, test))
        store[(n, seed)] = (subset, x_train, x_val, x_test, n_pairs, steps, seconds)
        print(f"n={n} seed={seed}: {steps} steps in {seconds:.0f} s", flush=True)
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        if best_c is None and all((t, 0) in store for t in TUNING_SIZES):
            # One regularisation setting for every run, as for the frozen baseline.
            tuning = []
            for c in C_GRID:
                for t in TUNING_SIZES:
                    sub, xt, xv = store[(t, 0)][:3]
                    clf = LogisticRegression(C=c, max_iter=5000).fit(xt, sub["label"])
                    tuning.append({"C": c, "n_per_class": t, "val_acc": round(clf.score(xv, val_df["label"]), 4)})
            tuning = pd.DataFrame(tuning)
            tuning.to_csv(RESULTS_DIR / "tuning_setfit.csv", index=False)
            best_c = float(tuning.groupby("C")["val_acc"].mean().idxmax())
            print(f"Using C={best_c}", flush=True)

    for (n, seed), (subset, x_train, x_val, x_test, n_pairs, steps, seconds) in sorted(store.items()):
        clf = LogisticRegression(C=best_c, max_iter=5000).fit(x_train, subset["label"])
        preds = clf.predict(x_test)
        save_predictions("setfit", n, seed, test["label"], preds)
        runs.append({"approach": "setfit", "model": MODEL, "n_per_class": n, "seed": seed,
                     "n_train": len(subset), "pairs": n_pairs, "steps": steps, "C": best_c,
                     "val_acc": round(clf.score(x_val, val_df["label"]), 4),
                     "test_acc": round(float((preds == test["label"].to_numpy()).mean()), 4),
                     "train_seconds": round(seconds, 1), "device": DEVICE.type, "date": str(date.today())})
        print(runs[-1], flush=True)
    pd.DataFrame(runs).to_csv(RESULTS_DIR / "runs_setfit.csv", index=False)


if __name__ == "__main__":
    main()
