"""Approach (c): a small model without fine-tuning.

Embed every message once with a frozen sentence-embedding model (bge-small,
33M parameters), then train only a logistic regression on top. Training takes
seconds, so this is the cheapest thing a team could try first.

Runs in the training environment (see requirements-train.txt):
    .venv-train/bin/python scripts/frozen_baseline.py

Writes test predictions to results/predictions/frozen/, one row per run to
results/runs_frozen.csv, and CPU timings to results/latency_local.csv.
Embeddings are cached in data/ (not committed).
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from fopstudy.data import DATA_DIR, learning_curve_subsets, load_split, validation_split
from fopstudy.results import RESULTS_DIR, save_latency, save_predictions

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SIZES = [5, 10, 25, 50]
SEEDS = [0, 1, 2]
C_GRID = [1, 3, 10, 30, 100, 300, 1000]


def embed(texts, batch_size=64):
    """CLS embeddings, L2-normalised, as the bge model card recommends."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
            cls = model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(out)


def cached_embeddings(split, df):
    path = DATA_DIR / f"emb_bge-small_{split}.npy"
    if not path.exists():
        np.save(path, embed(list(df["text"])))
    return np.load(path)


def fit(x, y, c):
    return LogisticRegression(C=c, max_iter=5000).fit(x, y)


def cpu_latency(classifier, texts, threads=2):
    """One message at a time on 2 CPU threads, a stand-in for a small cloud server.

    Times the embedding model and the classifier together, as a request would
    run them; tokenising is left out, as it is for the fine-tuned model.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
    times = []
    with torch.no_grad():
        for text in texts:
            batch = tok([text], truncation=True, max_length=128, return_tensors="pt")
            start = time.perf_counter()
            cls = model(**batch).last_hidden_state[:, 0]
            classifier.predict(torch.nn.functional.normalize(cls, dim=-1).numpy())
            times.append((time.perf_counter() - start) * 1000)
    torch.set_num_threads(previous)
    return times[10:]  # drop warm-up


def main():
    train, test = load_split("train"), load_split("test")
    x_train = pd.DataFrame(cached_embeddings("train", train), index=train.index)
    x_test = cached_embeddings("test", test)
    fit_df, val_df = validation_split(train, fraction=0.1, seed=0)

    # One regularisation setting for every size, picked on the validation set
    # (average over sizes, seed 0). Tuning per size would flatter the small sizes:
    # a team with 5 labels per intent would not have 1,003 more for tuning.
    tuning_sets = list(learning_curve_subsets(fit_df, SIZES, seed=0).values()) + [fit_df]
    val_acc = {}
    for c in C_GRID:
        scores = [fit(x_train.loc[s.index], s["label"], c).score(x_train.loc[val_df.index], val_df["label"])
                  for s in tuning_sets]
        val_acc[c] = float(np.mean(scores))
        print(f"C={c}: mean validation accuracy {val_acc[c]:.4f}")
    best_c = max(val_acc, key=val_acc.get)
    print(f"Using C={best_c}")

    runs = []
    jobs = [(n, seed, s) for seed in SEEDS
            for n, s in learning_curve_subsets(fit_df, SIZES, seed=seed).items()]
    jobs.append(("all", 0, fit_df))  # the full set has no sampling, so one run
    for n, seed, subset in jobs:
        start = time.time()
        model = fit(x_train.loc[subset.index], subset["label"], best_c)
        seconds = time.time() - start
        preds = model.predict(x_test)
        save_predictions("frozen", n, seed, test["label"], preds)
        runs.append({"approach": "frozen", "model": MODEL, "n_per_class": n, "seed": seed,
                     "n_train": len(subset), "C": best_c,
                     "val_acc": round(model.score(x_train.loc[val_df.index], val_df["label"]), 4),
                     "test_acc": round(float((preds == test["label"].values).mean()), 4),
                     "train_seconds": round(seconds, 2)})
        print(runs[-1])
    pd.DataFrame(runs).to_csv(RESULTS_DIR / "runs_frozen.csv", index=False)

    ms = cpu_latency(model, list(test["text"][:210]))  # the last model trained: all labels
    save_latency("frozen", ms)
    print(f"CPU latency p50 {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms")


if __name__ == "__main__":
    main()
