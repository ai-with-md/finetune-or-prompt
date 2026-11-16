"""Approach (a): fine-tune a small open model (ModernBERT-base, 150M parameters) on this Mac.

For every training size (5, 10, 25, 50 examples per intent, and all of them)
and three seeds, fine-tune the whole model and save its test predictions.

Fair-play rules, so the small sizes are not flattered:
- The learning rate is picked once, on the validation set (at 10 and 50
  examples per intent), and then used for every run. There is no early
  stopping on validation data: a team with 5 labels per intent would not
  have 1,003 more for tuning.
- The training length follows a fixed rule: at least 4 passes over the data
  and at least 500 steps.

Runs in the training environment (see requirements-train.txt):
    .venv-train/bin/python scripts/finetune.py

Finished runs are skipped on a rerun, so an interrupted job can resume.
Writes results/predictions/finetuned/, results/runs_finetuned.csv,
results/tuning_finetuned.csv and results/latency_local.csv. Model weights are
never saved.
"""

import math
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from fopstudy.data import learning_curve_subsets, load_split, validation_split
from fopstudy.results import RESULTS_DIR, prediction_path, save_predictions

MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
SIZES = [5, 10, 25, 50]
SEEDS = [0, 1, 2]
LR_GRID = [5e-5, 1e-4, 2e-4]
TUNING_SIZES = [10, 50]
BATCH = 32
MAX_LEN = 64
MIN_EPOCHS = 4
MIN_STEPS = 500
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def encode(texts):
    return tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


def predict(model, texts, device=DEVICE, batch=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = {k: v.to(device) for k, v in encode(texts[i:i + batch]).items()}
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def train(subset, lr, seed):
    """Fine-tune from the pretrained weights; returns the model and run details."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=77).to(DEVICE)
    texts, labels = list(subset["text"]), subset["label"].to_numpy()
    steps_per_epoch = math.ceil(len(texts) / BATCH)
    epochs = max(MIN_EPOCHS, math.ceil(MIN_STEPS / steps_per_epoch))
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    model.train()
    start = time.time()
    for _ in range(epochs):
        order = rng.permutation(len(texts))
        for i in range(0, len(texts), BATCH):
            idx = order[i:i + BATCH]
            enc = {k: v.to(DEVICE) for k, v in encode([texts[j] for j in idx]).items()}
            loss = model(**enc, labels=torch.tensor(labels[idx]).to(DEVICE)).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    return model, {"epochs": epochs, "steps": total, "train_seconds": round(time.time() - start, 1)}


def accuracy(model, df):
    return round(float((predict(model, list(df["text"])) == df["label"].to_numpy()).mean()), 4)


def pick_learning_rate(fit_df, val_df):
    """Validation accuracy for each learning rate at 10 and 50 examples per intent (seed 0).

    Rows are saved as they finish, so an interrupted search resumes where it stopped.
    """
    path = RESULTS_DIR / "tuning_finetuned.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["lr"], r["n_per_class"]) for r in rows}
    for lr in LR_GRID:
        for n in TUNING_SIZES:
            if (lr, n) in done:
                continue
            model, info = train(learning_curve_subsets(fit_df, [n], seed=0)[n], lr, seed=0)
            rows.append({"lr": lr, "n_per_class": n, "val_acc": accuracy(model, val_df), **info})
            pd.DataFrame(rows).to_csv(path, index=False)
            print(rows[-1], flush=True)
            del model
    tuning = pd.DataFrame(rows)
    mean_val = tuning[tuning["lr"].isin(LR_GRID)].groupby("lr")["val_acc"].mean()
    print(mean_val.to_string(), flush=True)
    return float(mean_val.idxmax())


def cpu_latency(model, texts, threads=2):
    """One message at a time on 2 CPU threads, a stand-in for a small cloud server."""
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.to("cpu").eval()
    times = []
    with torch.no_grad():
        for text in texts:
            enc = encode([text])
            start = time.perf_counter()
            model(**enc)
            times.append((time.perf_counter() - start) * 1000)
    torch.set_num_threads(previous)
    return times[10:]  # drop warm-up


def main():
    train_df, test = load_split("train"), load_split("test")
    fit_df, val_df = validation_split(train_df, fraction=0.1, seed=0)
    lr = pick_learning_rate(fit_df, val_df)
    print(f"Using learning rate {lr}", flush=True)

    runs_path = RESULTS_DIR / "runs_finetuned.csv"
    runs = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    for seed in SEEDS:
        subsets = {**learning_curve_subsets(fit_df, SIZES, seed=seed), "all": fit_df}
        for n, subset in subsets.items():
            if prediction_path("finetuned", n, seed).exists():
                continue
            model, info = train(subset, lr, seed)
            preds = predict(model, list(test["text"]))
            save_predictions("finetuned", n, seed, test["label"], preds)
            runs.append({"approach": "finetuned", "model": MODEL, "n_per_class": n, "seed": seed,
                         "n_train": len(subset), "lr": lr, **info,
                         "val_acc": accuracy(model, val_df),
                         "test_acc": round(float((preds == test["label"].to_numpy()).mean()), 4),
                         "device": DEVICE.type, "date": str(date.today())})
            pd.DataFrame(runs).to_csv(runs_path, index=False)
            print(runs[-1], flush=True)
            if n == "all" and seed == 0:
                ms = cpu_latency(model, list(test["text"][:210]))
                pd.DataFrame({"approach": "finetuned", "ms": np.round(ms, 2)}).to_csv(
                    RESULTS_DIR / "latency_local.csv", index=False)
                print(f"CPU latency p50 {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms", flush=True)
            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
