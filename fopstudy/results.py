"""Read and write test-set predictions, the hand-off between model runs and notebooks.

Every approach writes one small CSV per training size and seed, with one row
per test message: its position in the test set, the true label and the
prediction. Notebooks only read these files, so they rerun without model
weights, a GPU or an API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
Size = Union[int, str]  # examples per intent, or "all"


def prediction_path(approach: str, n_per_class: Size, seed: int,
                    results_dir: Path = RESULTS_DIR) -> Path:
    return Path(results_dir) / "predictions" / approach / f"n{n_per_class}_seed{seed}.csv"


def save_predictions(approach: str, n_per_class: Size, seed: int, labels, preds,
                     results_dir: Path = RESULTS_DIR) -> Path:
    labels, preds = list(labels), list(preds)
    if len(labels) != len(preds):
        raise ValueError("labels and preds must have the same length")
    path = prediction_path(approach, n_per_class, seed, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": range(len(labels)), "label": labels, "pred": preds}).to_csv(path, index=False)
    return path


def load_predictions(approach: str, n_per_class: Size, seed: int,
                     results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Predictions with a boolean `correct` column added."""
    df = pd.read_csv(prediction_path(approach, n_per_class, seed, results_dir))
    df["correct"] = df["label"] == df["pred"]
    return df


def available_runs(approach: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Every (n_per_class, seed) saved for an approach; n is an int or "all"."""
    rows = []
    for path in sorted((Path(results_dir) / "predictions" / approach).glob("n*_seed*.csv")):
        n, seed = path.stem[1:].split("_seed")
        rows.append({"n_per_class": int(n) if n.isdigit() else n, "seed": int(seed)})
    return pd.DataFrame(rows, columns=["n_per_class", "seed"])
