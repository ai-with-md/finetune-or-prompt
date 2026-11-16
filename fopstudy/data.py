"""Load Banking77 and cut it into the splits every notebook shares.

All splits are stratified by intent and fixed by a seed, so a notebook rerun
sees exactly the same examples. The learning-curve subsets are nested: the 10
examples per intent include the 5, the 25 include the 10, and so on. That way a
bigger training set only ever adds examples, and the curve measures the value
of more labels rather than the luck of a new draw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_labels(data_dir: Path = DATA_DIR) -> list[str]:
    """Intent names in label order, as published with the dataset."""
    with open(Path(data_dir) / "labels.json") as f:
        return json.load(f)


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load "train" or "test" with columns text, label and intent."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', not {split!r}")
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    df = pd.read_parquet(path)
    names = load_labels(data_dir)
    df["intent"] = [names[i] for i in df["label"]]
    return df[["text", "label", "intent"]]


def _shuffled_by_class(df: pd.DataFrame, seed: int, label_col: str) -> dict:
    """Index labels of each class in a fixed random order, classes in sorted order."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(label_col, sort=True).groups
    return {label: list(rng.permutation(np.asarray(idx))) for label, idx in groups.items()}


def per_class_sample(df: pd.DataFrame, n_per_class: int, seed: int = 0,
                     label_col: str = "label") -> pd.DataFrame:
    """Up to n_per_class rows of every class (all rows if a class has fewer).

    With the same seed, a larger n always contains the rows of a smaller n.
    """
    if n_per_class < 1:
        raise ValueError("n_per_class must be at least 1")
    order = _shuffled_by_class(df, seed, label_col)
    keep = [i for idx in order.values() for i in idx[:n_per_class]]
    return df.loc[keep]


def learning_curve_subsets(df: pd.DataFrame, sizes: Iterable[int], seed: int = 0,
                           label_col: str = "label") -> dict[int, pd.DataFrame]:
    """Nested training sets with the given number of examples per class."""
    return {n: per_class_sample(df, n, seed, label_col) for n in sorted(set(sizes))}


def validation_split(df: pd.DataFrame, fraction: float = 0.1, seed: int = 0,
                     label_col: str = "label") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a stratified share of every class for tuning; never tune on the test set.

    Each class gives round(fraction * size) rows, and at least one.
    Returns (train, validation) with the original index kept.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    order = _shuffled_by_class(df, seed, label_col)
    held = [i for idx in order.values() for i in idx[:max(1, round(fraction * len(idx)))]]
    val = df.loc[held]
    return df.drop(index=held), val
