"""Accuracy with intervals, alone and as a paired difference on the same test messages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_mean(values, n_boot: int = 10_000, level: float = 0.95,
                   seed: int = 0) -> tuple[float, float, float]:
    """Mean of `values` with a percentile bootstrap interval: (mean, low, high).

    For 0/1 correctness this is accuracy. Resampling is done by drawing counts of
    each distinct value, which is exact and fast for the few distinct values here.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1-d array")
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(values, return_counts=True)
    n = len(values)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):  # chunks keep memory small for many distinct values
        size = min(1000, n_boot - start)
        draws = rng.multinomial(n, counts / n, size=size)
        means[start:start + size] = (draws * uniq).sum(axis=1) / n  # no BLAS: avoids spurious warnings on macOS
    tail = (1 - level) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return float(values.mean()), float(lo), float(hi)


def paired_difference(correct_a, correct_b, **kwargs) -> tuple[float, float, float]:
    """Accuracy of A minus accuracy of B on the same messages, with an interval.

    Pairing removes the noise of which messages happen to be easy, so the
    interval is much tighter than comparing two separate accuracies.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("both arrays must score the same messages")
    return bootstrap_mean(a - b, **kwargs)


def learning_curve_summary(approach: str, results_dir=None, n_full: float = 9000 / 77,
                           **kwargs) -> pd.DataFrame:
    """Accuracy by training size for one approach, from its saved predictions.

    Per size: the mean accuracy over seeds, an interval from resampling test
    messages (on each message's correctness averaged over seeds), and the
    accuracy of each seed so the spread between runs stays visible.
    "all" is placed at its average number of examples per intent, `n_full`.
    """
    from fopstudy.results import RESULTS_DIR, available_runs, load_predictions

    results_dir = RESULTS_DIR if results_dir is None else results_dir
    runs = available_runs(approach, results_dir)
    rows = []
    for n, group in runs.groupby("n_per_class", sort=False):
        correct = np.mean([load_predictions(approach, n, s, results_dir)["correct"].to_numpy()
                           for s in group["seed"]], axis=0)
        acc, lo, hi = bootstrap_mean(correct, **kwargs)
        seed_accs = [float(load_predictions(approach, n, s, results_dir)["correct"].mean())
                     for s in sorted(group["seed"])]
        rows.append({"approach": approach, "n_per_class": n,
                     "x": n_full if n == "all" else float(n),
                     "accuracy": acc, "low": lo, "high": hi, "seed_accuracies": seed_accs})
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)
