"""Intervals must be centred on the observed accuracy and cover the truth about 95% of the time."""

import numpy as np
import pytest

from fopstudy.evaluate import bootstrap_mean, paired_difference


def test_bootstrap_mean_of_80_percent():
    mean, lo, hi = bootstrap_mean([1] * 80 + [0] * 20, seed=1)
    assert mean == pytest.approx(0.8)
    assert lo < 0.8 < hi
    # The binomial standard error is 0.04, so a 95% interval is about 0.16 wide.
    assert 0.13 < hi - lo < 0.19


def test_bootstrap_is_repeatable():
    data = np.random.default_rng(0).random(300)
    assert bootstrap_mean(data, seed=5) == bootstrap_mean(data, seed=5)


def test_interval_covers_the_true_accuracy_about_95_percent_of_the_time():
    rng = np.random.default_rng(42)
    hits = 0
    runs = 300
    for i in range(runs):
        sample = rng.random(200) < 0.7
        _, lo, hi = bootstrap_mean(sample, n_boot=2000, seed=i)
        hits += lo <= 0.7 <= hi
    assert 0.90 <= hits / runs <= 0.98


def test_paired_difference():
    a = np.array([1, 1, 1, 0, 1, 0, 1, 1])
    b = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    diff, lo, hi = paired_difference(a, b, seed=0)
    assert diff == pytest.approx(0.25)
    assert lo <= diff <= hi
    assert paired_difference(a, a) == (0.0, 0.0, 0.0)


def test_input_checks():
    with pytest.raises(ValueError):
        bootstrap_mean([])
    with pytest.raises(ValueError):
        paired_difference([1, 0], [1, 0, 1])


def test_learning_curve_summary(tmp_path):
    from fopstudy.evaluate import learning_curve_summary
    from fopstudy.results import save_predictions

    labels = [0, 1, 2, 3]
    save_predictions("frozen", 5, 0, labels, [0, 1, 0, 0], results_dir=tmp_path)      # 50%
    save_predictions("frozen", 5, 1, labels, [0, 1, 2, 0], results_dir=tmp_path)      # 75%
    save_predictions("frozen", "all", 0, labels, [0, 1, 2, 3], results_dir=tmp_path)  # 100%
    out = learning_curve_summary("frozen", results_dir=tmp_path, n_full=117.0, n_boot=500)
    assert list(out["n_per_class"]) == [5, "all"]
    assert list(out["x"]) == [5.0, 117.0]
    assert out.loc[0, "accuracy"] == pytest.approx(0.625)
    assert out.loc[0, "seed_accuracies"] == [0.5, 0.75]
    assert out.loc[1, "accuracy"] == 1.0
