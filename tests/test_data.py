"""Splits must be stratified, repeatable and nested. Uses made-up data, no download needed."""

import json

import numpy as np
import pandas as pd
import pytest

from fopstudy.data import (learning_curve_subsets, load_split, per_class_sample,
                           validation_split)


@pytest.fixture
def df():
    # Uneven classes like the real data: 3 intents with 40, 12 and 3 examples.
    sizes = {0: 40, 1: 12, 2: 3}
    labels = np.repeat(list(sizes), list(sizes.values()))
    out = pd.DataFrame({"text": [f"message {i}" for i in range(len(labels))], "label": labels})
    return out.sample(frac=1, random_state=7)  # shuffled, with a non-sorted index


def test_per_class_sample_takes_n_of_each_class_and_caps_small_ones(df):
    sample = per_class_sample(df, 5, seed=1)
    assert sample["label"].value_counts().to_dict() == {0: 5, 1: 5, 2: 3}


def test_per_class_sample_is_repeatable_and_depends_on_seed(df):
    a = per_class_sample(df, 5, seed=1).index
    assert list(a) == list(per_class_sample(df, 5, seed=1).index)
    assert set(a) != set(per_class_sample(df, 5, seed=2).index)


def test_learning_curve_subsets_are_nested(df):
    subsets = learning_curve_subsets(df, [10, 2, 5], seed=3)
    assert list(subsets) == [2, 5, 10]
    assert set(subsets[2].index) <= set(subsets[5].index) <= set(subsets[10].index)
    assert subsets[10]["label"].value_counts().to_dict() == {0: 10, 1: 10, 2: 3}


def test_per_class_sample_rejects_zero(df):
    with pytest.raises(ValueError):
        per_class_sample(df, 0)


def test_validation_split_is_stratified_and_disjoint(df):
    train, val = validation_split(df, fraction=0.25, seed=0)
    assert val["label"].value_counts().to_dict() == {0: 10, 1: 3, 2: 1}
    assert not set(train.index) & set(val.index)
    assert len(train) + len(val) == len(df)


def test_validation_split_rejects_bad_fraction(df):
    with pytest.raises(ValueError):
        validation_split(df, fraction=1.0)


def test_load_split_adds_intent_names(tmp_path):
    pd.DataFrame({"text": ["where is my card", "top up failed"], "label": [1, 0]}).to_parquet(
        tmp_path / "train.parquet")
    (tmp_path / "labels.json").write_text(json.dumps(["top_up_failed", "card_arrival"]))
    out = load_split("train", data_dir=tmp_path)
    assert list(out.columns) == ["text", "label", "intent"]
    assert list(out["intent"]) == ["card_arrival", "top_up_failed"]


def test_load_split_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_split("test", data_dir=tmp_path)


def test_contrastive_pairs_match_and_mismatch_classes():
    from fopstudy.data import contrastive_pairs

    labels = np.array([0, 0, 1, 1, 1, 2])
    pairs = contrastive_pairs(labels, iterations=3, seed=1)
    pos, neg = pairs[pairs[:, 2] == 1], pairs[pairs[:, 2] == 0]
    assert len(neg) == 3 * len(labels)
    assert len(pos) == 3 * 5  # the single example of class 2 has no partner
    assert (labels[pos[:, 0]] == labels[pos[:, 1]]).all() and (pos[:, 0] != pos[:, 1]).all()
    assert (labels[neg[:, 0]] != labels[neg[:, 1]]).all()
    assert (contrastive_pairs(labels, 3, seed=1) == pairs).all()
