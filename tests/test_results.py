import pytest

from fopstudy.results import available_runs, load_predictions, prediction_path, save_predictions


def test_round_trip(tmp_path):
    save_predictions("frozen", 5, 1, [3, 4, 5], [3, 0, 5], results_dir=tmp_path)
    df = load_predictions("frozen", 5, 1, results_dir=tmp_path)
    assert list(df["row"]) == [0, 1, 2]
    assert list(df["correct"]) == [True, False, True]


def test_path_layout(tmp_path):
    path = prediction_path("finetuned", "all", 0, results_dir=tmp_path)
    assert path == tmp_path / "predictions" / "finetuned" / "nall_seed0.csv"


def test_available_runs(tmp_path):
    for n, seed in [(5, 0), (5, 1), ("all", 0), (25, 2)]:
        save_predictions("frozen", n, seed, [1], [1], results_dir=tmp_path)
    runs = available_runs("frozen", results_dir=tmp_path)
    assert set(map(tuple, runs.values.tolist())) == {(5, 0), (5, 1), ("all", 0), (25, 2)}
    assert available_runs("missing", results_dir=tmp_path).empty


def test_lengths_must_match(tmp_path):
    with pytest.raises(ValueError):
        save_predictions("frozen", 5, 0, [1, 2], [1], results_dir=tmp_path)
    with pytest.raises(ValueError):
        save_predictions("frozen", 5, 0, [1, 2], [1, 2], rows=[7], results_dir=tmp_path)


def test_subset_rows_are_kept(tmp_path):
    save_predictions("sonnet", 0, 0, [4, 5], [4, -1], rows=[40, 12], results_dir=tmp_path)
    df = load_predictions("sonnet", 0, 0, results_dir=tmp_path)
    assert list(df["row"]) == [40, 12]
    assert list(df["correct"]) == [True, False]
