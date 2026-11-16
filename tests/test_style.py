import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from fopstudy import style  # noqa: E402


def test_every_approach_has_its_own_colour_and_label():
    assert set(style.APPROACH_ORDER) == set(style.APPROACH_COLORS) == set(style.APPROACH_LABELS)
    colours = list(style.APPROACH_COLORS.values()) + [style.HIGHLIGHT]
    assert len(set(colours)) == len(colours)


def test_headline_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "FIGURE_DIR", tmp_path)
    style.use_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    style.headline(fig, "A finding", "How it was measured")
    style.save(fig, "check")
    plt.close(fig)
    assert (tmp_path / "check.png").stat().st_size > 0


def test_learning_curve_plot_runs(tmp_path):
    import pandas as pd

    from fopstudy import plots

    summary = pd.DataFrame({"approach": "frozen", "n_per_class": [5, "all"], "x": [5.0, 117.0],
                            "accuracy": [0.8, 0.9], "low": [0.78, 0.89], "high": [0.82, 0.91],
                            "seed_accuracies": [[0.79, 0.81], [0.9]]})
    fig, ax = plots.learning_curve([summary])
    assert ax.get_xscale() == "log"
    plt.close(fig)
