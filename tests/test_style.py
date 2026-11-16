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
