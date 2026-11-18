"""Chart helpers shared by the notebooks, all in the study style."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fopstudy import style

# Zero examples (a prompt with no examples) cannot sit on a log axis; it is drawn here, labelled 0.
ZERO_X = 0.5


def learning_curve(summaries, ax=None, label_last=True):
    """Accuracy against examples per intent, one line per approach.

    `summaries` is a list of frames from evaluate.learning_curve_summary. Each
    approach gets its fixed colour, a band for the test-set interval, small dots
    for the individual seeds, and a direct label at its right end. Prompted
    models, whose examples go in the prompt rather than into training, are
    drawn dashed; a point at 0 examples is drawn at ZERO_X.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.6))
    else:
        fig = ax.figure
    for s in summaries:
        approach = s["approach"].iloc[0]
        colour = style.APPROACH_COLORS[approach]
        x = s["x"].where(s["x"] > 0, ZERO_X)
        ax.fill_between(x, s["low"], s["high"], color=colour, alpha=0.18, linewidth=0)
        ax.plot(x, s["accuracy"], color=colour, linewidth=2.5, marker="o", markersize=6,
                linestyle=(0, (5, 3)) if approach in style.PROMPTED else "-")
        for pos, accs in zip(x, s["seed_accuracies"]):
            ax.scatter(np.full(len(accs), pos), accs, color=colour, s=12, alpha=0.7, zorder=3)
        if label_last:
            last = s.iloc[-1]
            ax.annotate(f"{style.APPROACH_LABELS[approach]}  {last['accuracy']:.1%}",
                        (x.iloc[-1], last["accuracy"]), xytext=(10, 0), textcoords="offset points",
                        va="center", color=colour, fontsize=11, fontweight="bold")
    xs = sorted({x if x > 0 else ZERO_X for s in summaries for x in s["x"]})
    names = {x: ("0" if x == ZERO_X else f"all\n({x:.0f})" if x != round(x) else f"{x:.0f}")
             for x in xs}
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([names[x] for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training examples per intent")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    return fig, ax


def dot_intervals(rows, ax=None, value_format="{:+.1%}", zero_line=True):
    """One estimate per row, top to bottom: a dot, its 95% interval and its value.

    `rows` has columns label, value, low, high and colour. value_format is a
    format string or a function of the value. With zero_line, a highlighted line
    at 0 shows which differences are clear of no difference.
    """
    fmt = value_format if callable(value_format) else value_format.format
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8))
    else:
        fig = ax.figure
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows.itertuples()):
        ax.plot([r.low, r.high], [yi, yi], color=r.colour, linewidth=4, alpha=0.45)
        ax.scatter([r.value], [yi], color=r.colour, s=70, zorder=3)
        left = zero_line and r.value < 0  # label negatives on the left, clear of the zero line
        ax.annotate(fmt(r.value), (r.low if left else r.high, yi), xytext=(-8 if left else 8, 0),
                    textcoords="offset points", va="center", ha="right" if left else "left",
                    color=style.INK_2, fontsize=10)
    if zero_line:
        ax.axvline(0, color=style.HIGHLIGHT, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    return fig, ax
