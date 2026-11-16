"""Chart helpers shared by the notebooks, all in the study style."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fopstudy import style


def learning_curve(summaries, ax=None, label_last=True):
    """Accuracy against examples per intent, one line per approach.

    `summaries` is a list of frames from evaluate.learning_curve_summary. Each
    approach gets its fixed colour, a band for the test-set interval, small dots
    for the individual seeds, and a direct label at its right end.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.6))
    else:
        fig = ax.figure
    for s in summaries:
        approach = s["approach"].iloc[0]
        colour = style.APPROACH_COLORS[approach]
        ax.fill_between(s["x"], s["low"], s["high"], color=colour, alpha=0.18, linewidth=0)
        ax.plot(s["x"], s["accuracy"], color=colour, linewidth=2.5, marker="o", markersize=6)
        for x, accs in zip(s["x"], s["seed_accuracies"]):
            ax.scatter(np.full(len(accs), x), accs, color=colour, s=12, alpha=0.7, zorder=3)
        if label_last:
            last = s.iloc[-1]
            ax.annotate(f"{style.APPROACH_LABELS[approach]}  {last['accuracy']:.1%}",
                        (last["x"], last["accuracy"]), xytext=(10, 0), textcoords="offset points",
                        va="center", color=colour, fontsize=11, fontweight="bold")
    xs = sorted({x for s in summaries for x in s["x"]})
    names = {x: (f"all\n({x:.0f})" if x != round(x) else f"{x:.0f}") for x in xs}
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([names[x] for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training examples per intent")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    return fig, ax
