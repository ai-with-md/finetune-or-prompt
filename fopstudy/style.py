"""Shared visual style so every chart in the study looks like one family.

Same visual language as my experimentation toolkit: a dark surface, one colour
per approach that never changes, an acid lime highlight reserved for the point
the reader should look at (a break-even line, a crossing), and titles that state
the finding instead of naming the chart.
"""

from pathlib import Path

import matplotlib as mpl

SURFACE = "#111116"
INK = "#F4F1EA"
INK_2 = "#B9B5C2"
MUTED = "#85818F"
GRID = "#24232B"
BASELINE = "#3A3944"
HIGHLIGHT = "#D4FF3F"

# One fixed colour per approach. The no-fine-tuning baseline is a quiet grey on purpose.
APPROACH_COLORS = {
    "frozen": "#55535F",
    "finetuned": "#3B8FE0",
    "haiku": "#C4831C",
    "sonnet": "#7B68FF",
}
APPROACH_LABELS = {
    "frozen": "Small model, not fine-tuned",
    "finetuned": "Small model, fine-tuned",
    "haiku": "Claude Haiku 5.5, prompted",
    "sonnet": "Claude Sonnet 5.5, prompted",
}
APPROACH_ORDER = ["frozen", "finetuned", "haiku", "sonnet"]
PROMPTED = {"haiku", "sonnet"}  # examples go in the prompt, not into training

FONT_STACK = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]
FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def use_style() -> None:
    """Apply the study style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "ytick.major.size": 0,
        "xtick.major.size": 0,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.solid_capstyle": "round",
        "text.parse_math": False,  # so dollar signs in labels print as dollars
    })


def headline(fig, title: str, subtitle: str = "") -> None:
    """Big finding-first title with a quieter subtitle, left aligned."""
    from matplotlib.transforms import offset_copy

    fig.text(0.02, 0.98, title, ha="left", va="top", fontsize=17,
             fontweight="bold", color=INK)
    if subtitle:
        below = offset_copy(fig.transFigure, fig=fig, y=-30, units="points")
        fig.text(0.02, 0.98, subtitle, ha="left", va="top", fontsize=11,
                 color=MUTED, transform=below)


def save(fig, name: str) -> None:
    """Save a figure to docs/figures for use in the README."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.3)
