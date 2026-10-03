"""Draw histograms as outlines instead of filled bars, so they can be stacked.

A filled histogram hides whatever is behind it, so you can only ever look at one
particle at a time. That is why the old scripts wrote one PNG per particle and
you had to flick between them to compare.

Here we bin the data the same way, then draw only the line along the tops of the
bars. Nothing is filled in, so several particles can go on the same axes and be
read at once.

Everything below is plain matplotlib, no new dependencies.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a window
import matplotlib.pyplot as plt

# Colours, in the order curves are assigned them. Checked so that no two are
# confusable, including for the common forms of colour blindness. Do not
# reorder or extend past four without re-checking -- yellow next to orange in
# particular is not safe.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

# A second, redundant way to tell the curves apart, for print and for anyone who
# cannot rely on the colours at all.
LINESTYLES = ["-", "--", "-.", (0, (1, 1.4))]

# Chart ink. Kept off the series colours on purpose: text stays neutral so the
# only coloured things on the figure are the curves themselves.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

LINEWIDTH = 1.8

# The ratio panel is only drawn where the baseline bin holds at least this many
# events. A ratio of 2 events to 1 is pure noise, and drawing it makes the tails
# of every distribution look broken when they are just thinly populated.
MIN_RATIO_COUNTS = 10


def staircase(edges, counts, close=True):
    """Turn bin edges and bin counts into the x, y of the outline.

    matplotlib can draw this itself with histtype="step", but we bin separately
    because the ratio panel needs the raw counts anyway, and doing it once here
    keeps the two panels guaranteed to agree.

    close=True drops the outline down to zero at the first and last edge, so it
    shuts against the baseline instead of hanging in mid air. That is right for
    a count, but wrong for a ratio -- a ratio has no reason to be zero at the
    edge of the range, and drawing it that way puts a fake cliff on both ends.
    """
    if close:
        x = np.repeat(edges, 2)
        y = np.concatenate(([0.0], np.repeat(counts, 2), [0.0]))
    else:
        x = np.repeat(edges, 2)[1:-1]
        y = np.repeat(counts, 2)
    return x, y


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def outline_figure(
    series,
    bins,
    value_range,
    xlabel,
    title,
    filename,
    subtitle=None,
    ratio_panel=True,
    log_y=False,
    print_table=True,
    markers=None,
):
    """One figure holding an outline per series, plus an optional ratio panel.

    series       list of (label, values) -- one entry per curve, at most 4
    bins         number of bins
    value_range  (low, high) for the x axis
    ratio_panel  draw each later curve divided by the first one, underneath
    markers      optional list of x positions to mark with a dashed vertical
                 line, e.g. the edges of a detector acceptance

    The ratio panel is the "is particle 4 shaped like particle 3" question. Each
    curve keeps its own colour down there, so a line means the same particle in
    both panels.

    Labels and titles may contain matplotlib mathtext, e.g. "$e^{+}$".
    """
    if len(series) > len(PALETTE):
        raise ValueError(
            f"{len(series)} curves asked for but only {len(PALETTE)} distinguishable "
            "colours are defined -- split this into two figures instead"
        )

    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    counted = [(label, np.histogram(values, bins=edges)[0]) for label, values in series]

    show_ratio = ratio_panel and len(counted) > 1
    if show_ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(9, 6.8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax_ratio = None

    fig.patch.set_facecolor(SURFACE)

    # Main panel: the outlines.
    for i, (label, counts) in enumerate(counted):
        x, y = staircase(edges, counts)
        ax.plot(
            x, y,
            color=PALETTE[i], linestyle=LINESTYLES[i], linewidth=LINEWIDTH,
            label=label, zorder=3 + i, solid_joinstyle="miter",
        )

    _style_axes(ax)
    ax.set_xlim(*value_range)
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel("Events per bin", color=INK_SECONDARY, fontsize=10)

    # With two or more curves a legend is always present, so a curve is never
    # identified by colour alone. A single curve needs no legend box -- the
    # title already says what it is.
    if len(counted) > 1:
        legend = ax.legend(frameon=False, fontsize=9.5, loc="best")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    for x in markers or ():
        for axis in (ax, ax_ratio):
            if axis is not None:
                axis.axvline(x, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)

    # The subtitle sits in the gap the title pad opens up, so the two never
    # collide however long the title is.
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=24 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes,
            color=INK_MUTED, fontsize=9.5, va="bottom", ha="left",
        )

    # Ratio panel: everything relative to the first curve.
    if show_ratio:
        base_label, base = counted[0]
        # Thinly populated bins give a gap rather than a meaningless spike.
        base_safe = np.where(base >= MIN_RATIO_COUNTS, base, np.nan)
        for i, (label, counts) in enumerate(counted[1:], start=1):
            ratio = counts / base_safe
            x, y = staircase(edges, ratio, close=False)
            ax_ratio.plot(
                x, y,
                color=PALETTE[i], linestyle=LINESTYLES[i], linewidth=LINEWIDTH,
                zorder=3 + i, solid_joinstyle="miter",
            )
        ax_ratio.axhline(1.0, color=AXIS, linewidth=1.0, zorder=2)

        _style_axes(ax_ratio)
        ax_ratio.set_ylim(0, 2)
        # Name the baseline by its own legend label, minus the particle symbol,
        # so the ratio axis reads "ratio to Particle 3" or "ratio to Pair 34".
        base_short = base_label.split(":")[0].split(" (")[0]
        ax_ratio.set_ylabel(f"ratio to\n{base_short}", color=INK_SECONDARY, fontsize=9)
        ax_ratio.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    else:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)

    fig.savefig(filename, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {filename}")

    if print_table:
        _print_summary(series)


def plain(text):
    """Strip matplotlib mathtext down to plain ASCII, for printing.

    Windows terminals do not always handle Greek letters, so the console tables
    spell them out instead: "$e^{+}$" becomes "e+", "$\\gamma$" becomes "gamma".
    """
    for old, new in (
        (r"\rightarrow", "->"), (r"\gamma", "gamma"), (r"\mu", "mu"),
        (r"\bar{n}", "nbar"), (r"\,", " "), ("^", ""),
        ("$", ""), ("{", ""), ("}", ""), ("\\", ""),
    ):
        text = text.replace(old, new)
    return text


def _print_summary(series):
    """Print the same numbers as a small table.

    The figure is the quick read; this is so the exact values are available
    without opening the PNG, and so the content is not locked inside an image.
    """
    print(f"    {'curve':<28}{'n':>8}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}")
    for label, values in series:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        print(f"    {plain(label):<28}{values.size:>8}{values.mean():>12.4f}"
              f"{values.std():>12.4f}{values.min():>12.4f}{values.max():>12.4f}")
