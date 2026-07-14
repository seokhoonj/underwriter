"""Decision composition per coverage -- a stacked bar of each coverage's
auto-decided vs underwriter-referred share (or the category breakdown)."""

from __future__ import annotations

import polars as pl

from .theme import AUTO_BLUE, REFER_RED, import_pyplot, style_axes


def plot_decision(
    tabulation: pl.DataFrame,
    *,
    group: str = "auto",
    order: str = "auto_high",
    min_label: float = 0.03,
    title: str = "Decision composition per coverage",
    ax=None,
):
    """Stacked bar per coverage from a :meth:`Decision.tabulate` result. ``group``
    is ``"auto"`` (default; auto vs referred) or ``"category"``. ``order`` sorts
    coverages by ``"auto_high"`` (default), ``"auto_low"``, or ``"column"``."""
    plt = import_pyplot()
    tab = tabulation

    auto_share = (
        tab.group_by("coverage")
        .agg(share=pl.col("prop").filter(pl.col("auto") == 1).sum())
        .sort("share")
    )
    covs = auto_share["coverage"].to_list()
    if order == "auto_high":
        covs = covs[::-1]
    elif order == "column":
        covs = sorted(covs)

    data = tab.group_by(["coverage", group]).agg(prop=pl.col("prop").sum())
    levels = sorted(data[group].unique().to_list(), key=lambda v: str(v))
    if group == "auto":
        levels = [v for v in (1, 0) if v in levels]  # auto at bottom, referred on top
    colours = {1: AUTO_BLUE, 0: REFER_RED} if group == "auto" else None

    wide = {
        lv: dict(zip(*data.filter(pl.col(group) == lv).select("coverage", "prop").to_dict(as_series=False).values()))
        for lv in levels
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(covs) * 0.4), 4.5))
    bottoms = [0.0] * len(covs)
    for lv in levels:
        heights = [wide[lv].get(c, 0.0) for c in covs]
        colour = colours.get(lv) if colours else None
        ax.bar(range(len(covs)), heights, bottom=bottoms, color=colour, label=str(lv), width=0.85)
        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h > min_label:
                ax.text(i, b + h / 2, str(round(h * 100)), ha="center", va="center", fontsize=8)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(range(len(covs)))
    ax.set_xticklabels(covs, rotation=90)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.set_ylabel("percent")
    ax.set_title(title)
    ax.legend(title=group, frameon=False)
    style_axes(ax)
    return ax.figure
