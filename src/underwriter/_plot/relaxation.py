"""Relaxation plots: the per-disease marginal-impact ranking (horizontal bars)
and one rule's per-coverage before/after (a dumbbell)."""

from __future__ import annotations

import polars as pl

from .theme import BAR_BLUE, BASELINE_GREY, import_pyplot, style_axes


def plot_rule_impact(
    impact: pl.DataFrame,
    *,
    coverage: str | None = None,
    top: int = 12,
    fill: str = BAR_BLUE,
    title: str | None = None,
    ax=None,
):
    """Horizontal bar ranking of the diseases to relax, from ``list_rule_impact``.
    Pass ``coverage`` to pick one when the table is per-coverage."""
    plt = import_pyplot()
    data = impact
    if "coverage" in data.columns and coverage is not None:
        data = data.filter(pl.col("coverage") == coverage)
    if "coverage" in data.columns and data["coverage"].n_unique() > 1:
        raise ValueError('per-coverage ranking: pass coverage="<name>" to pick one.')
    if data.height == 0:
        raise ValueError("no rows to plot.")

    data = data.sort("auto_lift", descending=True).head(top).sort("auto_lift")
    labels = data["kcd_main"].to_list()
    lift = [v * 100 for v in data["auto_lift"].to_list()]
    n_id = data["n_id"].to_list()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.35)))
    ax.barh(range(len(labels)), lift, color=fill)
    for i, (v, n) in enumerate(zip(lift, n_id)):
        ax.text(v, i, f" {v:.2f}%p ({n:,})", va="center", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{coverage or 'overall'} automation-rate lift (%p)")
    ax.set_title(title or "Diseases to relax for the biggest automation-rate gain")
    ax.margins(x=0.2)
    style_axes(ax)
    return ax.figure


def plot_relaxed_rule(
    relaxed: pl.DataFrame,
    *,
    disease: str | None = None,
    title: str | None = None,
    ax=None,
):
    """Dumbbell of the automation rate per coverage before vs after relaxing,
    from ``relax_rule`` (only the coverages the relaxation moves)."""
    plt = import_pyplot()
    data = relaxed.filter(pl.col("auto_relaxed") != pl.col("auto_base")).sort("auto_relaxed")
    if data.height == 0:
        raise ValueError("this relaxation moved no coverage; nothing to plot.")

    covs = data["coverage"].to_list()
    base = [v * 100 for v in data["auto_base"].to_list()]
    relx = [v * 100 for v in data["auto_relaxed"].to_list()]
    lift = data["lift"].to_list()
    flipped = data["n_flipped"].to_list()
    y = range(len(covs))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, len(covs) * 0.4)))
    for i, (b, r) in enumerate(zip(base, relx)):
        ax.plot([b, r], [i, i], color="grey", linewidth=1.5, zorder=1)
    ax.scatter(base, list(y), color=BASELINE_GREY, s=40, label="baseline", zorder=2)
    ax.scatter(relx, list(y), color=BAR_BLUE, s=40, label="relaxed", zorder=2)
    for i, (r, lf, n) in enumerate(zip(relx, lift, flipped)):
        ax.text(r, i, f" {lf * 100:+.1f}%p ({n:,})", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(covs)
    ax.set_xlabel("automation rate (percent)")
    ax.set_title(title or (f"Relaxing {disease}: automation rate by coverage" if disease
                           else "Automation rate by coverage: before vs after relaxing"))
    ax.legend(frameon=False, loc="lower right")
    ax.margins(x=0.15)
    style_axes(ax)
    return ax.figure
