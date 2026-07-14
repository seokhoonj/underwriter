"""Relaxation what-if analysis: how much would each coverage's auto-decided share
rise if a disease's rule were relaxed (its underwriter referrals turned to the
standard code, keeping every exclusion / loading / reduction / decline)?

- ``relax_rule`` -- the joint lift of relaxing one or more diseases, per coverage
  (an exact re-combine).
- ``list_rule_impact`` -- every disease's marginal (independent) lift per coverage,
  the shortlist of what to relax, computed without a per-candidate re-combine.
- ``decompose_rule_impact`` -- split a joint relaxation into marginal / joint /
  synergy.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..engine.decision import Decision
from ..engine.match import MatchResult
from ..engine.underwriter import Underwriter


def _as_list(x: str | Sequence[str] | None) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def _auto_share(decision: Decision) -> pl.DataFrame:
    tab = decision.tabulate()
    return tab.group_by("coverage").agg(
        auto=pl.col("n").filter(pl.col("auto") == 1).sum() / pl.col("n").sum(),
        n_total=pl.col("n").sum(),
    )


def relax_rule(
    engine: Underwriter,
    match_result: MatchResult,
    kcd_main: str | Sequence[str],
    *,
    coverage: str | Sequence[str] | None = None,
    baseline: Decision | None = None,
) -> pl.DataFrame:
    """Relax the rule(s) for ``kcd_main`` (a regex, or a list of codes OR'd) and
    measure the per-coverage auto-share lift. Only referrals are relaxed, so the
    auto share can only rise. Returns one row per coverage sorted by ``lift``;
    draw it with ``plot_relaxed_rule``."""
    patterns = _as_list(kcd_main)
    if not patterns or any(not p for p in patterns):
        raise ValueError("kcd_main must be one or more non-empty patterns.")
    target = "|".join(patterns)
    grammar = engine.grammar
    standard, underwriter = grammar.standard, grammar.underwriter
    decision_columns = list(match_result.decision_columns)
    applied = match_result.applied
    if baseline is None:
        baseline = engine.decide(match_result)

    base = _auto_share(baseline).rename({"auto": "auto_base"})
    is_target = pl.col("kcd_main").str.contains(target)
    relaxed = applied.with_columns(
        [
            pl.when(is_target & ((pl.col("matched") == 0) | (pl.col(c) == underwriter)))
            .then(pl.lit(standard))
            .otherwise(pl.col(c))
            .alias(c)
            for c in decision_columns
        ]
    ).with_columns(
        matched=pl.when(is_target).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.col("matched"))
    )
    relaxed_share = _auto_share(engine.recombine(relaxed)).select(
        "coverage", auto_relaxed=pl.col("auto")
    )

    out = (
        base.join(relaxed_share, on="coverage")
        .with_columns(
            lift=pl.col("auto_relaxed") - pl.col("auto_base"),
            n_flipped=((pl.col("auto_relaxed") - pl.col("auto_base")) * pl.col("n_total"))
            .round()
            .cast(pl.Int64),
        )
        .select("coverage", "auto_base", "auto_relaxed", "lift", "n_flipped")
        .sort("lift", descending=True)
    )
    cov = _as_list(coverage)
    return out.filter(pl.col("coverage").is_in(cov)) if cov else out


def list_rule_impact(
    engine: Underwriter,
    match_result: MatchResult,
    *,
    coverage: str | Sequence[str] | None = None,
    baseline: Decision | None = None,
) -> pl.DataFrame:
    """Every disease's marginal auto-lift per coverage: the cells that would flip
    to auto if that disease's rule were relaxed on its own. A cell counts only when
    that disease is its sole source of the referral, so the marginal is exact
    without a per-candidate re-combine. Draw it with ``plot_rule_impact``."""
    grammar = engine.grammar
    underwriter = grammar.underwriter
    decision_columns = list(match_result.decision_columns)
    applied = match_result.applied
    if baseline is None:
        baseline = engine.decide(match_result)

    combined_long = (
        baseline.combined.unpivot(
            index="id", on=decision_columns, variable_name="coverage", value_name="decision"
        )
        .filter(pl.col("decision").is_not_null() & (pl.col("decision") != ""))
    )
    cov = _as_list(coverage)
    if cov:
        combined_long = combined_long.filter(pl.col("coverage").is_in(cov))
    referred_cells = combined_long.filter(pl.col("decision") == underwriter).select("id", "coverage")

    matched_referred = (
        applied.filter(pl.col("matched") == 1)
        .unpivot(index=["id", "kcd_main"], on=decision_columns, variable_name="coverage", value_name="code")
        .filter(pl.col("code") == underwriter)
        .select("id", "kcd_main", "coverage")
    )
    unmatched_referred = applied.filter(pl.col("matched") == 0).select("id", "kcd_main").join(
        pl.DataFrame({"coverage": decision_columns}), how="cross"
    )
    referred_src = (
        pl.concat([matched_referred, unmatched_referred])
        .join(referred_cells, on=["id", "coverage"], how="inner")
        .with_columns(n_causes=pl.col("kcd_main").n_unique().over(["id", "coverage"]))
    )
    sole = referred_src.filter(pl.col("n_causes") == 1)
    cov_counts = combined_long.group_by("coverage").agg(n_cov=pl.len())
    return (
        sole.group_by("kcd_main", "coverage")
        .agg(n_id=pl.col("id").n_unique(), n_flipped=pl.len())
        .join(cov_counts, on="coverage")
        .with_columns(auto_lift=pl.col("n_flipped") / pl.col("n_cov"))
        .select("coverage", "kcd_main", "n_id", "n_flipped", "auto_lift")
        .sort(["coverage", "n_flipped"], descending=[False, True])
    )


def decompose_rule_impact(
    engine: Underwriter,
    match_result: MatchResult,
    kcd_main: Sequence[str],
    *,
    coverage: str | Sequence[str] | None = None,
    baseline: Decision | None = None,
) -> pl.DataFrame:
    """Split relaxing a set of codes together into ``individual`` (sum of each
    code's marginal), ``joint`` (all at once), and ``synergy`` (joint - individual,
    the co-held cells the marginals miss), per coverage."""
    targets = list(dict.fromkeys(kcd_main))
    if len(targets) < 2:
        raise ValueError("kcd_main must name at least two representative codes.")
    if baseline is None:
        baseline = engine.decide(match_result)
    cov = _as_list(coverage)
    cov_cols = [c for c in match_result.decision_columns if cov is None or c in cov]

    def flips(regex: str) -> pl.DataFrame:
        return relax_rule(engine, match_result, regex, coverage=cov_cols, baseline=baseline).select(
            "coverage", "n_flipped"
        )

    joint = flips("^(" + "|".join(targets) + ")$").rename({"n_flipped": "joint"})
    marginal = pl.concat([flips(f"^{c}$") for c in targets]).group_by("coverage").agg(
        individual=pl.col("n_flipped").sum()
    )
    cov_counts = (
        baseline.combined.unpivot(index="id", on=cov_cols, variable_name="coverage", value_name="decision")
        .filter(pl.col("decision").is_not_null() & (pl.col("decision") != ""))
        .group_by("coverage")
        .agg(n_cov=pl.len())
    )
    merged = (
        joint.join(marginal, on="coverage", how="full", coalesce=True)
        .with_columns(pl.col("joint").fill_null(0), pl.col("individual").fill_null(0))
        .with_columns(synergy=pl.col("joint") - pl.col("individual"))
        .join(cov_counts, on="coverage", how="left")
    )
    return (
        merged.unpivot(
            index=["coverage", "n_cov"],
            on=["individual", "joint", "synergy"],
            variable_name="component",
            value_name="n_flipped",
        )
        .with_columns(auto_lift=pl.col("n_flipped") / pl.col("n_cov"))
        .select("coverage", "component", "n_flipped", "auto_lift")
        .sort(["coverage", "component"])
    )
