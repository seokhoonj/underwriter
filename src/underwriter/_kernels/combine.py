"""Combine per-disease decisions into one decision per (id, coverage), then wide
to one row per insured. Table-driven and letter-agnostic (see :class:`Grammar`).

Within-class combiners (priority / exclusion / loading / reduction) merge the
codes of one class; between-class composition by role (standard / decline /
underwriter) then builds the final cell. Fail-safe: an unresolvable or unmatched
code refers its coverage to the underwriter; an expired restriction relaxes to
standard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

from .tokens import Grammar, months_str, resolve_months, unresolved_reason

_CELL_SCHEMA = {"id": pl.String, "coverage": pl.String, "dec": pl.String}


def _empty_cells() -> pl.DataFrame:
    return pl.DataFrame(schema=_CELL_SCHEMA)


@dataclass(frozen=True)
class DecisionResult:
    combined: pl.DataFrame  # one row per insured
    cells: pl.DataFrame     # long (id, coverage, dec) audit trail
    unresolved: pl.DataFrame | None


def combine_decisions(
    applied: pl.DataFrame,
    decision_columns: list[str],
    grammar: Grammar,
    exclusion: pl.DataFrame,
    reduction: pl.DataFrame,
    loading_bands: pl.DataFrame,
) -> DecisionResult:
    exclusion_marks = exclusion["mark"].to_list()
    reduction_marks = reduction["mark"].to_list()
    carry = [c for c in ("id", "elp_day", "kcd_main", "no") if c in applied.columns]

    # unmatched inputs -> underwriter on every coverage, then melt to long
    unmatched_fill = [
        pl.when(pl.col("matched") == 0)
        .then(pl.lit(grammar.underwriter))
        .otherwise(pl.col(c))
        .alias(c)
        for c in decision_columns
    ]
    melted = (
        applied.with_columns(unmatched_fill)
        .unpivot(index=carry, on=decision_columns, variable_name="coverage", value_name="code")
        .filter(pl.col("code").is_not_null() & (pl.col("code") != ""))
        .with_columns(
            method=pl.col("code").str.slice(0, 1).replace_strict(
                grammar.combiner, default="priority"
            )
        )
    )
    all_cells = melted.select("id", "coverage").unique()

    # judge the whole vocabulary once; unresolvable cells go to the underwriter
    bad = {
        c: unresolved_reason(c, grammar, set(exclusion_marks), set(reduction_marks))
        for c in melted["code"].unique().to_list()
    }
    bad = {c: r for c, r in bad.items() if r}
    if bad:
        unresolved_cells = (
            melted.filter(pl.col("code").is_in(list(bad)))
            .select("id", "coverage")
            .unique()
            .with_columns(dec=pl.lit(grammar.underwriter))
        )
        resolved = melted.filter(~pl.col("code").is_in(list(bad)))
        report = (
            melted.filter(pl.col("code").is_in(list(bad)))
            .group_by("code")
            .agg(n_cell=pl.len(), n_id=pl.col("id").n_unique())
            .with_columns(reason=pl.col("code").replace_strict(bad, default=None))
            .sort("n_cell", descending=True)
        )
    else:
        unresolved_cells = _empty_cells()
        resolved = melted
        report = None

    parts = [
        _combine_priority(resolved.filter(pl.col("method") == "priority"), grammar),
        _combine_exclusion(resolved.filter(pl.col("method") == "exclusion"), exclusion_marks, grammar),
        _combine_loading(resolved.filter(pl.col("method") == "loading"), loading_bands, grammar),
        _combine_reduction(resolved.filter(pl.col("method") == "reduction"), reduction_marks, grammar),
        unresolved_cells,
    ]
    results = pl.concat([p for p in parts if p.height], how="vertical") if any(
        p.height for p in parts
    ) else _empty_cells()

    cells = _compose(results, grammar, all_cells)
    combined = cells.pivot(on="coverage", index="id", values="dec")
    if combined.height != applied["id"].n_unique():
        raise AssertionError("combine_decisions produced != one row per insured.")
    return DecisionResult(combined=combined, cells=cells, unresolved=report)


def _combine_priority(rows: pl.DataFrame, grammar: Grammar) -> pl.DataFrame:
    if rows.height == 0:
        return _empty_cells()
    worst = max(grammar.priority.values()) + 1
    return (
        rows.with_columns(
            _rank=pl.col("code").str.slice(0, 1).replace_strict(grammar.priority, default=worst)
        )
        .sort(["id", "coverage", "_rank"])
        .unique(subset=["id", "coverage"], keep="first")
        .select("id", "coverage", dec=pl.col("code"))
    )


def _combine_exclusion(
    rows: pl.DataFrame, marks: list[str], grammar: Grammar
) -> pl.DataFrame:
    if rows.height == 0:
        return _empty_cells()
    letter = re.escape(grammar.exclusion)
    sites = (
        rows.with_columns(token=pl.col("code").str.split(","))
        .explode("token", empty_as_null=True)  # split yields >=1 token; pin the future default
        .with_columns(
            site=pl.col("token").str.extract(rf"^{letter}([0-9]+)\(", 1),
            mark=pl.col("token").str.extract(rf"^{letter}[0-9]+\((.*)\)$", 1),
        )
        .with_columns(months=resolve_months(pl.col("mark"), pl.col("elp_day"), marks))
        .filter(pl.col("months").is_not_null() & (pl.col("months") > 0))
    )
    if sites.height == 0:
        return _empty_cells()
    per_site = (
        sites.group_by("id", "coverage", "site")
        .agg(months=pl.col("months").max())
        .sort(["id", "coverage", "site"])
        .with_columns(
            _tok=pl.concat_str(
                pl.lit(grammar.exclusion), pl.col("site"), pl.lit("("),
                months_str(pl.col("months")), pl.lit(")"),
            )
        )
    )
    built = per_site.group_by("id", "coverage", maintain_order=True).agg(
        n_site=pl.len(), dec=pl.col("_tok").str.join(",")
    )
    return built.with_columns(
        dec=pl.when(pl.col("n_site") > grammar.max_sites)
        .then(pl.lit(grammar.decline))
        .otherwise(pl.col("dec"))
    ).select("id", "coverage", "dec")


def _combine_loading(
    rows: pl.DataFrame, bands: pl.DataFrame, grammar: Grammar
) -> pl.DataFrame:
    if rows.height == 0:
        return _empty_cells()
    letter = re.escape(grammar.loading)
    totals = (
        rows.with_columns(
            _index=pl.col("code").str.extract(rf"^{letter}\(([0-9]+)\)", 1).cast(pl.Int64).fill_null(0)
        )
        .group_by("id", "coverage")
        .agg(total=pl.col("_index").sum())
        .sort("total")
    )
    banded = totals.join_asof(
        bands.sort("at_least"), left_on="total", right_on="at_least", strategy="backward"
    )
    return banded.select(
        "id", "coverage",
        dec=pl.when(pl.col("decision") == grammar.loading)
        .then(pl.concat_str(pl.lit(grammar.loading), pl.lit("("), pl.col("total").cast(pl.Utf8), pl.lit(")")))
        .otherwise(pl.col("decision")),
    )


def _combine_reduction(
    rows: pl.DataFrame, marks: list[str], grammar: Grammar
) -> pl.DataFrame:
    if rows.height == 0:
        return _empty_cells()
    letter = re.escape(grammar.reduction)
    kept = (
        rows.with_columns(mark=pl.col("code").str.extract(rf"^{letter}\((.*)\)$", 1))
        .with_columns(months=resolve_months(pl.col("mark"), pl.col("elp_day"), marks))
        .filter(pl.col("months").is_not_null() & (pl.col("months") > 0))
    )
    if kept.height == 0:
        return _empty_cells()
    return kept.group_by("id", "coverage").agg(
        dec=pl.concat_str(
            pl.lit(grammar.reduction), pl.lit("("), months_str(pl.col("months").max()), pl.lit(")")
        )
    ).select("id", "coverage", "dec")


def _compose(results: pl.DataFrame, grammar: Grammar, all_cells: pl.DataFrame) -> pl.DataFrame:
    if results.height == 0:
        return all_cells.with_columns(dec=pl.lit(grammar.standard))
    worst = max(grammar.priority.values()) + 1
    last = len(grammar.sheet_ord) + 1
    results = results.with_columns(_cls=pl.col("dec").str.slice(0, 1)).with_columns(
        _rank=pl.col("_cls").replace_strict(grammar.priority, default=worst),
        _has_terminal=pl.col("_cls").is_in(grammar.terminal).any().over(["id", "coverage"]),
    )
    alone = (
        results.filter(pl.col("_cls").is_in(grammar.terminal))
        .sort(["id", "coverage", "_rank"])
        .unique(subset=["id", "coverage"], keep="first")
        .select("id", "coverage", "dec")
    )
    rest = results.filter((~pl.col("_has_terminal")) & (pl.col("_cls") != grammar.standard))
    composed = (
        rest.with_columns(_pos=pl.col("_cls").replace_strict(grammar.sheet_ord, default=last))
        .sort(["id", "coverage", "_pos"])
        .group_by("id", "coverage", maintain_order=True)
        .agg(dec=pl.col("dec").str.join(","))
    )
    final = pl.concat([alone, composed], how="vertical") if (alone.height or composed.height) else _empty_cells()
    return (
        all_cells.join(final, on=["id", "coverage"], how="left")
        .with_columns(dec=pl.col("dec").fill_null(grammar.standard))
    )
