"""The band match -- attach each aggregated input to the rule it falls in.

Equality on ``kcd_main`` plus four closed bands (``age``, ``elp_day``,
``sur_cnt``, ``hos_day``; ``out_day`` / ``out_cnt`` are carried by the rule set
but never joined on). Only ``decl_yn == 0`` rules apply. A null band bound or a
null input value never matches (``is_between`` yields null, which ``filter``
drops -- the same "NA never matches" the R non-equi join gives for free). The
lowest-``ord`` match wins ties; multi-matches and genuine decision conflicts are
reported for rule-set cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_BANDS = (
    ("age", "age_min", "age_max"),
    ("elp_day", "elp_day_min", "elp_day_max"),
    ("sur_cnt", "sur_cnt_min", "sur_cnt_max"),
    ("hos_day", "hos_day_min", "hos_day_max"),
)
_RULE_KEYS = (
    "kcd_main", "no", "ord",
    "age_min", "age_max", "elp_day_min", "elp_day_max",
    "sur_cnt_min", "sur_cnt_max", "hos_day_min", "hos_day_max",
)


@dataclass(frozen=True)
class BandMatch:
    applied: pl.DataFrame
    unmatched: pl.DataFrame
    multi_matched: pl.DataFrame
    conflict: pl.DataFrame
    n_multi_matched: int
    n_conflict: int


def band_match(
    aggregated: pl.DataFrame,
    auto_rules: pl.DataFrame,
    decision_columns: list[str],
) -> BandMatch:
    """``auto_rules`` must already be filtered to ``decl_yn == 0``."""
    input_columns = aggregated.columns
    inputs = aggregated.with_row_index("_row")
    rules = auto_rules.select([*_RULE_KEYS, *decision_columns])

    condition = pl.lit(True)
    for value, low, high in _BANDS:
        condition = condition & pl.col(value).is_between(pl.col(low), pl.col(high))
    hits = inputs.join(rules, on="kcd_main", how="inner").filter(condition)

    per_input = hits.group_by("_row").len()
    multi_rows = per_input.filter(pl.col("len") > 1).select("_row")

    # lowest-ord match wins ties (near-all ties are identical duplicates)
    winner = (
        hits.sort(["_row", "ord"])
        .unique(subset="_row", keep="first")
        .select("_row", "no", "ord", *decision_columns)
    )
    applied = (
        inputs.join(winner, on="_row", how="left")
        .with_columns(matched=pl.col("no").is_not_null().cast(pl.Int8))
    )

    # a multi-matched input conflicts when its matched rules disagree on a decision
    conflict_rows = (
        hits.join(multi_rows, on="_row", how="inner")
        .group_by("_row")
        .agg(pl.struct(decision_columns).n_unique().alias("_n_distinct"))
        .filter(pl.col("_n_distinct") > 1)
        .select("_row")
    )
    applied = (
        applied.join(conflict_rows.with_columns(_conflict=True), on="_row", how="left")
        .with_columns(conflict=pl.col("_conflict").fill_null(False))
        .drop("_conflict", "_row")
    )

    unmatched = applied.filter(pl.col("matched") == 0).select(input_columns)
    multi_matched = (
        hits.join(multi_rows, on="_row", how="inner")
        .select("_row", "no", "ord", *decision_columns)
        .sort(["_row", "ord"])
    )
    conflict = multi_matched.join(conflict_rows, on="_row", how="inner")
    return BandMatch(
        applied=applied,
        unmatched=unmatched,
        multi_matched=multi_matched.drop("_row"),
        conflict=conflict.drop("_row"),
        n_multi_matched=multi_rows.height,
        n_conflict=conflict_rows.height,
    )
