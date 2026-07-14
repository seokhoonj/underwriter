"""``count_stay`` -- distinct hospital calendar days per group.

Across possibly overlapping admissions, count each calendar day once (a
dedup-union of day ranges), with the discharge date capped at the inquiry date
so days after the inquiry are never counted. This is the reference
implementation: explode each stay to its days and count the distinct set. It is
correct and parity-simple; an interval-sweep variant can replace it later,
pinned equal by a property test.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def count_stay(
    hospital_rows: pl.DataFrame,
    *,
    group: Sequence[str] = ("id", "kcd_main"),
    start: str = "sdate",
    end: str = "edate",
    cap: str = "inq_date",
    out: str = "hos_day",
) -> pl.DataFrame:
    """Count distinct in-hospital calendar days per ``group``.

    ``hospital_rows`` should already be the inpatient rows (``hos_day > 0``).
    Returns one row per ``group`` with an ``out`` (default ``hos_day``) Int32
    column. A stay whose start is after the capped end contributes nothing.
    """
    group = list(group)
    capped_end = pl.min_horizontal(pl.col(end), pl.col(cap))
    return (
        hospital_rows.with_columns(_capped_end=capped_end)
        .filter(pl.col(start) <= pl.col("_capped_end"))
        .with_columns(_stay_day=pl.date_ranges(pl.col(start), pl.col("_capped_end")))
        # ranges are non-empty (start <= capped_end filtered above); pin the future default
        .explode("_stay_day", empty_as_null=True)
        .group_by(group)
        .agg(pl.col("_stay_day").n_unique().cast(pl.Int32).alias(out))
    )
