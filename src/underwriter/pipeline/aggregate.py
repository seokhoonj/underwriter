"""Stage 4 -- aggregate mapped diagnoses to per-(id, kcd_main) underwriting inputs.

Reviewed rows only. Counts use a fixed 5-year window; elapsed days use each
disease's lookback window (IRREGULAR/UNMAPPED fall back to the 5-year window).
Every id leaves with at least one row: an insured whose every diagnosis aged out
(and has no VACANT line) is carried on an ``EXPIRED`` placeholder.
"""

from __future__ import annotations

import polars as pl

from .._kernels.io import mirror_output, to_polars
from .._kernels.stay import count_stay
from ..sentinels import Sentinel

_VACANT = Sentinel.VACANT.value
_NO_WINDOW = (Sentinel.IRREGULAR.value, Sentinel.UNMAPPED.value)
_EXPIRED = Sentinel.EXPIRED.value

_OUTPUT_COLUMNS = (
    "id", "kcd_main", "age", "hos_day", "sur_cnt", "out_cnt",
    "hos_elp_day", "sur_elp_day", "out_elp_day", "elp_day",
)


def _elapsed() -> pl.Expr:
    """Days since the most recent treatment, clamped at 0 (a treatment dated after
    the inquiry is "current" = 0 days); null when no treatment date at all."""
    tdate = pl.max_horizontal("acc_date", "sdate", "edate")
    days = (pl.col("inq_date") - tdate).dt.total_days()
    return (
        pl.when(days.is_null())
        .then(None)
        .otherwise(pl.max_horizontal(pl.lit(0, dtype=pl.Int64), days))
        .cast(pl.Int32)
    )


def _min_elapsed(rows: pl.DataFrame, out: str) -> pl.DataFrame:
    """Most-recent (minimum) elapsed day per (id, kcd_main). Empty in -> empty out
    with the right schema (never a spurious infinity)."""
    return rows.group_by("id", "kcd_main").agg(pl.col("elapsed").min().alias(out))


def aggregate_disease(mapped: object) -> object:
    frame, was_pandas = to_polars(mapped)
    reviewed = frame.filter(pl.col("review") == 1).with_columns(elapsed=_elapsed())

    # (id, kcd_main) universe: in the disease lookback window, or the 5-year window
    # for the windowless sentinels.
    in_scope = reviewed.filter(
        (pl.col("in_lookback") == 1)
        | (pl.col("kcd_main").is_in(_NO_WINDOW) & (pl.col("in_5yr") == 1))
    )
    # a VACANT line marks the line, not the insured: keep it only for ids with no
    # real diagnosis in scope.
    non_vacant = in_scope.filter(pl.col("kcd_main") != _VACANT)
    underwritable = non_vacant.select("id").unique()
    vacant_kept = in_scope.filter(pl.col("kcd_main") == _VACANT).join(
        underwritable, on="id", how="anti"
    )
    in_scope = pl.concat([non_vacant, vacant_kept])
    id_disease = in_scope.select("id", "kcd_main").unique()

    # per-treatment-type most-recent elapsed days
    hos_elp = _min_elapsed(in_scope.filter(pl.col("hos_day") > 0), "hos_elp_day")
    sur_elp = _min_elapsed(in_scope.filter(pl.col("sur_cnt") > 0), "sur_elp_day")
    out_elp = _min_elapsed(
        in_scope.filter((pl.col("hos_day") == 0) & (pl.col("sur_cnt") == 0)), "out_elp_day"
    )

    # counts over the fixed 5-year window
    within_5yr = reviewed.filter(pl.col("in_5yr") == 1)
    hosp = within_5yr.filter(pl.col("hos_day") > 0)
    hospital_days = (
        count_stay(hosp)
        if hosp.height
        else pl.DataFrame(schema={"id": pl.String, "kcd_main": pl.String, "hos_day": pl.Int32})
    )
    surgery_count = (
        within_5yr.filter(pl.col("sur_cnt") > 0)
        .group_by("id", "kcd_main")
        .agg(pl.col("acc_date").n_unique().cast(pl.Int32).alias("sur_cnt"))
    )
    outpatient_count = (
        within_5yr.filter((pl.col("hos_day") == 0) & (pl.col("sur_cnt") == 0))
        .group_by("id", "kcd_main")
        .agg(pl.col("acc_date").n_unique().cast(pl.Int32).alias("out_cnt"))
    )

    result = id_disease
    for part in (hospital_days, surgery_count, outpatient_count, hos_elp, sur_elp, out_elp):
        result = result.join(part, on=["id", "kcd_main"], how="left")
    result = result.with_columns(
        pl.col("hos_day").fill_null(0).cast(pl.Int32),
        pl.col("sur_cnt").fill_null(0).cast(pl.Int32),
        pl.col("out_cnt").fill_null(0).cast(pl.Int32),
        elp_day=pl.min_horizontal("hos_elp_day", "sur_elp_day", "out_elp_day"),
    )

    # age (per insured) from the whole mapped table -- so an id with no reviewed
    # row still has an age.
    ages = frame.group_by("id").agg(pl.col("age").cast(pl.Int32).max().alias("age"))
    result = result.join(ages, on="id", how="left")

    # EXPIRED: every id in `mapped` must leave with a row. Ids with nothing in
    # scope collapse to one EXPIRED placeholder carrying days since their most
    # recent reviewed treatment.
    outside = frame.join(result.select("id").unique(), on="id", how="anti")
    if outside.height:
        outside = outside.with_columns(elapsed=_elapsed())
        no_scope = outside.group_by("id").agg(
            pl.col("elapsed")
            .filter((pl.col("review") == 1) & pl.col("elapsed").is_not_null())
            .min()
            .alias("elp_day")
        )
        expired = no_scope.join(ages, on="id", how="left").select(
            pl.col("id"),
            pl.lit(_EXPIRED).alias("kcd_main"),
            pl.col("age"),
            pl.lit(0, dtype=pl.Int32).alias("hos_day"),
            pl.lit(0, dtype=pl.Int32).alias("sur_cnt"),
            pl.lit(0, dtype=pl.Int32).alias("out_cnt"),
            pl.lit(None, dtype=pl.Int32).alias("hos_elp_day"),
            pl.lit(None, dtype=pl.Int32).alias("sur_elp_day"),
            pl.lit(None, dtype=pl.Int32).alias("out_elp_day"),
            pl.col("elp_day").cast(pl.Int32),
        )
        result = pl.concat([result.select(_OUTPUT_COLUMNS), expired.select(_OUTPUT_COLUMNS)])
    else:
        result = result.select(_OUTPUT_COLUMNS)

    return mirror_output(result, was_pandas)
