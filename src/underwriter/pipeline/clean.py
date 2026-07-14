"""Stage 1 -- cleanse a raw ICIS claim table into a wide master, and keep each
id's latest inquiry.

One row per claim line: parse dates, reconcile the admission/discharge window,
normalize the diagnosis codes and pull them leftward (``kcd0`` = main), and drop
exact duplicate rows. No row is dropped for lacking a code: a codeless line
becomes ``VACANT`` (nothing to underwrite), a line whose codes are all
unreadable becomes ``IRREGULAR`` (routes to the underwriter). So an insured is
never lost between the raw feed and the final decision.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .._kernels.io import mirror_output, require_columns, to_polars
from .._kernels.kcd import split_kcd
from .._kernels.window import StayBasis, reconcile_stay_window
from .._types import FrameLike
from ..sentinels import Sentinel

_DEFAULT_KCD_COLUMNS = ("kcd0", "kcd1", "kcd2", "kcd3", "kcd4")
_DATE_COLUMNS = ("inq_date", "acc_date", "pay_date", "sdate", "edate")
_COLUMN_ORDER = (
    "id", "gender", "age", "inq_date", "pay_date", "acc_date",
    "sdate", "edate", "hos_day", "hos_cnt", "sur_cnt",
)


def _parse_ymd(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias(column)


def clean_icis(
    claims: FrameLike,
    *,
    kcd_columns: Sequence[str] = _DEFAULT_KCD_COLUMNS,
    method: StayBasis = "sdate",
) -> FrameLike:
    """Cleanse a raw ICIS claim table (one row per claim line).

    ``method`` picks which admission/discharge endpoint is trusted -- ``"sdate"``
    (default), ``"edate"``, or ``"auto"`` (see ``reconcile_stay_window``).
    Accepts and returns polars or pandas.
    """
    frame, was_pandas = to_polars(claims)
    kcd_columns = list(kcd_columns)
    require_columns(
        frame, ["id", "hos_day", *_DATE_COLUMNS, *kcd_columns], where="clean_icis"
    )

    # codes to str; remember which rows arrived with any code (VACANT vs IRREGULAR)
    frame = frame.with_columns(pl.col(c).cast(pl.Utf8) for c in kcd_columns)
    has_code = pl.any_horizontal(
        (pl.col(c).str.strip_chars() != "").fill_null(False) for c in kcd_columns
    )
    frame = frame.with_columns(_has_code=has_code)

    # parse dates, coerce counts, normalize gender to its two levels
    coercions = [_parse_ymd(c) for c in _DATE_COLUMNS]
    coercions.append(pl.col("hos_day").cast(pl.Int32))
    if "sur_cnt" in frame.columns:
        coercions.append(pl.col("sur_cnt").cast(pl.Int32))
    if "hos_cnt" in frame.columns:
        coercions.append(pl.col("hos_cnt").cast(pl.Int32))
    if "age" in frame.columns:
        coercions.append(pl.col("age").cast(pl.Int32))
    if "gender" in frame.columns:
        gender = pl.col("gender").cast(pl.Utf8)
        coercions.append(
            pl.when(gender.is_in(["1", "2"])).then(gender).otherwise(None).alias("gender")
        )
    frame = frame.with_columns(coercions)

    # reconcile the stay window
    frame = reconcile_stay_window(frame, method)

    # gather every valid code across the cells in order (this subsumes R's
    # redistribute-multi + normalize + pack-left in one pass), then stamp the
    # codeless rows by why they are codeless.
    joined = pl.concat_str(kcd_columns, separator=",", ignore_nulls=True).fill_null("")
    frame = frame.with_columns(_codes=split_kcd(joined))
    n_codes = pl.col("_codes").list.len().fill_null(0)
    frame = frame.with_columns(
        [
            pl.when(n_codes == 0)
            .then(
                pl.when(pl.col("_has_code"))
                .then(pl.lit(Sentinel.IRREGULAR.value))
                .otherwise(pl.lit(Sentinel.VACANT.value))
            )
            .otherwise(pl.col("_codes").list.get(0, null_on_oob=True))
            .alias(kcd_columns[0])
        ]
        + [
            pl.col("_codes").list.get(i, null_on_oob=True).alias(kcd_columns[i])
            for i in range(1, len(kcd_columns))
        ]
    )

    frame = frame.drop("_has_code", "_codes")
    frame = frame.unique(maintain_order=True)
    # order the known schema first, but keep any caller-supplied extra columns
    # (contract numbers, book/insurer discriminators) at the end -- never drop them.
    known = [c for c in (*_COLUMN_ORDER, *kcd_columns) if c in frame.columns]
    extra = [c for c in frame.columns if c not in known]
    frame = frame.select([*known, *extra])
    return mirror_output(frame, was_pandas)


def filter_latest_inquiry(cleaned: FrameLike) -> FrameLike:
    """Keep each id's most recent inquiry. All rows sharing an id's maximum
    ``inq_date`` are kept (one inquiry spans many claim rows); an id whose every
    ``inq_date`` is null keeps all its rows. Every id is preserved."""
    frame, was_pandas = to_polars(cleaned)
    require_columns(frame, ["id", "inq_date"], where="filter_latest_inquiry")

    latest = (
        frame.group_by("id")
        .agg(pl.col("inq_date").max().alias("_latest"))
        .filter(pl.col("_latest").is_not_null())
    )
    kept = (
        frame.join(latest, on="id", how="inner")
        .filter(pl.col("inq_date") == pl.col("_latest"))
        .drop("_latest")
    )
    no_date = frame.join(latest.select("id"), on="id", how="anti")
    result = pl.concat([kept, no_date]) if no_date.height else kept
    return mirror_output(result, was_pandas)
