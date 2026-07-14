"""Stage 3 -- map each diagnosis code to its representative disease and scope flags.

Exact ``kcd`` match first, then a 3-character-prefix fallback, then ``UNMAPPED``
for a still-unmatched valid code (kept, never dropped). The reserved codes
``VACANT`` / ``IRREGULAR`` carry through as their own ``kcd_main``.
"""

from __future__ import annotations

import polars as pl

from .._kernels.io import mirror_output, require_columns, to_polars
from .._kernels.months import minus_months
from .._types import FrameLike
from ..errors import RulebookError
from ..sentinels import Sentinel

_NO_WINDOW = (Sentinel.IRREGULAR.value, Sentinel.UNMAPPED.value)
_RESERVED = (Sentinel.VACANT.value, Sentinel.IRREGULAR.value)


def map_disease(melted: FrameLike, disease_table: FrameLike) -> FrameLike:
    """Attach ``kcd_main``, ``sub_chk``, ``lookback_mon`` and the scope flags
    ``review``, ``in_lookback``, ``in_5yr`` to each melted diagnosis row."""
    frame, was_pandas = to_polars(melted)
    disease, _ = to_polars(disease_table)
    require_columns(frame, ["kcd", "sub_kcd", "inq_date"], where="map_disease")
    require_columns(disease, ["kcd", "kcd_main", "sub_chk", "lookback_mon"], where="map_disease")

    # a kcd must map to one disease: the exact and fallback joins would resolve a
    # duplicate key differently, so reject it rather than depend on row order.
    key = disease.filter(pl.col("kcd").is_not_null())["kcd"]
    if key.n_unique() != key.len():
        dups = key.filter(key.is_duplicated()).unique().head(5).to_list()
        raise RulebookError(f"disease_table has duplicate `kcd` keys: {dups}.")

    lookup = disease.select("kcd", "kcd_main", "sub_chk", "lookback_mon")
    frame = frame.join(lookup, on="kcd", how="left")

    # 3-character-prefix fallback for rows the exact match missed
    fallback_lookup = lookup.rename(
        {"kcd": "_k3", "kcd_main": "_fb_main", "sub_chk": "_fb_sub", "lookback_mon": "_fb_look"}
    ).unique(subset="_k3", keep="first")
    matched = pl.col("kcd_main").is_not_null()
    frame = (
        frame.with_columns(_k3=pl.col("kcd").str.slice(0, 3))
        .join(fallback_lookup, on="_k3", how="left")
        .with_columns(
            kcd_main=pl.when(matched).then(pl.col("kcd_main")).otherwise(pl.col("_fb_main")),
            sub_chk=pl.when(matched).then(pl.col("sub_chk")).otherwise(pl.col("_fb_sub")),
            lookback_mon=pl.when(matched)
            .then(pl.col("lookback_mon"))
            .otherwise(pl.col("_fb_look")),
        )
        .drop("_k3", "_fb_main", "_fb_sub", "_fb_look")
    )

    # a valid code neither join found -> UNMAPPED (reviewed); then let the reserved
    # codes carry through as their own kcd_main (order matters: UNMAPPED first).
    still_null = pl.col("kcd_main").is_null()
    frame = frame.with_columns(
        sub_chk=pl.when(still_null).then(pl.lit(1)).otherwise(pl.col("sub_chk")),
        kcd_main=pl.when(still_null).then(pl.lit(Sentinel.UNMAPPED.value)).otherwise(pl.col("kcd_main")),
    )
    reserved = pl.col("kcd").is_in(_RESERVED)
    frame = frame.with_columns(
        kcd_main=pl.when(reserved).then(pl.col("kcd")).otherwise(pl.col("kcd_main")),
        sub_chk=pl.when(reserved).then(pl.lit(1)).otherwise(pl.col("sub_chk")),
    )
    frame = frame.with_columns(pl.col("sub_chk").cast(pl.Int8))

    # scope flags
    tdate = pl.max_horizontal("acc_date", "sdate", "edate")
    frame = frame.with_columns(
        review=((pl.col("sub_kcd") == 0) | (pl.col("sub_chk") == 1)).cast(pl.Int8),
        in_lookback=(tdate >= minus_months(pl.col("inq_date"), pl.col("lookback_mon"))).cast(pl.Int8),
        in_5yr=(tdate >= minus_months(pl.col("inq_date"), pl.lit(60))).cast(pl.Int8),
    )
    # VACANT can never age out: keep it in its lookback unconditionally.
    frame = frame.with_columns(
        in_lookback=pl.when(pl.col("kcd_main") == Sentinel.VACANT.value)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.col("in_lookback"))
    )
    return mirror_output(frame, was_pandas)
