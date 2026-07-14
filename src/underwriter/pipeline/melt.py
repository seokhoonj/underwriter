"""Stage 2 -- melt the wide cleaned master to one row per diagnosis code."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .._kernels.io import mirror_output, to_polars

_DEFAULT_KCD_COLUMNS = ("kcd0", "kcd1", "kcd2", "kcd3", "kcd4")


def melt_kcd(cleaned: object, *, kcd_columns: Sequence[str] = _DEFAULT_KCD_COLUMNS) -> object:
    """Wide (``kcd0..kcd4``) -> long: one row per non-null code, carrying every
    other column. ``sub_kcd`` is ``0`` for the main diagnosis (``kcd0``) and ``1``
    for a sub-diagnosis."""
    frame, was_pandas = to_polars(cleaned)
    kcd_columns = list(kcd_columns)
    id_vars = [c for c in frame.columns if c not in kcd_columns]
    long = (
        frame.unpivot(
            index=id_vars, on=kcd_columns, variable_name="_source", value_name="kcd"
        )
        .filter(pl.col("kcd").is_not_null())
        .with_columns(
            sub_kcd=pl.when(pl.col("_source") == kcd_columns[0])
            .then(0)
            .otherwise(1)
            .cast(pl.Int8)
        )
        .drop("_source")
    )
    return mirror_output(long, was_pandas)
