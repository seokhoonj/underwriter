"""Reconcile the hospital admission/discharge window through ``hos_day``.

``hos_day == 0`` is an outpatient visit (no stay): ``sdate`` is the visit date,
``edate`` is null. For a stay, one endpoint is trusted and the other derived
from ``hos_day``:

- ``sdate`` -- trust admission; ``edate = sdate + hos_day - 1``.
- ``edate`` -- trust discharge; ``sdate = edate - hos_day + 1``.
- ``auto`` -- per row prefer the sdate basis, but if its derived discharge would
  fall after pay/inquiry switch to the edate basis, and if that basis's derived
  admission would precede the accident date fall back to the sdate basis.

Assumes ``sdate``/``edate``/``pay_date``/``inq_date``/``acc_date`` are already
``pl.Date`` and ``hos_day`` is integer. Returns the frame with ``sdate``/``edate``
reconciled; no other column is touched.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from ..errors import InputError

#: Which admission/discharge endpoint the stay reconciliation trusts.
StayBasis = Literal["sdate", "edate", "auto"]


def _days(n: pl.Expr) -> pl.Expr:
    return pl.duration(days=n)


def reconcile_stay_window(frame: pl.DataFrame, method: StayBasis = "sdate") -> pl.DataFrame:
    if method == "sdate":
        return _reconcile_sdate(frame)
    if method == "edate":
        return _reconcile_edate(frame)
    if method == "auto":
        return _reconcile_auto(frame)
    raise InputError(f"method must be 'sdate', 'edate', or 'auto'; got {method!r}.")


def _reconcile_sdate(frame: pl.DataFrame) -> pl.DataFrame:
    is_stay = pl.col("hos_day") > 0
    # fill a missing admission from the discharge, then derive discharge from admission
    frame = frame.with_columns(
        sdate=pl.when(pl.col("sdate").is_null() & is_stay & pl.col("edate").is_not_null())
        .then(pl.col("edate") - _days(pl.col("hos_day") - 1))
        .otherwise(pl.col("sdate"))
    )
    return frame.with_columns(
        edate=pl.when(is_stay & pl.col("sdate").is_not_null())
        .then(pl.col("sdate") + _days(pl.col("hos_day") - 1))
        .otherwise(None)
    )


def _reconcile_edate(frame: pl.DataFrame) -> pl.DataFrame:
    is_stay = pl.col("hos_day") > 0
    frame = frame.with_columns(
        edate=pl.when(pl.col("edate").is_null() & is_stay & pl.col("sdate").is_not_null())
        .then(pl.col("sdate") + _days(pl.col("hos_day") - 1))
        .otherwise(pl.col("edate"))
    )
    frame = frame.with_columns(
        edate=pl.when(pl.col("hos_day") == 0).then(None).otherwise(pl.col("edate"))
    )
    return frame.with_columns(
        sdate=pl.when(is_stay & pl.col("edate").is_not_null())
        .then(pl.col("edate") - _days(pl.col("hos_day") - 1))
        .otherwise(pl.col("sdate"))
    )


def _reconcile_auto(frame: pl.DataFrame) -> pl.DataFrame:
    is_stay = (pl.col("hos_day") > 0).fill_null(False)
    has_sdate = pl.col("sdate").is_not_null()
    has_edate = pl.col("edate").is_not_null()

    discharge_upper = pl.min_horizontal("pay_date", "inq_date")  # na.rm per element
    admit_lower = pl.col("acc_date")
    edate_from_sdate = pl.col("sdate") + _days(pl.col("hos_day") - 1)
    sdate_from_edate = pl.col("edate") - _days(pl.col("hos_day") - 1)

    sdate_basis_ok = has_sdate & (discharge_upper.is_null() | (edate_from_sdate <= discharge_upper))
    edate_basis_ok = has_edate & (admit_lower.is_null() | (sdate_from_edate >= admit_lower))

    use_edate = (
        is_stay & has_edate & (~has_sdate | (~sdate_basis_ok & edate_basis_ok))
    ).fill_null(False)
    use_sdate = is_stay & has_sdate & ~use_edate

    return frame.with_columns(
        sdate=pl.when(use_edate).then(sdate_from_edate).otherwise(pl.col("sdate")),
        edate=pl.when(~is_stay)
        .then(None)
        .when(use_sdate)
        .then(edate_from_sdate)
        .otherwise(pl.col("edate")),
    )
