"""Two deliberately different month arithmetics.

``minus_months`` walks back a *calendar* window (used to scope a diagnosis by
its lookback), clamping to the target month's last valid day: ``2024-03-31``
minus one month is ``2024-02-29``, not ``2024-03-02``.

``resolve_months`` (added in Phase 2) converts a rulebook period *mark* to a
remaining-month count using fixed 30-day months. The two must not be confused --
each is pinned by its own tests.
"""

from __future__ import annotations

import polars as pl


def minus_months(date: pl.Expr, n_months: pl.Expr) -> pl.Expr:
    """``date`` shifted back ``n_months`` calendar months, clamped to month-end.

    A null ``n_months`` (a disease with no defined lookback) yields null. Relies
    on polars' month offset, which clamps an out-of-range day to the last day of
    the resulting month.
    """
    offset = pl.concat_str(
        [pl.lit("-"), n_months.cast(pl.Int64).cast(pl.Utf8), pl.lit("mo")]
    )
    return (
        pl.when(n_months.is_null())
        .then(None)
        .otherwise(date.dt.offset_by(offset))
    )
