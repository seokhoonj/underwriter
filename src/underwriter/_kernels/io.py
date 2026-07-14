"""Frame-boundary helpers: accept polars or pandas, validate columns, mirror the
output frame type back to whatever the caller passed in."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl


def to_polars(frame: object) -> tuple[pl.DataFrame, bool]:
    """Coerce a public input to a polars frame.

    Returns ``(polars_frame, was_pandas)`` so the caller can mirror the output
    back to pandas if that is what came in. A polars frame passes through
    untouched; a pandas frame is converted via Arrow.
    """
    if isinstance(frame, pl.DataFrame):
        return frame, False
    # duck-type pandas without importing it at module load
    if type(frame).__module__.split(".", 1)[0] == "pandas":
        return pl.from_pandas(frame), True
    raise TypeError(
        f"expected a polars or pandas DataFrame, got {type(frame).__name__!r}."
    )


def mirror_output(frame: pl.DataFrame, was_pandas: bool):
    """Return ``frame`` as pandas when the caller passed pandas in, else polars."""
    return frame.to_pandas() if was_pandas else frame


def require_columns(
    frame: pl.DataFrame, columns: Iterable[str], *, where: str
) -> None:
    """Raise a pointed ``ValueError`` if ``frame`` is missing any of ``columns``."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        have = ", ".join(frame.columns) or "(none)"
        raise ValueError(
            f"{where}: missing required column(s) {missing}. Present: {have}."
        )


def as_column_list(columns: str | Sequence[str]) -> list[str]:
    """Normalize a single name or a sequence of names to a list."""
    return [columns] if isinstance(columns, str) else list(columns)
