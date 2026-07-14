"""Frame-boundary helpers: accept polars or pandas, validate columns, mirror the
output frame type back to whatever the caller passed in."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .._types import FrameLike
from ..errors import FrameTypeError, MissingColumnsError


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
    raise FrameTypeError(
        f"expected a polars or pandas DataFrame, got {type(frame).__name__!r}."
    )


def to_polars_frame(frame: object) -> pl.DataFrame:
    """Coerce to polars and drop the ``was_pandas`` flag -- for callers that never
    mirror their output back (they always return polars)."""
    return to_polars(frame)[0]


def mirror_output(frame: pl.DataFrame, was_pandas: bool) -> FrameLike:
    """Return ``frame`` as pandas when the caller passed pandas in, else polars."""
    return frame.to_pandas() if was_pandas else frame


def require_columns(
    frame: pl.DataFrame, columns: Iterable[str], *, where: str
) -> None:
    """Raise a pointed ``MissingColumnsError`` if ``frame`` is missing any of ``columns``."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        have = ", ".join(frame.columns) or "(none)"
        raise MissingColumnsError(
            f"{where}: missing required column(s) {missing}. Present: {have}."
        )
