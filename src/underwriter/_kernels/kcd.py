"""KCD (Korean Classification of Diseases) code handling: normalize a raw code
cell to a canonical form, split a multi-code cell, and pack non-null codes to
the left across the ``kcd0..kcdN`` columns.

All functions are polars-expression builders (they return ``pl.Expr`` or operate
on a frame), so they compose inside a single lazy/eager query with no per-row
Python.
"""

from __future__ import annotations

import polars as pl

#: A canonical KCD code: one letter then at least two digits (e.g. ``M511``).
_KCD_PATTERN = r"^[A-Z][0-9]{2,}$"


def normalize_kcd(code: pl.Expr, length: int = 4) -> pl.Expr:
    """Canonicalize a raw code cell.

    Upper-case; keep only ``[A-Z0-9.,]``; take the part before the first comma;
    drop dots; truncate to ``length`` characters; then null out anything that is
    not a canonical code. Examples: ``"m00.0" -> "M000"``, ``"M51.13" -> "M511"``,
    ``"K63.5, S33" -> "K635"``, ``"xx" -> None``.
    """
    cleaned = (
        code.cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9.,]", "")
        .str.split(",")
        .list.first()
        .str.replace_all(r"\.", "")
        .str.slice(0, length)
    )
    return pl.when(cleaned.str.contains(_KCD_PATTERN)).then(cleaned).otherwise(None)


def split_kcd(code: pl.Expr, length: int = 4) -> pl.Expr:
    """Split a possibly comma-joined cell into a list of every canonical code it
    holds, dropping the ones that do not normalize. ``"K63.5, S33" -> ["K635",
    "S33"]`` (``"S33"`` kept only if it is a valid code)."""
    tokens = (
        code.cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9.,]", "")
        .str.split(",")
    )
    # normalize each token, then drop the nulls
    normalized = tokens.list.eval(
        pl.when(
            pl.element().str.replace_all(r"\.", "").str.slice(0, length).str.contains(_KCD_PATTERN)
        )
        .then(pl.element().str.replace_all(r"\.", "").str.slice(0, length))
        .otherwise(None)
    )
    return normalized.list.drop_nulls()


def pack_kcd_left(frame: pl.DataFrame, kcd_columns: list[str]) -> pl.DataFrame:
    """Shift non-null codes leftward so ``kcd0`` holds the first present code,
    ``kcd1`` the second, and trailing columns become null. Assumes the columns
    already hold canonical codes or null (call ``normalize_kcd`` first)."""
    packed = pl.concat_list([pl.col(c) for c in kcd_columns]).list.drop_nulls()
    return frame.with_columns(
        packed.list.get(i, null_on_oob=True).alias(column)
        for i, column in enumerate(kcd_columns)
    )
