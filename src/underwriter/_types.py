"""Shared type aliases. Import-light: heavy names live behind ``TYPE_CHECKING``
so a runtime import of the package never pulls in pandas."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import polars as pl

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    #: A frame the public boundary accepts: polars natively, pandas by mirroring.
    FrameLike: TypeAlias = pl.DataFrame | pd.DataFrame
else:
    FrameLike = object
