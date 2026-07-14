"""Synthetic example data for docs and tests -- no real claim data, no
proprietary rulebook.

``make_icis`` generates a raw ICIS-shaped claim table; ``make_disease_table``
a small KCD -> representative-disease lookup. Together they drive the front-half
pipeline (cleanse -> aggregate) in the public example, standing in for the
``claim`` / ``main`` inputs the original notebook read from disk. The real rule
set stays out of the repo.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# kcd, representative disease, sub-diagnosis review flag, lookback window (months)
_DISEASE = (
    ("M51", "M51", 1, 60), ("M511", "M51", 1, 60), ("M54", "M54", 1, 60),
    ("I10", "I10", 1, 24), ("E11", "E11", 1, 60), ("E119", "E11", 1, 60),
    ("C34", "C34", 1, 120), ("A00", "A00", 1, 12), ("S82", "S82", 1, 36),
    ("J45", "J45", 1, 24), ("K21", "K21", 1, 12), ("N39", "N39", 1, 12),
)
_CODES = [d[0] for d in _DISEASE]


def make_disease_table() -> pl.DataFrame:
    """A small synthetic KCD -> disease lookup for :func:`~underwriter.map_disease`."""
    return pl.DataFrame(
        {
            "kcd": [d[0] for d in _DISEASE],
            "kcd_main": [d[1] for d in _DISEASE],
            "sub_chk": [d[2] for d in _DISEASE],
            "lookback_mon": [d[3] for d in _DISEASE],
        }
    )


def _ymd(rng: np.random.Generator, lo_year: int, hi_year: int) -> str:
    year = int(rng.integers(lo_year, hi_year + 1))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 28))
    return f"{year:04d}{month:02d}{day:02d}"


def make_icis(n_insured: int = 300, seed: int = 0) -> pl.DataFrame:
    """A raw ICIS-shaped claim table (one row per claim line), synthetic and
    deterministic in ``seed``. Includes the shapes the pipeline must handle:
    inpatient and outpatient lines, multi-code cells, codeless lines (VACANT),
    unreadable codes (IRREGULAR), and aged-out histories (EXPIRED)."""
    rng = np.random.default_rng(seed)
    inquiry = "20240601"
    rows: list[dict] = []
    for i in range(n_insured):
        pid = f"P{i:04d}"
        gender = str(int(rng.integers(1, 3)))
        age = int(rng.integers(20, 71))
        draw = rng.random()
        n_lines = int(rng.integers(1, 5))
        for _ in range(n_lines):
            # 6% codeless (VACANT), 4% unreadable (IRREGULAR), else a real code
            r = rng.random()
            if r < 0.06:
                cells = [None, None, None, None, None]
            elif r < 0.10:
                cells = ["??", None, None, None, None]
            elif r < 0.16:  # a multi-code cell
                a, b = rng.choice(_CODES, size=2, replace=False)
                cells = [f"{a},{b}", None, None, None, None]
            else:
                cells = [str(rng.choice(_CODES)), None, None, None, None]

            aged_out = draw < 0.08  # a fraction have only old treatments -> EXPIRED
            acc = _ymd(rng, 2016, 2018) if aged_out else _ymd(rng, 2021, 2024)
            inpatient = rng.random() < 0.35
            hos_day = int(rng.integers(1, 21)) if inpatient else 0
            sur_cnt = int(rng.integers(0, 2)) if rng.random() < 0.2 else 0
            rows.append(
                {
                    "id": pid, "gender": gender, "age": age,
                    "inq_date": inquiry, "pay_date": acc, "acc_date": acc,
                    "sdate": acc, "edate": None,
                    "hos_day": hos_day, "hos_cnt": 1 if inpatient else 0, "sur_cnt": sur_cnt,
                    "kcd0": cells[0], "kcd1": cells[1], "kcd2": cells[2],
                    "kcd3": cells[3], "kcd4": cells[4],
                }
            )
    return pl.DataFrame(rows)
