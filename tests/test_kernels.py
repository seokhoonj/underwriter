"""Unit vectors for the numerical kernels, taken from the R package's documented
behaviour. These are the parity-critical primitives."""

from __future__ import annotations

import datetime as dt

import polars as pl

from underwriter._kernels.kcd import normalize_kcd, pack_kcd_left, split_kcd
from underwriter._kernels.months import minus_months
from underwriter._kernels.stay import count_stay


def test_normalize_kcd_examples():
    raw = pl.DataFrame({"kcd": ["m00.0", "M51.13", "K63.5, S33", "xx", "M0", None]})
    out = raw.select(normalize_kcd(pl.col("kcd")))["kcd"].to_list()
    assert out == ["M000", "M511", "K635", None, None, None]


def test_split_kcd_keeps_every_valid_code():
    raw = pl.DataFrame({"kcd": ["K63.5, S33", "m00.0", "xx,Z01"]})
    out = raw.select(split_kcd(pl.col("kcd")))["kcd"].to_list()
    assert out == [["K635", "S33"], ["M000"], ["Z01"]]


def test_pack_kcd_left_shifts_non_nulls():
    frame = pl.DataFrame(
        {
            "kcd0": [None, "A00"],
            "kcd1": ["A00", None],
            "kcd2": [None, "B11"],
            "kcd3": ["B11", None],
            "kcd4": [None, None],
        }
    )
    packed = pack_kcd_left(frame, ["kcd0", "kcd1", "kcd2", "kcd3", "kcd4"])
    assert packed.row(0) == ("A00", "B11", None, None, None)
    assert packed.row(1) == ("A00", "B11", None, None, None)


def test_minus_months_clamps_to_month_end():
    frame = pl.DataFrame(
        {"date": [dt.date(2024, 3, 31), dt.date(2024, 1, 15)], "n": [1, 12]}
    )
    out = frame.select(minus_months(pl.col("date"), pl.col("n")).alias("out"))["out"].to_list()
    assert out == [dt.date(2024, 2, 29), dt.date(2023, 1, 15)]


def test_minus_months_null_lookback_is_null():
    frame = pl.DataFrame({"date": [dt.date(2024, 3, 31)], "n": [None]})
    out = frame.select(minus_months(pl.col("date"), pl.col("n")).alias("out"))["out"].to_list()
    assert out == [None]


def test_count_stay_dedups_overlap_and_caps_at_inquiry():
    rows = pl.DataFrame(
        {
            "id": ["x", "x", "x"],
            "kcd_main": ["M51", "M51", "M51"],
            # two overlapping stays 01-01..01-05 and 01-04..01-08 -> 8 distinct days
            "sdate": [dt.date(2024, 1, 1), dt.date(2024, 1, 4), dt.date(2024, 1, 20)],
            "edate": [dt.date(2024, 1, 5), dt.date(2024, 1, 8), dt.date(2024, 1, 31)],
            # inquiry caps the third stay at 01-25 -> 6 days (20..25)
            "inq_date": [dt.date(2024, 2, 1)] * 3,
        }
    )
    # cap the last stay at inquiry 2024-01-25 for this check
    rows = rows.with_columns(inq_date=pl.lit(dt.date(2024, 1, 25)))
    out = count_stay(rows)
    # days: {01-01..01-08}=8 distinct  +  {01-20..01-25}=6  = 14
    assert out.row(0) == ("x", "M51", 14) or out.sort("kcd_main").row(0) == ("x", "M51", 14)


def test_package_imports():
    import underwriter

    assert underwriter.Sentinel.VACANT == "VACANT"
    assert underwriter.SENTINELS == ("VACANT", "IRREGULAR", "UNMAPPED", "EXPIRED")
