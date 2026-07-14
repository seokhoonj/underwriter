"""End-to-end front-half pipeline: clean -> melt -> map -> aggregate.

The load-bearing invariant is no-insured-left-behind: every id in the raw feed
leaves aggregate_disease with at least one row, via a real disease, VACANT,
UNMAPPED, or an EXPIRED placeholder.
"""

from __future__ import annotations

import polars as pl

import underwriter as uw


def _disease_table():
    # M51 (disc) reviewed, 60-month lookback; A00 short 12-month lookback
    return pl.DataFrame(
        {
            "kcd": ["M51", "M511", "A00"],
            "kcd_main": ["M51", "M51", "A00"],
            "sub_chk": [1, 1, 1],
            "lookback_mon": [60, 60, 12],
        }
    )


def _claim(**overrides):
    row = {
        "id": "1", "gender": "1", "age": 40,
        "inq_date": "20240601", "pay_date": "20240110", "acc_date": "20240101",
        "sdate": None, "edate": None, "hos_day": 0, "hos_cnt": 0, "sur_cnt": 0,
        "kcd0": None, "kcd1": None, "kcd2": None, "kcd3": None, "kcd4": None,
    }
    row.update(overrides)
    return row


def _run(rows):
    claims = pl.DataFrame(rows)
    cleaned = uw.filter_latest_inquiry(uw.clean_icis(claims))
    mapped = uw.map_disease(uw.melt_kcd(cleaned), _disease_table())
    return claims, uw.aggregate_disease(mapped)


def test_no_insured_left_behind():
    rows = [
        # A: a real recent disc diagnosis, inpatient 5 days
        _claim(id="A", kcd0="M51.13", hos_day=5, sdate="20240501"),
        # B: only a codeless line -> VACANT survives
        _claim(id="B"),
        # C: a code the disease table has no row for -> UNMAPPED
        _claim(id="C", kcd0="Z99", sdate="20240301"),
        # D: an A00 diagnosis whose treatment is far older than its 12-mo lookback
        #    (2019) -> aged out -> EXPIRED placeholder
        _claim(id="D", kcd0="A00", acc_date="20190101", pay_date="20190101", sdate="20190101"),
    ]
    claims, agg = _run(rows)
    assert set(agg["id"].to_list()) == set(claims["id"].to_list())  # nobody dropped
    assert agg["id"].n_unique() == agg.height  # one row per (id, kcd_main), ids here distinct


def test_expired_placeholder_for_aged_out():
    claims, agg = _run(
        [_claim(id="D", kcd0="A00", acc_date="20190101", pay_date="20190101", sdate="20190101")]
    )
    row = agg.filter(pl.col("id") == "D")
    assert row["kcd_main"].to_list() == ["EXPIRED"]
    assert row.select(["hos_day", "sur_cnt", "out_cnt"]).row(0) == (0, 0, 0)


def test_unmapped_valid_code_survives_as_unmapped():
    # Z99 is a well-formed code the disease table has no row for -> UNMAPPED, kept
    claims, agg = _run([_claim(id="C", kcd0="Z99", sdate="20240301")])
    assert agg.filter(pl.col("id") == "C")["kcd_main"].to_list() == ["UNMAPPED"]


def test_vacant_only_insured_survives_as_vacant():
    claims, agg = _run([_claim(id="B")])
    assert agg.filter(pl.col("id") == "B")["kcd_main"].to_list() == ["VACANT"]


def test_inpatient_stay_days_and_elapsed():
    # inpatient 2024-05-01..05-05 (5 days), inquiry 2024-06-01
    claims, agg = _run([_claim(id="A", kcd0="M51.13", hos_day=5, sdate="20240501")])
    row = agg.filter(pl.col("id") == "A")
    assert row["kcd_main"].to_list() == ["M51"]
    assert row["hos_day"].to_list() == [5]                  # 5 distinct stay days
    assert row["elp_day"].to_list() == [27]                 # 06-01 minus discharge 05-05
