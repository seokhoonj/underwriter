"""clean_icis / filter_latest_inquiry behaviour, from the R package's contract."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from underwriter.pipeline.clean import clean_icis, filter_latest_inquiry


def _claim(**overrides):
    row = {
        "id": "1", "gender": "1", "age": 40,
        "inq_date": "20240201", "pay_date": "20240110", "acc_date": "20240101",
        "sdate": None, "edate": None, "hos_day": 0, "hos_cnt": 0, "sur_cnt": 0,
        "kcd0": None, "kcd1": None, "kcd2": None, "kcd3": None, "kcd4": None,
    }
    row.update(overrides)
    return row


def test_sdate_basis_derives_discharge_and_normalizes_code():
    df = pl.DataFrame([_claim(kcd0="M51.13", hos_day=5, sdate="20240101")])
    out = clean_icis(df)
    assert out["kcd0"].to_list() == ["M511"]
    assert out["sdate"].to_list() == [dt.date(2024, 1, 1)]
    assert out["edate"].to_list() == [dt.date(2024, 1, 5)]  # sdate + 5 - 1


def test_outpatient_has_null_discharge():
    df = pl.DataFrame([_claim(id="2", kcd0="A00", hos_day=0, sdate="20240115")])
    out = clean_icis(df)
    assert out["edate"].to_list() == [None]
    assert out["sdate"].to_list() == [dt.date(2024, 1, 15)]


def test_codeless_line_becomes_vacant():
    df = pl.DataFrame([_claim(id="3")])
    out = clean_icis(df)
    assert out["kcd0"].to_list() == ["VACANT"]


def test_unreadable_code_becomes_irregular():
    df = pl.DataFrame([_claim(id="4", kcd0="xx")])
    out = clean_icis(df)
    assert out["kcd0"].to_list() == ["IRREGULAR"]


def test_multi_code_cell_is_spread_across_columns():
    df = pl.DataFrame([_claim(id="5", kcd0="K63.5,S33", kcd1="M00")])
    out = clean_icis(df)
    assert out.select(["kcd0", "kcd1", "kcd2"]).row(0) == ("K635", "S33", "M00")


def test_exact_duplicate_rows_are_dropped():
    df = pl.DataFrame([_claim(kcd0="A00"), _claim(kcd0="A00")])
    out = clean_icis(df)
    assert out.height == 1


def test_clean_icis_keeps_caller_supplied_extra_columns():
    # a book/contract discriminator the pipeline does not know must not be dropped
    df = pl.DataFrame([_claim(kcd0="A00", policy_no="X1", book="alpha")])
    out = clean_icis(df)
    assert "policy_no" in out.columns and "book" in out.columns
    assert out.select(["policy_no", "book"]).row(0) == ("X1", "alpha")


def test_clean_icis_mirrors_pandas_input_type():
    pd = pytest.importorskip("pandas")
    out = clean_icis(pd.DataFrame([_claim(kcd0="A00")]))
    assert isinstance(out, pd.DataFrame)
    assert out["kcd0"].tolist() == ["A00"]


def test_edate_method_derives_admission_from_discharge():
    df = pl.DataFrame([_claim(id="6", kcd0="A00", hos_day=5, sdate=None, edate="20240105")])
    out = clean_icis(df, method="edate")
    assert out["sdate"].to_list() == [dt.date(2024, 1, 1)]  # edate - 5 + 1
    assert out["edate"].to_list() == [dt.date(2024, 1, 5)]


def test_auto_method_uses_edate_basis_when_admission_is_missing():
    df = pl.DataFrame([_claim(id="7", kcd0="A00", hos_day=3, sdate=None, edate="20240110")])
    out = clean_icis(df, method="auto")
    assert out["sdate"].to_list() == [dt.date(2024, 1, 8)]  # edate - 3 + 1


def test_filter_latest_inquiry_keeps_max_and_never_drops_ids():
    df = pl.DataFrame(
        [
            _claim(id="1", inq_date="20240101", kcd0="A00"),
            _claim(id="1", inq_date="20240301", kcd0="B00"),  # later inquiry
            _claim(id="2", inq_date=None, kcd0="C00"),         # no date -> kept whole
        ]
    )
    cleaned = clean_icis(df)
    kept = filter_latest_inquiry(cleaned)
    assert set(kept["id"].to_list()) == {"1", "2"}          # no id dropped
    id1 = kept.filter(pl.col("id") == "1")
    assert id1["inq_date"].to_list() == [dt.date(2024, 3, 1)]  # only the latest
    assert id1["kcd0"].to_list() == ["B00"]
