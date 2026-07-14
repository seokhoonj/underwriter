"""Data-quality report for an ICIS claim table (raw, cleansed, or mapped).

Per-id consistency, row-level anomalies, a value profile, the no-diagnosis split,
and -- on a mapped table -- the lookback scope. Every anomaly is counted as
affected rows and affected insured, with percentages. Aggregates only; no raw
rows are printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import polars as pl

from .._kernels.io import to_polars
from .._types import FrameLike
from ..errors import InputError
from ..sentinels import Sentinel

_DATE_COLUMNS = ("acc_date", "sdate", "edate", "pay_date", "inq_date")
_KCD_COLUMNS = ("kcd0", "kcd1", "kcd2", "kcd3", "kcd4")


class CountStat(TypedDict):
    """Affected rows and insured for one anomaly, with their percentages."""

    n_row: int
    n_id: int
    pct_row: float
    pct_id: float


@dataclass(frozen=True)
class IcisDiagnosis:
    n_row: int
    n_id: int
    id_consistency: dict[str, float]
    date_chronology: dict[str, CountStat] | None
    row_anomaly: dict[str, CountStat]
    hos_day_span: dict[str, CountStat] | None
    gender_dist: dict[str, int] | None
    age_stat: dict[str, int | None] | None
    missing_required: dict[str, int]
    no_diagnosis: dict[str, CountStat | int] | None
    scope: dict[str, CountStat | int] | None

    def __repr__(self) -> str:
        def cnt(stat: dict) -> str:
            return f"n_row {stat['n_row']:>8,} ({stat['pct_row']:6.2f}%) | n_id {stat['n_id']:>8,} ({stat['pct_id']:6.2f}%)"

        ic = self.id_consistency
        lines = [
            f"n_row={self.n_row:,} | n_id={self.n_id:,} | rows/id median={ic['rows_per_id_median']}",
            "\n== id_consistency ==",
            f"  multi-gender ids : {ic['multi_gender_ids']:,}",
            f"  multi-age ids    : {ic['multi_age_ids']:,}  (age-span>=2yr: {ic['age_span_ge2_ids']:,})",
            f"  multi-inq ids    : {ic['multi_inq_ids']:,}",
        ]
        if self.date_chronology:
            lines.append("\n== date_chronology (acc <= sdate <= edate <= pay <= inq) ==")
            for k, v in self.date_chronology.items():
                lines.append(f"  {k:<20}: {cnt(v)}")
        lines.append("\n== row_anomaly ==")
        for k, v in self.row_anomaly.items():
            if v is not None:
                lines.append(f"  {k:<20}: {cnt(v)}")
        if self.no_diagnosis:
            nd = self.no_diagnosis
            lines.append("\n== no diagnosis (all kcd empty; kept on the no-diagnosis code) ==")
            lines.append(f"  all kcd empty        : {cnt(nd['all_empty'])}")
            lines.append(f"    inpatient/surgery  : {cnt(nd['inpatient'])}")
            lines.append(f"    outpatient (visit) : {cnt(nd['outpatient'])}")
            lines.append(f"    empty (no tx/date) : {cnt(nd['empty'])}")
            lines.append(f"  ids with no coded row: {nd['all_empty_ids']:,}")
        if self.scope:
            sc = self.scope
            lines.append("\n== scope (which diagnoses the lookback windows admit) ==")
            lines.append(f"  not reviewed             : {cnt(sc['not_reviewed'])}")
            lines.append(f"  reviewed but out of window: {cnt(sc['out_of_window'])}")
            lines.append(f"  ids with nothing in scope: {sc['no_scope_ids']:,}")
            lines.append(f"    no diagnosis ever      : {sc['never_coded_ids']:,}  -> VACANT")
            lines.append(f"    every diagnosis aged out: {sc['aged_out_ids']:,}  -> EXPIRED")
        return "\n".join(lines)


def _parse_date(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8)
    return (
        pl.when(text.str.contains("-"))
        .then(text.str.to_date("%Y-%m-%d", strict=False))
        .otherwise(text.str.to_date("%Y%m%d", strict=False))
    )


def _present(column: str, frame: pl.DataFrame) -> pl.Expr:
    if frame.schema[column] == pl.Utf8:
        return pl.col(column).is_not_null() & (pl.col(column).str.strip_chars() != "")
    return pl.col(column).is_not_null()


def _count_mask(frame: pl.DataFrame, mask: pl.Expr, *, n_row: int, n_id: int) -> CountStat:
    """Affected rows and insured for ``mask``, as a fraction of the whole table."""
    sub = frame.select(mask.fill_null(False).alias("m"), pl.col("id"))
    nr = int(sub.select(pl.col("m").sum()).item())
    ni = int(sub.filter("m")["id"].n_unique())
    return {"n_row": nr, "n_id": ni,
            "pct_row": round(100 * nr / n_row, 3),
            "pct_id": round(100 * ni / n_id, 3) if n_id else 0.0}


def _count_within(frame: pl.DataFrame, mask: pl.Expr, base: pl.Expr) -> CountStat:
    """Affected rows and insured for ``mask``, as a fraction of the ``base`` subset."""
    sub = frame.select(mask.fill_null(False).alias("m"), base.fill_null(False).alias("b"), pl.col("id"))
    hit = sub.filter(pl.col("m") & pl.col("b"))
    base_rows = sub.filter("b")
    nr = hit.height
    ni = hit["id"].n_unique()
    base_nr = max(base_rows.height, 1)
    base_ni = max(base_rows["id"].n_unique(), 1)
    return {"n_row": nr, "n_id": ni,
            "pct_row": round(100 * nr / base_nr, 3),
            "pct_id": round(100 * ni / base_ni, 3)}


def _id_consistency(frame: pl.DataFrame, has: set[str]) -> dict[str, float]:
    per_id = frame.group_by("id").agg(
        n_gender=pl.col("gender").n_unique() if "gender" in has else pl.lit(None),
        n_age=pl.col("age").n_unique() if "age" in has else pl.lit(None),
        n_inq=pl.col("inq_date").n_unique() if "inq_date" in has else pl.lit(None),
        age_span=(pl.col("age").max() - pl.col("age").min()) if "age" in has else pl.lit(None),
        rows=pl.len(),
    )
    return {
        "rows_per_id_median": float(per_id["rows"].median()),
        "multi_gender_ids": int((per_id["n_gender"] > 1).sum()) if "gender" in has else 0,
        "multi_age_ids": int((per_id["n_age"] > 1).sum()) if "age" in has else 0,
        "age_span_ge2_ids": int((per_id["age_span"] >= 2).sum()) if "age" in has else 0,
        "multi_inq_ids": int((per_id["n_inq"] > 1).sum()) if "inq_date" in has else 0,
        "multi_age_single_inq": int(((per_id["n_age"] > 1) & (per_id["n_inq"] == 1)).sum())
        if {"age", "inq_date"} <= has else 0,
    }


def _date_chronology(frame: pl.DataFrame, has: set[str], *, n_row: int, n_id: int) -> dict[str, CountStat] | None:
    if not all(c in has for c in _DATE_COLUMNS):
        return None
    chain = ["_acc_date", "_sdate", "_edate", "_pay_date", "_inq_date"]
    labels = list(_DATE_COLUMNS)
    return {
        f"{labels[i]} > {labels[i + 1]}":
        _count_mask(frame, pl.col(chain[i]) > pl.col(chain[i + 1]), n_row=n_row, n_id=n_id)
        for i in range(4)
    }


def _row_anomaly(
    frame: pl.DataFrame, has: set[str], original_columns: list[str],
    *, have_span: bool, n_row: int, n_id: int,
) -> dict[str, CountStat]:
    def mask(expr: pl.Expr) -> dict:
        return _count_mask(frame, expr, n_row=n_row, n_id=n_id)

    anomaly: dict = {}
    if {"hos_day", "sdate"} <= has:
        anomaly["hos_day>0 & no sdate"] = mask((pl.col("hos_day") > 0) & ~_present("sdate", frame))
    if {"sdate", "edate"} <= has:
        anomaly["edate but no sdate"] = mask(~_present("sdate", frame) & _present("edate", frame))
    if have_span and "hos_day" in has:
        is_hosp = (pl.col("hos_day") > 0) & pl.col("_span").is_not_null()
        anomaly["date-span > hos_day"] = mask(is_hosp & (pl.col("_span") > pl.col("hos_day")))
        anomaly["date-span < hos_day"] = mask(is_hosp & (pl.col("_span") < pl.col("hos_day")))
    if {"hos_day", "hos_cnt"} <= has:
        anomaly["hos_day < hos_cnt"] = mask(pl.col("hos_day") < pl.col("hos_cnt"))
    # R's duplicated(): the 2nd+ occurrence of a row (extras), not the whole group
    anomaly["duplicate rows"] = mask(~pl.struct(original_columns).is_first_distinct())
    return anomaly


def _hos_day_span(frame: pl.DataFrame, has: set[str], *, have_span: bool, n_row: int, n_id: int) -> dict[str, CountStat] | None:
    if not (have_span and "hos_day" in has):
        return None
    both = _present("sdate", frame) & _present("edate", frame) & (pl.col("hos_day") > 0) & pl.col("_span").is_not_null()
    return {
        "n_base": _count_mask(frame, both, n_row=n_row, n_id=n_id),
        "match": _count_within(frame, pl.col("_span") == pl.col("hos_day"), both),
        "span_gt": _count_within(frame, pl.col("_span") > pl.col("hos_day"), both),
        "span_lt": _count_within(frame, pl.col("_span") < pl.col("hos_day"), both),
    }


def _gender_dist(frame: pl.DataFrame, has: set[str]) -> dict[str, int] | None:
    return dict(frame["gender"].value_counts().iter_rows()) if "gender" in has else None


def _age_stat(frame: pl.DataFrame, has: set[str]) -> dict[str, int | None] | None:
    if "age" not in has:
        return None
    return {
        "min": frame["age"].min(), "max": frame["age"].max(),
        "n_zero": int((frame["age"] == 0).sum()), "n_na": int(frame["age"].null_count()),
    }


def _missing_required(frame: pl.DataFrame, has: set[str]) -> dict[str, int]:
    required = [c for c in ("id", "gender", "age", "inq_date", "acc_date", "pay_date", "kcd0") if c in has]
    missing_required = {}
    for c in required:
        if frame.schema[c] == pl.Utf8:
            miss = int(frame.select((pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")).sum()).item())
        else:
            miss = int(frame[c].null_count())
        if miss:
            missing_required[c] = miss
    return missing_required


def _no_diagnosis(frame: pl.DataFrame, has: set[str], *, n_row: int, n_id: int) -> dict[str, CountStat | int] | None:
    kcd_cols = [c for c in _KCD_COLUMNS if c in has]
    if not kcd_cols:
        return None
    no_code = pl.all_horizontal([~_present(c, frame) for c in kcd_cols])
    tx_cols = [c for c in ("hos_day", "sur_cnt", "hos_cnt") if c in has]
    has_tx = pl.any_horizontal([(pl.col(c).is_not_null() & (pl.col(c) > 0)) for c in tx_cols]) if tx_cols else pl.lit(False)
    has_visit = _present("acc_date", frame) if "acc_date" in has else pl.lit(False)
    coded_ids = frame.filter(~no_code.fill_null(False))["id"].n_unique()
    return {
        "all_empty": _count_mask(frame, no_code, n_row=n_row, n_id=n_id),
        "inpatient": _count_within(frame, has_tx, no_code),
        "outpatient": _count_within(frame, ~has_tx & has_visit, no_code),
        "empty": _count_within(frame, ~has_tx & ~has_visit, no_code),
        "all_empty_ids": n_id - coded_ids,
    }


def _scope(frame: pl.DataFrame, has: set[str], *, n_row: int, n_id: int) -> dict[str, CountStat | int] | None:
    if not ({"kcd_main", "review", "in_lookback", "in_5yr"} <= has):
        return None
    coded = pl.col("kcd_main") != Sentinel.VACANT.value
    reviewed = pl.col("review") == 1
    windowed = (
        (pl.col("in_lookback") == 1)
        | (pl.col("kcd_main").is_in([Sentinel.IRREGULAR.value, Sentinel.UNMAPPED.value]) & (pl.col("in_5yr") == 1))
    ).fill_null(False)
    underwritable = reviewed & coded & windowed
    all_ids = set(frame["id"].unique().to_list())
    uw_ids = set(frame.filter(underwritable)["id"].unique().to_list())
    coded_ids = set(frame.filter(coded)["id"].unique().to_list())
    no_scope = all_ids - uw_ids
    never_coded = all_ids - coded_ids
    return {
        "not_reviewed": _count_mask(frame, ~reviewed, n_row=n_row, n_id=n_id),
        "out_of_window": _count_mask(frame, reviewed & coded & ~windowed, n_row=n_row, n_id=n_id),
        "no_scope_ids": len(no_scope),
        "never_coded_ids": len(never_coded),
        "aged_out_ids": len(no_scope - never_coded),
    }


def diagnose_icis(frame_like: FrameLike) -> IcisDiagnosis:
    """Data-quality report for a raw, cleansed, or mapped ICIS claim table.

    Each section is scored independently against the whole table; aggregates only,
    no raw rows are printed. Sections that need absent columns return ``None``.
    """
    frame, _ = to_polars(frame_like)
    if frame.height == 0:
        raise InputError("`frame_like` has no rows to diagnose.")
    n_row = frame.height
    n_id = frame["id"].n_unique()
    original_columns = frame.columns  # for the duplicate check, before augmenting
    has = set(frame.columns)

    date_cols = [c for c in _DATE_COLUMNS if c in has]
    frame = frame.with_columns([_parse_date(c).alias(f"_{c}") for c in date_cols])
    have_span = "sdate" in has and "edate" in has
    if have_span:
        frame = frame.with_columns(
            _span=(pl.col("_edate") - pl.col("_sdate")).dt.total_days() + 1
        )

    return IcisDiagnosis(
        n_row=n_row,
        n_id=n_id,
        id_consistency=_id_consistency(frame, has),
        date_chronology=_date_chronology(frame, has, n_row=n_row, n_id=n_id),
        row_anomaly=_row_anomaly(frame, has, original_columns, have_span=have_span, n_row=n_row, n_id=n_id),
        hos_day_span=_hos_day_span(frame, has, have_span=have_span, n_row=n_row, n_id=n_id),
        gender_dist=_gender_dist(frame, has),
        age_stat=_age_stat(frame, has),
        missing_required=_missing_required(frame, has),
        no_diagnosis=_no_diagnosis(frame, has, n_row=n_row, n_id=n_id),
        scope=_scope(frame, has, n_row=n_row, n_id=n_id),
    )
