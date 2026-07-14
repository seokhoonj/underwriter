"""Data-quality report for an ICIS claim table (raw, cleansed, or mapped).

Per-id consistency, row-level anomalies, a value profile, the no-diagnosis split,
and -- on a mapped table -- the lookback scope. Every anomaly is counted as
affected rows and affected insured, with percentages. Aggregates only; no raw
rows are printed.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..sentinels import Sentinel

_DATE_COLUMNS = ("acc_date", "sdate", "edate", "pay_date", "inq_date")
_KCD_COLUMNS = ("kcd0", "kcd1", "kcd2", "kcd3", "kcd4")


@dataclass(frozen=True)
class IcisDiagnosis:
    n_row: int
    n_id: int
    id_consistency: dict
    date_chronology: dict | None
    row_anomaly: dict
    hos_day_span: dict | None
    gender_dist: dict | None
    age_stat: dict | None
    missing_required: dict
    no_diagnosis: dict | None
    scope: dict | None

    def __repr__(self) -> str:
        def cnt(v: dict) -> str:
            return f"n_row {v['n_row']:>8,} ({v['pct_row']:6.2f}%) | n_id {v['n_id']:>8,} ({v['pct_id']:6.2f}%)"

        L = [f"n_row={self.n_row:,} | n_id={self.n_id:,} | rows/id median={self.id_consistency['rows_per_id_median']}"]
        ic = self.id_consistency
        L.append("\n== id_consistency ==")
        L.append(f"  multi-gender ids : {ic['multi_gender_ids']:,}")
        L.append(f"  multi-age ids    : {ic['multi_age_ids']:,}  (age-span>=2yr: {ic['age_span_ge2_ids']:,})")
        L.append(f"  multi-inq ids    : {ic['multi_inq_ids']:,}")
        if self.date_chronology:
            L.append("\n== date_chronology (acc <= sdate <= edate <= pay <= inq) ==")
            for k, v in self.date_chronology.items():
                L.append(f"  {k:<20}: {cnt(v)}")
        L.append("\n== row_anomaly ==")
        for k, v in self.row_anomaly.items():
            if v is not None:
                L.append(f"  {k:<20}: {cnt(v)}")
        if self.no_diagnosis:
            nd = self.no_diagnosis
            L.append("\n== no diagnosis (all kcd empty; kept on the no-diagnosis code) ==")
            L.append(f"  all kcd empty        : {cnt(nd['all_empty'])}")
            L.append(f"    inpatient/surgery  : {cnt(nd['inpatient'])}")
            L.append(f"    outpatient (visit) : {cnt(nd['outpatient'])}")
            L.append(f"    empty (no tx/date) : {cnt(nd['empty'])}")
            L.append(f"  ids with no coded row: {nd['all_empty_ids']:,}")
        if self.scope:
            sc = self.scope
            L.append("\n== scope (which diagnoses the lookback windows admit) ==")
            L.append(f"  not reviewed             : {cnt(sc['not_reviewed'])}")
            L.append(f"  reviewed but out of window: {cnt(sc['out_of_window'])}")
            L.append(f"  ids with nothing in scope: {sc['no_scope_ids']:,}")
            L.append(f"    no diagnosis ever      : {sc['never_coded_ids']:,}  -> VACANT")
            L.append(f"    every diagnosis aged out: {sc['aged_out_ids']:,}  -> EXPIRED")
        return "\n".join(L)


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


def diagnose_icis(dt: object) -> IcisDiagnosis:
    from .._kernels.io import to_polars

    frame, _ = to_polars(dt)
    if frame.height == 0:
        raise ValueError("`dt` has no rows to diagnose.")
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

    def count_mask(mask: pl.Expr) -> dict:
        sub = frame.select(mask.fill_null(False).alias("m"), pl.col("id"))
        nr = int(sub.select(pl.col("m").sum()).item())
        ni = int(sub.filter("m")["id"].n_unique())
        return {"n_row": nr, "n_id": ni,
                "pct_row": round(100 * nr / n_row, 3),
                "pct_id": round(100 * ni / n_id, 3) if n_id else 0.0}

    def count_within(mask: pl.Expr, base: pl.Expr) -> dict:
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

    # --- id_consistency ---
    per_id = frame.group_by("id").agg(
        n_gender=pl.col("gender").n_unique() if "gender" in has else pl.lit(None),
        n_age=pl.col("age").n_unique() if "age" in has else pl.lit(None),
        n_inq=pl.col("inq_date").n_unique() if "inq_date" in has else pl.lit(None),
        age_span=(pl.col("age").max() - pl.col("age").min()) if "age" in has else pl.lit(None),
        rows=pl.len(),
    )
    id_consistency = {
        "rows_per_id_median": float(per_id["rows"].median()),
        "multi_gender_ids": int((per_id["n_gender"] > 1).sum()) if "gender" in has else 0,
        "multi_age_ids": int((per_id["n_age"] > 1).sum()) if "age" in has else 0,
        "age_span_ge2_ids": int((per_id["age_span"] >= 2).sum()) if "age" in has else 0,
        "multi_inq_ids": int((per_id["n_inq"] > 1).sum()) if "inq_date" in has else 0,
        "multi_age_single_inq": int(((per_id["n_age"] > 1) & (per_id["n_inq"] == 1)).sum())
        if {"age", "inq_date"} <= has else 0,
    }

    # --- date_chronology ---
    date_chronology = None
    if all(c in has for c in _DATE_COLUMNS):
        chain = ["_acc_date", "_sdate", "_edate", "_pay_date", "_inq_date"]
        labels = list(_DATE_COLUMNS)
        date_chronology = {
            f"{labels[i]} > {labels[i + 1]}": count_mask(pl.col(chain[i]) > pl.col(chain[i + 1]))
            for i in range(4)
        }

    # --- row_anomaly ---
    row_anomaly: dict = {}
    if {"hos_day", "sdate"} <= has:
        row_anomaly["hos_day>0 & no sdate"] = count_mask(
            (pl.col("hos_day") > 0) & ~_present("sdate", frame)
        )
    if {"sdate", "edate"} <= has:
        row_anomaly["edate but no sdate"] = count_mask(
            ~_present("sdate", frame) & _present("edate", frame)
        )
    if have_span and "hos_day" in has:
        is_hosp = (pl.col("hos_day") > 0) & pl.col("_span").is_not_null()
        row_anomaly["date-span > hos_day"] = count_mask(is_hosp & (pl.col("_span") > pl.col("hos_day")))
        row_anomaly["date-span < hos_day"] = count_mask(is_hosp & (pl.col("_span") < pl.col("hos_day")))
    if {"hos_day", "hos_cnt"} <= has:
        row_anomaly["hos_day < hos_cnt"] = count_mask(pl.col("hos_day") < pl.col("hos_cnt"))
    # R's duplicated(): the 2nd+ occurrence of a row (extras), not the whole group
    row_anomaly["duplicate rows"] = count_mask(~pl.struct(original_columns).is_first_distinct())

    # --- hos_day vs date-span ---
    hos_day_span = None
    if have_span and "hos_day" in has:
        both = _present("sdate", frame) & _present("edate", frame) & (pl.col("hos_day") > 0) & pl.col("_span").is_not_null()
        hos_day_span = {
            "n_base": count_mask(both),
            "match": count_within(pl.col("_span") == pl.col("hos_day"), both),
            "span_gt": count_within(pl.col("_span") > pl.col("hos_day"), both),
            "span_lt": count_within(pl.col("_span") < pl.col("hos_day"), both),
        }

    # --- value profile ---
    gender_dist = (
        dict(frame["gender"].value_counts().iter_rows()) if "gender" in has else None
    )
    age_stat = None
    if "age" in has:
        age_stat = {
            "min": frame["age"].min(), "max": frame["age"].max(),
            "n_zero": int((frame["age"] == 0).sum()), "n_na": int(frame["age"].null_count()),
        }
    required = [c for c in ("id", "gender", "age", "inq_date", "acc_date", "pay_date", "kcd0") if c in has]
    missing_required = {}
    for c in required:
        if frame.schema[c] == pl.Utf8:
            miss = int(frame.select((pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")).sum()).item())
        else:
            miss = int(frame[c].null_count())
        if miss:
            missing_required[c] = miss

    # --- no_diagnosis ---
    no_diagnosis = None
    kcd_cols = [c for c in _KCD_COLUMNS if c in has]
    if kcd_cols:
        no_code = pl.all_horizontal([~_present(c, frame) for c in kcd_cols])
        tx_cols = [c for c in ("hos_day", "sur_cnt", "hos_cnt") if c in has]
        has_tx = pl.any_horizontal([(pl.col(c).is_not_null() & (pl.col(c) > 0)) for c in tx_cols]) if tx_cols else pl.lit(False)
        has_visit = _present("acc_date", frame) if "acc_date" in has else pl.lit(False)
        coded_ids = frame.filter(~no_code.fill_null(False))["id"].n_unique()
        no_diagnosis = {
            "all_empty": count_mask(no_code),
            "inpatient": count_within(has_tx, no_code),
            "outpatient": count_within(~has_tx & has_visit, no_code),
            "empty": count_within(~has_tx & ~has_visit, no_code),
            "all_empty_ids": n_id - coded_ids,
        }

    # --- scope (mapped tables only) ---
    scope = None
    if {"kcd_main", "review", "in_lookback", "in_5yr"} <= has:
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
        scope = {
            "not_reviewed": count_mask(~reviewed),
            "out_of_window": count_mask(reviewed & coded & ~windowed),
            "no_scope_ids": len(no_scope),
            "never_coded_ids": len(never_coded),
            "aged_out_ids": len(no_scope - never_coded),
        }

    return IcisDiagnosis(
        n_row=n_row, n_id=n_id, id_consistency=id_consistency, date_chronology=date_chronology,
        row_anomaly=row_anomaly, hos_day_span=hos_day_span, gender_dist=gender_dist,
        age_stat=age_stat, missing_required=missing_required, no_diagnosis=no_diagnosis, scope=scope,
    )
