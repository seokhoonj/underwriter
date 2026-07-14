"""Band-match behaviour, from the R match_rule contract, on inline fixtures.
(Full R-parity on real data is checked locally by refs/kdb/parity.py.)"""

from __future__ import annotations

import polars as pl

from underwriter._kernels.band import band_match

_DECISIONS = ["life", "adb"]


def _rule(no, kcd_main, *, ord=1, decl_yn=0, age_band=(0, 999), elp_day_band=(0, 9999),
          sur_cnt_band=(0, 999), hos_day_band=(0, 9999), life="S", adb="S"):
    return {
        "no": no, "kcd_main": kcd_main, "ord": ord, "decl_yn": decl_yn,
        "age_min": age_band[0], "age_max": age_band[1],
        "elp_day_min": elp_day_band[0], "elp_day_max": elp_day_band[1],
        "sur_cnt_min": sur_cnt_band[0], "sur_cnt_max": sur_cnt_band[1],
        "hos_day_min": hos_day_band[0], "hos_day_max": hos_day_band[1],
        "life": life, "adb": adb,
    }


def _input(id, kcd_main, *, age=40, elp_day=100, sur_cnt=0, hos_day=0):
    return {"id": id, "kcd_main": kcd_main, "age": age, "elp_day": elp_day,
            "sur_cnt": sur_cnt, "hos_day": hos_day}


def _match(inputs, rules):
    agg = pl.DataFrame(inputs)
    rs = pl.DataFrame(rules).filter(pl.col("decl_yn") == 0)
    return band_match(agg, rs, _DECISIONS)


def test_input_in_band_matches_and_carries_decisions():
    bm = _match([_input("1", "M51")], [_rule(10, "M51", life="D", adb="U")])
    assert bm.applied["matched"].to_list() == [1]
    assert bm.applied.select(["no", "life", "adb"]).row(0) == (10, "D", "U")


def test_out_of_band_is_unmatched_and_kept():
    bm = _match([_input("1", "M51", elp_day=50000)], [_rule(10, "M51")])
    assert bm.applied["matched"].to_list() == [0]
    assert bm.unmatched.height == 1              # kept, not dropped


def test_null_input_value_never_matches():
    bm = _match([_input("1", "M51", elp_day=None)], [_rule(10, "M51")])
    assert bm.applied["matched"].to_list() == [0]


def test_lowest_ord_wins_ties():
    bm = _match(
        [_input("1", "M51")],
        [_rule(10, "M51", ord=5, life="U"), _rule(11, "M51", ord=2, life="S")],
    )
    assert bm.applied.select(["no", "life"]).row(0) == (11, "S")  # ord 2 beats ord 5
    assert bm.n_multi_matched == 1


def test_conflict_flagged_when_matched_rules_disagree():
    bm = _match(
        [_input("1", "M51")],
        [_rule(10, "M51", ord=1, life="S"), _rule(11, "M51", ord=2, life="D")],
    )
    assert bm.n_multi_matched == 1
    assert bm.n_conflict == 1


def test_duplicate_rules_multi_but_no_conflict():
    bm = _match(
        [_input("1", "M51")],
        [_rule(10, "M51", ord=1, life="S"), _rule(11, "M51", ord=2, life="S")],
    )
    assert bm.n_multi_matched == 1
    assert bm.n_conflict == 0
