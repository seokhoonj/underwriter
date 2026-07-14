"""The decision-code grammar's edges: the exclusion site cap, the elapsed-aware
``i`` mark (fixed 30-day months), and the loading-band join. These are the pieces
the small end-to-end rulebook in test_engine.py does not reach."""

from __future__ import annotations

import polars as pl

import underwriter as uw


def _rule(no, kcd_main, *, ord=1, decl_yn=0, age_band=(0, 999), elp_day_band=(0, 9999),
          sur_cnt_band=(0, 999), hos_day_band=(0, 9999), life="S", hos="S"):
    return {
        "no": no, "kcd_main": kcd_main, "ord": ord, "decl_yn": decl_yn,
        "age_min": age_band[0], "age_max": age_band[1],
        "elp_day_min": elp_day_band[0], "elp_day_max": elp_day_band[1],
        "sur_cnt_min": sur_cnt_band[0], "sur_cnt_max": sur_cnt_band[1],
        "hos_day_min": hos_day_band[0], "hos_day_max": hos_day_band[1],
        "out_day_min": 0, "out_day_max": 9999, "life": life, "hos": hos,
    }


def _grammar_rulebook() -> uw.Rulebook:
    decision = pl.DataFrame(
        {
            "priority": [1, 2, 3, 4, 4, 4, 4, 5],
            "code": ["D", "U", "C", "M", "E", "L", "R", "S"],
            "combiner": ["priority", "priority", "priority", "priority",
                         "loading", "reduction", "exclusion", "priority"],
            "role": ["decline", "underwriter", None, None, None, None, None, "standard"],
            "auto": [1, 0, 1, 1, 1, 1, 1, 1],
            "max_sites": [None, None, None, None, None, None, 4, None],
        }
    )
    codes = ["E1", "E2", "E3", "E4", "E5", "LD", "IM"]
    disease = pl.DataFrame(
        {"kcd": codes, "kcd_main": codes, "sub_chk": [1] * len(codes), "lookback_mon": [60] * len(codes)}
    )
    ruleset = pl.DataFrame(
        [
            _rule(1, "E1", life="R01(3)"), _rule(2, "E2", life="R02(3)"),
            _rule(3, "E3", life="R03(3)"), _rule(4, "E4", life="R04(3)"),
            _rule(5, "E5", life="R05(3)"),          # a 5th distinct exclusion site
            _rule(6, "LD", life="E(120)"),          # a loading index of 120
            _rule(7, "IM", life="R06(5i)"),         # an elapsed-aware exclusion mark
        ]
    )
    sentinel = pl.DataFrame(
        [{k: v for k, v in _rule(0, s, life="S").items() if k != "no"}
         for s in ("VACANT", "EXPIRED", "IRREGULAR", "UNMAPPED")]
    )
    exclusion = pl.DataFrame({"mark": ["3", "5i"], "desc": ["3yr", "5yr-elapsed"]})
    reduction = pl.DataFrame({"mark": ["3", "5i"], "period": ["3yr", "5yr-elapsed"]})
    loading = pl.DataFrame({"at_least": [0, 50, 201], "decision": ["S", "E", "D"]})
    return uw.Rulebook.from_frames(
        disease=disease, ruleset=ruleset, sentinel=sentinel, decision=decision,
        exclusion=exclusion, reduction=reduction, loading=loading,
    )


def _aggregated(rows):
    base = {"age": 40, "hos_day": 0, "sur_cnt": 0, "out_cnt": 0,
            "hos_elp_day": None, "sur_elp_day": None, "out_elp_day": None, "elp_day": 100}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_exclusion_sites_above_max_declines():
    engine = uw.Underwriter(_grammar_rulebook())
    agg = _aggregated([{"id": "MS", "kcd_main": k} for k in ("E1", "E2", "E3", "E4", "E5")])
    actual_life = engine.underwrite(agg).combined.filter(pl.col("id") == "MS")["life"].to_list()
    assert actual_life == ["D"]  # 5 sites > max_sites 4 -> decline


def test_exclusion_sites_at_max_are_kept():
    engine = uw.Underwriter(_grammar_rulebook())
    agg = _aggregated([{"id": "M4", "kcd_main": k} for k in ("E1", "E2", "E3", "E4")])
    actual_life = engine.underwrite(agg).combined.filter(pl.col("id") == "M4")["life"].to_list()
    assert actual_life == ["R01(36),R02(36),R03(36),R04(36)"]  # 4 sites == cap, all kept


def test_elapsed_mark_uses_truncated_30_day_months():
    engine = uw.Underwriter(_grammar_rulebook())
    agg = _aggregated([{"id": "IM", "kcd_main": "IM", "elp_day": 100}])
    actual_life = engine.underwrite(agg).combined.filter(pl.col("id") == "IM")["life"].to_list()
    # 100 // 30 = 3 months elapsed; "5i" -> 5*12 - 3 = 57 remaining months
    assert actual_life == ["R06(57)"]


def test_loading_total_selects_backward_band():
    engine = uw.Underwriter(_grammar_rulebook())
    agg = _aggregated([{"id": "LD", "kcd_main": "LD"}])
    actual_life = engine.underwrite(agg).combined.filter(pl.col("id") == "LD")["life"].to_list()
    # index 120 falls in the [50, 201) band, whose decision is the loading code E
    assert actual_life == ["E(120)"]
