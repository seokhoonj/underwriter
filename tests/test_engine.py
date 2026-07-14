"""Engine end-to-end on a small inline rulebook: match -> decide, plus the
diagnostics and relaxation. (Full R-parity on the real rule set is checked
locally by refs/kdb/parity.py.)"""

from __future__ import annotations

import polars as pl
import pytest

import underwriter as uw


def _rule(no, kcd_main, *, ord=1, decl_yn=0, age=(0, 999), elp=(0, 9999), sur=(0, 999),
          hos=(0, 9999), life="S", hoscov="S"):
    return {
        "no": no, "kcd_main": kcd_main, "ord": ord, "decl_yn": decl_yn,
        "age_min": age[0], "age_max": age[1], "elp_day_min": elp[0], "elp_day_max": elp[1],
        "sur_cnt_min": sur[0], "sur_cnt_max": sur[1], "hos_day_min": hos[0], "hos_day_max": hos[1],
        "out_day_min": 0, "out_day_max": 9999, "life": life, "hos": hoscov,
    }


def _mini_rulebook() -> uw.Rulebook:
    decision = pl.DataFrame(
        {
            "priority": [1, 2, 3, 4, 4, 4, 4, 5],
            "code": ["D", "U", "C", "M", "E", "L", "R", "S"],
            "combiner": ["priority", "priority", "priority", "priority", "loading", "reduction", "exclusion", "priority"],
            "role": ["decline", "underwriter", None, None, None, None, None, "standard"],
            "auto": [1, 0, 1, 1, 1, 1, 1, 1],
            "max_sites": [None, None, None, None, None, None, 4, None],
        }
    )
    disease = pl.DataFrame(
        {"kcd": ["M51", "A00"], "kcd_main": ["M51", "A00"], "sub_chk": [1, 1], "lookback_mon": [60, 12]}
    )
    ruleset = pl.DataFrame(
        [
            _rule(1, "M51", life="S", hoscov="D"),          # decline on hos
            _rule(2, "A00", life="U", hoscov="S"),          # refer life
            _rule(3, "E11", life="R03(3)", hoscov="S"),     # 3yr site exclusion on life
        ]
    )
    sentinel = pl.DataFrame(
        [
            {k: v for k, v in _rule(0, s, life=life, hoscov=life).items() if k != "no"}
            for s, life in [("VACANT", "S"), ("EXPIRED", "S"), ("IRREGULAR", "U"), ("UNMAPPED", "U")]
        ]
    )
    exclusion = pl.DataFrame({"code": ["01", "02"], "mark": ["3", "5i"], "desc": ["3yr", "5yr-elapsed"]})
    reduction = pl.DataFrame({"mark": ["3", "5i"], "period": ["3yr", "5yr-elapsed"]})
    loading = pl.DataFrame({"at_least": [0, 50, 201], "decision": ["S", "U", "D"]})
    return uw.Rulebook.from_frames(
        disease=disease, ruleset=ruleset, sentinel=sentinel, decision=decision,
        exclusion=exclusion, reduction=reduction, loading=loading,
    )


def _aggregated(rows):
    base = {"age": 40, "hos_day": 0, "sur_cnt": 0, "out_cnt": 0,
            "hos_elp_day": None, "sur_elp_day": None, "out_elp_day": None, "elp_day": 100}
    return pl.DataFrame([{**base, **r} for r in rows])


def test_underwrite_one_row_per_insured_and_decisions():
    book = _mini_rulebook()
    agg = _aggregated([
        {"id": "1", "kcd_main": "M51"},   # -> hos decline
        {"id": "2", "kcd_main": "A00"},   # -> life refer
    ])
    dec = uw.Underwriter(book).underwrite(agg)
    assert dec.combined.height == 2
    row1 = dec.combined.filter(pl.col("id") == "1")
    assert row1.select(["life", "hos"]).row(0) == ("S", "D")
    row2 = dec.combined.filter(pl.col("id") == "2")
    assert row2.select(["life", "hos"]).row(0) == ("U", "S")


def test_exclusion_resolves_to_months():
    book = _mini_rulebook()
    agg = _aggregated([{"id": "3", "kcd_main": "E11", "elp_day": 100}])
    dec = uw.Underwriter(book).underwrite(agg)
    # R03(3) = site 03, 3-year fixed mark -> 36 months
    assert dec.combined.filter(pl.col("id") == "3")["life"].to_list() == ["R03(36)"]


def test_unmatched_is_referred_to_underwriter():
    book = _mini_rulebook()
    agg = _aggregated([{"id": "9", "kcd_main": "Z99"}])  # no rule
    m = uw.Underwriter(book).match(agg)
    assert m.n_unmatched == 1
    dec = uw.Underwriter(book).underwrite(agg)
    assert dec.combined.filter(pl.col("id") == "9").select(["life", "hos"]).row(0) == ("U", "U")


def test_diagnose_ruleset_is_clean_on_the_mini_rulebook():
    d = uw.diagnose_ruleset(_mini_rulebook())
    assert d.missing_sentinel["n_kcd"] == 0
    assert d.latent_conflict["n_pair"] == 0


def test_relax_rule_lifts_only_referrals():
    book = _mini_rulebook()
    agg = _aggregated([{"id": "2", "kcd_main": "A00"}])  # life referred
    eng = uw.Underwriter(book)
    m = eng.match(agg)
    out = uw.relax_rule(eng, m, "A00", coverage="life")
    assert out["lift"][0] > 0  # relaxing A00 lifts life's auto share


def test_rulebook_rejects_duplicate_disease_key():
    with pytest.raises(ValueError):
        uw.Rulebook.from_frames(
            disease=pl.DataFrame({"kcd": ["M51", "M51"], "kcd_main": ["M51", "M51"],
                                  "sub_chk": [1, 1], "lookback_mon": [60, 60]}),
            ruleset=pl.DataFrame([_rule(1, "M51")]),
            decision=pl.DataFrame({"priority": [2, 5], "code": ["U", "S"],
                                   "combiner": ["priority", "priority"],
                                   "role": ["underwriter", "standard"], "auto": [0, 1],
                                   "max_sites": [None, None]}),
            exclusion=pl.DataFrame({"mark": ["3"]}), reduction=pl.DataFrame({"mark": ["3"]}),
            loading=pl.DataFrame({"at_least": [0], "decision": ["S"]}),
        )
