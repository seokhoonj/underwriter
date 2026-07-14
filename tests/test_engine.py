"""Engine end-to-end on a small inline rulebook: match -> decide, plus the
diagnostics and relaxation. (Full R-parity on the real rule set is checked
locally by refs/kdb/parity.py.)"""

from __future__ import annotations

import polars as pl
import pytest

import underwriter as uw
from underwriter.errors import RulebookError, UnderwriterError


def _rule(no, kcd_main, *, ord=1, decl_yn=0, age_band=(0, 999), elp_day_band=(0, 9999),
          sur_cnt_band=(0, 999), hos_day_band=(0, 9999), life="S", hoscov="S"):
    return {
        "no": no, "kcd_main": kcd_main, "ord": ord, "decl_yn": decl_yn,
        "age_min": age_band[0], "age_max": age_band[1],
        "elp_day_min": elp_day_band[0], "elp_day_max": elp_day_band[1],
        "sur_cnt_min": sur_cnt_band[0], "sur_cnt_max": sur_cnt_band[1],
        "hos_day_min": hos_day_band[0], "hos_day_max": hos_day_band[1],
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


def test_diagnose_ruleset_separates_genuine_and_shadow_explained_conflicts():
    # GN: two overlapping rules whose life decision differs -> a genuine conflict.
    # SH: also differ, but a shadow column (recover) explains it -> not genuine.
    decision = pl.DataFrame(
        {"priority": [1, 2, 5], "code": ["D", "U", "S"], "combiner": ["priority"] * 3,
         "role": ["decline", "underwriter", "standard"], "auto": [1, 0, 1], "max_sites": [None] * 3}
    )
    disease = pl.DataFrame(
        {"kcd": ["GN", "SH"], "kcd_main": ["GN", "SH"], "sub_chk": [1, 1], "lookback_mon": [60, 60]}
    )

    def rule(no, kcd_main, life, age_band, recover=""):
        return {
            "no": no, "kcd_main": kcd_main, "ord": no, "decl_yn": 0,
            "age_min": age_band[0], "age_max": age_band[1], "elp_day_min": 0, "elp_day_max": 9999,
            "sur_cnt_min": 0, "sur_cnt_max": 999, "hos_day_min": 0, "hos_day_max": 9999,
            "out_day_min": 0, "out_day_max": 9999, "recover": recover, "life": life, "hos": "S",
        }

    ruleset = pl.DataFrame([
        rule(1, "GN", "S", (0, 50)), rule(2, "GN", "U", (40, 99)),
        rule(3, "SH", "S", (0, 50), recover="a"), rule(4, "SH", "U", (40, 99), recover="b"),
    ])
    sentinel = pl.DataFrame(
        [{k: v for k, v in rule(0, s, "S", (0, 999)).items() if k != "no"}
         for s in ("VACANT", "EXPIRED", "IRREGULAR", "UNMAPPED")]
    )
    book = uw.Rulebook.from_frames(
        disease=disease, ruleset=ruleset, sentinel=sentinel, decision=decision,
        exclusion=pl.DataFrame({"mark": ["3"]}), reduction=pl.DataFrame({"mark": ["3"]}),
        loading=pl.DataFrame({"at_least": [0], "decision": ["S"]}),
    )
    conflict = uw.diagnose_ruleset(book).latent_conflict
    assert conflict == {"n_pair": 2, "n_kcd": 2, "n_genuine": 1}


def test_run_returns_one_row_per_raw_insured():
    # the whole pipeline (clean -> map -> aggregate -> underwrite) must preserve
    # every insured exactly once, even when most codes fall through to UNMAPPED.
    claims = uw.make_icis(80, seed=7)
    dec = uw.Underwriter(_mini_rulebook()).run(claims)
    assert dec.combined.height == claims["id"].n_unique()


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


def test_diagnose_icis_counts_duplicates_and_no_diagnosis_rows():
    rows = [
        {"id": "1", "gender": "1", "age": 40, "inq_date": "20240601", "pay_date": "20240110",
         "acc_date": "20240101", "sdate": None, "edate": None, "hos_day": 0, "hos_cnt": 0,
         "sur_cnt": 0, "kcd0": "A00", "kcd1": None, "kcd2": None, "kcd3": None, "kcd4": None},
    ]
    rows.append(dict(rows[0]))                       # an exact duplicate row
    rows.append({**rows[0], "id": "2", "kcd0": None})  # a codeless line -> no diagnosis
    report = uw.diagnose_icis(pl.DataFrame(rows))
    assert (report.n_row, report.n_id) == (3, 2)
    assert report.row_anomaly["duplicate rows"]["n_row"] == 1
    assert report.no_diagnosis["all_empty"]["n_row"] == 1
    assert report.no_diagnosis["all_empty_ids"] == 1


def test_rulebook_error_is_an_underwriter_error_and_a_value_error():
    # domain errors stay catchable as ValueError (back-compat) and as UnderwriterError
    with pytest.raises(UnderwriterError):
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
    assert issubclass(RulebookError, ValueError)


def test_public_input_guards_raise_underwriter_error():
    # every public entry point's argument guard is catchable as UnderwriterError
    book = _mini_rulebook()
    agg = _aggregated([{"id": "2", "kcd_main": "A00"}])
    match = uw.Underwriter(book).match(agg)
    with pytest.raises(UnderwriterError):
        uw.relax_rule(uw.Underwriter(book), match, [])          # no patterns
    with pytest.raises(UnderwriterError):
        uw.diagnose_icis(pl.DataFrame({"id": []}))              # empty frame
    with pytest.raises(ValueError):                             # still a ValueError too
        uw.diagnose_icis(pl.DataFrame({"id": []}))


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
