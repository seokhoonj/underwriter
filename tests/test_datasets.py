"""The synthetic example data drives the front half and loses no insured."""

from __future__ import annotations

import underwriter as uw


def test_make_icis_is_deterministic():
    a = uw.make_icis(n_insured=50, seed=1)
    b = uw.make_icis(n_insured=50, seed=1)
    assert a.equals(b)
    assert uw.make_icis(n_insured=50, seed=2).equals(a) is False


def test_synthetic_front_half_loses_no_insured():
    claims = uw.make_icis(n_insured=200, seed=0)
    disease = uw.make_disease_table()
    agg = uw.aggregate_disease(
        uw.map_disease(uw.melt_kcd(uw.filter_latest_inquiry(uw.clean_icis(claims))), disease)
    )
    assert agg["id"].n_unique() == claims["id"].n_unique() == 200


def test_synthetic_exercises_the_sentinels():
    claims = uw.make_icis(n_insured=400, seed=0)
    disease = uw.make_disease_table()
    agg = uw.aggregate_disease(
        uw.map_disease(uw.melt_kcd(uw.filter_latest_inquiry(uw.clean_icis(claims))), disease)
    )
    seen = set(agg["kcd_main"].unique().to_list())
    # the generator produces codeless, unreadable, and aged-out histories
    assert {"IRREGULAR"} & seen
    assert "M51" in seen  # and real diseases
