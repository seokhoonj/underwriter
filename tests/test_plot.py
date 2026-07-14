"""Plots render without error (headless Agg backend)."""

from __future__ import annotations

import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import underwriter as uw
from test_engine import _aggregated, _mini_rulebook


def _decision():
    book = _mini_rulebook()
    agg = _aggregated([{"id": "1", "kcd_main": "M51"}, {"id": "2", "kcd_main": "A00"}])
    eng = uw.Underwriter(book)
    return eng, eng.match(agg)


def test_decision_plot_renders():
    eng, m = _decision()
    fig = eng.decide(m).plot()
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_rule_impact_plot_renders():
    eng, m = _decision()
    impact = uw.list_rule_impact(eng, m, coverage="life")
    if impact.height:
        fig = uw.plot_rule_impact(impact, coverage="life")
        assert fig is not None
        matplotlib.pyplot.close(fig)


def test_relaxed_rule_plot_renders():
    eng, m = _decision()
    relaxed = uw.relax_rule(eng, m, "A00")
    fig = uw.plot_relaxed_rule(relaxed, disease="A00")
    assert fig is not None
    matplotlib.pyplot.close(fig)
