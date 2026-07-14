"""underwriter -- automated underwriting from insurance claim history.

A polars pipeline that cleanses claim history, maps diagnosis codes to
representative diseases, aggregates per-insured underwriting inputs, matches
them against a rule set, and combines the per-disease decisions into a final
per-insured decision. The engine is data-agnostic: the rulebook is supplied at
run time, so it is not tied to any one insurer.

See also
--------
Sibling R package: https://github.com/seokhoonj/underwriter-r
"""

from __future__ import annotations

from .datasets import make_disease_table, make_icis
from .diagnostics.icis import IcisDiagnosis, diagnose_icis
from .diagnostics.relaxation import decompose_rule_impact, list_rule_impact, relax_rule
from .diagnostics.ruleset import RulesetDiagnosis, diagnose_ruleset
from ._plot.relaxation import plot_relaxed_rule, plot_rule_impact
from .engine.decision import Decision
from .engine.match import MatchResult
from .engine.underwriter import Underwriter
from .errors import (
    FrameTypeError,
    GrammarError,
    InputError,
    MissingColumnsError,
    RulebookError,
    UnderwriterError,
)
from .pipeline.aggregate import aggregate_disease
from .pipeline.clean import clean_icis, filter_latest_inquiry
from .pipeline.map import map_disease
from .pipeline.melt import melt_kcd
from .rulebook.rulebook import Rulebook
from .sentinels import SENTINELS, Sentinel

__version__ = "0.1.0"

__all__ = [
    "SENTINELS",
    "Decision",
    "FrameTypeError",
    "GrammarError",
    "IcisDiagnosis",
    "InputError",
    "MatchResult",
    "MissingColumnsError",
    "Rulebook",
    "RulebookError",
    "RulesetDiagnosis",
    "Sentinel",
    "Underwriter",
    "UnderwriterError",
    "__version__",
    "aggregate_disease",
    "clean_icis",
    "decompose_rule_impact",
    "diagnose_icis",
    "diagnose_ruleset",
    "filter_latest_inquiry",
    "list_rule_impact",
    "make_disease_table",
    "make_icis",
    "map_disease",
    "melt_kcd",
    "plot_relaxed_rule",
    "plot_rule_impact",
    "relax_rule",
]
