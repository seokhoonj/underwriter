"""The ``Underwriter`` -- the configure-then-apply engine.

Built once from a :class:`Rulebook` (validated rules + sentinels + decision
tables), then applied to many books of aggregated inputs via ``match`` (attach
a rule) and, in Phase 2, ``decide`` (compose the final per-insured decision).
The lossratio ``RegimeDetector(...).detect(tri)`` shape.
"""

from __future__ import annotations

import polars as pl

from .._kernels.band import band_match
from .._kernels.combine import combine_decisions
from .._kernels.io import require_columns, to_polars
from .._kernels.tokens import Grammar
from ..pipeline.aggregate import aggregate_disease
from ..pipeline.clean import clean_icis, filter_latest_inquiry
from ..pipeline.map import map_disease
from ..pipeline.melt import melt_kcd
from ..rulebook.rulebook import Rulebook
from .decision import Decision
from .match import MatchResult

_BAND_INPUTS = ("kcd_main", "age", "elp_day", "sur_cnt", "hos_day")


class Underwriter:
    """Apply a rulebook to aggregated underwriting inputs."""

    def __init__(self, rulebook: Rulebook) -> None:
        self._rulebook = rulebook
        # the engine only ever applies no-declaration rules
        self._auto_rules = rulebook.ruleset.filter(pl.col("decl_yn") == 0)
        self._decision_columns = list(rulebook.decision_columns)
        self._grammar = Grammar.from_tables(rulebook.decision, rulebook.loading)
        self._loading_bands = _validated_loading(rulebook.loading)

    @property
    def rulebook(self) -> Rulebook:
        return self._rulebook

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    def match(self, aggregated: object) -> MatchResult:
        """Attach each ``(id, kcd_main)`` input to the rule it falls in. Unmatched
        inputs are kept (``matched == 0``) so they can be referred to the
        underwriter -- no insured is dropped."""
        frame, _ = to_polars(aggregated)
        require_columns(frame, _BAND_INPUTS, where="Underwriter.match")
        bm = band_match(frame, self._auto_rules, self._decision_columns)
        return MatchResult(
            applied=bm.applied,
            unmatched=bm.unmatched,
            multi_matched=bm.multi_matched,
            conflict=bm.conflict,
            decision_columns=tuple(self._decision_columns),
            n_multi_matched=bm.n_multi_matched,
            n_conflict=bm.n_conflict,
        )

    def decide(self, match_result: MatchResult) -> Decision:
        """Collapse the per-disease decisions into one final decision per insured
        (one row per id, one column per coverage)."""
        return self.recombine(match_result.applied)

    def recombine(self, applied: object) -> Decision:
        """Run the decision composition on an ``applied`` table -- the same step as
        :meth:`decide`, exposed for what-if relaxation that edits ``applied`` and
        re-combines."""
        frame, _ = to_polars(applied)
        result = combine_decisions(
            frame,
            self._decision_columns,
            self._grammar,
            self._rulebook.exclusion,
            self._rulebook.reduction,
            self._loading_bands,
        )
        return Decision(
            combined=result.combined,
            cells=result.cells,
            decision_columns=tuple(self._decision_columns),
            auto_by_code=self._grammar.auto_by_code,
            unresolved=result.unresolved,
        )

    def underwrite(self, aggregated: object) -> Decision:
        """The whole engine: ``match`` then ``decide``."""
        return self.decide(self.match(aggregated))

    def run(self, claims: object) -> Decision:
        """The whole pipeline from raw claims to the final decision: cleanse,
        map (with the rulebook's disease table), aggregate, then underwrite.

        The same call serves a batch simulation (pass the whole book of claims)
        and the production rule engine (pass one applicant's claim history) --
        every stage groups by insured, so one person's decision is identical to
        that person's row in the batch."""
        cleaned = filter_latest_inquiry(clean_icis(claims))
        aggregated = aggregate_disease(map_disease(melt_kcd(cleaned), self._rulebook.disease))
        return self.underwrite(aggregated)

    def __repr__(self) -> str:
        return (
            f"<Underwriter: {self._auto_rules.height:,} auto rules, "
            f"{len(self._decision_columns)} coverages>"
        )


def _validated_loading(loading: pl.DataFrame) -> pl.DataFrame:
    bands = loading.sort("at_least")
    if bands.height == 0 or bands["at_least"].null_count():
        raise ValueError("loading_table needs at least one band with a non-null `at_least`.")
    if bands["at_least"].n_unique() != bands.height:
        raise ValueError("loading_table has duplicate `at_least` bounds.")
    if int(bands["at_least"][0]) != 0:
        raise ValueError("loading_table's first `at_least` must be 0.")
    return bands
