"""``MatchResult`` -- the frozen, named result of matching aggregated inputs
against a rule set. Replaces the R ``match_rule`` untyped list of five tables;
counts are derived properties that cannot drift from the frames."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class MatchResult:
    applied: pl.DataFrame        # inputs + matched/conflict/no/ord + one column per coverage
    unmatched: pl.DataFrame      # inputs no rule matched
    multi_matched: pl.DataFrame  # each multi-matched input joined to every rule it hit
    conflict: pl.DataFrame       # the multi-matched rows whose decisions disagree
    decision_columns: tuple[str, ...]
    n_multi_matched: int
    n_conflict: int

    @property
    def n_matched(self) -> int:
        return int(self.applied.select((pl.col("matched") == 1).sum()).item())

    @property
    def n_unmatched(self) -> int:
        return self.unmatched.height

    def __repr__(self) -> str:
        return (
            f"<MatchResult: {self.applied.height:,} inputs, "
            f"unmatched {self.n_unmatched:,}, multi {self.n_multi_matched:,}, "
            f"conflict {self.n_conflict:,}>"
        )
