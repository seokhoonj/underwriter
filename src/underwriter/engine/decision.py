"""``Decision`` -- the per-insured final decision, one row per id and one column
per coverage. Its methods only read/transform the result."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Decision:
    combined: pl.DataFrame          # one row per insured, one column per coverage
    cells: pl.DataFrame             # long (id, coverage, dec) audit trail
    decision_columns: tuple[str, ...]
    auto_by_code: dict[str, int]
    unresolved: pl.DataFrame | None

    def tabulate(self) -> pl.DataFrame:
        """Decision distribution per coverage: ``coverage, decision, category,
        auto, n, prop`` (proportions sum to 1 within each coverage)."""
        cats = (
            pl.col("decision")
            .str.split(",")
            .list.eval(pl.element().str.extract(r"^([A-Za-z]+)", 1))
            .list.unique()
            .list.sort()
        )
        return (
            self.combined.unpivot(
                index="id", on=list(self.decision_columns),
                variable_name="coverage", value_name="decision",
            )
            .filter(pl.col("decision").is_not_null() & (pl.col("decision") != ""))
            .group_by("coverage", "decision")
            .agg(n=pl.len())
            .with_columns(_cats=cats)
            .with_columns(
                category=pl.col("_cats").list.join(","),
                auto=pl.col("_cats")
                .list.eval(pl.element().replace_strict(self.auto_by_code, default=1))
                .list.min()
                .cast(pl.Int8),
            )
            .with_columns(prop=pl.col("n") / pl.col("n").sum().over("coverage"))
            .drop("_cats")
            .sort(["coverage", "n"], descending=[False, True])
        )

    def trace(self, insured_id: str) -> pl.DataFrame:
        """The per-coverage decision cells for one insured."""
        return self.cells.filter(pl.col("id") == insured_id).sort("coverage")

    def plot(self, **kwargs):
        """Stacked bar of the decision composition per coverage (needs
        matplotlib). Keyword args pass through to ``plot_decision``."""
        from .._plot.decision import plot_decision

        return plot_decision(self.tabulate(), **kwargs)

    def __repr__(self) -> str:
        return (
            f"<Decision: {self.combined.height:,} insured x "
            f"{len(self.decision_columns)} coverages>"
        )
