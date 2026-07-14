"""The ``Rulebook`` -- the Python-native replacement for the xlsx workbook.

A frozen bundle of the seven rule tables, validated and assembled once. The
engine is data-agnostic: a rulebook is runtime data, so it is not tied to any one
insurer. ``combine_ruleset`` (appending the sentinel catch-all rows below the
disease rules and renumbering) is absorbed into construction, so the engine can
never run on a half-assembled rule set.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .._kernels.io import to_polars

# rule-set columns that are keys / bands / declaration attributes; every other
# column is a per-coverage decision output.
_NON_DECISION_COLUMNS = frozenset(
    {
        "no", "kcd_main", "kcd_main_ko", "n", "ord", "decl_yn",
        "age_min", "age_max", "elp_day_min", "elp_day_max",
        "sur_cnt_min", "sur_cnt_max", "hos_day_min", "hos_day_max",
        "out_day_min", "out_day_max",
        "recover", "recur", "treat", "severe", "cause", "medical_checkup",
    }
)

_SHEETS = ("disease", "ruleset", "sentinel", "decision", "exclusion", "reduction", "loading")


def combine_ruleset(ruleset: pl.DataFrame, sentinel: pl.DataFrame | None) -> pl.DataFrame:
    """Append the sentinel catch-all rows below the disease rules, continuing the
    rule number as ``max(no) + 1, 2, ...`` (a sentinel's own ``no`` is not
    trusted). Kept apart in authoring so re-writing the disease rules cannot drop
    the sentinels."""
    if sentinel is None or sentinel.height == 0:
        return ruleset
    start = int(ruleset["no"].max())
    sent = sentinel.drop("no") if "no" in sentinel.columns else sentinel
    sent = sent.with_columns(no=pl.int_range(1, sent.height + 1) + start)
    return pl.concat([ruleset, sent], how="diagonal_relaxed")


@dataclass(frozen=True)
class Rulebook:
    """The seven assembled rule tables plus the derived coverage-column list."""

    disease: pl.DataFrame
    ruleset: pl.DataFrame  # disease rules + sentinels, engine-ready
    decision: pl.DataFrame
    exclusion: pl.DataFrame
    reduction: pl.DataFrame
    loading: pl.DataFrame
    decision_columns: tuple[str, ...]

    @classmethod
    def from_frames(
        cls,
        *,
        disease: object,
        ruleset: object,
        decision: object,
        exclusion: object,
        reduction: object,
        loading: object,
        sentinel: object | None = None,
    ) -> "Rulebook":
        disease = to_polars(disease)[0]
        ruleset = to_polars(ruleset)[0]
        decision = to_polars(decision)[0]
        exclusion = to_polars(exclusion)[0]
        reduction = to_polars(reduction)[0]
        loading = to_polars(loading)[0]
        sentinel = to_polars(sentinel)[0] if sentinel is not None else None

        _validate_disease(disease)
        combined = combine_ruleset(ruleset, sentinel)
        decision_columns = tuple(
            c for c in combined.columns if c not in _NON_DECISION_COLUMNS
        )
        return cls(
            disease=disease,
            ruleset=combined,
            decision=decision,
            exclusion=exclusion,
            reduction=reduction,
            loading=loading,
            decision_columns=decision_columns,
        )

    @classmethod
    def from_excel(cls, path: str) -> "Rulebook":
        """Load a rulebook from an xlsx workbook with the seven named sheets
        (``disease``, ``ruleset``, ``sentinel``, ``decision``, ``exclusion``,
        ``reduction``, ``loading``). Storage-format compatibility only -- the
        canonical constructor is :meth:`from_frames`."""
        sheets = {s: pl.read_excel(path, sheet_name=s) for s in _SHEETS}
        return cls.from_frames(
            disease=sheets["disease"],
            ruleset=sheets["ruleset"],
            sentinel=sheets["sentinel"],
            decision=sheets["decision"],
            exclusion=sheets["exclusion"],
            reduction=sheets["reduction"],
            loading=sheets["loading"],
        )

    def __repr__(self) -> str:
        return (
            f"<Rulebook: {self.disease.height:,} disease rows, "
            f"{self.ruleset.height:,} rules, {len(self.decision_columns)} coverages>"
        )


def _validate_disease(disease: pl.DataFrame) -> None:
    required = {"kcd", "kcd_main", "sub_chk", "lookback_mon"}
    missing = required - set(disease.columns)
    if missing:
        raise ValueError(f"disease table missing column(s): {sorted(missing)}.")
    key = disease.filter(pl.col("kcd").is_not_null())["kcd"]
    if key.n_unique() != key.len():
        dups = key.filter(key.is_duplicated()).unique().head(5).to_list()
        raise ValueError(f"disease table has duplicate `kcd` keys: {dups}.")
