"""The decision-code grammar, compiled once from the decision table, plus the
period-mark arithmetic. Letter-agnostic: no decision code is hard-wired -- the
class letters and their roles are read off the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_COMBINERS = ("priority", "exclusion", "loading", "reduction")
_REGEX_METACHARACTERS = set(r".^$*+?()[]{}|\\")


@dataclass(frozen=True)
class Grammar:
    priority: dict[str, int]
    combiner: dict[str, str]
    exclusion: str | None
    loading: str | None
    reduction: str | None
    standard: str
    decline: str
    underwriter: str
    terminal: tuple[str, ...]
    sheet_ord: dict[str, int]
    max_sites: int | None
    auto_by_code: dict[str, int]

    @classmethod
    def from_tables(cls, decision: pl.DataFrame, loading: pl.DataFrame) -> "Grammar":
        codes = decision["code"].to_list()
        _check_codes(codes)
        combiner = dict(zip(codes, decision["combiner"].to_list()))
        unknown = sorted({m for m in combiner.values() if m not in _COMBINERS})
        if unknown:
            raise ValueError(f"decision combiners must be one of {_COMBINERS}; found {unknown}.")
        priority = dict(zip(codes, (int(p) for p in decision["priority"].to_list())))

        def by_combiner(name: str) -> str | None:
            for c, m in combiner.items():
                if m == name:
                    return c
            return None

        role = (
            dict(zip(codes, decision["role"].to_list()))
            if "role" in decision.columns
            else {}
        )

        def by_role(name: str) -> str | None:
            for c, r in role.items():
                if r == name:
                    return c
            return None

        standard = by_role("standard") or max(priority, key=priority.get)
        decline = by_role("decline") or min(priority, key=priority.get)
        underwriter = by_role("underwriter")
        if underwriter is None:
            raise ValueError(
                "decision table needs a row with role == 'underwriter'; "
                "unmatched diseases are referred there."
            )
        exclusion = by_combiner("exclusion")
        auto = (
            {c: int(a) for c, a in zip(codes, decision["auto"].to_list())}
            if "auto" in decision.columns
            else {}
        )
        return cls(
            priority=priority,
            combiner=combiner,
            exclusion=exclusion,
            loading=by_combiner("loading"),
            reduction=by_combiner("reduction"),
            standard=standard,
            decline=decline,
            underwriter=underwriter,
            terminal=tuple(dict.fromkeys([decline, underwriter])),
            sheet_ord={c: i + 1 for i, c in enumerate(codes)},
            max_sites=_max_sites(decision, exclusion),
            auto_by_code=auto,
        )


def _check_codes(codes: list[str]) -> None:
    wide = sorted({c for c in codes if len(str(c)) != 1})
    if wide:
        raise ValueError(f"decision codes must be one character; found {wide}.")
    meta = sorted({c for c in codes if c in _REGEX_METACHARACTERS})
    if meta:
        raise ValueError(f"decision codes must not be regex metacharacters; found {meta}.")
    dups = sorted({c for c in codes if codes.count(c) > 1})
    if dups:
        raise ValueError(f"decision table has duplicate codes: {dups}.")


def _max_sites(decision: pl.DataFrame, exclusion: str | None) -> int | None:
    if exclusion is None:
        return None
    if "max_sites" not in decision.columns:
        raise ValueError("decision table needs a `max_sites` column for the exclusion code.")
    cap = decision.filter(pl.col("code") == exclusion)["max_sites"].to_list()
    if not cap or cap[0] is None or int(cap[0]) < 1:
        raise ValueError(
            f"decision table needs a positive `max_sites` on exclusion code {exclusion!r}; "
            "write a large number for no cap."
        )
    return int(cap[0])


def unresolved_reason(
    code: str, grammar: Grammar, exclusion_marks: set[str], reduction_marks: set[str]
) -> str:
    """Empty string if the code is resolvable, else why it is not. Judged on the
    code text alone (independent of elapsed days / merging)."""
    import re

    letter = code[0]
    if letter not in grammar.priority:  # priority holds every declared code
        return f'class letter "{letter}" is not in decision table'
    method = grammar.combiner[letter]
    if method == "exclusion":
        for token in code.split(","):
            if not re.fullmatch(rf"{re.escape(letter)}[0-9]+\(.*\)", token):
                return f'"{token}" is not of the form {letter}<site>(<mark>)'
            mark = re.fullmatch(rf"{re.escape(letter)}[0-9]+\((.*)\)", token).group(1)
            if mark not in exclusion_marks:
                return f'mark "{mark}" is not in exclusion table'
    elif method == "reduction":
        m = re.fullmatch(rf"{re.escape(letter)}\((.*)\)", code)
        if not m:
            return f'"{code}" is not of the form {letter}(<mark>)'
        if m.group(1) not in reduction_marks:
            return f'mark "{m.group(1)}" is not in reduction table'
    elif method == "loading":
        if not re.fullmatch(rf"{re.escape(letter)}\([0-9]+\)", code):
            return f'"{code}" does not carry a numeric index'
    return ""  # priority payload has no syntax to check


def resolve_months(mark: pl.Expr, elp_day: pl.Expr, valid_marks: list[str]) -> pl.Expr:
    """A period mark ("5i" = 5yr minus elapsed, "3" = 3yr fixed, "99" = whole
    period) to a remaining-month count. <= 0 means expired; null means invalid.
    30 days = 1 month, truncated."""
    elapsed = (elp_day // 30).cast(pl.Int64)
    base = mark.str.replace("i", "", literal=True).cast(pl.Int64, strict=False)
    months = pl.when(mark.str.ends_with("i")).then(base * 12 - elapsed).otherwise(base * 12)
    months = pl.when(mark == "99").then(pl.lit(9999)).otherwise(months)
    return pl.when(mark.is_in(valid_marks)).then(months).otherwise(None).cast(pl.Int64)


def months_str(months: pl.Expr) -> pl.Expr:
    return (
        pl.when(months >= 9999).then(pl.lit("99")).otherwise(months.cast(pl.Utf8))
    )
