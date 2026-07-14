"""Static validation of a rule set -- no claim data. Finds the authoring defects
the engine can only surface at run time (and only once some input lands in the
offending band): shadow conditions, latent conflicts, exact duplicates, diseases
with no auto rule, and missing sentinels.

Only ``decl_yn == 0`` rows are examined (those the engine applies). Band tests use
the four joined bands (``age``, ``elp_day``, ``sur_cnt``, ``hos_day``);
``out_day`` is carried but never joined, so a constrained ``out_day`` band counts
as a shadow condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..rulebook.rulebook import Rulebook
from ..sentinels import SENTINELS

_SHADOW_COLUMNS = ("recover", "recur", "treat", "severe", "cause", "medical_checkup")
_BAND_LO = ("age_min", "elp_day_min", "sur_cnt_min", "hos_day_min")
_BAND_HI = ("age_max", "elp_day_max", "sur_cnt_max", "hos_day_max")


@dataclass(frozen=True)
class RulesetDiagnosis:
    n_rule: int
    n_kcd: int
    n_auto: int
    shadow_condition: dict
    latent_conflict: dict
    exact_duplicate: dict
    no_auto_rule: dict
    missing_sentinel: dict

    def __repr__(self) -> str:
        sc, lc = self.shadow_condition, self.latent_conflict
        ed, na, ms = self.exact_duplicate, self.no_auto_rule, self.missing_sentinel
        lines = [
            f"n_rule={self.n_rule:,} | n_kcd={self.n_kcd:,} | decl_yn==0 rows={self.n_auto:,}",
            f"  shadow_condition  : {sc['n_row']} rows across {sc['n_kcd']} kcd_main "
            f"({', '.join(f'{k}={v}' for k, v in sc['by_col'].items()) or 'none'})",
            f"  latent_conflict   : {lc['n_pair']} pairs across {lc['n_kcd']} kcd_main "
            f"({lc['n_genuine']} genuine)",
            f"  exact_duplicate   : {ed['n_group']} groups ({ed['n_extra']} redundant rows)",
            f"  no_auto_rule      : {na['n_kcd']} kcd_main"
            + (f" ({', '.join(na['kcds'][:8])})" if na["n_kcd"] else ""),
            f"  missing_sentinel  : {ms['n_kcd']}"
            + (f" ({', '.join(ms['kcds'])})" if ms["n_kcd"] else ""),
        ]
        return "\n".join(lines)


def _is_condition(column: str) -> pl.Expr:
    stripped = pl.col(column).cast(pl.Utf8).str.strip_chars()
    return stripped.is_in(["", "*"]).not_() & pl.col(column).is_not_null()


def diagnose_ruleset(rulebook: Rulebook) -> RulesetDiagnosis:
    rs = rulebook.ruleset
    decision_columns = list(rulebook.decision_columns)
    auto = rs.filter(pl.col("decl_yn") == 0)

    shadow_columns = [c for c in _SHADOW_COLUMNS if c in auto.columns]
    flag_exprs = {c: _is_condition(c).alias(f"_sc_{c}") for c in shadow_columns}
    if {"out_day_min", "out_day_max"} <= set(auto.columns):
        flag_exprs["out_day"] = (
            ((pl.col("out_day_min") > 0) | (pl.col("out_day_max") < 9999))
            .fill_null(False)
            .alias("_sc_out_day")
        )
    flagged = auto.with_columns(list(flag_exprs.values()))
    keys = list(flag_exprs)
    by_col = {
        k: int(flagged.select(pl.col(f"_sc_{k}").sum()).item()) for k in keys
    }
    by_col = {k: v for k, v in by_col.items() if v > 0}
    any_condition = pl.any_horizontal([pl.col(f"_sc_{k}") for k in keys])
    sh_rows = flagged.filter(any_condition)
    shadow_condition = {
        "n_row": sh_rows.height,
        "n_kcd": sh_rows["kcd_main"].n_unique(),
        "by_col": by_col,
    }

    latent_conflict = _latent_conflict(auto, decision_columns, shadow_columns)

    dup_key = ["kcd_main", *_BAND_LO, *_BAND_HI, *decision_columns]
    groups = auto.group_by(dup_key).len().filter(pl.col("len") > 1)
    exact_duplicate = {
        "n_group": groups.height,
        "n_extra": int((groups["len"] - 1).sum()) if groups.height else 0,
    }

    present = set(rs["kcd_main"].unique().to_list())
    with_auto = set(auto["kcd_main"].unique().to_list())
    no_auto = sorted(present - with_auto)
    missing = [s for s in SENTINELS if s not in with_auto]

    return RulesetDiagnosis(
        n_rule=rs.height,
        n_kcd=rs["kcd_main"].n_unique(),
        n_auto=auto.height,
        shadow_condition=shadow_condition,
        latent_conflict=latent_conflict,
        exact_duplicate=exact_duplicate,
        no_auto_rule={"n_kcd": len(no_auto), "kcds": no_auto},
        missing_sentinel={"n_kcd": len(missing), "kcds": missing},
    )


def _latent_conflict(auto: pl.DataFrame, decision_columns: list[str], shadow_columns: list[str]) -> dict:
    """Pairwise within kcd_main: decl_yn==0 rows whose four bands overlap and whose
    decisions differ. A pair also differing in a shadow / out_day column is
    ``shadow_explained`` (legitimate but engine-unresolvable), not genuine."""
    shadow_cmp = [c for c in shadow_columns if c in auto.columns]
    shadow_cmp += [c for c in ("out_day_min", "out_day_max") if c in auto.columns]
    n_pair = n_genuine = 0
    kcds: set[str] = set()
    for group in auto.partition_by("kcd_main"):
        m = group.height
        if m < 2:
            continue
        lo = group.select(_BAND_LO).to_numpy()
        hi = group.select(_BAND_HI).to_numpy()
        dec = group.select(decision_columns).fill_null("").to_numpy()
        shm = group.select(shadow_cmp).fill_null("").to_numpy() if shadow_cmp else None
        kcd = group["kcd_main"][0]
        for i in range(m - 1):
            for j in range(i + 1, m):
                overlap = (lo[i] <= hi[j]).all() and (lo[j] <= hi[i]).all()
                if overlap and (dec[i] != dec[j]).any():
                    n_pair += 1
                    kcds.add(kcd)
                    if shm is None or not (shm[i] != shm[j]).any():
                        n_genuine += 1
    return {"n_pair": n_pair, "n_kcd": len(kcds), "n_genuine": n_genuine}
