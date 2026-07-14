"""The four sentinel ``kcd_main`` values.

These are reserved codes that are *not* diagnoses. They are the type-level
expression of the package's prime invariant: **no insured ever disappears from
any pipeline stage** -- every id leaves every stage with at least one row. A
real KCD code matches ``^[A-Z][0-9]{2,}$``, so none of these words can collide
with one.

Each sentinel is stamped at exactly one stage and carries an *intended*
disposition. The disposition is only intended here -- the actual decision is
authored in the rulebook's sentinel rows; this enum only guarantees the
vocabulary (and lets ``diagnose_ruleset`` verify every sentinel has a rule).

===========  ==================  ==========================  ================
sentinel     stamped by          meaning                     intended
===========  ==================  ==========================  ================
VACANT       clean_icis          the line carried no code    standard
IRREGULAR    clean_icis          a code was written but       underwriter
                                 could not be read
UNMAPPED     map_disease         a valid code the disease     underwriter
                                 table has no row for
EXPIRED      aggregate_disease   every diagnosis aged out of  standard
                                 its lookback window
===========  ==================  ==========================  ================

This module is the single source of the four names. Nothing else in the package
should spell ``"VACANT"`` etc. as a bare string literal.
"""

from __future__ import annotations

from enum import Enum


class Sentinel(str, Enum):
    """A reserved ``kcd_main`` value. A ``str`` subclass, so a member compares
    equal to its spelling and can be used wherever a code string is expected;
    pass ``.value`` into polars expressions for an explicit literal."""

    VACANT = "VACANT"
    IRREGULAR = "IRREGULAR"
    UNMAPPED = "UNMAPPED"
    EXPIRED = "EXPIRED"

    def __str__(self) -> str:  # keep f-strings / polars literals as the bare code
        return self.value


#: All four sentinel codes as plain strings, in stamping order. This is the set
#: ``diagnose_ruleset`` checks for a missing ``decl_yn == 0`` rule.
SENTINELS: tuple[str, ...] = tuple(s.value for s in Sentinel)
