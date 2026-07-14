"""The package exception hierarchy.

Every failure this package raises for a bad rulebook, a missing column, or a
malformed decision grammar is an :class:`UnderwriterError`, so a caller can
``except UnderwriterError`` to catch *this* package's failures without swallowing
unrelated errors from polars or numpy. The concrete errors also subclass the
stdlib exception a caller would naturally reach for (``ValueError`` for bad data,
``TypeError`` for a wrong argument type), so existing ``except ValueError`` code
keeps working.
"""

from __future__ import annotations


class UnderwriterError(Exception):
    """Base class for every error this package raises deliberately."""


class RulebookError(UnderwriterError, ValueError):
    """A rulebook is malformed: a bad disease table, loading bands, or sentinels."""


class MissingColumnsError(UnderwriterError, ValueError):
    """A frame is missing a column a pipeline stage or the engine requires."""


class GrammarError(UnderwriterError, ValueError):
    """The decision table does not define a usable decision-code grammar."""


class FrameTypeError(UnderwriterError, TypeError):
    """A public boundary was handed something that is neither polars nor pandas."""


class InputError(UnderwriterError, ValueError):
    """A public function was called with an unusable argument or empty input
    (an empty frame to diagnose, a bad relaxation pattern, nothing to plot)."""
