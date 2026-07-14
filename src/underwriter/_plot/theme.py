"""Shared plot styling: white panel, black border, no gridlines."""

from __future__ import annotations

# decision colours: auto-decided blue, underwriter-referred red
AUTO_BLUE = "#80B1D3"
REFER_RED = "#FB8072"
BAR_BLUE = "#4E79A7"
BASELINE_GREY = "grey"


def import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "plotting needs matplotlib -- install with: pip install 'underwriter[plot]'"
        ) from exc
    return plt


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.grid(False)
