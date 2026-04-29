"""Shared accessibility helpers for the plotting scripts.

The plotting scripts use color to separate series, but readers with color
deficiency or printed black-and-white copies can't rely on that alone.  These
helpers expose ordered cycles of markers, hatch patterns, and (optional)
linestyles so each series gets a redundant non-color cue.

Design rules:
- Lines stay solid by default to keep the figures clean and professional.
  Series are differentiated by *marker shape*, which combines well with color.
- Bars and stacked bars use distinct hatch patterns combined with color.
- For very dense lines, callers should pass ``markevery`` to thin markers so
  shapes stay readable without crowding the curve.
- For curves with long flat regions at a floor or ceiling (e.g., saturation
  plots that hit 1.0 across all series), use ``interesting_marker_indices``
  to put markers only where the curves are doing something.

Usage::

    from _plot_style import marker_for, hatch_for, line_style_for

    for i, group in enumerate(groups):
        ax.plot(x, y, label=group, marker=marker_for(i), markersize=5)

    ax.bar(positions, heights, hatch=hatch_for(stack_index))
"""

from __future__ import annotations

import numpy as np

MARKER_CYCLE: tuple[str, ...] = (
    "o",  # circle
    "s",  # square
    "^",  # triangle up
    "D",  # diamond
    "v",  # triangle down
    "P",  # filled plus
    "X",  # filled x
    "*",  # star
    "<",  # triangle left
    ">",  # triangle right
    "h",  # hexagon
    "p",  # pentagon
)

LINESTYLE_CYCLE: tuple[object, ...] = (
    "-",
    (0, (5, 2)),
    (0, (1, 2)),
    (0, (4, 2, 1, 2)),
    (0, (5, 2, 1, 2, 1, 2)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (5, 5)),
)

HATCH_CYCLE: tuple[str, ...] = (
    "",
    "//",
    "..",
    "xx",
    "\\\\",
    "++",
    "--",
    "||",
    "oo",
    "**",
)


def marker_for(idx: int) -> str:
    """Return a distinct marker shape for the given series index."""
    return MARKER_CYCLE[idx % len(MARKER_CYCLE)]


def line_style_for(idx: int) -> object:
    """Return a distinct linestyle (matplotlib spec) for the given series index."""
    return LINESTYLE_CYCLE[idx % len(LINESTYLE_CYCLE)]


def hatch_for(idx: int) -> str:
    """Return a distinct hatch pattern for the given bar/wedge index."""
    return HATCH_CYCLE[idx % len(HATCH_CYCLE)]


def interesting_marker_indices(
    values,
    *,
    floor: float | None = None,
    ceiling: float | None = None,
    max_markers: int = 18,
):
    """Pick marker indices in the ``(floor, ceiling)`` band of ``values``.

    Designed for ``markevery`` on plots that have long flat regions at a
    floor or ceiling (saturation plateau, log-scale floor, sharing-rate at
    shallow depth, etc.). Without this, every series ends up stacking
    markers in the same flat region and the figure looks busy.

    Returns a Python ``list`` of indices into ``values`` where the value is
    strictly above ``floor`` (when given) and strictly below ``ceiling``
    (when given), uniformly subsampled to about ``max_markers`` entries.
    Falls back to ``slice(None)`` (every point) when no value is in the
    band, so callers preserve their previous behavior on edge cases.
    """
    arr = np.asarray(values, dtype=float)
    mask = np.ones(arr.shape, dtype=bool)
    if floor is not None:
        mask &= arr > float(floor)
    if ceiling is not None:
        mask &= arr < float(ceiling)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        return slice(None)
    if candidates.size <= max_markers:
        return list(candidates)
    step = max(1, candidates.size // max_markers)
    return list(candidates[::step])
