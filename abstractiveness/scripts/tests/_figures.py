"""
Shared figure inspection of the check scripts: the figures a function closes, and what their scatter panels draw.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, PolyCollection


def closed_figures(function, *args, **kwargs) -> tuple:
    """The function's result and the figures it closed, kept open for inspection."""
    figures, close = [], plt.close
    plt.close = lambda fig=None: figures.append(plt.gcf() if fig is None else fig)
    try:
        result = function(*args, **kwargs)
    finally:
        plt.close = close
    return result, figures


def points(ax) -> np.ndarray:
    """Every scattered point of a panel, one row per point."""
    offsets = [c.get_offsets() for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())]
    return np.vstack(offsets) if offsets else np.zeros((0, 2))


def line_slope(ax) -> float:
    """Slope of the panel's first drawn line, NaN without one."""
    if not ax.lines:
        return np.nan
    x, y = (np.asarray(values, dtype=float) for values in ax.lines[0].get_data())
    return float(np.polyfit(x, y, 1)[0])


def has_band(ax) -> bool:
    """Whether the panel draws a filled band."""
    return any(isinstance(c, PolyCollection) for c in ax.collections)


def legend_texts(ax) -> list:
    """The panel's legend entries, empty without a legend."""
    legend = ax.get_legend()
    return [text.get_text() for text in legend.get_texts()] if legend else []


def same_points(drawn: np.ndarray, x, y) -> bool:
    """Whether the drawn points are exactly the given x and y, in any order."""
    expected = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if drawn.shape != expected.shape:
        return False
    order = lambda array: array[np.lexsort(array.T[::-1])]
    return bool(np.allclose(order(drawn), order(expected)))
