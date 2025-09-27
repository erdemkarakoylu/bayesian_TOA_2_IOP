
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Tuple, Sequence

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np

# --- PNAS column widths (cm) and converter ---
PNAS_WIDTHS_CM = {"one": 8.7, "onehalf": 11.4, "two": 17.8}  # 1, 1.5, 2 columns
CM2IN = 1 / 2.54

# Okabe–Ito colorblind-safe palette
OKABE_ITO: Sequence[str] = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

def pnas_figsize(width: str = "one", aspect: float = 0.62) -> Tuple[float, float]:
    """Return (w,h) in inches for PNAS figures.
    width ∈ {'one','onehalf','two'}. aspect = h/w."""
    if width not in PNAS_WIDTHS_CM:
        raise ValueError(f"width must be one of {tuple(PNAS_WIDTHS_CM)}")
    w_in = PNAS_WIDTHS_CM[width] * CM2IN
    h_in = w_in * aspect
    return (w_in, h_in)

def set_pnas_style(
    base_font_size: int = 9,
    font_stack: Iterable[str] = ("Arial", "Helvetica", "DejaVu Sans"),
    use_tex: bool = False,
    show_grid: bool = False,
) -> None:
    """Set Matplotlib rcParams for PNAS-style figures."""
    mpl.rcParams.update({
        # Fonts & text
        "text.usetex": use_tex,
        "font.family": "sans-serif",
        "font.sans-serif": list(font_stack),
        "mathtext.fontset": "dejavusans",  # sans math without TeX
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 2,

        # Lines, ticks, spines
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grid & legend
        "axes.grid": show_grid,
        "grid.alpha": 0.2,
        "legend.frameon": False,
        "legend.handlelength": 1.4,

        # Color cycle
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),

        # Save/export (vector, editable fonts)
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.format": "pdf",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Layout: we’ll call tight_layout() / constrained_layout per-figure
        "figure.autolayout": False,
    })

def fig_ax(
    width: str = "one",
    aspect: float = 0.62,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool | str = False,
    sharey: bool | str = False,
    squeeze: bool = True,
    constrained_layout: bool = False,
):
    """Create a figure/axes sized to PNAS columns."""
    figsize = pnas_figsize(width, aspect)
    fig, axes = pp.subplots(
        nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
        squeeze=squeeze, figsize=figsize, constrained_layout=constrained_layout
    )
    return fig, axes

def label_panels(
    axes: Iterable[pp.Axes],
    labels: Iterable[str] = ("a", "b", "c", "d"),
    loc: str = "upper left",
    dx: float = 0.02,
    dy: float = 0.02,
    weight: str = "bold",
):
    """Add (a), (b), ... labels to multiple axes in figure coordinates."""
    loc_map = {
        "upper left": (dx, 1 - dy), "upper right": (1 - dx, 1 - dy),
        "lower left": (dx, dy),     "lower right": (1 - dx, dy),
    }
    if loc not in loc_map:
        raise ValueError(f"loc must be one of {tuple(loc_map)}")
    x0, y0 = loc_map[loc]
    for ax, lab in zip(axes, labels):
        ax.text(x0, y0, f"({lab})", transform=ax.transAxes,
                ha="left" if "left" in loc else "right",
                va="top" if "upper" in loc else "bottom",
                fontsize=mpl.rcParams["axes.labelsize"], fontweight=weight)

def save_pnas(fig: pp.Figure, path: str | Path) -> None:
    """Save as vector PDF with tight bbox; create dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)  # format/bbox from rcParams

@contextmanager
def pnas_figure(path: str | Path, width: str = "one", aspect: float = 0.62, **subplots_kw):
    """Context manager: create, yield, then save & close a PNAS-sized figure."""
    fig, ax = fig_ax(width=width, aspect=aspect, **subplots_kw)
    try:
        yield (fig, ax)
    finally:
        save_pnas(fig, path)
        pp.close(fig)

