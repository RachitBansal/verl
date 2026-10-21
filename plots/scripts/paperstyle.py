"""Shared figure style for the paper. Figure 1 (plot_fig1_panels.py) is the reference.

Every figure is drawn on a canvas K_CANVAS = 2.45 times the width it is printed at, so the type sizes below come out at
about 8-9 pt in the PDF and the lines at about 1.6 pt, whatever the figure's printed width:

    fig = plt.figure(figsize=paperstyle.canvas(0.495, aspect=0.75))   # a half-width subfigure, 3:4

An identical copy lives in critical-batch-rlvr/analysis/iclr_figs/paperstyle.py (cbs repo); keep the two in sync.
"""
import matplotlib as mpl

TEXTWIDTH_IN = 5.5          # ICLR text width
K_CANVAS = 2.45             # canvas inches per printed inch

# colours: one per sweep, as in Figure 1
SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
C = dict(k16="#1f77b4", k64="#d66300", b128="#2ca02c", b256="#8c564b", b1024="#9467bd",
         upsample="#c2378f",                       # upsampling (generate 64, train K)
         kl3="#1f77b4", kl2="#c2185b")             # K = 16 sweep at KL 1e-3 (= the Figure 1 blue) and at KL 1e-2
MARK = dict(k16="o", k64="h", b128="p", b256="s", b1024="v", upsample="D")

# type sizes (points on the canvas; divide by K_CANVAS for the printed size)
FS = dict(tick=20, label=22, legend=19, annot=20, title=22)
# line widths and marker sizes on the canvas
LW = dict(line=4.0, ref=2.6, crit=3.0, grid=0.8, edge=1.5)
MS = dict(marker=13, scatter=190, ring=15)


def canvas(frac=1.0, aspect=0.6):
    """(width, height) in inches for a figure printed at `frac` of the text width, height = aspect * width"""
    w = frac * TEXTWIDTH_IN * K_CANVAS
    return (w, w * aspect)


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
        "font.size": FS["label"], "axes.labelsize": FS["label"], "axes.titlesize": FS["title"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
        "legend.title_fontsize": FS["legend"], "legend.frameon": False,
        "lines.linewidth": LW["line"], "lines.markersize": MS["marker"],
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finish(ax, grid=True):
    """Figure 1 look: no frame, light major grid behind the data, no tick marks"""
    for sp in ax.spines.values():
        sp.set_visible(False)
    if grid:
        ax.grid(True, which="major", color=GRID, lw=LW["grid"], zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0, colors=INK, labelsize=FS["tick"])


def subcaption(ax, text, xlabel=None, fontsize=None):
    """Sub-figure caption inside the figure, under the panel (below its x-axis label), instead of a LaTeX \\subcaption or a
    title on top. Use \\phantomsubcaption in LaTeX to keep \\label / \\ref working for (a), (b), ..."""
    ax.set_xlabel(f"{xlabel}\n{text}" if xlabel else text, linespacing=1.7, fontsize=fontsize or FS["label"])
