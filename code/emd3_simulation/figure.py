"""Figure: measurements and model assumptions informing EMD3 attribution.

Same conventions as the EMD1, EMD2 and EMD4 figures: exact 183 mm double-column
canvas, shared type scale with a 6 pt floor, Type 42 fonts, colour by role,
solid tints rather than alpha, vector + 600 dpi raster export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import design as dz
from .conditions import DESIGNS, MATCH_DAY, TRUTHS

MM = 1 / 25.4
W_DOUBLE = 183 * MM

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans",
                        "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.default": "regular",
    "figure.dpi": 160, "savefig.dpi": 600,
    "savefig.bbox": None, "savefig.pad_inches": 0,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

INK, INK2, MUTED, RULE = "#12253a", "#3d4a58", "#6b7785", "#c9d1da"
IND, SEL, CYT = "#1baf7a", "#2a78d6", "#b3323f"
FLAG = "#b3323f"          # the "this is the dangerous one" accent
TS = {"fignote": 6.6, "body": 7.2, "label": 7.6, "head": 8.6, "title": 9.4}
TRUTH_COLOR = {"induction": IND, "selection": SEL, "cytotoxicity": CYT}


def tint(hex_color: str, frac: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * (1 - frac)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax, title, ylabel, tag, xlabel=None):
    ax.set_title(title, fontsize=TS["label"], color=INK, pad=3.5, loc="left")
    ax.set_ylabel(ylabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    ax.tick_params(labelsize=TS["fignote"], colors=INK2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.text(-0.24, 1.16, tag, transform=ax.transAxes, fontsize=TS["head"],
            fontweight="bold", color=INK, va="top", ha="left")


# Publication export set. The 600-dpi TIFF is ~49 MB and exists for the journal
# submission alone; an interactive walkthrough only needs the PNG, so callers
# can narrow the set rather than paying for it.
FORMATS = ("pdf", "svg", "png", "eps", "tif")


def make_figure(mp, sep, ctx, outdir: Path,
                formats: tuple = FORMATS) -> list[Path]:
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.66))
    gs = fig.add_gridspec(2, 3, left=0.070, right=0.980, top=0.850, bottom=0.105,
                          wspace=0.44, hspace=0.62)

    # --- a: the endpoint trap ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    t = sep["t"]
    ax.plot(t, sep["induction"], color=IND, linewidth=1.3, label="induction")
    ax.plot(t, sep["selection"], color=SEL, linewidth=1.3, label="selection")
    ax.axvline(MATCH_DAY, color=RULE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax.plot([MATCH_DAY], [ctx["pair"]["x_ind_at_match"]], "o", color=INK,
            markersize=3.0, zorder=5)
    ax.annotate("same endpoint", xy=(MATCH_DAY, ctx["pair"]["x_ind_at_match"]),
                xytext=(38, 0.10), fontsize=TS["fignote"] - 0.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    _style(ax, "Two mechanisms, one endpoint", "stem-like fraction", "a", "day")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 0.40)
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="lower right",
              handlelength=1.2, borderaxespad=0.2, labelspacing=0.22)

    # --- b: the exact degeneracy -----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    d = ctx["degeneracy"]
    ax.plot(d["t"], d["x_sel"], color=SEL, linewidth=2.0, label="selection")
    ax.plot(d["t"], d["x_cyt"], color=CYT, linewidth=1.0,
            linestyle=(0, (3, 2)), label="differential death")
    _style(ax, "Identical in the fraction", "stem-like fraction", "b", "day")
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="lower right",
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.22)
    axi = ax.inset_axes([0.13, 0.58, 0.40, 0.36])
    axi.plot(d["t"], d["logn_sel"], color=SEL, linewidth=1.0)
    axi.plot(d["t"], d["logn_cyt"], color=CYT, linewidth=1.0, linestyle=(0, (3, 2)))
    axi.set_title("log total count", fontsize=TS["fignote"] - 1.2, color=INK2,
                  pad=1.5)
    axi.tick_params(labelsize=TS["fignote"] - 2.0, colors=INK2, pad=1.0)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)

    # --- c: the same data under two structures ----------------------------
    # Plotting the false-positive RATE here would be six identical 100% bars:
    # with a properly endpoint-matched cytotoxic control every agent yields a
    # confident interval under the induction-only structure. The panel now
    # carries V16 as well: the SAME fractions, profiled with the fitness gap
    # free, put the interval back across zero for the two non-plastic agents.
    # That is the sharpest statement the build has -- one dataset, two
    # structures, opposite conclusions -- so the two intervals are drawn on one
    # row rather than in separate panels.
    ax = fig.add_subplot(gs[0, 2])
    rows = [(dn, tn) for dn in DESIGNS for tn in TRUTHS]
    for j, (dn, tn) in enumerate(rows):
        y = len(rows) - 1 - j
        ri = ctx["fp"][dn][tn]
        rb = ctx["fp_both"][dn][tn]
        ax.plot([rb["lo"], rb["hi"]], [y - 0.22, y - 0.22],
                color=tint(TRUTH_COLOR[tn], 0.42), linewidth=1.5,
                solid_capstyle="butt", zorder=2)
        ax.plot([ri["lo"], ri["hi"]], [y + 0.20, y + 0.20],
                color=TRUTH_COLOR[tn], linewidth=2.4, solid_capstyle="butt",
                zorder=3)
        ax.plot([ri["point"]], [y + 0.20], "o", color=TRUTH_COLOR[tn],
                markersize=2.8, zorder=4)
    ax.axvline(0.0, color=FLAG, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{tn[:6]}·{dn}" for dn, tn in rows][::-1],
                       fontsize=TS["fignote"] - 1.4)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(left=-0.006)
    _style(ax, r"Profile-interval summaries", "", "c",
           xlabel=r"$\Delta\beta$ profile interval")
    ax.plot([], [], color=INK, linewidth=2.4, label="induction-only")
    ax.plot([], [], color=MUTED, linewidth=1.5, label="gap free")
    ax.legend(fontsize=TS["fignote"] - 1.8, frameon=False, loc="lower right",
              handlelength=1.1, borderaxespad=0.2, labelspacing=0.18)

    # --- d: which ANALYSIS false-positives --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    tns = list(TRUTHS)
    xw = np.arange(len(tns))
    # Three analyses of the same data. Fitting induction ALONE is the one that
    # false-positives; a comparison that includes a differential-death
    # structure does not. Counting any structure that asserts a non-zero beta,
    # not just "ind" -- with counts the comparison often prefers the combined
    # structure, which is still an induction call.
    alone = [ctx["fp"]["extended"][tn]["ci_excl"] * 100 for tn in tns]
    gapfree = [ctx["fp_both"]["extended"][tn]["ci_excl"] * 100 for tn in tns]
    fo = [ctx["counts"][tn]["frac_picks_beta"] * 100 for tn in tns]
    wc = [ctx["counts"][tn]["picks_beta"] * 100 for tn in tns]
    ax.bar(xw - 0.30, alone, width=0.19, color=FLAG, edgecolor="none",
           label="fit induction alone")
    ax.bar(xw - 0.10, gapfree, width=0.19, color=tint(SEL, 0.55),
           edgecolor="none", label=r"profile $\Delta\beta$, gap free")
    ax.bar(xw + 0.10, fo, width=0.19, color=tint(CYT, 0.5), edgecolor="none",
           label="compare structures (fractions)")
    ax.bar(xw + 0.30, wc, width=0.19, color=IND, edgecolor="none",
           label="compare structures (+ counts)")
    ax.set_xticks(xw)
    ax.set_xticklabels(["induction", "selection", "cytotox."],
                       fontsize=TS["fignote"] - 0.8, rotation=20, ha="right")
    ax.set_ylim(0, 140)
    _style(ax, r"Inference of exposure-induced conversion",
           "synthetic datasets (%)", "d")
    ax.legend(fontsize=TS["fignote"] - 1.6, frameon=False, loc="upper center",
              handlelength=0.9, borderaxespad=0.15, labelspacing=0.16)

    # --- e: the design answer, as POWER -----------------------------------
    # A verdict ("the true contrast clears the critical value") is 50% power at
    # the boundary, and the 100x design sits almost exactly there. Plotting the
    # power curve instead of a pass/fail verdict is what stops a coin flip from
    # being read as a working design.
    ax = fig.add_subplot(gs[1, 1])
    folds = np.array([10, 20, 50, 100, 141, 200, 300])
    pw = [dz.purity_dependence(mp, eps_lo=0.001, eps_hi=0.001 * fo,
                               day=7.0)["power"] for fo in folds]
    ax.plot(folds, np.array(pw) * 100, color=IND, linewidth=1.3, marker="o",
            markersize=2.6)
    ax.axhline(80, color=FLAG, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.axhline(50, color=RULE, linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 300])
    ax.set_xticklabels(["10x", "20x", "50x", "100x", "200x", "300x"])
    ax.set_ylim(0, 108)
    for fo, lab in ((100, "56%"), (141, "80%")):
        i = int(np.flatnonzero(folds == fo)[0])
        ax.plot([fo], [pw[i] * 100], "o", color=FLAG if fo == 100 else INK,
                markersize=3.4, zorder=5)
    ax.annotate("56% power\nat n = 3", xy=(100, pw[3] * 100),
                xytext=(11, 86), fontsize=TS["fignote"] - 1.4, color=FLAG,
                arrowprops=dict(arrowstyle="-", color=FLAG, linewidth=0.6))
    _style(ax, "Sorting contrast: power and starting composition",
           "% power", "e", "residual purity separation")

    # --- f: Boolean layer -------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    # The two knockdowns carry V13: they do not RESTORE the control basin, they
    # abolish it, going below the level the control network carries anyway.
    conds = ["control", "arsenic", "arsenic_FZD10_kd", "arsenic_WNT_inhib",
             "unsupported"]
    # Short, rotated. Five two-line labels in a one-third-width panel collide
    # into unreadable overlap; the supported/unsupported distinction moves to
    # the caption rather than being crammed onto the axis.
    labels = ["control", "arsenic", "+FZD10 kd", "+WNT inh", "+p53 loss"]
    xb = np.arange(len(conds))
    basin = [ctx["bool"][c]["stem_basin"] * 100 for c in conds]
    reach = [ctx["bool"][c]["reach_frac"] * 100 for c in conds]
    # V15 needs BOTH immortal columns. Plotting reachability alone showed a flat
    # row of zeros and left the misreading it guards against invisible: the
    # immortal BASIN does move with exposure, and that is the statistic a reader
    # would otherwise quote as evidence of immortalisation.
    imm_basin = [ctx["bool"][c]["immortal_basin"] * 100 for c in conds]
    imm = [ctx["bool"][c]["reach_immortal"] * 100 for c in conds]
    # Basins are Monte Carlo estimates over update orders, so they carry an
    # error bar. Without it the bars invite the third significant figure to be
    # read as real; the earlier one-trajectory-per-start version had an MC
    # error near +-1.2 points and was quoted to 0.1.
    basin_se = [ctx["bool"][c]["stem_basin_se"] * 100 for c in conds]
    imm_se = [ctx["bool"][c]["immortal_basin_se"] * 100 for c in conds]
    ebar = dict(ecolor=INK2, elinewidth=0.6, capsize=1.2, capthick=0.6)
    ax.bar(xb - 0.30, basin, width=0.19, color=tint(IND, 0.5), edgecolor="none",
           yerr=basin_se, error_kw=ebar, label="stem basin")
    ax.bar(xb - 0.10, reach, width=0.19, color=IND, edgecolor="none",
           label="stem reached")
    ax.bar(xb + 0.10, imm_basin, width=0.19, color=tint(CYT, 0.5), edgecolor="none",
           yerr=imm_se, error_kw=ebar, label="immortal basin")
    ax.bar(xb + 0.30, imm, width=0.19, color=CYT, edgecolor="none",
           label="immortal reached")
    # V17: the immortal zero is a property of the TERT-negative START as well as
    # of the wiring. From a telomerase-positive start -- which the exemplar line
    # is -- the unsupported perturbation scores immortalisation in every run.
    # Drawn as an open overlay so the conditional cannot be read off the figure
    # as a flat row of zeros.
    imm_on = [ctx["bool_tert_on"][c]["reach_immortal"] * 100 for c in conds]
    ax.bar(xb + 0.30, imm_on, width=0.19, facecolor="none", edgecolor=CYT,
           linewidth=0.7, linestyle=(0, (1.6, 1.2)),
           label="immortal reached\n(TERT-on start)")
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, fontsize=TS["fignote"] - 1.2, rotation=28,
                       ha="right")
    ax.axhline(basin[0], color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)),
               zorder=0)
    # Sits over the two knockdown columns, which are zero: at the left edge the
    # label ran into the arsenic stem-basin bar once the fourth bar narrowed the
    # group spacing.
    ax.text(2.5, basin[0] + 3.5, "control basin",
            fontsize=TS["fignote"] - 1.6, color=MUTED, ha="center")
    ax.set_ylim(0, 118)
    _style(ax, "Attractor structure", "% ", "f")
    ax.legend(fontsize=TS["fignote"] - 2.0, frameon=False, loc="upper left",
              handlelength=0.9, borderaxespad=0.2, labelspacing=0.12,
              ncol=2, columnspacing=0.6)

    fig.text(0.070, 0.980,
             "EMD3: distinguishing exposure-induced conversion from population "
             "selection",
             fontsize=TS["title"], fontweight="bold", color=INK, va="top")
    fig.text(0.070, 0.947,
             "Synthetic population dynamics under induction, selection and "
             "differential-death scenarios, with the arsenic E2F2/FZD10–Wnt "
             "pathway as biological context.",
             fontsize=TS["fignote"], color=MUTED, va="top")
    fig.text(0.070, 0.925,
             "Rates and observation models are specified, not estimated. "
             "Population and Boolean results are conditional model analyses, "
             "not independent experimental evidence.",
             fontsize=TS["fignote"], color=MUTED, va="top")

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        pth = outdir / f"figure2e_emd3_simulation.{ext}"
        fig.savefig(pth, format=ext)
        paths.append(pth)
    plt.close(fig)
    print(f"  wrote {paths[0].parent / 'figure2e_emd3_simulation'}"
          f".{{{','.join(formats)}}}")
    return paths
