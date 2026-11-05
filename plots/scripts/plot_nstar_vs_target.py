"""Fitted critical batch N* against the target accuracy, per sweep, using only the columns whose LR bracket is complete.

Sweeps (each a set of columns x with N sequences per step):
  prompts, K = 16, KL 1e-3     x = B, N = 16 B     (csv/val_curves_n16.json)
  prompts, K = 16, KL 1e-2     x = B, N = 16 B     (same file)
  prompts, K = 64, KL 1e-3     x = B, N = 64 B     (--n64 csv/val_curves_n64.json; fixed-code runs only)
  rollouts, B = 128, KL 1e-3   x = K, N = 128 K    (--b128 csv/val_curves_bsz128.json; pre-fix runs for K <= 16, fixed-code for 32/64/128)

For each target T (--min-target, default 40%, .. 50%, 0.5% steps), every column is reduced to its fastest interpolated crossing over the LRs
tried, and the bracket of that best LR is judged by the rule of the interactive LR grid (build_lr_grid_interactive.py):
complete when, on each side, some tested LR within 2.5x of the best (or else the nearest one) is slower or conclusively did not
reach T (has run at least as long as the best crossing, capped at SHORT_STEPS); inconclusive when a neighbour is too short to judge yet; open when
no LR was tried on that side. S(N) = S_min (1 + N*/N) is then fitted in log S to the complete-bracket columns (solid)
and, for comparison, to every column with a crossing (faint). Markers are filled where every crossing column is
complete. The lower panel counts the complete columns per sweep.

Usage: python plot_nstar_vs_target.py csv/val_curves_n16.json png/nstar_vs_target.png [--n64 FILE] [--b128 FILE]
                                      [--min-bsz 4] [--min-cols 5] [--min-target 0.40] [--max-target 0.50] [--target-step 0.005] [--all-cols] [--keep-first-interval] [--sweeps n16,n16kl2,n64,b128] [--fixed-min-bsz 128]
A column whose best crossing is interpolated between step 0 and the run's first validation (a coarse 25-step interval for
most runs) is left out of every fit (the lower panel counts only what each fit uses); --keep-first-interval restores the old behaviour.
Writes the PNG, paper/<stem>.pdf, csv/<stem>_table.csv (per sweep and target) and csv/<stem>_columns.csv.
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
SHORT_STEPS = 200
MAX_GAP = 2.5           # any LR within this factor of the best that closes a side completes it (else the nearest LR decides)
VALID_N_PREFIX = {1, 2, 4, 8, 16}
FIXED_ONLY_N = {32, 64, 128}


def opt(flag, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
json_in, png_out = args[0], args[1]
MIN_BSZ, MIN_COLS = opt("--min-bsz", 4, int), opt("--min-cols", 5, int)
FIXED_MIN_BSZ = opt("--fixed-min-bsz", None, int)   # K = 16 sweeps: drop pre-fix runs at batch sizes >= this (as plot_steps_to_50_combined.py)
k16ok = lambda r: FIXED_MIN_BSZ is None or r["fixed"] or r["bsz"] < FIXED_MIN_BSZ
ALL_COLS = "--all-cols" in sys.argv          # ignore bracketing: fit every point that reached the target, no hollow/faint distinction
T_LO = opt("--min-target", 0.40, float)
T_STEP = opt("--target-step", 0.005, float)
T_HI = opt("--max-target", 0.50, float)
TARGETS = np.round(np.arange(T_LO, T_HI + 1e-9, T_STEP), 3)
KEEP_FIRST = "--keep-first-interval" in sys.argv   # by default a column whose crossing precedes its first eval is left out of the fits

n16 = json.load(open(json_in))
SERIES = [
    dict(key="n16", name="prompts, K = 16, KL 1e-3", runs=[r for r in n16 if abs(r["kl"] - 1e-3) < 1e-12 and r["bsz"] >= MIN_BSZ and k16ok(r)], x="bsz", per=16, color="#1f77b4", marker="o"),
    dict(key="n16kl2", name="prompts, K = 16, KL 1e-2", runs=[r for r in n16 if abs(r["kl"] - 1e-2) < 1e-12 and r["bsz"] >= MIN_BSZ and k16ok(r)], x="bsz", per=16, color="#8fb8e8", marker="D"),
]
if opt("--n64"):
    n64 = json.load(open(opt("--n64")))
    SERIES.append(dict(key="n64", name="prompts, K = 64, KL 1e-3", runs=[r for r in n64 if abs(r["kl"] - 1e-3) < 1e-12 and r["fixed"] and r["bsz"] >= MIN_BSZ], x="bsz", per=64, color="#d66300", marker="h"))
if opt("--b128"):
    b128 = json.load(open(opt("--b128")))
    roll = [r for r in b128 if not r["downsample"] and r["kl"] is not None and abs(r["kl"] - 1e-3) < 1e-12
            and (r["n"] in VALID_N_PREFIX or (r["fixed"] and r["n"] in FIXED_ONLY_N))]
    SERIES.append(dict(key="b128", name="rollouts, B = 128, KL 1e-3", runs=roll, x="n", per=128, color="#2ca02c", marker="p"))


def crossing(steps, vals, t):
    """Interpolated crossing step and whether it lies inside the first evaluation interval (step 0 .. first eval) when that
    interval is longer than one step; a crossing within step 1 is as fine as one optimizer step allows and is treated as measured."""
    for i, v in enumerate(vals):
        if v >= t:
            if i == 0:
                return float(steps[0]), steps[0] == 0
            s0, s1, v0, v1 = steps[i - 1], steps[i], vals[i - 1], vals[i]
            return max(1.0, float(s1) if v1 == v0 else s0 + (s1 - s0) * (t - v0) / (v1 - v0)), (i == 1 and steps[0] == 0 and steps[1] > 1)   # a crossing inside step 1 counts as one full step (floored), not a fraction of one: a step is the unit of training   # resumed segments start mid-run; a crossing inside step 1 of a run evaluated every step cannot be resolved finer, so it counts as measured
    return None, None


def fit(N, S):
    N = np.asarray(N, float); y = np.log(np.asarray(S, float)); best = None
    for ln in np.linspace(np.log(8), np.log(2 ** 28), 6000):
        g = np.log1p(np.exp(ln) / N); m = (y - g).mean(); r = np.sqrt(((y - g - m) ** 2).mean())
        if best is None or r < best[2]:
            best = (float(np.exp(m)), float(np.exp(ln)), float(r))
    if best[1] > 2 ** 26:          # ran to the edge of the search range: the points carry no bend, N* is undefined
        return (np.nan, np.nan, np.nan)
    return best


rows, cols_out = [], []
ONLY = opt("--sweeps", "")                      # --sweeps n16,n16kl2,n64,b128 draws only those series
if ONLY:
    SERIES = [ser for ser in SERIES if ser.get("key") in ONLY.split(",")]
for ser in SERIES:
    xk = ser["x"]
    for t in TARGETS:
        cfg = {}
        for r in ser["runs"]:
            if not r["steps"] or r["steps"][-1] == 0:      # only the base-model reading at step 0: the run never trained, so it is no neighbour
                continue
            c = cfg.setdefault((r[xk], r["lr"]), dict(x=r[xk], lr=r["lr"], steps=np.inf, longest=0, running=False, first=True, first_eval=None, miss=0))
            xx, first = crossing(r["steps"], r["vals"], t)
            # a measured crossing always beats a first-interval guess; among equals, the earlier one wins
            if xx is not None and ((c["first"] and not first) or (c["first"] == first and xx < c["steps"])):
                c["steps"], c["first"], c["first_eval"] = xx, first, r["steps"][1] if len(r["steps"]) > 1 else None
            if xx is None:
                c["miss"] = max(c["miss"], r["steps"][-1])
            c["longest"] = max(c["longest"], r["steps"][-1]); c["running"] |= r["state"] == "running"
        # a first-interval guess is overruled by a run of the same config evaluated through that whole interval without crossing
        for c in cfg.values():
            if c["first"] and np.isfinite(c["steps"]) and c["first_eval"] is not None and c["miss"] >= c["first_eval"]:
                c["steps"], c["first"] = np.inf, False
        if not cfg:
            continue
        d = pd.DataFrame(cfg.values()); d["steps"] = d["steps"].replace(np.inf, np.nan)
        cols = []
        for x, g in d.groupby("x"):
            g = g.sort_values("lr").reset_index(drop=True)
            hit = g.dropna(subset=["steps"])
            if hit.empty:
                continue
            ib = hit["steps"].idxmin(); ref = g.loc[ib, "steps"]
            verdict = lambda r: "hit" if not np.isnan(r["steps"]) else ("never" if r["longest"] >= min(ref, SHORT_STEPS) else "inconclusive")
            g["verdict"] = g.apply(verdict, axis=1)
            # a neighbour closes its side only with a measured verdict: a slower crossing that is itself a first-interval guess does not
            adj = lambda j: "open" if j < 0 or j >= len(g) else ("inconclusive" if g.loc[j, "verdict"] == "inconclusive" else
                                                                 "guess" if (g.loc[j, "verdict"] == "hit" and bool(g.loc[j, "first"])) else "complete")
            closes = lambda j: g.loc[j, "verdict"] == "never" or (g.loc[j, "verdict"] == "hit" and not bool(g.loc[j, "first"]))

            def side(step):
                # any tested LR on that side within MAX_GAP x of the best that closes it completes the side, even when the
                # nearest LR is inconclusive or a guess; otherwise the nearest LR decides, as before
                j = ib + step
                while 0 <= j < len(g) and abs(np.log(g.loc[j, "lr"] / g.loc[ib, "lr"])) <= np.log(MAX_GAP) + 1e-6:
                    if closes(j):
                        return "complete"
                    j += step
                return adj(ib + step)
            lo, up = side(-1), side(+1)
            status = "open" if "open" in (lo, up) else "inconclusive" if "inconclusive" in (lo, up) else "guess" if "guess" in (lo, up) else "complete"
            first = bool(g.loc[ib, "first"])
            if first and not KEEP_FIRST:
                status = "first-interval"      # crossing interpolated between step 0 and the first eval: too coarse to use
            cols.append(dict(series=ser["name"], target=t, x=int(x), N=int(x) * ser["per"], best_lr=g.loc[ib, "lr"], steps=ref, status=status, first_interval=first,
                             tried=" ".join(f"{r['lr']:g}{'' if r['verdict'] == 'hit' else '×' if r['verdict'] == 'never' else '?'}" for _, r in g.iterrows())))
        cols_out += cols
        c = pd.DataFrame(cols)
        dropped = c[c["status"] == "first-interval"]; c_ok = c[c["status"] != "first-interval"]
        comp = c[c["status"] == "complete"]
        row = dict(series=ser["name"], target=t, n_cols=len(c), n_first_interval=len(dropped), n_complete=len(comp), all_complete=len(comp) == len(c),
                   x_complete=" ".join(str(b) for b in comp["x"]), x_first_interval=" ".join(str(b) for b in dropped["x"]))
        if len(c_ok) >= 3:
            s_all, n_all, r_all = fit(c_ok["N"], c_ok["steps"]); row.update(smin_all=s_all, nstar_all=n_all, rms_all=r_all)
        if len(comp) >= MIN_COLS:
            s_c, n_c, r_c = fit(comp["N"], comp["steps"]); row.update(smin_complete=s_c, nstar_complete=n_c, rms_complete=r_c)
        rows.append(row)
tab = pd.DataFrame(rows)
stem = os.path.basename(png_out).replace(".png", "")
csv_dir = os.path.dirname(os.path.abspath(json_in))
tab.to_csv(os.path.join(csv_dir, stem + "_table.csv"), index=False)
pd.DataFrame(cols_out).to_csv(os.path.join(csv_dir, stem + "_columns.csv"), index=False)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13.5, "legend.fontsize": 11})
fig, (ax, axn) = plt.subplots(2, 1, figsize=(10.5, 7.8), facecolor=SURFACE, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.1], hspace=0.12))
for a in (ax, axn):
    a.set_facecolor(SURFACE); a.grid(True, which="major", color=GRID, lw=0.7); a.tick_params(which="both", colors=INK2, length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
for ser in SERIES:
    f = tab[tab["series"] == ser["name"]].sort_values("target")
    if f.empty:
        continue
    x = f["target"] * 100; col, mk = ser["color"], ser["marker"]
    if ALL_COLS:
        ok = f.dropna(subset=["nstar_all"]) if "nstar_all" in f else f.iloc[0:0]
        ax.plot(ok["target"] * 100, ok["nstar_all"], ls="-", lw=2, color=col, zorder=3)
        ax.scatter(ok["target"] * 100, ok["nstar_all"], s=42, marker=mk, color=col, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        axn.plot(x, f["n_cols"] - f["n_first_interval"], lw=1.8, color=col, zorder=3, drawstyle="steps-mid")
        continue
    if "nstar_all" in f:
        ax.plot(x, f["nstar_all"], ls="-", lw=1.2, color=col, alpha=0.3, zorder=2)
    if "nstar_complete" in f:
        ax.plot(x, f["nstar_complete"], ls="-", lw=2, color=col, zorder=3)
        full, part = f[f["all_complete"]], f[~f["all_complete"]]
        ax.scatter(full["target"] * 100, full["nstar_complete"], s=42, marker=mk, color=col, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        ax.scatter(part["target"] * 100, part["nstar_complete"], s=42, marker=mk, facecolors="none", edgecolors=col, linewidths=1.3, zorder=4)
    axn.plot(x, f["n_complete"], lw=1.8, color=col, zorder=3, drawstyle="steps-mid")
    axn.plot(x, f["n_cols"] - f["n_first_interval"], lw=1, ls=":", color=col, zorder=2, drawstyle="steps-mid")
ax.set_yscale("log"); ax.set_ylim(2e3, 5e5)
ax.set_ylabel("fitted N*  (sequences per step)", color=INK)
ax.set_title("critical batch vs target accuracy, fitted to every batch size that reached the target (bracketing ignored)" if ALL_COLS else
             "critical batch vs target accuracy, fitted to the batch sizes whose LR bracket is complete", loc="left", fontsize=13, color=INK)
axn.set_ylabel("batch sizes in the fit", color=INK); axn.set_xlabel("target AIME 1983–2024 accuracy (%)", color=INK)
axn.set_ylim(0, tab["n_cols"].max() + 1)
if not ALL_COLS:
    axn.axhline(MIN_COLS, color=INK2, lw=1, ls="--", zorder=1)
    axn.text(TARGETS[0] * 100 + 0.2, MIN_COLS + 0.3, f"{MIN_COLS} batch sizes: minimum for a solid line", fontsize=10, color=INK2, va="bottom")
legend = [Line2D([], [], marker=s["marker"], ls="-", lw=2, ms=7, color=s["color"], mec=SURFACE, label=s["name"]) for s in SERIES]
if ALL_COLS:
    legend += [Line2D([], [], ls="-", lw=1.8, color=INK2, label="lower panel: batch sizes in the fit")]
else:
  legend += [Line2D([], [], ls="-", lw=2, color=INK2, label=f"solid: fit to the bracketed batch sizes (needs {MIN_COLS})"),
             Line2D([], [], ls="-", lw=1.2, color=INK2, alpha=0.35, label="faint: fit to every batch size that reached the target"),
             Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=INK2, mew=1.3, label="hollow: some batch sizes lack a complete bracket"),
             Line2D([], [], ls="-", lw=1.8, color=INK2, label="lower panel: batch sizes in the solid fit (dotted: in the faint fit)")]
ax.legend(handles=legend, loc="upper left", frameon=False, labelcolor=INK, ncol=2, fontsize=10, columnspacing=1.2)
fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.09)
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, stem + ".pdf"), facecolor=SURFACE)
print("wrote", png_out)

# ------------------------------------------------------------------ --paper: one panel in the steps_to_50 / Figure 4 style
# (26-inch line width, 28-34 pt type, a sub-figure at 0.53 of the line width; no title, no count panel). y axis B* = N*/K (prompts per step). Per series: the bracketed
# fit as a line, every target as a dot (hollow where some batch size lacks a closed LR bracket), with a 1-sigma band on the fit. Writes <stem>_paper.png and paper/<stem>_paper.pdf.
if "--paper" in sys.argv:
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    TICK, LABEL, LEGEND = 28, 31, 34
    plt.rcParams.update(plt.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(0.53 * 26.0, 8.6), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=INK, labelsize=TICK, length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    hollow_any, ys = False, []
    for ser in SERIES:
        f = tab[tab["series"] == ser["name"]].dropna(subset=["nstar_complete"]).sort_values("target") if "nstar_complete" in tab else tab.iloc[0:0]
        if f.empty:
            continue
        col = ser["color"]; div = ser["per"] if ser["x"] == "bsz" else 1      # B* = N*/K prompts per step
        x = f["target"] * 100; y = f["nstar_complete"] / div; ys += list(y)
        # B*(acc): a power law in the target accuracy, B* = c acc^alpha (same exponent as N*), fitted in log-log to every target shown
        la, ly = np.log(x / 100), np.log(y)
        alpha, lc = np.polyfit(la, ly, 1); resid = ly - (lc + alpha * la)
        se = np.sqrt(resid.var(ddof=2) / ((la - la.mean()) ** 2).sum())
        xs = np.linspace(x.min(), x.max(), 50); lx = np.log(xs / 100)
        band = np.sqrt(resid.var(ddof=2) * (1 / len(la) + (lx - la.mean()) ** 2 / ((la - la.mean()) ** 2).sum()))   # 1-sigma band of the fitted mean
        ax.fill_between(xs, np.exp(lc + alpha * lx - band), np.exp(lc + alpha * lx + band), color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, np.exp(lc + alpha * lx), ls="-", lw=4.8, color=col, zorder=4)
        fit_handle = Line2D([], [], ls="-", lw=4.8, color=col, label=f"power-law fit  $B^* \\propto \\mathrm{{acc}}^{{{alpha:.1f}}}$")
        print(f"B*(acc) fit: B* = {np.exp(lc):,.0f} acc^{alpha:.3f} (±{se:.3f});  B*(35%) = {np.exp(lc) * 0.35 ** alpha:,.0f}, B*(50%) = {np.exp(lc) * 0.5 ** alpha:,.0f}")
        full, part = f[f["all_complete"]], f[~f["all_complete"]]; hollow_any |= len(part) > 0
        ax.scatter(full["target"] * 100, full["nstar_complete"] / div, s=9.5 ** 2 * 1.5, color=col, edgecolors=SURFACE, linewidths=1.2, zorder=5)
        ax.scatter(part["target"] * 100, part["nstar_complete"] / div, s=9.5 ** 2 * 1.5, facecolors=SURFACE, edgecolors=col, linewidths=2.4, zorder=5)
    ax.set_yscale("log")
    lo, hi = min(ys) / 1.12, max(ys) * 1.08; ax.set_ylim(lo, hi)
    yt = [m * 10 ** e for e in range(3, 7) for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)]
    ax.yaxis.set_major_locator(FixedLocator([v for v in yt if lo <= v <= hi])); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    tx = np.arange(np.ceil(TARGETS[0] * 100 / 5) * 5, TARGETS[-1] * 100 + 0.01, 5)
    ax.set_xticks(tx); ax.set_xticklabels([f"{t:.0f}%" for t in tx]); ax.set_xlim(TARGETS[0] * 100 - 0.6, TARGETS[-1] * 100 + 0.6)
    ax.set_xlabel("target accuracy", color=INK, fontsize=LABEL); ax.set_ylabel("critical batch size $B^*$ (prompts)", color=INK, fontsize=LABEL)
    hs = [fit_handle] if len(SERIES) == 1 else []
    if hollow_any:
        hs.append(Line2D([], [], marker="o", ls="", ms=12, mfc=SURFACE, mec=INK2, mew=2.4, label="LR bracket open at one batch size"))
    if hs:
        ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=TICK - 2, labelcolor=INK, handlelength=1.6, handletextpad=0.5, borderaxespad=0.2)
    out = png_out.replace(".png", "_paper.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE); fig.savefig(os.path.join(pdf_dir, stem + "_paper.pdf"), facecolor=SURFACE)
    print("wrote", out)
for ser in SERIES:
    f = tab[tab["series"] == ser["name"]]
    full = f[f["all_complete"]]
    print(f"\n{ser['name']}: fully bracketed at " + (", ".join(f"{r.target*100:g}%" for r in full.itertuples()) or "no target"))
    print("   " + "  ".join(f"{r.target*100:g}%:{r.nstar_complete:,.0f}[{r.n_complete}/{r.n_cols}]" for r in f.itertuples() if hasattr(r, "nstar_complete") and not np.isnan(r.nstar_complete)))
