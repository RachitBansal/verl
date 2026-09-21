"""How the best learning rate scales with batch size, as a function of target accuracy and KL coef.

For each target T (35% .. 50%, 0.5% steps), KL coef and batch size B (n = 16, B >= MIN_BSZ):
  best LR(B, T) = LR of the run with the fewest interpolated steps to T at that batch.
Then a least-squares fit of log10(best LR) against log2(B) gives the exponent alpha in
  best LR  ∝  B^alpha        (alpha = 1: LR doubles per batch doubling; 0.5: sqrt scaling; 0: flat)
reported per (T, KL) with its standard error and the number of batches that reached T.

Left panel: alpha vs target for both KLs (band = ±1 s.e.).  Right panel: the fitted best-LR lines
at a few targets, with the underlying best-LR points, to show what alpha summarises.

Usage: python plot_lr_slope_vs_target.py csv/val_curves_n16.json png/lr_slope_vs_target.png [--min-bsz 8]
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
KLS = {1e-3: dict(color=BLUE, marker="o", ls="-", label="KL coef 1e-3"),
       1e-2: dict(color=ORANGE, marker="D", ls="--", label="KL coef 1e-2")}
TARGETS = np.round(np.arange(0.35, 0.50 + 1e-9, 0.005), 3)
SHOW_T = [0.35, 0.40, 0.45, 0.50]                 # right panel
RAMP = ["#9ec5f4", "#3987e5", "#184f95", "#0d366b"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
json_in, png_out = args[0], args[1]
MIN_BSZ = int(sys.argv[sys.argv.index("--min-bsz") + 1]) if "--min-bsz" in sys.argv else 8
runs = json.load(open(json_in))


def crossing(steps, vals, t):
    for i, v in enumerate(vals):
        if v >= t:
            if i == 0:
                return float(steps[0])
            s0, s1, v0, v1 = steps[i - 1], steps[i], vals[i - 1], vals[i]
            return float(s1) if v1 == v0 else s0 + (s1 - s0) * (t - v0) / (v1 - v0)
    return None


best_rows, fit_rows = [], []
for kl in KLS:
    rs = [r for r in runs if abs(r["kl"] - kl) < 1e-12 and r["bsz"] >= MIN_BSZ]
    for t in TARGETS:
        best = {}
        for r in rs:
            c = crossing(r["steps"], r["vals"], t)
            if c is None or c <= 0:
                continue
            if r["bsz"] not in best or c < best[r["bsz"]][0]:
                best[r["bsz"]] = (c, r["lr"], r["name"])
        for b, (c, lr, name) in sorted(best.items()):
            best_rows.append(dict(kl=kl, target=t, bsz=b, best_lr=lr, steps=c, run=name))
        if len(best) < 3:
            continue
        x = np.log2(np.array(sorted(best), float))
        y = np.log10(np.array([best[b][1] for b in sorted(best)]))
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope_log10_per_doubling, intercept = coef
        n = len(x)
        resid = y - A @ coef
        se = np.sqrt(resid @ resid / (n - 2) / ((x - x.mean()) ** 2).sum()) if n > 2 else np.nan
        alpha = slope_log10_per_doubling / np.log10(2)          # LR ∝ B^alpha
        fit_rows.append(dict(kl=kl, target=t, alpha=alpha, alpha_se=se / np.log10(2), n_bsz=n,
                             bsz_min=int(2 ** x.min()), bsz_max=int(2 ** x.max()),
                             lr_at_bsz128=10 ** (intercept + slope_log10_per_doubling * 7),
                             lr_ratio_per_doubling=2 ** alpha))
best_df, fit = pd.DataFrame(best_rows), pd.DataFrame(fit_rows)
stem = os.path.join(os.path.dirname(os.path.abspath(json_in)), os.path.basename(png_out).replace(".png", ""))
fit.to_csv(stem + "_table.csv", index=False)
best_df.to_csv(stem + "_bestlr_table.csv", index=False)

# ---------------------------------------------------------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.25))
for ax in (axL, axR):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.8); ax.tick_params(which="both", colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

for kl, st in KLS.items():
    f = fit[np.isclose(fit["kl"], kl)].sort_values("target")
    axL.fill_between(f["target"] * 100, f["alpha"] - f["alpha_se"], f["alpha"] + f["alpha_se"], color=st["color"], alpha=0.15, lw=0)
    axL.plot(f["target"] * 100, f["alpha"], color=st["color"], lw=2, ls=st["ls"], zorder=3)
    axL.scatter(f["target"] * 100, f["alpha"], color=st["color"], s=28, marker=st["marker"], edgecolors=SURFACE, linewidths=0.8, zorder=4)
for ref, lab in [(1.0, "LR ∝ B  (linear)"), (0.5, "LR ∝ √B"), (0.0, "flat")]:
    axL.axhline(ref, color=MUTED, lw=1, ls=":", zorder=1)
    axL.text(50.3, ref, lab, color=INK2, fontsize=9, va="center", ha="left")
axL.set_xlim(34.5, 53.5); axL.set_xlabel("target AIME 1983–2024 accuracy (%)", color=INK2)
axL.set_ylabel("exponent α in  best LR ∝ batch^α", color=INK2)
axL.set_title("how fast the best LR grows with batch size", loc="left", color=INK, fontsize=12)
axL.legend(handles=[Line2D([], [], color=s["color"], lw=2, ls=s["ls"], marker=s["marker"], mec=SURFACE, label=s["label"]) for s in KLS.values()]
           + [Line2D([], [], color=MUTED, lw=6, alpha=0.3, label="±1 s.e. of the fit")],
           loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2)

# right: best-LR points and fitted lines at a few targets, KL 1e-3 solid / KL 1e-2 dashed, colour = target
for ti, t in enumerate(SHOW_T):
    for kl, st in KLS.items():
        b = best_df[np.isclose(best_df["kl"], kl) & np.isclose(best_df["target"], t)].sort_values("bsz")
        f = fit[np.isclose(fit["kl"], kl) & np.isclose(fit["target"], t)]
        if b.empty or f.empty:
            continue
        axR.scatter(b["bsz"], b["best_lr"], color=RAMP[ti], s=30, marker=st["marker"], edgecolors=SURFACE, linewidths=0.7, alpha=0.9, zorder=3)
        xs = np.array([b["bsz"].min(), b["bsz"].max()], float)
        a, l128 = f["alpha"].iloc[0], f["lr_at_bsz128"].iloc[0]
        axR.plot(xs, l128 * (xs / 128) ** a, color=RAMP[ti], lw=1.8, ls=st["ls"], zorder=2)
axR.set_xscale("log", base=2); axR.set_yscale("log")
bs = sorted(best_df["bsz"].unique()); axR.set_xticks(bs); axR.set_xticklabels([str(int(v)) for v in bs], fontsize=9)
axR.set_xlabel("batch size (prompts per step)", color=INK2); axR.set_ylabel("best learning rate", color=INK2)
axR.set_title("best LR per batch and the fitted power law", loc="left", color=INK, fontsize=12)
axR.legend(handles=[Line2D([], [], color=RAMP[i], lw=2, label=f"target {int(t * 100)}%") for i, t in enumerate(SHOW_T)]
           + [Line2D([], [], color=INK2, lw=1.5, ls="-", marker="o", mec=SURFACE, label="KL 1e-3"),
              Line2D([], [], color=INK2, lw=1.5, ls="--", marker="D", mec=SURFACE, label="KL 1e-2")],
           loc="lower right", frameon=False, fontsize=9, labelcolor=INK2, ncol=2)
fig.suptitle(f"GRPO on-policy, n = 16: scaling of the best learning rate with batch size, by target accuracy  (batches ≥ {MIN_BSZ})",
             color=INK, fontsize=12.5, x=0.02, ha="left")
fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.12, wspace=0.25)
fig.savefig(png_out, dpi=150, facecolor=SURFACE)
print("wrote", png_out)
for kl in KLS:
    f = fit[np.isclose(fit["kl"], kl)]
    print(f"\nKL {kl:g}: alpha (LR ∝ B^alpha), n batches")
    print("  " + "  ".join(f"{r.target*100:g}%:{r.alpha:.2f}±{r.alpha_se:.2f}({r.n_bsz})" for r in f.itertuples()))
