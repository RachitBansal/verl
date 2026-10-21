"""The doubling corollary against the data: from a configuration (B, K), is doubling the prompts at least as good as
doubling the rollouts? Both land on 2BK sequences per step.

Corollary: eps_opt(2B, K) >= eps_opt(B, 2K), so steps to target should satisfy S(2B, K) <= S(B, 2K). With the joint fit
of Theorem 1 (S = S0 (1 + P/B + R/(BK))) the predicted ratio is
    S(B, 2K) / S(2B, K) = (1 + P/B + R/(2BK)) / (1 + P/(2B) + R/(2BK)),
which exceeds 1 by an amount that grows with P/B relative to the rollout term.

Every (B, K) cell with a measured crossing is collected from the sweeps (fastest LR per cell, KL 1e-3; pre-fix runs
admitted only where the paper admits them, i.e. the K = 16 prompt sweep and K <= 16 at B = 128). A starting cell is
usable when both (2B, K) and (B, 2K) are measured.

  left:  S(B, 2K) against S(2B, K) for each usable start, with the diagonal; the corollary puts every point above it
  right: the measured ratio against the ratio the joint fit predicts, per start

Usage: python plot_doubling_corollary.py csv/steps_to_50_seqs.csv png/doubling_corollary.png [--no-b512] [--bracketed-only]\n  --bracketed-only keeps only starts whose three cells each have a complete LR bracket; otherwise those are drawn hollow / faded
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE, ORANGE = "#1f77b4", "#d66300"
SHORT_STEPS = 200

csv_in, png_out = sys.argv[1], sys.argv[2]
NO_B512 = "--no-b512" in sys.argv
df = pd.read_csv(csv_in)
COL = "steps_to_50_interp"
df = df[np.isclose(df["kl"].astype(float), 1e-3)].dropna(subset=["seqs", "lr"]).copy()
df = df[~df["downsample"].astype(bool)]
# no length pre-filter: a short run that never crossed is judged against its column's best crossing below
df["fixed"] = df["fixed"].astype(bool)
# admit pre-fix runs only where the paper does
ok_prefix = (df["n"] == 16) | ((df["bsz"] == 128) & (df["n"] <= 16))
df = df[df["fixed"] | ok_prefix]

BRACKETED_ONLY = "--bracketed-only" in sys.argv
cells, bracket = {}, {}
for (B, K), g in df.groupby(["bsz", "n"]):
    rows = []
    for lr, gg in g.groupby("lr"):
        gg = gg.sort_values([COL, "last_val_step"], na_position="last", ascending=[True, False]); p = gg.iloc[0]
        rows.append(dict(lr=lr, steps=p[COL], longest=gg["last_val_step"].max(), running=(gg["state"] == "running").any()))
    d = pd.DataFrame(rows).sort_values("lr").reset_index(drop=True); hit = d.dropna(subset=["steps"])
    if hit.empty:
        continue
    ib = hit["steps"].idxmin(); ref = d.loc[ib, "steps"]
    cells[(int(B), int(K))] = float(ref)
    verdict = lambda r: "hit" if not np.isnan(r["steps"]) else ("never" if r["longest"] >= min(ref, SHORT_STEPS) else "inconclusive")
    d["v"] = d.apply(verdict, axis=1)
    side = lambda j: "open" if j < 0 or j >= len(d) else ("inconclusive" if d.loc[j, "v"] == "inconclusive" else "complete")
    lo, up = side(ib - 1), side(ib + 1)
    bracket[(int(B), int(K))] = "open" if "open" in (lo, up) else "inconclusive" if "inconclusive" in (lo, up) else "complete"

# joint-fit scales (plot_theorem_joint.py): S0, P, R
S0, P, R = 4.77, 1271.0, 54866.0
rows = []
for (B, K), S in cells.items():
    a, b = cells.get((2 * B, K)), cells.get((B, 2 * K))
    if a is None or b is None:
        continue
    in_sweeps = lambda b, k: k in (16, 64) or b in (128, 1024)        # the four sweeps the paper reports
    extra = not (in_sweeps(2 * B, K) and in_sweeps(B, 2 * K))
    pred = (1 + P / B + R / (2 * B * K)) / (1 + P / (2 * B) + R / (2 * B * K))
    sts = [bracket[(B, K)], bracket[(2 * B, K)], bracket[(B, 2 * K)]]
    allb = all(x == "complete" for x in sts)
    rows.append(dict(B=B, K=K, N2=2 * B * K, S_start=S, S_2B=a, S_2K=b, ratio=b / a, pred_ratio=pred, uses_b512=extra,
                     bracket_start=sts[0], bracket_2B=sts[1], bracket_2K=sts[2], all_bracketed=allb))
t = pd.DataFrame(rows).sort_values(["B", "K"])
if NO_B512:
    t = t[~t["uses_b512"]]
if BRACKETED_ONLY:
    t = t[t["all_bracketed"]]
t.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), "doubling_corollary_table.csv"), index=False)

plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 13, "legend.fontsize": 10.5})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1.15]))
for ax in (axL, axR):
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.7); ax.tick_params(which="both", colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

# left: S(B,2K) vs S(2B,K)
lo, hi = t[["S_2B", "S_2K"]].min().min() / 1.5, t[["S_2B", "S_2K"]].max().max() * 1.5
axL.plot([lo, hi], [lo, hi], ls=":", lw=1.2, color=MUTED, zorder=1)
axL.fill_between([lo, hi], [lo, hi], [hi, hi], color=BLUE, alpha=0.05, zorder=0)
for _, r in t.iterrows():
    mk = dict(marker="o", s=90, color=BLUE, edgecolors=SURFACE, linewidths=1.2) if r["all_bracketed"] else dict(marker="o", s=90, facecolors="none", edgecolors=BLUE, linewidths=1.6)
    axL.scatter([r["S_2B"]], [r["S_2K"]], zorder=5, **mk)
    axL.annotate(f"from ({r['B']:.0f}, {r['K']:.0f})", (r["S_2B"], r["S_2K"]), xytext=(8, 4), textcoords="offset points", fontsize=10, color=INK2)
axL.set_xscale("log"); axL.set_yscale("log"); axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
axL.set_xlabel("steps to 50% after doubling prompts,  S(2B, K)", color=INK); axL.set_ylabel("steps to 50% after doubling rollouts,  S(B, 2K)", color=INK)
axL.set_title("same 2BK sequences per step; corollary: every point on or above the diagonal" + ("\n(starts whose three configurations all have a complete LR bracket)" if BRACKETED_ONLY else ""), loc="left", fontsize=11.5, color=INK)
axL.text(0.03, 0.95, "doubling prompts\nis faster", transform=axL.transAxes, ha="left", va="top", fontsize=10.5, color=BLUE)
axL.text(0.97, 0.05, "doubling rollouts\nis faster", transform=axL.transAxes, ha="right", va="bottom", fontsize=10.5, color=INK2)

# right: measured vs predicted ratio per start
labels = [f"({r['B']:.0f}, {r['K']:.0f})" for _, r in t.iterrows()]
x = np.arange(len(t))
axR.bar(x - 0.18, t["ratio"], width=0.36, color=BLUE, label="measured  S(B, 2K) / S(2B, K)", zorder=3, alpha=[0.45 if u else 1 for u in t["uses_b512"]] if False else 1)
axR.bar(x + 0.18, t["pred_ratio"], width=0.36, color=ORANGE, label=f"Theorem 1 with the joint-fit scales  (σ²_prompt = {P:,.0f}, σ²_rollout = {R / 1000:.0f}k)", zorder=3)
for i, (_, r) in enumerate(t.iterrows()):
    if not r["all_bracketed"]:
        axR.bar([i - 0.18, i + 0.18], [r["ratio"], r["pred_ratio"]], width=0.36, color=SURFACE, alpha=0.55, zorder=4)
axR.axhline(1, color=INK2, lw=1, zorder=2)
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=10.5, rotation=35, ha="right"); axR.set_xlabel("starting configuration  (B, K)", color=INK)
axR.set_ylabel("steps ratio: double rollouts / double prompts", color=INK); axR.set_ylim(0, max(2.2, t[["ratio", "pred_ratio"]].max().max() * 1.15))
axR.set_title("above 1: doubling prompts reaches the target in fewer steps", loc="left", fontsize=11.5, color=INK)
hnd = [Line2D([], [], marker="s", ls="", ms=10, color=BLUE, label="measured  S(B, 2K) / S(2B, K)"),
       Line2D([], [], marker="s", ls="", ms=10, color=ORANGE, label=f"Theorem 1 with the joint-fit scales  (σ²_prompt = {P:,.0f}, σ²_rollout = {R / 1000:.0f}k)")]
if (~t["all_bracketed"]).any():
    hnd.append(Line2D([], [], marker="s", ls="", ms=10, color=BLUE, alpha=0.45, label="faded / hollow: at least one of the three cells lacks a complete LR bracket"))
axR.legend(handles=hnd, loc="upper left", frameon=False, labelcolor=INK)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.25)
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf")), facecolor=SURFACE)
print("wrote", png_out)
print(t.round(2).to_string(index=False))
