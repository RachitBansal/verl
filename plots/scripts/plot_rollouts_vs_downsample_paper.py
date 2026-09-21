"""Paper figure: more rollouts vs downsampled rollouts, at B = 128 prompts, on a sequences-trained axis.

  rollout sweep - plain GRPO, B = 128, n rollouts per prompt varied (fastest LR at each n); n = 32/64/128
                  from the fixed-code reruns only
  downsampling  - N = 64 rollouts generated per prompt, K kept for training, B = 128, fixed code
                  (fastest LR at each K)
x = sequences trained per optimizer step = 128 * n (rollouts) or 128 * K (downsampling), so the two
series are compared at equal training cost per step.

Bracket check for every K: all LRs tried at that K are drawn beside the best one -- a smaller filled
marker if that LR also crossed (slower), a hollow marker at its last validated step if it finished
without crossing after running at least as long as the best crossing, and an x if it is inconclusive
(still running or too short). The ring around the best LR is green when a worse LR exists on both
sides, amber when a neighbour is inconclusive, red when no LR was tried on one side.

Usage: python plot_rollouts_vs_downsample_paper.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_paper.png
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
VIOLET, ORANGE = "#4a3aa7", "#eb6834"
GOOD, WARN, BAD = "#008300", "#b8860b", "#c62828"
B = 128
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}
FIXED_ONLY_N = {32, 64, 128}

csv_in, png_out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_in)
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df = df[np.isclose(df["kl"].astype(float), 1e-3) & (df["bsz"] == B)].dropna(subset=["lr"]).copy()
df["last_val_step"] = df["last_val_step"].fillna(0)
df["downsample"] = df["downsample"].astype(bool); df["fixed"] = df["fixed"].astype(bool)


def per_config(src, key):
    """one row per (key, lr): fastest crossing, longest attempt, running flag."""
    rows = []
    for (k, lr), s in src.groupby([key, "lr"]):
        hit = s.dropna(subset=[COL])
        rows.append(dict(k=int(k), lr=lr, steps=hit[COL].min() if len(hit) else np.nan,
                         longest=s["last_val_step"].max(), running=(s["state"] == "running").any(),
                         max_val=s["max_val"].max(), attempts=len(s),
                         run=(hit.sort_values(COL).iloc[0]["name"] if len(hit) else s.sort_values("last_val_step").iloc[-1]["name"])))
    return pd.DataFrame(rows)


def columns(cfg):
    """per k: best LR and bracket status."""
    out = []
    for k, s in cfg.groupby("k"):
        s = s.sort_values("lr").reset_index(drop=True)
        hits = s.dropna(subset=["steps"])
        if hits.empty:
            out.append(dict(k=k, best=None, status="no crossing", items=s)); continue
        bi = hits["steps"].idxmin(); best = s.loc[bi]
        ref = best["steps"]
        verdict = lambda r: "hit" if not np.isnan(r["steps"]) else ("never" if (not r["running"] and r["longest"] >= min(ref, SHORT_STEPS)) else "inconclusive")
        s["verdict"] = s.apply(verdict, axis=1)
        side = lambda j: "open" if j < 0 or j >= len(s) else ("inconclusive" if s.loc[j, "verdict"] == "inconclusive" else "complete")
        lo, up = side(bi - 1), side(bi + 1)
        status = "open" if "open" in (lo, up) else "inconclusive" if "inconclusive" in (lo, up) else "complete"
        out.append(dict(k=k, best=best, lower=lo, upper=up, status=status, items=s))
    return out


roll_src = df[(~df["downsample"]) & (df["n"].isin(VALID_N_PREFIX) | (df["fixed"] & df["n"].isin(FIXED_ONLY_N)))]
roll = columns(per_config(roll_src.rename(columns={"n": "kk"}), "kk"))
ds = columns(per_config(df[df["downsample"] & df["fixed"]], "dsk"))

# table
rows = []
for series, cols in [("rollouts_n", roll), ("downsample_K", ds)]:
    for c in cols:
        for _, r in c["items"].iterrows():
            rows.append(dict(series=series, k=c["k"], seqs=B * c["k"], lr=r["lr"], steps=r["steps"], longest=r["longest"], max_val=r["max_val"],
                             verdict=r.get("verdict", "no crossing"), is_best=(c["best"] is not None and r["lr"] == c["best"]["lr"]),
                             bracket=c["status"], attempts=r["attempts"], run=r["run"]))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 13, "legend.fontsize": 10.5})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.22))
for ax in (axL, axR):
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.7); ax.tick_params(which="both", colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
STAT_COL = {"complete": GOOD, "inconclusive": WARN, "open": BAD}

# ---- left: the two curves, ring = bracket status of the best LR
for cols, color, marker, tag, side in [(roll, VIOLET, "p", "n", "left"), (ds, ORANGE, "o", "K", "right")]:
    best = sorted((B * c["k"], c["best"]["steps"], c) for c in cols if c["best"] is not None)
    axL.plot([b[0] for b in best], [b[1] for b in best], lw=2, color=color, zorder=4)
    dx, ha = (10, "left") if side == "right" else (-10, "right")
    for x, y, c in best:
        axL.scatter([x], [y], s=110, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=6)
        axL.scatter([x], [y], s=300, marker=marker, facecolors="none", edgecolors=STAT_COL[c["status"]], linewidths=2.2, zorder=7)
        axL.annotate(f"{tag}={c['k']}", (x, y), xytext=(dx, 7 if side == "left" else -13), textcoords="offset points", ha=ha, fontsize=10.5, color=color, zorder=8)
    never = [c for c in cols if c["best"] is None]
    if never:
        txt = "; ".join(f"{tag}={c['k']} never reaches 50% ({c['items']['max_val'].max():.0%} after {int(c['items']['longest'].max())} steps)" for c in never)
        axL.text(0.99, 0.97 if side == "left" else 0.91, txt, transform=axL.transAxes, fontsize=9.5, color=color, ha="right", va="top")
axL.set_xscale("log", base=2); axL.set_yscale("log")
xs = sorted({B * c["k"] for c in roll + ds if c["best"] is not None})
axL.set_xticks(xs); axL.set_xticklabels([f"{v:,}" if v < 10000 else f"{v // 1024}k" for v in xs], fontsize=11)
axL.set_xlabel(f"sequences trained per step  ({B} prompts × rollouts trained per prompt)", color=INK)
axL.set_ylabel("steps to 50% AIME 1983–2024", color=INK)
axL.set_title("steps to target, fastest LR per point", loc="left", fontsize=13, color=INK)

# ---- right: the LR ladder - every LR tried at each n / K, ring on the best coloured by bracket status
def ladder(cols, color, marker, nudge):
    for c in cols:
        x = c["k"] * 2.0 ** nudge
        for _, r in c["items"].iterrows():
            v = r.get("verdict", "inconclusive") if c["best"] is not None else ("inconclusive" if r["running"] or r["longest"] < SHORT_STEPS else "never")
            if v == "hit":
                axR.scatter([x], [r["lr"]], s=70, marker=marker, color=color, edgecolors=SURFACE, linewidths=0.9, zorder=5)
            elif v == "never":
                axR.scatter([x], [r["lr"]], s=60, marker=marker, facecolors="none", edgecolors=color, linewidths=1.5, zorder=5)
            else:
                axR.scatter([x], [r["lr"]], s=55, marker="x", color=MUTED, linewidths=1.5, zorder=5)
        if c["best"] is not None:
            axR.scatter([x], [c["best"]["lr"]], s=260, marker=marker, facecolors="none", edgecolors=STAT_COL[c["status"]], linewidths=2.2, zorder=7)
    b = sorted((c["k"] * 2.0 ** nudge, c["best"]["lr"]) for c in cols if c["best"] is not None)
    axR.plot([q[0] for q in b], [q[1] for q in b], lw=1.3, ls="--", color=color, alpha=0.7, zorder=3)
ladder(roll, VIOLET, "p", -0.12)
ladder(ds, ORANGE, "o", +0.12)
axR.set_xscale("log", base=2); axR.set_yscale("log")
ks = sorted({c["k"] for c in roll + ds})
axR.set_xticks(ks); axR.set_xticklabels([str(k) for k in ks], fontsize=11)
axR.set_xlim(min(ks) / 1.8, max(ks) * 1.8)
axR.set_xlabel("rollouts trained per prompt  (n, or K of 64 generated)", color=INK)
axR.set_ylabel("learning rate", color=INK)
axR.set_title("every LR tried; ring = fastest, colour = bracket", loc="left", fontsize=13, color=INK)

legend = [
    Line2D([], [], marker="p", ls="-", lw=2, ms=9, color=VIOLET, mec=SURFACE, label="more rollouts: n generated and trained (B = 128)"),
    Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=ORANGE, mec=SURFACE, label="downsampling: 64 generated, K trained (B = 128)"),
    Line2D([], [], marker="o", ls="", ms=7, color=INK2, mec=SURFACE, label="LR reached 50%"),
    Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=INK2, mew=1.4, label="LR did not reach 50% (finished, ran at least as long as the best)"),
    Line2D([], [], marker="x", ls="", ms=7, color=MUTED, mew=1.4, label="inconclusive: running or too short"),
    Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=GOOD, mew=2, label="best LR bracketed by a worse LR on both sides"),
    Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=WARN, mew=2, label="a neighbouring LR is inconclusive"),
    Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=BAD, mew=2, label="no LR tried on one side"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.005), columnspacing=2.2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.26, wspace=0.22)
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper")
os.makedirs(pdf_dir, exist_ok=True)
pdf_out = os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf"))
fig.savefig(pdf_out, facecolor=SURFACE)
print("wrote", png_out, "and", pdf_out)
for name, cols in [("rollouts", roll), ("downsample", ds)]:
    print(f"\n{name}:")
    for c in cols:
        b = c["best"]
        print(f"  k={c['k']:<4d} best={'—' if b is None else f'{b[chr(108)+chr(114)]:g} -> {b[chr(115)+chr(116)+chr(101)+chr(112)+chr(115)]:.0f}'}  bracket={c['status']:12s} tried=" +
              ", ".join(f"{r['lr']:g}{'' if r.get('verdict')=='hit' else '×' if r.get('verdict')=='never' else '?'}" for _, r in c["items"].iterrows()))
