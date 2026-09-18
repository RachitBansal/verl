"""Steps to 50% AIME vs sequences trained per step: downsampling vs plain GRPO.

x: sequences reaching the actor per optimizer step (prompts x rollouts trained per
   prompt), log2.  Plain GRPO n=16 at batch b trains 16*b; downsample N=64->K at
   batch 128 trains 128*K.
y: steps to reach 50% AIME 1983-2024 mean@1 (interpolated between the bracketing validation
   readings when the CSV has steps_to_50_interp; else the first checkpoint at/above 50%), log.

Series
  plain GRPO (n = 16, no downsampling), pre-fix sweep, KL 1e-3:
     - headline: fastest LR at each batch size (filled blue circles, solid line)
     - matched-LR: the lr = 1e-5 run at each batch size (hollow blue, dashed)
  plain GRPO n = 16 on the FIXED loss scaling, any LR (filled blue squares, LR labelled)
  downsample (N = 64, K in {2,4,8,16}, bsz 128), fixed loss scaling, KL 1e-3 (orange).
  --prefix: also the PRE-FIX downsample runs (green diamonds). Their effective LR was
     ~0.10/0.13/0.18/0.30 x nominal for K = 1/2/4/16 (Adam eps regime), so the LR
     printed next to each is nominal only.
Runs that have not reached 50% are drawn as open triangles at their last validated
step (a lower bound on steps-to-50%).

Usage: python plot_steps_vs_seqs.py steps_to_50_seqs.csv out.png [--prefix] [--nsweep]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated pair)
VIOLET = "#4a3aa7"                           # slot 7; passes the validator beside blue/orange (CVD dE 24.7)
PREFIX = "#008300"                           # slot 6 green, requested for pre-fix runs; diamond shape is the secondary encoding
SHORT_STEPS = 200
MATCHED_LR = 1e-5

csv_in, png_out = sys.argv[1], sys.argv[2]
SHOW_PREFIX = "--prefix" in sys.argv[3:]
SHOW_NSWEEP = "--nsweep" in sys.argv[3:]      # add the rollout sweep: plain GRPO, bsz 128, n varied
VALID_N = {1, 2, 4, 8, 16}                    # n=32/64 pre-fix runs sat in Adam's eps regime (step factor 0.69/0.46) -> excluded
df = pd.read_csv(csv_in)
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"   # interpolated crossing when available
df = df[np.isclose(df["kl"].astype(float), 1e-3)]
df = df.dropna(subset=["seqs", "lr"])
df["seqs"] = df["seqs"].astype(int)
long_enough = (df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[COL].notna()   # short runs count once they reach 50%


def pick_fastest(s):
    return s.sort_values([COL, "last_val_step"], na_position="last",
                         ascending=[True, False]).iloc[0]


def collapse(src, keys, series, k_from=None):
    rows = []
    for key, s in src.groupby(keys):
        p = pick_fastest(s)
        rows.append(dict(series=series, seqs=int(p["seqs"]), lr=p["lr"],
                         k=int(p[k_from]) if k_from else 16, n=p.get("n"),
                         steps=p[COL], last=p["last_val_step"], max_val=p["max_val"], run=p["name"]))
    return pd.DataFrame(rows).sort_values("seqs") if rows else pd.DataFrame(
        columns=["series", "seqs", "lr", "k", "n", "steps", "last", "max_val", "run"])


# ---- plain GRPO, n = 16, pre-fix sweep --------------------------------------
plain = collapse(df[(~df["downsample"]) & (df["n"] == 16) & (~df["fixed"]) & long_enough], ["seqs", "lr"], "plain")
plain_hit = plain.dropna(subset=["steps"])
matched = plain[np.isclose(plain["lr"], MATCHED_LR)].sort_values("seqs")

# ---- plain GRPO, n = 16, fixed scaling --------------------------------------
pfix = collapse(df[(~df["downsample"]) & (df["n"] == 16) & df["fixed"] & long_enough], ["seqs", "lr"], "plain_fixed")

# headline "fastest LR at each sequence count" is the best available plain run, pre-fix or fixed;
# fixed-code points on it are drawn as squares
all_plain_hit = pd.concat([plain_hit.assign(fixed=False), pfix.dropna(subset=["steps"]).assign(fixed=True)])
best = all_plain_hit.loc[all_plain_hit.groupby("seqs")["steps"].idxmin()].sort_values("seqs")

# ---- downsample, fixed scaling ----------------------------------------------
ds = collapse(df[df["downsample"] & df["fixed"]], ["dsk", "lr"], "downsample", k_from="dsk")
ds_hit = ds.dropna(subset=["steps"])
ds_cens = ds[ds["steps"].isna()]

# ---- downsample, pre-fix (optional) ------------------------------------------
pre = collapse(df[df["downsample"] & (~df["fixed"]) & long_enough], ["dsk", "lr"], "downsample_prefix", k_from="dsk") \
    if SHOW_PREFIX else collapse(df.iloc[0:0], ["dsk", "lr"], "downsample_prefix", k_from="dsk")
pre_hit = pre.dropna(subset=["steps"])
pre_cens = pre[pre["steps"].isna()]

# ---- rollout sweep: plain GRPO at bsz 128 with n varied (optional) ----------------------
nsw_src = df[(~df["downsample"]) & (df["bsz"] == 128) & df["n"].isin(VALID_N) & long_enough] if SHOW_NSWEEP else df.iloc[0:0]
nsw_all = collapse(nsw_src, ["n", "lr"], "nsweep") if len(nsw_src) else collapse(df.iloc[0:0], ["n", "lr"], "nsweep")
if len(nsw_all):
    nsw_all["k"] = nsw_all["n"].astype(int)
    _hit = nsw_all.dropna(subset=["steps"])
    nsw_best = _hit.loc[_hit.groupby("n")["steps"].idxmin()].sort_values("seqs")            # fastest LR per n
    _never_n = set(nsw_all["n"]) - set(_hit["n"])                                           # n where no LR reached 50%
    nsw_never = nsw_all[nsw_all["n"].isin(_never_n)].sort_values("last", ascending=False).drop_duplicates("n")
else:
    nsw_best = nsw_never = nsw_all

pd.concat([best.assign(role="plain_fastest_lr"), matched.assign(role="plain_lr1e-5"),
           pfix.assign(role="plain_fixed_scaling"), ds.assign(role="downsample_fixed"),
           pre.assign(role="downsample_prefix"), nsw_all.assign(role="plain_nsweep_bsz128")]).to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ---- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

b0, s0 = best["seqs"].iloc[0], best["steps"].iloc[0]
allx = pd.concat([plain["seqs"], ds["seqs"], pfix["seqs"], pre["seqs"], nsw_all["seqs"]])
xs = np.array([allx.min() / 1.4, allx.max() * 1.4])
ax.plot(xs, s0 * b0 / xs, ls=":", lw=1.2, color=MUTED, zorder=1)
ax.annotate("perfect scaling (steps ∝ 1/sequences)", (xs[1], s0 * b0 / xs[1]), xytext=(-4, -6), va="top",
            textcoords="offset points", ha="right", fontsize=8.2, color=INK2)

# plain GRPO, fastest LR per batch (headline)
ax.plot(best["seqs"], best["steps"], lw=2, color=BLUE, zorder=3)
bp, bf = best[~best["fixed"]], best[best["fixed"]]
ax.scatter(bp["seqs"], bp["steps"], s=95, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=4)
ax.scatter(bf["seqs"], bf["steps"], s=110, marker="s", color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=4)
for _, r in best.iterrows():
    above = SHOW_NSWEEP and r["seqs"] == 128
    ax.annotate(f"lr {r['lr']:g}", (r["seqs"], r["steps"]), xytext=(0, 9 if above else -15), textcoords="offset points",
                ha="center", fontsize=7.8, color=INK2, zorder=6)

# plain GRPO at the matched LR (1e-5), pre-fix
m_hit = matched.dropna(subset=["steps"])
ax.plot(m_hit["seqs"], m_hit["steps"], lw=1.4, ls="--", color=BLUE, zorder=3)
ax.scatter(m_hit["seqs"], m_hit["steps"], s=95, facecolors=SURFACE, edgecolors=BLUE, linewidths=1.8, zorder=4)
for _, r in matched[matched["steps"].isna()].iterrows():
    ax.scatter([r["seqs"]], [r["last"]], s=80, marker="^", facecolors=SURFACE, edgecolors=BLUE,
               linewidths=1.6, zorder=4)

# plain GRPO, fixed scaling: only points that are NOT the fastest at their x (those sit on the headline as squares)
pf_hit = pfix.dropna(subset=["steps"])
pf_hit = pf_hit[~pf_hit["run"].isin(best["run"])]
ax.scatter(pf_hit["seqs"], pf_hit["steps"], s=110, marker="s", color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=5)
for _, grp in pf_hit.groupby("seqs"):
    for i, (_, r) in enumerate(grp.sort_values("steps", ascending=False).iterrows()):
        ax.annotate(f"lr {r['lr']:g}", (r["seqs"], r["steps"]), xytext=(10, 4 if i % 2 == 0 else -11),
                    textcoords="offset points", fontsize=7.4, color=INK2, zorder=7)

# downsample, fixed scaling
ax.plot(ds_hit["seqs"], ds_hit["steps"], lw=2, color=ORANGE, zorder=5)
ax.scatter(ds_hit["seqs"], ds_hit["steps"], s=95, color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=6)
HIT_POS = {4: dict(xytext=(10, 2), ha="left"), 16: dict(xytext=(-10, -4), ha="right")}
for _, r in ds_hit.iterrows():
    pos = HIT_POS.get(int(r["k"]), dict(xytext=(0, 9), ha="center"))
    ax.annotate(f"K={r['k']}, lr {r['lr']:g}", (r["seqs"], r["steps"]), textcoords="offset points",
                fontsize=7.8, color=INK, zorder=7, **pos)
CENS_POS = {2: dict(xytext=(0, -13), ha="center") if (SHOW_PREFIX or SHOW_NSWEEP) else dict(xytext=(-10, -3), ha="right"),
            8: dict(xytext=(10, -3), ha="left"), 16: dict(xytext=(10, 2), ha="left")}
for _, r in ds_cens.iterrows():
    ax.scatter([r["seqs"]], [r["last"]], s=85, marker="^", facecolors=SURFACE, edgecolors=ORANGE,
               linewidths=1.8, zorder=6)
    pos = CENS_POS.get(int(r["k"]), dict(xytext=(10, 2), ha="left"))
    lr_txt = "" if np.isclose(r["lr"], MATCHED_LR) else f", lr {r['lr']:g}"
    ax.annotate(f"K={r['k']}{lr_txt}: {r['max_val']:.0%} @ {int(r['last'])}", (r["seqs"], r["last"]),
                textcoords="offset points", fontsize=7.4, color=INK2, zorder=7, **pos)

# downsample, pre-fix (green). Shape carries identity too: diamonds (hit) / triangles (not yet).
if SHOW_PREFIX:
    ax.scatter(pre_hit["seqs"], pre_hit["steps"], s=80, marker="D", color=PREFIX, edgecolors=SURFACE,
               linewidths=1.3, zorder=5)
    # label hits only; stack labels for co-located points (same x) to the left, alternating vertical offset
    for seqs, grp in pre_hit.groupby("seqs"):
        grp = grp.sort_values("steps", ascending=False)
        for i, (_, r) in enumerate(grp.iterrows()):
            n_txt = f" (N={int(r['n'])})" if r["n"] != 64 else ""
            if r["k"] == 31:
                pos = dict(xytext=(9, -3), ha="left")
            else:
                pos = dict(xytext=(9, 3 if i % 2 == 0 else -9), ha="left")
            ax.annotate(f"K={r['k']}{n_txt}, lr {r['lr']:g}", (r["seqs"], r["steps"]), textcoords="offset points",
                        fontsize=7.4, color=PREFIX, zorder=7, **pos)
    # censored pre-fix runs: markers only (K is given by x; nominal LRs are in the table CSV)
    ax.scatter(pre_cens["seqs"], pre_cens["last"], s=70, marker="^", facecolors=SURFACE, edgecolors=PREFIX,
               linewidths=1.6, zorder=5)

# rollout sweep (violet pentagons): fastest LR per n at bsz 128; n with no crossing as open pentagon at its longest run
if SHOW_NSWEEP and len(nsw_best):
    ax.plot(nsw_best["seqs"], nsw_best["steps"], lw=2, color=VIOLET, zorder=5)
    ax.scatter(nsw_best["seqs"], nsw_best["steps"], s=120, marker="p", color=VIOLET, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    NSW_POS = {2: dict(xytext=(10, 4), ha="left"), 4: dict(xytext=(-9, -3), ha="right"), 8: dict(xytext=(10, 6), ha="left")}
    for _, r in nsw_best.iterrows():
        if int(r["n"]) == 16:   # same run as the blue headline point at 2048 sequences; already labelled
            continue
        pos = NSW_POS.get(int(r["n"]), dict(xytext=(10, 6), ha="left"))
        lab = f"n={int(r['n'])}" if int(r["n"]) == 4 else f"n={int(r['n'])}, lr {r['lr']:g}"   # n=4 shares its x with K=4; short label fits left of it
        ax.annotate(lab, (r["seqs"], r["steps"]), textcoords="offset points",
                    fontsize=7.8, color=VIOLET, zorder=7, **pos)
    for _, r in nsw_never.iterrows():
        ax.scatter([r["seqs"]], [r["last"]], s=110, marker="p", facecolors=SURFACE, edgecolors=VIOLET, linewidths=1.8, zorder=6)
        ax.annotate(f"n={int(r['n'])}: never ({r['max_val']:.0%} max)", (r["seqs"], r["last"]), xytext=(-10, -12), textcoords="offset points",
                    ha="right", fontsize=7.4, color=VIOLET, zorder=7)

ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = sorted(set(pd.concat([plain["seqs"], ds["seqs"], pfix["seqs"], nsw_all["seqs"]]).unique()))
ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks])
ax.set_xlabel("sequences trained per optimizer step  (prompts × rollouts kept per prompt)", color=INK2, fontsize=10)
ax.set_ylabel("steps to 50% AIME 1983-2024 (interpolated)", color=INK2, fontsize=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
title = "Downsampling (64 generated → K trained, 128 prompts) vs plain GRPO (16 rollouts, varying batch)"
if SHOW_PREFIX:
    title += "\ngreen = pre-fix downsample runs (effective LR ≈ 0.1–0.3× the nominal LR shown)"
if SHOW_NSWEEP:
    title += "\nviolet = plain GRPO at batch 128 with n rollouts varied (pre-fix; n=32/64 excluded, eps-regime)"
ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)

legend = [
    Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=BLUE, mec=SURFACE, label="plain GRPO n=16 — fastest LR at each batch"),
    Line2D([], [], marker="o", ls="--", lw=1.4, ms=8, color=BLUE, mfc=SURFACE, mec=BLUE, mew=1.8, label="plain GRPO n=16 — lr 1e-5 (pre-fix scaling)"),
    Line2D([], [], marker="s", ls="", ms=8, color=BLUE, mec=SURFACE, label="plain GRPO n=16 — fixed scaling (LR labelled)"),
    Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=ORANGE, mec=SURFACE, label="downsample N=64→K — lr 1e-5, fixed scaling"),
]
if SHOW_PREFIX:
    legend.append(Line2D([], [], marker="D", ls="", ms=7, color=PREFIX, mec=SURFACE, label="downsample N=64→K — pre-fix scaling (nominal LR labelled)"))
if SHOW_NSWEEP:
    legend.append(Line2D([], [], marker="p", ls="-", lw=2, ms=9, color=VIOLET, mec=SURFACE, label="plain GRPO, bsz 128, n varied — fastest LR per n"))
legend += [
    Line2D([], [], marker="^", ls="", ms=8, mfc=SURFACE, mec=INK2, mew=1.6, label="not yet at 50% — shown at last validated step"),
    Line2D([], [], ls=":", lw=1.2, color=MUTED, label="perfect 1/sequences scaling"),
]
ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8.0, labelcolor=INK2)
fig.savefig(png_out, dpi=170, bbox_inches="tight", facecolor=SURFACE)

print("plain GRPO, fastest LR per sequences:\n", best[["seqs", "lr", "steps", "run"]].to_string(index=False))
print("\nplain GRPO at lr 1e-5 (pre-fix):\n", matched[["seqs", "steps", "last", "max_val", "run"]].to_string(index=False))
print("\nplain GRPO, fixed scaling:\n", pfix[["seqs", "lr", "steps", "last", "max_val", "run"]].to_string(index=False))
print("\ndownsample (fixed scaling):\n", ds[["seqs", "k", "lr", "steps", "last", "max_val", "run"]].to_string(index=False))
if SHOW_PREFIX:
    print("\ndownsample (pre-fix):\n", pre[["seqs", "k", "n", "lr", "steps", "last", "max_val", "run"]].to_string(index=False))
print("wrote", png_out)
