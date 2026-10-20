"""Paper figure: steps to 50% AIME vs sequences trained per step, for three ways of growing the step.

  batch sweep, n = 16  - plain GRPO, 16 rollouts per prompt, batch size B varied (fastest LR at each B)
  rollout sweep        - plain GRPO, B = 128 prompts, rollouts per prompt n varied (fastest LR at each n)
  batch sweep, n = 64  - plain GRPO, 64 rollouts per prompt, batch size B varied (fastest LR at each B; fixed loss scaling only)

All series use KL coef 1e-3 and the fastest available run per point (pre-fix or fixed loss scaling;
for n = 32/64/128 only the fixed-code reruns are eligible because the pre-fix runs sat in Adam's eps regime).
No pre/post-fix comparison, no downsampling series. Writes the PNG next to the other figures and the PDF to plots/paper/.

--fit: least-squares fit (in log steps) of the critical-batch form  S(N) = S_min (1 + N*/N)  to each series, where N is
sequences per step; the fitted curves are drawn dashed and N* (the critical number of sequences, at which steps are twice
S_min) is annotated, also as B* = N*/n prompts for the batch sweeps and n* = N*/128 rollouts for the rollout sweep.

Usage: python plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_paper.png [--fit] [--knee TOL] [--b512] [--b1024] [--x-prompts] [--kl2]
  --kl2 adds the K = 16 batch sweep at KL coef 1e-2 as a separate series (light blue squares; fitted with --fit). All other series stay KL 1e-3.\n  --b512 adds the second rollout sweep (B = 512, K varied, fixed code); runs that have not crossed are hollow at their last step.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE, TEAL, VIOLET = "#1f77b4", "#d66300", "#2ca02c"   # tab:blue, a deeper tab:orange, tab:green for the n=16 sweep, n=64 sweep, rollout sweep
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}          # n where a pre-fix run is an honest measurement
FIXED_ONLY_N = {32, 64, 128}               # n where only fixed-code reruns count

csv_in, png_out = sys.argv[1], sys.argv[2]
FIT = "--fit" in sys.argv[3:]
# --knee TOL: marker at (TOL - 1) N*, where the fitted curve is TOL x the perfect-scaling asymptote S_min N*/N.
# The default TOL = 2 is the fitted N* itself, which the paper calls N_crit (steps = 2 S_min); TOL = 1.25 instead
# marks where steps first exceed perfect scaling by 25%, the visible end of the linear regime.
KNEE_TOL = float(sys.argv[sys.argv.index("--knee") + 1]) if "--knee" in sys.argv else 2.0
B512 = "--b512" in sys.argv[3:]          # add the second rollout sweep: plain GRPO at B = 512 with K varied (fixed code)
B1024 = "--b1024" in sys.argv[3:]        # add the third rollout sweep: plain GRPO at B = 1024 with K varied (fixed code)
XPROMPTS = "--x-prompts" in sys.argv[3:]  # x-axis = prompts per step B instead of sequences B K (rollout sweeps collapse to one column)
RED = "#d62728"                          # tab:red for that series
PURPLE = "#9467bd"                       # tab:purple for the B = 1024 rollout sweep
KL2 = "--kl2" in sys.argv[3:]            # add the K = 16 batch sweep at KL 1e-2 as its own series (never pooled with KL 1e-3)
LIGHTBLUE = "#8fb8e8"                    # for that series, as in plot_nstar_vs_target.py
df = pd.read_csv(csv_in)


def fit_cbs(seqs, steps):
    """S = S_min (1 + N*/N), least squares in log S. For fixed N* the optimal log S_min is the mean residual, so the fit
    is a 1-D search over log N*. Returns (S_min, N*, rms log residual)."""
    N = np.asarray(seqs, float); y = np.log(np.asarray(steps, float))
    best = None
    for ln_ns in np.linspace(np.log(8), np.log(2 ** 26), 6000):
        g = np.log1p(np.exp(ln_ns) / N)
        ln_smin = (y - g).mean()
        rms = np.sqrt(((y - g - ln_smin) ** 2).mean())
        if best is None or rms < best[2]:
            best = (float(np.exp(ln_smin)), float(np.exp(ln_ns)), float(rms))
    return best


def fmt_k(v):
    """two significant figures, thousands as k"""
    if v >= 1e5:
        return f"{v / 1000:.0f}k"
    return f"{v / 1000:.2g}k" if v >= 1000 else f"{float(f'{v:.2g}'):.0f}"
COL = "steps_to_50_interp" if "steps_to_50_interp" in df else "steps_to_50"
df_all = df.dropna(subset=["seqs", "lr"]).copy()
df = df[np.isclose(df["kl"].astype(float), 1e-3)].dropna(subset=["seqs", "lr"]).copy()
df["seqs"] = df["seqs"].astype(int)
df = df[~df["downsample"].astype(bool)]
long_enough = (df["last_val_step"].fillna(0) >= SHORT_STEPS) | df[COL].notna()
df = df[long_enough]
# a run validated every step ("_dense") that crossed replaces the coarsely validated runs of the same (n, B, LR) cell,
# whose crossing was interpolated over a 10- or 25-step bracket; applies to every series
_cell = ["n", "bsz", "lr"]
_dkey = set(df.loc[df["name"].str.contains("_dense") & df[COL].notna(), _cell].astype(str).agg("|".join, axis=1))
df = df[~(df[_cell].astype(str).agg("|".join, axis=1).isin(_dkey) & ~df["name"].str.contains("_dense"))]


def fastest_per(src, key):
    """fastest run at each value of `key` (any LR); returns hits and, separately, keys with no crossing."""
    rows = []
    for k, s in src.groupby(key):
        s = s.sort_values([COL, "last_val_step"], na_position="last", ascending=[True, False])
        p = s.iloc[0]
        rows.append(dict(key=k, seqs=int(p["seqs"]), lr=p["lr"], steps=p[COL], last=p["last_val_step"],
                         max_val=p["max_val"], run=p["name"]))
    out = pd.DataFrame(rows).sort_values("seqs")
    return out.dropna(subset=["steps"]), out[out["steps"].isna()]


batch_src = df[df["n"] == 16]
batch, _ = fastest_per(batch_src, "bsz")
# the same sweep at KL 1e-2 (K = 16 only; the other sweeps have no KL 1e-2 runs)
if KL2:
    kl2_src = df_all[np.isclose(df_all["kl"].astype(float), 1e-2) & (df_all["n"] == 16) & ~df_all["downsample"].astype(bool)]
    kl2_src = kl2_src[(kl2_src["last_val_step"].fillna(0) >= SHORT_STEPS) | kl2_src[COL].notna()]
    kl2_src = kl2_src.assign(seqs=kl2_src["seqs"].astype(int))
    bkl2, _ = fastest_per(kl2_src, "bsz")
else:
    bkl2 = batch.iloc[0:0]

roll_src = df[(df["bsz"] == 128) & (df["n"].isin(VALID_N_PREFIX) | (df["fixed"].astype(bool) & df["n"].isin(FIXED_ONLY_N)))]
roll, roll_never = fastest_per(roll_src, "n")

# batch sweep at n = 64 (fixed code only; every run is post-fix)
b64_src = df[(df["n"] == 64) & df["fixed"].astype(bool)]
# (per-step-validated "_dense" reruns already replace the coarse runs of their cell, see above)
b64, b64_never = fastest_per(b64_src, "bsz") if len(b64_src) else (roll.iloc[0:0], roll.iloc[0:0])

# second rollout sweep at B = 512 (fixed code only); points that have not crossed are kept separately
if B512:
    r512_src = df[(df["bsz"] == 512) & df["fixed"].astype(bool) & (df["n"] != 16) | ((df["bsz"] == 512) & (df["n"] == 16) & np.isclose(df["kl"].astype(float), 1e-3))]
    r512, r512_never = fastest_per(r512_src, "n")
else:
    r512, r512_never = roll.iloc[0:0], roll.iloc[0:0]
if B1024:
    r1k_src = df[(df["bsz"] == 1024) & df["fixed"].astype(bool) & np.isclose(df["kl"].astype(float), 1e-3)]
    r1k, r1k_never = fastest_per(r1k_src, "n")
else:
    r1k, r1k_never = roll.iloc[0:0], roll.iloc[0:0]

table = pd.concat([batch.assign(series="batch_sweep_n16"), roll.assign(series="rollout_sweep_bsz128"),
                   r512.assign(series="rollout_sweep_bsz512"), r512_never.assign(series="rollout_sweep_bsz512_never"),
                   r1k.assign(series="rollout_sweep_bsz1024"), r1k_never.assign(series="rollout_sweep_bsz1024_never"),
                   roll_never.assign(series="rollout_sweep_bsz128_never"),
                   b64.assign(series="batch_sweep_n64"), b64_never.assign(series="batch_sweep_n64_never"),
                   bkl2.assign(series="batch_sweep_n16_kl1e-2")])
table.to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)),
                          os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

if XPROMPTS:    # re-express every x as prompts per step; the "seqs" column is what the plotting code reads
    batch["seqs"] = batch["key"].astype(int)
    if len(b64):
        b64["seqs"] = b64["key"].astype(int)
    if len(bkl2):
        bkl2["seqs"] = bkl2["key"].astype(int)
    for d_, B_ in ((roll, 128), (roll_never, 128), (r512, 512), (r512_never, 512), (r1k, 1024), (r1k_never, 1024)):
        if len(d_):
            d_["seqs"] = B_

# ------------------------------------------------------------------ figure
TOKENS_PER_SEQ = 1024 + 3072       # data.max_prompt_length + data.max_response_length in on_policy*.sh: the token budget per sequence
plt.rcParams.update({"font.size": 17, "axes.labelsize": 18, "legend.fontsize": 15})
fig, ax = plt.subplots(figsize=(10.6, 6.3), facecolor=SURFACE, layout="constrained")
ax.set_facecolor(SURFACE)

# perfect 1/sequences scaling through the smallest batch point
b0 = batch.iloc[0]
xmax = max(batch["seqs"].max(), roll["seqs"].max(), r1k["seqs"].max() if len(r1k) else 0, b64["seqs"].max() if len(b64) else 0)
xs = np.array([batch["seqs"].min() / 2.5, xmax * 1.6])
ax.plot(xs, b0["steps"] * b0["seqs"] / xs, ls=":", lw=1.2, color=MUTED, zorder=1)
# the same reference along the K = 64 curve, anchored on its own smallest batch point
if len(b64):
    c0 = b64.iloc[0]
    xs64 = np.array([c0["seqs"] / 2.5, xmax * 1.6])
    ax.plot(xs64, c0["steps"] * c0["seqs"] / xs64, ls=":", lw=1.2, color=TEAL, alpha=0.55, zorder=1)

# markers only (no joining lines); the fitted curves, when asked for, are the lines
if len(b64):
    ax.scatter(b64["seqs"], b64["steps"], s=120, marker="h", color=TEAL, edgecolors=SURFACE, linewidths=1.2, zorder=4)
ax.scatter(roll["seqs"], roll["steps"], s=120, marker="p", color=VIOLET, edgecolors=SURFACE, linewidths=1.2, zorder=5)
if len(r512):
    ax.scatter(r512["seqs"], r512["steps"], s=100, marker="D", color=RED, edgecolors=SURFACE, linewidths=1.2, zorder=5.5)
    for _, r in r512.iterrows():
        ax.annotate(f"K={int(r['key'])}", (r["seqs"], r["steps"]), xytext=(9, 7), textcoords="offset points", ha="left", fontsize=11, color=RED, zorder=8)
    for _, r in r512_never.iterrows():       # did not reach the target: hollow marker at its last validated step
        ax.scatter([r["seqs"]], [r["last"]], s=100, marker="D", facecolors="none", edgecolors=RED, linewidths=1.6, zorder=5.5)
        ax.annotate(f"K={int(r['key'])}: {r['max_val']:.0%} after {int(r['last'])} steps", (r["seqs"], r["last"]), xytext=(9, 7),
                    textcoords="offset points", ha="left", fontsize=10, color=RED, zorder=8)
if len(r1k):
    ax.scatter(r1k["seqs"], r1k["steps"], s=100, marker="v", color=PURPLE, edgecolors=SURFACE, linewidths=1.2, zorder=5.5)
    for _, r in r1k_never.iterrows():
        ax.scatter([r["seqs"]], [r["last"]], s=100, marker="v", facecolors="none", edgecolors=PURPLE, linewidths=1.6, zorder=5.5)
        ax.annotate(f"K={int(r['key'])}: {r['max_val']:.0%} after {int(r['last'])} steps", (r["seqs"], r["last"]), xytext=(0, -15),
                    textcoords="offset points", ha="center", fontsize=10, color=PURPLE, zorder=8)
if len(bkl2):
    ax.scatter(bkl2["seqs"], bkl2["steps"], s=100, marker="s", color=LIGHTBLUE, edgecolors=SURFACE, linewidths=1.2, zorder=6)
ax.scatter(batch["seqs"], batch["steps"], s=90, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=7)


# --fit: critical-batch fits per series, solid in the series colour, with N* annotated
fits = []
if FIT:
    series = [("more prompts, K = 16", batch, BLUE, 16, "B*"), ("more rollouts, B = 128", roll, VIOLET, 128, "K*")]
    if len(b64):
        series.append(("more prompts, K = 64", b64, TEAL, 64, "B*"))
    if len(r512) >= 3:
        series.append(("more rollouts, B = 512", r512, RED, 512, "K*"))
    if len(r1k) >= 3:
        series.append(("more rollouts, B = 1024", r1k, PURPLE, 1024, "K*"))
    if len(bkl2) >= 3:
        series.append(("more prompts, K = 16, KL 1e-2", bkl2, LIGHTBLUE, 16.01, "B*"))   # per = 16.01: K = 16, tagged apart from the KL 1e-3 series
    series = [t for t in series if t[1]["seqs"].nunique() >= 3]     # a rollout sweep has one x in prompts mode: nothing to fit
    for label, d, col, per, unit in series:
        smin, ns, rms = fit_cbs(d["seqs"], d["steps"])
        fits.append(dict(series=label, s_min=smin, n_star_seqs=ns, per=per, star_unit=unit, star=ns / round(per), rms_log=rms, n_points=len(d)))
        fx = np.geomspace(d["seqs"].min() / 1.5, d["seqs"].max() * 1.5, 200)
        ax.plot(fx, smin * (1 + ns / fx), ls="-", lw=1.8, color=col, alpha=0.9, zorder=3)
    # N_crit of each series is the fitted parameter N* (the batch at which steps are twice S_min). With --knee TOL the
    # marker moves to (TOL - 1) N* instead, where the curve is TOL x perfect scaling; marked on the curve with a
    # dotted vertical to the axis, labelled in the series colour
    ymin = min(batch["steps"].min(), roll["steps"].min(), r1k["steps"].min() if len(r1k) else np.inf, b64["steps"].min() if len(b64) else np.inf) / 4
    # label each critical batch in its own units, placed in open space and tied to the circle by an arrow;
    # position = (x factor, y factor) applied to the circle's coordinates
    LABEL_AT = {16: (0.5, 0.34), 64: (2.2, 5.2), 128: (2.7, 2.1)}   # K=16 label raised so it clears the tick labels with the fitted-N* knee
    if KNEE_TOL < 2 and not KL2:
        LABEL_AT[1024] = (1.3, 0.38)                                  # paper Figure 1 (--b1024 --knee 1.25): below-right of its circle
    if KL2:
        LABEL_AT[16.01] = (2.4, 0.42)                                 # the KL 1e-2 label below-right of its circle, clear of the B = 1024 curve
        LABEL_AT[16] = (0.36, 0.3)                                    # the K = 16 label further down-left, clear of the purple N_crit label
    for f in fits:
        n_c = (KNEE_TOL - 1) * f["n_star_seqs"]; s_c = f["s_min"] * (1 + f["n_star_seqs"] / n_c)
        f["n_crit_seqs"] = n_c; f["crit"] = n_c / round(f["per"])
    # paper series: each critical batch labelled in its own units, in open space, tied to its circle by an arrow (LABEL_AT);
    # other series (--b512/--b1024) or prompts mode: a horizontal label beside the dotted line in the band under the curves,
    # one level per line (leftmost highest) so they never overlap
    order = sorted(fits, key=lambda f: f["n_crit_seqs"])
    levels = [ymin * 1.25 * 1.45 ** i for i in range(len(order))][::-1]
    for f, y_lab in zip(order, levels):
        col = {16: BLUE, 128: VIOLET, 64: TEAL, 512: RED, 1024: PURPLE, 16.01: LIGHTBLUE}[f["per"]]
        n_c = f["n_crit_seqs"]; s_c = f["s_min"] * (1 + f["n_star_seqs"] / n_c)
        unit = "B" if round(f["per"]) in (16, 64) else "K"
        ax.axvline(n_c, ls=":", lw=2.6, color=col, zorder=2.5)
        ax.plot([n_c], [s_c], marker="o", ms=13, mfc=SURFACE, mec=col, mew=2.2, ls="", zorder=9)
        if f["per"] in LABEL_AT and not XPROMPTS:
            fx, fy = LABEL_AT[f["per"]]
            ax.annotate(f"${unit}_\\mathrm{{crit}} \\approx {fmt_k(f['crit'])}$", xy=(n_c, s_c), xytext=(n_c * fx, s_c * fy),
                        textcoords="data", ha="center", va="center", fontsize=17, color=col, zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, shrinkA=2, shrinkB=9, zorder=9))
        else:
            txt = f"$B_\\mathrm{{crit}}$ ≈ {fmt_k(n_c)}" if XPROMPTS else f"$N_\\mathrm{{crit}}$ ≈ {fmt_k(n_c)}  (${unit}$ ≈ {fmt_k(f['crit'])})"
            ax.annotate(txt, (n_c, y_lab), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=13, color=col, zorder=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_ylim(bottom=ymin)
    pd.DataFrame(fits).to_csv(os.path.join(os.path.dirname(os.path.abspath(csv_in)), os.path.basename(png_out).replace(".png", "_fit_table.csv")), index=False)

ax.set_xscale("log", base=2); ax.set_yscale("log")
ticks = sorted(set(batch["seqs"]) | set(roll["seqs"]) | set(b64["seqs"]) | set(r512["seqs"]) | set(r512_never["seqs"]) | set(r1k["seqs"]) | set(r1k_never["seqs"]) | set(bkl2["seqs"]))
ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" if t < 1024 else f"{t // 1024}k" for t in ticks], fontsize=15)   # 1k, 2k, ... keep the larger type from colliding
ax.set_xlabel("prompts per optimizer step  B" if XPROMPTS else "sequences per optimizer step  (prompts B × rollouts K)", color=INK)
ax.set_ylabel("steps to target AIME 1983–2024", color=INK)
ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)     # very light grid, no border
ax.set_axisbelow(True)
ax.tick_params(which="both", colors=INK, length=0)
for sp in ax.spines.values():
    sp.set_visible(False)


def fmt_tokens(v):
    return f"{v / 1e3:.0f}k" if v < 1e6 else (f"{v / 1e6:.1f}M" if v < 1e7 else f"{v / 1e6:.0f}M")


# secondary x-axis: maximum tokens per optimizer step = sequences × (max prompt + max response length); not defined in prompts mode
if not XPROMPTS:
    top = ax.secondary_xaxis("top", functions=(lambda n: n * TOKENS_PER_SEQ, lambda t: t / TOKENS_PER_SEQ))
    top.set_xticks([t * TOKENS_PER_SEQ for t in ticks]); top.set_xticklabels([fmt_tokens(t * TOKENS_PER_SEQ) for t in ticks], fontsize=14)
    top.set_xlabel(f"maximum tokens per optimizer step  (sequences × {TOKENS_PER_SEQ:,} tokens)", color=INK, fontsize=15.5)
    top.tick_params(which="both", colors=INK, length=0)
    for sp in top.spines.values():
        sp.set_visible(False)

nstar = {}       # critical batches are annotated on the curves, not in the legend
legend = [Line2D([], [], marker="o", ls="", ms=8, color=BLUE, mec=SURFACE, label="scaling prompts, fixed K = 16" + nstar.get(16, ""))]
if len(bkl2):
    legend.append(Line2D([], [], marker="s", ls="", ms=8, color=LIGHTBLUE, mec=SURFACE, label="scaling prompts, fixed K = 16, KL 1e-2"))
if len(b64):
    legend.append(Line2D([], [], marker="h", ls="", ms=9, color=TEAL, mec=SURFACE, label="scaling prompts, fixed K = 64" + nstar.get(64, "")))
legend.append(Line2D([], [], marker="p", ls="", ms=9, color=VIOLET, mec=SURFACE, label="scaling rollouts, fixed B = 128" + nstar.get(128, "")))
if len(r512):
    legend.append(Line2D([], [], marker="D", ls="", ms=8, color=RED, mec=SURFACE, label="scaling rollouts, fixed B = 512" + nstar.get(512, "")))
if len(r1k):
    legend.append(Line2D([], [], marker="v", ls="", ms=8, color=PURPLE, mec=SURFACE, label="scaling rollouts, fixed B = 1024" + nstar.get(1024, "")))
if len(r512_never) or len(r1k_never):
    legend.append(Line2D([], [], marker="D", ls="", ms=8, mfc="none", mec=INK2, mew=1.6, label="did not reach the target, shown at its last step"))
# series in the lower left; reference lines (perfect scaling, fit, critical batch) in a row below the axes
refs = [Line2D([], [], ls=":", lw=1.2, color=MUTED, label="perfect scaling, S ∝ 1/B" if XPROMPTS else "perfect scaling, S ∝ 1/N")]
if FIT:
    refs.append(Line2D([], [], ls="-", lw=1.8, color=INK2, label="fit, S = S_min (1 + B*/B)" if XPROMPTS else "fit, S = S_min (1 + N*/N)"))
    refs.append(Line2D([], [], marker="o", ms=9, mfc=SURFACE, mec=INK2, mew=1.8, ls=":", lw=1.6, color=INK2,
                       label="critical batch, the fitted N*" if KNEE_TOL == 2.0 else f"critical batch, {int(round((KNEE_TOL - 1) * 100))}% above perfect scaling"))
ax.add_artist(ax.legend(handles=legend, loc="lower left", frameon=False, labelcolor=INK))
fig.legend(handles=refs, loc="outside lower center", ncol=3, frameon=False, labelcolor=INK, fontsize=13,
           columnspacing=1.3, handlelength=2.0, handletextpad=0.5)
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper")   # plots/paper/
os.makedirs(pdf_dir, exist_ok=True)
pdf_out = os.path.join(pdf_dir, os.path.basename(png_out).replace(".png", ".pdf"))
fig.savefig(pdf_out, facecolor=SURFACE)
print("wrote", png_out, "and", pdf_out)
print("\nbatch sweep:\n", batch[["key", "seqs", "lr", "steps", "run"]].to_string(index=False))
print("\nrollout sweep:\n", roll[["key", "seqs", "lr", "steps", "run"]].to_string(index=False))
print("\nrollout sweep, never:\n", roll_never[["key", "seqs", "last", "max_val"]].to_string(index=False))
if len(b64):
    print("\nbatch sweep, K = 64:\n", b64[["key", "seqs", "lr", "steps", "run"]].to_string(index=False))
    print("\nbatch sweep, K = 64, never:\n", b64_never[["key", "seqs", "last", "max_val"]].to_string(index=False))
if fits:
    print("\ncritical-batch fits S = S_min (1 + N*/N):\n", pd.DataFrame(fits).to_string(index=False))
