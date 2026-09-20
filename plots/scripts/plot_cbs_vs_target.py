"""Critical batch size vs target accuracy, from the full n=16 validation curves.

For each target T (30% .. 55% in 0.5% steps) and each KL coef:
  S(B)  = fastest interpolated steps-to-T over all runs at batch B (any LR)
  For each pair of adjacent tested batches B1 < B2 (normally a doubling), perfect scaling
  would give S(B2) = S(B1) * B1 / B2, i.e. steps halve per doubling.  The doubling "fails"
  when steps fall by less than the required fraction:  S(B2) / S(B1) > RATIO per doubling
  (RATIO = 0.7 by default: a doubling "pays off" only if it cuts steps by >= 30%; 0.5 is the
  strict "must halve" rule).  Gaps larger than one doubling are normalised per doubling.
  CBS   = B2 of the first doubling that starts STREAK consecutive failures (--rule first, default;
          --streak 1 by default; --streak 2 requires two noisy doublings in a row), or the smallest B2 from
          which every later doubling fails (--rule sustained).  If no doubling fails the
          point is censored at "> largest batch that reached T".
Default rule (--rule slope): for each tested batch B, least-squares slope of log2(steps) vs log2(batch) over
the forward window [B, B*2^WINDOW] (WINDOW = 2 doublings); CBS = the first B whose window slope is shallower
than -1 + SLOPE_TOL (SLOPE_TOL = 0.3, i.e. slope > -0.7). Averaging over a window keeps one noisy doubling
from triggering it. Outliers: batches below MIN_BSZ (8) are ignored, and a batch whose fastest run is slower
than the next-smaller batch's is dropped (--keep-nonmonotone disables). --rule first / sustained are the per-doubling rules above.
An orange line adds the McCandlish-style fit B_crit from E(B) = B*S(B) = E_min*(1 + B/B_crit)
(least squares in log E; needs >= 4 batches).

Usage: python plot_cbs_vs_target.py csv/val_curves_n16.json png/cbs_vs_target.png
         [--rule slope|first|sustained] [--window 2] [--slope-tol 0.3] [--min-bsz 8] [--keep-nonmonotone]
         [--ratio 0.7] [--streak 1]
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
KLS = [1e-3, 1e-2]
TARGETS = np.round(np.arange(0.30, 0.55 + 1e-9, 0.005), 3)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
json_in, png_out = args[0], args[1]
RULE = sys.argv[sys.argv.index("--rule") + 1] if "--rule" in sys.argv else "slope"
WINDOW = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 2          # doublings per slope window
SLOPE_TOL = float(sys.argv[sys.argv.index("--slope-tol") + 1]) if "--slope-tol" in sys.argv else 0.3   # off the line when slope > -1 + tol
MIN_BSZ = int(sys.argv[sys.argv.index("--min-bsz") + 1]) if "--min-bsz" in sys.argv else 8              # ignore tiny, single-run batches
DROP_NONMONO = "--keep-nonmonotone" not in sys.argv     # drop a batch that is slower than the next-smaller batch (outlier)
RATIO = float(sys.argv[sys.argv.index("--ratio") + 1]) if "--ratio" in sys.argv else 0.7
STREAK = int(sys.argv[sys.argv.index("--streak") + 1]) if "--streak" in sys.argv else 1   # consecutive failing doublings required

runs = json.load(open(json_in))


def crossing(steps, vals, t):
    for i, v in enumerate(vals):
        if v >= t:
            if i == 0:
                return float(steps[0])
            s0, s1, v0, v1 = steps[i - 1], steps[i], vals[i - 1], vals[i]
            return float(s1) if v1 == v0 else s0 + (s1 - s0) * (t - v0) / (v1 - v0)
    return None


def fit_bcrit(S):
    """E(B) = E_min (1 + B/B_crit): grid over log B_crit, closed-form E_min in log space."""
    B = np.array(sorted(S), float); E = B * np.array([S[b] for b in sorted(S)])
    if len(B) < 4:
        return None
    best = None
    for lb in np.linspace(np.log2(B.min()) - 2, np.log2(B.max()) + 4, 600):
        bc = 2 ** lb
        logEmin = np.mean(np.log(E) - np.log1p(B / bc))
        res = np.sum((np.log(E) - logEmin - np.log1p(B / bc)) ** 2)
        if best is None or res < best[0]:
            best = (res, bc)
    return best[1]


rows = []
for kl in KLS:
    rs = [r for r in runs if abs(r["kl"] - kl) < 1e-12]
    bszs = sorted({r["bsz"] for r in rs})
    for t in TARGETS:
        S = {}
        for b in bszs:
            c = [crossing(r["steps"], r["vals"], t) for r in rs if r["bsz"] == b]
            c = [x for x in c if x is not None and x > 0]
            if c:
                S[b] = min(c)
        if len(S) < 2:
            continue
        S = {b: v for b, v in S.items() if b >= MIN_BSZ}
        if DROP_NONMONO:                     # a larger batch that needs MORE steps than the next-smaller one is an outlier
            for b1, b2 in zip(sorted(S)[:-1], sorted(S)[1:]):
                if S.get(b2, 0) >= S.get(b1, np.inf):
                    S.pop(b2, None)
        if len(S) < 2:
            continue
        Bs = sorted(S)
        maxB = max(Bs)                       # largest batch that actually reached this target
        # per-doubling step ratio for each adjacent pair, normalised when a batch is missing
        pairs = []
        for b1, b2 in zip(Bs[:-1], Bs[1:]):
            n_doublings = np.log2(b2 / b1)
            per_doubling = (S[b2] / S[b1]) ** (1.0 / n_doublings)
            pairs.append((b1, b2, per_doubling))
        fails = [r > RATIO for _, _, r in pairs]
        cbs = None
        slope_at = None
        if RULE == "slope":
            # average log-log slope over a forward window of WINDOW doublings [B, B*2^WINDOW] (least squares over
            # every tested batch in the window); CBS = first B whose window slope is shallower than -1 + SLOPE_TOL
            lB, lS = np.log2(Bs), np.log2([S[b] for b in Bs])
            for i, b in enumerate(Bs):
                m = (lB >= lB[i] - 1e-9) & (lB <= lB[i] + WINDOW + 1e-9)
                if m.sum() < 2 or lB[m].max() < lB[i] + WINDOW - 1e-9:
                    continue                      # window not fully covered by tested batches
                slope = np.polyfit(lB[m], lS[m], 1)[0]
                if slope > -1 + SLOPE_TOL:
                    cbs, slope_at = b, slope
                    break
        elif RULE == "first":   # first doubling that starts a run of STREAK consecutive failures (noise guard)
            for i in range(len(pairs)):
                if all(fails[i:i + STREAK]) and len(fails[i:i + STREAK]) == STREAK:
                    cbs = pairs[i][1]
                    break
        else:   # sustained: first b2 such that this and every later doubling fails
            for i, (_, b2, _) in enumerate(pairs):
                if all(fails[i:]):
                    cbs = b2
                    break
        ratio_at = next((r for (_, b2, r) in pairs if b2 == cbs), None)
        rows.append(dict(kl=kl, target=t, rule=RULE, ratio_threshold=RATIO, streak=STREAK, window=WINDOW, slope_tol=SLOPE_TOL, min_bsz=MIN_BSZ,
                         cbs=cbs, step_ratio_at_cbs=ratio_at, slope_at_cbs=slope_at,
                         censored=cbs is None, max_bsz_tested=maxB, bcrit_fit=fit_bcrit(S), n_bsz=len(Bs),
                         steps_by_bsz=" ".join(f"{b}:{S[b]:.0f}" for b in Bs),
                         step_ratio_by_doubling=" ".join(f"{b1}->{b2}:{r:.2f}" for b1, b2, r in pairs)))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(os.path.dirname(os.path.abspath(json_in)),
                        os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, facecolor=SURFACE)
for ax, kl in zip(axes, KLS):
    ax.set_facecolor(SURFACE)
    d = tab[np.isclose(tab["kl"], kl)].sort_values("target")
    maxB = int(d["max_bsz_tested"].max())            # this panel's largest batch that reached any target
    cens_y = int(tab["max_bsz_tested"].max()) * 2    # shared censor row across panels (y axis is shared)
    hit, cen = d[~d["censored"]], d[d["censored"]]
    ax.plot(hit["target"] * 100, hit["cbs"], color=BLUE, lw=1.4, alpha=0.55, zorder=2)
    ax.scatter(hit["target"] * 100, hit["cbs"], s=44, color=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=4)
    ax.scatter(cen["target"] * 100, [cens_y] * len(cen), s=44, facecolors=SURFACE, edgecolors=BLUE,
               linewidths=1.4, zorder=4)
    f = d.dropna(subset=["bcrit_fit"])
    ax.plot(f["target"] * 100, f["bcrit_fit"], color=ORANGE, lw=2, zorder=3)
    ax.axhline(cens_y, color=GRID, lw=1, ls=":", zorder=1)
    ax.text(30.2, cens_y * 1.12, f"hollow = no departure found up to batch {maxB}" if RULE == "slope" else f"hollow = every doubling up to batch {maxB} pays off",
            color=INK2, fontsize=9, va="bottom")
    ax.set_yscale("log", base=2)
    yt = [2 ** k for k in range(2, int(np.log2(cens_y)) + 1)] + [cens_y * 2]
    ax.set_yticks(yt)
    ax.set_yticklabels([str(v) if v < cens_y else ("beyond tested" if v == cens_y else f"{v} (fit only)") for v in yt])
    ax.set_ylim(3, cens_y * 2.6)
    ax.set_xlim(29.5, 55.5)
    ax.set_xlabel("target AIME 1983-2024 accuracy (%)", color=INK2)
    ax.set_title(f"KL coef {kl:g}", color=INK, fontsize=12, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
axes[0].set_ylabel("critical batch size (prompts per step)", color=INK2)
pct = int(round((1 - RATIO) * 100))
what = "fails to halve steps-to-target" if abs(RATIO - 0.5) < 1e-9 else f"cuts steps-to-target by less than {pct}%"
if RULE == "slope":
    cbs_label = (f"CBS: first batch B whose average log-log slope over the next {WINDOW} doublings is shallower than "
                 f"{-1 + SLOPE_TOL:g}  (perfect scaling = -1)")
else:
    cbs_label = f"CBS: batch reached by the first doubling that {what}  (steps(2B) > {RATIO:g} x steps(B))"
handles = [Line2D([], [], marker="o", color=BLUE, lw=0, markersize=7, label=cbs_label),
           Line2D([], [], color=ORANGE, lw=2, label="B_crit fit: batch x steps = E_min (1 + batch/B_crit)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2,
           bbox_to_anchor=(0.5, 0.0), columnspacing=2.5)
rule_txt = ((f"first of {STREAK} consecutive doublings that stop paying off" if STREAK > 1 else "first doubling that stops paying off")
            if RULE == "first" else "first doubling from which no later doubling pays off" if RULE == "sustained"
            else f"first batch where the {WINDOW}-doubling average slope of steps vs batch leaves -1 by more than {SLOPE_TOL:g}"
                 f"  (batches < {MIN_BSZ}" + (" and non-monotone outliers" if DROP_NONMONO else "") + " ignored)")
fig.suptitle("GRPO on-policy, n = 16 rollouts: critical batch size vs target accuracy", color=INK, fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.925, rule_txt if RULE == "slope" else f"{rule_txt};  a doubling pays off when it " + ("halves steps-to-target" if abs(RATIO - 0.5) < 1e-9 else f"cuts steps-to-target by at least {pct}%"),
         color=INK2, fontsize=10, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.92))
fig.savefig(png_out, dpi=150, facecolor=SURFACE)
print("wrote", png_out)

for kl in KLS:
    d = tab[np.isclose(tab["kl"], kl)].sort_values("target")
    print(f"\nKL {kl:g}  rule={RULE}" + (f" window={WINDOW} slope_tol={SLOPE_TOL}" if RULE == "slope" else f" ratio={RATIO} streak={STREAK}"))
    print("  " + " ".join(f"{t*100:g}%->{'>' + str(int(m)) if c else int(b)}"
                          for t, b, c, m in zip(d["target"], d["cbs"].fillna(0), d["censored"], d["max_bsz_tested"])))
