"""Critical batch size vs target accuracy, from the full n=16 validation curves.

For each target T (30% .. 55% in 0.5% steps) and each KL coef:
  S(B)  = fastest interpolated steps-to-T over all runs at batch B (any LR)
  line  = perfect 1/batch scaling S_perf(B) = S(B0) * B0 / B through the anchor batch B0
  CBS   = the smallest B > B0 whose S(B) exceeds tol * S_perf(B), i.e. the first point
          that leaves the 1/batch line; if no tested batch leaves it, CBS is censored at
          "> largest batch tested".
Rule (--rule): "first" (default) = the literal first batch above the anchor that leaves the line;
"sustained" = the smallest batch from which every larger tested batch is off the line (ignores an
isolated under-tuned batch). An orange line adds the McCandlish-style fit B_crit from
E(B) = B*S(B) = E_min * (1 + B/B_crit) (least squares in log E; needs >= 4 batches).
Anchor (--anchor): "min" (default) = the most prompt-efficient batch (smallest B*S(B)),
"smallest" = the smallest batch that reaches T (matches the dotted line in steps_vs_bsz*.png).

Usage: python plot_cbs_vs_target.py csv/val_curves_n16.json png/cbs_vs_target.png [--anchor min|smallest] [--rule first|sustained]
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
RAMP = ["#9ec5f0", "#2a78d6", "#0d3a7a"]          # blue sequential: loose -> strict tolerance
TOLS = [2.0, 1.5, 1.25]                            # "not near" = steps > tol x perfect-line steps
MARK = {2.0: "o", 1.5: "s", 1.25: "D"}
KLS = [1e-3, 1e-2]
TARGETS = np.round(np.arange(0.30, 0.55 + 1e-9, 0.005), 3)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
json_in, png_out = args[0], args[1]
ANCHOR = sys.argv[sys.argv.index("--anchor") + 1] if "--anchor" in sys.argv else "min"
RULE = sys.argv[sys.argv.index("--rule") + 1] if "--rule" in sys.argv else "first"
ORANGE = "#eb6834"

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

runs = json.load(open(json_in))

def crossing(steps, vals, t):
    for i, v in enumerate(vals):
        if v >= t:
            if i == 0:
                return float(steps[0])
            s0, s1, v0, v1 = steps[i - 1], steps[i], vals[i - 1], vals[i]
            return float(s1) if v1 == v0 else s0 + (s1 - s0) * (t - v0) / (v1 - v0)
    return None

rows = []
for kl in KLS:
    rs = [r for r in runs if abs(r["kl"] - kl) < 1e-12]
    bszs = sorted({r["bsz"] for r in rs})
    maxB = max(bszs)
    for t in TARGETS:
        S = {}
        for b in bszs:
            c = [crossing(r["steps"], r["vals"], t) for r in rs if r["bsz"] == b]
            c = [x for x in c if x is not None and x > 0]
            if c:
                S[b] = min(c)
        if len(S) < 2:
            continue
        Bs = sorted(S)
        b0 = min(Bs, key=lambda b: b * S[b]) if ANCHOR == "min" else Bs[0]
        bcrit = fit_bcrit(S)
        above = [b for b in Bs if b > b0]
        ratio = {b: S[b] / (S[b0] * b0 / b) for b in above}   # actual steps / perfect-line steps
        for tol in TOLS:
            off = [ratio[b] > tol for b in above]
            cbs = None
            if RULE == "first":
                cbs = next((b for b, o in zip(above, off) if o), None)
            else:   # sustained: smallest b such that every tested batch >= b is off the line
                for i, b in enumerate(above):
                    if all(off[i:]):
                        cbs = b
                        break
            rows.append(dict(kl=kl, target=t, tol=tol, anchor_bsz=b0, anchor_steps=S[b0],
                             cbs=cbs, ratio_at_cbs=ratio.get(cbs), censored=cbs is None, max_bsz_tested=maxB,
                             bcrit_fit=bcrit, n_bsz=len(Bs),
                             steps_by_bsz=" ".join(f"{b}:{S[b]:.0f}" for b in Bs)))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(os.path.dirname(os.path.abspath(json_in)),
                        os.path.basename(png_out).replace(".png", "_table.csv")), index=False)

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, facecolor=SURFACE)
for ax, kl in zip(axes, KLS):
    ax.set_facecolor(SURFACE)
    d = tab[np.isclose(tab["kl"], kl)]
    maxB = int(d["max_bsz_tested"].max())
    cens_y = maxB * 2
    for tol, col in zip(TOLS, RAMP):
        s = d[d["tol"] == tol].sort_values("target")
        hit = s[~s["censored"]]
        cen = s[s["censored"]]
        # slight horizontal stagger so coincident tolerances stay visible
        dx = {2.0: -0.0012, 1.5: 0.0, 1.25: 0.0012}[tol]
        ax.plot(hit["target"] * 100 + dx * 100, hit["cbs"], color=col, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter(hit["target"] * 100 + dx * 100, hit["cbs"], s=44, marker=MARK[tol], color=col,
                   edgecolors=SURFACE, linewidths=0.8, zorder=4)
        ax.scatter(cen["target"] * 100 + dx * 100, [cens_y] * len(cen), s=44, marker=MARK[tol],
                   facecolors=SURFACE, edgecolors=col, linewidths=1.4, zorder=4)
    f = d[d["tol"] == TOLS[0]].sort_values("target").dropna(subset=["bcrit_fit"])
    ax.plot(f["target"] * 100, f["bcrit_fit"], color=ORANGE, lw=2, zorder=3)
    ax.axhline(cens_y, color=GRID, lw=1, ls=":", zorder=1)
    ax.text(30.2, cens_y * 1.12, f"hollow = still on the line at batch {maxB} (largest tested)",
            color=INK2, fontsize=9, va="bottom")
    ax.set_yscale("log", base=2)
    yt = [2 ** k for k in range(2, int(np.log2(cens_y)) + 1)]
    yt.append(cens_y * 2)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(v) if v < cens_y else (f"> {maxB}" if v == cens_y else f"{v} (fit only)") for v in yt])
    ax.set_ylim(3, cens_y * 2.6)
    ax.set_xlim(29.5, 55.5)
    ax.set_xlabel("target AIME 1983-2024 accuracy (%)", color=INK2)
    ax.set_title(f"KL coef {kl:g}", color=INK, fontsize=12, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    # anchor batch as a thin grey trace along the bottom, for context
    a = d[d["tol"] == TOLS[0]].sort_values("target")
    ax.plot(a["target"] * 100, a["anchor_bsz"], color=MUTED, lw=1, ls="--", zorder=1)
axes[0].set_ylabel("critical batch size (prompts per step)", color=INK2)
handles = [Line2D([], [], marker=MARK[t], color=c, lw=0, markersize=7,
                  label=f"steps > {t:g}x the line") for t, c in zip(TOLS, RAMP)]
handles.append(Line2D([], [], color=ORANGE, lw=2, label="B_crit fit: batch x steps = E_min (1 + batch/B_crit)"))
handles.append(Line2D([], [], color=MUTED, lw=1, ls="--",
                      label=("anchor batch (most prompt-efficient)" if ANCHOR == "min" else "anchor batch (smallest that reaches target)")))
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0), columnspacing=2.5)
anchor_txt = ("1/batch line through the most prompt-efficient batch (min batch x steps)" if ANCHOR == "min"
              else "1/batch line through the smallest batch that reaches the target")
rule_txt = "first batch off the line" if RULE == "first" else "first batch from which every larger batch is off the line"
fig.suptitle("GRPO on-policy, n = 16 rollouts: critical batch size vs target accuracy", color=INK, fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.925, f"{rule_txt};  {anchor_txt}", color=INK2, fontsize=10, ha="left")
fig.tight_layout(rect=(0, 0.09, 1, 0.92))
fig.savefig(png_out, dpi=150, facecolor=SURFACE)
print("wrote", png_out)

# console summary
for kl in KLS:
    d = tab[np.isclose(tab["kl"], kl)]
    print(f"\nKL {kl:g}  (anchor={ANCHOR})")
    for tol in TOLS:
        s = d[d["tol"] == tol].sort_values("target")
        print(f"  tol {tol:g}x: " + " ".join(f"{int(t*100+0.5) if abs(t*100-round(t*100))<1e-6 else t*100:g}%->{'>' + str(int(m)) if c else int(b)}"
                                              for t, b, c, m in zip(s["target"], s["cbs"].fillna(0), s["censored"], s["max_bsz_tested"])))
