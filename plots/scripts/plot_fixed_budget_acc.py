"""Accuracy after a FIXED number of prompts trained on, against the prompt batch size, and the batch size where it drops off.

A run at batch size B has trained on D prompts after D / B steps. Its accuracy there is read from the two evaluations
around that step, but only when the step is close to an evaluation: the nearest evaluation must lie within
--max-interp x (D / B) steps (default 0.1), and the step must not fall before the run's first post-training evaluation.
Runs that stopped before D / B steps are not used (whether they crashed or passed the 52% stop threshold).
Per batch size the best learning rate at the budget is the one with the highest usable accuracy.

A batch size is BRACKETED at the budget when the learning rate just below and just above its best one were both tried
and both have a usable reading at the budget with lower accuracy. A neighbour that is unusable at the budget (ended
earlier, or no evaluation near the step) leaves the bracket inconclusive; a side with no learning rate tried is open.
Only bracketed batch sizes enter the drop-off (unbracketed ones are not drawn; --all-points uses every usable one).

Drop-off batch size for a budget: the largest B whose accuracy is within --tol of the best accuracy over all B,
interpolated in log B to the first B below that line (--tol default 0.0164, one s.e. of an AIME reading at 50%).
It is undefined when no tried B falls below the line, or when fewer than --min-points batch sizes are usable.

Usage:
  single budget:  python plot_fixed_budget_acc.py csv/val_curves_n16.json png/fixed_budget_acc_12k.png --prompts 12500
  budget sweep:   python plot_fixed_budget_acc.py csv/val_curves_n16.json png/fixed_budget_acc.png --budgets 1000,2000,4000,8000,16000,32000,64000
                  -> <stem>.png: accuracy vs B, one curve per budget;  <stem>_dropoff.png: drop-off batch size vs budget
Options: --k 16 --kl 1e-3 --tol 0.0164 --max-interp 0.1 --min-bsz 4 --min-points 4 --all-points --epoch-prompts 17917 (marks 1, 2, 4 epochs over the training set)
Writes paper/<stem>.pdf and csv/<stem>_table.csv (per budget, batch size, LR: step, accuracy, usable) and csv/<stem>_dropoff_table.csv.
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
BLUE = "#1f77b4"


def opt(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


json_in, png_out = sys.argv[1], sys.argv[2]
K = opt("--k", 16, int); KL = opt("--kl", 1e-3, float); TOL = opt("--tol", 0.0164, float)
MAX_INTERP = opt("--max-interp", 0.1, float); MIN_BSZ = opt("--min-bsz", 4, int); MIN_PTS = opt("--min-points", 4, int)
EPOCH = opt("--epoch-prompts", 17917, int)      # prompts in the DAPO training set: one pass over the data
BUDGETS = [float(b) for b in opt("--budgets", "", str).split(",") if b] or [opt("--prompts", 12500.0, float)]
SWEEP = "--budgets" in sys.argv
ALL_POINTS = "--all-points" in sys.argv

runs = [r for r in json.load(open(json_in)) if r.get("kl") is not None and abs(r["kl"] - KL) < 1e-12 and r.get("n", K) == K and r["bsz"] >= MIN_BSZ and r["steps"]]
curves = {}
for r in runs:                                   # stitch resumed segments of one run name
    c = curves.setdefault((r["bsz"], float(r["lr"]), r["name"].replace("_resume", "")), {})
    for s, v in zip(r["steps"], r["vals"]):
        c[s] = max(c.get(s, 0), v)


def read_at(st, v, s):
    """accuracy at step s, and whether the reading is trustworthy under the interpolation rule"""
    if s > st[-1]:
        return None, "ended before the budget"
    if st[0] == 0 and len(st) > 1 and s < st[1]:
        return None, "before first evaluation"
    gap = np.min(np.abs(st - s))
    if gap > MAX_INTERP * s and gap > 0.5:
        return None, f"nearest evaluation {gap:.0f} steps away"
    return float(np.interp(s, st, v)), "ok"


rows = []
for D in BUDGETS:
    for (B, lr, name), c in curves.items():
        st = np.array(sorted(c), float); v = np.array([c[s] for s in sorted(c)])
        s = D / B; acc, why = read_at(st, v, s)
        rows.append(dict(budget=D, bsz=int(B), lr=lr, run=name, step_at_budget=s, last_step=int(st[-1]), acc=acc, usable=acc is not None, why=why))
tab = pd.DataFrame(rows)
stem = png_out.replace(".png", ""); csv_dir = os.path.dirname(os.path.abspath(json_in))
tab.sort_values(["budget", "bsz", "lr"]).to_csv(os.path.join(csv_dir, os.path.basename(stem) + "_table.csv"), index=False)

def bracket_of(g, best_lr, best_acc):
    """g: all rows (every LR, usable or not) of one batch size at one budget, one row per LR (best reading per LR)"""
    lrs = sorted(g["lr"].unique()); i = lrs.index(best_lr)
    def side(j):
        if j < 0 or j >= len(lrs):
            return "open"
        r = g[g["lr"] == lrs[j]]
        if r["usable"].any():
            return "worse" if r.loc[r["usable"], "acc"].max() <= best_acc else "better"
        return "inconclusive"
    lo, up = side(i - 1), side(i + 1)
    status = "complete" if lo == up == "worse" else "open" if "open" in (lo, up) else "inconclusive"
    return status, lo, up


best_all, drop = {}, []
for D in BUDGETS:
    rows_b = []
    for B, g in tab[tab["budget"] == D].groupby("bsz"):
        u = g[g["usable"]]
        if u.empty:
            continue
        w = u.sort_values("acc", ascending=False).iloc[0]
        status, lo, up = bracket_of(g, w["lr"], w["acc"])
        rows_b.append(dict(budget=D, bsz=int(B), lr=w["lr"], run=w["run"], step_at_budget=w["step_at_budget"], acc=w["acc"], bracket=status, lower=lo, upper=up,
                           n_lrs=g["lr"].nunique(), bracketed=(status == "complete")))
    best = pd.DataFrame(rows_b).sort_values("bsz").reset_index(drop=True) if rows_b else pd.DataFrame(columns=["budget", "bsz", "lr", "acc", "bracketed"])
    best_all[D] = best
    best = best[best["bracketed"]].reset_index(drop=True) if not ALL_POINTS else best
    rec = dict(budget=D, n_bsz=len(best_all[D]), n_bracketed=int(best_all[D]["bracketed"].sum()) if len(best_all[D]) else 0, B_crit=np.nan, best_acc=np.nan, b_lo=np.nan, b_hi=np.nan)
    if len(best) >= MIN_PTS:
        top = best["acc"].max(); line = top - TOL; ok = best[best["acc"] >= line]
        rec["best_acc"] = top
        if len(ok) and ok["bsz"].max() < best["bsz"].max():
            i = best.index[best["bsz"] == ok["bsz"].max()][0]
            b0, a0 = best.loc[i, "bsz"], best.loc[i, "acc"]; b1, a1 = best.loc[i + 1, "bsz"], best.loc[i + 1, "acc"]
            rec["B_crit"] = float(np.exp(np.log(b0) + (np.log(b1) - np.log(b0)) * (a0 - line) / (a0 - a1))) if a0 != a1 else float(b0)
            rec["b_lo"], rec["b_hi"] = b0, b1
    drop.append(rec)
drop = pd.DataFrame(drop)
pd.concat(best_all.values()).to_csv(os.path.join(csv_dir, os.path.basename(stem) + "_best_table.csv"), index=False)
drop.to_csv(os.path.join(csv_dir, os.path.basename(stem) + "_dropoff_table.csv"), index=False)

plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 13, "legend.fontsize": 10.5})
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.7); ax.tick_params(which="both", colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


fk = lambda x: f"{x:,.0f}" if x < 10000 else f"{x / 1000:,.0f}k"
# ---------------------------------------------------------------- figure 1: accuracy vs B, one curve per budget
fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor=SURFACE); style(ax)
cmap = plt.get_cmap("viridis"); nB = len(BUDGETS)
for j, D in enumerate(BUDGETS):
    best = best_all[D]; col = cmap(0.1 + 0.8 * j / max(nB - 1, 1)) if SWEEP else BLUE
    if best.empty:
        continue
    br = best[best["bracketed"] | ALL_POINTS]
    if br.empty:
        continue
    ax.plot(br["bsz"], br["acc"], lw=1.8, color=col, alpha=0.85, zorder=4)
    ax.scatter(br["bsz"], br["acc"], s=55, color=col, edgecolors=SURFACE, linewidths=1.1, zorder=5, label=f"{fk(D)} prompts")
    if not SWEEP:
        u = tab[(tab["budget"] == D) & tab["usable"]]; oth = u[~u.set_index(["bsz", "lr", "run"]).index.isin(best.set_index(["bsz", "lr", "run"]).index)]
        ax.scatter(oth["bsz"], oth["acc"], s=26, color=col, alpha=0.3, zorder=3)
        for r in best.itertuples():
            ax.annotate(f"{r.lr:g}", (r.bsz, r.acc), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    rec = drop[drop["budget"] == D].iloc[0]
    if not np.isnan(rec["B_crit"]):
        ax.axvline(rec["B_crit"], color=col, lw=1.1, ls=":", alpha=0.9, zorder=2)
allB = sorted(tab[tab["usable"]]["bsz"].unique())
ax.set_xscale("log", base=2); ax.set_xticks(allB); ax.set_xticklabels([fk(b) for b in allB])
shown = pd.concat([b[b["bracketed"] | ALL_POINTS] for b in best_all.values()])
ax.set_ylim(0.2, max(0.56, shown["acc"].max() + 0.03))
allB = sorted(shown["bsz"].unique()); ax.set_xticks(allB); ax.set_xticklabels([fk(b) for b in allB])
ax.set_xlabel(f"prompt batch size B  (K = {K} rollouts per prompt)", color=INK)
ax.set_ylabel("AIME 1983–2024 accuracy after the budget", color=INK)
ax.set_title(f"accuracy after a fixed number of prompts trained on, best learning rate per point, KL {KL:g}" + ("" if SWEEP else f"  ({fk(BUDGETS[0])} prompts)"), loc="left", fontsize=12, color=INK)
hnd, lab = ax.get_legend_handles_labels()
hnd += [Line2D([], [], ls=":", lw=1.1, color=INK2, label="dotted: batch size where accuracy drops off (best − 1 s.e.)" + ("" if ALL_POINTS else ", bracketed points only"))]
if not ALL_POINTS:
    hnd += [Line2D([], [], ls="", label="only batch sizes whose best LR is bracketed by a worse LR on both sides")]
if not SWEEP:
    hnd += [Line2D([], [], marker="o", ls="", ms=5, color=BLUE, alpha=0.3, label="other learning rates")]
ax.legend(handles=hnd, loc="lower left", frameon=False, labelcolor=INK, ncol=2 if SWEEP else 1, fontsize=9.5)
fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12)
fig.savefig(png_out, dpi=200, facecolor=SURFACE); fig.savefig(os.path.join(pdf_dir, os.path.basename(stem) + ".pdf"), facecolor=SURFACE)
print("wrote", png_out)

# ---------------------------------------------------------------- figure 2: drop-off batch size vs budget
if SWEEP:
    fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE); style(ax)
    d = drop.dropna(subset=["B_crit"])
    ax.plot(d["budget"], d["B_crit"], lw=2, color=BLUE, zorder=4)
    ax.scatter(d["budget"], d["B_crit"], s=70, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=5)
    ax.vlines(d["budget"], d["b_lo"], d["b_hi"], color=BLUE, lw=1, alpha=0.5, zorder=3)
    nkey = "n_bsz" if ALL_POINTS else "n_bracketed"
    und = drop[drop["B_crit"].isna() & (drop[nkey] >= MIN_PTS)]
    used = lambda b: best_all[b] if ALL_POINTS else best_all[b][best_all[b]["bracketed"]]
    for r in und.itertuples():
        ax.annotate("no drop-off\nwithin tried B", (r.budget, used(r.budget)["bsz"].max()), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.scatter(und["budget"], [used(b)["bsz"].max() for b in und["budget"]], s=70, marker="^", facecolors="none", edgecolors=INK2, linewidths=1.4, zorder=5)
    few = drop[drop["B_crit"].isna() & (drop[nkey] < MIN_PTS)]
    y_few = min(d["b_lo"].min() if len(d) else 32, 32) / 1.5
    ax.scatter(few["budget"], [y_few] * len(few), s=60, marker="x", color=INK2, linewidths=1.4, zorder=5)
    for r in few.itertuples():
        ax.annotate(f"only {int(getattr(r, nkey))} bracketed\nbatch sizes", (r.budget, y_few), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log", base=2); ax.margins(x=0.1)
    xlo, xhi = ax.get_xlim()
    ax.axvspan(EPOCH, xhi * 10, color=GRID, alpha=0.35, zorder=0)
    for e in (1, 2, 4):
        if xlo <= EPOCH * e <= xhi:
            ax.axvline(EPOCH * e, color=INK2, lw=1, ls="--", zorder=1)
            ax.text(EPOCH * e * 1.03, ax.get_ylim()[0] * 1.08, f"{e} epoch{'s' if e > 1 else ''}\n({EPOCH * e / 1000:,.0f}k prompts)", fontsize=9, color=INK2, va="bottom", ha="left")
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("prompts trained on  (batch size × steps)", color=INK); ax.set_ylabel(f"batch size where accuracy drops off  (K = {K})", color=INK)
    ax.set_title("the batch size that stops being free, against the data budget", loc="left", fontsize=12.5, color=INK)
    yt = [b for b in sorted(tab["bsz"].unique()) if y_few <= b <= max(d["b_hi"].max() * 1.5, 1)] if len(d) else allB
    ax.set_yticks(yt); ax.set_yticklabels([fk(b) for b in yt])
    hnd = [Line2D([], [], marker="o", ls="-", ms=8, color=BLUE, mec=SURFACE, label="drop-off B: largest B within 1 s.e. of the best accuracy, interpolated to the next B"),
           Line2D([], [], ls="-", lw=1, color=BLUE, alpha=0.5, label="bar: the two tried batch sizes it was interpolated between"),
           Line2D([], [], marker="^", ls="", ms=8, mfc="none", mec=INK2, mew=1.4, label="no batch size below the line yet (largest usable B shown)"),
           Line2D([], [], marker="x", ls="", ms=7, color=INK2, mew=1.4, label=f"fewer than {MIN_PTS} bracketed batch sizes: not estimated"),
           Line2D([], [], ls="--", lw=1, color=INK2, label=f"dashed: whole passes over the {EPOCH:,} training prompts (shaded: beyond one epoch)")]
    if not ALL_POINTS:
        hnd.insert(1, Line2D([], [], ls="", label="only batch sizes whose best LR is bracketed by a worse LR on both sides"))
    ax.legend(handles=hnd, loc="upper left", frameon=False, labelcolor=INK, fontsize=9.5)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.12)
    fig.savefig(stem + "_dropoff.png", dpi=200, facecolor=SURFACE); fig.savefig(os.path.join(pdf_dir, os.path.basename(stem) + "_dropoff.pdf"), facecolor=SURFACE)
    print("wrote", stem + "_dropoff.png")

print("\nper budget: usable batch sizes, best accuracy, drop-off B")
print(drop.round(3).to_string(index=False))
for D in BUDGETS:
    u = tab[tab["budget"] == D]
    bb = best_all[D]
    print(f"\n{fk(D)} prompts: bracketed B = {sorted(bb[bb['bracketed']]['bsz'])}; usable but not bracketed: " + ", ".join(f"B={r.bsz} ({r.lower}/{r.upper})" for r in bb[~bb["bracketed"]].itertuples()) + "; dropped: " + "; ".join(f"B={b}: {w}" for b, w in u[~u["usable"]].groupby("bsz")["why"].agg(lambda x: x.value_counts().index[0]).items() if b not in set(best_all[D]["bsz"])))
