"""Fitted critical batch N* against the training tokens actually generated to reach each target accuracy.

Series: prompts at K = 16 with KL 1e-3 and KL 1e-2 as two separate curves (never pooled), prompts at K = 64, rollouts at B = 128.

Rachit's Figure 5b (iclr_figs/make_cbs_vs_tokens.py) places each N*(target) at S_min * N* * 4096 tokens: two fitted
parameters times the per-sequence token budget. Here the x position is measured instead:

  for each sweep and target, every column (a (B, K) cell) is reduced to its fastest LR run and that run's crossing step;
  the tokens that run generated up to the crossing are summed from its logged per-step mean prompt and response lengths
  (tokens by step s = B K * sum_{steps <= s} (prompt_len + resp_len), interpolated at the fractional crossing);
  the x position is the smallest such token count over the columns in the fit -- the cheapest way we actually
  reached the target, the measured counterpart of E_min = S_min N*.

N* is fitted exactly as in plot_nstar_vs_target.py: S = S_min (1 + N*/N) in log S over the columns with a complete LR
bracket (solid, needs --min-cols of them) or, with --all-cols, over every column that reached the target. Markers are
filled where every crossing column is bracketed. --x fitted reproduces Rachit's placement but with the measured mean
sequence length in place of 4096.

Usage: python plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_tokens.png --tokens csv/token_series.json
          [--n64 csv/val_curves_n64.json] [--b128 csv/val_curves_bsz128.json] [--min-bsz 4] [--min-cols 5]
          [--min-target 0.40] [--target-step 0.005] [--all-cols] [--x measured|fitted|prompts|prompts-fitted] [--keep-first-interval]
  --x prompts-fitted puts the fitted prompt budget S_min x B* (= S_min N* / K; the prompt-unit analogue of --x fitted) on the x-axis.
  --x prompts puts the prompts trained on to reach the target (steps x B of the cheapest batch size in the fit) on the x-axis.
Columns whose crossing is interpolated before the run's first validation are left out of the fits unless --keep-first-interval.
  --sweeps n16,n16kl2,n64,b128 draws only those series (default: all available).
  --fit draws a least-squares power law N* = c x^alpha (log-log, over every drawn target of a series with at least 3 points),
  dashed, with the exponent and its OLS standard error in the legend; written to csv/<stem>_fit_table.csv. Neighbouring
  targets share runs, so that standard error understates the real uncertainty.
Writes the PNG, paper/<stem>.pdf and csv/<stem>_table.csv.
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

SURFACE, INK, INK2, GRID, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a2"
SHORT_STEPS = 200
VALID_N_PREFIX = {1, 2, 4, 8, 16}
FIXED_ONLY_N = {32, 64, 128}


def opt(flag, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
json_in, png_out = args[0], args[1]
MIN_BSZ, MIN_COLS = opt("--min-bsz", 4, int), opt("--min-cols", 5, int)
ALL_COLS = "--all-cols" in sys.argv
XMODE = opt("--x", "measured")
T_LO, T_STEP = opt("--min-target", 0.40, float), opt("--target-step", 0.005, float)
TARGETS = np.round(np.arange(T_LO, 0.50 + 1e-9, T_STEP), 3)
KEEP_FIRST = "--keep-first-interval" in sys.argv   # by default a column whose crossing precedes its first eval is left out of the fits
TOK_RAW = json.load(open(opt("--tokens", "csv/token_series.json")))
# resumed runs are separate wandb runs with the same name: merge their per-step length series by name, first segment wins on overlap
TOK_BY_NAME = {}
for rid, d in TOK_RAW.items():
    acc = TOK_BY_NAME.setdefault(d["name"], {})
    for st, pl, rl in zip(d["steps"], d["prompt_len"], d["resp_len"]):
        acc.setdefault(st, pl + rl)
TOK = {rid: dict(steps=np.array(sorted(TOK_BY_NAME[d["name"]])), L=np.array([TOK_BY_NAME[d["name"]][k] for k in sorted(TOK_BY_NAME[d["name"]])]))
       for rid, d in TOK_RAW.items()}

n16 = json.load(open(json_in))
SERIES = [dict(key="n16", name="prompts, K = 16, KL 1e-3", runs=[r for r in n16 if abs(r["kl"] - 1e-3) < 1e-12 and r["bsz"] >= MIN_BSZ], x="bsz", per=16, color="#1f77b4", marker="o"),
          dict(key="n16kl2", name="prompts, K = 16, KL 1e-2", runs=[r for r in n16 if abs(r["kl"] - 1e-2) < 1e-12 and r["bsz"] >= MIN_BSZ], x="bsz", per=16, color="#8fb8e8", marker="D")]
if opt("--n64"):
    n64 = json.load(open(opt("--n64")))
    SERIES.append(dict(key="n64", name="prompts, K = 64, KL 1e-3", runs=[r for r in n64 if abs(r["kl"] - 1e-3) < 1e-12 and r["fixed"] and r["bsz"] >= MIN_BSZ], x="bsz", per=64, color="#d66300", marker="h"))
if opt("--b128"):
    b128 = json.load(open(opt("--b128")))
    roll = [r for r in b128 if not r["downsample"] and r["kl"] is not None and abs(r["kl"] - 1e-3) < 1e-12
            and (r["n"] in VALID_N_PREFIX or (r["fixed"] and r["n"] in FIXED_ONLY_N))]
    SERIES.append(dict(key="b128", name="rollouts, B = 128, KL 1e-3", runs=roll, x="n", per=128, color="#2ca02c", marker="p"))


def crossing(steps, vals, t):
    """Interpolated crossing step and whether it lies inside the first evaluation interval (step 0 .. first eval)."""
    for i, v in enumerate(vals):
        if v >= t:
            if i == 0:
                return float(steps[0]), steps[0] == 0
            s0, s1, v0, v1 = steps[i - 1], steps[i], vals[i - 1], vals[i]
            return (float(s1) if v1 == v0 else s0 + (s1 - s0) * (t - v0) / (v1 - v0)), (i == 1 and steps[0] == 0)   # resumed segments start mid-run
    return None, None


def tokens_until(run_id, N, step):
    """Tokens generated by (fractional) training step `step`: N sequences per step times the logged mean length per step."""
    d = TOK.get(run_id)
    if not d or not len(d["steps"]):
        return np.nan, np.nan
    s = d["steps"].astype(float); L = d["L"].astype(float)
    cum = np.concatenate([[0.0], np.cumsum(L)]); s_edges = np.concatenate([[s[0] - 1], s])    # tokens/N after each logged step
    if step > s[-1] + 1:            # crossing beyond the logged length series (should not happen)
        return np.nan, np.nan
    seqs_len = np.interp(step, s_edges, cum)
    mean_len = seqs_len / max(step - s_edges[0], 1e-9)
    return N * seqs_len, mean_len


def fit(N, S):
    N = np.asarray(N, float); y = np.log(np.asarray(S, float)); best = None
    for ln in np.linspace(np.log(8), np.log(2 ** 28), 6000):
        g = np.log1p(np.exp(ln) / N); m = (y - g).mean(); r = np.sqrt(((y - g - m) ** 2).mean())
        if best is None or r < best[2]:
            best = (float(np.exp(m)), float(np.exp(ln)), float(r))
    if best[1] > 2 ** 26:
        return (np.nan, np.nan, np.nan)
    return best


rows, cols_out = [], []
for ser in SERIES:
    xk = ser["x"]
    for t in TARGETS:
        cfg = {}
        for r in ser["runs"]:
            if not r["steps"]:
                continue
            c = cfg.setdefault((r[xk], r["lr"]), dict(x=r[xk], lr=r["lr"], steps=np.inf, longest=0, run=None, first=True))
            xx, first = crossing(r["steps"], r["vals"], t)
            # a measured crossing always beats a first-interval guess; among equals, the earlier one wins
            if xx is not None and ((c["first"] and not first) or (c["first"] == first and xx < c["steps"])):
                c["steps"], c["run"], c["first"] = xx, r["id"], first
            c["longest"] = max(c["longest"], r["steps"][-1])
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
            side = lambda j: "open" if j < 0 or j >= len(g) else ("inconclusive" if g.loc[j, "verdict"] == "inconclusive" else "complete")
            lo, up = side(ib - 1), side(ib + 1)
            status = "open" if "open" in (lo, up) else "inconclusive" if "inconclusive" in (lo, up) else "complete"
            if bool(g.loc[ib, "first"]) and not KEEP_FIRST:
                status = "first-interval"
            N = int(x) * ser["per"]
            toks, mlen = tokens_until(g.loc[ib, "run"], N, ref)
            prompts = ref * (int(x) if ser["x"] == "bsz" else ser["per"])          # prompts drawn to reach the target: steps x B
            cols.append(dict(series=ser["name"], target=t, x=int(x), N=N, best_lr=g.loc[ib, "lr"], run=g.loc[ib, "run"], steps=ref,
                             tokens=toks, mean_len=mlen, prompts=prompts, status=status))
        cols_out += cols
        c = pd.DataFrame(cols)
        comp = c[c["status"] == "complete"]; c_ok = c[c["status"] != "first-interval"]
        use = c_ok if ALL_COLS else comp
        row = dict(series=ser["name"], target=t, n_cols=len(c), n_first_interval=len(c) - len(c_ok), n_complete=len(comp), all_complete=len(comp) == len(c))
        if len(use) >= (3 if ALL_COLS else MIN_COLS):
            smin, nstar, rms = fit(use["N"], use["steps"])
            u = use.dropna(subset=["tokens"])
            i_min = u["tokens"].idxmin() if len(u) else None
            ip = use["prompts"].idxmin()
            row.update(smin=smin, nstar=nstar, rms=rms, tokens_min=u.loc[i_min, "tokens"] if i_min is not None else np.nan,
                       tokens_min_N=u.loc[i_min, "N"] if i_min is not None else np.nan, mean_len=u["mean_len"].mean() if len(u) else np.nan,
                       prompts_min=use.loc[ip, "prompts"], prompts_min_N=use.loc[ip, "N"])
            row["tokens_fitted"] = smin * nstar * row["mean_len"]
            row["prompts_fitted"] = smin * nstar / ser["per"] if ser["x"] == "bsz" else smin * ser["per"]   # S_min B* prompts; a rollout sweep spends S_min x B
        rows.append(row)
tab = pd.DataFrame(rows)
stem = os.path.basename(png_out).replace(".png", "")
csv_dir = os.path.dirname(os.path.abspath(json_in))
tab.to_csv(os.path.join(csv_dir, stem + "_table.csv"), index=False)
pd.DataFrame(cols_out).to_csv(os.path.join(csv_dir, stem + "_columns.csv"), index=False)

XCOL = {"measured": "tokens_min", "fitted": "tokens_fitted", "prompts": "prompts_min", "prompts-fitted": "prompts_fitted"}[XMODE]
ONLY = opt("--sweeps", "")
if ONLY:
    SERIES = [s_ for s_ in SERIES if s_["key"] in ONLY.split(",")]
FIT = "--fit" in sys.argv
XSYM = {"measured": "T", "fitted": "T", "prompts": "P", "prompts-fitted": "P"}[XMODE]

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13.5, "legend.fontsize": 11})
fig, ax = plt.subplots(figsize=(10, 6.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE); ax.grid(True, which="major", color=GRID, lw=0.7); ax.tick_params(which="both", colors=INK2, length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
drawn = []
for ser in SERIES:
    f = tab[tab["series"] == ser["name"]].sort_values("target")
    if "nstar" not in f or f["nstar"].isna().all():
        continue
    drawn.append(ser)
    f = f.dropna(subset=["nstar", XCOL]); col, mk = ser["color"], ser["marker"]
    ax.plot(f[XCOL], f["nstar"], ls="-", lw=2, color=col, alpha=0.55, zorder=3)
    full, part = f[f["all_complete"] | ALL_COLS], f[~(f["all_complete"] | ALL_COLS)]
    ax.scatter(full[XCOL], full["nstar"], s=48, marker=mk, color=col, edgecolors=SURFACE, linewidths=0.8, zorder=4)
    ax.scatter(part[XCOL], part["nstar"], s=48, marker=mk, facecolors="none", edgecolors=col, linewidths=1.3, zorder=4)
    if FIT and len(f) >= 3:
        lx, ly = np.log(f[XCOL].values), np.log(f["nstar"].values)
        (a, c), cov = np.polyfit(lx, ly, 1, cov=True)
        xs = np.geomspace(f[XCOL].min() / 1.3, f[XCOL].max() * 1.3, 50)
        ax.plot(xs, np.exp(c) * xs ** a, ls="--", lw=1.5, color=col, alpha=0.9, zorder=3)
        ser["fit"] = dict(series=ser["name"], x=XCOL, alpha=float(a), alpha_se=float(np.sqrt(cov[0, 0])), coef=float(np.exp(c)), n_targets=len(f),
                          x_min=float(f[XCOL].min()), x_max=float(f[XCOL].max()), nstar_min=float(f["nstar"].min()), nstar_max=float(f["nstar"].max()))
    for r in f.itertuples():
        pct = r.target * 100
        if abs(pct - round(pct)) < 1e-6 and int(round(pct)) % 5 == 0:
            ax.annotate(f"{pct:.0f}%", (getattr(r, XCOL), r.nstar), xytext=(0, 10 if ser["x"] == "bsz" else -16), textcoords="offset points",
                        ha="center", fontsize=10.5, color=col)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 2.5)
ax.set_ylabel("fitted N*  (sequences per step)", color=INK)
if XMODE in ("prompts", "prompts-fitted"):
    fmt_x = lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"
    ticks = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
else:
    fmt_x = lambda v, _: f"{v / 1e9:.1f}B" if v >= 1e9 else f"{v / 1e6:.0f}M"
    ticks = [1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9, 5e9]
ax.xaxis.set_major_formatter(FuncFormatter(fmt_x)); ax.xaxis.set_minor_locator(NullLocator())
lo, hi = ax.get_xlim()
ax.xaxis.set_major_locator(FixedLocator([v for v in ticks if lo <= v <= hi]))
if XMODE == "prompts":
    ax.set_xlabel("prompts trained on to reach the target  (steps × B, cheapest batch size in the fit)", color=INK)
    ttl = f"critical batch vs the prompts spent reaching each target, {T_LO:.0%}–50%"
elif XMODE == "prompts-fitted":
    ax.set_xlabel("S_min · B*  (fitted prompts to reach the target in the small-batch limit)", color=INK)
    ttl = f"critical batch vs fitted prompts to target P_min = S_min B*, {T_LO:.0%}–50%"
elif XMODE == "measured":
    ax.set_xlabel("training tokens generated to reach the target  (cheapest batch size in the fit, measured)", color=INK)
    ttl = f"critical batch vs the tokens actually spent reaching each target, {T_LO:.0%}–50%"
else:
    ax.set_xlabel("S_min · N* · mean sequence length  (fitted small-batch limit, measured lengths)", color=INK)
    ttl = f"critical batch vs fitted tokens to target E_min = S_min N*, {T_LO:.0%}–50%"
ax.set_title(ttl + ("  (bracketing ignored)" if ALL_COLS else "  (LR-bracketed batch sizes only)"), loc="left", fontsize=12, color=INK)
exp_txt = lambda fd: f"   fit N* ∝ {XSYM}$^{{{fd['alpha']:.2f} ± {fd['alpha_se']:.2f}}}$ ({fd['n_targets']} targets)"
legend = [Line2D([], [], marker=s["marker"], ls="-", lw=2, ms=7, color=s["color"], mec=SURFACE, label=s["name"] + (exp_txt(s["fit"]) if "fit" in s else "")) for s in drawn]
if FIT:
    fits = [s["fit"] for s in drawn if "fit" in s]
    pd.DataFrame(fits).to_csv(os.path.join(csv_dir, stem + "_fit_table.csv"), index=False)
missing = [s["name"] for s in SERIES if s not in drawn]
if not ALL_COLS:
    legend += [Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=INK2, mew=1.3, label="hollow: some batch sizes lack a complete LR bracket"),
               Line2D([], [], ls="", label=f"drawn where at least {MIN_COLS} bracketed batch sizes exist")]
    if missing:
        legend += [Line2D([], [], ls="", label="not drawn, too few bracketed batch sizes: " + "; ".join(missing))]
ax.legend(handles=legend, loc="upper left", frameon=False, labelcolor=INK, fontsize=10.5)
fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.1)
fig.savefig(png_out, dpi=200, facecolor=SURFACE)
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(png_out))), "paper"); os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(os.path.join(pdf_dir, stem + ".pdf"), facecolor=SURFACE)
print("wrote", png_out)
if FIT and any("fit" in s_ for s_ in drawn):
    print("\npower-law fits N* = c x^alpha over the drawn targets:\n", pd.DataFrame([s_["fit"] for s_ in drawn if "fit" in s_]).round(3).to_string(index=False))
for ser in SERIES:
    f = tab[tab["series"] == ser["name"]]
    if "nstar" not in f:
        continue
    f = f.dropna(subset=["nstar"])
    print(f"\n{ser['name']}")
    for r in f.itertuples():
        if abs(r.target * 100 - round(r.target * 100)) < 1e-6:
            print(f"  {r.target:.0%}: N* {r.nstar:9,.0f}  S_min {r.smin:5.1f}  measured min tokens {r.tokens_min / 1e6:7.0f}M (at N = {r.tokens_min_N:,.0f})"
                  f"   fitted S_min N* L = {r.tokens_fitted / 1e6:7.0f}M   mean len {r.mean_len:5.0f}  [{r.n_complete}/{r.n_cols} bracketed]")
