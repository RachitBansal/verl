"""
Visualize non-math capability regression from SFT vs RL training.

Two figures:
  1. Absolute curves per task: base, SFT, RL-from-base, RL-from-SFT vs pretrain step.
     Five tasks: wikitext BPB, HellaSwag acc_norm, LAMBADA perplexity,
                 SQuAD-completion BPB, IFEval BPB.
  2. Tradeoff scatter: GSM8K Pass@1 accuracy gain (vs base at the same pretrain step)
     vs non-math regression, for wikitext / LAMBADA / SQuAD-lm / IFEval BPB.

Usage:
    python notebooks/plot_regression.py
"""

from pathlib import Path
from collections import defaultdict
import ast
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# ─── Config ───────────────────────────────────────────────────────────────────
LM_EVAL_RESULTS_DIR = Path(__file__).parent.parent / "lm_eval_harness" / "results"
EVAL_RESULTS_DIRS = [
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results"),
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results_sunny"),
]
TOKEN_MULTIPLIER = 2_000_000
DATASET_KEY = "test_score/openai/gsm8k"

# pseudo-task key → (lm-eval task, metric key, short title, direction ["lower"=better])
# Ordering: accuracy (higher=better) panels first, BPB/perplexity (lower=better) after.
TASKS = {
    "lambada_acc":         ("lambada_openai",      "acc,none",           "LAMBADA",       "higher"),
    "hellaswag":           ("hellaswag",           "acc_norm,none",      "HellaSwag",   "higher"),
    "arc_easy":            ("arc_easy",            "acc_norm,none",      "ARC-Easy",    "higher"),
    "arc_challenge":       ("arc_challenge",       "acc_norm,none",      "ARC-Challenge", "higher"),
    "piqa":                ("piqa",                "acc_norm,none",      "PIQA",        "higher"),
    "winogrande":          ("winogrande",          "acc,none",           "WinoGrande Acc",         "higher"),
    "openbookqa":          ("openbookqa",          "acc_norm,none",      "OpenBookQA",  "higher"),
    "wikitext":            ("wikitext",            "bits_per_byte,none", "WikiText BPB",          "lower"),
    "lambada_ppl":         ("lambada_openai",      "perplexity,none",    "LAMBADA Perplexity",     "lower"),
    "squad_completion_lm": ("squad_completion_lm", "bits_per_byte,none", "SQuAD-Completion BPB",   "lower"),
    "ifeval_lm":           ("ifeval_lm",           "bits_per_byte,none", "IFEval BPB",             "lower"),
}
TASK_KEYS = list(TASKS.keys())
LOG_Y_TASKS = {"lambada_ppl"}  # log y-axis for perplexity panels

# Tasks used in the tradeoff scatter (same order — acc first, then lower-is-better)
TRADEOFF_TASKS = ["lambada_acc", "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande",
                  "openbookqa", "wikitext", "lambada_ppl", "squad_completion_lm", "ifeval_lm"]


# ─── Parsers ──────────────────────────────────────────────────────────────────

RE_BASE = re.compile(r"^OLMo2-1B-stage1-50B_step(?P<step>\d+)-hf$")
RE_SFT = re.compile(
    r"^OLMo2-1B_step(?P<step>\d+)_interleave_twoloader_n\d+_sft_\d+_ppo_\d+_rgsm_step(?P<sft_step>\d+)$"
)
# Same pattern as SFT (rgsm) but trained with `gsm` data instead of `rgsm`.
RE_SFT_GSM = re.compile(
    r"^OLMo2-1B_step(?P<step>\d+)_interleave_twoloader_n\d+_sft_\d+_ppo_\d+_gsm_step(?P<sft_step>\d+)$"
)
# RL from SFT-ed base: `stepN_sfted` OR `stepNsfted`
RE_RL_SFT = re.compile(
    r"^olmo2_1b_step(?P<step>\d+)_?sfted_omigsm8k_n\d+(_v(?P<seed>\d+))?_step(?P<rl_step>\d+)$"
)
# RL from raw base
RE_RL_BASE = re.compile(
    r"^olmo2_1b_step(?P<step>\d+)_omigsm8k_n\d+(_v(?P<seed>\d+))?_step(?P<rl_step>\d+)$"
)


def load_nonmath_results():
    """Return dicts: base[step], sft[step], sft_gsm[step], rl_base[step], rl_sft[step].

    Each value is {task: metric_value}. For RL groups we average across seeds.
    """
    base, sft, sft_gsm = {}, {}, {}
    rl_base_raw, rl_sft_raw = defaultdict(list), defaultdict(list)

    for f in sorted(LM_EVAL_RESULTS_DIR.glob("*/results.json")):
        data = json.loads(f.read_text())
        name = data["run_name"]
        results = data.get("results", {})
        if "wikitext" not in results:  # skip stale c4-only runs
            continue
        row = {key: results.get(task, {}).get(metric) for key, (task, metric, _, _) in TASKS.items()}

        m = RE_BASE.match(name)
        if m:
            base[int(m.group("step"))] = row
            continue
        m = RE_SFT.match(name)
        if m:
            sft[int(m.group("step"))] = row
            continue
        m = RE_SFT_GSM.match(name)
        if m:
            sft_gsm[int(m.group("step"))] = row
            continue
        m = RE_RL_SFT.match(name)  # check sfted first (more specific)
        if m:
            rl_sft_raw[int(m.group("step"))].append(row)
            continue
        m = RE_RL_BASE.match(name)
        if m:
            rl_base_raw[int(m.group("step"))].append(row)
            continue

    def avg(raw):
        out = {}
        for step, rows in raw.items():
            out[step] = {
                t: np.mean([r[t] for r in rows if r[t] is not None]) if any(r[t] is not None for r in rows) else None
                for t in TASK_KEYS
            }
        return out

    return base, sft, sft_gsm, avg(rl_base_raw), avg(rl_sft_raw)


# ─── GSM accuracy loading ─────────────────────────────────────────────────────

def read_score(result_path, samples):
    if not result_path.exists():
        return {}
    lines = result_path.read_text().splitlines()
    if samples > 1 or "majority" in result_path.name:
        scores = {}
        for line in lines:
            if "Pass@" in line and ":" in line:
                km = re.search(r"Pass@(\d+)", line)
                if km:
                    parts = line.split(":")
                    if len(parts) > 1:
                        try:
                            scores[int(km.group(1))] = float(parts[1].split()[0])
                        except Exception:
                            pass
        if scores:
            return scores
    for line in reversed(lines):
        try:
            payload = ast.literal_eval(line.strip())
        except Exception:
            continue
        if isinstance(payload, dict) and DATASET_KEY in payload:
            return {samples: payload[DATASET_KEY]}
    return {}


RE_GSM_BASE = re.compile(r"1B-step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$")
RE_GSM_SFT = re.compile(
    r"OLMo2-1B_step(?P<step>\d+)_interleave_twoloader_n(?P<n>\d+)_sft_\d+_ppo_\d+_rgsm-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)
RE_GSM_RL = [
    re.compile(r"olmo2_1b_step(?P<step>\d+)_omigsm8k_n(?P<n>\d+)(_v(?P<seed>\d+))?-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
]
RE_GSM_RL_SFT = [
    re.compile(r"olmo2_1b_step(?P<step>\d+)_?sfted_omigsm8k_n(?P<n>\d+)(_v(?P<seed>\d+))?-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
]


def load_gsm_accuracy():
    """Return (base_acc, sft_acc, rl_base_acc, rl_sft_acc). Pass@1, temp 0.6."""
    base_acc = {}
    sft_raw = defaultdict(list)
    rl_base_raw = defaultdict(list)
    rl_sft_raw = defaultdict(list)

    for base_dir in EVAL_RESULTS_DIRS:
        if not base_dir.exists():
            continue
        for path in base_dir.iterdir():
            if not path.is_dir():
                continue
            name = path.name

            mb = RE_GSM_BASE.match(name)
            if mb and not any(t in name for t in ["-rl-", "-sft-", "-hf"]):
                if int(mb.group("shot")) == 8 and mb.group("temp") == "0.6":
                    samples = int(mb.group("samples"))
                    file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
                    for k, sc in read_score(path / file, samples).items():
                        if k == 1:
                            step = int(mb.group("step"))
                            base_acc[step] = max(base_acc.get(step, 0), sc)
                continue

            ms = RE_GSM_SFT.match(name)
            if ms and ms.group("temp") == "0.6" and int(ms.group("n")) == 32:
                samples = int(ms.group("samples"))
                file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
                for k, sc in read_score(path / file, samples).items():
                    if k == 1:
                        sft_raw[int(ms.group("step"))].append((sc, int(ms.group("rl_step"))))
                continue

            matched = False
            for pat in RE_GSM_RL_SFT:  # sfted first (more specific)
                m = pat.match(name)
                if m and m.group("temp") == "0.6" and int(m.group("n")) == 32:
                    samples = int(m.group("samples"))
                    file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
                    seed = int(m.groupdict().get("seed") or 1)
                    for k, sc in read_score(path / file, samples).items():
                        if k == 1:
                            rl_sft_raw[int(m.group("step"))].append((sc, int(m.group("rl_step")), seed))
                    matched = True
                    break
            if matched:
                continue
            for pat in RE_GSM_RL:
                m = pat.match(name)
                if m and m.group("temp") == "0.6" and int(m.group("n")) == 32:
                    samples = int(m.group("samples"))
                    file = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
                    seed = int(m.groupdict().get("seed") or 1)
                    for k, sc in read_score(path / file, samples).items():
                        if k == 1:
                            rl_base_raw[int(m.group("step"))].append((sc, int(m.group("rl_step")), seed))
                    break

    def last_step_best_score(raw):
        out = {}
        for step, entries in raw.items():
            # per-seed: pick latest rl_step; across seeds: pick best score
            by_seed = defaultdict(list)
            for sc, rl_step, seed in entries:
                by_seed[seed].append((sc, rl_step))
            best_per_seed = [max(es, key=lambda x: x[1])[0] for es in by_seed.values()]
            out[step] = max(best_per_seed)
        return out

    def sft_last(raw):
        out = {}
        for step, entries in raw.items():
            out[step] = max(entries, key=lambda x: x[1])[0]
        return out

    return base_acc, sft_last(sft_raw), last_step_best_score(rl_base_raw), last_step_best_score(rl_sft_raw)


# ─── Style ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "legend.fontsize": 13,
})

C_BASE = "#777777"
C_SFT = "#009E73"
C_SFT_GSM = "#FFB000"  # amber/orange to distinguish from green SFT-rgsm
C_RL_BASE = "#E24A33"
C_RL_SFT = "#9467BD"

STYLE_BASE = dict(color=C_BASE, marker="o", ls=":", ms=8, lw=2.0, label=r"Base ($\mathcal{M}_t$)")
STYLE_SFT = dict(color=C_SFT, marker="d", ls="-.", ms=9, lw=2.0, label=r"SFT-Multi ($\mathcal{M}_t^{\text{SFT-multi}})$")
STYLE_SFT_GSM = dict(color=C_SFT_GSM, marker="^", ls="-.", ms=9, lw=2.0, label=r"SFT-Single ($\mathcal{M}_t^{\text{SFT-single}})$")
STYLE_RL_BASE = dict(color=C_RL_BASE, marker="*", ls="-", ms=14, lw=2.5, label=r"RL-from-base ($\mathcal{M}_t^{\text{RL}})$")
STYLE_RL_SFT = dict(color=C_RL_SFT, marker="s", ls="--", ms=8, lw=2.0, label=r"RL-from-SFT ($\mathcal{M}_t^{\text{SFT}\to\text{RL}})$")


def dual_axis_formatter(num, pos):
    tok, mag = num, 0
    while abs(tok) >= 1000:
        mag += 1
        tok /= 1000.0
    tok_str = "%.0f%s" % (tok, ["", "K", "M", "B", "T"][mag])
    step, smag = num / TOKEN_MULTIPLIER, 0
    while abs(step) >= 1000:
        smag += 1
        step /= 1000.0
    return f"{tok_str}\n{step_str(step, smag)}"


def step_str(step, smag):
    return "%.0f%s" % (step, ["", "k", "m", "b"][smag])


formatter = FuncFormatter(dual_axis_formatter)


# ─── Load data ────────────────────────────────────────────────────────────────

base, sft, sft_gsm, rl_base, rl_sft = load_nonmath_results()
base_acc, sft_acc, rl_base_acc, rl_sft_acc = load_gsm_accuracy()

all_steps = sorted(set(base) | set(sft) | set(sft_gsm) | set(rl_base) | set(rl_sft))

print("=" * 110)
print("Non-math task values (averaged across seeds for RL)")
print("=" * 110)
header = f"{'step':>7}  {'group':<12}  " + "  ".join(f"{t:<10}" for t in TASK_KEYS)
print(header)
print("-" * len(header))
for step in all_steps:
    for label, d in [("base", base), ("SFT-rgsm", sft), ("SFT-gsm", sft_gsm), ("RL-base", rl_base), ("RL-sft", rl_sft)]:
        if step in d:
            vals = "  ".join(
                ("%10.4f" % d[step][t]) if d[step][t] is not None else "        --"
                for t in TASK_KEYS
            )
            print(f"{step:>7}  {label:<12}  {vals}")

print()
print("GSM8K Pass@1 (temp 0.6):")
for step in all_steps:
    parts = []
    for label, d in [("base", base_acc), ("SFT", sft_acc), ("RL-base", rl_base_acc), ("RL-sft", rl_sft_acc)]:
        if step in d:
            parts.append(f"{label}={d[step]:.3f}")
    if parts:
        print(f"  step {step}: " + ", ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Absolute curves per task
# ═══════════════════════════════════════════════════════════════════════════════

pct_formatter = FuncFormatter(lambda x, _: f"{x:.0f}")


def plot_line(ax, d, style):
    steps = sorted(s for s in d if d[s].get(task) is not None)
    if not steps:
        return
    xs = [s * TOKEN_MULTIPLIER for s in steps]
    ys = [d[s][task] * 100 for s in steps]
    ax.plot(xs, ys, **style)


N1 = len(TASK_KEYS)
NCOL1 = 4
NROW1 = (N1 + NCOL1 - 1) // NCOL1
fig1, axes = plt.subplots(NROW1, NCOL1, figsize=(6 * NCOL1, 4.5 * NROW1))
axes = axes.ravel()

for i, task in enumerate(TASK_KEYS):
    ax = axes[i]
    _, _, title, direction = TASKS[task]
    plot_line(ax, base, STYLE_BASE)
    plot_line(ax, sft, STYLE_SFT)
    plot_line(ax, sft_gsm, STYLE_SFT_GSM)
    plot_line(ax, rl_base, STYLE_RL_BASE)
    plot_line(ax, rl_sft, STYLE_RL_SFT)
    arrow = "↓ better" if direction == "lower" else "↑ better"
    ax.set_title(f"{title}  ({arrow})", pad=10)
    ax.set_xlabel("Pre-training tokens (steps)")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(pct_formatter)
    ax.grid(True, linestyle=":", color="gray", alpha=0.7)
    if task in LOG_Y_TASKS:
        ax.set_yscale("log")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

# Hide unused panels
for ax in axes[N1:]:
    ax.set_visible(False)

# Legend below figure
handles, labels = axes[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04),
            ncol=len(handles), frameon=True, fontsize=13)

fig1.suptitle("Non-math capability regression: base vs. SFT vs. RL (limit=100 per task)",
              fontsize=18, y=1.00)
plt.tight_layout()
plt.subplots_adjust(bottom=0.09)
out1 = Path(__file__).parent / "regression_absolute.pdf"
fig1.savefig(out1, bbox_inches="tight")
fig1.savefig(out1.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"\nSaved absolute curves -> {out1}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Tradeoff scatter — GSM Pass@1 gain vs non-math regression
# ═══════════════════════════════════════════════════════════════════════════════

rows = []
# SFT-gsm has no GSM accuracy data (no GSM math eval run on these ckpts yet),
# so it appears as no-acc entries — drawn at acc_gain=None on the scatter.
groups = [
    ("SFT-rgsm", sft,     sft_acc,     C_SFT,     "d"),
    ("SFT-gsm",  sft_gsm, {},          C_SFT_GSM, "^"),
    ("RL-base",  rl_base, rl_base_acc, C_RL_BASE, "*"),
    ("RL-sft",   rl_sft,  rl_sft_acc,  C_RL_SFT,  "s"),
]
for label, bpb_d, acc_d, color, marker in groups:
    for step in bpb_d:
        if step not in base:
            continue
        base_row = base[step]
        method_row = bpb_d[step]
        acc_gain = None
        if step in acc_d and step in base_acc:
            acc_gain = (acc_d[step] - base_acc[step]) * 100
        for task in TRADEOFF_TASKS:
            b, m = base_row.get(task), method_row.get(task)
            if b is None or m is None:
                continue
            delta = m - b
            rows.append({
                "method": label, "color": color, "marker": marker,
                "step": step, "task": task,
                "acc_gain": acc_gain, "delta": delta,
            })
df = pd.DataFrame(rows)

N2 = len(TRADEOFF_TASKS)
NCOL2 = 4
NROW2 = (N2 + NCOL2 - 1) // NCOL2
fig2, axes2 = plt.subplots(NROW2, NCOL2, figsize=(6 * NCOL2, 4.5 * NROW2))
axes2 = axes2.ravel()
for i, task in enumerate(TRADEOFF_TASKS):
    ax = axes2[i]
    sub = df[df["task"] == task]
    if sub.empty:
        ax.set_visible(False)
        continue
    has_acc = sub[sub["acc_gain"].notna()]
    no_acc = sub[sub["acc_gain"].isna()]

    texts = []
    for label, _, _, color, marker in groups:
        s = has_acc[has_acc["method"] == label]
        if s.empty:
            continue
        size = 280 if marker == "*" else 140
        ax.scatter(s["acc_gain"], s["delta"],
                   color=color, marker=marker, s=size,
                   edgecolors="k", linewidths=0.6, label=label, zorder=5)
        for _, r in s.iterrows():
            texts.append(ax.text(r["acc_gain"], r["delta"], f"{int(r['step']/1000)}k",
                                 fontsize=10, color=color, fontweight="bold"))
    if HAS_ADJUST_TEXT and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.5))

    ax.axhline(0, color=C_BASE, ls="--", lw=1.2, alpha=0.7)
    ax.axvline(0, color=C_BASE, ls="--", lw=1.2, alpha=0.5)
    _, _, title, direction = TASKS[task]
    ax.set_title(f"{title}", pad=10)
    ax.set_xlabel("GSM8K Pass@1 gain vs base (pp)")
    regr_dir = "positive = regression" if direction == "lower" else "negative = regression"
    ax.set_ylabel(f"Δ (method − base): {regr_dir}")
    if task in LOG_Y_TASKS:
        # Use symlog so we can still show small negatives if any
        ax.set_yscale("symlog", linthresh=max(1.0, abs(sub["delta"].abs().min() or 1.0)))
    ax.grid(True, linestyle=":", color="gray", alpha=0.7)
    # Note if any rows had no GSM accuracy available
    n_missing = len(no_acc["method"].unique()) if not no_acc.empty else 0
    if n_missing:
        missing_methods = ", ".join(sorted(no_acc["method"].unique()))
        ax.text(0.02, 0.98, f"(no GSM acc: {missing_methods})",
                transform=ax.transAxes, fontsize=9, color="gray",
                va="top", ha="left")

# Hide unused panels (2x3 grid, 5 tasks)
for ax in axes2[len(TRADEOFF_TASKS):]:
    ax.set_visible(False)

handles, labels = [], []
seen = set()
for ax in axes2:
    if not ax.get_visible():
        continue
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in seen:
            handles.append(hi); labels.append(li); seen.add(li)
fig2.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
            ncol=len(handles), frameon=True, fontsize=13)
fig2.suptitle("Math gain vs non-math regression (labels = pretrain step)", fontsize=16, y=1.00)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
out2 = Path(__file__).parent / "regression_tradeoff.pdf"
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved tradeoff scatter -> {out2}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Condensed version for the main paper
# Layout: large left panel with the average across benchmarks, and a 3x2 grid
# on the right with one panel per benchmark (minimal labeling).
# ═══════════════════════════════════════════════════════════════════════════════

CONDENSED_TASKS = ["lambada_acc", "hellaswag", "arc_easy", "arc_challenge", "piqa", "openbookqa"]
CONDENSED_TITLE_OVERRIDES = {"lambada_acc": "LAMBADA"}


def token_only_formatter(num, pos):
    tok, mag = num, 0
    while abs(tok) >= 1000:
        mag += 1
        tok /= 1000.0
    return "%.0f%s" % (tok, ["", "K", "M", "B", "T"][mag])


formatter_tokens_only = FuncFormatter(token_only_formatter)


def average_across_tasks(d, tasks):
    """For each step, average the given task values (in %). Skips Nones."""
    out = {}
    for step, row in d.items():
        vals = [row[t] for t in tasks if row.get(t) is not None]
        if vals:
            out[step] = {"_avg": float(np.mean(vals))}
    return out


def plot_line_avg(ax, d_avg, style):
    steps = sorted(d_avg)
    if not steps:
        return
    xs = [s * TOKEN_MULTIPLIER for s in steps]
    ys = [d_avg[s]["_avg"] * 100 for s in steps]
    ax.plot(xs, ys, **style)


fig3 = plt.figure(figsize=(13, 4.5))
gs = fig3.add_gridspec(3, 3, width_ratios=[1.6, 1, 1], hspace=0.08, wspace=0.10)

# Left: big average panel spanning all 3 rows
ax_avg = fig3.add_subplot(gs[:, 0])
base_avg = average_across_tasks(base, CONDENSED_TASKS)
sft_avg = average_across_tasks(sft, CONDENSED_TASKS)
sft_gsm_avg = average_across_tasks(sft_gsm, CONDENSED_TASKS)
rl_base_avg = average_across_tasks(rl_base, CONDENSED_TASKS)
plot_line_avg(ax_avg, base_avg, STYLE_BASE)
plot_line_avg(ax_avg, sft_gsm_avg, STYLE_SFT_GSM)
plot_line_avg(ax_avg, sft_avg, STYLE_SFT)
plot_line_avg(ax_avg, rl_base_avg, STYLE_RL_BASE)
ax_avg.set_title("Average across 6 benchmarks", pad=10, fontsize=20)
ax_avg.set_xlabel("Pre-training tokens", fontsize=18)
ax_avg.set_ylabel("Accuracy (%)", fontsize=18)
ax_avg.xaxis.set_major_formatter(formatter_tokens_only)
ax_avg.yaxis.set_major_formatter(pct_formatter)
ax_avg.tick_params(axis="both", labelsize=16)
ax_avg.grid(True, linestyle=":", color="gray", alpha=0.7)
for spine in ax_avg.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.0)

# Right: 3x2 grid of small square per-benchmark panels
small_axes = []
for i, task in enumerate(CONDENSED_TASKS):
    r, c = i // 2, i % 2
    ax = fig3.add_subplot(gs[r, 1 + c])
    small_axes.append(ax)
    _, _, title, _ = TASKS[task]
    title = CONDENSED_TITLE_OVERRIDES.get(task, title)
    plot_line(ax, base, STYLE_BASE)
    plot_line(ax, sft_gsm, STYLE_SFT_GSM)
    plot_line(ax, sft, STYLE_SFT)
    plot_line(ax, rl_base, STYLE_RL_BASE)
    # Benchmark name inside panel at bottom-right (frees vertical space → taller panels)
    ax.text(0.97, 0.05, title, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12, fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75))
    # Minimal labeling: just two y ticks; only bottom row keeps x-axis
    is_bottom = (r == 2)
    if is_bottom:
        ax.xaxis.set_major_formatter(formatter_tokens_only)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.tick_params(axis="x", labelsize=12)
    else:
        ax.set_xlabel("")
        ax.set_xticks([])
    # Y-ticks at panel top and bottom (same relative spot across panels)
    task_vals = []
    for d in (base, sft_gsm, sft, rl_base):
        for s in d:
            v = d[s].get(task)
            if v is not None:
                task_vals.append(v * 100)
    if task_vals:
        ylo = int(np.floor(min(task_vals)))
        yhi = int(np.ceil(max(task_vals)))
        pad = max(1, (yhi - ylo) * 0.08)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_yticks([ylo, yhi])
    ax.yaxis.set_major_formatter(pct_formatter)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, linestyle=":", color="gray", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)

handles, labels = ax_avg.get_legend_handles_labels()
ax_avg.legend(handles, labels, loc="lower right", frameon=True, fontsize=11,
              framealpha=0.92)

plt.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.10)
out3 = Path(__file__).parent / "regression_main.pdf"
fig3.savefig(out3, bbox_inches="tight")
fig3.savefig(out3.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved condensed main-paper figure -> {out3}")
print(f"Saved png version -> {out3.with_suffix('.png')}")

plt.show()
