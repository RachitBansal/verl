"""
Parallel-avg RL: GSM8K pass@k + non-math 5-benchmark average vs RL training step.

Four panels (Pass@1, Pass@8, Pass@32, Non-math avg). Three curves per panel
(SFT LR runs), plus horizontal reference lines for the base step10000 ckpt and
the SFT-Single step10000 ckpt. The GSM panels add a Direct-RL reference at
~step10000 read from manual_rl_gsm.json.

Usage:
    python notebooks/parallel_avg_analysis.py
"""

from pathlib import Path
import ast
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ───────────────────────────────────────────────────────────────────
EVAL_DIRS = [
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results_sunny"),
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results"),
]
LM_EVAL_RESULTS_DIR = Path(__file__).parent.parent / "lm_eval_harness" / "results"

EXPERIMENTS = {
    "sft1e-6": "OLMo2-1B_step10000_parallel_avg_n32_rl1e-6_sft1e-6_smbatch",
    "sft1e-7": "OLMo2-1B_step10000_parallel_avg_n32_rl1e-6_sft1e-7_smbatch",
    "sft4e-5": "OLMo2-1B_step10000_parallel_avg_n32_rl1e-6_sft4e-5_smbatch",
}

PLOT_SAMPLES = [1, 8, 32]
N_SAMPLES = 32
TEMP = 0.6
PRETRAIN_STEP = 10000
PRETRAIN_SHOT = 8
TOKEN_MULTIPLIER = 2_000_000
PRETRAIN_TOKENS_B = PRETRAIN_STEP * TOKEN_MULTIPLIER / 1e9  # 20.0 B

# 5 condensed non-math tasks (mirrors CONDENSED_TASKS in plot_regression.py).
NONMATH_TASKS = {
    "lambada_acc": ("lambada_openai", "acc,none"),
    "hellaswag":   ("hellaswag",      "acc_norm,none"),
    "arc_easy":    ("arc_easy",       "acc_norm,none"),
    "piqa":        ("piqa",           "acc_norm,none"),
    "openbookqa":  ("openbookqa",     "acc_norm,none"),
}
BASE_LM_NAME = f"OLMo2-1B-stage1-50B_step{PRETRAIN_STEP}-hf"
SFT_SINGLE_LM_NAME = f"OLMo2-1B_step{PRETRAIN_STEP}_interleave_twoloader_n32_sft_50000_ppo_0_gsm_step9500"
SFT_MULTI_LM_NAME = f"OLMo2-1B_step{PRETRAIN_STEP}_interleave_twoloader_n32_sft_50000_ppo_0_rgsm_step9500"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def read_pass_at_k(result_path: Path):
    if not result_path.exists():
        return {}
    scores = {}
    for line in result_path.read_text().splitlines():
        m = re.search(r"Pass@(\d+)\s*:\s*([\d.]+)", line)
        if m:
            scores[int(m.group(1))] = float(m.group(2))
    return scores


def read_dict_score(result_path: Path, key: str = "test_score/openai/gsm8k"):
    if not result_path.exists():
        return None
    for line in reversed(result_path.read_text().splitlines()):
        try:
            payload = ast.literal_eval(line.strip())
        except Exception:
            continue
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    return None


def load_nonmath_avg(run_name: str):
    p = LM_EVAL_RESULTS_DIR / run_name / "results.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text()).get("results", {})
    vals = [data.get(task, {}).get(metric) for task, metric in NONMATH_TASKS.values()]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def load_direct_rl_nonmath_avg(pretrain_step: int):
    """Direct-RL (RL from raw base) non-math avg at the given pretrain step.
    Per seed take the latest rl_step, then average across seeds.
    Mirrors RE_RL_BASE in plot_regression.py."""
    pat = re.compile(
        rf"^olmo2_1b_step{pretrain_step}_omigsm8k_n\d+(_v(?P<seed>\d+))?_step(?P<rl_step>\d+)$"
    )
    by_seed = {}  # seed -> (rl_step, avg)
    for d in LM_EVAL_RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        avg = load_nonmath_avg(d.name)
        if avg is None:
            continue
        seed = int(m.group("seed") or 1)
        rl_step = int(m.group("rl_step"))
        if seed not in by_seed or rl_step > by_seed[seed][0]:
            by_seed[seed] = (rl_step, avg)
    if not by_seed:
        return None
    return sum(v[1] for v in by_seed.values()) / len(by_seed)


# ─── Load GSM curves ──────────────────────────────────────────────────────────
gsm_rows = []
for label, exp_name in EXPERIMENTS.items():
    for base in EVAL_DIRS:
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            m = re.match(
                rf"^{re.escape(exp_name)}-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
                d.name,
            )
            if not m:
                continue
            step = int(m.group(1))
            for k, s in read_pass_at_k(d / "gsm8k_majority_results.txt").items():
                gsm_rows.append({"experiment": label, "step": step, "k": k, "score": s})

gsm_df = pd.DataFrame(gsm_rows).drop_duplicates(subset=["experiment", "step", "k"])
print(f"GSM: {len(gsm_df)} rows across {gsm_df['experiment'].nunique()} experiments")
print(gsm_df.groupby("experiment")["step"].agg(["min", "max", "count"]))


# ─── Load non-math curves (one point per experiment for now) ─────────────────
nm_rows = []
for label, exp_name in EXPERIMENTS.items():
    prefix = exp_name + "_step"
    for d in LM_EVAL_RESULTS_DIR.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        m = re.match(rf"^{re.escape(exp_name)}_step(\d+)$", d.name)
        if not m:
            continue
        avg = load_nonmath_avg(d.name)
        if avg is None:
            continue
        nm_rows.append({"experiment": label, "step": int(m.group(1)), "avg": avg})

nm_df = pd.DataFrame(nm_rows).sort_values(["experiment", "step"])
print(f"\nNon-math: {len(nm_df)} rows")
if not nm_df.empty:
    print(nm_df.to_string(index=False))


# ─── Pretrain (base) 8-shot pass@k references ────────────────────────────────
def find_pretrain_score(samples: int, shot: int = PRETRAIN_SHOT, step: int = PRETRAIN_STEP):
    dir_name = f"1B-step{step}-{shot}shot-{samples}samples-temp{TEMP}"
    fname = "gsm8k_majority_results.txt" if samples > 1 else "gsm8k_results.txt"
    for base in EVAL_DIRS:
        p = base / dir_name / fname
        if not p.exists():
            continue
        scores = read_pass_at_k(p)
        if samples in scores:
            return scores[samples]
        v = read_dict_score(p)
        if v is not None:
            return v
    return None


pretrain_scores = {k: find_pretrain_score(k) for k in PLOT_SAMPLES}
print(f"\nPretrain step{PRETRAIN_STEP} {PRETRAIN_SHOT}-shot scores: {pretrain_scores}")


# ─── Direct-RL references at ~step10000 ──────────────────────────────────────
manual_rl_path = Path(__file__).parent / "manual_rl_gsm.json"
with open(manual_rl_path) as f:
    MANUAL_RL = {int(k): v for k, v in json.load(f).items() if k.isdigit()}


def direct_rl_at_step(samples: int):
    points = MANUAL_RL.get(samples, [])
    if not points:
        return None
    best = min(points, key=lambda p: abs(p[0] - PRETRAIN_TOKENS_B))
    return best[1]


direct_rl_scores = {k: direct_rl_at_step(k) for k in PLOT_SAMPLES}
print(f"Direct-RL @ ~step{PRETRAIN_STEP} ({PRETRAIN_TOKENS_B}B tok) scores: {direct_rl_scores}")


# ─── SFT-Single GSM reference (last RL step) ─────────────────────────────────
sft_single_pat = re.compile(
    rf"^OLMo2-1B_step{PRETRAIN_STEP}_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_gsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$"
)
sft_single_dir, sft_single_step = None, -1
for base in EVAL_DIRS:
    if not base.exists():
        continue
    for d in base.iterdir():
        m = sft_single_pat.match(d.name)
        if m and int(m.group(1)) > sft_single_step:
            sft_single_step = int(m.group(1))
            sft_single_dir = d

sft_single_gsm_scores = {}
if sft_single_dir is not None:
    sft_single_gsm_scores = read_pass_at_k(sft_single_dir / "gsm8k_majority_results.txt")
    print(f"SFT-Single GSM baseline: {sft_single_dir.name} -> {sft_single_gsm_scores}")


# ─── SFT-Single->RL GSM reference (10k pretrain ckpt, last RL step) ──────────
sft_single_rl_pat = re.compile(
    rf"^OLMo2-1B_step{PRETRAIN_STEP}sfted_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_gsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$"
)
sft_single_rl_dir, sft_single_rl_step = None, -1
for base in EVAL_DIRS:
    if not base.exists():
        continue
    for d in base.iterdir():
        m = sft_single_rl_pat.match(d.name)
        if m and int(m.group(1)) > sft_single_rl_step:
            sft_single_rl_step = int(m.group(1))
            sft_single_rl_dir = d

sft_single_rl_scores = {}
if sft_single_rl_dir is not None:
    sft_single_rl_scores = read_pass_at_k(sft_single_rl_dir / "gsm8k_majority_results.txt")
    print(f"SFT-Single->RL GSM baseline: {sft_single_rl_dir.name} -> {sft_single_rl_scores}")


# ─── SFT-Multi GSM reference (last RL step) ──────────────────────────────────
sft_multi_pat = re.compile(
    rf"^OLMo2-1B_step{PRETRAIN_STEP}_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_rgsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$"
)
sft_multi_dir, sft_multi_step = None, -1
for base in EVAL_DIRS:
    if not base.exists():
        continue
    for d in base.iterdir():
        m = sft_multi_pat.match(d.name)
        if m and int(m.group(1)) > sft_multi_step:
            sft_multi_step = int(m.group(1))
            sft_multi_dir = d

sft_multi_gsm_scores = {}
if sft_multi_dir is not None:
    sft_multi_gsm_scores = read_pass_at_k(sft_multi_dir / "gsm8k_majority_results.txt")
    print(f"SFT-Multi GSM baseline: {sft_multi_dir.name} -> {sft_multi_gsm_scores}")


# ─── Non-math base + SFT-Single/Multi references ────────────────────────────
nm_base_avg = load_nonmath_avg(BASE_LM_NAME)
nm_sft_single_avg = load_nonmath_avg(SFT_SINGLE_LM_NAME)
nm_sft_multi_avg = load_nonmath_avg(SFT_MULTI_LM_NAME)
nm_direct_rl_avg = load_direct_rl_nonmath_avg(PRETRAIN_STEP)
print(f"\nNon-math base ({BASE_LM_NAME}): {nm_base_avg}")
print(f"Non-math SFT-Single ({SFT_SINGLE_LM_NAME}): {nm_sft_single_avg}")
print(f"Non-math SFT-Multi ({SFT_MULTI_LM_NAME}): {nm_sft_multi_avg}")
print(f"Non-math Direct-RL @ step{PRETRAIN_STEP}: {nm_direct_rl_avg}")


# ─── Plot ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.fontsize": 22,
})

# Blue gradient by SFT LR — largest LR is the bluest (closest to the SFT-Single
# baseline color), smallest LR is the darkest navy.
_blues = plt.get_cmap("Blues")
styles = {
    "sft4e-5": {"color": _blues(0.45), "marker": "^", "ls": "-",
                "label": r"Parallel avg: $\eta_{\text{SFT}}=4\times10^{-5},\ \eta_{\text{RL}}=10^{-6}$"},
    "sft1e-6": {"color": _blues(0.70), "marker": "o", "ls": "-",
                "label": r"Parallel avg: $\eta_{\text{SFT}}=10^{-6},\ \eta_{\text{RL}}=10^{-6}$"},
    "sft1e-7": {"color": _blues(0.95), "marker": "s", "ls": "-",
                "label": r"Parallel avg: $\eta_{\text{SFT}}=10^{-7},\ \eta_{\text{RL}}=10^{-6}$"},
}

PRETRAIN_REF_STYLE = {"color": "#777777", "ls": ":", "linewidth": 2.5,
                      "label": r"$\mathcal{M}_{10\mathrm{k}}$ (Base, 8-shot)"}
DIRECT_RL_REF_STYLE = {"color": "#E24A33", "ls": "--", "linewidth": 2.5,
                       "label": r"$\mathcal{M}_{10\mathrm{k}}^{\text{RL}}$ (Direct RL)"}
SFT_SINGLE_REF_STYLE = {"color": "#56B4E9", "ls": "--", "linewidth": 2.5,
                        "label": r"$\mathcal{M}_{10\mathrm{k}}^{\text{SFT-Single}}$"}
SFT_MULTI_REF_STYLE = {"color": "#009E73", "ls": "--", "linewidth": 2.5,
                       "label": r"$\mathcal{M}_{10\mathrm{k}}^{\text{SFT-Multi}}$"}
SFT_SINGLE_RL_REF_STYLE = {"color": "#7B3294", "ls": "-.", "linewidth": 2.5,
                           "label": r"$\mathcal{M}_{10\mathrm{k}}^{\text{SFT-Single}\to\text{RL}}$"}

fig, axes = plt.subplots(1, 4, figsize=(24, 6))


def autoscale(ax, ys):
    ys = [y for y in ys if y is not None]
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    pad = max(2.0, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)


# GSM panels (Pass@1, Pass@8, Pass@32)
for idx, k in enumerate(PLOT_SAMPLES):
    ax = axes[idx]
    for exp_label, st in styles.items():
        sub = gsm_df[(gsm_df["experiment"] == exp_label) & (gsm_df["k"] == k)].sort_values("step")
        if sub.empty:
            continue
        ax.plot(
            sub["step"], sub["score"] * 100,
            color=st["color"], marker=st["marker"], ls=st["ls"],
            markersize=10, linewidth=2.5,
            label=st["label"],
        )

    if direct_rl_scores.get(k) is not None:
        ax.axhline(direct_rl_scores[k], **DIRECT_RL_REF_STYLE)
    if sft_single_gsm_scores.get(k) is not None:
        ax.axhline(sft_single_gsm_scores[k] * 100, **SFT_SINGLE_REF_STYLE)
    if sft_multi_gsm_scores.get(k) is not None:
        ax.axhline(sft_multi_gsm_scores[k] * 100, **SFT_MULTI_REF_STYLE)
    if sft_single_rl_scores.get(k) is not None:
        ax.axhline(sft_single_rl_scores[k] * 100, **SFT_SINGLE_RL_REF_STYLE)

    ax.set_title(f"GSM8K Pass@{k}", pad=15)
    ax.set_xlabel("RL training steps")
    if idx == 0:
        ax.set_ylabel("GSM8K Accuracy (%)")

    ys = list(gsm_df[gsm_df["k"] == k]["score"] * 100)
    if direct_rl_scores.get(k) is not None:
        ys.append(direct_rl_scores[k])
    if sft_single_gsm_scores.get(k) is not None:
        ys.append(sft_single_gsm_scores[k] * 100)
    if sft_multi_gsm_scores.get(k) is not None:
        ys.append(sft_multi_gsm_scores[k] * 100)
    if sft_single_rl_scores.get(k) is not None:
        ys.append(sft_single_rl_scores[k] * 100)
    autoscale(ax, ys)

    ax.grid(True, linestyle=":", color="gray", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)


# Non-math panel (5-benchmark average) — bar plot, parallel-avg runs first then baselines
ax = axes[3]
NM_BASE_STYLE = {**PRETRAIN_REF_STYLE, "label": r"$\mathcal{M}_{10\mathrm{k}}$ (Base, 8-shot)"}

PARA_TICK_NAMES = {
    "sft4e-5": "para-avg sft4e-5",
    "sft1e-6": "para-avg sft1e-6",
    "sft1e-7": "para-avg sft1e-7",
}

nm_bar_entries = []  # (legend_label, value%, color, tick_label)
for exp_label, st in styles.items():
    sub = nm_df[nm_df["experiment"] == exp_label].sort_values("step")
    if sub.empty:
        continue
    nm_bar_entries.append((st["label"], sub["avg"].iloc[-1] * 100, st["color"], PARA_TICK_NAMES[exp_label]))
if nm_base_avg is not None:
    nm_bar_entries.append((NM_BASE_STYLE["label"], nm_base_avg * 100, NM_BASE_STYLE["color"], "Base"))
if nm_direct_rl_avg is not None:
    nm_bar_entries.append((DIRECT_RL_REF_STYLE["label"], nm_direct_rl_avg * 100, DIRECT_RL_REF_STYLE["color"], "Direct-RL"))
if nm_sft_single_avg is not None:
    nm_bar_entries.append((SFT_SINGLE_REF_STYLE["label"], nm_sft_single_avg * 100, SFT_SINGLE_REF_STYLE["color"], "SFT-Single"))
if nm_sft_multi_avg is not None:
    nm_bar_entries.append((SFT_MULTI_REF_STYLE["label"], nm_sft_multi_avg * 100, SFT_MULTI_REF_STYLE["color"], "SFT-Multi"))

xs = list(range(len(nm_bar_entries)))
bars = ax.bar(
    xs,
    [e[1] for e in nm_bar_entries],
    color=[e[2] for e in nm_bar_entries],
    edgecolor="black", linewidth=1.0,
)
for rect, e in zip(bars, nm_bar_entries):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3,
            f"{e[1]:.1f}", ha="center", va="bottom", fontsize=16)

ax.set_xticks(xs)
ax.set_xticklabels([e[3] for e in nm_bar_entries], rotation=35, ha="right")
ax.set_title("Non-math 5-bench avg", pad=15)
ax.set_xlabel("")
ax.set_ylabel("Mean accuracy (%)")

ys_nm = [e[1] for e in nm_bar_entries]
if ys_nm:
    lo, hi = min(ys_nm), max(ys_nm)
    pad = max(1.0, 0.20 * (hi - lo))
    ax.set_ylim(max(0, lo - pad), hi + pad)

ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.7)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)


# Combined legend
handles, labels = [], []
seen = set()
for a in axes:
    h, l = a.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in seen:
            handles.append(hi)
            labels.append(li)
            seen.add(li)

plt.tight_layout()
plt.subplots_adjust(bottom=0.38)
fig.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.22),
    ncol=3, frameon=True, framealpha=1.0, borderpad=0.3,
)

out = Path(__file__).parent / "parallel_avg_analysis.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"\nSaved to {out}")
