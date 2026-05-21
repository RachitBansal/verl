"""
Combined parallel-avg RL analysis: one row per pretrain ckpt
(step{3000, 5000, 10000, 14000, 22000}), four columns
(GSM Pass@1, Pass@8, Pass@32, Non-math 5-bench avg).

Column titles render only on the top row; a single shared legend sits at the
bottom of the figure.

For pretrain step10000 three SFT LR families are plotted (sft1e-6, sft1e-7,
sft4e-5). For the other rows only sft1e-6 was trained, so each panel has one
curve. The non-math parallel-avg bar is shown as an empty hatched outline when
lm_eval data is missing for that LR/pretrain combination.

Usage:
    python notebooks/parallel_avg_analysis_combined.py
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

PRETRAIN_STEPS = [3000, 5000, 10000, 14000, 22000]

PLOT_SAMPLES = [1, 8, 32]
N_SAMPLES = 32
TEMP = 0.6
PRETRAIN_SHOT = 8
TOKEN_MULTIPLIER = 2_000_000

NONMATH_TASKS = {
    "lambada_acc": ("lambada_openai", "acc,none"),
    "hellaswag":   ("hellaswag",      "acc_norm,none"),
    "arc_easy":    ("arc_easy",       "acc_norm,none"),
    "piqa":        ("piqa",           "acc_norm,none"),
    "openbookqa":  ("openbookqa",     "acc_norm,none"),
}


# ─── Per-pretrain-step experiment LR families ───────────────────────────────
def get_experiments(pretrain_step: int) -> dict:
    """Return mapping of {LR label: experiment dir name}. Only step10000 has
    multiple SFT LR runs; the other steps use sft1e-6 alone."""
    base = f"OLMo2-1B_step{pretrain_step}_parallel_avg_n32_rl1e-6"
    if pretrain_step == 10000:
        return {
            "sft1e-6": f"{base}_sft1e-6_smbatch",
            "sft1e-7": f"{base}_sft1e-7_smbatch",
            "sft4e-5": f"{base}_sft4e-5_smbatch",
        }
    return {"sft1e-6": f"{base}_sft1e-6_smbatch"}


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
    pat = re.compile(
        rf"^olmo2_1b_step{pretrain_step}_omigsm8k_n\d+(_v(?P<seed>\d+))?_step(?P<rl_step>\d+)$"
    )
    by_seed = {}
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


def find_pretrain_score(step: int, samples: int, shot: int = PRETRAIN_SHOT):
    """Try the dedicated {samples}-sample dir first; fall back to the 32-sample
    majority file (which lists pass@1, pass@2, ..., pass@32)."""
    candidate_dirs = []
    for prefix in (f"1B-stage1-50B-step{step}", f"1B-step{step}"):
        candidate_dirs.append((f"{prefix}-{shot}shot-{samples}samples-temp{TEMP}", samples))
        if samples != N_SAMPLES:
            candidate_dirs.append((f"{prefix}-{shot}shot-{N_SAMPLES}samples-temp{TEMP}", samples))
    for dir_name, want_k in candidate_dirs:
        fname = "gsm8k_majority_results.txt" if dir_name.endswith(f"{N_SAMPLES}samples-temp{TEMP}") or want_k > 1 else "gsm8k_results.txt"
        for base in EVAL_DIRS:
            p = base / dir_name / fname
            if not p.exists():
                continue
            scores = read_pass_at_k(p)
            if want_k in scores:
                return scores[want_k]
            if want_k == samples:
                v = read_dict_score(p)
                if v is not None:
                    return v
    return None


def gsm_dir_to_lm_eval_name(gsm_dir_name: str) -> str:
    base = re.sub(r"-rl-0shot-boxed-\d+samples-temp[\d.]+$", "", gsm_dir_name)
    return re.sub(r"-step(\d+)$", r"_step\1", base)


def find_last_dir(suffix_pattern: str):
    pat = re.compile(suffix_pattern)
    best_dir, best_step = None, -1
    for base in EVAL_DIRS:
        if not base.exists():
            continue
        for d in base.iterdir():
            m = pat.match(d.name)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best_dir = d
    return best_dir


# ─── Direct-RL reference (manual) ────────────────────────────────────────────
manual_rl_path = Path(__file__).parent / "manual_rl_gsm.json"
with open(manual_rl_path) as f:
    MANUAL_RL = {int(k): v for k, v in json.load(f).items() if k.isdigit()}


def direct_rl_at_step(pretrain_step: int, samples: int):
    points = MANUAL_RL.get(samples, [])
    if not points:
        return None
    target_tokens_b = pretrain_step * TOKEN_MULTIPLIER / 1e9
    best = min(points, key=lambda p: abs(p[0] - target_tokens_b))
    return best[1]


# ─── Plot styles ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.7)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.fontsize": 30,
})

_blues = plt.get_cmap("Blues")
LR_STYLES = {
    "sft4e-5": {"color": _blues(0.45), "marker": "^", "ls": "-",
                "label": r"$\mathcal{M}^{\text{Parallel}}$ ($\eta_{\text{SFT}}=4\times10^{-5}$)"},
    "sft1e-6": {"color": _blues(0.70), "marker": "o", "ls": "-",
                "label": r"$\mathcal{M}^{\text{Parallel}}$ ($\eta_{\text{SFT}}=10^{-6}$)"},
    "sft1e-7": {"color": _blues(0.95), "marker": "s", "ls": "-",
                "label": r"$\mathcal{M}^{\text{Parallel}}$ ($\eta_{\text{SFT}}=10^{-7}$)"},
}

PRETRAIN_REF_STYLE = {"color": "#777777", "ls": ":", "linewidth": 4.0,
                      "label": r"$\mathcal{M}$ (Base, 8-shot)"}
DIRECT_RL_REF_STYLE = {"color": "#E24A33", "ls": "--", "linewidth": 4.0,
                       "label": r"$\mathcal{M}^{\text{RL}}$ (Direct RL)"}
SFT_SINGLE_REF_STYLE = {"color": "#E69F00", "ls": "--", "linewidth": 4.0,
                        "label": r"$\mathcal{M}^{\text{SFT-Single}}$"}
SFT_MULTI_REF_STYLE = {"color": "#009E73", "ls": "--", "linewidth": 4.0,
                       "label": r"$\mathcal{M}^{\text{SFT-Multi}}$"}
SFT_SINGLE_RL_REF_STYLE = {"color": "#7B3294", "ls": "-.", "linewidth": 4.0,
                           "label": r"$\mathcal{M}^{\text{SFT-Single}\to\text{RL}}$"}


def autoscale(ax, ys):
    ys = [y for y in ys if y is not None]
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    pad = max(2.0, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)


# ─── Per-row data loading ────────────────────────────────────────────────────
def load_row_data(pretrain_step: int):
    experiments = get_experiments(pretrain_step)

    gsm_rows, nm_rows = [], []
    for label, exp_name in experiments.items():
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
                rl_step = int(m.group(1))
                for k, s in read_pass_at_k(d / "gsm8k_majority_results.txt").items():
                    gsm_rows.append({"lr": label, "step": rl_step, "k": k, "score": s})

        # lm_eval non-math per parallel-avg LR
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
            nm_rows.append({"lr": label, "step": int(m.group(1)), "avg": avg})

    gsm_df = pd.DataFrame(gsm_rows).drop_duplicates(subset=["lr", "step", "k"])
    nm_df = pd.DataFrame(nm_rows).sort_values(["lr", "step"]) if nm_rows else pd.DataFrame()

    pretrain_scores = {k: find_pretrain_score(pretrain_step, k) for k in PLOT_SAMPLES}
    direct_rl_scores = {k: direct_rl_at_step(pretrain_step, k) for k in PLOT_SAMPLES}

    sft_single_dir = find_last_dir(
        rf"^OLMo2-1B_step{pretrain_step}_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_gsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
    )
    sft_single_gsm = read_pass_at_k(sft_single_dir / "gsm8k_majority_results.txt") if sft_single_dir else {}

    sft_multi_dir = find_last_dir(
        rf"^OLMo2-1B_step{pretrain_step}_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_rgsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
    )
    sft_multi_gsm = read_pass_at_k(sft_multi_dir / "gsm8k_majority_results.txt") if sft_multi_dir else {}

    sft_single_rl_dir = find_last_dir(
        rf"^OLMo2-1B_step{pretrain_step}sfted_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_gsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
    )
    sft_single_rl_gsm = read_pass_at_k(sft_single_rl_dir / "gsm8k_majority_results.txt") if sft_single_rl_dir else {}

    base_lm_name = f"OLMo2-1B-stage1-50B_step{pretrain_step}-hf"
    nm_base = load_nonmath_avg(base_lm_name)
    nm_sft_single = load_nonmath_avg(gsm_dir_to_lm_eval_name(sft_single_dir.name)) if sft_single_dir else None
    nm_sft_multi = load_nonmath_avg(gsm_dir_to_lm_eval_name(sft_multi_dir.name)) if sft_multi_dir else None
    nm_direct_rl = load_direct_rl_nonmath_avg(pretrain_step)

    return {
        "experiments": experiments,
        "gsm_df": gsm_df,
        "nm_df": nm_df,
        "pretrain_scores": pretrain_scores,
        "direct_rl_scores": direct_rl_scores,
        "sft_single_gsm": sft_single_gsm,
        "sft_multi_gsm": sft_multi_gsm,
        "sft_single_rl_gsm": sft_single_rl_gsm,
        "nm_base": nm_base,
        "nm_sft_single": nm_sft_single,
        "nm_sft_multi": nm_sft_multi,
        "nm_direct_rl": nm_direct_rl,
    }


# ─── Draw one row of the figure ──────────────────────────────────────────────
def draw_row(axes_row, pretrain_step: int, data: dict, show_titles: bool):
    pretrain_tokens_b = pretrain_step * TOKEN_MULTIPLIER / 1e9

    # GSM panels
    for idx, k in enumerate(PLOT_SAMPLES):
        ax = axes_row[idx]
        for lr_label, st in LR_STYLES.items():
            if lr_label not in data["experiments"]:
                continue
            sub = data["gsm_df"][(data["gsm_df"]["lr"] == lr_label) & (data["gsm_df"]["k"] == k)].sort_values("step")
            if sub.empty:
                continue
            ax.plot(
                sub["step"], sub["score"] * 100,
                color=st["color"], marker=st["marker"], ls=st["ls"],
                markersize=9, linewidth=2.3,
                label=st["label"],
            )

        if data["direct_rl_scores"].get(k) is not None:
            ax.axhline(data["direct_rl_scores"][k], **DIRECT_RL_REF_STYLE)
        if data["sft_single_gsm"].get(k) is not None:
            ax.axhline(data["sft_single_gsm"][k] * 100, **SFT_SINGLE_REF_STYLE)
        if data["sft_multi_gsm"].get(k) is not None:
            ax.axhline(data["sft_multi_gsm"][k] * 100, **SFT_MULTI_REF_STYLE)
        if data["sft_single_rl_gsm"].get(k) is not None:
            ax.axhline(data["sft_single_rl_gsm"][k] * 100, **SFT_SINGLE_RL_REF_STYLE)

        if show_titles:
            ax.set_title(f"GSM8K Pass@{k}", pad=12)
        if idx == 0:
            ax.set_ylabel(rf"Pre-train step {pretrain_tokens_b:.0f}B" + "\nGSM8K Acc (%)")

        ys = list(data["gsm_df"][data["gsm_df"]["k"] == k]["score"] * 100)
        if data["direct_rl_scores"].get(k) is not None:
            ys.append(data["direct_rl_scores"][k])
        if data["sft_single_gsm"].get(k) is not None:
            ys.append(data["sft_single_gsm"][k] * 100)
        if data["sft_multi_gsm"].get(k) is not None:
            ys.append(data["sft_multi_gsm"][k] * 100)
        if data["sft_single_rl_gsm"].get(k) is not None:
            ys.append(data["sft_single_rl_gsm"][k] * 100)
        autoscale(ax, ys)

        ax.grid(True, linestyle=":", color="gray", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

    # Non-math bar panel
    ax = axes_row[3]
    # Per-LR parallel-avg bars first (one per LR family present)
    entries = []
    for lr_label in data["experiments"]:
        sub = data["nm_df"][data["nm_df"]["lr"] == lr_label] if not data["nm_df"].empty else pd.DataFrame()
        val = sub["avg"].iloc[-1] * 100 if not sub.empty else None
        tick = LR_STYLES[lr_label]["label"].split("(")[1].rstrip(")")  # e.g. "η_SFT=10^-6"
        # Use a short tick for compactness
        short = {"sft1e-6": r"$\mathcal{M}^{\text{Para}}_{6}$",
                 "sft1e-7": r"$\mathcal{M}^{\text{Para}}_{7}$",
                 "sft4e-5": r"$\mathcal{M}^{\text{Para}}_{4e\!-\!5}$"}.get(lr_label, "Para")
        entries.append((LR_STYLES[lr_label]["label"], val, LR_STYLES[lr_label]["color"], short))

    entries.extend([
        (PRETRAIN_REF_STYLE["label"], data["nm_base"] * 100 if data["nm_base"] is not None else None,
         PRETRAIN_REF_STYLE["color"], "Base"),
        (DIRECT_RL_REF_STYLE["label"], data["nm_direct_rl"] * 100 if data["nm_direct_rl"] is not None else None,
         DIRECT_RL_REF_STYLE["color"], "Direct-RL"),
        (SFT_SINGLE_REF_STYLE["label"], data["nm_sft_single"] * 100 if data["nm_sft_single"] is not None else None,
         SFT_SINGLE_REF_STYLE["color"], "SFT-Single"),
        (SFT_MULTI_REF_STYLE["label"], data["nm_sft_multi"] * 100 if data["nm_sft_multi"] is not None else None,
         SFT_MULTI_REF_STYLE["color"], "SFT-Multi"),
    ])

    xs = list(range(len(entries)))
    for x, (_, val, color, _tick) in zip(xs, entries):
        if val is None:
            ax.bar(x, 0, color="none", edgecolor=color, linewidth=1.5, hatch="//")
        else:
            rect = ax.bar(x, val, color=color, edgecolor="black", linewidth=1.0)[0]
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=18)

    ax.set_xticks(xs)
    ax.set_xticklabels([e[3] for e in entries], rotation=35, ha="right", fontsize=20)
    if show_titles:
        ax.set_title("Non-math 5-bench avg", pad=12)

    ys_nm = [e[1] for e in entries if e[1] is not None]
    if ys_nm:
        lo, hi = min(ys_nm), max(ys_nm)
        pad = max(1.0, 0.20 * (hi - lo))
        ax.set_ylim(max(0, lo - pad), hi + pad)

    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)


# ─── Build figure ────────────────────────────────────────────────────────────
n_rows = len(PRETRAIN_STEPS)
fig, axes = plt.subplots(n_rows, 4, figsize=(26, 4.6 * n_rows))
if n_rows == 1:
    axes = [axes]

for row_idx, pt_step in enumerate(PRETRAIN_STEPS):
    print(f"\n=== Pretrain step{pt_step} ===")
    data = load_row_data(pt_step)
    print(f"  GSM rows: {len(data['gsm_df'])}; non-math rows: {len(data['nm_df'])}")
    print(f"  pretrain pass@k: {data['pretrain_scores']}")
    print(f"  direct-RL pass@k: {data['direct_rl_scores']}")
    draw_row(axes[row_idx], pt_step, data, show_titles=(row_idx == 0))

# Bottom-row x-axis labels for GSM panels
for ax in axes[-1][:3]:
    ax.set_xlabel("RL training steps")
axes[-1][3].set_xlabel("")

# Shared legend at the bottom
handles, labels = [], []
seen = set()
for row in axes:
    for a in row:
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                handles.append(hi)
                labels.append(li)
                seen.add(li)

# Add baseline reference handles (axhlines emit handles only when labeled
# the first time we draw them — they were drawn with **STYLE but without label
# above, so build them manually from the style dicts for the legend).
from matplotlib.lines import Line2D
extra_refs = [
    PRETRAIN_REF_STYLE,
    DIRECT_RL_REF_STYLE,
    SFT_SINGLE_REF_STYLE,
    SFT_MULTI_REF_STYLE,
    SFT_SINGLE_RL_REF_STYLE,
]
for st in extra_refs:
    if st["label"] not in seen:
        handles.append(Line2D([0, 1], [0, 0], color=st["color"], ls=st["ls"], linewidth=st["linewidth"]))
        labels.append(st["label"])
        seen.add(st["label"])

# Order legend: 3 parallel-avg entries first (row 1), then 5 baselines.
desired_order = [
    LR_STYLES["sft1e-6"]["label"],
    LR_STYLES["sft1e-7"]["label"],
    LR_STYLES["sft4e-5"]["label"],
    DIRECT_RL_REF_STYLE["label"],
    SFT_SINGLE_REF_STYLE["label"],
    SFT_MULTI_REF_STYLE["label"],
    SFT_SINGLE_RL_REF_STYLE["label"],
    PRETRAIN_REF_STYLE["label"],
]
order_lookup = {lbl: i for i, lbl in enumerate(desired_order)}
sorted_pairs = sorted(zip(handles, labels), key=lambda pair: order_lookup.get(pair[1], 99))
sorted_handles, sorted_labels = zip(*sorted_pairs)

plt.tight_layout()
plt.subplots_adjust(bottom=0.10, wspace=0.32, hspace=0.50)
fig.legend(
    sorted_handles, sorted_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.06),
    ncol=3, frameon=True, framealpha=1.0, borderpad=0.4,
)

out = Path(__file__).parent / "parallel_avg_analysis_combined.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"\nSaved {out}")
print(f"Saved {out.with_suffix('.png')}")
