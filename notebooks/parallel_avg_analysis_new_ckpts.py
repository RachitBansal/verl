"""
Parallel-avg RL (sft1e-6 only) GSM8K pass@k + non-math 5-benchmark average for
the new pretraining checkpoints: step{3000, 5000, 14000, 22000}.

Mirrors notebooks/parallel_avg_analysis.py (which targets step10000 with 3 SFT
LR families). For these new pretrain ckpts only the sft1e-6 family has been
trained/evaluated, so each panel has a single curve.

The non-math panel's parallel-avg bar is left as an empty outline because that
lm_eval run has not been computed yet.

Usage:
    python notebooks/parallel_avg_analysis_new_ckpts.py
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

PRETRAIN_STEPS = [3000, 5000, 14000, 22000]
EXP_TEMPLATE = "OLMo2-1B_step{step}_parallel_avg_n32_rl1e-6_sft1e-6_smbatch"

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
    """Direct-RL non-math avg: per seed take the latest rl_step, then average."""
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
    """Convert a GSM eval dir name like
        OLMo2-1B_step3000_..._gsm-step9500-rl-0shot-boxed-32samples-temp0.6
    into the lm_eval results dir name:
        OLMo2-1B_step3000_..._gsm_step9500
    """
    base = re.sub(r"-rl-0shot-boxed-\d+samples-temp[\d.]+$", "", gsm_dir_name)
    return re.sub(r"-step(\d+)$", r"_step\1", base)


def find_last_dir(pretrain_step: int, suffix_pattern: str):
    """Return the eval dir matching the given regex (with `(\\d+)` rl_step group)
    that has the largest rl_step, searched across EVAL_DIRS."""
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

_blues = plt.get_cmap("Blues")
PARA_STYLE = {
    "color": _blues(0.70), "marker": "o", "ls": "-",
    "label": r"$\mathcal{M}^{\text{Parallel}}$ ($\eta_{\text{SFT}}=10^{-6}$)",
}

PRETRAIN_REF_STYLE = {"color": "#777777", "ls": ":", "linewidth": 2.5}
DIRECT_RL_REF_STYLE = {"color": "#E24A33", "ls": "--", "linewidth": 2.5}
SFT_SINGLE_REF_STYLE = {"color": "#56B4E9", "ls": "--", "linewidth": 2.5}
SFT_MULTI_REF_STYLE = {"color": "#009E73", "ls": "--", "linewidth": 2.5}
SFT_SINGLE_RL_REF_STYLE = {"color": "#7B3294", "ls": "-.", "linewidth": 2.5}


def autoscale(ax, ys):
    ys = [y for y in ys if y is not None]
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    pad = max(2.0, 0.08 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)


def make_plot(pretrain_step: int):
    exp_name = EXP_TEMPLATE.format(step=pretrain_step)
    pretrain_tokens_b = pretrain_step * TOKEN_MULTIPLIER / 1e9
    base_lm_name = f"OLMo2-1B-stage1-50B_step{pretrain_step}-hf"

    # ─── Load GSM curves (sft1e-6 only) ─────────────────────────────────────
    gsm_rows = []
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
                gsm_rows.append({"step": step, "k": k, "score": s})
    gsm_df = pd.DataFrame(gsm_rows).drop_duplicates(subset=["step", "k"])
    print(f"\n=== Pretrain step{pretrain_step} ===")
    print(f"GSM: {len(gsm_df)} rows")
    if not gsm_df.empty:
        print(gsm_df.groupby("k")["step"].agg(["min", "max", "count"]))

    # ─── Non-math (lm_eval) ─────────────────────────────────────────────────
    nm_rows = []
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
        nm_rows.append({"step": int(m.group(1)), "avg": avg})
    nm_df = pd.DataFrame(nm_rows).sort_values("step") if nm_rows else pd.DataFrame()

    # ─── References ─────────────────────────────────────────────────────────
    pretrain_scores = {k: find_pretrain_score(pretrain_step, k) for k in PLOT_SAMPLES}
    direct_rl_scores = {k: direct_rl_at_step(pretrain_step, k) for k in PLOT_SAMPLES}

    sft_single_dir = find_last_dir(
        pretrain_step,
        rf"^OLMo2-1B_step{pretrain_step}_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_gsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
    )
    sft_single_gsm = read_pass_at_k(sft_single_dir / "gsm8k_majority_results.txt") if sft_single_dir else {}

    sft_multi_dir = find_last_dir(
        pretrain_step,
        rf"^OLMo2-1B_step{pretrain_step}_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_rgsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
    )
    sft_multi_gsm = read_pass_at_k(sft_multi_dir / "gsm8k_majority_results.txt") if sft_multi_dir else {}

    sft_single_rl_dir = find_last_dir(
        pretrain_step,
        rf"^OLMo2-1B_step{pretrain_step}sfted_interleave_twoloader_n{N_SAMPLES}_sft_\d+_ppo_\d+_gsm-step(\d+)-rl-0shot-boxed-{N_SAMPLES}samples-temp{TEMP}$",
    )
    sft_single_rl_gsm = read_pass_at_k(sft_single_rl_dir / "gsm8k_majority_results.txt") if sft_single_rl_dir else {}

    nm_base = load_nonmath_avg(base_lm_name)
    nm_sft_single = load_nonmath_avg(gsm_dir_to_lm_eval_name(sft_single_dir.name)) if sft_single_dir else None
    nm_sft_multi = load_nonmath_avg(gsm_dir_to_lm_eval_name(sft_multi_dir.name)) if sft_multi_dir else None
    nm_direct_rl = load_direct_rl_nonmath_avg(pretrain_step)

    print(f"  Base pretrain GSM @ {pretrain_step}: {pretrain_scores}")
    print(f"  Direct-RL @ ~{pretrain_tokens_b}B tok: {direct_rl_scores}")
    print(f"  SFT-Single dir: {sft_single_dir.name if sft_single_dir else None}")
    print(f"  SFT-Multi  dir: {sft_multi_dir.name if sft_multi_dir else None}")
    print(f"  SFT-Single->RL dir: {sft_single_rl_dir.name if sft_single_rl_dir else None}")
    print(f"  Non-math base/SFT-Single/SFT-Multi/Direct-RL: "
          f"{nm_base} / {nm_sft_single} / {nm_sft_multi} / {nm_direct_rl}")

    # ─── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # GSM panels
    for idx, k in enumerate(PLOT_SAMPLES):
        ax = axes[idx]
        sub = gsm_df[gsm_df["k"] == k].sort_values("step")
        if not sub.empty:
            ax.plot(
                sub["step"], sub["score"] * 100,
                color=PARA_STYLE["color"], marker=PARA_STYLE["marker"], ls=PARA_STYLE["ls"],
                markersize=10, linewidth=2.5,
                label=PARA_STYLE["label"],
            )
        if direct_rl_scores.get(k) is not None:
            ax.axhline(direct_rl_scores[k], **DIRECT_RL_REF_STYLE,
                       label=r"$\mathcal{M}^{\text{RL}}$ (Direct RL)")
        if sft_single_gsm.get(k) is not None:
            ax.axhline(sft_single_gsm[k] * 100, **SFT_SINGLE_REF_STYLE,
                       label=r"$\mathcal{M}^{\text{SFT-Single}}$")
        if sft_multi_gsm.get(k) is not None:
            ax.axhline(sft_multi_gsm[k] * 100, **SFT_MULTI_REF_STYLE,
                       label=r"$\mathcal{M}^{\text{SFT-Multi}}$")
        if sft_single_rl_gsm.get(k) is not None:
            ax.axhline(sft_single_rl_gsm[k] * 100, **SFT_SINGLE_RL_REF_STYLE,
                       label=r"$\mathcal{M}^{\text{SFT-Single}\to\text{RL}}$")

        ax.set_title(f"GSM8K Pass@{k}", pad=15)
        ax.set_xlabel("RL training steps")
        if idx == 0:
            ax.set_ylabel("GSM8K Accuracy (%)")

        ys = list(sub["score"] * 100) if not sub.empty else []
        if direct_rl_scores.get(k) is not None:
            ys.append(direct_rl_scores[k])
        if sft_single_gsm.get(k) is not None:
            ys.append(sft_single_gsm[k] * 100)
        if sft_multi_gsm.get(k) is not None:
            ys.append(sft_multi_gsm[k] * 100)
        if sft_single_rl_gsm.get(k) is not None:
            ys.append(sft_single_rl_gsm[k] * 100)
        autoscale(ax, ys)

        ax.grid(True, linestyle=":", color="gray", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

    # Non-math bar panel
    ax = axes[3]
    para_avg = nm_df["avg"].iloc[-1] * 100 if not nm_df.empty else None
    entries = [
        ("Para sft1e-6", para_avg, PARA_STYLE["color"], r"$\mathcal{M}^{\text{Para}}$"),
        ("Base",        nm_base * 100 if nm_base is not None else None,
                        PRETRAIN_REF_STYLE["color"], "Base"),
        ("Direct-RL",   nm_direct_rl * 100 if nm_direct_rl is not None else None,
                        DIRECT_RL_REF_STYLE["color"], "Direct-RL"),
        ("SFT-Single",  nm_sft_single * 100 if nm_sft_single is not None else None,
                        SFT_SINGLE_REF_STYLE["color"], "SFT-Single"),
        ("SFT-Multi",   nm_sft_multi * 100 if nm_sft_multi is not None else None,
                        SFT_MULTI_REF_STYLE["color"], "SFT-Multi"),
    ]

    xs = list(range(len(entries)))
    for x, (_, val, color, _tick) in zip(xs, entries):
        if val is None:
            # Empty outlined bar to indicate missing data.
            ax.bar(x, 0, color="none", edgecolor=color, linewidth=1.5, hatch="//")
        else:
            rect = ax.bar(x, val, color=color, edgecolor="black", linewidth=1.0)[0]
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=16)

    ax.set_xticks(xs)
    ax.set_xticklabels([e[3] for e in entries], rotation=35, ha="right")
    ax.set_title("Non-math 5-bench avg", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Mean accuracy (%)")

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

    # Legend
    handles, labels = [], []
    seen = set()
    for a in axes:
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                handles.append(hi)
                labels.append(li)
                seen.add(li)

    fig.suptitle(
        rf"Pretrain step{pretrain_step} (~{pretrain_tokens_b:.0f}B tok)",
        y=1.02, fontsize=24,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38, wspace=0.35)
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.22),
        ncol=3, frameon=True, framealpha=1.0, borderpad=0.3,
    )

    out = Path(__file__).parent / f"parallel_avg_analysis_step{pretrain_step}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved {out}")
    plt.close(fig)


for s in PRETRAIN_STEPS:
    make_plot(s)
