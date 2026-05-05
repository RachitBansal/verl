"""
Plot RL-effectiveness (test set) vs base-model Pass@k on the matching RL train set.

A long 1xN figure: x-axis of each panel is base-model Pass@k on the RL train set
(omi_gsm parquet for GSM rows, omi_math parquet for MATH rows). Two delta lines
per dataset: ΔPass@1 (RL - base) and ΔPass@32 on test.
Marker shape differentiates dataset (o=GSM, s=MATH); marker color encodes
pretraining step.

Caches per-parquet scoring under <parquet_dir>/{omi_gsm,omi_math}_scores_cache.json.

Usage:
    python notebooks/plot_base_metric_rl.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Make verl scoring importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIRS = [
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results"),
    Path("/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results_sunny"),
]
TARGET_TEMP = 0.6
TARGET_SHOT = 8

# Panels (columns) in the long figure: x-axis = base-model Pass@k on the omi train set
X_PASS_KS = [2, 4, 8, 16, 32]
# Two y-series per dataset: ΔPass@1 and ΔPass@32 on test set
Y_DELTA_KS = [1, 32]


# ─── Per-dataset configuration ────────────────────────────────────────────────
def _gsm_score_one(args):
    response, gt = args
    from verl.utils.reward_score.openmathinstruct import compute_score
    try:
        return float(compute_score(response, gt))
    except Exception:
        return 0.0


def _math_score_one(args):
    response, gt = args
    from verl.utils.reward_score.math_reward import compute_score
    try:
        return float(compute_score(response, gt))
    except Exception:
        return 0.0


DATASETS = {
    "gsm": {
        "label": "GSM8K",
        "color": "#1f77b4",   # blue
        "marker": "o",
        "omi_parquet": "omi_gsm_predictions.parquet",
        "omi_cache": "omi_gsm_scores_cache.json",
        "omi_scorer": _gsm_score_one,
        "test_majority": "gsm8k_majority_results.txt",
        # Each treatment: model trained from base via different recipe; we plot
        # treatment_pass_k - base_pass_k vs base Pass@k on omi train.
        "treatments": {
            "rl": {
                "label": "RL",
                "rollouts": 32,
                "patterns": [
                    re.compile(r"olmo2_1b_step(?P<pt_step>\d+)_omigsm8k_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
                    re.compile(r"olmo2_1b_step(?P<pt_step>\d+)_omigsm8k_n(?P<num_rollouts>\d+)_v(?P<seed>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
                ],
            },
            "sft_gsm": {
                "label": "SFT (gsm)",
                "rollouts": 32,
                "patterns": [
                    re.compile(r"OLMo2-1B_step(?P<pt_step>\d+)_interleave_twoloader_n(?P<num_rollouts>\d+)_sft_\d+_ppo_0_gsm-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
                ],
            },
            "sft_rgsm": {
                "label": "SFT (rgsm)",
                "rollouts": 32,
                "patterns": [
                    re.compile(r"OLMo2-1B_step(?P<pt_step>\d+)_interleave_twoloader_n(?P<num_rollouts>\d+)_sft_\d+_ppo_0_rgsm-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
                ],
            },
        },
    },
    "math": {
        "label": "MATH",
        "color": "#d62728",   # red
        "marker": "s",
        "omi_parquet": "omi_math_predictions.parquet",
        "omi_cache": "omi_math_scores_cache.json",
        "omi_scorer": _math_score_one,
        "test_majority": "math_majority_results.txt",
        "treatments": {
            "rl": {
                "label": "RL",
                "rollouts": 64,
                "patterns": [
                    re.compile(r"olmo2_1b_step(?P<pt_step>\d+)_omi_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
                    re.compile(r"olmo2_1b_step(?P<pt_step>\d+)_omi_n(?P<num_rollouts>\d+)_v(?P<seed>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
                ],
            },
        },
    },
}


# ─── Score parsing from majority files ───────────────────────────────────────
def read_majority_metrics(result_path: Path) -> dict:
    """Parse Pass@k + mean@k from a *_majority_results.txt file."""
    if not result_path.exists():
        return {}
    text = result_path.read_text().splitlines()
    pass_scores: dict[int, float] = {}
    mean_correct: float | None = None
    samples_per_example: int | None = None
    for line in text:
        if "Pass@" in line and ":" in line:
            m = re.search(r"Pass@(\d+)\s*:\s*([0-9.]+)", line)
            if m:
                pass_scores[int(m.group(1))] = float(m.group(2))
        avg_m = re.search(r"Average # Correct per Example:\s*([0-9.]+)\s*/\s*(\d+)", line)
        if avg_m:
            mean_correct = float(avg_m.group(1))
            samples_per_example = int(avg_m.group(2))
    out = {"pass": pass_scores}
    if mean_correct is not None and samples_per_example:
        out["mean"] = {samples_per_example: mean_correct / samples_per_example}
    return out


# ─── Score parquet predictions ───────────────────────────────────────────────
def score_parquet(parquet_path: Path, cache_filename: str, scorer_fn, n_workers: int = 16) -> dict:
    """Score a predictions parquet -> {'pass_at_k': {k: float}, 'mean_at_n': {n: float}}.

    Cached at <parquet_dir>/<cache_filename>.
    """
    cache_path = parquet_path.parent / cache_filename
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if "cond_mean_at_n" in cached:  # schema-v2
                cached["pass_at_k"] = {int(k): v for k, v in cached.get("pass_at_k", {}).items()}
                cached["mean_at_n"] = {int(k): v for k, v in cached.get("mean_at_n", {}).items()}
                cached["cond_mean_at_n"] = {int(k): v for k, v in cached.get("cond_mean_at_n", {}).items()}
                return cached
        except Exception:
            pass

    if not parquet_path.exists():
        return {}

    df = pd.read_parquet(parquet_path)
    n_samples = len(df["responses"].iloc[0])

    work = []
    per_example_idx = []
    for _, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else row["expected_answer"]
        for resp in row["responses"]:
            work.append((resp, gt))
        per_example_idx.append(len(row["responses"]))

    print(f"  Scoring {len(work)} responses ({len(df)} examples × {n_samples}) from {parquet_path.parent.name}/{parquet_path.name}...")

    from multiprocessing import Pool
    with Pool(n_workers) as pool:
        flat_scores = pool.map(scorer_fn, work)

    scores_per_example: list[list[float]] = []
    cursor = 0
    for n in per_example_idx:
        scores_per_example.append(flat_scores[cursor : cursor + n])
        cursor += n

    import math
    n = n_samples
    correct_counts = [int(round(sum(s > 0 for s in row))) for row in scores_per_example]

    def pass_at_k(num_correct: int, n: int, k: int) -> float:
        if n - num_correct < k:
            return 1.0
        return 1.0 - math.comb(n - num_correct, k) / math.comb(n, k)

    pass_at_k_scores = {}
    for k in [1, 2, 4, 8, 16, 32]:
        if k > n:
            continue
        pass_at_k_scores[k] = sum(pass_at_k(c, n, k) for c in correct_counts) / len(correct_counts)
    mean_at_n = {n: sum(correct_counts) / (len(correct_counts) * n)}
    nonzero = [c for c in correct_counts if c > 0]
    cond_mean_at_n = {n: (sum(nonzero) / (len(nonzero) * n)) if nonzero else 0.0}

    result = {
        "pass_at_k": pass_at_k_scores,
        "mean_at_n": mean_at_n,
        "cond_mean_at_n": cond_mean_at_n,
        "num_nonzero_examples": len(nonzero),
        "n_samples": n,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    return result


# ─── Locate eval dirs ────────────────────────────────────────────────────────
PRE_PATTERN = re.compile(
    r"1B-(?:stage1-50B-)?step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
)


def find_first(name: str) -> Path | None:
    for base in BASE_DIRS:
        p = base / name
        if p.exists():
            return p
    return None


def collect_base_pretrain_steps() -> list[int]:
    steps = set()
    for base in BASE_DIRS:
        if not base.exists():
            continue
        for path in base.iterdir():
            if not path.is_dir():
                continue
            m = PRE_PATTERN.match(path.name)
            if not m:
                continue
            if int(m.group("samples")) != 32 or int(m.group("shot")) != TARGET_SHOT:
                continue
            if float(m.group("temp")) != TARGET_TEMP:
                continue
            steps.add(int(m.group("step")))
    return sorted(steps)


def base_test_majority_path(step: int, filename: str) -> Path | None:
    # Prefer the new 1B-stage1-50B-step... dirs (where current sweep writes test-set
    # majorities); fall back to the old 1B-step... dirs.
    for name in (
        f"1B-stage1-50B-step{step}-{TARGET_SHOT}shot-32samples-temp{TARGET_TEMP}",
        f"1B-step{step}-{TARGET_SHOT}shot-32samples-temp{TARGET_TEMP}",
    ):
        p = find_first(name)
        if p is None:
            continue
        candidate = p / filename
        if candidate.exists():
            return candidate
    return None


def base_omi_parquet(step: int, filename: str) -> Path | None:
    name = f"1B-stage1-50B-step{step}-{TARGET_SHOT}shot-32samples-temp{TARGET_TEMP}"
    p = find_first(name)
    return None if p is None else p / filename


def collect_treatment_runs(patterns, rollouts: int):
    """pt_step -> list[(rl_step, seed, dir_path)] (samples=32, temp=0.6, given rollouts)."""
    runs: dict[int, list[tuple[int, int, Path]]] = {}
    for base in BASE_DIRS:
        if not base.exists():
            continue
        for path in base.iterdir():
            if not path.is_dir():
                continue
            for pat in patterns:
                m = pat.match(path.name)
                if not m:
                    continue
                if int(m.group("samples")) != 32 or float(m.group("temp")) != TARGET_TEMP:
                    break
                if int(m.group("num_rollouts")) != rollouts:
                    break
                pt_step = int(m.group("pt_step"))
                rl_step = int(m.group("rl_step"))
                seed = int(m.group("seed")) if "seed" in m.groupdict() and m.group("seed") else 1
                runs.setdefault(pt_step, []).append((rl_step, seed, path))
                break
    return runs


def best_treatment_metrics(runs_for_step, test_filename: str):
    """Pick the run with the highest Pass@32; among multiple rl_steps per seed, take the latest.

    Eval dirs are duplicated across BASE_DIRS (some copies missing test files); for
    each (seed, rl_step) prefer a path that actually has the test file.
    """
    if not runs_for_step:
        return None
    candidates: dict[int, dict[int, list[Path]]] = {}
    for rl_step, seed, path in runs_for_step:
        candidates.setdefault(seed, {}).setdefault(rl_step, []).append(path)

    best = None
    for seed, by_step in candidates.items():
        latest_step = max(by_step)
        chosen = next((p for p in by_step[latest_step] if (p / test_filename).exists()), None)
        if chosen is None:
            continue
        metrics = read_majority_metrics(chosen / test_filename)
        score32 = metrics.get("pass", {}).get(32)
        if score32 is None:
            continue
        if best is None or score32 > best["pass"].get(32, -1):
            best = metrics
            best["_rl_step"] = latest_step
            best["_seed"] = seed
    return best


# ─── Build joined table for one dataset ──────────────────────────────────────
def build_dataset_rows(dataset_key: str, cfg: dict) -> list[dict]:
    rows = []
    treatments = cfg["treatments"]
    treatment_runs = {
        tname: collect_treatment_runs(t["patterns"], t["rollouts"])
        for tname, t in treatments.items()
    }
    for step in collect_base_pretrain_steps():
        # Base — omi parquet (RL train set). This is the only required input.
        omi_path = base_omi_parquet(step, cfg["omi_parquet"])
        if omi_path is None or not omi_path.exists():
            print(f"  [{dataset_key} skip step={step}] no {cfg['omi_parquet']}")
            continue
        omi_scores = score_parquet(omi_path, cfg["omi_cache"], cfg["omi_scorer"])
        if not omi_scores:
            print(f"  [{dataset_key} skip step={step}] omi scoring failed")
            continue

        # Base — test set (majority); optional
        base_test = {}
        base_test_path = base_test_majority_path(step, cfg["test_majority"])
        if base_test_path is not None and base_test_path.exists():
            base_test = read_majority_metrics(base_test_path)

        row = {
            "dataset": dataset_key,
            "pt_step": step,
            "base_test_pass_1": base_test.get("pass", {}).get(1),
            "base_test_pass_32": base_test.get("pass", {}).get(32),
            "base_test_mean_32": base_test.get("mean", {}).get(32),
            "omi_mean_32": omi_scores["mean_at_n"].get(32),
        }
        for kk in [1, 2, 4, 8, 16, 32]:
            row[f"omi_pass_{kk}"] = omi_scores["pass_at_k"].get(kk)

        for tname, runs in treatment_runs.items():
            best = best_treatment_metrics(runs.get(step, []), cfg["test_majority"])
            row[f"{tname}_test_pass_1"] = best["pass"].get(1) if best else None
            row[f"{tname}_test_pass_32"] = best["pass"].get(32) if best else None
            row[f"{tname}_rl_step"] = best.get("_rl_step") if best else None
        rows.append(row)
    return rows


def build_table() -> pd.DataFrame:
    all_rows = []
    for key, cfg in DATASETS.items():
        all_rows.extend(build_dataset_rows(key, cfg))
    return pd.DataFrame(all_rows)


def _col(metric_kind: str, k: int, kind: str) -> str:
    """kind in {'omi','base_test','rl_test'}; metric_kind in {'pass','mean','cond_mean'}."""
    suffix = {"pass": "pass", "mean": "mean", "cond_mean": "cond_mean"}[metric_kind]
    if kind == "omi":
        return f"omi_{suffix}_{k}"
    # cond_mean only available on omi (parquet); fall back gracefully
    return f"{kind}_{suffix}_{k}"


# ─── Plot ────────────────────────────────────────────────────────────────────
def plot(df: pd.DataFrame, output_path: Path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "legend.fontsize": 10,
    })

    n_panels = len(X_PASS_KS)
    # One row per (dataset, treatment); each row plots Δ = treatment - base on test
    row_specs: list[tuple[str, str]] = []
    for ds_key, cfg in DATASETS.items():
        for tname in cfg["treatments"]:
            row_specs.append((ds_key, tname))
    n_rows = len(row_specs)
    fig, axes = plt.subplots(n_rows, n_panels,
                             figsize=(3.7 * n_panels, 3.6 * n_rows),
                             sharey="row")

    # Color scale across all pretraining steps that appear with usable data (any treatment)
    treatment_cols = [f"{t}_test_pass_32" for ds in DATASETS.values() for t in ds["treatments"]]
    usable_mask = (
        df["base_test_pass_32"].notna() & df[treatment_cols].notna().any(axis=1)
    ) if not df.empty else pd.Series(dtype=bool)
    pt_steps = sorted(df.loc[usable_mask, "pt_step"].unique()) if not df.empty else []
    cmap = plt.get_cmap("viridis")
    if len(pt_steps) > 1:
        norm = mpl.colors.LogNorm(vmin=min(pt_steps), vmax=max(pt_steps))
    elif pt_steps:
        norm = mpl.colors.Normalize(vmin=pt_steps[0], vmax=pt_steps[0] + 1)
    else:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

    for row_idx, (ds_key, tname) in enumerate(row_specs):
        cfg = DATASETS[ds_key]
        tlabel = cfg["treatments"][tname]["label"]
        for col_idx, k in enumerate(X_PASS_KS):
            ax = axes[row_idx, col_idx]
            sub = df[df["dataset"] == ds_key].dropna(subset=[
                f"omi_pass_{k}", "base_test_pass_1", "base_test_pass_32",
                f"{tname}_test_pass_1", f"{tname}_test_pass_32",
            ]).copy()

            if not sub.empty:
                sub["_x"] = sub[f"omi_pass_{k}"] * 100
                for series_k in Y_DELTA_KS:
                    sub_d = sub.copy()
                    sub_d["_y"] = (sub_d[f"{tname}_test_pass_{series_k}"] - sub_d[f"base_test_pass_{series_k}"]) * 100
                    sub_d = sub_d.sort_values("_x")
                    colors = cmap(norm(sub_d["pt_step"].values))
                    if series_k == 32:
                        ax.plot(sub_d["_x"], sub_d["_y"], color="gray", linewidth=0.8,
                                linestyle="-", alpha=0.45, zorder=1)
                        ax.scatter(sub_d["_x"], sub_d["_y"], c=colors,
                                   marker=cfg["marker"], s=95,
                                   edgecolors="k", linewidths=0.6, zorder=10)
                    else:
                        ax.plot(sub_d["_x"], sub_d["_y"], color="gray", linewidth=0.8,
                                linestyle="--", alpha=0.45, zorder=1)
                        ax.scatter(sub_d["_x"], sub_d["_y"], facecolors="none",
                                   edgecolors=colors, marker=cfg["marker"], s=95,
                                   linewidths=1.6, zorder=10)

            ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.4)
            if row_idx == 0:
                ax.set_title(f"k = {k}")
            ax.set_xlabel(f"Base Pass@{k} on {cfg['label']} train (omi) (%)")
            ax.grid(True, linestyle=":", color="gray", alpha=0.6)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor("black"); spine.set_linewidth(1.0)

        axes[row_idx, 0].set_ylabel(
            f"{cfg['label']}  {tlabel}\n" + r"$\Delta$Pass on test (%) [trt $-$ Base]"
        )

    # Style legend (shape ↔ dataset, fill/linestyle ↔ k)
    legend_handles = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
               markeredgecolor="k", linestyle="-", markersize=9,
               label=r"GSM8K  $\Delta$Pass@32"),
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markeredgecolor="gray", linestyle="--", markersize=9,
               markeredgewidth=1.5, label=r"GSM8K  $\Delta$Pass@1"),
        Line2D([0], [0], marker="s", color="gray", markerfacecolor="gray",
               markeredgecolor="k", linestyle="-", markersize=9,
               label=r"MATH  $\Delta$Pass@32"),
        Line2D([0], [0], marker="s", color="gray", markerfacecolor="none",
               markeredgecolor="gray", linestyle="--", markersize=9,
               markeredgewidth=1.5, label=r"MATH  $\Delta$Pass@1"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    # Colorbar for pretraining step
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right",
                        pad=0.015, fraction=0.012)
    cbar.set_label("Pretrain step")
    if pt_steps:
        cbar.set_ticks(pt_steps)
        cbar.ax.set_yticklabels([str(s) for s in pt_steps])

    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.94, bottom=0.05, left=0.06, right=0.92)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_table()
    print(f"Built table with {len(df)} rows across {df['pt_step'].nunique() if not df.empty else 0} pt_steps "
          f"× {df['dataset'].nunique() if not df.empty else 0} datasets.")
    if not df.empty:
        print(df.to_string(index=False))
    output_path = Path(__file__).parent / "base_metric_rl_effectiveness.pdf"
    plot(df, output_path)
