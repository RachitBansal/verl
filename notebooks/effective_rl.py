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

import numpy as np
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

# Manual overrides for direct-RL test scores: most olmo2_1b_step*_omi_n* /
# olmo2_1b_step*_omigsm8k_n* eval dirs are empty placeholders, so we read
# Pass@1 / Pass@32 from notebooks/manual_rl_{math,gsm}.json. Each maps
# pt_step -> {1: pass@1_frac, 32: pass@32_frac}.
MANUAL_RL_MATH_PATH = Path(__file__).parent / "manual_rl_math.json"
MANUAL_RL_GSM_PATH = Path(__file__).parent / "manual_rl_gsm.json"
# Manual points (sorted by x_billions) correspond to these canonical 1B/50B-stage1 pretrain steps:
MANUAL_RL_MATH_PT_STEPS = [1000, 3000, 5000, 10000, 14000, 22000]
MANUAL_RL_GSM_PT_STEPS = [1000, 2000, 3000, 5000, 6000, 7000, 10000, 14000, 22000]


def _load_manual_rl_test(path: Path, pt_steps: list[int]) -> dict[int, dict[int, float]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[int, dict[int, float]] = {step: {} for step in pt_steps}
    for k_str, points in raw.items():
        if not k_str.isdigit():
            continue
        k = int(k_str)
        if k not in (1, 32):
            continue
        sorted_pts = sorted(points, key=lambda p: p[0])
        if len(sorted_pts) != len(pt_steps):
            continue
        for step, (_x, acc_pct) in zip(pt_steps, sorted_pts):
            out[step][k] = acc_pct / 100.0
    return out


MANUAL_RL_MATH_TEST = _load_manual_rl_test(MANUAL_RL_MATH_PATH, MANUAL_RL_MATH_PT_STEPS)
MANUAL_RL_GSM_TEST = _load_manual_rl_test(MANUAL_RL_GSM_PATH, MANUAL_RL_GSM_PT_STEPS)

# Manual direct-RL Pass@1 measured on the omi-math train distribution.
# Used by `plot_omi_rl_effectiveness_math` (x = base omi-math Pass@32, y =
# RL omi-math Pass@1 − base omi-math Pass@1).
MANUAL_RL_OMI_MATH_PATH = Path(__file__).parent / "manual_rl_omi_math.json"
_RL_OMI_RAW_TO_DS = {"math_1b": "math", "math_60b": "math_60b"}


def _load_manual_rl_omi(path: Path) -> dict[str, dict[int, float]]:
    """Returns {dataset_key: {pt_step: rl_omi_pass1_frac}}."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict[int, float]] = {}
    for raw_key, by_step in raw.items():
        if raw_key.startswith("_") or not isinstance(by_step, dict):
            continue
        ds_key = _RL_OMI_RAW_TO_DS.get(raw_key)
        if ds_key is None:
            continue
        cleaned: dict[int, float] = {}
        for k, v in by_step.items():
            if k.startswith("_") or not str(k).isdigit():
                continue
            try:
                cleaned[int(k)] = float(v) / 100.0
            except (TypeError, ValueError):
                continue
        if cleaned:
            out[ds_key] = cleaned
    return out


MANUAL_RL_OMI = _load_manual_rl_omi(MANUAL_RL_OMI_MATH_PATH)


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
        "color": "#2166AC",   # colorbrewer blue
        "marker": "o",
        "omi_parquet": "omi_gsm_predictions.parquet",
        "omi_cache": "omi_gsm_scores_cache.json",
        "omi_scorer": _gsm_score_one,
        "test_majority": "gsm8k_majority_results.txt",
        "base_pattern": re.compile(
            r"1B-(?:stage1-50B-)?step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
        ),
        "base_test_templates": [
            "1B-stage1-50B-step{step}-{shot}shot-32samples-temp{temp}",
            "1B-step{step}-{shot}shot-32samples-temp{temp}",
        ],
        "base_omi_template": "1B-stage1-50B-step{step}-{shot}shot-32samples-temp{temp}",
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
        "color": "#D6604D",   # colorbrewer soft red
        "marker": "s",
        "omi_parquet": "omi_math_predictions.parquet",
        "omi_cache": "omi_math_scores_cache.json",
        "omi_scorer": _math_score_one,
        "test_majority": "math_majority_results.txt",
        "base_pattern": re.compile(
            r"1B-(?:stage1-50B-)?step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
        ),
        "base_test_templates": [
            "1B-stage1-50B-step{step}-{shot}shot-32samples-temp{temp}",
            "1B-step{step}-{shot}shot-32samples-temp{temp}",
        ],
        "base_omi_template": "1B-stage1-50B-step{step}-{shot}shot-32samples-temp{temp}",
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
    "math_60b": {
        "label": "MATH (1B-MATH60B)",
        "color": "#762A83",   # purple
        "marker": "D",
        "omi_parquet": "omi_math_predictions.parquet",
        "omi_cache": "omi_math_scores_cache.json",
        "omi_scorer": _math_score_one,
        "test_majority": "math_majority_results.txt",
        # Base test files live under 1B-MATH60B-step*; omi_math parquet lives under 1B-stage1-60B-step*
        "base_pattern": re.compile(
            r"1B-MATH60B-step(?P<step>\d+)-(?P<shot>\d+)shot-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"
        ),
        "base_test_templates": [
            "1B-MATH60B-step{step}-{shot}shot-32samples-temp{temp}",
        ],
        "base_omi_template": "1B-stage1-60B-step{step}-{shot}shot-32samples-temp{temp}",
        "treatments": {
            "rl": {
                "label": "RL",
                "rollouts": 32,
                "patterns": [
                    re.compile(r"olmo2_1b_60bmath_step(?P<pt_step>\d+)_omi_n(?P<num_rollouts>\d+)-step(?P<rl_step>\d+)-rl-0shot-boxed-(?P<samples>\d+)samples-temp(?P<temp>[\d.]+)$"),
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

def find_first(name: str) -> Path | None:
    for base in BASE_DIRS:
        p = base / name
        if p.exists():
            return p
    return None


def collect_base_pretrain_steps(pre_pattern: re.Pattern) -> list[int]:
    steps = set()
    for base in BASE_DIRS:
        if not base.exists():
            continue
        for path in base.iterdir():
            if not path.is_dir():
                continue
            m = pre_pattern.match(path.name)
            if not m:
                continue
            if int(m.group("samples")) != 32 or int(m.group("shot")) != TARGET_SHOT:
                continue
            if float(m.group("temp")) != TARGET_TEMP:
                continue
            steps.add(int(m.group("step")))
    return sorted(steps)


def _format_template(template: str, step: int) -> str:
    return template.format(step=step, shot=TARGET_SHOT, temp=TARGET_TEMP)


def base_test_majority_path(step: int, filename: str, templates: list[str]) -> Path | None:
    for tpl in templates:
        p = find_first(_format_template(tpl, step))
        if p is None:
            continue
        candidate = p / filename
        if candidate.exists():
            return candidate
    return None


def base_omi_parquet(step: int, filename: str, template: str) -> Path | None:
    p = find_first(_format_template(template, step))
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
    for step in collect_base_pretrain_steps(cfg["base_pattern"]):
        # Base — omi parquet (RL train set). This is the only required input.
        omi_path = base_omi_parquet(step, cfg["omi_parquet"], cfg["base_omi_template"])
        if omi_path is None or not omi_path.exists():
            print(f"  [{dataset_key} skip step={step}] no {cfg['omi_parquet']}")
            continue
        omi_scores = score_parquet(omi_path, cfg["omi_cache"], cfg["omi_scorer"])
        if not omi_scores:
            print(f"  [{dataset_key} skip step={step}] omi scoring failed")
            continue

        # Base — test set (majority); optional
        base_test = {}
        base_test_path = base_test_majority_path(step, cfg["test_majority"], cfg["base_test_templates"])
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

        manual_rl_omi = MANUAL_RL_OMI.get(dataset_key, {})
        if step in manual_rl_omi:
            row["rl_omi_pass_1"] = manual_rl_omi[step]

        for tname, runs in treatment_runs.items():
            best = best_treatment_metrics(runs.get(step, []), cfg["test_majority"])
            row[f"{tname}_test_pass_1"] = best["pass"].get(1) if best else None
            row[f"{tname}_test_pass_32"] = best["pass"].get(32) if best else None
            row[f"{tname}_rl_step"] = best.get("_rl_step") if best else None
            if tname == "rl":
                manual_table = (
                    MANUAL_RL_MATH_TEST if dataset_key == "math"
                    else MANUAL_RL_GSM_TEST if dataset_key == "gsm"
                    else None
                )
                if manual_table is not None:
                    manual = manual_table.get(step)
                    if manual and 1 in manual and 32 in manual:
                        row[f"{tname}_test_pass_1"] = manual[1]
                        row[f"{tname}_test_pass_32"] = manual[32]
        rows.append(row)
    return rows


def build_table() -> pd.DataFrame:
    all_rows = []
    for key, cfg in DATASETS.items():
        all_rows.extend(build_dataset_rows(key, cfg))
    return pd.DataFrame(all_rows)


# ─── 4B math (custom) ────────────────────────────────────────────────────────
# 4B base eval dirs lack omi_math_predictions.parquet, so we can't use
# build_dataset_rows. Per user instruction:
#   step5000 RL  -> new rmath ckpt (sft_0_ppo_50000_rmath)
#   step14000 RL -> old olmo2_4b ckpt (omi)
MATH_4B_CFG = {
    "label": "MATH (4B / 50B-stage1)",
    "color": "#1A9641",   # green star (matches base_metric_rl_comparison)
    "marker": "*",
}

_MATH_4B_RL_PATTERNS = {
    5000: re.compile(
        r"OLMo2-4B_step5000_interleave_twoloader_n32_sft_0_ppo_50000_rmath"
        r"-step(?P<rl_step>\d+)-rl-0shot-boxed-32samples-temp0\.6$"
    ),
    14000: re.compile(
        r"olmo2_4b_step14000_omi_n\d+"
        r"-step(?P<rl_step>\d+)-rl-0shot-boxed-32samples-temp0\.6$"
    ),
}


def build_math_4b_rows() -> list[dict]:
    base_template = "4B-stage1-50B-step{step}-8shot-32samples-temp0.6"
    rows = []
    for pt_step, rl_re in _MATH_4B_RL_PATTERNS.items():
        base_dir = find_first(base_template.format(step=pt_step))
        if base_dir is None:
            print(f"  [math_4b skip step={pt_step}] no base eval dir")
            continue
        base_test = read_majority_metrics(base_dir / "math_majority_results.txt")
        if not base_test.get("pass"):
            print(f"  [math_4b skip step={pt_step}] no base test majority")
            continue

        candidates: list[tuple[int, Path]] = []
        for base in BASE_DIRS:
            if not base.exists():
                continue
            for path in base.iterdir():
                if not path.is_dir():
                    continue
                m = rl_re.match(path.name)
                if m:
                    candidates.append((int(m.group("rl_step")), path))
        if not candidates:
            print(f"  [math_4b skip step={pt_step}] no RL eval dir")
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = next((p for _s, p in candidates if (p / "math_majority_results.txt").exists()), None)
        if chosen is None:
            print(f"  [math_4b skip step={pt_step}] no RL test majority")
            continue
        rl_metrics = read_majority_metrics(chosen / "math_majority_results.txt")

        rows.append({
            "dataset": "math_4b",
            "pt_step": pt_step,
            "base_test_pass_1": base_test.get("pass", {}).get(1),
            "base_test_pass_32": base_test.get("pass", {}).get(32),
            "rl_test_pass_1": rl_metrics.get("pass", {}).get(1),
            "rl_test_pass_32": rl_metrics.get("pass", {}).get(32),
        })
    return rows


def _col(metric_kind: str, k: int, kind: str) -> str:
    """kind in {'omi','base_test','rl_test'}; metric_kind in {'pass','mean','cond_mean'}."""
    suffix = {"pass": "pass", "mean": "mean", "cond_mean": "cond_mean"}[metric_kind]
    if kind == "omi":
        return f"omi_{suffix}_{k}"
    # cond_mean only available on omi (parquet); fall back gracefully
    return f"{kind}_{suffix}_{k}"


# ─── Plot ────────────────────────────────────────────────────────────────────
def plot_first_row(df: pd.DataFrame, output_path: Path, relative: bool = False):
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

    PANEL_KS = [1, 32]
    # Each panel is one (group_label, [dataset_keys]) — math panel overlays both
    # the 1B/50B-stage1 series (red) and the 1B/60BMATH series (purple).
    PANEL_GROUPS = [
        ("GSM8K", ["gsm"]),
        ("MATH",  ["math", "math_60b"]),
    ]
    panel_specs = [(label, ds_keys, k) for label, ds_keys in PANEL_GROUPS for k in PANEL_KS]

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(3.7 * len(panel_specs), 3.6))
    axes = axes.ravel()

    for col_idx, (panel_label, ds_keys, k) in enumerate(panel_specs):
        ax = axes[col_idx]
        for ds_key in ds_keys:
            cfg = DATASETS[ds_key]
            sub = df[df["dataset"] == ds_key].dropna(subset=[
                f"omi_pass_{k}", "base_test_pass_1", "base_test_pass_32",
                f"rl_test_pass_1", f"rl_test_pass_32",
            ]).copy()
            if sub.empty:
                continue
            sub["_x"] = sub[f"omi_pass_{k}"] * 100
            sub["_y"] = sub[f"rl_test_pass_{k}"] * 100
            sub = sub.sort_values("_x")
            ax.scatter(sub["_x"], sub["_y"], color=cfg["color"],
                       marker=cfg["marker"], s=95,
                       edgecolors="k", linewidths=0.6, zorder=10)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lo = min(x0, y0)
        hi = max(x1, y1)
        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=0.9,
                linestyle="--", alpha=0.6, zorder=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        ax.set_title(f"{panel_label}  (k = {k})")
        ax.set_xlabel(f"Base Pass@{k} on Train (%)")
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor("black"); spine.set_linewidth(1.0)

    axes[0].set_ylabel("Direct RL Pass@k on test (%)")

    legend_handles = [
        Line2D([0], [0], marker=DATASETS["gsm"]["marker"],
               color=DATASETS["gsm"]["color"], markerfacecolor=DATASETS["gsm"]["color"],
               markeredgecolor="#222222", markeredgewidth=0.6, markersize=9,
               linestyle="None", label="GSM8K"),
        Line2D([0], [0], marker=DATASETS["math"]["marker"],
               color=DATASETS["math"]["color"], markerfacecolor=DATASETS["math"]["color"],
               markeredgecolor="#222222", markeredgewidth=0.6, markersize=9,
               linestyle="None", label="MATH (1B / 50B-stage1)"),
        Line2D([0], [0], marker=DATASETS["math_60b"]["marker"],
               color=DATASETS["math_60b"]["color"], markerfacecolor=DATASETS["math_60b"]["color"],
               markeredgecolor="#222222", markeredgewidth=0.6, markersize=9,
               linestyle="None", label="MATH (1B / 60BMATH)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=True,
               framealpha=0.95, edgecolor="#cccccc",
               fontsize=11, handletextpad=0.5, columnspacing=1.2)

    fig.subplots_adjust(wspace=0.3, top=0.92, bottom=0.22, left=0.08, right=0.97)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved to {output_path.with_suffix('.png')}")
    plt.close()


def plot_first_row_xtest(df: pd.DataFrame, output_path: Path):
    """Combined version: per-panel x is base 8-shot Pass@k on that panel's own test
    set. So GSM panels use base GSM Pass@k as x; MATH panels use base MATH Pass@k
    as x. y is direct-RL Pass on test. A y=x dashed line is drawn as a reference
    in each panel without changing the x/y ranges.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "legend.fontsize": 18,
    })

    PANEL_KS = [1, 32]
    PANEL_GROUPS = [
        ("GSM8K", "GSM8K", ["gsm"]),
        ("MATH",  "MATH",  ["math", "math_60b", "math_4b"]),
    ]
    panel_specs = [(label, x_test_label, ds_keys, k)
                   for label, x_test_label, ds_keys in PANEL_GROUPS
                   for k in PANEL_KS]

    # Star marker renders smaller than circles/squares/diamonds at the same
    # `s`, so bump it up to keep visual sizes consistent.
    marker_scatter_size = {"*": 220}
    marker_legend_size = {"*": 16}

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(4.4 * len(panel_specs), 4.2))
    axes = axes.ravel()

    for col_idx, (panel_label, x_test_label, ds_keys, k) in enumerate(panel_specs):
        ax = axes[col_idx]
        for ds_key in ds_keys:
            cfg = MATH_4B_CFG if ds_key == "math_4b" else DATASETS[ds_key]
            sub = df[df["dataset"] == ds_key].dropna(subset=[
                f"base_test_pass_{k}", "rl_test_pass_1", "rl_test_pass_32",
            ]).copy()
            if sub.empty:
                continue
            sub["_x"] = sub[f"base_test_pass_{k}"] * 100
            sub["_y"] = sub[f"rl_test_pass_{k}"] * 100
            sub = sub.sort_values("_x")
            ax.scatter(sub["_x"], sub["_y"], color=cfg["color"],
                       marker=cfg["marker"],
                       s=marker_scatter_size.get(cfg["marker"], 95),
                       edgecolors="k", linewidths=0.6, zorder=10)

        # y=x reference line over the data extent — drawn after autoscale, then
        # restore the ranges so the line doesn't zoom out the axes.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # MATH (k=1) panel: base Pass@1 is tiny (~0–3%) but RL Pass@1 spans 0–30%,
        # so the y=x reference is squashed into the bottom corner. Stretch x so
        # the line shows a meaningful diagonal.
        if panel_label == "MATH" and k == 1:
            x1 = max(x1, 5)
            ax.set_xlim(x0, x1)
        lo = min(x0, y0)
        hi = max(x1, y1)
        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=2.5,
                linestyle="--", alpha=0.35, zorder=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        ax.set_title(f"{panel_label}  (k = {k})")
        ax.set_xlabel(f"Base 8-shot Pass@{k}")
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor("black"); spine.set_linewidth(1.0)

    axes[0].set_ylabel("Direct RL Pass@k (%)")

    legend_handles = [
        Line2D([0], [0], marker=DATASETS["gsm"]["marker"],
               color=DATASETS["gsm"]["color"], markerfacecolor=DATASETS["gsm"]["color"],
               markeredgecolor="#222222", markeredgewidth=0.6, markersize=11,
               linestyle="None", label="GSM8K"),
        Line2D([0], [0], marker=DATASETS["math"]["marker"],
               color=DATASETS["math"]["color"], markerfacecolor=DATASETS["math"]["color"],
               markeredgecolor="#222222", markeredgewidth=0.6, markersize=11,
               linestyle="None", label="MATH (N=1B, D=50B)"),
        Line2D([0], [0], marker=DATASETS["math_60b"]["marker"],
               color=DATASETS["math_60b"]["color"], markerfacecolor=DATASETS["math_60b"]["color"],
               markeredgecolor="#222222", markeredgewidth=0.6, markersize=11,
               linestyle="None", label="MATH (N=1B, D=60B)"),
        Line2D([0], [0], marker=MATH_4B_CFG["marker"],
               color=MATH_4B_CFG["color"], markerfacecolor=MATH_4B_CFG["color"],
               markeredgecolor="#222222", markeredgewidth=0.6,
               markersize=marker_legend_size.get(MATH_4B_CFG["marker"], 11),
               linestyle="None", label="MATH (N=4B, D=50B)"),
        Line2D([0], [0], color="gray", linewidth=2.5, linestyle="--",
               alpha=0.35, label="No improvement"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.08), frameon=True,
               framealpha=0.95, edgecolor="#cccccc",
               fontsize=18, handletextpad=0.5, columnspacing=1.2)

    fig.subplots_adjust(wspace=0.3, top=0.92, bottom=0.22, left=0.08, right=0.97)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved to {output_path.with_suffix('.png')}")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_table()
    math_4b_rows = build_math_4b_rows()
    if math_4b_rows:
        df = pd.concat([df, pd.DataFrame(math_4b_rows)], ignore_index=True)
    print(f"Built table with {len(df)} rows across {df['pt_step'].nunique() if not df.empty else 0} pt_steps "
          f"× {df['dataset'].nunique() if not df.empty else 0} datasets.")
    if not df.empty:
        print(df.to_string(index=False))
    plot_first_row(df, Path(__file__).parent / "base_metric_rl_effectiveness_row1.pdf", relative=False)
    plot_first_row_xtest(df, Path(__file__).parent / "base_metric_rl_effectiveness_row1_xtest.pdf")