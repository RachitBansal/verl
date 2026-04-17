#!/usr/bin/env python3
"""Run lm-evaluation-harness on a HuggingFace checkpoint.

Evaluates a group of tasks defined in a task-config YAML (group/task list with
per-task num_fewshot), so we can swap in different capability suites without
editing Python.

Usage:
    python run_eval.py --model_path /path/to/hf_checkpoint
    python run_eval.py --model_path /path/to/hf_checkpoint --limit 100  # signal screening
    python run_eval.py --model_path /path/to/hf_checkpoint --quick       # limit=5 dry run
    python run_eval.py --model_path /path/to/hf_checkpoint --task_config eval_tasks.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = SCRIPT_DIR / "eval_tasks_nonmath.yaml"


def load_task_config(config_path: Path):
    """Parse a task-group YAML into a list of (task_name, num_fewshot) tuples."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tasks = []
    for entry in cfg.get("task", []):
        name = entry["task"]
        num_fewshot = entry.get("num_fewshot")
        tasks.append((name, num_fewshot))
    if not tasks:
        raise ValueError(f"No tasks found in {config_path}")
    return tasks


def derive_run_name(model_path: str) -> str:
    """Derive a short run name from the model path."""
    parts = Path(model_path).resolve().parts
    try:
        hf_idx = parts.index("hf_model")
        experiment = parts[hf_idx - 1]
        step = parts[hf_idx + 1]
        return f"{experiment}_{step}"
    except (ValueError, IndexError):
        return "_".join(parts[-2:])


def run_lm_eval(model_args, task, num_fewshot, limit, output_path, batch_size):
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--include_path", str(SCRIPT_DIR),
        "--tasks", task,
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print("=" * 60)
    print(f"Task: {task} (num_fewshot={num_fewshot}, limit={limit})")
    print(" \\\n  ".join(cmd))
    print("=" * 60)
    print()

    result = subprocess.run(cmd)
    return result.returncode


def consolidate_results(output_path: Path, run_name: str):
    """Merge all lm_eval result JSONs into a single clean results.json."""
    merged = {"run_name": run_name, "results": {}}

    for json_file in sorted(output_path.rglob("results_*.json")):
        data = json.loads(json_file.read_text())
        if "results" in data:
            merged["results"].update(data["results"])
        for key in ("config", "model_args"):
            if key in data:
                merged[key] = data[key]

    out_file = output_path / "results.json"
    out_file.write_text(json.dumps(merged, indent=2))

    print("=" * 60)
    print(f"Consolidated results -> {out_file}")
    print("=" * 60)
    for task, metrics in merged["results"].items():
        bpb = metrics.get("bits_per_byte,none")
        acc = metrics.get("acc,none")
        em = metrics.get("exact_match,none")
        ppl = metrics.get("perplexity,none") or metrics.get("word_perplexity,none")
        parts = []
        if bpb is not None:
            parts.append(f"bpb={bpb}")
        if acc is not None:
            parts.append(f"acc={acc}")
        if em is not None:
            parts.append(f"exact_match={em}")
        if ppl is not None:
            parts.append(f"ppl={ppl}")
        summary = ", ".join(parts) if parts else "(see json)"
        print(f"  {task}: {summary}")
    print()

    return merged


def main():
    parser = argparse.ArgumentParser(description="Run lm_eval on a HuggingFace checkpoint")
    parser.add_argument("--model_path", required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--task_config", default=str(DEFAULT_TASK_CONFIG),
                        help="Path to task-group YAML (fields: group, task: [{task, num_fewshot}])")
    parser.add_argument("--limit", type=int, default=None,
                        help="Global per-task sample limit (overrides task defaults); e.g. 100 for screening")
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "results"), help="Results output directory")
    parser.add_argument("--batch_size", default="auto", help="Batch size (default: auto)")
    parser.add_argument("--quick", action="store_true", help="Shortcut for --limit 5 (dry run)")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    args = parser.parse_args()

    tasks = load_task_config(Path(args.task_config))

    run_name = derive_run_name(args.model_path)
    output_path = Path(args.output_dir) / run_name
    model_args = f"pretrained={args.model_path},dtype={args.dtype},max_length={args.max_length}"

    limit = 5 if args.quick else args.limit

    for task, num_fewshot in tasks:
        rc = run_lm_eval(model_args, task, num_fewshot, limit, output_path, args.batch_size)
        if rc != 0:
            print(f"ERROR: {task} failed with return code {rc}")
            sys.exit(rc)

    consolidate_results(output_path, run_name)
    print("All tasks completed. Results saved to:", output_path / "results.json")


if __name__ == "__main__":
    main()
