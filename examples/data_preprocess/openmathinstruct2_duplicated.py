# Author: Sunny + Claude
"""
Preprocess the nvidia/OpenMathInstruct-2 dataset to parquet format (with duplicates in train).

Unlike openmathinstruct2.py, this script does NOT deduplicate the training set.
The validation sets use unique questions; the training set keeps all remaining
rows (including duplicate questions with different solutions).

Creates three datasets:
1. train.parquet: All samples for training (no dedup, val questions excluded)
2. val_gsm8k.parquet: 1k unique GSM8K questions for validation
3. val_math.parquet: 1k unique MATH questions for validation

Note: Different data_source values for validation sets enable separate metric reporting in training logs.

Usage:
     python examples/data_preprocess/openmathinstruct2_duplicated.py \
      --n_val_gsm8k 1000 \
      --n_val_math 1000 \
      --seed 42 \
      --local_save_dir /n/netscratch/dam_lab/Lab/sqin/rl_pretrain/data/openmathinstruct2_duplicated \
      --cache_dir /n/netscratch/dam_lab/Lab/sqin/cache/datasets
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/openmathinstruct2_duplicated", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--cache_dir", default=None, help="Cache directory for downloading datasets from HuggingFace."
    )
    parser.add_argument(
        "--n_val_gsm8k", default=1000, type=int, help="Number of unique GSM8K questions for validation."
    )
    parser.add_argument(
        "--n_val_math", default=1000, type=int, help="Number of unique MATH questions for validation."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for sampling."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "nvidia/OpenMathInstruct-2"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    if args.cache_dir:
        print(f"Using cache directory: {args.cache_dir}", flush=True)
        os.makedirs(args.cache_dir, exist_ok=True)

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, cache_dir=args.cache_dir)
    else:
        dataset = datasets.load_dataset(data_source, cache_dir=args.cache_dir)

    # The dataset is typically stored in the 'train' split
    train_dataset = dataset["train"]

    print(f"Original dataset size: {len(train_dataset)}", flush=True)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx, split_name, custom_data_source=None):
        question_raw = example["problem"]
        question = question_raw + " " + instruction_following

        generated_solution = example["generated_solution"]
        expected_answer = example["expected_answer"]
        problem_source = example["problem_source"]

        # Use custom_data_source if provided (for validation sets), otherwise use default
        source = custom_data_source if custom_data_source is not None else data_source

        data = {
            "data_source": source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(expected_answer)},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "problem": question_raw,
                "generated_solution": generated_solution,
                "problem_source": problem_source,
            },
        }
        return data

    # === Build GSM8K validation set from unique questions ===
    print(f"\n=== Creating GSM8K validation set ===", flush=True)
    gsm8k_filter_fn = lambda x: x["problem_source"] in ["gsm8k", "augmented_gsm8k"]
    gsm8k_all = train_dataset.filter(gsm8k_filter_fn)

    gsm8k_count = sum(1 for x in gsm8k_all if x["problem_source"] == "gsm8k")
    augmented_gsm8k_count = sum(1 for x in gsm8k_all if x["problem_source"] == "augmented_gsm8k")

    print(f"GSM8K-related dataset size (with duplicates): {len(gsm8k_all)}", flush=True)
    print(f"  - gsm8k: {gsm8k_count} rows", flush=True)
    print(f"  - augmented_gsm8k: {augmented_gsm8k_count} rows", flush=True)

    # Deduplicate to get unique questions, then shuffle and pick n_val
    seen_problems = set()
    unique_indices = []
    for idx, example in enumerate(gsm8k_all):
        problem = example["problem"]
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_indices.append(idx)

    gsm8k_unique = gsm8k_all.select(unique_indices)
    print(f"Unique GSM8K questions available: {len(gsm8k_unique)}", flush=True)

    gsm8k_unique_shuffled = gsm8k_unique.shuffle(seed=args.seed)
    n_val_gsm8k = min(args.n_val_gsm8k, len(gsm8k_unique_shuffled))

    val_gsm8k = gsm8k_unique_shuffled.select(range(n_val_gsm8k))
    val_gsm8k_problems = set(ex["problem"] for ex in val_gsm8k)

    val_gsm8k = val_gsm8k.map(
        function=lambda ex, idx: process_fn(ex, idx, "val_gsm8k", custom_data_source="OpenMathInstruct-2/gsm8k"),
        with_indices=True
    )
    print(f"val_gsm8k size: {len(val_gsm8k)}", flush=True)

    # === Build MATH validation set from unique questions ===
    print(f"\n=== Creating MATH validation set ===", flush=True)
    math_filter_fn = lambda x: x["problem_source"] in ["math", "augmented_math"]
    math_all = train_dataset.filter(math_filter_fn)

    math_count = sum(1 for x in math_all if x["problem_source"] == "math")
    augmented_math_count = sum(1 for x in math_all if x["problem_source"] == "augmented_math")

    print(f"MATH-related dataset size (with duplicates): {len(math_all)}", flush=True)
    print(f"  - math: {math_count} rows", flush=True)
    print(f"  - augmented_math: {augmented_math_count} rows", flush=True)

    # Deduplicate to get unique questions, then shuffle and pick n_val
    seen_problems = set()
    unique_indices = []
    for idx, example in enumerate(math_all):
        problem = example["problem"]
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_indices.append(idx)

    math_unique = math_all.select(unique_indices)
    print(f"Unique MATH questions available: {len(math_unique)}", flush=True)

    math_unique_shuffled = math_unique.shuffle(seed=args.seed)
    n_val_math = min(args.n_val_math, len(math_unique_shuffled))

    val_math = math_unique_shuffled.select(range(n_val_math))
    val_math_problems = set(ex["problem"] for ex in val_math)

    val_math = val_math.map(
        function=lambda ex, idx: process_fn(ex, idx, "val_math", custom_data_source="OpenMathInstruct-2/math"),
        with_indices=True
    )
    print(f"val_math size: {len(val_math)}", flush=True)

    # === Build training set: all rows whose question is NOT in val ===
    print(f"\n=== Building training set (no dedup, val questions excluded) ===", flush=True)

    val_problems = val_gsm8k_problems | val_math_problems
    print(f"Total validation questions to exclude: {len(val_problems)}", flush=True)

    train_data = train_dataset.filter(lambda x: x["problem"] not in val_problems)
    train_data = train_data.map(
        function=lambda ex, idx: process_fn(ex, idx, "train"),
        with_indices=True
    )
    print(f"train size: {len(train_data)}", flush=True)

    # === Save ===
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)

    print(f"\n=== Saving datasets to {local_dir} ===", flush=True)

    output_file = os.path.join(local_dir, "train.parquet")
    train_data.to_parquet(output_file)
    print(f"✓ Saved train.parquet ({len(train_data)} examples)", flush=True)

    output_file = os.path.join(local_dir, "val_gsm8k.parquet")
    val_gsm8k.to_parquet(output_file)
    print(f"✓ Saved val_gsm8k.parquet ({len(val_gsm8k)} examples)", flush=True)

    output_file = os.path.join(local_dir, "val_math.parquet")
    val_math.to_parquet(output_file)
    print(f"✓ Saved val_math.parquet ({len(val_math)} examples)", flush=True)

    example = train_data[0]
    example_file = os.path.join(local_dir, "train_example.json")
    with open(example_file, "w") as f:
        json.dump(example, f, indent=2)
    print(f"✓ Saved example to train_example.json", flush=True)

    example = val_gsm8k[0]
    example_file = os.path.join(local_dir, "val_gsm8k_example.json")
    with open(example_file, "w") as f:
        json.dump(example, f, indent=2)
    print(f"✓ Saved example to val_gsm8k_example.json", flush=True)

    example = val_math[0]
    example_file = os.path.join(local_dir, "val_math_example.json")
    with open(example_file, "w") as f:
        json.dump(example, f, indent=2)
    print(f"✓ Saved example to val_math_example.json", flush=True)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"Preprocessing complete! All datasets saved to {local_dir}", flush=True)
    print(f"  - train.parquet: {len(train_data)} examples (with duplicates, val excluded)", flush=True)
    print(f"  - val_gsm8k.parquet: {len(val_gsm8k)} examples (unique GSM8K questions)", flush=True)
    print(f"  - val_math.parquet: {len(val_math)} examples (unique MATH questions)", flush=True)
    print(f"{'='*60}", flush=True)
