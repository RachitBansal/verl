# Author: Sunny + Claude
"""
Preprocess the nvidia/OpenMathInstruct-2 dataset for GSM8K subset (with duplicates in train).

Unlike openmathinstruct2_gsm8k.py, this script does NOT deduplicate the training set.
The validation set uses 1000 unique questions; the training set keeps all remaining
rows (including duplicate questions with different solutions).

Creates two datasets:
1. train_gsm8k.parquet: GSM8K questions for training (no dedup, val questions excluded)
2. val_gsm8k.parquet: 1k unique GSM8K questions for validation

Usage:
     python examples/data_preprocess/openmathinstruct2_gsm8k_duplicated.py \
      --n_val 1000 \
      --seed 42 \
      --local_save_dir /n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_gsm8k_duplicated \
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
        "--n_val", default=1000, type=int, help="Number of unique GSM8K questions for validation."
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

    def process_fn(example, idx, split_name):
        question_raw = example["problem"]
        question = question_raw + " " + instruction_following

        generated_solution = example["generated_solution"]
        expected_answer = example["expected_answer"]
        problem_source = example["problem_source"]

        data = {
            "data_source": data_source,
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

    # === Filter for GSM8K-related questions ===
    print(f"\n=== Filtering for GSM8K-related questions ===", flush=True)
    gsm8k_filter_fn = lambda x: x["problem_source"] in ["gsm8k", "augmented_gsm8k"]
    gsm8k_all = train_dataset.filter(gsm8k_filter_fn)

    gsm8k_count = sum(1 for x in gsm8k_all if x["problem_source"] == "gsm8k")
    augmented_gsm8k_count = sum(1 for x in gsm8k_all if x["problem_source"] == "augmented_gsm8k")

    print(f"GSM8K-related dataset size (with duplicates): {len(gsm8k_all)}", flush=True)
    print(f"  - gsm8k: {gsm8k_count} rows", flush=True)
    print(f"  - augmented_gsm8k: {augmented_gsm8k_count} rows", flush=True)

    # === Build validation set from unique questions ===
    print(f"\n=== Building validation set from unique questions ===", flush=True)

    # Deduplicate to get unique questions, then shuffle and pick n_val
    seen_problems = set()
    unique_indices = []
    for idx, example in enumerate(gsm8k_all):
        problem = example["problem"]
        if problem not in seen_problems:
            seen_problems.add(problem)
            unique_indices.append(idx)

    gsm8k_unique = gsm8k_all.select(unique_indices)
    print(f"Unique questions available: {len(gsm8k_unique)}", flush=True)

    gsm8k_unique_shuffled = gsm8k_unique.shuffle(seed=args.seed)
    n_val = min(args.n_val, len(gsm8k_unique_shuffled))

    val_gsm8k = gsm8k_unique_shuffled.select(range(n_val))
    val_problems = set(ex["problem"] for ex in val_gsm8k)

    val_gsm8k = val_gsm8k.map(
        function=lambda ex, idx: process_fn(ex, idx, "val_gsm8k"),
        with_indices=True
    )
    print(f"val_gsm8k size: {len(val_gsm8k)}", flush=True)

    # === Build training set: all rows whose question is NOT in val ===
    print(f"\n=== Building training set (no dedup, val questions excluded) ===", flush=True)

    train_gsm8k = gsm8k_all.filter(lambda x: x["problem"] not in val_problems)
    train_gsm8k = train_gsm8k.map(
        function=lambda ex, idx: process_fn(ex, idx, "train_gsm8k"),
        with_indices=True
    )
    print(f"train_gsm8k size: {len(train_gsm8k)}", flush=True)

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

    output_file = os.path.join(local_dir, "train_gsm8k.parquet")
    train_gsm8k.to_parquet(output_file)
    print(f"✓ Saved train_gsm8k.parquet ({len(train_gsm8k)} examples)", flush=True)

    output_file = os.path.join(local_dir, "val_gsm8k.parquet")
    val_gsm8k.to_parquet(output_file)
    print(f"✓ Saved val_gsm8k.parquet ({len(val_gsm8k)} examples)", flush=True)

    example = val_gsm8k[0]
    example_file = os.path.join(local_dir, "val_gsm8k_example.json")
    with open(example_file, "w") as f:
        json.dump(example, f, indent=2)
    print(f"✓ Saved example to val_gsm8k_example.json", flush=True)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"Preprocessing complete! All datasets saved to {local_dir}", flush=True)
    print(f"  - train_gsm8k.parquet: {len(train_gsm8k)} examples (with duplicates, val excluded)", flush=True)
    print(f"  - val_gsm8k.parquet: {len(val_gsm8k)} examples (unique questions)", flush=True)
    print(f"{'='*60}", flush=True)
