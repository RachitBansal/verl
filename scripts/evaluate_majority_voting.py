#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate predictions using majority voting (consensus@k) strategy.
Useful when N_SAMPLES > 1.
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import comb
from tqdm import tqdm


def extract_answer(response: str, dataset_type: str, ground_truth: str) -> float:
    """Extract and evaluate a single response."""
    # Use the default scoring function
    if dataset_type == "gsm8k":
        from verl.utils.reward_score.gsm8k import compute_score
        return compute_score(response, ground_truth, method="flexible")
    elif dataset_type == "math":
        from verl.utils.reward_score.math_reward import compute_score
        return compute_score(response, ground_truth)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k using the analytical unbiased estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: total number of samples
        c: number of correct samples
        k: k value for pass@k

    Returns:
        pass@k score
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(comb(n - c, k, exact=True) / comb(n, k, exact=True))


def majority_vote(responses: list, dataset_type: str, ground_truth: str) -> tuple:
    """
    Evaluate responses using majority voting.

    Args:
        responses: List of response strings
        dataset_type: 'gsm8k' or 'math'
        ground_truth: Ground truth answer

    Returns:
        (majority_vote_correct, any_correct, num_correct)
    """
    scores = []
    for response in responses:
        score = extract_answer(response, dataset_type, ground_truth)
        scores.append(score)

    # Any correct (pass@k)
    any_correct = float(max(scores))

    # Number correct
    num_correct = sum(scores)

    # Majority voting: most common prediction
    # If there's a tie, we consider it incorrect
    if num_correct > len(responses) / 2:
        majority_vote_correct = 1.0
    else:
        majority_vote_correct = 0.0

    return majority_vote_correct, any_correct, num_correct


def main():
    parser = argparse.ArgumentParser(description="Evaluate with majority voting")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Input parquet file with predictions")
    parser.add_argument("--dataset_type", type=str, required=True, 
                        choices=["gsm8k", "math"],
                        help="Dataset type")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.input_file}")
    df = pd.read_parquet(args.input_file)
    
    n_samples = len(df["responses"].iloc[0]) if len(df) > 0 else 0
    print(f"Dataset: {args.dataset_type}")
    print(f"Number of test examples: {len(df)}")
    print(f"Samples per example: {n_samples}")
    print("")
    
    # majority_scores = []
    pass_at_k_scores = []
    num_correct_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        responses = row["responses"]
        ground_truth = row["reward_model"]["ground_truth"]
        
        majority_correct, any_correct, num_correct = majority_vote(
            responses, args.dataset_type, ground_truth
        )
        
        # majority_scores.append(majority_correct)
        pass_at_k_scores.append(any_correct)
        num_correct_list.append(num_correct)
    
    # Compute pass@k for all powers of 2 up to n_samples
    k_values = []
    k = 1
    while k <= n_samples:
        k_values.append(k)
        k *= 2

    # Compute pass@k for each k value using analytical estimator
    pass_at_k_results = {}
    for k in k_values:
        pass_at_k_scores_for_k = []
        for num_correct in num_correct_list:
            pass_at_k_score = compute_pass_at_k(n_samples, int(num_correct), k)
            pass_at_k_scores_for_k.append(pass_at_k_score)
        pass_at_k_results[k] = np.mean(pass_at_k_scores_for_k)

    avg_correct = np.mean(num_correct_list)

    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Number of samples per prompt: {n_samples}")
    print("")
    print("Pass@k Metrics (Analytical Estimator):")
    for k in k_values:
        pass_at_k_val = pass_at_k_results[k]
        print(f"  Pass@{k:<2}: {pass_at_k_val:.4f} ({pass_at_k_val*100:.2f}%)")
    print("")
    print(f"Average # Correct per Example: {avg_correct:.2f} / {n_samples}")
    print("=" * 60)
    print("")
    
    # Additional statistics
    print("Distribution of correct answers per example:")
    correct_counts = Counter(num_correct_list)
    for k in sorted(correct_counts.keys()):
        count = correct_counts[k]
        percentage = count / len(num_correct_list) * 100
        print(f"  {k} correct: {count} examples ({percentage:.1f}%)")


if __name__ == "__main__":
    main()


