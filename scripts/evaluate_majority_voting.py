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
from tqdm import tqdm

from verl.utils.reward_score import default_compute_score


def extract_answer(response: str, dataset_type: str, ground_truth: str) -> float:
    """Extract and evaluate a single response."""
    # Use the default scoring function
    if dataset_type == "gsm8k":
        data_source = "openai/gsm8k"
    elif dataset_type == "math":
        data_source = "lighteval/MATH"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    score = default_compute_score(data_source, response, ground_truth)
    return score


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
    if not responses:
        return 0.0, 0.0, 0
    
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
    
    majority_scores = []
    pass_at_k_scores = []
    num_correct_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        responses = row["responses"]
        ground_truth = row["reward_model"]["ground_truth"]
        
        majority_correct, any_correct, num_correct = majority_vote(
            responses, args.dataset_type, ground_truth
        )
        
        majority_scores.append(majority_correct)
        pass_at_k_scores.append(any_correct)
        num_correct_list.append(num_correct)
    
    # Compute metrics
    majority_vote_acc = np.mean(majority_scores)
    pass_at_k = np.mean(pass_at_k_scores)
    avg_correct = np.mean(num_correct_list)
    
    # Also compute single sample accuracy (first response)
    single_sample_scores = []
    for idx, row in df.iterrows():
        responses = row["responses"]
        ground_truth = row["reward_model"]["ground_truth"]
        score = extract_answer(responses[0], args.dataset_type, ground_truth)
        single_sample_scores.append(score)
    
    single_sample_acc = np.mean(single_sample_scores)
    
    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Number of samples per prompt: {n_samples}")
    print("")
    print(f"Single Sample Accuracy:           {single_sample_acc:.4f} ({single_sample_acc*100:.2f}%)")
    print(f"Majority Voting Accuracy:         {majority_vote_acc:.4f} ({majority_vote_acc*100:.2f}%)")
    print(f"Pass@{n_samples} (Any Correct):            {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
    print(f"Average # Correct per Example:    {avg_correct:.2f} / {n_samples}")
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


