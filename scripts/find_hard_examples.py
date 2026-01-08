#!/usr/bin/env python3
"""
Find Hard Examples from OpenMathInstruct2 Dataset

This script identifies "hard" examples where the model has low accuracy (around 1/64).
These examples are valuable for adaptive rollout experiments where we want to focus
compute on samples where the model struggles.

Usage:
    python scripts/find_hard_examples.py \
        --model_path /path/to/model \
        --output_dir /path/to/output \
        --n_samples 128 \
        --target_accuracy 0.015625 \
        --tolerance 0.02 \
        --n_hard_examples 1000

Author: Sunny + Claude
"""

# IMPORTANT: Set environment variables for vLLM BEFORE any imports
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import sys
from pathlib import Path
import numpy as np

import pandas as pd
import torch
from tqdm import tqdm

# Add verl to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Find hard examples from OpenMathInstruct2")
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="allenai/OLMo-2-0425-1B",  # HuggingFace model ID
        help="Path to the model checkpoint (local path or HuggingFace model ID)"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2/train.parquet",
        help="Path to input dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_hard",
        help="Directory to save hard examples"
    )
    
    # Generation configuration
    parser.add_argument("--n_samples", type=int, default=128, help="Number of generations per example")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt length")
    parser.add_argument("--max_response_length", type=int, default=2048, help="Max response length")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization")
    
    # Hard example selection
    parser.add_argument(
        "--target_accuracy",
        type=float,
        default=1/64,  # ~0.015625
        help="Target accuracy for hard examples (default: 1/64)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Tolerance around target accuracy (e.g., 0.03 means 1/64 ± 3%)"
    )
    parser.add_argument(
        "--n_hard_examples",
        type=int,
        default=1000,
        help="Number of hard examples to collect"
    )
    parser.add_argument(
        "--max_examples_to_scan",
        type=int,
        default=50000,
        help="Maximum examples to scan before giving up"
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_all_scores", action="store_true", help="Save all accuracy scores for analysis")
    
    return parser.parse_args()


def load_reward_function():
    """Load the reward/scoring function for math problems."""
    # Import the scoring functions from verl
    from verl.utils.reward_score.openmathinstruct import compute_score as omi_compute_score
    from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
    from verl.utils.reward_score.math_reward import compute_score as math_compute_score
    
    def compute_score(response: str, ground_truth: str, problem_source: str = "openmath") -> float:
        """
        Compute score for a response.
        
        Uses different scoring functions based on problem_source:
        - augmented_gsm8k, gsm8k -> GSM8K scoring (numerical answer extraction)
        - augmented_math, math -> MATH scoring (boxed answer extraction with equivalence)
        - default -> OpenMathInstruct scoring (boxed format with MATH-style checking)
        """
        try:
            problem_source_lower = problem_source.lower()
            
            if "gsm" in problem_source_lower:
                # GSM8K-style problems
                return gsm8k_compute_score(response, ground_truth)
            elif "math" in problem_source_lower and "openmath" not in problem_source_lower:
                # MATH-style problems  
                return math_compute_score(response, ground_truth)
            else:
                # Default to OpenMathInstruct scoring (handles both)
                return omi_compute_score(response, ground_truth)
        except Exception as e:
            print(f"Warning: Error computing score for '{problem_source}': {e}")
            return 0.0
    
    return compute_score


def setup_vllm_engine(model_path: str, args):
    """Setup vLLM inference engine."""
    from vllm import LLM, SamplingParams
    
    print(f"Loading model from: {model_path}")
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_response_length,
        dtype="bfloat16",
        enforce_eager=True,
    )
    
    # Use n parameter to generate multiple samples efficiently
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=20,
        max_tokens=args.max_response_length,
        n=args.n_samples,  # Generate all samples at once
    )
    
    return llm, sampling_params


def generate_and_score_single(
    llm,
    sampling_params,
    prompt: str,
    ground_truth: str,
    problem_source: str,
    compute_score_fn,
) -> dict:
    """
    Generate n_samples responses for a single prompt and compute accuracy.
    
    Returns dict with:
        - n_correct: int
        - n_total: int
        - accuracy: float
    """
    # Generate all samples at once (using n parameter in sampling_params)
    outputs = llm.generate([prompt], sampling_params)

    if np.random.random() < 0.10:
        print(f"Sample examle:")
        print(f"Prompt: {prompt}")
        print(f"Ground truth: {ground_truth}")
        print(f"Problem source: {problem_source}")
        print(f"Outputs: {outputs[0]}")
    
    # Score all responses
    n_correct = 0
    n_total = 0
    
    for output in outputs:
        for completion in output.outputs:
            response = completion.text
            score = compute_score_fn(response, ground_truth, problem_source)
            if score > 0.5:  # Threshold for correct
                n_correct += 1
            n_total += 1
    
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
    }


def main():
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Finding Hard Examples from OpenMathInstruct2")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Target accuracy: {args.target_accuracy:.4f} (1/{int(1/args.target_accuracy)})")
    print(f"Tolerance: ±{args.tolerance:.4f}")
    print(f"Target hard examples: {args.n_hard_examples}")
    print(f"Samples per example: {args.n_samples}")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} examples")
    
    # Shuffle dataset for random sampling
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Extract prompts and ground truths
    # The prompt column contains list of chat messages: [{'content': ..., 'role': 'user'}]
    prompts = []
    if 'prompt' in df.columns:
        for prompt_data in df['prompt']:
            if isinstance(prompt_data, list):
                # Chat format: extract content from messages
                prompt_text = ""
                for msg in prompt_data:
                    if isinstance(msg, dict) and 'content' in msg:
                        prompt_text += msg['content']
                prompts.append(prompt_text)
            elif isinstance(prompt_data, str):
                prompts.append(prompt_data)
            else:
                prompts.append(str(prompt_data))
    elif 'problem' in df.columns:
        # Fallback to problem column
        prompts = df['problem'].tolist()
    else:
        raise ValueError(f"Cannot find prompt column. Available: {df.columns.tolist()}")
    
    # Get ground truth from reward_model column (it's a dict with 'ground_truth' key)
    ground_truths = []
    if 'reward_model' in df.columns:
        for rm in df['reward_model']:
            if isinstance(rm, dict):
                ground_truths.append(rm.get('ground_truth', ''))
            else:
                ground_truths.append(str(rm))
    elif 'expected_answer' in df.columns:
        ground_truths = df['expected_answer'].tolist()
    elif 'ground_truth' in df.columns:
        ground_truths = df['ground_truth'].tolist()
    else:
        raise ValueError(f"Cannot find ground_truth column. Available: {df.columns.tolist()}")
    
    # Get data source / problem source for scoring
    if 'problem_source' in df.columns:
        data_sources = df['problem_source'].tolist()
    elif 'data_source' in df.columns:
        data_sources = df['data_source'].tolist()
    else:
        data_sources = ['openmath'] * len(prompts)
    
    print(f"Sample prompt: {prompts[0][:300]}...")
    print(f"Sample ground_truth: {ground_truths[0]}")
    print(f"Sample data_source: {data_sources[0]}")
    
    # Load scoring function
    print("\nLoading scoring function...")
    compute_score_fn = load_reward_function()
    
    # Setup vLLM
    print("\nSetting up vLLM engine...")
    llm, sampling_params = setup_vllm_engine(args.model_path, args)
    
    # Define accuracy bounds
    acc_lower = args.target_accuracy - args.tolerance
    acc_upper = args.target_accuracy + args.tolerance
    print(f"\nAccepting examples with accuracy in [{acc_lower:.4f}, {acc_upper:.4f}]")
    
    # Collect hard examples
    hard_examples = []
    all_scores = []  # For analysis
    
    n_scanned = 0
    max_to_scan = min(len(prompts), args.max_examples_to_scan)
    pbar = tqdm(total=args.n_hard_examples, desc="Finding hard examples")
    
    for idx in range(max_to_scan):
        if len(hard_examples) >= args.n_hard_examples:
            break
        
        prompt = prompts[idx]
        gt = ground_truths[idx]
        ps = data_sources[idx]
        
        # Generate and score
        result = generate_and_score_single(
            llm=llm,
            sampling_params=sampling_params,
            prompt=prompt,
            ground_truth=gt,
            problem_source=ps,
            compute_score_fn=compute_score_fn,
        )
        
        n_scanned += 1
        acc = result['accuracy']
        all_scores.append({
            'idx': idx,
            'accuracy': acc,
            'n_correct': result['n_correct'],
            'problem_source': ps,
        })
        
        # Check if this is a hard example
        if acc_lower <= acc <= acc_upper:
            # Add original dataframe row info
            example = df.iloc[idx].to_dict()
            example['measured_accuracy'] = acc
            example['n_correct'] = result['n_correct']
            example['n_total'] = result['n_total']
            
            hard_examples.append(example)
            pbar.update(1)
            
            # Log progress
            if len(hard_examples) % 50 == 0:
                tqdm.write(f"Found {len(hard_examples)}/{args.n_hard_examples} hard examples "
                          f"(scanned {n_scanned})")
        
        # Progress logging
        if n_scanned % 100 == 0:
            n_hard = len(hard_examples)
            hit_rate = n_hard / n_scanned if n_scanned > 0 else 0
            tqdm.write(f"Scanned {n_scanned} examples, found {n_hard} hard examples "
                      f"(hit rate: {hit_rate:.2%})")
    
    pbar.close()
    
    print(f"\n{'=' * 60}")
    print(f"Results Summary")
    print(f"{'=' * 60}")
    print(f"Total scanned: {n_scanned}")
    print(f"Hard examples found: {len(hard_examples)}")
    print(f"Hit rate: {len(hard_examples) / n_scanned:.2%}" if n_scanned > 0 else "N/A")
    
    if len(hard_examples) < args.n_hard_examples:
        print(f"\nWARNING: Only found {len(hard_examples)} hard examples "
              f"(target was {args.n_hard_examples})")
        print("Consider:")
        print("  - Increasing --tolerance")
        print("  - Increasing --max_examples_to_scan")
        print("  - Adjusting --target_accuracy")
    
    # Save hard examples
    if hard_examples:
        hard_df = pd.DataFrame(hard_examples)
        
        # Save as parquet
        output_path = os.path.join(args.output_dir, "hard_examples.parquet")
        hard_df.to_parquet(output_path)
        print(f"\nSaved {len(hard_examples)} hard examples to: {output_path}")
        
        # Also save train/val split
        n_train = int(len(hard_df) * 0.9)
        train_df = hard_df.iloc[:n_train]
        val_df = hard_df.iloc[n_train:]
        
        train_path = os.path.join(args.output_dir, "train.parquet")
        val_path = os.path.join(args.output_dir, "val.parquet")
        
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)
        
        print(f"Saved train split ({len(train_df)} examples): {train_path}")
        print(f"Saved val split ({len(val_df)} examples): {val_path}")
        
        # Print statistics
        print(f"\nAccuracy statistics of hard examples:")
        print(f"  Mean: {hard_df['measured_accuracy'].mean():.4f}")
        print(f"  Std: {hard_df['measured_accuracy'].std():.4f}")
        print(f"  Min: {hard_df['measured_accuracy'].min():.4f}")
        print(f"  Max: {hard_df['measured_accuracy'].max():.4f}")
    
    # Save all scores for analysis
    if args.save_all_scores and all_scores:
        scores_df = pd.DataFrame(all_scores)
        scores_path = os.path.join(args.output_dir, "all_accuracy_scores.parquet")
        scores_df.to_parquet(scores_path)
        print(f"\nSaved all accuracy scores to: {scores_path}")
        
        # Print overall statistics
        print(f"\nOverall accuracy statistics (n={len(scores_df)}):")
        print(f"  Mean: {scores_df['accuracy'].mean():.4f}")
        print(f"  Median: {scores_df['accuracy'].median():.4f}")
        print(f"  Std: {scores_df['accuracy'].std():.4f}")
        
        # Distribution of accuracies
        print(f"\nAccuracy distribution:")
        bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        for i in range(len(bins) - 1):
            count = ((scores_df['accuracy'] >= bins[i]) & (scores_df['accuracy'] < bins[i+1])).sum()
            pct = count / len(scores_df) * 100
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} ({pct:.1f}%)")
    
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

