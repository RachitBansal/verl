#!/usr/bin/env python3
"""
Script to download and prepare the OpenMathInstruct-1 dataset for GRPO training.

The OpenMathInstruct-1 dataset from NVIDIA contains 1.8M math problems with solutions
that use a mix of text reasoning and code blocks.

Dataset URL: https://huggingface.co/datasets/nvidia/OpenMathInstruct-1
"""

import os
import argparse
from pathlib import Path


def prepare_openmath_dataset(output_dir: str, subset_size: int = None):
    """
    Download and prepare OpenMathInstruct-1 dataset.
    
    Args:
        output_dir: Directory to save the processed parquet files
        subset_size: If specified, only use a subset of the data for testing
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found. Please install it:")
        print("  pip install datasets")
        return False
    
    print("Loading OpenMathInstruct-1 dataset from HuggingFace...")
    print("Note: This is a large dataset (~6M rows), so it may take some time.")
    
    # Load the dataset
    dataset = load_dataset("nvidia/OpenMathInstruct-1")
    
    print(f"Dataset loaded!")
    print(f"  Train split: {len(dataset['train'])} samples")
    print(f"  Validation split: {len(dataset['validation'])} samples")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process and save the data
    for split_name in ['train', 'validation']:
        print(f"\nProcessing {split_name} split...")
        split_data = dataset[split_name]
        
        # If subset_size is specified, take a random subset
        if subset_size is not None and len(split_data) > subset_size:
            print(f"  Taking a random subset of {subset_size} samples...")
            split_data = split_data.shuffle(seed=42).select(range(subset_size))
        
        # Filter to only keep correct solutions for training
        # This ensures the model learns from high-quality examples
        print(f"  Filtering to keep only correct solutions...")
        split_data = split_data.filter(lambda x: x['is_correct'])
        print(f"  After filtering: {len(split_data)} samples")
        
        # Transform the data to match expected format
        # The dataset has 'question' field which will be used as prompt
        # We'll keep the 'generated_solution' as reference (though GRPO generates its own)
        def transform_sample(sample):
            return {
                'prompt': sample['question'],
                'reference_solution': sample['generated_solution'],
                'expected_answer': sample['expected_answer'],
                'dataset_source': sample['dataset'],  # 'gsm8k' or 'math'
            }
        
        print(f"  Transforming data...")
        split_data = split_data.map(transform_sample, remove_columns=dataset[split_name].column_names)
        
        # Save as parquet
        output_file = output_path / f"{split_name}.parquet"
        print(f"  Saving to {output_file}...")
        split_data.to_parquet(str(output_file))
        print(f"  ✓ Saved {len(split_data)} samples")
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"\nYou can now run the training script with:")
    print(f"  sbatch examples/grpo_trainer/run_olmo2-1b_openmath_slurm.sh")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare OpenMathInstruct-1 dataset for GRPO training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/n/netscratch/dam_lab/Lab/sqin/tmp_data/openmath",
        help="Directory to save the processed parquet files"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="If specified, only use a subset of the data (useful for testing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode: only process 1000 samples per split"
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in TEST mode with 1000 samples per split")
        args.subset_size = 1000
    
    success = prepare_openmath_dataset(args.output_dir, args.subset_size)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

