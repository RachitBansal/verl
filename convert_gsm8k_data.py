#!/usr/bin/env python3
"""
Convert existing GSM8K parquet files to VERL RLHF format
"""

import argparse
import os
import re
import pandas as pd


def extract_solution(solution_str):
    """Extract the numerical answer from the GSM8K answer string"""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        # If no #### found, return empty string
        return ""
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def convert_row(row, idx, split):
    """Convert a single row to VERL RLHF format"""
    question_raw = row["question"]
    answer_raw = row["answer"]
    
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    question = question_raw + " " + instruction_following
    
    solution = extract_solution(answer_raw)
    
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
    }


def convert_parquet(input_path, output_path, split):
    """Convert a parquet file to VERL format"""
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Converting {len(df)} rows...")
    converted_rows = []
    for idx, row in df.iterrows():
        converted_rows.append(convert_row(row, idx, split))
    
    converted_df = pd.DataFrame(converted_rows)
    
    print(f"Saving to {output_path}...")
    converted_df.to_parquet(output_path)
    print(f"Done! Saved {len(converted_df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GSM8K parquet files to VERL RLHF format")
    parser.add_argument("--train_input", required=True, help="Input train parquet file")
    parser.add_argument("--test_input", required=True, help="Input test parquet file")
    parser.add_argument("--output_dir", default="./data/gsm8k_verl", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert train and test files
    convert_parquet(args.train_input, os.path.join(args.output_dir, "train.parquet"), "train")
    convert_parquet(args.test_input, os.path.join(args.output_dir, "test.parquet"), "test")
    
    print(f"\nConversion complete! Files saved in {args.output_dir}")

