"""
Convert parquet predictions file to human-readable text format.
"""

import pandas as pd
import sys

def convert_parquet_to_text(parquet_path, output_path):
    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    print(f"Loaded {len(df)} examples from {parquet_path}")
    print(f"Columns: {df.columns.tolist()}")

    # Write to text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Predictions from: {parquet_path}\n")
        f.write(f"Total examples: {len(df)}\n")
        f.write("=" * 100 + "\n\n")

        for idx, row in df.iterrows():
            f.write(f"Example {idx + 1}\n")
            f.write("-" * 100 + "\n")

            # Write all columns for this example
            for col in df.columns:
                value = row[col]
                f.write(f"{col}:\n")
                f.write(f"{value}\n")
                f.write("\n")

            f.write("=" * 100 + "\n\n")

    print(f"Converted to: {output_path}")

if __name__ == "__main__":
    parquet_path = "/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results/1B-step22000-rl-step500-0shot-boxed-1samples-temp0.0/gsm8k_predictions.parquet"
    output_path = "/n/netscratch/dam_lab/Everyone/rl_pretrain/eval_results/1B-step22000-rl-step500-0shot-boxed-1samples-temp0.0/gsm8k_predictions.txt"

    convert_parquet_to_text(parquet_path, output_path)
