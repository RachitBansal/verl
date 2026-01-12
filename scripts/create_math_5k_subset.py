#!/usr/bin/env python3
"""
Create a 5k subset of MATH examples from OpenMathInstruct2.
"""

import pandas as pd
import os

# Input/output paths
DATA_DIR = "/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2"
OUTPUT_DIR = "/n/netscratch/dam_lab/Everyone/rl_pretrain/data/openmathinstruct2_math"
TRAIN_FILE = os.path.join(DATA_DIR, "train.parquet")

# Load the full training data
print(f"Loading {TRAIN_FILE}...")
df = pd.read_parquet(TRAIN_FILE)
print(f"Total examples: {len(df)}")

# Check problem_source values
print("\nProblem source distribution:")
print(df['problem_source'].value_counts())

# Filter for MATH-related examples (not GSM8K)
math_sources = df['problem_source'].str.contains('math', case=False) & ~df['problem_source'].str.contains('gsm', case=False)
df_math = df[math_sources]
print(f"\nMATH-related examples: {len(df_math)}")
print("MATH problem sources:")
print(df_math['problem_source'].value_counts())

# Randomly sample examples
SAMPLE_SIZE = 100000
if len(df_math) < SAMPLE_SIZE:
    print(f"Warning: Only {len(df_math)} MATH examples available, using all")
    df_sample = df_math
else:
    df_sample = df_math.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"\nSampled {len(df_sample)} examples")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the subset
output_file = os.path.join(OUTPUT_DIR, f"train_math_{SAMPLE_SIZE}.parquet")
df_sample.to_parquet(output_file, index=False)
print(f"\nSaved {len(df_sample)} examples to {output_file}")

# Save a sample JSON for reference
sample_json = os.path.join(OUTPUT_DIR, f"train_math_{SAMPLE_SIZE}_example.json")
df_sample.iloc[0].to_json(sample_json, indent=2)
print(f"Saved example to {sample_json}")

# Copy val_math.parquet if it exists
val_math_src = os.path.join(DATA_DIR, "val_math.parquet")
if os.path.exists(val_math_src):
    import shutil
    val_math_dst = os.path.join(OUTPUT_DIR, "val_math.parquet")
    shutil.copy(val_math_src, val_math_dst)
    print(f"Copied val_math.parquet to {OUTPUT_DIR}")

# Write info file
info_file = os.path.join(OUTPUT_DIR, "info.txt")
with open(info_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write(f"MATH subset created from {DATA_DIR}\n")
    f.write(f"  - train_math_{SAMPLE_SIZE}.parquet: {len(df_sample)} examples (random MATH subset)\n")
    f.write(f"  - val_math.parquet: 1000 examples (MATH validation)\n")
    f.write(f"\nProblem sources in subset:\n")
    for src, count in df_sample['problem_source'].value_counts().items():
        f.write(f"  - {src}: {count}\n")
    f.write("=" * 60 + "\n")
print(f"Saved info to {info_file}")

print("\nDone!")

