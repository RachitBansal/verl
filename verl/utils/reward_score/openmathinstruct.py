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
Reward scoring function for OpenMathInstruct-2 dataset.

This dataset contains a mix of GSM8K-style and MATH-style questions.
We use:
1. Boxed format extraction (like MATH dataset)
2. MATH-style equivalence checking (handles both simple and complex LaTeX)
"""

from typing import Optional

# Import extraction and comparison functions from existing modules
from verl.utils.reward_score.math_reward import (
    extract_boxed_answer,
    is_equiv as math_is_equiv,
)


def extract_answer(solution_str: str) -> Optional[str]:
    """
    Extract answer from solution string using boxed format.

    Since we instruct models to use \\boxed{} format for OpenMathInstruct-2,
    we primarily use boxed extraction (inherited from MATH dataset logic).

    Args:
        solution_str: The model's solution text

    Returns:
        Extracted answer string or None if no answer found
    """
    # Use MATH-style boxed extraction
    answer = extract_boxed_answer(solution_str)
    return answer


def check_correctness_math_logic(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer is correct using MATH-style comparison only.

    Since we extract answers using \\boxed{} format (MATH-style), we use
    only MATH-style equivalence checking to avoid false positives from
    buggy numerical fallbacks in GSM8K-style checking.

    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if answer is correct
    """
    if not predicted or not ground_truth:
        return False

    # Use only MATH-style equivalence check
    # (handles both simple numbers and complex LaTeX formatting correctly)
    try:
        if math_is_equiv(predicted, ground_truth):
            return True
    except Exception:
        pass

    return False


def compute_score(solution_str: str, ground_truth: str, format_score: float = 0.0, score: float = 1.0) -> float:
    """
    Compute reward score for OpenMathInstruct-2 dataset.

    Process:
    1. Extract answer using boxed format (MATH-style)
    2. Check correctness using MATH-style equivalence checking
    3. Return score if correct, format_score if answer found but incorrect, 0 if no answer

    Args:
        solution_str: The model's solution text
        ground_truth: Ground truth answer
        format_score: Score to give if answer is extracted but incorrect (default: 0.0)
        score: Score to give if answer is correct (default: 1.0)

    Returns:
        Score value (0.0, format_score, or score)
    """
    # Extract answer using boxed format
    answer = extract_answer(solution_str)

    if answer is None:
        # No answer found in expected format
        return 0.0

    # Check if answer is correct using MATH-style checking
    if check_correctness_math_logic(answer, ground_truth):
        return score
    else:
        # Answer found but incorrect
        return format_score
