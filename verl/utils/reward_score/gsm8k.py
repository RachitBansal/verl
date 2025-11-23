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

import re
from typing import Optional
import numpy as np


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    - Remove whitespace
    - Convert to lowercase
    - Remove LaTeX formatting
    - Handle common equivalences

    Adapted from interleaved-rl/src/evaluation/math_eval.py
    """
    if not answer:
        return ""

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    # Remove LaTeX commands but keep content
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', answer)

    # Remove common LaTeX symbols
    answer = answer.replace('\\%', '%')
    answer = answer.replace('\\$', '$')
    answer = answer.replace('\\,', '')
    answer = answer.replace('\\:', '')
    answer = answer.replace('\\;', '')
    answer = answer.replace('\\!', '')

    # Remove dollar signs (for math mode)
    answer = answer.replace('$', '')

    # Convert to lowercase for comparison
    answer = answer.lower()

    # Remove trailing punctuation
    answer = answer.rstrip('.')

    return answer.strip()


def extract_number(text: str) -> Optional[float]:
    """
    Extract numerical value from text for comparison.

    Adapted from interleaved-rl/src/evaluation/math_eval.py
    """
    if not text:
        return None

    # Remove common units and symbols
    text = text.replace(',', '')  # Remove thousands separator
    text = text.replace('$', '')
    text = text.replace('%', '')

    # Try to extract number
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass

    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} format (common in MATH dataset).

    Adapted from interleaved-rl/src/evaluation/math_eval.py
    """
    # Match \boxed{...} including nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer (typically the final answer)
        return matches[-1].strip()

    return None


def extract_solution(solution_str, method="strict"):
    """
    Extract numerical answer from GSM8K format.
    GSM8K answers typically end with #### followed by the number.

    Adapted from interleaved-rl/src/evaluation/math_eval.py

    Args:
        solution_str: the solution text
        method: 'strict' uses only #### format, 'flexible' tries multiple patterns

    Returns:
        Extracted answer string or None
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # First try to find #### format (takes FIRST match like interleaved-rl)
        match = re.search(r'####\s*(.+?)(?:\n|$)', solution_str)
        if match:
            return match.group(1).strip().replace(",", "").replace("$", "")
        return None

    elif method == "flexible":
        # First try to find #### format
        match = re.search(r'####\s*(.+?)(?:\n|$)', solution_str)
        if match:
            return match.group(1).strip()

        # Otherwise, try to find boxed format
        boxed = extract_boxed_answer(solution_str)
        if boxed:
            return boxed

        # Last resort: extract the last number mentioned
        numbers = re.findall(r'-?\d+\.?\d*', solution_str)
        if numbers:
            return numbers[-1]

        return None


def check_answer_equivalence(predicted: str, ground_truth: str, strict: bool = False) -> bool:
    """
    Check if predicted answer matches ground truth.

    Adapted from interleaved-rl/src/evaluation/math_eval.py

    Args:
        predicted: Model's predicted answer
        ground_truth: Ground truth answer
        strict: If True, require exact match after normalization

    Returns:
        True if answers match
    """
    if not predicted or not ground_truth:
        return False

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    if strict:
        return False

    # Try numerical comparison
    pred_num = extract_number(pred_norm)
    gt_num = extract_number(gt_norm)

    if pred_num is not None and gt_num is not None:
        # Use relative tolerance for floating point comparison
        return np.isclose(pred_num, gt_num, rtol=1e-4, atol=1e-8)

    return False


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        # Use normalization and numerical comparison with tolerance
        if check_answer_equivalence(answer, ground_truth, strict=(method == "strict")):
            return score
        else:
            return format_score
