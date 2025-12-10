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

_SOLUTION_CLIP_CHARS = 300

_STRICT_PATTERNS = [
    re.compile(r"####\s*(?:final\s+answer|answer)?\s*[:=]?\s*([^\n]+)", re.IGNORECASE),
]

_FLEXIBLE_PATTERNS = _STRICT_PATTERNS + [
    re.compile(r"(?:^|\n)\s*final\s+answer\s*[:=]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*answer\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:the\s+answer|our\s+answer)\s*(?:is|=)\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:the\s+result|result)\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE),
]

_BOXED_START_PATTERN = re.compile(r"\\boxed\s*\{")
_FRAC_PATTERN = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")


def _strip_braces(token):
    token = token.strip()
    while token.startswith("{") and token.endswith("}") and len(token) >= 2:
        token = token[1:-1].strip()
    return token


def _iter_boxed_segments(text):
    for match in _BOXED_START_PATTERN.finditer(text):
        start = match.start()
        index = match.end()
        depth = 1
        end = index
        while end < len(text) and depth > 0:
            char = text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            end += 1
        if depth == 0:
            yield start, text[index:end - 1], end


def _strip_latex_wrappers(text):
    if not text:
        return text

    unwrapped = text
    # Iteratively remove boxed expressions in case of nesting.
    while True:
        pieces = []
        last_index = 0
        changed = False
        for start, content, end in _iter_boxed_segments(unwrapped):
            pieces.append(unwrapped[last_index:start])
            pieces.append(content)
            last_index = end
            changed = True
        if not changed:
            break
        pieces.append(unwrapped[last_index:])
        unwrapped = "".join(pieces)

    def _frac_replacer(match):
        numerator = _strip_braces(match.group(1))
        denominator = _strip_braces(match.group(2))
        return f"{numerator}/{denominator}"

    unwrapped = _FRAC_PATTERN.sub(_frac_replacer, unwrapped)
    unwrapped = re.sub(r"\\text\{([^}]*)\}", r"\1", unwrapped)
    unwrapped = re.sub(r"\\(?:left|right)\s*[\{\}\(\)\[\]]", "", unwrapped)
    unwrapped = unwrapped.replace("\\,", "")
    unwrapped = unwrapped.replace("\\!", "")
    return unwrapped


def _normalize_candidate(raw_candidate):
    if not raw_candidate:
        return None
    candidate = raw_candidate.strip()
    if not candidate:
        return None

    candidate = _strip_latex_wrappers(candidate)
    candidate = candidate.replace(",", "")
    candidate = candidate.replace("$", "")
    candidate = candidate.replace("\\%", "%")
    candidate = candidate.strip()
    if not candidate:
        return None

    candidate = re.sub(r"[{}]", "", candidate)
    candidate = candidate.strip()
    if not candidate:
        return None

    frac_match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", candidate)
    if frac_match:
        numerator, denominator = frac_match.groups()
        numerator = numerator.rstrip(".")
        denominator = denominator.rstrip(".")
        if numerator and denominator:
            return f"{numerator}/{denominator}"

    number_match = _NUMBER_PATTERN.search(candidate)
    if number_match:
        number = number_match.group(0).replace(" ", "").rstrip(".")
        return number if number else None

    return None


def _extract_candidates(solution_str, patterns, include_boxed=False):
    candidates = []
    for pattern in patterns:
        for match in pattern.finditer(solution_str):
            raw = match.group(match.lastindex or 0)
            normalized = _normalize_candidate(raw)
            if normalized:
                candidates.append((match.start(), normalized))

    if include_boxed:
        for start, content, _ in _iter_boxed_segments(solution_str):
            normalized = _normalize_candidate(content)
            if normalized:
                candidates.append((start, normalized))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]
    return None


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # Require the presence of the expected "####" format, but allow for latex wrappers.
        final_answer = _extract_candidates(solution_str, _STRICT_PATTERNS, include_boxed=False)
    elif method == "flexible":
        final_answer = _extract_candidates(solution_str, _FLEXIBLE_PATTERNS, include_boxed=True)
        if final_answer is None:
            preprocessed = _strip_latex_wrappers(solution_str).replace(",", "")
            numeric_candidates = _NUMBER_PATTERN.findall(preprocessed)
            cleaned = []
            for candidate in numeric_candidates:
                value = candidate.replace(" ", "").rstrip(".")
                if value and value not in {"", "."}:
                    cleaned.append(value)
            if cleaned:
                final_answer = cleaned[-1]
    else:
        final_answer = None
    return final_answer


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
        if answer == ground_truth:
            return score
        else:
            return format_score
