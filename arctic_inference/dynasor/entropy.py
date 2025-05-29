# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional

from arctic_inference.dynasor.evaluator import math_equal


def entropy(Plist: List[float]) -> float:
    """Calculate the Shannon entropy of a probability distribution.
    
    Args:
        Plist: List of probabilities that sum to 1
        
    Returns:
        float: The entropy value in bits (using log base 2)
    """
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0


def norm(Olist: List[float]) -> List[float]:
    """Normalize a list of numbers to sum to 1.
    
    Args:
        Olist: List of numbers to normalize
        
    Returns:
        List[float]: Normalized list where sum equals 1
    """
    s = sum(Olist)
    return [o / s for o in Olist]


def count(Olist: List[Any]) -> List[float]:
    """Count occurrences of each unique element in a list.
    
    Args:
        Olist: List of elements to count
        
    Returns:
        List[float]: List of counts for each unique element
    """
    x_dict = defaultdict(lambda: 0.0)
    for x in Olist:
        x_dict[x] += 1
    cc = [c for _, c in x_dict.items()]
    # print(cc)
    return cc


def item_entropy(answers: List[Any]) -> float:
    """Calculate the entropy of a list of answers.
    
    Args:
        answers: List of answers to calculate entropy for
        
    Returns:
        float: Entropy value in bits
    """
    return entropy(norm(count(answers)))


def count_not_empty(answers: List[str]) -> int:
    """Count the number of non-empty strings in a list.
    
    Args:
        answers: List of strings to check
        
    Returns:
        int: Number of non-empty strings
    """
    return sum(1 for answer in answers if answer != "")


def equal_group(answers: List[Any]) -> bool:
    """Check if all answers in a list are equivalent.
    
    Args:
        answers: List of answers to compare
        
    Returns:
        bool: True if all answers are equivalent, False otherwise
    """
    equiv_classes = []

    for answer in answers:
        weight = 1
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer, rep):
                flag = 1
                break
        if flag:
            continue
        equiv_classes.append(answer)

    return len(equiv_classes) == 1


def majority_voting(answers: List[Any]) -> Any:
    """Find the most common answer using majority voting.
    
    Args:
        answers: List of answers to vote on
        
    Returns:
        Any: The most common answer
    """
    equiv_classes = []
    equiv_weights = []
    max_vote = 0
    for answer in answers:
        weight = 1
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer, rep):
                flag = 1
                equiv_weights[i] = equiv_weights[i] + weight
                if equiv_weights[i] > max_vote:
                    max_vote = equiv_weights[i]
                    max_rep = answer
                break
        if flag:
            continue
        equiv_classes.append(answer)
        equiv_weights.append(weight)
        if max_vote == 0:
            max_vote = weight
            max_rep = answer
    return max_rep


def obtain_answer(s: str) -> str:
    """Extract the first complete answer from a string by matching braces.
    
    Args:
        s: Input string containing potential answer
        
    Returns:
        str: The first complete answer found, or empty string if none found
    """
    # Find first unpaired } by counting { and }
    stack = []
    for i, c in enumerate(s):
        if c == "{":
            stack.append(c)
        elif c == "}":
            if not stack:  # No matching { found
                return s[:i]
            stack.pop()
    return ""


uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]


def is_certain_answer(probe_response_text: str, uncertain_words: List[str]) -> bool:
    """Check if the answer is certain by looking for uncertain words.
    
    Args:
        probe_response_text: Text to check for uncertainty
        uncertain_words: List of words that indicate uncertainty
        
    Returns:
        bool: True if the answer is certain, False otherwise
    """
    return not any(word in probe_response_text.lower() for word in uncertain_words)


def has_value(x: Any) -> bool:
    """Check if a value exists and is non-empty.
    
    Args:
        x: Value to check
        
    Returns:
        bool: True if value exists and is non-empty, False otherwise
    """
    if x is None:
        return False
    if isinstance(x, str):
        return len(x.strip()) > 0
    if isinstance(x, list):
        return len(x) > 0
    return True


def should_early_exit(
    answers: List[str],
    probe_response_text: str,
    uncertain_words: List[str],
    continue_certain_bar: int,
    is_certains: List[bool],
) -> bool:
    """Check if the answer is consistent and certain enough to exit early.
    1. Number of answers should be greater than the threshold
    2. The probe response text should not contain any uncertain words
    3. The answers should be consistent
    
    Args:
        answers: List of answers to check
        probe_response_text: Text of the probe response
        uncertain_words: List of words that indicate uncertainty
        continue_certain_bar: Threshold for number of consistent answers needed
        is_certains: List of booleans indicating if each answer is certain
        
    Returns:
        bool: True if should exit early, False otherwise

    """

    # Number of answers should be greater than the threshold
    if len(answers) < continue_certain_bar:
        return False

    # The probe response text should not contain any uncertain words
    probe_response_text_lower = probe_response_text.lower()
    if any(word in probe_response_text_lower for word in uncertain_words):
        return False

    # The last answer window should be consistent
    answer_candidates = answers[-continue_certain_bar:]
    is_certains = is_certains[-continue_certain_bar:]
    if equal_group(answer_candidates):
        if count_not_empty(answer_candidates) == continue_certain_bar:
            if sum(is_certains) == continue_certain_bar:
                # logger.debug(f"Early exit on: {answer_candidates = } ({is_certains = })")
                return True

    return True
