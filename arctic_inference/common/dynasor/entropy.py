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
from typing import List

from arctic_inference.common.dynasor.evaluator import math_equal


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0


def norm(Olist):
    s = sum(Olist)
    return [o / s for o in Olist]


def count(Olist):
    x_dict = defaultdict(lambda: 0.0)
    for x in Olist:
        x_dict[x] += 1
    cc = [c for _, c in x_dict.items()]
    # print(cc)
    return cc


def item_entropy(answers: List) -> float:
    return entropy(norm(count(answers)))


def count_not_empty(answers):
    return sum(1 for answer in answers if answer != "")


def eqaul_group(answers):
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


def majority_voting(answers):
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


def obtaint_answer(s):
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


def obtain_answer(s):
    return obtaint_answer(s)


uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]


def is_certain_answer(probe_response_text: str, uncertain_words: list[str]) -> bool:
    """Check if the answer is certain"""
    return not any(word in probe_response_text.lower() for word in uncertain_words)


def has_value(x) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return len(x.strip()) > 0
    if isinstance(x, list):
        return len(x) > 0
    return True


def should_early_exit(
    answers: list[str],
    probe_response_text: str,
    uncertain_words: list[str],
    continue_certain_bar: int,
    is_certains: list[bool],
) -> bool:
    """
    Check if the answer is consistent or certain.
    1. Number of answers should be greater than the threshold
    2. The probe response text should not contain any uncertain words
    3. The answers should be consistent
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
    if eqaul_group(answer_candidates):
        if count_not_empty(answer_candidates) == continue_certain_bar:
            if sum(is_certains) == continue_certain_bar:
                # logger.debug(f"Early exit on: {answer_candidates = } ({is_certains = })")
                return True

    return True