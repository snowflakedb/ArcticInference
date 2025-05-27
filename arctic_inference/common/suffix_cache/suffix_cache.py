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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, List, Optional, Sequence, Union

from arctic_inference.common.suffix_cache._C import SuffixTree, Candidate


@dataclass
class SuffixSpecResult:
    """
    A dataclass representing the result of a speculation using SuffixDecoding.

    Attributes:
        token_ids (List[int]): List of token IDs in the speculation result.
        parents (List[int]): List of parent indices for each token used to
            encode the tree structure. The parent token of token_ids[i] is
            token_ids[parents[i]].
        probs (List[float]): List of estimated probabilities for each token.
        score (float): The overall score of the suffix match computed as the
            sum of the estimated probabilities of each speculated token.
        match_len (int): The length of the pattern match that yielded this
            speculation result.
    """
    token_ids: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_candidate(candidate: Candidate) -> SuffixSpecResult:
        return SuffixSpecResult(
            token_ids=candidate.token_ids,
            parents=candidate.parents,
            probs=candidate.probs,
            score=candidate.score,
            match_len=candidate.match_len,
        )


class SuffixCache:
    
    def __init__(self, max_depth: int = 64):
        self._max_depth = max_depth
        self._suffix_tree = SuffixTree(max_depth)
        self._prompt_trees = {}
        self._req_to_seq_id = {}

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def has_cached_prompt(self, req_id: Hashable) -> bool:
        return req_id in self._prompt_trees

    def cached_prompt_ids(self) -> List[Hashable]:
        return list(self._prompt_trees.keys())

    def cache_prompt(self, req_id: Hashable, prompt_token_ids: Sequence[int]):
        """
        Cache a prompt for a specific request ID. Future speculations for the
        same request may also source draft tokens from this prompt.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt to be cached.

        Raises:
            ValueError: If a prompt already exists for the given request ID.

        Note:
            The caller should evict the cached prompt using `evict_prompt` once
            the prompt is no longer needed (i.e. the request is completed).
        """
        if req_id in self._prompt_trees:
            raise ValueError(f"Prompt already exists for request '{req_id}'")
        self._prompt_trees[req_id] = SuffixTree(self._max_depth)
        self._prompt_trees[req_id].extend(0, prompt_token_ids)

    def evict_prompt(self, req_id: Hashable):
        """
        Evicts a prompt from the cache for a specific request.

        Args:
            req_id (Hashable): The unique identifier for the request whose
                prompt should be evicted.

        Raises:
            ValueError: If no prompt exists for the given request identifier.
        """
        if req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        del self._prompt_trees[req_id]

    def _get_or_assign_seq_id(self, req_id: Hashable) -> int:
        if req_id not in self._req_to_seq_id:
            self._req_to_seq_id[req_id] = len(self._req_to_seq_id)
        return self._req_to_seq_id[req_id]

    def update_response(
        self,
        req_id: Hashable,
        token_ids: Union[int | Sequence[int]],
    ):
        """
        Update the cached response for a given request by adding token(s) to
        its end. It does not rely on the prompt being cached for the request,
        and its lifetime does not depend on the prompt's existence. Once the
        response is updated, the new tokens can be used for future speculations
        for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Notes:
            - If req_id doesn't exist, a new empty sequence will be initialized.
            - If token_ids is a single integer, it's added as a single token.
            - If token_ids is a sequence, all tokens in the sequence are added.
        """
        seq_id = self._get_or_assign_seq_id(req_id)
        if isinstance(token_ids, int):
            self._suffix_tree.append(seq_id, token_ids)
            if req_id in self._prompt_trees:
                self._prompt_trees[req_id].append(0, token_ids)
        else:
            self._suffix_tree.extend(seq_id, token_ids)
            if req_id in self._prompt_trees:
                self._prompt_trees[req_id].extend(0, token_ids)

    def speculate(
        self,
        req_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
        use_cached_prompt: bool = True,
    ) -> SuffixSpecResult:
        """
        Speculates and returns the most likely continuation of a given token
        pattern using the request-specific prompt cache (if available) and the
        global cache of previous responses.

        Args:
            req_id (Hashable): The unique identifier for the request.
            pattern (Sequence[int]): The sequence of token IDs to match and
                continue from.
            max_spec_tokens (int): Maximum number of tokens to speculate. If 0,
                uses the cache's max_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched pattern length.
            min_token_prob (float): Minimum estimated probability threshold for
                candidate tokens.
            use_tree_spec (bool): If True, uses tree-based speculation.
            use_cached_prompt (bool): If True, uses the cached prompt for the
                request in addition to the global cache of previous responses.
        
        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the prompt doesn't exist for the given req_id when
                use_cached_prompt is True, or if the pattern is invalid.
        """
        if use_cached_prompt and req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        if not pattern:
            raise ValueError("Pattern must not be empty")

        if max_spec_tokens is None:
            max_spec_tokens = self.max_depth

        if len(pattern) > self._max_depth:
            pattern = pattern[-self._max_depth :]

        if use_cached_prompt:
            prompt_tree = self._prompt_trees[req_id]
            candidate = prompt_tree.speculate(
                pattern,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec)
            result = SuffixSpecResult.from_candidate(candidate)
        else:
            result = SuffixSpecResult()

        candidate = self._suffix_tree.speculate(
            pattern,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec)
        if candidate.score > result.score:
            result = SuffixSpecResult.from_candidate(candidate)
        return result
