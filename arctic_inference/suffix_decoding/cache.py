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
from typing import Dict, Hashable, KeysView, List, Optional, Sequence, Tuple
import threading
import queue
import time

import numpy as np

from arctic_inference.suffix_decoding._C import SuffixTree, Draft


@dataclass
class SuffixDecodingDraft:
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
        match_len (int): The length of the context match that yielded this
            speculation result.
    """
    token_ids: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_native(draft: Draft) -> SuffixDecodingDraft:
        return SuffixDecodingDraft(
            token_ids=draft.token_ids,
            parents=draft.parents,
            probs=draft.probs,
            score=draft.score,
            match_len=draft.match_len,
        )


class AsyncBatchProcessor:
    """High-performance batch processor for suffix cache updates."""

    def __init__(self, global_tree: SuffixTree, max_batch_size: int = 1000,
                 max_latency_ms: float = 2.0):
        self._global_tree = global_tree
        self._max_batch_size = max_batch_size
        self._max_latency_s = max_latency_ms / 1000.0

        self._update_queue: "queue.Queue[tuple[int, list[int]]]" = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._batch_lock = threading.Lock()

        self._worker_thread = threading.Thread(
            target=self._batch_worker_loop,
            name="suffix-cache-batch-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def submit_update(self, seq_id: int, token_ids: list[int]) -> bool:
        try:
            self._update_queue.put((int(seq_id), [int(t) for t in token_ids]),
                                   timeout=0.001)
            return True
        except queue.Full:
            self._global_tree.extend(int(seq_id), [int(t) for t in token_ids])
            return False

    def _batch_worker_loop(self) -> None:
        batch: dict[int, list[int]] = {}
        total_tokens = 0
        last_process_time = time.time()

        while not self._stop_event.is_set():
            try:
                items: list[tuple[int, list[int]]] = []
                timeout = max(0.001,
                              self._max_latency_s - (time.time() -
                                                     last_process_time))
                try:
                    first_item = self._update_queue.get(timeout=timeout)
                    items.append(first_item)
                except queue.Empty:
                    first_item = None

                if first_item is not None:
                    while len(items) < self._max_batch_size:
                        try:
                            item = self._update_queue.get_nowait()
                            items.append(item)
                        except queue.Empty:
                            break

                if items:
                    for seq_id, token_ids in items:
                        if seq_id not in batch:
                            batch[seq_id] = token_ids
                        else:
                            batch[seq_id].extend(token_ids)
                        total_tokens += len(token_ids)
                        self._update_queue.task_done()

                current_time = time.time()
                should_process = (
                    total_tokens >= self._max_batch_size or
                    (self._max_latency_s > 0 and (current_time -
                                                  last_process_time)
                     >= self._max_latency_s) or len(batch) >= 100)

                if should_process and batch:
                    self._process_batch(batch)
                    batch.clear()
                    total_tokens = 0
                    last_process_time = current_time
            except Exception:
                # Continue processing even if one batch fails
                continue

        # Final flush on shutdown
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: dict[int, list[int]]) -> None:
        if not batch:
            return
        # Convert to native format and apply in a single call
        batch_updates: list[tuple[int, list[int]]] = []
        for seq_id, tokens in batch.items():
            if tokens:
                batch_updates.append((int(seq_id), [int(t) for t in tokens]))
        if batch_updates:
            self._global_tree.extend_batch(batch_updates)

    def close(self) -> None:
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)


class SuffixDecodingCache:
    
    def __init__(self,
                 max_tree_depth: int = 64,
                 max_cached_requests: int = -1,
                 enable_async_updates: bool = False,
                 async_max_batch_tokens: int = 4096,
                 async_max_latency_ms: int = 1):
        """
        Initialize the SuffixDecodingCache.

        Args:
            max_tree_depth (int): The maximum depth of the suffix trees.
            max_cached_requests (int, optional): The maximum number of cached
                requests. Eviction is triggered when the limit is reached. `-1`
                means no limit on the number of cached requests.
        """
        if max_cached_requests > 0x7FFFFFFF:
            raise ValueError("max_cached_requests must be at most 2^31")

        self._max_tree_depth = max_tree_depth
        self._max_cached_requests = max_cached_requests

        # Global suffix tree caches previous responses in a single tree.
        self._global_tree = SuffixTree(max_tree_depth)

        # Local suffix trees cache prompts for each active request separately.
        self._local_trees = {}

        # Maps between Python request ID and int32_t sequence ID. Tracks all
        # request IDs that are in the global tree.
        self._req_to_seq_id: Dict[Hashable, int] = {}
        self._seq_to_req_id: Dict[int, Hashable] = {}

        # Unused sequence ID to assign to a new request ID.
        self._next_seq_id = 0

        # Async update configuration and state.
        self._async_enabled = enable_async_updates
        self._global_lock = threading.RLock()
        self._batch_processor: Optional[AsyncBatchProcessor] = None
        if self._async_enabled:
            self._batch_processor = AsyncBatchProcessor(
                self._global_tree,
                max_batch_size=int(async_max_batch_tokens),
                max_latency_ms=float(async_max_latency_ms),
            )

    @property
    def max_tree_depth(self) -> int:
        return self._max_tree_depth

    @property
    def max_cached_requests(self) -> int:
        return self._max_cached_requests

    @property
    def active_requests(self) -> KeysView:
        """
        Returns a view of the currently active request IDs. Active requests are
        those that have been started via `start_request` and not yet stopped
        via `stop_request`. The prompts of active requests are stored so they
        can be used during speculation for the same request.
        """
        return self._local_trees.keys()

    @property
    def cached_requests(self) -> KeysView:
        """
        Returns a view of all request IDs that have their responses cached in
        the global suffix tree. The response for the cached request can be used
        during speculation for other requests, until the response is evicted.
        """
        return self._req_to_seq_id.keys()

    def start_request(self, req_id: Hashable, prompt_token_ids: Sequence[int]):
        """
        This method should be called when starting to process a new request. It
        will store the prompt for the request, allowing future speculations for
        the same request to use the prompt context. The prompt will be stored
        until `stop_request` is called. If `max_cached_requests != 0`, then a
        new slot is allocated in the global cache for the response, triggering
        cache eviction (FIFO order) if needed.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt of the request.

        Raises:
            ValueError: If a request with the same `req_id` is already active
                or cached.
        """
        if req_id in self._local_trees:
            raise ValueError(f"Request '{req_id}' is already active")
        self._local_trees[req_id] = SuffixTree(self._max_tree_depth)
        self._local_trees[req_id].extend(0, prompt_token_ids)
        if self._max_cached_requests != 0:
            # Global cache is enabled.
            if req_id in self._req_to_seq_id:
                # Evict existing cached response for the request if present.
                self.evict_cached_response(req_id)
            # Allocate a new seq_id for the request.
            self._generate_seq_id(req_id)

    def stop_request(self, req_id: Hashable):
        """
        This method should be called when a request is completed. It will evict
        the prompt for the request, freeing up memory. The request's response
        may still be cached in the global cache until it is evicted.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")
        del self._local_trees[req_id]

    def add_active_response(
        self,
        req_id: Hashable,
        token_ids: np.ndarray | Sequence[int],
    ):
        """
        Update the cached response for a given request by appending token(s) to
        its end. Once the response is updated, the new tokens can be used for
        future speculations for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if isinstance(token_ids, np.ndarray):
            self._validate_ndarray(token_ids)

        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")

        # Update the local tree for the active request.
        self._local_trees[req_id].extend(0, token_ids)

        # Also update the response if the request is in the global cache (it
        # may be evicted from the global cache before the request is stopped).
        if req_id in self._req_to_seq_id:
            seq_id = self._req_to_seq_id[req_id]
            if self._async_enabled and self._batch_processor is not None:
                ok = self._batch_processor.submit_update(seq_id, token_ids)
                if not ok:
                    self._global_tree.extend(seq_id, token_ids)
            else:
                self._global_tree.extend(seq_id, token_ids)

    def evict_cached_response(self, req_id: Hashable):
        """
        Evicts the given request's response from the global cache. `req_id` can
        be safely reused for a new request after eviction.

        Args:
            req_id (Hashable): The unique identifier for the request that
                should be evicted.

        Raises:
            ValueError: If no response exists for the given request identifier.
        """
        if req_id not in self._req_to_seq_id:
            raise ValueError(f"Request '{req_id}' is not cached")
        seq_id = self._req_to_seq_id.pop(req_id)
        self._seq_to_req_id.pop(seq_id)
        self._global_tree.remove(seq_id)

    def speculate(
        self,
        req_id: Hashable,
        context: np.ndarray | Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> SuffixDecodingDraft:
        """
        Speculates and returns the most likely continuation of a given token
        context using the request's prompt and the global cache of previous
        responses. This method can only be called for active requests (i.e.
        after calling `start_request` and before calling `stop_request`).

        Args:
            req_id (Hashable): The unique identifier for the request.
            context (Sequence[int]): The sequence of token IDs to match and
                continue from.
            max_spec_tokens (int): Maximum number of tokens to speculate. If 0,
                uses the cache's max_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched context length.
            min_token_prob (float): Minimum estimated probability threshold for
                draft tokens.
            use_tree_spec (bool): If True, uses tree-based speculation.
        
        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if isinstance(context, np.ndarray):
            # If context is a numpy array, use the zero-copy ndarray overload.
            self._validate_ndarray(context)  # Make sure the array is valid.
            spec_func = SuffixTree.speculate_ndarray
        else:
            spec_func = SuffixTree.speculate

        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")

        if max_spec_tokens is None:
            max_spec_tokens = self.max_depth

        if len(context) > self._max_tree_depth:
            context = context[-self._max_tree_depth :]

        draft1 = spec_func(
            self._local_trees[req_id],
            context,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec)

        draft2 = spec_func(
            self._global_tree,
            context,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec)

        draft = draft1 if draft1.score >= draft2.score else draft2

        return SuffixDecodingDraft.from_native(draft)

    def _generate_seq_id(self, req_id: Hashable) -> int:
        # Find the next available seq_id not used by an active request.
        while True:
            seq_id = self._next_seq_id
            # Increment to the next non-negative int32_t value.
            self._next_seq_id = (self._next_seq_id + 1) & 0x7FFFFFFF
            if (seq_id not in self._seq_to_req_id or
                    self._seq_to_req_id[seq_id] not in self._local_trees):
                break
        # Check if the seq_id is used by an inactive but cached request.
        if seq_id in self._seq_to_req_id:
            # This seq_id is already used, should be a very rare case that
            # only happens when the seq_id has wrapped around and collided.
            # We evict the old cached request to free up the seq_id.
            del self._req_to_seq_id[self._seq_to_req_id[seq_id]]
            del self._seq_to_req_id[seq_id]
            self._global_tree.remove(seq_id)
        # Allocate the seq_id to the new req_id.
        self._req_to_seq_id[req_id] = seq_id
        self._seq_to_req_id[seq_id] = req_id
        self._maybe_evict_requests(seq_id)
        return seq_id

    def _maybe_evict_requests(self, new_seq_id: int):
        if self._max_cached_requests < 0:
            # Negative value means no global cache size limit.
            return
        assert self._max_cached_requests != 0  # Global cache must be enabled.
        while len(self._req_to_seq_id) > self._max_cached_requests:
            # Evict the first eligible request. Should be FIFO order in Python
            # 3.7+ since dict preserves insertion order. Avoid evicting the
            # request that was just added (new_seq_id).
            for req_id, seq_id in self._req_to_seq_id.items():
                if seq_id != new_seq_id:
                    self.evict_cached_response(req_id)
                    break

    def _validate_ndarray(self, arr: np.ndarray):
        if arr.ndim != 1:
            raise ValueError(f"ndarray input must have ndim=1, "
                             f"got ndim={arr.ndim}")
        if arr.dtype != np.int32:
            raise ValueError(f"ndarray input must have dtype=int32, "
                             f"got dtype={arr.dtype.name}")
        if not arr.flags["CONTIGUOUS"]:
            raise ValueError(f"ndarray input must be contiguous")
