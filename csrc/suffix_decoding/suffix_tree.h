// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <deque>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "int32_map.h"

struct Group;

struct Node {
    // Number of suffixes from the root that end at or pass through this node.
    int64_t count = 0;

    // Token referenced by this node. Node can refer to a sequence of tokens,
    // this is just the ID of the first token.
    int token = 0;

    // Number of tokens in this node.
    int length = 0;

    // Reference sequence ID and starting index for the tokens in this node.
    // Implements the path compression optimization to achieve O(N) memory.
    int ref_seq = 0;
    int ref_idx = -1;

    // This map tracks all the suffixes that end at this node. Maps seq_id to
    // the end index of that suffix (may be truncated due to tree depth). Used
    // to find a new ref_seq and ref_idx if the reference sequence is deleted.
    Int32Map<int> endpoints;

    // Pointer to parent node.
    Node* parent = nullptr;

    // Children nodes, the key should always be the first token of the child.
    Int32Map<std::unique_ptr<Node>> children;

    // All the children of each node are kept in order of decreasing count in
    // a doubly linked list for efficient speculation. head_child points to the
    // first child (highest count) and tail_child points to the last child.
    Node* head_child = nullptr;
    Node* tail_child = nullptr;

    // Pointers to the next and previous siblings in the doubly linked list.
    Node* next_sibling = nullptr;
    Node* prev_sibling = nullptr;

    // To enable efficient reordering of the siblings list when counts change,
    // nodes with the same count are grouped together. Each node holds a shared
    // pointer to its group, and the groups also form a doubly linked list.
    std::shared_ptr<Group> group = nullptr;

    Node() = default;

    Node(int64_t count, int token, int length, int ref_seq, int ref_idx)
        : count(count),
          token(token),
          length(length),
          ref_seq(ref_seq),
          ref_idx(ref_idx) {}

    // Memory usage of this node.
    size_t memory_usage() const {
        size_t total = sizeof(*this);
        total += children.memory_usage();
        total += endpoints.memory_usage();
        return total;
    }
};

struct Group {
    // Pointer to the head node of this group. All nodes before the head node
    // have a strictly higher count, and all nodes after the head node have a
    // lower or equal count.
    Node* head = nullptr;

    // Pointers to the next and previous groups in the doubly linked list.
    Group* next = nullptr;
    Group* prev = nullptr;

    Group(Node* head) : head(head) {}
};

struct Draft {
    // The token ids of the speculation draft.
    std::vector<int32_t> token_ids;

    // For each token, the index of its parent token (-1 if no parent).
    std::vector<int32_t> parents;

    // For each token, the estimated probability of the token.
    std::vector<float> probs;

    // Floating point score of the draft (sum of all probs).
    float score = 0.0;

    // Length of the prefix match for the speculated tokens.
    int match_len = 0;
};

class SuffixTree {
public:

    SuffixTree(int max_depth);

    int num_seqs() const {
        return static_cast<int>(_seqs.size());
    }

    // Append a new element to the sequence with id seq_id.
    void append(int seq_id, int token);

    // Append multiple new elements to the sequence with id seq_id.
    void extend(int seq_id, std::span<const int32_t> tokens);

    // Remove the sequence with id seq_id.
    void remove(int seq_id);

    // Given a context, speculate the next tokens using the suffix tree.
    Draft speculate(std::span<const int32_t> context,
                    int max_spec_tokens,
                    float max_spec_factor,
                    float max_spec_offset,
                    float min_token_prob,
                    bool use_tree_spec);

    // Check the integrity of the suffix tree, return empty string if ok,
    // otherwise return an error message.
    std::string check_integrity();

    // Estimate memory usage of the suffix tree, for debugging only. It
    // walks the entire tree so can be slow.
    size_t estimate_memory() const;

private:

    // Maximum depth of the suffix tree.
    int _max_depth;

    // The root node of the suffix tree.
    std::unique_ptr<Node> _root;

    // Mapping from seq id to its sequence of tokens (vectors of int32_t).
    Int32Map<std::vector<int32_t>> _seqs;

    // For each sequence, a sliding window of active nodes. Maintains at most
    // _max_depth active nodes for each sequence. Queue is shifted when a new
    // token is added to the sequence. Each active node is in the queue for at
    // most _max_depth iterations before being removed.
    Int32Map<std::deque<Node*>> _active_nodes;

    std::pair<Node*, int> _match_context(std::span<const int32_t> context);

    Draft _speculate_path(Node* node, int idx, int max_spec_tokens,
                          float min_token_prob);

    Draft _speculate_tree(Node* node, int idx, int max_spec_tokens,
                          float min_token_prob);

    std::string _check_node_integrity(Node* node);
};
