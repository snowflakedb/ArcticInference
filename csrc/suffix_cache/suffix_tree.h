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
#include <unordered_map>
#include <utility>
#include <vector>

#include <iterator>

class ReferenceMap {
/*
 * For a given suffix tree node, this class keeps track of all the sequences and the starting
 * index within those sequences that contain the tokens represented by the node. Supports an
 * efficient way to increment/decrement all indices, which is needed by certain tree operations.
 */
friend class SuffixTree;

public:

    int get_idx(int seq_id) {
        return _idx_in_seq[seq_id] + _idx_offset;
    }

    std::pair<int, int> get_any() {
        auto it = _idx_in_seq.begin();
        assert(it != _idx_in_seq.end());
        return {it->first, it->second + _idx_offset};
    }

    void set_idx(int seq_id, int idx) {
        _idx_in_seq[seq_id] = idx - _idx_offset;
    }

    void erase(int seq_id) {
        if (_idx_in_seq.count(seq_id)) {
            _idx_in_seq.erase(seq_id);
        }
    }

    void add_all(int delta) {
        _idx_offset += delta;
    }

    size_t size() const {
        return _idx_in_seq.size();
    }

    bool contains(int seq_id) const {
        return _idx_in_seq.count(seq_id) > 0;
    }

    class const_iterator {
        using Inner = std::unordered_map<int,int>::const_iterator;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<int,int>;
        using difference_type = std::ptrdiff_t;

        const_iterator(Inner it, int off) : _it(it), _off(off) {}
        value_type operator*() const { return {_it->first, _it->second + _off}; }
        const_iterator& operator++() { ++_it; return *this; }
        bool operator==(const const_iterator& o) const { return _it == o._it; }
        bool operator!=(const const_iterator& o) const { return _it != o._it; }

    private:
        Inner _it;
        int _off;
    };

    const_iterator begin() const {
        return const_iterator(_idx_in_seq.begin(), _idx_offset);
    }

    const_iterator end() const {
        return const_iterator(_idx_in_seq.end(), _idx_offset);
    }

private:
    std::unordered_map<int, int> _idx_in_seq;
    int _idx_offset = 0;
};

struct Node {
    // Token referenced by this node. Node can refer to a sequence of tokens,
    // this is just the ID of the first token.
    int token = 0;

    // Number of suffixes from the root that end at or pass through this node.
    int count = 0;

    // Parent node.
    Node* parent = nullptr;

    // Children nodes, the key should always be the first token of the child.
    std::unordered_map<int, std::unique_ptr<Node>> children;

    // For each sequence that contains this node, tracks the start index of the
    // tokens in this node within that sequence.
    ReferenceMap refs;

    // Number of tokens in this node.
    int length = 0;
};

struct Candidate {
    // The token ids of the speculation candidate.
    std::vector<int> token_ids;

    // For each token, the index of its parent token (-1 if no parent).
    std::vector<int> parents;

    // For each token, the estimated probability of the token.
    std::vector<float> probs;

    // Floating point score of the candidate (sum of all probs).
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
    void extend(int seq_id, const std::vector<int>& tokens);

    // Remove the sequence with id seq_id.
    void remove(int seq_id);

    Candidate speculate(const std::vector<int>& pattern,
                        int max_spec_tokens,
                        float max_spec_factor = 1.0f,
                        float max_spec_offset = 0.0f,
                        float min_token_prob = 0.1f,
                        bool use_tree_spec = false);

    std::string check_integrity();

    std::string check_integrity(Node* node);

private:

    // Maximum depth of the suffix tree.
    int _max_depth;

    // The root node of the suffix tree.
    std::unique_ptr<Node> _root;

    // Mapping from seq id to its sequence (vector of ints).
    std::unordered_map<int, std::vector<int>> _seqs;

    // For each sequence, a sliding window of active nodes. Maintains at most
    // _max_depth active nodes for each sequence. Queue is shifted when a new
    // token is added to the sequence. Each active node is in the queue for at
    // most _max_depth iterations before being removed.
    std::unordered_map<int, std::deque<Node*>> _active_nodes;

    std::pair<Node*, int> _match_pattern(const std::vector<int>& pattern,
                                         int start_idx = 0);

    Candidate _speculate_path(Node* node, int idx, int max_spec_tokens,
                              float min_token_prob);

    Candidate _speculate_tree(Node* node, int idx, int max_spec_tokens,
                              float min_token_prob);
};
