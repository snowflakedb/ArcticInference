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

#include <cassert>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "suffix_tree.h"

SuffixTree::SuffixTree(int max_depth)
    : _max_depth(max_depth), _root(new Node()) {
}

// Append a new element to a new or existing sequence.
void SuffixTree::append(int seq_id, int token) {
    // Initialize the sequence if it doesn't exist.
    _seqs.try_emplace(seq_id);
    _active_nodes.try_emplace(seq_id);
    // Insert a new active node at the root.
    _active_nodes[seq_id].push_back(_root.get());
    _root->count += 1;
    // Ensure the number of active nodes doesn't exceed max_depth.
    if (_active_nodes[seq_id].size() > static_cast<size_t>(_max_depth)) {
        _active_nodes[seq_id].pop_front();
    }
    _seqs[seq_id].push_back(token);
    
    // Iterate over all active nodes for this sequence.
    for (size_t i = 0; i < _active_nodes[seq_id].size(); ++i) {
        Node* node = _active_nodes[seq_id][i];
        auto it = node->children.find(token);
        Node* child = (it != node->children.end()) ? it->second.get() : nullptr;

        if (child == nullptr) {
            // No existing child node.
            if (node->count == 1 && node != _root.get()) {
                // The active node is a leaf (only one suffix) so simply extend its length.
                assert(node->seq_id == seq_id);
                assert(node->children.empty());
                node->length += 1;
            } else {
                // Create a new leaf node for the current suffix.
                Node* new_child = new Node();
                new_child->parent = node;
                new_child->count = 1;
                new_child->seq_id = seq_id;
                new_child->start = static_cast<int>(_seqs[seq_id].size()) - 1;
                new_child->length = 1;
                node->children.emplace(token, new_child);
                _active_nodes[seq_id][i] = new_child;
            }
        }
        else if (node->count == child->count + 1 && node != _root.get()) {
            // The active node should have only one child.
            assert(node->children.size() == 1);
            if (child->length == 1) {
                // Fuse the active node and its single child.
                Node* parent = node->parent;
                // Update child to take the place of the current node.
                child->count += 1;
                child->seq_id = seq_id;
                child->length = node->length + 1;
                child->start = static_cast<int>(_seqs[seq_id].size()) - child->length;
                child->parent = parent;
                // Give ownership of child pointer to parent.
                int tok = _seqs[child->seq_id][child->start];
                assert(parent->children[tok].get() == node);
                parent->children[tok] = std::move(node->children[token]);
                // Replace active node with child node.
                _active_nodes[seq_id][i] = child;
            } else {
                // Extend the active node without overlapping with the child.
                assert(child->length > 1);
                node->seq_id = seq_id;
                node->length += 1;
                node->start = static_cast<int>(_seqs[seq_id].size()) - node->length;
                child->start += 1;
                child->length -= 1;
                int tok = _seqs[child->seq_id][child->start];
                if (tok != token) {
                    node->children[tok] = std::move(node->children[token]);
                    node->children.erase(token);
                }
            }
        }
        else {
            // Either the node has multiple children or the suffix does not uniquely end here.
            if (child->length == 1) {
                child->count += 1;
                _active_nodes[seq_id][i] = child;
            } else {
                // Split the child node.
                assert(child->length > 1);
                Node* new_child = new Node();
                new_child->parent = node;
                new_child->count = child->count + 1;
                new_child->seq_id = seq_id;
                new_child->start = static_cast<int>(_seqs[seq_id].size()) - 1;
                new_child->length = 1;
                int tok = _seqs[child->seq_id][child->start + 1];
                new_child->children[tok] = std::move(node->children[token]);
                node->children[token].reset(new_child);
                child->parent = new_child;
                child->start += 1;
                child->length -= 1;
                _active_nodes[seq_id][i] = new_child;
            }
        }
    }
}

// Extend a new or existing sequence.
void SuffixTree::extend(int seq_id, const std::vector<int>& tokens) {
    for (int token : tokens) {
        append(seq_id, token);
    }
}

Candidate SuffixTree::speculate(const std::vector<int>& pattern,
                                int max_spec_tokens,
                                float max_spec_factor,
                                float max_spec_offset,
                                float min_token_prob,
                                bool use_tree_spec) {
    Candidate result;
    int start_idx = std::max(static_cast<int>(pattern.size()) - _max_depth, 0);
    for ( ; start_idx < pattern.size(); start_idx++) {
        auto[node, idx] = _match_pattern(pattern, start_idx);
        if (node == nullptr) {
            continue;
        }
        int match_len = static_cast<int>(pattern.size()) - start_idx;
        int max_tokens = std::min(max_spec_tokens,
                                  static_cast<int>(match_len * max_spec_factor
                                                   + max_spec_offset + 1e-6));
        max_tokens = std::max(max_tokens, 0);
        Candidate candidate;
        if (use_tree_spec) {
            candidate = _speculate_tree(node, idx, max_tokens, min_token_prob);
        } else {
            candidate = _speculate_path(node, idx, max_tokens, min_token_prob);
        }
        if (candidate.score > result.score) {
            result = std::move(candidate);
            result.match_len = match_len;
        }
    }
    return result;
}

std::pair<Node*, int> SuffixTree::_match_pattern(
        const std::vector<int>& pattern, int start_idx) {
    Node* node = _root.get();
    int idx = 0;
    for (int i = start_idx; i < pattern.size(); i++) {
        int c = pattern[i];
        if (idx >= node->length) {
            auto it = node->children.find(c);
            if (it == node->children.end()) {
                return {nullptr, -1};
            }
            node = it->second.get();
            idx = 0;
        }
        assert(idx < node->length);
        if (_seqs[node->seq_id][node->start + idx] != c) {
            return {nullptr, -1};
        }
        idx++;
    }
    return {node, idx};
}

Candidate SuffixTree::_speculate_path(Node* node, int idx,
                                      int max_spec_tokens,
                                      float min_token_prob) {
    Candidate ret;
    float prob = 1.0f;
    while (ret.token_ids.size() < max_spec_tokens && prob >= min_token_prob) {
        if (idx < node->length) {
            // Use previous token index as parent; if none, mark as -1.
            ret.parents.push_back(static_cast<int>(ret.token_ids.size()) - 1);
            int token = _seqs[node->seq_id][node->start + idx];
            ret.token_ids.push_back(token);
            ret.probs.push_back(prob);
            ret.score += prob;
            idx++;
        } else {
            Node* child = nullptr;
            int count = 0;
            // Choose the child with the maximum count.
            for (auto& kv : node->children) {
                Node* ch = kv.second.get();
                if (ch->count > count) {
                    child = ch;
                    count = ch->count;
                }
            }
            if (child == nullptr) {
                break;
            }
            prob *= static_cast<float>(count) / node->count;
            node = child;
            idx = 0;
        }
    }
    return ret;
}

struct HeapItem {
    float prob;
    Node* node;
    int idx;
    int parent;   // index in the candidate token list; -1 if none.

    HeapItem(float p, Node* n, int i, int par)
        : prob(p), node(n), idx(i), parent(par) {}
};

struct HeapItemCompare {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        // In C++ priority_queue by default returns the largest element.
        // Thus, we compare probabilities so that the highest prob is returned.
        return a.prob < b.prob;
    }
};

// Get a candidate token tree using a priority queue.
Candidate SuffixTree::_speculate_tree(Node* node, int idx,
                                      int max_spec_tokens,
                                      float min_token_prob) {
    Candidate ret;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapItemCompare> queue;
    queue.emplace(1.0, node, idx, -1);
    while (ret.token_ids.size() < max_spec_tokens && !queue.empty()) {
        HeapItem item = queue.top();
        queue.pop();
        if (item.idx < item.node->length) {
            int token = _seqs[item.node->seq_id][item.node->start + item.idx];
            ret.token_ids.push_back(token);
            ret.parents.push_back(item.parent);
            ret.probs.push_back(item.prob);
            ret.score += item.prob;
            queue.emplace(item.prob, item.node, item.idx + 1,
                          static_cast<int>(ret.token_ids.size()) - 1);
        } else {
            for (auto& kv : item.node->children) {
                Node* child = kv.second.get();
                float prob = item.prob * child->count / 
                    static_cast<float>(item.node->count);
                if (prob >= min_token_prob) {
                    queue.emplace(prob, child, 0, item.parent);
                }
            }
        }
    }
    return ret;
}
