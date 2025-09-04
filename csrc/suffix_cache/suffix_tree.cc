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
    //std::cout << seq_id << " insert " << token << std::endl;
    for (size_t i = 0; i < _active_nodes[seq_id].size(); ++i) {
        //std::cout << "i = " << i << std::endl;
        Node* node = _active_nodes[seq_id][i];
        auto it = node->children.find(token);
        Node* child = (it != node->children.end()) ? it->second.get() : nullptr;

        if (child == nullptr) {
            // No existing child node for the new token.
            if (node->count == 1 && node != _root.get()) {
                //std::cout << "case 1" << std::endl;
                // The active node has count = 1, which means the only suffix that ends here is the
                // one that's being extended right now. Then this node should be a leaf node, and
                // we can simply extend the length of this node.
                assert(node->children.empty());  // Check is a leaf node
                assert(node->idx_in_seq.size() == 1);  // Check only current sequence ends here
                assert(node->idx_in_seq.find(seq_id) != node->idx_in_seq.end());
                node->length += 1;  // Valid since we just appended a token to this sequence
            } else {
                //std::cout << "case 2" << std::endl;
                // Either this is the root node, or the current suffix is not the only one that
                // ends here. Either case, we need to extend the current suffix into a new child.
                Node* new_child = new Node();
                new_child->token = token;
                new_child->parent = node;
                new_child->count = 1;
                new_child->idx_in_seq[seq_id] = static_cast<int>(_seqs[seq_id].size()) - 1;
                new_child->length = 1;
                node->children.emplace(token, new_child);
                _active_nodes[seq_id][i] = new_child;
            }
        }
        else if (node->count == child->count + 1 && node != _root.get()) {
            // The active node has a child for the new token, and the child's count is exactly one
            // fewer than the active node's count. Since the suffix for the active node ends here,
            // that means all other suffixes that pass through this node must go to that child.
            assert(node->children.size() == 1);  // The active node should have only one child.
            if (child->length == 1) {
                //std::cout << "case 3" << std::endl;
                // The child only has length 1. If we append the new token to the current suffix,
                // then it will perfectly overlap with the child. In this case, we should just fuse
                // the current suffix into the child and eliminate the current node.
                Node* parent = node->parent;
                // Update child to take the place of the current node.
                child->token = node->token;
                child->count += 1;  // Current suffix extends into the child
                child->length = node->length + 1;
                child->idx_in_seq = std::move(node->idx_in_seq);
                child->idx_offset = node->idx_offset;
                child->idx_in_seq[seq_id] =
                    static_cast<int>(_seqs[seq_id].size()) - child->idx_offset - child->length;
                child->parent = parent;
                // Give ownership of child pointer to parent and should also free the current node.
                assert(parent->children.count(child->token));
                assert(parent->children[child->token].get() == node);
                parent->children[child->token] = std::move(node->children[token]);
                // Replace active node with child node.
                _active_nodes[seq_id][i] = child;
            } else {
                //std::cout << "case 4" << std::endl;
                // The child has length > 1. If we append the new token to the current suffix, then
                // it still does not reach the child node. In this case, we keep both nodes but
                // extend the length of the current node by 1 into the child node.
                assert(child->length > 1);
                node->length += 1;
                node->idx_in_seq[seq_id] =
                    static_cast<int>(_seqs[seq_id].size()) - node->idx_offset - node->length;
                child->idx_offset += 1;
                child->length -= 1;
                // The child node's first token should be updated to its second token.
                auto elem = child->idx_in_seq.begin();  // Take an arbitrary reference sequence
                assert(elem != child->idx_in_seq.end());
                child->token = _seqs[elem->first][elem->second + child->idx_offset];
                if (child->token != token) {
                    node->children[child->token] = std::move(node->children[token]);
                    node->children.erase(token);
                }
            }
        }
        else {
            // There is a child for the new token, and should move the active node into that child.
            if (child->length == 1) {
                //std::cout << "case 5" << std::endl;
                // The child node has length 1, just update the active node pointer to it.
                child->count += 1;
                child->idx_in_seq[seq_id] =
                    static_cast<int>(_seqs[seq_id].size()) - child->idx_offset - 1;
                _active_nodes[seq_id][i] = child;
            } else {
                //std::cout << "case 6" << std::endl;
                // The child node has length > 1. If we extend the current suffix into it, then it
                // must be split into a segment of length 1 and another segment with the remainder.
                assert(child->length > 1);
                Node* new_child = new Node();
                new_child->token = token;
                new_child->count = child->count + 1;
                new_child->parent = node;
                new_child->idx_in_seq = child->idx_in_seq;  // TODO: optimize
                new_child->idx_offset = child->idx_offset;
                new_child->idx_in_seq[seq_id] =
                    static_cast<int>(_seqs[seq_id].size()) - new_child->idx_offset - 1;
                new_child->length = 1;
                // The child node's first token should be updated to its second token.
                auto elem = child->idx_in_seq.begin();  // Take an arbitrary reference sequence
                child->token = _seqs[elem->first][elem->second + child->idx_offset + 1];
                new_child->children[child->token] = std::move(node->children[token]);
                node->children[token].reset(new_child);
                child->parent = new_child;
                child->idx_offset += 1;
                child->length -= 1;
                _active_nodes[seq_id][i] = new_child;
            }
        }
    }
    //check_integrity();
}

// Extend a new or existing sequence.
void SuffixTree::extend(int seq_id, const std::vector<int>& tokens) {
    for (int token : tokens) {
        append(seq_id, token);
    }
}

// Remove an existing sequence.
void SuffixTree::remove(int seq_id) {
    const std::vector<int>& seq = _seqs[seq_id];
    // Loop through all suffix starting indices.
    for (int start = 0; start < seq.size(); start++) {
        Node *node = _root.get();
        node->count--;
        int idx = start;
        // Loop through the nodes for this suffix.
        while (idx < seq.size()) {
            int token = seq[idx];
            //std::cout << token << std::endl;
            auto it = node->children.find(token);
            if (it == node->children.end()) {
                //std::cout << "Token not found in children: " << token << std::endl;
                break;
            }
            //std::cout << "Found token in children: " << token << std::endl;
            Node* child = it->second.get();
            assert(child->count > 0);
            child->count--;
            if (child->count == 0) {
                //std::cout << "Erasing child node for token: " << token << std::endl;
                //std::cout << node->children.size() << std::endl;
                node->children.erase(token);
                break;
            }
            if (child->idx_in_seq.count(seq_id)) {
                child->idx_in_seq.erase(seq_id);
            }
            idx += child->length;
            // if (node->count == child->count) {
            //     // Merge node into child.
            //     child->length += node->length;
            //     child->idx_in_seq = std::move(node->idx_in_seq);
            //     child->idx_offset = node->idx_offset;
            //     child->parent = node->parent;
            //     child->parent->children[token] = std::move(node->children[token]);
            // }
            node = child;
        }
        if (node != _root.get() && node->children.size() == 1) {
            auto& [token, child] = *node->children.begin();
            if (node->count == child->count) {
                // Merge node into child.
                //std::cout << "Merging node into child for token: " << token << std::endl;
                child->token = node->token;
                child->length += node->length;
                child->idx_in_seq = std::move(node->idx_in_seq);
                child->idx_offset = node->idx_offset;
                child->parent = node->parent;
                node = child.release();
                //assert(node->count > 0);
                //assert(node->idx_in_seq.size() > 0);
                node->parent->children[node->token].reset(node);
            }
        }
    }
    _seqs.erase(seq_id);
    _active_nodes.erase(seq_id);
    check_integrity();
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

bool SuffixTree::check_integrity() {
    std::queue<Node*> queue;
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        check_integrity(node);
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    for (int seq_id = 0; seq_id < _seqs.size(); seq_id++) {
        const std::vector<int>& seq = _seqs[seq_id];
        // Loop through all suffix starting indices.
        for (int start = 0; start < seq.size(); start++) {
            Node *node = _root.get();
            int idx = start;
            // Loop through the nodes for this suffix.
            while (idx < seq.size()) {
                int token = seq[idx];
                auto it = node->children.find(token);
                if (it == node->children.end()) {
                    break;
                }
                Node* child = it->second.get();
                assert(child->count > 0);
                assert(child->idx_in_seq.count(seq_id));
                node = child;
            }
        }
    }
    return true;
}

bool SuffixTree::check_integrity(Node* node) {
    int children_count = 0;
    for (const auto& [token, child] : node->children) {
        // Do all my children have me as their parent?
        assert(child->parent == node);
        children_count++;
    }
    // Is my counter at least the sum of my childrens' counters?
    assert(children_count <= node->count);
    if (node == _root.get()) {
        // Root node can stop here after some simple checks.
        assert(node->count >= 0);
        assert(node->parent == nullptr);
        assert(node->length == 0);
        assert(node->idx_in_seq.empty());
        assert(node->idx_offset == 0);
        return true;
    }
    // Is my length positive? Otherwise, I shouldn't exist.
    assert(node->length > 0);
    // Is my count positive? Otherwise, I shouldn't exist.
    assert(node->count > 0);
    // Are all my children's counts less than mine?
    for (const auto& [token, child] : node->children) {
        assert(child->count < node->count);
    }
    // Find what my first token is.
    auto elem = node->idx_in_seq.begin();  // Take an arbitrary element
    assert(elem != node->idx_in_seq.end());
    assert(_seqs.find(elem->first) != _seqs.end());
    assert(elem->second + node->idx_offset + node->length <= static_cast<int>(_seqs[elem->first].size()));
    int token = _seqs[elem->first][elem->second + node->idx_offset];
    // Check I am my parent's child.
    assert(node->parent->children.find(token) != node->parent->children.end());
    assert(node->parent->children[token].get() == node);
    // Check all my sequence references are correct.
    for (int i = 0; i < node->length; ++i) {
        int tok = _seqs[elem->first][elem->second + node->idx_offset + i];
        for (const auto& [seq_id, idx] : node->idx_in_seq) {
            assert(_seqs.find(seq_id) != _seqs.end());
            assert(idx + node->idx_offset + i < static_cast<int>(_seqs[seq_id].size()));
            assert(_seqs[seq_id][idx + node->idx_offset + i] == tok);
        }
    }
    return true;
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
        auto elem = node->idx_in_seq.begin();  // Take an arbitrary element
        if (_seqs[elem->first][elem->second + node->idx_offset + idx] != c) {
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
            auto elem = node->idx_in_seq.begin();  // Take an arbitrary element
            int token = _seqs[elem->first][elem->second + node->idx_offset + idx];
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
            auto elem = item.node->idx_in_seq.begin();  // Take an arbitrary element
            int token = _seqs[elem->first][elem->second + item.node->idx_offset + item.idx];
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
