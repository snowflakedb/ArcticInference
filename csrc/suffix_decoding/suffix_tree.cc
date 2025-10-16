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
#include <string>
#include <unordered_map>
#include <vector>
#include "suffix_tree.h"

#define CHECK_OR_RETURN(cond) \
    if (!(cond)) return "Integrity check failed (line " + std::to_string(__LINE__) + "): " + #cond;

SuffixTree::SuffixTree(int max_depth)
    : _max_depth(max_depth), _root(new Node()) {
}

void _remove_from_siblings(Node* node) {
    // Remove a node from the siblings and groups linked lists.
    assert(node->parent);  // Should only be called on non-root nodes.
    // Take care of the groups linked list.
    Group* group = node->group.get();
    if (group->head == node) {
        if (node->next_sibling && node->next_sibling->count == node->count) {
            // There are other nodes in the same group, update its head and
            // remove the node from the group.
            group->head = node->next_sibling;
            node->group.reset();
        } else {
            // Otherwise, the node is the only member of its group. Remove the
            // group together with the node.
            if (group->prev) {
                group->prev->next = group->next;
            }
            if (group->next) {
                group->next->prev = group->prev;
            }
            group->prev = group->next = nullptr;
        }
    } else {
        // The node is not the head of its group, just remove it.
        node->group.reset();
    }
    // Take care of the siblings linked list.
    if (node->next_sibling) {
        node->next_sibling->prev_sibling = node->prev_sibling;
    } else {
        node->parent->tail_child = node->prev_sibling;
    }
    if (node->prev_sibling) {
        node->prev_sibling->next_sibling = node->next_sibling;
    } else {
        node->parent->head_child = node->next_sibling;
    }
    node->prev_sibling = node->next_sibling = nullptr;
}

void _insert_into_siblings_before(Node* node, Node* other) {
    // Insert a node before another in the siblings and groups linked lists.
    assert(node->parent);  // Should only be called on non-root nodes.
    assert(node->parent == other->parent);  // Should be siblings.
    // Take care of the siblings linked list.
    if (other->prev_sibling) {
        other->prev_sibling->next_sibling = node;
    } else {
        node->parent->head_child = node;
    }
    node->next_sibling = other;
    node->prev_sibling = other->prev_sibling;
    other->prev_sibling = node;
    // Take care of the groups linked list.
    Node* prev_sibling = node->prev_sibling;
    if (prev_sibling && node->count == prev_sibling->count) {
        // If the previous sibling has the same count, just join its group.
        node->group = prev_sibling->group;  // std::shared_ptr assignment
    } else if (node->count == other->count) {
        // Previous sibling has different count, but next sibling has the same
        // count. Join as the head of the next sibling's group.
        node->group = other->group;  // std::shared_ptr assignment
        node->group->head = node;
    } else {
        // Previous and next siblings both have different counts. The node
        // belongs in a group by itself.
        Group* group = node->group.get();
        if (!group) {
            // The node does not come with a group, create a new one.
            group = new Group();
            group->head = node;
            node->group.reset(group);  // std::shared_ptr assignment
        }
        assert(group->head == node && !group->next && !group->prev);
        // Insert the node's group into the linked list.
        if (prev_sibling) {
            group->prev = prev_sibling->group.get();
            group->prev->next = group;
        }
        group->next = other->group.get();
        group->next->prev = group;
    }
}

void _insert_into_siblings_after(Node* node, Node* other) {
    // Insert a node after another in the siblings and groups linked lists.
    assert(node->parent);  // Should only be called on non-root nodes.
    assert(node->parent == other->parent);  // Should be siblings.
    // Take care of the siblings linked list.
    if (other->next_sibling) {
        other->next_sibling->prev_sibling = node;
    } else {
        node->parent->tail_child = node;
    }
    node->prev_sibling = other;
    node->next_sibling = other->next_sibling;
    other->next_sibling = node;
    // Take care of the groups linked list.
    Node* next_sibling = node->next_sibling;
    if (next_sibling && node->count == next_sibling->count) {
        // If the next sibling has the same count, join its group and maybe
        // update the head of the group.
        node->group = next_sibling->group;  // std::shared_ptr assignment
        if (node->group->head == next_sibling) {
            node->group->head = node;
        }
    } else if (node->count == other->count) {
        // Next sibling has different count, but previous sibling has the same
        // count. Join as the tail of the previous sibling's group.
        node->group = other->group;  // std::shared_ptr assignment
    } else {
        // Previous and next siblings both have different counts. The node
        // belongs in a group by itself.
        Group* group = node->group.get();
        if (!group) {
            // The node does not come with a group, create a new one.
            group = new Group();
            group->head = node;
            node->group.reset(group);  // std::shared_ptr assignment
        }
        assert(group->head == node && !group->next && !group->prev);
        // Insert the node's group into the linked list.
        if (next_sibling) {
            group->next = next_sibling->group.get();
            group->next->prev = group;
        }
        group->prev = other->group.get();
        group->prev->next = group;
    }
}

void _replace_in_siblings(Node* old_node, Node* new_node) {
    // Replace a node with another in the siblings and groups linked lists.
    assert(old_node->count == new_node->count);  // Should have the same count.
    assert(old_node->parent);  // Should only be called on non-root nodes.
    // Take care of the siblings linked list.
    if (old_node->next_sibling) {
        old_node->next_sibling->prev_sibling = new_node;
    } else {
        old_node->parent->tail_child = new_node;
    }
    if (old_node->prev_sibling) {
        old_node->prev_sibling->next_sibling = new_node;
    } else {
        old_node->parent->head_child = new_node;
    }
    new_node->prev_sibling = old_node->prev_sibling;
    new_node->next_sibling = old_node->next_sibling;
    old_node->prev_sibling = old_node->next_sibling = nullptr;
    // Take care of the groups linked list.
    Group* group = old_node->group.get();
    if (group->head == old_node) {
        group->head = new_node;
    }
    new_node->group = old_node->group;  // std::shared_ptr assignment
    old_node->group.reset();
}

void _increment_count(Node* node) {
    // Increment the count of a node by 1, and update its position in the
    // sibling and group linked lists if necessary.
    if (!node->parent) {
        // Root node has no siblings, update its count and return.
        node->count += 1;
        return;
    }
    if (!node->prev_sibling || node->prev_sibling->count > node->count + 1) {
        // The node does not need to move, and will not join the previous group
        // after its count is incremented.
        assert(node->group->head == node);
        if (!node->next_sibling || node->next_sibling->count < node->count) {
            // The node should be the only member of its group and will not
            // join the previous group, so just update its count.
            assert(node->group.use_count() == 1);
            node->count += 1;
            // std::cout << "Increment 1" << std::endl;
        } else {
            // The node will split off from its current group to a new group.
            assert(node->next_sibling->count == node->count);
            Group* orig_group = node->group.get();
            orig_group->head = node->next_sibling;
            Group* new_group = new Group();
            new_group->head = node;
            new_group->next = orig_group;
            if (orig_group->prev) {
                new_group->prev = orig_group->prev;
                new_group->prev->next = new_group;
            }
            orig_group->prev = new_group;
            node->group.reset(new_group);  // std::shared_ptr assignment
            node->count += 1;
            // std::cout << "Increment 2" << std::endl;
        }
    } else {
        // The node needs to be moved.
        assert(node->prev_sibling->count >= node->count);
        Node* other = node->prev_sibling->group->head;
        _remove_from_siblings(node);
        node->count += 1;
        _insert_into_siblings_before(node, other);
        // std::cout << "Increment 3" << std::endl;
    }
}

// Append a new element to a new or existing sequence.
void SuffixTree::append(int seq_id, int token) {
    // Initialize the sequence if it doesn't exist.
    _seqs.try_emplace(seq_id);
    _active_nodes.try_emplace(seq_id);

    // Insert a new active node at the root.
    _active_nodes[seq_id].push_back(_root.get());
    _root->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
    _root->count += 1;

    // Ensure the number of active nodes doesn't exceed max_depth.
    if (_active_nodes[seq_id].size() > static_cast<size_t>(_max_depth)) {
        _active_nodes[seq_id].pop_front();
    }
    _seqs[seq_id].push_back(token);
    
    // Iterate over all active nodes for this sequence.
    for (size_t i = 0; i < _active_nodes[seq_id].size(); ++i) {
        Node* node = _active_nodes[seq_id][i];
        Node* child = nullptr;
        if (node->children.contains(token)) {
            child = node->children[token].get();
        }

        assert(node->endpoints.contains(seq_id));
        assert(node->endpoints[seq_id] == _seqs[seq_id].size() - 1);

        if (child == nullptr) {
            // No existing child node for the new token.
            if (node->count == 1 && node != _root.get()) {
                // The active node has count = 1, which means the only suffix that ends here is the
                // one that's being extended right now. Then this node should be a leaf node, and
                // we can simply extend the length of this node.
                assert(node->children.empty());
                assert(node->ref_seq == seq_id);
                node->length += 1;
                node->endpoints[seq_id] += 1;
            } else {
                // Either this is the root node, or the current suffix is not the only one that
                // ends here. Either case, we need to extend the current suffix into a new child.
                Node* new_child = new Node();
                new_child->token = token;
                new_child->parent = node;
                new_child->count = 1;
                new_child->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                new_child->ref_seq = seq_id;
                new_child->ref_idx = static_cast<int>(_seqs[seq_id].size()) - 1;
                new_child->length = 1;
                node->children.emplace(token, new_child);
                node->endpoints.erase(seq_id);
                // Maintain doubly-linked list of children and groups.
                if (node->tail_child == nullptr) {
                    // This should be the first child being added.
                    assert(node->head_child == nullptr && node->children.size() == 1);
                    node->head_child = node->tail_child = new_child;
                    // Create a new group for this child.
                    Group* group = new Group();
                    group->head = new_child;
                    new_child->group.reset(group);  // std::shared_ptr assignment
                } else if (node->tail_child->count == 1) {
                    // Prefer to join as the head of the last group if possible.
                    _insert_into_siblings_before(new_child, node->tail_child->group->head);
                } else {
                    // Otherwise, just insert as the tail child.
                    assert(node->tail_child->count > 1);
                    _insert_into_siblings_after(new_child, node->tail_child);
                }
                // Update the active node to the new child.
                _active_nodes[seq_id][i] = new_child;
            }
        } else if (node->count == child->count + 1 && node != _root.get()) {
            // The active node has a child for the new token, and the child's count is exactly one
            // fewer than the active node's count. Since the suffix for the active node ends here,
            // that means all other suffixes that pass through this node must go to that child.
            assert(node->children.size() == 1);  // The active node should have only one child.
            assert(node->endpoints.size() == 1);  // Only the current suffix should end here.
            if (child->length == 1) {
                // The child only has length 1. If we append the new token to the current suffix,
                // then it will perfectly overlap with the child. In this case, we should just fuse
                // the current suffix into the child and eliminate the current node.
                Node* parent = node->parent;
                // Update child to take the place of the current node.
                child->token = node->token;
                child->count += 1;  // Current suffix extends into the child.
                child->length = node->length + 1;
                child->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                child->ref_seq = seq_id;
                child->ref_idx = static_cast<int>(_seqs[seq_id].size()) - child->length;
                child->parent = parent;
                // Child now has the same count as the node, so it can simply take the node's place
                // in the siblings and groups linked lists.
                _replace_in_siblings(node, child);
                // Give ownership of child pointer to parent and should also free the current node.
                assert(parent->children.contains(child->token));
                assert(parent->children[child->token].get() == node);
                Node* tmp = node->children[token].release();
                parent->children[child->token].reset(tmp);
                node = nullptr;
                // Replace active node with child node.
                _active_nodes[seq_id][i] = child;
            } else {
                // The child has length > 1. If we append the new token to the current suffix, then
                // it still does not reach the child node. In this case, we keep both nodes but
                // extend the length of the current node by 1 into the child node.
                node->length += 1;
                node->endpoints[seq_id] += 1;
                node->ref_seq = seq_id;
                node->ref_idx = static_cast<int>(_seqs[seq_id].size()) - node->length;
                child->length -= 1;
                child->ref_idx += 1;
                // The child node's first token should be updated to its second token.
                child->token = _seqs[child->ref_seq][child->ref_idx];
                if (child->token != token) {
                    Node* tmp = node->children[token].release();
                    node->children.emplace(child->token, tmp);
                    node->children.erase(token);
                }
            }
        } else {
            // There is a child for the new token, and should move the active node into that child.
            if (child->length == 1) {
                // The child node has length 1, just update the active node pointer to it.
                node->endpoints.erase(seq_id);
                child->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                child->ref_seq = seq_id;
                child->ref_idx = static_cast<int>(_seqs[seq_id].size()) - 1;
                // Increment the child count and update linked lists.
                _increment_count(child);
                // Replace active node with child node.
                _active_nodes[seq_id][i] = child;
            } else {
                // The child node has length > 1. If we extend the current suffix into it, then it
                // must be split into a segment of length 1 and another segment with the remainder.
                Node* new_node = new Node();
                new_node->token = token;
                new_node->count = child->count;
                new_node->parent = node;
                new_node->length = 1;
                new_node->endpoints[seq_id] = static_cast<int>(_seqs[seq_id].size());
                new_node->ref_seq = seq_id;
                new_node->ref_idx = static_cast<int>(_seqs[seq_id].size()) - new_node->length;
                // Replace the child with the new node in the linked lists.
                _replace_in_siblings(child, new_node);
                // The child node's first token should be updated to its second token.
                child->token = _seqs[child->ref_seq][child->ref_idx + 1];
                Node* tmp = node->children[token].release();
                new_node->children.emplace(child->token, tmp);
                node->children[token].reset(new_node);
                node->endpoints.erase(seq_id);
                child->parent = new_node;
                child->length -= 1;
                child->ref_idx += 1;
                // Create a new group for the child node.
                new_node->head_child = new_node->tail_child = child;
                Group* group = new Group();
                group->head = child;
                child->group.reset(group);  // std::shared_ptr assignment
                // Increment the new node count and update linked lists.
                _increment_count(new_node);
                // Update active node.
                _active_nodes[seq_id][i] = new_node;
            }
        }
    }
}

// Extend a new or existing sequence.
void SuffixTree::extend(int seq_id, std::span<const int32_t> tokens) {
    for (int token : tokens) {
        append(seq_id, token);
    }
}

// Remove an existing sequence.
void SuffixTree::remove(int seq_id) {
    const std::vector<int>& seq = _seqs[seq_id];
    std::vector<Node*> path;  // Declare here to avoid repeated allocations.
    // Loop through all suffix starting indices.
    for (int start = 0; start < seq.size(); start++) {
        Node *node = _root.get();
        node->count--;
        int idx = start;
        path.clear();
        // Loop through the nodes for this suffix.
        while (idx < seq.size()) {
            int token = seq[idx];
            if (!node->children.contains(token)) {
                break;
            }
            Node* child = node->children[token].get();
            assert(child->count > 0);
            child->count--;
            if (child->count == 0) {
                node->children.erase(token);
                break;
            }
            if (child->endpoints.contains(seq_id)) {
                child->endpoints.erase(seq_id);
            }
            idx += child->length;
            node = child;
            path.push_back(node);
        }
        // The last visited node may be mergeable with its child.
        if (node != _root.get() && node->children.size() == 1) {
            const auto& it = *node->children.begin();
            std::unique_ptr<Node>& child_uptr = node->children[it.first];
            if (node->count == child_uptr->count) {
                // Merge node into child.
                child_uptr->token = node->token;
                child_uptr->length += node->length;
                child_uptr->ref_idx -= node->length;
                child_uptr->parent = node->parent;
                path.back() = node = child_uptr.release();
                node->parent->children[node->token].reset(node);
            }
        }
        // ref_seq and ref_idx of all nodes in the path may need to be updated.
        // 1. Go to an arbitrary leaf to get its endpoints.
        Node* leaf = node;
        int distance = 0;  // Distance from node to leaf.
        while (!leaf->children.empty()) {
            leaf = (*leaf->children.begin()).second.get();
            distance += leaf->length;
        }
        // 2. Pick an arbitrary endpoint for the reference sequence and index.
        if (leaf->endpoints.empty() || leaf->endpoints.contains(seq_id)) {
            // Still need to visit this leaf later when removing this sequence.
            // We can skip updating the refs until the next time it's visited.
            continue;
        }
        const auto& ref = *leaf->endpoints.begin();
        // 3. Go back up the path to update all nodes' refs.
        int32_t ref_seq = ref.first;
        int32_t ref_idx = ref.second - distance;
        while (!path.empty()) {
            Node* n = path.back();
            path.pop_back();
            ref_idx -= n->length;
            if (n->ref_seq == seq_id) {
                n->ref_seq = ref_seq;
                n->ref_idx = ref_idx;
            }
        }
    }
    _seqs.erase(seq_id);
    _active_nodes.erase(seq_id);
}

Draft SuffixTree::speculate(std::span<const int32_t> context,
                            int max_spec_tokens,
                            float max_spec_factor,
                            float max_spec_offset,
                            float min_token_prob,
                            bool use_tree_spec) {
    Draft best_draft;
    for (int match_len = 1; match_len < context.size(); match_len++) {
        auto[node, idx] = _match_context(
            context.subspan(context.size() - match_len, match_len));
        if (node == nullptr) {
            break;
        }
        int max_tokens = std::min(max_spec_tokens,
                                  static_cast<int>(match_len * max_spec_factor
                                                   + max_spec_offset + 1e-6));
        max_tokens = std::max(max_tokens, 0);
        Draft draft;
        if (use_tree_spec) {
            draft = _speculate_tree(node, idx, max_tokens, min_token_prob);
        } else {
            draft = _speculate_path(node, idx, max_tokens, min_token_prob);
        }
        if (draft.score >= best_draft.score) {
            best_draft = std::move(draft);
            best_draft.match_len = match_len;
        }
    }
    return best_draft;
}

std::string SuffixTree::check_integrity() {
    // 1. Check structural integrity of all nodes.
    std::queue<Node*> queue;
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        std::string ret = _check_node_integrity(node);
        if (!ret.empty()) {
            return ret;
        }
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    // 2. Check all sequences are represented in the tree.
    std::unordered_map<Node*, int64_t> visit_count;
    for (int seq_id = 0; seq_id < _seqs.size(); seq_id++) {
        const std::vector<int>& seq = _seqs[seq_id];
        // Loop through all suffix starting indices.
        for (int start = 0; start < seq.size(); start++) {
            int idx = start;
            // Traverse the tree along this suffix.
            Node* node = _root.get();
            visit_count[node]++;
            while (idx < seq.size() && idx - start < _max_depth) {
                // There should be a child for the next token.
                CHECK_OR_RETURN(node->children.contains(seq[idx]));
                node = node->children[seq[idx]].get();
                visit_count[node]++;
                // Sequence should not end in the middle of a node.
                CHECK_OR_RETURN(idx + node->length <= seq.size());
                for (int i = 0; i < node->length; ++i) {
                    int ref_seq = node->ref_seq;
                    int ref_idx = node->ref_idx + i;
                    // Reference tokens should match sequence tokens.
                    CHECK_OR_RETURN(seq[idx + i] == _seqs[ref_seq][ref_idx]);
                }
                idx += node->length;
            }
            // The last node on this path should have an endpoint.
            CHECK_OR_RETURN(node->endpoints.contains(seq_id));
        }
    }
    // 3. Check all nodes were visited the correct number of times.
    assert(queue.empty());
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        // The visit count should match the node count.
        CHECK_OR_RETURN(node->count == visit_count[node]);
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    return "";
}

std::string SuffixTree::_check_node_integrity(Node* node) {
    int64_t children_count = 0;
    for (const auto& [token, child] : node->children) {
        // All children should have the correct parent pointer.
        CHECK_OR_RETURN(child->parent == node);
        children_count++;
    }
    // Node count should be at least the sum of all children counts.
    CHECK_OR_RETURN(children_count <= node->count);
    if (node == _root.get()) {
        // Root node should not contain any tokens, do some basic checks.
        CHECK_OR_RETURN(node->count >= 0);
        CHECK_OR_RETURN(node->parent == nullptr);
        CHECK_OR_RETURN(node->length == 0);
        CHECK_OR_RETURN(node->endpoints.empty());
        CHECK_OR_RETURN(node->ref_idx == -1);
    } else {
        // Node length should be positive.
        CHECK_OR_RETURN(node->length > 0);
        // Node count should be positive.
        CHECK_OR_RETURN(node->count > 0);
        // Each child count should be strictly less than the node count. Otherwise, the node and
        // the child should have been merged into a single node.
        for (const auto& [token, child] : node->children) {
            CHECK_OR_RETURN(child->count < node->count);
        }
        // Internal nodes must have a valid reference sequence and index.
        CHECK_OR_RETURN(_seqs.count(node->ref_seq));
        CHECK_OR_RETURN(node->ref_idx >= 0);
        CHECK_OR_RETURN(node->ref_idx + node->length <= _seqs[node->ref_seq].size());
        // Check the first token of the node is correct.
        CHECK_OR_RETURN(node->token == _seqs[node->ref_seq][node->ref_idx]);
        // Check the node is in its parent's children map.
        CHECK_OR_RETURN(node->parent->children.contains(node->token));
        CHECK_OR_RETURN(node->parent->children[node->token].get() == node);
        // Check all endpoint references are correct.
        for (auto [seq_id, end_idx] : node->endpoints) {
            // Endpoint should refer to a sequence id that exists.
            CHECK_OR_RETURN(_seqs.count(seq_id));
            // Endpoint index should be within the sequence length.
            CHECK_OR_RETURN(end_idx > 0 && end_idx <= _seqs[seq_id].size());
            // Check all tokens from the start of the suffix to the endpoint.
            Node* n = node;
            int idx = end_idx;
            // Walk up the tree and check all tokens agree with the suffix ending at this endpoint.
            do {
                // Check the index in the sequence is not underflowed.
                CHECK_OR_RETURN(n->length <= idx);
                idx -= n->length;
                for (int i = 0; i < n->length; ++i) {
                    int tok = _seqs[n->ref_seq][n->ref_idx + i];
                    // Check each token in this node agrees with the sequence.
                    CHECK_OR_RETURN(_seqs[seq_id][idx + i] == tok);
                }
                n = n->parent;
            } while (n != nullptr);
        }
    }
    // Check siblings list integrity.
    if (!node->head_child && !node->tail_child) {
        CHECK_OR_RETURN(node->children.empty());
    } else {
        // If there is a child then there must be both a head and a tail child.
        CHECK_OR_RETURN(node->head_child && node->tail_child);
        // Check head and tail child pointers are correct.
        CHECK_OR_RETURN(node->head_child->prev_sibling == nullptr);
        CHECK_OR_RETURN(node->tail_child->next_sibling == nullptr);
        // Check all children are in the siblings linked list.
        int count = 0;
        Node* child = node->head_child;
        Node* prev_child = nullptr;
        while (child != nullptr) {
            count++;
            // Check the child is in the children map.
            CHECK_OR_RETURN(node->children.contains(child->token));
            // Check the group pointer is valid.
            CHECK_OR_RETURN(child->group != nullptr);
            if (prev_child) {
                // Check the siblings are ordered in nonincreasing count.
                CHECK_OR_RETURN(child->count <= prev_child->count);
                // Check the sibling pointers are correct.
                CHECK_OR_RETURN(child->prev_sibling == prev_child);
                CHECK_OR_RETURN(prev_child->next_sibling == child);
                // Check the group pointers are correct.
                if (child->count == prev_child->count) {
                    // If the next sibling has the same count, they should be in the same group.
                    CHECK_OR_RETURN(child->group == prev_child->group);
                } else {
                    // Otherwise, they should be in different groups.
                    CHECK_OR_RETURN(child->group != prev_child->group);
                    // The child should be the head of its group.
                    CHECK_OR_RETURN(child->group->head == child);
                    // Check group pointers are correct.
                    CHECK_OR_RETURN(child->group->prev == prev_child->group.get());
                    CHECK_OR_RETURN(prev_child->group->next == child->group.get());

                }
            } else {
                CHECK_OR_RETURN(child == node->head_child);
            }
            prev_child = child;
            child = child->next_sibling;
        }
        // Check the last child reached is the tail child.
        CHECK_OR_RETURN(prev_child == node->tail_child);
        // Check the number of children matches the size of the children map.
        CHECK_OR_RETURN(count == node->children.size());
    }
    return "";
}

std::pair<Node*, int> SuffixTree::_match_context(std::span<const int32_t> context) {
    Node* node = _root.get();
    int idx = 0;
    for (int i = 0; i < context.size(); i++) {
        int c = context[i];
        if (idx >= node->length) {
            if (!node->children.contains(c)) {
                return {nullptr, -1};
            }
            node = node->children[c].get();
            idx = 0;
        }
        assert(idx < node->length);
        if (_seqs[node->ref_seq][node->ref_idx + idx] != c) {
            return {nullptr, -1};
        }
        idx++;
    }
    return {node, idx};
}

Draft SuffixTree::_speculate_path(Node* node, int idx,
                                  int max_spec_tokens,
                                  float min_token_prob) {
    Draft ret;
    float prob = 1.0f;
    while (ret.token_ids.size() < max_spec_tokens && prob >= min_token_prob) {
        if (idx < node->length) {
            // Use previous token index as parent; if none, mark as -1.
            ret.parents.push_back(static_cast<int>(ret.token_ids.size()) - 1);
            int token = _seqs[node->ref_seq][node->ref_idx + idx];
            ret.token_ids.push_back(token);
            ret.probs.push_back(prob);
            ret.score += prob;
            idx++;
        } else {
            Node* child = node->head_child;
            if (child == nullptr) {
                break;
            }
            int64_t count = child->count;
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
    int parent;   // index in the draft token list; -1 if none.

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

// Get a draft token tree using a priority queue.
Draft SuffixTree::_speculate_tree(Node* node, int idx,
                                  int max_spec_tokens,
                                  float min_token_prob) {
    Draft ret;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapItemCompare> queue;
    queue.emplace(1.0, node, idx, -1);
    while (ret.token_ids.size() < max_spec_tokens && !queue.empty()) {
        HeapItem item = queue.top();
        queue.pop();
        if (item.idx < item.node->length) {
            int token = _seqs[item.node->ref_seq][item.node->ref_idx + item.idx];
            ret.token_ids.push_back(token);
            ret.parents.push_back(item.parent);
            ret.probs.push_back(item.prob);
            ret.score += item.prob;
            queue.emplace(item.prob, item.node, item.idx + 1,
                          static_cast<int>(ret.token_ids.size()) - 1);
        } else {
            for (const auto& kv : item.node->children) {
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

size_t SuffixTree::estimate_memory() const {
    size_t total = sizeof(*this);
    std::vector<Node*> stack;
    stack.push_back(_root.get());
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        total += node->memory_usage();
        for (const auto& [token, child] : node->children) {
            stack.push_back(child.get());
        }
    }
    for (const auto& [seq_id, seq] : _seqs) {
        total += sizeof(decltype(seq)::value_type) * seq.capacity();
    }
    for (const auto& [seq_id, active_nodes] : _active_nodes) {
        total += sizeof(decltype(active_nodes)::value_type) * active_nodes.size();
    }
    return total;
}
