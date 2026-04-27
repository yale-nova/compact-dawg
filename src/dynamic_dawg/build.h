#pragma once

#include "../dynamic_dawg.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

inline uint32_t DynamicDawg::Finalize(const Node &node)
{
    if (node.edges.empty())
        return BUILD_TERMINAL_NODE;

    auto it = memo_.find(node);
    if (it != memo_.end())
        return it->second;

    uint32_t start_idx = temp_edges_.size();
    for (size_t i = 0; i < node.edges.size(); ++i) {
        bool is_last = (i == node.edges.size() - 1);
        temp_edges_.push_back({node.edges[i].label, node.edges[i].target, is_last, 1});
    }

    memo_.emplace(node, start_idx);
    return start_idx;
}

inline void DynamicDawg::RunPathCompression()
{
    if (temp_edges_.empty())
        return;

    std::vector<uint32_t> node_starts;
    for (size_t i = 0; i < temp_edges_.size(); ++i) {
        if (i == 0 || temp_edges_[i - 1].is_last) {
            node_starts.push_back(static_cast<uint32_t>(i));
        }
    }

    std::unordered_set<uint32_t> node_starts_set(node_starts.begin(), node_starts.end());

    std::unordered_map<uint32_t, uint32_t> in_degree;
    for (const auto &e : temp_edges_) {
        if (e.target != BUILD_TERMINAL_NODE && node_starts_set.count(e.target)) {
            in_degree[e.target]++;
        }
    }

    std::unordered_map<uint32_t, uint32_t> degree;
    for (size_t k = 0; k < node_starts.size(); ++k) {
        uint32_t s = node_starts[k];
        uint32_t next = (k + 1 < node_starts.size()) ? node_starts[k + 1]
                                                      : static_cast<uint32_t>(temp_edges_.size());
        degree[s] = next - s;
    }

    std::unordered_set<uint32_t> compressible;
    for (uint32_t s : node_starts) {
        if (s != root_index_ && degree[s] == 1 && in_degree[s] == 1) {
            compressible.insert(s);
        }
    }

    if (compressible.empty())
        return;

    std::vector<TempEdge> new_edges;
    new_edges.reserve(temp_edges_.size() - compressible.size());
    std::unordered_map<uint32_t, uint32_t> old_to_new;

    for (uint32_t s : node_starts) {
        if (compressible.count(s))
            continue;

        old_to_new[s] = static_cast<uint32_t>(new_edges.size());

        uint32_t end_of_node =
            (degree.count(s) ? s + degree[s] : static_cast<uint32_t>(temp_edges_.size()));
        for (uint32_t i = s; i < end_of_node; ++i) {
            std::string concat_label = temp_edges_[i].label;
            uint32_t target = temp_edges_[i].target;
            uint32_t segment_count = temp_edges_[i].segment_count;

            while (target != BUILD_TERMINAL_NODE && compressible.count(target)) {
                concat_label += temp_edges_[target].label;
                segment_count += temp_edges_[target].segment_count;
                target = temp_edges_[target].target;
            }

            new_edges.push_back(
                {std::move(concat_label), target, temp_edges_[i].is_last, segment_count});
        }
    }

    for (auto &e : new_edges) {
        if (e.target != BUILD_TERMINAL_NODE) {
            auto it = old_to_new.find(e.target);
            if (it != old_to_new.end()) {
                e.target = it->second;
            }
        }
    }

    root_index_ = old_to_new.at(root_index_);
    temp_edges_ = std::move(new_edges);
}

inline void DynamicDawg::Insert(const std::string &bitstring)
{
    if (!plan_.valid()) {
        throw std::logic_error("Insert: SegmentPlan must be valid before inserting keys");
    }
    if (bitstring.size() != plan_.total_bits) {
        throw std::invalid_argument("Insert: bitstring length must match plan.total_bits");
    }

    if (active_path_.empty()) {
        active_path_.resize(plan_.depth() + 1);
    }
    if (last_key_.empty()) {
        last_key_.resize(plan_.depth());
        for (size_t i = 0; i < last_key_.size(); ++i) {
            AssignSegment(&last_key_[i], bitstring, i);
        }
        return;
    }

    size_t lcp = 0;
    while (lcp < last_key_.size() && SegmentEquals(bitstring, lcp, last_key_[lcp])) {
        lcp++;
    }

    for (size_t i = last_key_.size(); i > lcp; --i) {
        uint32_t offset = Finalize(active_path_[i]);
        active_path_[i - 1].edges.push_back({last_key_[i - 1], offset});
        active_path_[i].edges.clear();
    }

    for (size_t i = lcp; i < last_key_.size(); ++i) {
        AssignSegment(&last_key_[i], bitstring, i);
    }
}

inline void DynamicDawg::Finish(bool print_timings)
{
    if (last_key_.empty())
        return;

    auto start_finish = std::chrono::high_resolution_clock::now();

    for (size_t i = last_key_.size(); i > 0; --i) {
        uint32_t offset = Finalize(active_path_[i]);
        active_path_[i - 1].edges.push_back({last_key_[i - 1], offset});
        active_path_[i].edges.clear();
    }
    root_index_ = Finalize(active_path_[0]);
    active_path_[0].edges.clear();

    auto end_finalize = std::chrono::high_resolution_clock::now();

    if (path_compress_) {
        RunPathCompression();
    }

    uint32_t max_offset = temp_edges_.size();
    offset_bits_ = BitsNeededForValue(max_offset);

    terminal_node_ = (1ULL << offset_bits_) - 1;
    offset_mask_ = terminal_node_;

    auto start_alloc = std::chrono::high_resolution_clock::now();

    uint64_t total_label_bits = 0;
    uint32_t max_label_segments = 0;
    for (const auto &e : temp_edges_) {
        uint32_t lbits = static_cast<uint32_t>(e.label.size());
        total_label_bits += lbits;
        if (e.segment_count > max_label_segments)
            max_label_segments = e.segment_count;
    }

    length_bits_ = BitsNeededForValue(max_label_segments);
    label_offset_bits_ = BitsNeededForValue(total_label_bits);

    labels_.Init(total_label_bits);
    labels_.Clear();
    label_offsets_.Init(temp_edges_.size() * label_offset_bits_);
    label_offsets_.Clear();
    label_lengths_.Init(temp_edges_.size() * length_bits_);
    label_lengths_.Clear();
    offsets_.Init(temp_edges_.size() * offset_bits_);
    offsets_.Clear();
    is_last_.Init(temp_edges_.size() * 1);
    is_last_.Clear();

    auto end_alloc = std::chrono::high_resolution_clock::now();
    auto start_write = std::chrono::high_resolution_clock::now();

    uint64_t label_bit_pos = 0;
    for (size_t i = 0; i < temp_edges_.size(); ++i) {
        const auto &e = temp_edges_[i];

        label_offsets_.SetValPos(static_cast<uint64_t>(i) * label_offset_bits_, label_bit_pos,
                                 label_offset_bits_);
        label_lengths_.SetValPos(static_cast<uint64_t>(i) * length_bits_, e.segment_count,
                                 length_bits_);

        for (size_t j = 0; j < e.label.size(); ++j) {
            if (e.label[j] == '1') {
                labels_.SetBit(label_bit_pos + j);
            }
        }
        label_bit_pos += e.label.size();

        uint32_t target = e.target;
        if (target == BUILD_TERMINAL_NODE)
            target = terminal_node_;
        offsets_.SetValPos(static_cast<uint64_t>(i) * offset_bits_, target & offset_mask_,
                           offset_bits_);

        if (e.is_last)
            is_last_.SetBit(i);
    }

    auto end_write = std::chrono::high_resolution_clock::now();

    std::vector<TempEdge>().swap(temp_edges_);
    memo_.clear();
    active_path_.clear();
    last_key_.clear();

    auto end_finish = std::chrono::high_resolution_clock::now();

    if (print_timings) {
        std::chrono::duration<double> finalize_dur = end_finalize - start_finish;
        std::chrono::duration<double> alloc_dur = end_alloc - start_alloc;
        std::chrono::duration<double> write_dur = end_write - start_write;
        std::chrono::duration<double> cleanup_dur = end_finish - end_write;

        std::cout << "      [Finish] Finalize Time: " << std::fixed << std::setprecision(4)
                  << finalize_dur.count() << " s" << std::endl;
        std::cout << "      [Finish] Alloc Time:    " << std::fixed << std::setprecision(4)
                  << alloc_dur.count() << " s" << std::endl;
        std::cout << "      [Finish] Write Time:    " << std::fixed << std::setprecision(4)
                  << write_dur.count() << " s" << std::endl;
        std::cout << "      [Finish] Cleanup Time:  " << std::fixed << std::setprecision(4)
                  << cleanup_dur.count() << " s" << std::endl;
    }
}
