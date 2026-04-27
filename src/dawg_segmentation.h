#pragma once

/**
 * dawg_segmentation.h
 *
 * Segmentation utilities for DynamicDawg: compute variable-width group plans
 * that chunk a Morton bitstring into segments with controlled branching.
 *
 * Two strategies:
 *   - from_widths():  accept a pre-computed width sequence (Strategy A)
 *   - greedy():       compute segment widths from sorted keys using ρ-threshold (Strategy B)
 *
 * Decoupled from the DAWG itself — this is pure data analysis.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace dawg_seg {

// -----------------------------------------------------------------------
// SegmentPlan: immutable description of how to chunk a bitstring
// -----------------------------------------------------------------------

struct SegmentPlan {
    std::vector<uint32_t> widths;       // per-segment group width in bits
    std::vector<uint32_t> bit_offsets;  // cumulative: bit_offsets[i] = sum(widths[0..i-1])
    uint32_t total_bits = 0;

    size_t depth() const { return widths.size(); }

    // Chunk a bitstring into variable-width pieces according to this plan.
    // Returns a vector of binary substrings (one per segment).
    std::vector<std::string> chunk_key(const std::string &bitstring) const
    {
        std::vector<std::string> key;
        key.reserve(widths.size());
        for (size_t i = 0; i < widths.size(); ++i) {
            uint32_t start = bit_offsets[i];
            uint32_t w = widths[i];
            if (start + w <= bitstring.size()) {
                key.push_back(bitstring.substr(start, w));
            } else {
                // Pad with zeros if the bitstring is shorter than expected
                std::string chunk;
                if (start < bitstring.size()) {
                    chunk = bitstring.substr(start, bitstring.size() - start);
                }
                while (chunk.size() < w)
                    chunk += '0';
                key.push_back(std::move(chunk));
            }
        }
        return key;
    }

    // Validate the plan: widths must sum to total_bits
    bool valid() const
    {
        if (widths.empty())
            return false;
        if (bit_offsets.size() != widths.size())
            return false;

        uint32_t sum = 0;
        uint32_t expected_offset = 0;
        for (size_t i = 0; i < widths.size(); ++i) {
            if (bit_offsets[i] != expected_offset)
                return false;
            sum += widths[i];
            expected_offset += widths[i];
        }
        return sum == total_bits;
    }
};

inline std::vector<uint32_t> default_candidate_widths()
{
    std::vector<uint32_t> widths;
    for (uint32_t g = 1; g <= 1024; g *= 2)
        widths.push_back(g);
    return widths;
}

inline void sort_candidate_widths_desc(std::vector<uint32_t> *candidate_widths)
{
    if (!candidate_widths)
        return;
    std::sort(candidate_widths->begin(), candidate_widths->end(), std::greater<uint32_t>());
}

// -----------------------------------------------------------------------
// Strategy A: construct from a pre-computed width sequence
// -----------------------------------------------------------------------

inline SegmentPlan from_widths(std::vector<uint32_t> widths, uint32_t total_bits)
{
    SegmentPlan plan;
    plan.widths = std::move(widths);
    plan.total_bits = total_bits;
    plan.bit_offsets.resize(plan.widths.size());
    uint32_t offset = 0;
    for (size_t i = 0; i < plan.widths.size(); ++i) {
        plan.bit_offsets[i] = offset;
        offset += plan.widths[i];
    }
    return plan;
}

// -----------------------------------------------------------------------
// Cardinality helpers (shared with analyze_bitstring_structure.cpp)
// -----------------------------------------------------------------------

// Count unique g-bit symbols at a given bit position across all keys.
inline size_t count_unique_symbols(const std::vector<std::string> &keys,
                                   uint32_t start_bit, uint32_t width)
{
    std::unordered_set<std::string_view> seen;
    seen.reserve(keys.size());
    for (const auto &k : keys) {
        std::string_view sv(k.data() + start_bit, width);
        seen.insert(sv);
    }
    return seen.size();
}

inline size_t count_adjacent_changes(const std::vector<std::string> &keys,
                                     uint32_t start_bit, uint32_t width)
{
    if (keys.size() < 2)
        return 0;

    size_t changes = 0;
    std::string_view prev(keys.front().data() + start_bit, width);
    for (size_t i = 1; i < keys.size(); ++i) {
        std::string_view current(keys[i].data() + start_bit, width);
        if (current != prev)
            ++changes;
        prev = current;
    }
    return changes;
}

// K_max = min(N, 2^g). For g >= 64, min(N, 2^g) = N for any practical N.
inline size_t symbol_capacity(size_t n_keys, uint32_t g)
{
    if (g >= 64u)
        return n_keys;
    const uint64_t two_g = 1ULL << g;
    if (two_g >= n_keys)
        return n_keys;
    return static_cast<size_t>(two_g);
}

inline uint32_t ceil_log2_u64(uint64_t value)
{
    if (value <= 1)
        return 1;
    uint32_t bits = 0;
    uint64_t x = value - 1;
    while (x > 0) {
        ++bits;
        x >>= 1;
    }
    return bits;
}

struct SegmentScoreFeatures {
    size_t cardinality = 0;
    size_t adjacent_changes = 0;
    size_t kmax = 0;
    double rho = 0.0;
    double branch_pressure = 0.0;
    double adjacency_rate = 0.0;
};

inline SegmentScoreFeatures compute_segment_features(const std::vector<std::string> &sorted_keys,
                                                     uint32_t start_bit, uint32_t width)
{
    SegmentScoreFeatures out;
    out.cardinality = count_unique_symbols(sorted_keys, start_bit, width);
    out.adjacent_changes = count_adjacent_changes(sorted_keys, start_bit, width);
    out.kmax = symbol_capacity(sorted_keys.size(), width);

    if (!sorted_keys.empty()) {
        out.branch_pressure =
            static_cast<double>(out.cardinality) / static_cast<double>(sorted_keys.size());
        out.adjacency_rate =
            static_cast<double>(out.adjacent_changes) /
            static_cast<double>(std::max<size_t>(1, sorted_keys.size() - 1));
    }
    if (out.kmax > 0) {
        out.rho = static_cast<double>(out.cardinality) / static_cast<double>(out.kmax);
    }
    return out;
}

inline double estimate_segment_cost(uint32_t start_bit, uint32_t width, uint32_t total_bits,
                                    const SegmentScoreFeatures &f)
{
    const double depth_cost = 14.0 / static_cast<double>(std::max<uint32_t>(1, width));
    const double label_cost = static_cast<double>(width) / 8.0;
    const double offset_meta_bits =
        static_cast<double>(ceil_log2_u64(static_cast<uint64_t>(total_bits) + 1));
    const double length_meta_bits = static_cast<double>(ceil_log2_u64(width + 1));
    const double metadata_cost = (offset_meta_bits + length_meta_bits + 1.0) / 8.0;

    const double high_entropy_penalty =
        22.0 * f.rho + 12.0 * f.branch_pressure + 10.0 * f.adjacency_rate;
    const double compression_bonus =
        10.0 * (1.0 - f.rho) + 8.0 * (1.0 - f.branch_pressure) + 8.0 * (1.0 - f.adjacency_rate);

    const double normalized_pos =
        (total_bits == 0) ? 0.0 : static_cast<double>(start_bit) / static_cast<double>(total_bits);
    const bool edge_phase = (normalized_pos < 0.20 || normalized_pos > 0.70);
    const double width_bias = edge_phase ? -0.10 * static_cast<double>(width)
                                         : 0.15 * static_cast<double>(width);

    return label_cost + metadata_cost + depth_cost + high_entropy_penalty - compression_bonus +
           width_bias;
}

// -----------------------------------------------------------------------
// Strategy B: greedy segmentation from sorted keys
// -----------------------------------------------------------------------

inline SegmentPlan greedy(const std::vector<std::string> &sorted_keys,
                          uint32_t total_bits,
                          double rho_threshold = 0.5,
                          std::vector<uint32_t> candidate_widths = {})
{
    if (candidate_widths.empty())
        candidate_widths = default_candidate_widths();
    sort_candidate_widths_desc(&candidate_widths);

    std::vector<uint32_t> widths;
    uint32_t pos = 0;
    size_t n = sorted_keys.size();

    while (pos < total_bits) {
        uint32_t best_width = 0;
        bool found = false;

        for (uint32_t g : candidate_widths) {
            if (pos + g > total_bits)
                continue;

            size_t card = count_unique_symbols(sorted_keys, pos, g);
            size_t kmax = symbol_capacity(n, g);
            double rho = static_cast<double>(card) / static_cast<double>(kmax);

            if (rho <= rho_threshold) {
                best_width = g;
                found = true;
                break; // largest g that fits
            }

            // Track smallest g as fallback
            if (best_width == 0 || g < best_width) {
                best_width = g;
            }
        }

        if (!found && best_width == 0) {
            best_width = 1; // absolute fallback
        }

        widths.push_back(best_width);
        pos += best_width;
    }

    return from_widths(std::move(widths), total_bits);
}

// -----------------------------------------------------------------------
// Strategy C: greedy segmentation with a DAWG-oriented cost heuristic
// -----------------------------------------------------------------------

inline SegmentPlan greedy_cost_aware(const std::vector<std::string> &sorted_keys,
                                     uint32_t total_bits,
                                     double rho_threshold = 0.5,
                                     uint32_t min_width = 1,
    std::vector<uint32_t> candidate_widths = {})
{
    if (candidate_widths.empty())
        candidate_widths = default_candidate_widths();
    sort_candidate_widths_desc(&candidate_widths);

    std::vector<uint32_t> widths;
    uint32_t pos = 0;

    while (pos < total_bits) {
        double best_score = std::numeric_limits<double>::infinity();
        double best_rho = std::numeric_limits<double>::infinity();
        uint32_t best_width = 1;
        uint32_t fallback_width = 0;

        for (uint32_t g : candidate_widths) {
            if (g == 0 || pos + g > total_bits)
                continue;
            if (g < min_width)
                continue;

            SegmentScoreFeatures f = compute_segment_features(sorted_keys, pos, g);
            double score = estimate_segment_cost(pos, g, total_bits, f);
            bool passes_rho = (f.rho <= rho_threshold);
            bool choose = false;

            if (passes_rho) {
                const double rel_margin =
                    (std::isfinite(best_score) && best_score > 0.0) ? std::abs(best_score) * 0.05
                                                                     : 0.05;
                if (score + rel_margin < best_score) {
                    choose = true;
                } else if (std::abs(score - best_score) <= rel_margin) {
                    if (f.rho > 0.75 || f.adjacency_rate > 0.75) {
                        choose = (g < best_width);
                    } else if (f.rho < 0.20 && f.adjacency_rate < 0.25) {
                        choose = (g > best_width);
                    } else {
                        choose = (f.rho < best_rho) || (f.rho == best_rho && g > best_width);
                    }
                }
            }

            if (choose) {
                best_score = score;
                best_rho = f.rho;
                best_width = g;
            }

            if (fallback_width == 0 || g < fallback_width)
                fallback_width = g;
        }

        if (!std::isfinite(best_score))
            best_width = (fallback_width > 0) ? fallback_width : 1;

        widths.push_back(best_width);
        pos += best_width;
    }

    return from_widths(std::move(widths), total_bits);
}

inline SegmentPlan greedy_cost_aware_min_width(const std::vector<std::string> &sorted_keys,
                                               uint32_t total_bits,
                                               double rho_threshold = 0.5,
                                               uint32_t min_width = 4)
{
    std::vector<uint32_t> candidate_widths;
    for (uint32_t g = min_width; g <= 1024; g *= 2)
        candidate_widths.push_back(g);
    return greedy_cost_aware(sorted_keys, total_bits, rho_threshold, min_width,
                             std::move(candidate_widths));
}

inline SegmentPlan phase_aware(uint32_t total_bits,
                               double prefix_end = 0.20,
                               double suffix_start = 0.70,
                               uint32_t prefix_width = 1024,
                               uint32_t middle_width = 16,
                               uint32_t suffix_width = 128)
{
    if (total_bits == 0)
        return {};

    prefix_end = std::max(0.0, std::min(1.0, prefix_end));
    suffix_start = std::max(prefix_end, std::min(1.0, suffix_start));

    uint32_t prefix_end_bit = static_cast<uint32_t>(total_bits * prefix_end);
    uint32_t suffix_start_bit = static_cast<uint32_t>(total_bits * suffix_start);

    std::vector<uint32_t> widths;
    uint32_t pos = 0;

    auto append_phase = [&](uint32_t end_bit, uint32_t width) {
        width = std::max<uint32_t>(1, width);
        while (pos < end_bit) {
            uint32_t remaining = end_bit - pos;
            uint32_t w = std::min(width, remaining);
            widths.push_back(w);
            pos += w;
        }
    };

    append_phase(prefix_end_bit, prefix_width);
    append_phase(suffix_start_bit, middle_width);
    append_phase(total_bits, suffix_width);

    return from_widths(std::move(widths), total_bits);
}

// -----------------------------------------------------------------------
// Uniform plan: equivalent to fixed GROUP_BITS
// -----------------------------------------------------------------------

inline SegmentPlan uniform(uint32_t group_bits, uint32_t total_bits)
{
    std::vector<uint32_t> widths;
    uint32_t n_chunks = total_bits / group_bits;
    widths.resize(n_chunks, group_bits);
    // Handle remainder if total_bits is not divisible by group_bits
    uint32_t remainder = total_bits % group_bits;
    if (remainder > 0) {
        widths.push_back(remainder);
    }
    return from_widths(std::move(widths), total_bits);
}

} // namespace dawg_seg
