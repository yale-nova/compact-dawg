#pragma once

/**
 * dynamic_dawg.h
 *
 * DynamicDawg: a DAWG/DAFSA index for fixed-length Morton bitstrings with
 * **variable-width grouping**. Unlike CompactDawg (which uses a compile-time
 * GROUP_BITS), DynamicDawg takes a runtime `dawg_seg::SegmentPlan` that assigns
 * different group sizes to different bit positions.
 *
 * Motivation: IEEE-754 float Morton codes have three distinct phases:
 *   1. Dead prefix   (~0–20% of bits): cardinality=1, use maximum group width
 *   2. High-entropy  (~20–70%):        high branching, use small group width
 *   3. Low-entropy   (~70–100%):       cardinality drops, widen groups again
 *
 * Variable-width grouping exploits all three phases for better compression.
 *
 * Construction uses the incremental Daciuk algorithm (same as CompactDawg V1).
 * Suffix sharing via memo hashing is identical.
 * Edge labels are variable-length, stored in bit-packed vectors with per-edge
 * offsets and lengths (similar to V1's PATH_COMPRESS representation).
 *
 * Spatial range queries are deferred to a follow-up phase.
 */

#include "dawg_segmentation.h"
#include "ds-lib/bit_vector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class DynamicDawg {
public:
    static constexpr uint32_t BUILD_TERMINAL_NODE = 0xFFFFFFFF;

private:
    struct Edge {
        std::string label; // variable-length binary string
        uint32_t target;
        bool operator==(const Edge &other) const
        {
            return label == other.label && target == other.target;
        }
    };

    struct Node {
        std::vector<Edge> edges;
        bool operator==(const Node &other) const
        {
            if (edges.size() != other.edges.size())
                return false;
            for (size_t i = 0; i < edges.size(); ++i) {
                if (!(edges[i] == other.edges[i]))
                    return false;
            }
            return true;
        }
    };

    struct NodeHasher {
        size_t operator()(const Node &n) const
        {
            size_t hash = 0;
            for (const auto &e : n.edges) {
                size_t edge_hash = std::hash<std::string>{}(e.label);
                edge_hash ^= std::hash<uint32_t>{}(e.target) + 0x9e3779b97f4a7c15ULL +
                             (edge_hash << 6) + (edge_hash >> 2);
                hash ^= edge_hash + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    // --- Packed storage ---
    bits::BitVector labels_;
    bits::BitVector offsets_;
    bits::BitVector is_last_;
    bits::BitVector label_offsets_;
    bits::BitVector label_lengths_;

    // --- Build-time state ---
    std::unordered_map<Node, uint32_t, NodeHasher> memo_;
    std::vector<Node> active_path_;
    std::vector<std::string> last_key_; // chunked by plan

    uint32_t root_index_ = BUILD_TERMINAL_NODE;

    struct TempEdge {
        std::string label;
        uint32_t target;
        bool is_last;
        uint32_t segment_count;
    };
    std::vector<TempEdge> temp_edges_;

    // --- Packed storage metadata ---
    uint8_t offset_bits_ = 0;
    uint32_t terminal_node_ = 0;
    uint32_t offset_mask_ = 0;
    uint8_t label_offset_bits_ = 0;
    uint8_t length_bits_ = 0;

    // --- Segmentation plan ---
    dawg_seg::SegmentPlan plan_;
    bool path_compress_ = false;

    static uint8_t BitsNeededForValue(uint64_t value)
    {
        uint8_t bits = 0;
        while (value > 0) {
            ++bits;
            value >>= 1;
        }
        return bits == 0 ? 1 : bits;
    }

    bool TryReadPackedTarget(uint32_t edge_idx, uint32_t *target) const
    {
        if (!target)
            return false;
        const uint64_t off_pos = static_cast<uint64_t>(edge_idx) * offset_bits_;
        if (off_pos + offset_bits_ > offsets_.GetSizeInBits())
            return false;
        *target = offsets_.GetValPos(off_pos, offset_bits_);
        return true;
    }

    // -----------------------------------------------------------------------
    // Build helpers
    // -----------------------------------------------------------------------

    bool SegmentEquals(const std::string &bitstring, size_t segment_idx,
                       const std::string &segment) const
    {
        if (segment_idx >= plan_.widths.size())
            return false;
        uint32_t width = plan_.widths[segment_idx];
        if (segment.size() != width)
            return false;
        return bitstring.compare(plan_.bit_offsets[segment_idx], width, segment) == 0;
    }

    void AssignSegment(std::string *out, const std::string &bitstring, size_t segment_idx) const
    {
        if (!out || segment_idx >= plan_.widths.size())
            return;
        out->assign(bitstring, plan_.bit_offsets[segment_idx], plan_.widths[segment_idx]);
    }

    uint32_t Finalize(const Node &node);

    // -----------------------------------------------------------------------
    // Path compression (optional, post-build)
    // -----------------------------------------------------------------------

    void RunPathCompression();

    // -----------------------------------------------------------------------
    // Packed storage accessors
    // -----------------------------------------------------------------------

    uint64_t GetLabelBitOffset(uint32_t edge_idx) const
    {
        return label_offsets_.GetValPos(
            static_cast<uint64_t>(edge_idx) * label_offset_bits_, label_offset_bits_);
    }

    uint32_t GetLabelSegmentCount(uint32_t edge_idx) const
    {
        return label_lengths_.GetValPos(
            static_cast<uint64_t>(edge_idx) * length_bits_, length_bits_);
    }

    uint32_t PlanBitLength(size_t start_segment, uint32_t segment_count) const
    {
        uint32_t total = 0;
        size_t end = std::min(plan_.widths.size(), start_segment + segment_count);
        for (size_t i = start_segment; i < end; ++i) {
            total += plan_.widths[i];
        }
        return total;
    }

    std::string ReadLabelString(uint32_t edge_idx, size_t start_segment) const
    {
        uint64_t bit_off = GetLabelBitOffset(edge_idx);
        uint32_t n_bits = PlanBitLength(start_segment, GetLabelSegmentCount(edge_idx));
        std::string result;
        result.reserve(n_bits);
        for (uint32_t j = 0; j < n_bits; ++j) {
            result += labels_.GetBit(bit_off + j) ? '1' : '0';
        }
        return result;
    }

public:
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Construct with a pre-built segmentation plan (Strategy A).
    explicit DynamicDawg(dawg_seg::SegmentPlan plan, bool path_compress = false)
        : plan_(std::move(plan)), path_compress_(path_compress)
    {
    }

    /// Default constructor (must call SetPlan() before Insert).
    DynamicDawg() = default;

    void SetPlan(dawg_seg::SegmentPlan plan, bool path_compress = false)
    {
        plan_ = std::move(plan);
        path_compress_ = path_compress;
    }

    // -----------------------------------------------------------------------
    // Insertion (Daciuk incremental algorithm)
    // -----------------------------------------------------------------------

    void Insert(const std::string &bitstring);

    // -----------------------------------------------------------------------
    // Finish: finalize all pending nodes and pack into bit-vectors
    // -----------------------------------------------------------------------

    void Finish(bool print_timings = false);

    // -----------------------------------------------------------------------
    // Contains: exact-match lookup
    // -----------------------------------------------------------------------

    bool Contains(const std::string &bitstring) const;

    // -----------------------------------------------------------------------
    // LexicographicSearch: range query on bitstring ordering
    // -----------------------------------------------------------------------

    void LexicographicSearch(const std::string &start_bitstring, const std::string &end_bitstring,
                             std::vector<std::string> *results) const;

    // -----------------------------------------------------------------------
    // Size and metadata
    // -----------------------------------------------------------------------

    size_t size_in_bytes() const
    {
        size_t total_bits =
            labels_.GetSizeInBits() + offsets_.GetSizeInBits() + is_last_.GetSizeInBits() +
            label_offsets_.GetSizeInBits() + label_lengths_.GetSizeInBits();
        return (total_bits + 7) / 8 + get_plan_bytes();
    }

    size_t size_in_bytes_bitvectors_only() const
    {
        size_t total_bits =
            labels_.GetSizeInBits() + offsets_.GetSizeInBits() + is_last_.GetSizeInBits() +
            label_offsets_.GetSizeInBits() + label_lengths_.GetSizeInBits();
        return (total_bits + 7) / 8;
    }

    size_t get_total_edges() const { return is_last_.GetSizeInBits(); }

    size_t get_total_label_bits() const { return labels_.GetSizeInBits(); }

    size_t get_label_offsets_bits() const { return label_offsets_.GetSizeInBits(); }

    size_t get_label_lengths_bits() const { return label_lengths_.GetSizeInBits(); }

    size_t get_offsets_bits() const { return offsets_.GetSizeInBits(); }

    size_t get_is_last_bits() const { return is_last_.GetSizeInBits(); }

    size_t get_plan_bytes() const
    {
        return plan_.widths.size() * sizeof(uint32_t) * 2 + sizeof(uint32_t);
    }

    size_t get_non_label_metadata_bits() const
    {
        return offsets_.GetSizeInBits() + is_last_.GetSizeInBits();
    }

    double get_average_label_bits() const
    {
        return get_total_edges() == 0
                   ? 0.0
                   : static_cast<double>(labels_.GetSizeInBits()) /
                         static_cast<double>(get_total_edges());
    }

    double get_label_metadata_share() const
    {
        const double total_bits = static_cast<double>(
            labels_.GetSizeInBits() + offsets_.GetSizeInBits() + is_last_.GetSizeInBits() +
            label_offsets_.GetSizeInBits() + label_lengths_.GetSizeInBits());
        if (total_bits <= 0.0)
            return 0.0;
        return static_cast<double>(label_offsets_.GetSizeInBits() + label_lengths_.GetSizeInBits()) /
               total_bits;
    }

    uint8_t get_offset_bits() const { return offset_bits_; }

    uint8_t get_length_bits() const { return length_bits_; }

    const dawg_seg::SegmentPlan &get_plan() const { return plan_; }

    uint32_t get_target(uint32_t edge_idx) const
    {
        uint64_t off_pos = static_cast<uint64_t>(edge_idx) * offset_bits_;
        return offsets_.GetValPos(off_pos, offset_bits_);
    }

    size_t get_node_count() const
    {
        size_t count = 0;
        for (size_t i = 0; i < is_last_.GetSizeInBits(); ++i) {
            if (is_last_.GetBit(i))
                count++;
        }
        return count;
    }

private:
    // -----------------------------------------------------------------------
    // LexicographicSearch recursive traversal
    // -----------------------------------------------------------------------

    void LexSearchRecursive(uint32_t current_idx, const std::vector<std::string> &start_key,
                            const std::vector<std::string> &end_key, size_t depth,
                            bool start_bound, bool end_bound, std::string &current_string,
                            std::vector<std::string> *results) const;
};

#include "dynamic_dawg/build.h"
#include "dynamic_dawg/query_lex.h"
