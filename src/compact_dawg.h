#pragma once

#include "ds-lib/bit_vector.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef NUM_DIMENSIONS
#include "mdtrie/data_point.h"
#endif

template <uint32_t GROUP_BITS, bool PATH_COMPRESS = false, bool TRACK_SHARING = false>
class CompactDawg
{
public:
    static constexpr uint32_t LABEL_BITS = GROUP_BITS;
    static constexpr uint32_t BUILD_TERMINAL_NODE = 0xFFFFFFFF;

    struct SharingStats {
        size_t finalize_calls = 0;
        size_t memo_hits = 0;
        size_t unique_nodes = 0;
        size_t trie_edges = 0;
        size_t dawg_edges = 0;
    };

private:
    struct Edge {
        std::string label;
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
                hash ^= (std::hash<std::string>{}(e.label) << 1) ^ std::hash<uint32_t>{}(e.target);
            }
            return hash;
        }
    };

    bits::BitVector labels_;
    bits::BitVector offsets_;
    bits::BitVector is_last_;
    bits::BitVector label_offsets_;
    bits::BitVector label_lengths_;

    std::unordered_map<Node, uint32_t, NodeHasher> memo_;
    std::vector<Node> active_path_;
    std::vector<std::string> last_key_;
    uint32_t root_index_ = BUILD_TERMINAL_NODE;

    struct TempEdge {
        std::string label;
        uint32_t target;
        bool is_last;
    };
    std::vector<TempEdge> temp_edges_;

    uint8_t offset_bits_ = 0;
    uint32_t terminal_node_ = 0;
    uint32_t offset_mask_ = 0;
    uint8_t label_offset_bits_ = 0;
    uint8_t length_bits_ = 0;

    struct NoSharingState {};
    struct SharingState {
        SharingStats stats;
        std::vector<size_t> per_depth_finalize;
        std::vector<size_t> per_depth_hits;
        std::vector<size_t> per_depth_dawg_edges;
    };
    [[no_unique_address]] std::conditional_t<TRACK_SHARING, SharingState, NoSharingState> sharing_;

    static uint8_t BitsNeededForValue(uint64_t value)
    {
        uint8_t bits = 0;
        while (value > 0) {
            ++bits;
            value >>= 1;
        }
        return bits == 0 ? 1 : bits;
    }

    static std::vector<std::string> ChunkBitstring(const std::string &bitstring, char pad_char)
    {
        std::vector<std::string> key;
        key.reserve((bitstring.size() + GROUP_BITS - 1) / GROUP_BITS);
        for (size_t i = 0; i < bitstring.size(); i += GROUP_BITS) {
            size_t end = std::min(bitstring.size(), i + GROUP_BITS);
            std::string chunk = bitstring.substr(i, end - i);
            while (chunk.size() < GROUP_BITS)
                chunk += pad_char;
            key.push_back(std::move(chunk));
        }
        return key;
    }

    bool TryReadPackedTarget(uint32_t edge_idx, uint32_t *target) const
    {
        if (!target)
            return false;
        if (offset_bits_ == 0) {
            *target = terminal_node_;
            return true;
        }
        const uint64_t off_pos = static_cast<uint64_t>(edge_idx) * offset_bits_;
        if (off_pos + offset_bits_ > offsets_.GetSizeInBits())
            return false;
        *target = offsets_.GetValPos(off_pos, offset_bits_);
        return true;
    }

    bool HasNoRoot() const
    {
        return root_index_ == BUILD_TERMINAL_NODE || root_index_ == terminal_node_;
    }

    uint32_t Finalize(const Node &node, size_t depth = SIZE_MAX);

    uint64_t GetLabelBitOffset(uint32_t edge_idx) const
    {
        if constexpr (PATH_COMPRESS) {
            return label_offsets_.GetValPos(
                static_cast<uint64_t>(edge_idx) * label_offset_bits_, label_offset_bits_);
        } else {
            return static_cast<uint64_t>(edge_idx) * LABEL_BITS;
        }
    }

    uint32_t GetLabelChunks(uint32_t edge_idx) const
    {
        if constexpr (PATH_COMPRESS) {
            return label_lengths_.GetValPos(
                static_cast<uint64_t>(edge_idx) * length_bits_, length_bits_);
        } else {
            return 1;
        }
    }

    std::string ReadLabelString(uint32_t edge_idx) const
    {
        uint64_t bit_off = GetLabelBitOffset(edge_idx);
        uint32_t n_bits = GetLabelChunks(edge_idx) * GROUP_BITS;
        std::string result;
        result.reserve(n_bits);
        for (uint32_t j = 0; j < n_bits; ++j) {
            result += labels_.GetBit(bit_off + j) ? '1' : '0';
        }
        return result;
    }

    void RunPathCompression();

public:
    CompactDawg() {}

    void Insert(const std::string &bitstring);

    void Finish(bool print_timings = false);

    bool Contains(const std::string &bitstring) const;

    void LexicographicSearch(const std::string &start_bitstring, const std::string &end_bitstring,
                            std::vector<std::string> *results) const;

#ifdef NUM_DIMENSIONS
    void SpatialRangeSearch(data_point<NUM_DIMENSIONS> start_point,
                            data_point<NUM_DIMENSIONS> end_point,
                            std::vector<std::string> *results) const;
#endif

    size_t size_in_bytes() const
    {
        size_t total_bits =
            labels_.GetSizeInBits() + offsets_.GetSizeInBits() + is_last_.GetSizeInBits();
        if constexpr (PATH_COMPRESS) {
            total_bits += label_offsets_.GetSizeInBits() + label_lengths_.GetSizeInBits();
        }
        return (total_bits + 7) / 8;
    }

    static uint8_t OffsetBitsForEdgeCount(size_t edge_count)
    {
        return BitsNeededForValue(edge_count);
    }

    static size_t PackedFixedWidthBytesForEdgeCount(size_t edge_count)
    {
        const size_t offset_bits = OffsetBitsForEdgeCount(edge_count);
        const size_t total_bits =
            edge_count * (static_cast<size_t>(LABEL_BITS) + offset_bits + 1);
        return (total_bits + 7) / 8;
    }

    size_t get_total_edges() const { return is_last_.GetSizeInBits(); }

    uint8_t get_offset_bits() const { return offset_bits_; }

    uint8_t get_length_bits() const { return length_bits_; }

    uint32_t get_target(uint32_t edge_idx) const
    {
        uint32_t target = terminal_node_;
        TryReadPackedTarget(edge_idx, &target);
        return target;
    }

    size_t get_node_count() const
    {
        size_t n = 0;
        size_t total = is_last_.GetSizeInBits();
        for (size_t i = 0; i < total; ++i) {
            if (i == 0 || is_last_.GetBit(i - 1))
                n++;
        }
        return n;
    }

    const SharingStats &GetSharingStats() const
        requires(TRACK_SHARING)
    {
        return sharing_.stats;
    }

    const std::vector<size_t> &GetPerDepthFinalize() const
        requires(TRACK_SHARING)
    {
        return sharing_.per_depth_finalize;
    }

    const std::vector<size_t> &GetPerDepthHits() const
        requires(TRACK_SHARING)
    {
        return sharing_.per_depth_hits;
    }

    const std::vector<size_t> &GetPerDepthDawgEdges() const
        requires(TRACK_SHARING)
    {
        return sharing_.per_depth_dawg_edges;
    }

private:
    void LexicographicSearchRecursive(uint32_t current_idx, const std::vector<std::string> &start_key,
                                     const std::vector<std::string> &end_key, size_t depth,
                                     bool start_bound, bool end_bound, std::string &current_string,
                                     std::vector<std::string> *results) const;

#ifdef NUM_DIMENSIONS
    void SpatialRangeSearchRecursive(uint32_t current_idx, data_point<NUM_DIMENSIONS> start_point,
                                     data_point<NUM_DIMENSIONS> end_point, size_t dawg_depth,
                                     std::string &current_string,
                                     std::vector<std::string> *results) const;
#endif
};

#include "compact_dawg/build.h"
#include "compact_dawg/query_lex.h"
#ifdef NUM_DIMENSIONS
#include "compact_dawg/query_spatial.h"
#endif
