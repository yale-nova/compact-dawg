#pragma once

#include "../compact_dawg.h"

template <uint32_t GROUP_BITS, bool PATH_COMPRESS, bool TRACK_SHARING>
void CompactDawg<GROUP_BITS, PATH_COMPRESS, TRACK_SHARING>::SpatialRangeSearch(
    data_point<NUM_DIMENSIONS> start_point, data_point<NUM_DIMENSIONS> end_point,
    std::vector<std::string> *results) const
{
    if (!results)
        return;
    if (root_index_ == BUILD_TERMINAL_NODE || root_index_ == terminal_node_)
        return;

    std::string current_path;
    current_path.reserve(NUM_DIMENSIONS * 32);

    SpatialRangeSearchRecursive(root_index_, start_point, end_point, 0, current_path, results);
}

template <uint32_t GROUP_BITS, bool PATH_COMPRESS, bool TRACK_SHARING>
void CompactDawg<GROUP_BITS, PATH_COMPRESS, TRACK_SHARING>::SpatialRangeSearchRecursive(
    uint32_t current_idx, data_point<NUM_DIMENSIONS> start_point,
    data_point<NUM_DIMENSIONS> end_point, size_t dawg_depth, std::string &current_string,
    std::vector<std::string> *results) const
{
    if (current_idx == BUILD_TERMINAL_NODE || current_idx == terminal_node_) {
        results->push_back(current_string);
        return;
    }

    if constexpr (!PATH_COMPRESS) {
        uint32_t trie_level = (dawg_depth * GROUP_BITS) / NUM_DIMENSIONS;
        uint32_t bit_index_in_level = (dawg_depth * GROUP_BITS) % NUM_DIMENSIONS;

        morton_t start_symbol = start_point.leaf_to_symbol(trie_level);
        morton_t end_symbol = end_point.leaf_to_symbol(trie_level);
        morton_t bound_magic = ~(start_symbol ^ end_symbol);

        std::string required_bits(GROUP_BITS, 'X');
        for (size_t j = 0; j < GROUP_BITS; ++j) {
            if (bit_index_in_level + j >= NUM_DIMENSIONS)
                break;
            uint32_t morton_bit_idx = NUM_DIMENSIONS - 1 - (bit_index_in_level + j);
            if (bound_magic.get_bit_unsafe(morton_bit_idx)) {
                required_bits[j] = start_symbol.get_bit_unsafe(morton_bit_idx) ? '1' : '0';
            }
        }

        while (true) {
            if (current_idx >= is_last_.GetSizeInBits())
                return;

            uint32_t offset = 0;
            if (!TryReadPackedTarget(current_idx, &offset))
                return;
            bool is_last = is_last_.GetBit(current_idx);

            bool match = true;
            std::string label;
            label.reserve(LABEL_BITS);
            for (size_t j = 0; j < LABEL_BITS; ++j) {
                if (current_idx * LABEL_BITS + j >= labels_.GetSizeInBits()) {
                    match = false;
                    break;
                }
                bool bit = labels_.GetBit(current_idx * LABEL_BITS + j);
                char c = bit ? '1' : '0';
                label += c;
                if (required_bits[j] != 'X' && required_bits[j] != c) {
                    match = false;
                    break;
                }
            }

            if (match) {
                data_point<NUM_DIMENSIONS> next_start = start_point;
                data_point<NUM_DIMENSIONS> next_end = end_point;

                bool shrink_now = ((bit_index_in_level + GROUP_BITS) >= NUM_DIMENSIONS);
                if (shrink_now) {
                    morton_t current_symbol(0ULL);
                    size_t level_start_idx = trie_level * NUM_DIMENSIONS;
                    size_t bits_from_string =
                        std::min(static_cast<size_t>(NUM_DIMENSIONS),
                                 current_string.size() - level_start_idx);
                    for (size_t i = 0; i < bits_from_string; ++i) {
                        if (current_string[level_start_idx + i] == '1') {
                            current_symbol.set_bit_unsafe(NUM_DIMENSIONS - 1 - i, true);
                        }
                    }
                    size_t bits_from_label = NUM_DIMENSIONS - bits_from_string;
                    for (size_t i = 0; i < bits_from_label && i < label.size(); ++i) {
                        if (label[i] == '1') {
                            current_symbol.set_bit_unsafe(
                                NUM_DIMENSIONS - 1 - (bits_from_string + i), true);
                        }
                    }
                    data_point<NUM_DIMENSIONS>::shrink_query_bounds(&next_start, &next_end,
                                                                    current_symbol, trie_level);
                }

                size_t original_len = current_string.size();
                current_string += label;

                SpatialRangeSearchRecursive(offset, next_start, next_end, dawg_depth + 1,
                                            current_string, results);

                current_string.resize(original_len);
            }

            if (is_last)
                break;
            current_idx++;
        }
    } else {
        while (true) {
            if (current_idx >= is_last_.GetSizeInBits())
                return;

            uint32_t label_chunks = GetLabelChunks(current_idx);
            uint64_t label_bit_off = GetLabelBitOffset(current_idx);

            uint32_t offset = 0;
            if (!TryReadPackedTarget(current_idx, &offset))
                return;
            bool is_last = is_last_.GetBit(current_idx);

            size_t original_len = current_string.size();
            bool match = true;
            data_point<NUM_DIMENSIONS> running_start = start_point;
            data_point<NUM_DIMENSIONS> running_end = end_point;

            for (uint32_t c = 0; c < label_chunks; ++c) {
                size_t inner_depth = dawg_depth + c;
                uint32_t trie_level = (inner_depth * GROUP_BITS) / NUM_DIMENSIONS;
                uint32_t bit_index_in_level = (inner_depth * GROUP_BITS) % NUM_DIMENSIONS;

                morton_t start_symbol = running_start.leaf_to_symbol(trie_level);
                morton_t end_symbol = running_end.leaf_to_symbol(trie_level);
                morton_t bound_magic = ~(start_symbol ^ end_symbol);

                uint64_t chunk_off = label_bit_off + static_cast<uint64_t>(c) * GROUP_BITS;
                std::string chunk_bits;
                chunk_bits.reserve(GROUP_BITS);

                for (size_t j = 0; j < GROUP_BITS; ++j) {
                    if (chunk_off + j >= labels_.GetSizeInBits()) {
                        match = false;
                        break;
                    }
                    bool bit = labels_.GetBit(chunk_off + j);
                    char ch = bit ? '1' : '0';
                    chunk_bits += ch;

                    if (bit_index_in_level + j < NUM_DIMENSIONS) {
                        uint32_t morton_bit_idx = NUM_DIMENSIONS - 1 - (bit_index_in_level + j);
                        if (bound_magic.get_bit_unsafe(morton_bit_idx)) {
                            char required =
                                start_symbol.get_bit_unsafe(morton_bit_idx) ? '1' : '0';
                            if (ch != required) {
                                match = false;
                                break;
                            }
                        }
                    }
                }

                if (!match)
                    break;

                current_string += chunk_bits;

                bool shrink_now = ((bit_index_in_level + GROUP_BITS) >= NUM_DIMENSIONS);
                if (shrink_now) {
                    morton_t current_symbol(0ULL);
                    size_t level_start_idx = trie_level * NUM_DIMENSIONS;
                    size_t avail = current_string.size() - level_start_idx;
                    size_t bits_to_read = std::min(avail, static_cast<size_t>(NUM_DIMENSIONS));
                    for (size_t i = 0; i < bits_to_read; ++i) {
                        if (current_string[level_start_idx + i] == '1') {
                            current_symbol.set_bit_unsafe(NUM_DIMENSIONS - 1 - i, true);
                        }
                    }
                    data_point<NUM_DIMENSIONS>::shrink_query_bounds(&running_start, &running_end,
                                                                    current_symbol, trie_level);
                }
            }

            if (match) {
                SpatialRangeSearchRecursive(offset, running_start, running_end,
                                            dawg_depth + label_chunks, current_string, results);
            }

            current_string.resize(original_len);

            if (is_last)
                break;
            current_idx++;
        }
    }
}
