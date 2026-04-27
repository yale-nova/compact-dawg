#pragma once

#include "../compact_dawg.h"

template <uint32_t GROUP_BITS, bool PATH_COMPRESS, bool TRACK_SHARING>
bool CompactDawg<GROUP_BITS, PATH_COMPRESS, TRACK_SHARING>::Contains(
    const std::string &bitstring) const
{
    std::vector<std::string> key = ChunkBitstring(bitstring, '0');

    if (HasNoRoot())
        return false;

    uint32_t current_idx = root_index_;
    size_t key_pos = 0;

    while (key_pos < key.size()) {
        if (current_idx == terminal_node_)
            return false;

        bool found = false;

        while (true) {
            if (current_idx >= is_last_.GetSizeInBits())
                return false;

            uint32_t label_chunks = GetLabelChunks(current_idx);
            uint64_t label_bit_off = GetLabelBitOffset(current_idx);

            uint32_t offset = 0;
            if (!TryReadPackedTarget(current_idx, &offset))
                return false;

            bool match = true;
            if (key_pos + label_chunks > key.size()) {
                match = false;
            } else {
                for (uint32_t c = 0; c < label_chunks && match; ++c) {
                    const std::string &kc = key[key_pos + c];
                    uint64_t chunk_off = label_bit_off + static_cast<uint64_t>(c) * GROUP_BITS;
                    for (size_t j = 0; j < GROUP_BITS; ++j) {
                        if (chunk_off + j >= labels_.GetSizeInBits()) {
                            match = false;
                            break;
                        }
                        bool bit = labels_.GetBit(chunk_off + j);
                        if ((bit ? '1' : '0') != kc[j]) {
                            match = false;
                            break;
                        }
                    }
                }
            }

            bool is_last = is_last_.GetBit(current_idx);

            if (match) {
                key_pos += label_chunks;
                current_idx = offset;
                found = true;
                break;
            }
            if (is_last)
                break;
            current_idx++;
        }

        if (!found)
            return false;
    }
    return current_idx == terminal_node_;
}

template <uint32_t GROUP_BITS, bool PATH_COMPRESS, bool TRACK_SHARING>
void CompactDawg<GROUP_BITS, PATH_COMPRESS, TRACK_SHARING>::LexicographicSearch(
    const std::string &start_bitstring, const std::string &end_bitstring,
    std::vector<std::string> *results) const
{
    if (!results)
        return;
    if (HasNoRoot())
        return;

    std::vector<std::string> start_key = ChunkBitstring(start_bitstring, '0');
    std::vector<std::string> end_key = ChunkBitstring(end_bitstring, '1');

    std::string current_path;
    current_path.reserve(start_bitstring.size());

    LexicographicSearchRecursive(root_index_, start_key, end_key, 0, true, true, current_path,
                                 results);
}

template <uint32_t GROUP_BITS, bool PATH_COMPRESS, bool TRACK_SHARING>
void CompactDawg<GROUP_BITS, PATH_COMPRESS, TRACK_SHARING>::LexicographicSearchRecursive(
    uint32_t current_idx, const std::vector<std::string> &start_key,
    const std::vector<std::string> &end_key, size_t depth, bool start_bound, bool end_bound,
    std::string &current_string, std::vector<std::string> *results) const
{
    if (current_idx == BUILD_TERMINAL_NODE || current_idx == terminal_node_) {
        results->push_back(current_string);
        return;
    }

    while (true) {
        if (current_idx >= is_last_.GetSizeInBits())
            return;

        uint32_t label_chunks = GetLabelChunks(current_idx);
        uint64_t label_bit_off = GetLabelBitOffset(current_idx);

        uint32_t offset = 0;
        if (!TryReadPackedTarget(current_idx, &offset))
            return;

        bool in_range = true;
        bool next_start_bound = start_bound;
        bool next_end_bound = end_bound;
        std::string full_label;
        full_label.reserve(label_chunks * GROUP_BITS);

        for (uint32_t c = 0; c < label_chunks; ++c) {
            size_t d = depth + c;
            std::string min_label = (next_start_bound && d < start_key.size())
                                        ? start_key[d]
                                        : std::string(GROUP_BITS, '0');
            std::string max_label = (next_end_bound && d < end_key.size())
                                        ? end_key[d]
                                        : std::string(GROUP_BITS, '1');

            std::string chunk_str;
            chunk_str.reserve(GROUP_BITS);
            uint64_t chunk_off = label_bit_off + static_cast<uint64_t>(c) * GROUP_BITS;
            for (size_t j = 0; j < GROUP_BITS; ++j) {
                if (chunk_off + j >= labels_.GetSizeInBits()) {
                    in_range = false;
                    break;
                }
                chunk_str += labels_.GetBit(chunk_off + j) ? '1' : '0';
            }
            if (!in_range)
                break;

            if (chunk_str < min_label || chunk_str > max_label) {
                in_range = false;
                break;
            }

            next_start_bound = next_start_bound && (chunk_str == min_label);
            next_end_bound = next_end_bound && (chunk_str == max_label);
            full_label += chunk_str;
        }

        bool is_last = is_last_.GetBit(current_idx);

        if (in_range) {
            size_t original_len = current_string.size();
            current_string += full_label;

            LexicographicSearchRecursive(offset, start_key, end_key, depth + label_chunks,
                                         next_start_bound, next_end_bound, current_string,
                                         results);

            current_string.resize(original_len);
        }

        if (is_last)
            break;
        current_idx++;
    }
}
