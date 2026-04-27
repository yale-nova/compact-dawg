#pragma once

#include "../dynamic_dawg.h"

inline bool DynamicDawg::Contains(const std::string &bitstring) const
{
    if (bitstring.size() != plan_.total_bits)
        return false;

    std::vector<std::string> key = plan_.chunk_key(bitstring);

    if (root_index_ == BUILD_TERMINAL_NODE || root_index_ == terminal_node_)
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

            uint32_t label_segments = GetLabelSegmentCount(current_idx);
            uint32_t label_bits = PlanBitLength(key_pos, label_segments);
            uint64_t label_bit_off = GetLabelBitOffset(current_idx);

            uint32_t offset = 0;
            if (!TryReadPackedTarget(current_idx, &offset))
                return false;

            bool match = true;
            uint32_t bits_consumed = 0;
            size_t chunks_consumed = 0;

            while (chunks_consumed < label_segments && bits_consumed < label_bits &&
                   key_pos + chunks_consumed < key.size()) {
                const std::string &kc = key[key_pos + chunks_consumed];
                uint32_t chunk_width = static_cast<uint32_t>(kc.size());

                uint32_t bits_to_check = std::min(chunk_width, label_bits - bits_consumed);

                for (uint32_t j = 0; j < bits_to_check; ++j) {
                    if (label_bit_off + bits_consumed + j >= labels_.GetSizeInBits()) {
                        match = false;
                        break;
                    }
                    bool bit = labels_.GetBit(label_bit_off + bits_consumed + j);
                    if ((bit ? '1' : '0') != kc[j]) {
                        match = false;
                        break;
                    }
                }

                if (!match)
                    break;

                bits_consumed += bits_to_check;
                chunks_consumed++;
            }

            if (match && (bits_consumed < label_bits || chunks_consumed != label_segments)) {
                match = false;
            }

            bool is_last = is_last_.GetBit(current_idx);

            if (match) {
                key_pos += chunks_consumed;
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

inline void DynamicDawg::LexicographicSearch(const std::string &start_bitstring,
                                             const std::string &end_bitstring,
                                             std::vector<std::string> *results) const
{
    if (!results)
        return;
    if (root_index_ == BUILD_TERMINAL_NODE || root_index_ == terminal_node_)
        return;

    std::vector<std::string> start_key = plan_.chunk_key(start_bitstring);
    std::vector<std::string> end_key = plan_.chunk_key(end_bitstring);

    for (size_t i = 0; i < end_key.size(); ++i) {
        uint32_t expected_width = plan_.widths[i];
        while (end_key[i].size() < expected_width)
            end_key[i] += '1';
    }

    std::string current_path;
    current_path.reserve(start_bitstring.size());

    LexSearchRecursive(root_index_, start_key, end_key, 0, true, true, current_path, results);
}

inline void DynamicDawg::LexSearchRecursive(uint32_t current_idx,
                                            const std::vector<std::string> &start_key,
                                            const std::vector<std::string> &end_key, size_t depth,
                                            bool start_bound, bool end_bound,
                                            std::string &current_string,
                                            std::vector<std::string> *results) const
{
    if (current_idx == BUILD_TERMINAL_NODE || current_idx == terminal_node_) {
        results->push_back(current_string);
        return;
    }

    while (true) {
        if (current_idx >= is_last_.GetSizeInBits())
            return;

        uint32_t label_segments = GetLabelSegmentCount(current_idx);
        uint32_t label_bits = PlanBitLength(depth, label_segments);
        uint64_t label_bit_off = GetLabelBitOffset(current_idx);

        uint32_t offset = 0;
        if (!TryReadPackedTarget(current_idx, &offset))
            return;

        bool in_range = true;
        bool next_start_bound = start_bound;
        bool next_end_bound = end_bound;
        std::string full_label;
        full_label.reserve(label_bits);

        uint32_t bits_read = 0;
        size_t d = depth;

        uint32_t segments_read = 0;
        while (segments_read < label_segments && bits_read < label_bits && d < plan_.widths.size()) {
            uint32_t chunk_width = plan_.widths[d];
            uint32_t bits_to_read = std::min(chunk_width, label_bits - bits_read);

            std::string min_label, max_label;
            if (next_start_bound && d < start_key.size()) {
                min_label = start_key[d].substr(0, bits_to_read);
            } else {
                min_label = std::string(bits_to_read, '0');
            }
            if (next_end_bound && d < end_key.size()) {
                max_label = end_key[d].substr(0, bits_to_read);
            } else {
                max_label = std::string(bits_to_read, '1');
            }

            std::string chunk_str;
            chunk_str.reserve(bits_to_read);
            for (uint32_t j = 0; j < bits_to_read; ++j) {
                uint64_t pos = label_bit_off + bits_read + j;
                if (pos >= labels_.GetSizeInBits()) {
                    in_range = false;
                    break;
                }
                chunk_str += labels_.GetBit(pos) ? '1' : '0';
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

            bits_read += bits_to_read;
            d++;
            segments_read++;
        }

        if (segments_read != label_segments || bits_read != label_bits) {
            in_range = false;
        }

        bool is_last = is_last_.GetBit(current_idx);

        if (in_range) {
            size_t original_len = current_string.size();
            current_string += full_label;

            LexSearchRecursive(offset, start_key, end_key, d, next_start_bound, next_end_bound,
                               current_string, results);

            current_string.resize(original_len);
        }

        if (is_last)
            break;
        current_idx++;
    }
}
