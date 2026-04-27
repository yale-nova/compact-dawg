#ifndef MD_TRIE_DATA_POINT_H
#define MD_TRIE_DATA_POINT_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>

#include "bitmap_utils.h"
#include "defs.h"
#include "ordered_types.h"

/**
 * Data point struct is used both to represent a data point and the return
 * "struct" for range search. Range search can either return a vector of
 * coordinates or a vector of primary keys.
 *
 * TODO(yash): split the coordinate_t to use 32 bits only (less brittle).
 *             And split off an ordered_t so we don't forget to transform
 *             ordered and regular coordinates...
 */
template <n_dimensions_t DIMENSION> class data_point
{
private:
    coordinate_t coordinates_[DIMENSION] = {0}; // 12.9 todo: add alignment

public:
    explicit data_point() { std::memset(coordinates_, '\0', sizeof(coordinate_t) * DIMENSION); }

    // Copy constructor
    data_point(const data_point &other)
    {
        std::memcpy(this->coordinates_, other.coordinates_, sizeof(coordinate_t) * DIMENSION);
    }

    // Move constructor
    data_point(data_point &&other) noexcept
    {
        std::memcpy(this->coordinates_, other.coordinates_, sizeof(coordinate_t) * DIMENSION);
    }

    // Copy assignment operator
    data_point &operator=(const data_point &other)
    {
        if (this != &other) {
            std::memcpy(this->coordinates_, other.coordinates_, sizeof(coordinate_t) * DIMENSION);
        }
        return *this;
    }

    // Move assignment operator
    data_point &operator=(data_point &&other) noexcept
    {
        if (this != &other) {
            std::memcpy(this->coordinates_, other.coordinates_, sizeof(coordinate_t) * DIMENSION);
        }
        return *this;
    }

    /// @brief Constructs from an array of coordinates
    explicit data_point(const coordinate_t coords[DIMENSION])
    {
        std::memcpy(coordinates_, coords, sizeof(coordinate_t) * DIMENSION);
    }

    // Equality so data_point can be used as a key in associative containers
    inline void clear() { std::memset(coordinates_, '\0', sizeof(coordinate_t) * DIMENSION); }

    // Equality so data_point can be used as a key in associative containers
    inline bool operator==(const data_point &other) const
    {
        return !std::memcmp(coordinates_, other.coordinates_, sizeof(coordinate_t) * DIMENSION);
    }

    // Less-than comparison based on morton code ordering at each trie level.
    // Used for sorting data points in morton order.
    inline bool operator<(const data_point &other) const
    {
        for (trie_level_t level = 0; level < MAX_TRIE_DEPTH; level++) {
            morton_t a = this->leaf_to_symbol(level);
            morton_t b = other.leaf_to_symbol(level);

            if (a < b) {
                return true;
            } else if (a > b) {
                return false;
            }
            // If a == b, continue to next level
        }

        // All levels are equal, so this point is not less than other.
        return false;
    }

    // Used to indicate intent: the output is encoded in an ordered format already
    inline ordered_coordinate_t get_ordered_coordinate(n_dimensions_t index) const
    {
        return static_cast<ordered_coordinate_t>(coordinates_[index]);
    }

    // Used to indicate intent: the input is encoded in an ordered format already
    inline void set_ordered_coordinate(n_dimensions_t index, ordered_coordinate_t value)
    {
        coordinates_[index] = value;
    }

    // This function automatically converts floats from their ordered representation
    inline float get_float_coordinate(n_dimensions_t index) const
    {
        return ordered_types::ordered_u32_to_float(
            static_cast<ordered_coordinate_t>(coordinates_[index]));
    }

    // This function automatically converts floats to their ordered representation
    inline void set_float_coordinate(n_dimensions_t index, float value)
    {
        coordinates_[index] = ordered_types::float_to_ordered_u32(value);
    }

    // Each dimension can be returned as a morton
    //
    // Each `data_point` can be encoded as a single-dimensional morton_code.
    // This function returns the `level`th symbol in that morton code.
    inline morton_t leaf_to_symbol(trie_level_t cur_trie_level) const
    {
        // The AVX implementations aren't correct for DIMS <64 at the moment.
        if constexpr (DIMENSION >= 64)
            return leaf_to_symbol_avx2(cur_trie_level);
        return leaf_to_symbol_noavx(cur_trie_level);
    }

    /// @brief Shift the point based on the already-determined morton bits.
    ///
    /// Suppose we are using this data point as the bounds to a range query.
    /// If our search algorithm is recursive (ie: searches different subtrees as
    /// we go on), then we may want to _narrow_ our range query bounds based on
    /// what symbols are already "locked in" for each subtree.
    ///
    /// Eg: we have an initial query bounds [0, 0] through [3, 3]. This turns
    ///     into morton encodings [0b0000, 0b1111]. Now if we are searching in the
    ///     subtree in position root[0b11] it means that the "0th bit"
    ///     of both of our dimensions must be 1 in this branch. So we can narrow our
    ///     search to exclude points with `coordinate_bits[i][0] = 1`.
    ///     The new bounds will be [2, 2] through [3, 3]...AKA morton interval
    ///     [0b1100, 0b1111]!
    ///
    /// This function changes the begin and end range points.
    static inline void shrink_query_bounds(data_point *start_range, data_point *end_range,
                                           const morton_t &current_symbol, trie_level_t trie_level)
    {
        if constexpr (DIMENSION >= 8)
            return shrink_query_bounds_avx2(start_range, end_range, current_symbol, trie_level);
        return shrink_query_bounds_noavx(start_range, end_range, current_symbol, trie_level);
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "[";
        for (n_dimensions_t i = 0; i < DIMENSION; i++) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "0x" << std::hex << std::setw(8) << std::setfill('0') << coordinates_[i];
        }
        oss << "]";
        return oss.str();
    }

private:
    // An implementation of `leaf_to_symbol` using plain sequential code.
    // (no data parallel instructions, such as AVX).
    inline morton_t leaf_to_symbol_noavx(trie_level_t cur_trie_level) const
    {
        morton_t result(uint64_t(0));
        n_dimensions_t cur_result_bit = DIMENSION - 1;
        trie_level_t offset = MAX_TRIE_DEPTH - cur_trie_level - 1;

        // Retrieve only the bits relevant to this dimension.
        for (size_t i = 0; i < DIMENSION; i++) {
            coordinate_t coordinate = coordinates_[i];
            bool bit = GETBIT(coordinate, offset);
            result.set_bit_unsafe(cur_result_bit, bit);
            cur_result_bit--;
        }

        return result;
    }

    /// @brief Ultra-optimized AVX2 version of leaf_to_symbol
    ///
    /// Processes 16 coordinates per iteration using two 256-bit loads,
    /// filling a 64-bit block in just 4 iterations instead of 8.
    /// Same optimization pattern as AVX512 v2 but using AVX2.
    inline morton_t leaf_to_symbol_avx2(trie_level_t cur_trie_level) const
        requires(DIMENSION >= 64)
    {
        constexpr n_dimensions_t dimension = DIMENSION;
        constexpr size_t num_blocks = dimension / 64;
        trie_level_t offset = MAX_TRIE_DEPTH - cur_trie_level - 1;

        // Static LUT for 8-bit reversal
        static const uint8_t bit_reverse_lut[256] = {
            0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0,
            0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8,
            0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94,
            0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC,
            0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2,
            0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA,
            0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86,
            0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
            0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE,
            0x7E, 0xFE, 0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1,
            0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99,
            0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5,
            0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD,
            0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3,
            0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B, 0x8B,
            0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
            0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7,
            0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF,
            0x3F, 0xBF, 0x7F, 0xFF};

        const uint32_t bit_mask = 1U << offset;
        const __m256i v_bit_mask = _mm256_set1_epi32(static_cast<int32_t>(bit_mask));
        const __m256i v_zero = _mm256_setzero_si256();

        morton_t result(uint64_t(0));
        uint64_t *result_blocks = result.blocks();

        // Helper lambda to extract 8-bit mask from 8 coords
        auto get_mask8 = [&](size_t j) -> uint8_t {
            __m256i coords =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&coordinates_[j]));
            __m256i masked = _mm256_and_si256(coords, v_bit_mask);
            __m256i cmp = _mm256_cmpeq_epi32(masked, v_zero);
            int mask8 = (~_mm256_movemask_ps(_mm256_castsi256_ps(cmp))) & 0xFF;
            return bit_reverse_lut[mask8];
        };

        // Process 64 coordinates per block with fully unrolled inner loop
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            size_t coord_base = block_idx * 64;

            // 8 groups of 8 coords each, fully unrolled
            uint8_t b7 = get_mask8(coord_base);
            uint8_t b6 = get_mask8(coord_base + 8);
            uint8_t b5 = get_mask8(coord_base + 16);
            uint8_t b4 = get_mask8(coord_base + 24);
            uint8_t b3 = get_mask8(coord_base + 32);
            uint8_t b2 = get_mask8(coord_base + 40);
            uint8_t b1 = get_mask8(coord_base + 48);
            uint8_t b0 = get_mask8(coord_base + 56);

            uint64_t block_val =
                (static_cast<uint64_t>(b7) << 56) | (static_cast<uint64_t>(b6) << 48) |
                (static_cast<uint64_t>(b5) << 40) | (static_cast<uint64_t>(b4) << 32) |
                (static_cast<uint64_t>(b3) << 24) | (static_cast<uint64_t>(b2) << 16) |
                (static_cast<uint64_t>(b1) << 8) | static_cast<uint64_t>(b0);

            result_blocks[num_blocks - 1 - block_idx] = block_val;
        }

        return result;
    }

    // An implementation of `shrink_query_bounds()` using plain sequential code.
    // (no data parallel instructions, such as AVX).
    static inline void shrink_query_bounds_noavx(data_point *start_range, data_point *end_range,
                                                 const morton_t &current_symbol,
                                                 trie_level_t trie_level)
    {
        constexpr n_dimensions_t dimension = DIMENSION;
        trie_level_t offset = MAX_TRIE_DEPTH - trie_level - 1U;
        assert(MAX_TRIE_DEPTH >= trie_level);

        for (n_dimensions_t j = 0; j < dimension; j++) {
            ordered_coordinate_t start_coordinate = start_range->get_ordered_coordinate(j);
            ordered_coordinate_t end_coordinate = end_range->get_ordered_coordinate(j);

            // Before "offset" point, we know that all bits are the same in (start) and (end).
            // This function's goal is to make sure 1 more "symbol"'s worth of
            // bits match between (start) and (end).
            //
            // Eg:
            // start: 0011_001
            // end:   0011_101
            //
            // We have already made sure that the first 4 bits match. (in this subtree).
            // Now, we need to make one more "symbol"'s worth of bits match.
            //
            // Dimension = 2, so let's "match" 1 additional bit from each input dimension.
            //
            // new_start:  0011_0_00
            // new_end:    0011_1_11
            n_dimensions_t symbol_offset = NUM_DIMENSIONS - j - 1;

            bool start_bit = GETBIT(start_coordinate, offset);
            bool end_bit = GETBIT(end_coordinate, offset);
            bool symbol_bit = current_symbol.get_bit_unsafe(symbol_offset);

            // Bring the start of the search range to second half
            if (symbol_bit && !start_bit) {
                start_coordinate = start_coordinate & low_bits_unset[offset];
                SETBIT(start_coordinate, offset);
            }
            // Bring the end of the search range to first half
            if (!symbol_bit && end_bit) {
                end_coordinate = end_coordinate | low_bits_set[offset];
                CLRBIT(end_coordinate, offset);
            }
            start_range->set_ordered_coordinate(j, start_coordinate);
            end_range->set_ordered_coordinate(j, end_coordinate);
        }
    }

    static inline void shrink_query_bounds_avx2(data_point *start_range, data_point *end_range,
                                                const morton_t &current_symbol,
                                                trie_level_t trie_level)
        requires(DIMENSION >= 8)
    {
        constexpr n_dimensions_t dimension = DIMENSION;
        trie_level_t offset = MAX_TRIE_DEPTH - trie_level - 1U;

        // Precompute 32-bit masks
        const uint32_t bit_mask_32 = 1U << offset;
        const uint32_t low_mask_32 = static_cast<uint32_t>(low_bits_unset[offset]);
        const uint32_t fill_mask_32 = static_cast<uint32_t>(low_bits_set[offset]);

        // Broadcast masks to 256-bit vectors (8 x 32-bit)
        const __m256i v_bit_mask = _mm256_set1_epi32(static_cast<int32_t>(bit_mask_32));
        const __m256i v_low_mask = _mm256_set1_epi32(static_cast<int32_t>(low_mask_32));
        const __m256i v_fill_mask = _mm256_set1_epi32(static_cast<int32_t>(fill_mask_32));
        const __m256i v_zero = _mm256_setzero_si256();
        const __m256i v_all_ones = _mm256_set1_epi32(-1); // Hoisted
        // Reversed bit positions for symbol extraction (lane0=bit7, lane1=bit6, etc.)
        const __m256i v_bit_positions =
            _mm256_setr_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

        // Get direct pointer access to coordinate arrays (now uint32_t!)
        coordinate_t *start_coords = start_range->coordinates_;
        coordinate_t *end_coords = end_range->coordinates_;
        const uint64_t *symbol_blocks = current_symbol.blocks();

        // Process 8 dimensions at a time with direct SIMD loads/stores
        for (size_t j = 0; j < dimension; j += 8) {
            // Direct load of 8 x 32-bit coordinates (no packing needed!)
            __m256i start_vec =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&start_coords[j]));
            __m256i end_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&end_coords[j]));

            // ===== VECTORIZED SYMBOL BIT EXTRACTION (BRANCHLESS) =====
            // For dims [j, j+1, ..., j+7], symbol bits are at morton positions:
            //   [DIM-1-j, DIM-2-j, ..., DIM-8-j] (8 consecutive bits, descending)

            // Calculate position of the lowest bit we need (for dim j+7)
            n_dimensions_t low_bit_pos = dimension - 8 - j;
            size_t block_idx = low_bit_pos / 64;
            size_t bit_start = low_bit_pos % 64;

            // Extract 8 bits - handle block boundary
            uint64_t bits8;
            if (bit_start + 8 <= 64) {
                // All 8 bits fit in one block (common case)
                bits8 = (symbol_blocks[block_idx] >> bit_start) & 0xFFULL;
            } else {
                // Bits span two blocks (happens every 8th iteration when near boundary)
                size_t bits_in_first = 64 - bit_start;
                uint64_t first = symbol_blocks[block_idx] >> bit_start;
                uint64_t second =
                    symbol_blocks[block_idx + 1] & ((1ULL << (8 - bits_in_first)) - 1);
                bits8 = first | (second << bits_in_first);
            }

            // Expand 8 bits to 8 x 32-bit lanes using AVX2 with reversed bit positions
            __m256i v_bits = _mm256_set1_epi32(static_cast<int32_t>(bits8));
            __m256i symbol_vec = _mm256_and_si256(v_bits, v_bit_positions);

            // Create masks: symbol_bit!=0 (set) and symbol_bit==0 (not set)
            __m256i symbol_is_zero = _mm256_cmpeq_epi32(symbol_vec, v_zero);
            __m256i symbol_mask = _mm256_andnot_si256(symbol_is_zero, v_all_ones);
            __m256i not_symbol_mask = symbol_is_zero;

            // Check start_bit and end_bit for each coordinate
            __m256i start_bits = _mm256_and_si256(start_vec, v_bit_mask);
            __m256i end_bits = _mm256_and_si256(end_vec, v_bit_mask);

            // Conditions
            __m256i start_bit_zero = _mm256_cmpeq_epi32(start_bits, v_zero);
            __m256i end_bit_set =
                _mm256_andnot_si256(_mm256_cmpeq_epi32(end_bits, v_zero), v_all_ones);

            // Compute new values
            __m256i new_start_val =
                _mm256_or_si256(_mm256_and_si256(start_vec, v_low_mask), v_bit_mask);
            __m256i new_end_val =
                _mm256_andnot_si256(v_bit_mask, _mm256_or_si256(end_vec, v_fill_mask));

            // Apply conditionally using blend
            // Note: blendv_epi8 is byte-level, but works here because apply_start/apply_end
            // are guaranteed to be 0xFFFFFFFF or 0x00000000 per 32-bit lane (from cmpeq results)
            __m256i apply_start = _mm256_and_si256(symbol_mask, start_bit_zero);
            __m256i apply_end = _mm256_and_si256(not_symbol_mask, end_bit_set);

            start_vec = _mm256_blendv_epi8(start_vec, new_start_val, apply_start);
            end_vec = _mm256_blendv_epi8(end_vec, new_end_val, apply_end);

            // Direct store of 8 x 32-bit coordinates (no unpacking needed!)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&start_coords[j]), start_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&end_coords[j]), end_vec);
        }
    }
};

// Provide a std::hash specialization so data_point can be used as key in
// unordered_map/unordered_set. This is an allowed user-defined specialization
// of std::hash for a user type.
namespace std
{
template <n_dimensions_t DIM> struct hash<data_point<DIM>> {
    size_t operator()(const data_point<DIM> &dp) const noexcept
    {
        // Use a variation of boost::hash_combine
        size_t seed = 0;
        std::hash<coordinate_t> h;
        for (n_dimensions_t i = 0; i < DIM; ++i) {
            seed ^=
                h(dp.get_ordered_coordinate(i)) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
} // namespace std

#endif // MD_TRIE_DATA_POINT_H
