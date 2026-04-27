// Templated Morton key storing an arbitrary number of bits in 64-bit blocks.
#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <x86intrin.h> // AVX2 intrinsics

// Disables compile warnings with -Wpedantic (int128 is a GCC extension).
__extension__ typedef __int128 int128;
__extension__ typedef unsigned __int128 uint128;

template <size_t DIM_BITS> class morton_key
{
    static_assert(DIM_BITS > 0, "DIM_BITS must be > 0");
    // For DIM_BITS <= 64, allow any positive bit width.
    // For DIM_BITS  > 64, require multiple of 64 as per your assumption.
    static_assert((DIM_BITS <= 64) || (DIM_BITS % 64 == 0),
                  "If DIM_BITS > 64, it must be a multiple of 64");

public:
    static constexpr size_t bits = DIM_BITS;
    static constexpr size_t bits_per_block = 64;
    static constexpr size_t num_blocks = (bits + bits_per_block - 1) / bits_per_block;

    using block_t = uint64_t;

private:
    // Plain C-array storage (little-endian block order: data_[0] is least
    // significant)
    block_t data_[num_blocks]{}; // zero-initialized
    bool null_{false};

    // Mask for the top (most significant) block to ensure unused high bits are
    // zero
    static constexpr block_t top_block_mask()
    {
        const size_t rem = bits % bits_per_block;
        return rem == 0 ? ~block_t(0) : (block_t(1) << rem) - 1;
    }

    static void zero_blocks(block_t *p) noexcept
    {
        std::memset(p, 0, sizeof(block_t) * num_blocks);
    }

    static constexpr block_t top_excess_mask() noexcept { return ~top_block_mask(); }

public:
    static void mask_top(block_t *p) noexcept { p[num_blocks - 1] &= top_block_mask(); }
    // default constructs to zero
    morton_key() = default;

    static morton_key<DIM_BITS> zero()
    {
        morton_key key;
        key.null_ = false;
        return key;
    }

    static morton_key<DIM_BITS> null()
    {
        morton_key key;
        key.null_ = true;
        return key;
    }

    // Returns the maximum possible morton_key (all valid bits set to 1)
    static morton_key<DIM_BITS> maximum()
    {
        morton_key key;
        for (size_t i = 0; i < num_blocks; ++i)
            key.data_[i] = ~block_t(0);
        mask_top(key.data_);
        key.null_ = false;
        return key;
    }

    /// @brief Factory: creates a morton_key containing (v << shift_bits) in O(1)
    ///
    /// This is much more efficient than `morton_key(v) << shift_bits` because:
    /// - `morton_key(v) << shift` does: zero_blocks + set data_[0] + mask_top + O(num_blocks) shift
    /// loop
    /// - This function does: default init + direct write to 1-2 blocks + conditional mask_top
    ///
    /// For CHUNK_WIDTH_SHIFT * H_LEVEL = 1008 bits (block 15, offset 48):
    /// - Old way: 32+ block operations in shift loops
    /// - This way: 2 block writes
    ///
    /// @param v The uint64_t value to place at shifted position
    /// @param shift_bits Number of bits to shift left
    /// @return New morton_key containing (v << shift_bits)
    static morton_key from_shifted_u64(uint64_t v, size_t shift_bits) noexcept
    {
        morton_key key; // default {} aggregate init - NRVO eligible

        if (v == 0 || shift_bits >= bits)
            return key;

        const size_t block_idx = shift_bits / bits_per_block;
        const size_t bit_offset = shift_bits % bits_per_block;

        if (block_idx >= num_blocks)
            return key;

        // Directly place value at correct position - O(1)
        key.data_[block_idx] = v << bit_offset;

        // Handle overflow into next block if bit_offset != 0
        if (bit_offset != 0 && block_idx + 1 < num_blocks) {
            key.data_[block_idx + 1] = v >> (bits_per_block - bit_offset);
        }

        // Only mask if we touched the top block
        if (block_idx == num_blocks - 1 || (bit_offset != 0 && block_idx + 1 == num_blocks - 1)) {
            mask_top(key.data_);
        }

        return key;
    }

    /// @brief In-place set: this = (v << shift_bits), assuming this is zero
    ///
    /// This is the member function equivalent of from_shifted_u64 but modifies
    /// the caller in-place instead of returning a new object. Assumes this is
    /// already zero-initialized.
    ///
    /// @param v The uint64_t value to place at shifted position
    /// @param shift_bits Number of bits to shift left
    /// @return Reference to this (for chaining)
    morton_key &set_from_shifted_u64(uint64_t v, size_t shift_bits) noexcept
    {

        if (v == 0 || shift_bits >= bits)
            return *this;

        const size_t block_idx = shift_bits / bits_per_block;
        const size_t bit_offset = shift_bits % bits_per_block;

        if (block_idx >= num_blocks)
            return *this;

        // Directly place value at correct position - O(1)
        data_[block_idx] = v << bit_offset;

        // Handle overflow into next block if bit_offset != 0
        if (bit_offset != 0 && block_idx + 1 < num_blocks) {
            data_[block_idx + 1] = v >> (bits_per_block - bit_offset);
        }

        // Only mask if we touched the top block
        if (block_idx == num_blocks - 1 || (bit_offset != 0 && block_idx + 1 == num_blocks - 1)) {
            mask_top(data_);
        }

        return *this;
    }

    /// @brief In-place replace: clears bits at [shift_bits, shift_bits + bit_count), then sets to v
    ///
    /// This is a fused "mask and set" operation for when you need to replace bits at a
    /// specific level during DFS traversal. It clears the bits in the range first,
    /// then ORs in the new value.
    ///
    /// @param v The uint64_t value to place at shifted position
    /// @param shift_bits Starting bit position to replace
    /// @param bit_count Number of bits to clear and replace (must be <= 64)
    /// @return Reference to this (for chaining)
    morton_key &replace_shifted_u64(uint64_t v, size_t shift_bits, size_t bit_count) noexcept
    {
        assert(bit_count <= 64);

        if (shift_bits >= bits)
            return *this;

        const size_t block_idx = shift_bits / bits_per_block;
        const size_t bit_offset = shift_bits % bits_per_block;

        if (block_idx >= num_blocks)
            return *this;

        // Create mask for the bits we want to clear
        const uint64_t clear_mask_low =
            (bit_count >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bit_count) - 1);

        // Clear bits in first block
        data_[block_idx] &= ~(clear_mask_low << bit_offset);

        // Set new value in first block
        data_[block_idx] |= (v << bit_offset);

        // Handle overflow into next block if bit_offset != 0
        const size_t bits_in_first = bits_per_block - bit_offset;
        if (bit_count > bits_in_first && block_idx + 1 < num_blocks) {
            const size_t bits_in_second = bit_count - bits_in_first;
            const uint64_t clear_mask_high =
                (bits_in_second >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bits_in_second) - 1);

            // Clear bits in second block
            data_[block_idx + 1] &= ~clear_mask_high;

            // Set new value in second block
            data_[block_idx + 1] |= (v >> bits_in_first);
        }

        // Only mask if we touched the top block
        if (block_idx == num_blocks - 1 ||
            (bit_count > bits_in_first && block_idx + 1 == num_blocks - 1)) {
            mask_top(data_);
        }

        return *this;
    }

    /// @brief In-place clear: clears bits at [shift_bits, shift_bits + bit_count)
    ///
    /// Use this when you only need to mask out bits without setting new values.
    ///
    /// @param shift_bits Starting bit position to clear
    /// @param bit_count Number of bits to clear (must be <= 64)
    /// @return Reference to this (for chaining)
    morton_key &clear_bits_at(size_t shift_bits, size_t bit_count) noexcept
    {
        assert(bit_count <= 64);

        if (shift_bits >= bits)
            return *this;

        const size_t block_idx = shift_bits / bits_per_block;
        const size_t bit_offset = shift_bits % bits_per_block;

        if (block_idx >= num_blocks)
            return *this;

        // Create mask for the bits we want to clear
        const uint64_t clear_mask_low =
            (bit_count >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bit_count) - 1);

        // Clear bits in first block
        data_[block_idx] &= ~(clear_mask_low << bit_offset);

        // Handle overflow into next block
        const size_t bits_in_first = bits_per_block - bit_offset;
        if (bit_count > bits_in_first && block_idx + 1 < num_blocks) {
            const size_t bits_in_second = bit_count - bits_in_first;
            const uint64_t clear_mask_high =
                (bits_in_second >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bits_in_second) - 1);

            // Clear bits in second block
            data_[block_idx + 1] &= ~clear_mask_high;
        }

        return *this;
    }

    /// @brief In-place add: this += (v << shift_bits)
    ///
    /// This is the most efficient path for computing parent + (small_int << large_shift).
    /// It directly modifies this morton_key by adding the shifted value in-place,
    /// avoiding temporary object creation and 16-block zero initialization.
    ///
    /// @param v The uint64_t value to shift and add (must fit in 64 bits)
    /// @param shift_bits Number of bits to shift left before adding
    /// @return Reference to this (for chaining)
    morton_key &add_shifted_u64(uint64_t v, size_t shift_bits) noexcept
    {
        if (v == 0 || shift_bits >= bits)
            return *this;

        const size_t block_idx = shift_bits / bits_per_block;
        const size_t bit_offset = shift_bits % bits_per_block;

        if (block_idx >= num_blocks)
            return *this;

        // Add v << bit_offset to block[block_idx], propagating carry
        uint64_t to_add_low = v << bit_offset;
        uint64_t to_add_high = (bit_offset != 0) ? (v >> (bits_per_block - bit_offset)) : 0;

        // Add low part with carry propagation
        uint128 sum = static_cast<uint128>(data_[block_idx]) + to_add_low;
        data_[block_idx] = static_cast<uint64_t>(sum);
        uint64_t carry = static_cast<uint64_t>(sum >> 64);

        // Add high part (if any) to next block
        if (to_add_high != 0 && block_idx + 1 < num_blocks) {
            sum = static_cast<uint128>(data_[block_idx + 1]) + to_add_high + carry;
            data_[block_idx + 1] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        } else if (carry != 0 && block_idx + 1 < num_blocks) {
            sum = static_cast<uint128>(data_[block_idx + 1]) + carry;
            data_[block_idx + 1] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }

        // Continue propagating carry if needed
        for (size_t i = block_idx + 2; i < num_blocks && carry != 0; ++i) {
            sum = static_cast<uint128>(data_[i]) + carry;
            data_[i] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }

        mask_top(data_);
        return *this;
    }

    /// @brief Fused copy + add: returns base + (v << shift_bits) efficiently
    ///
    /// Uses AVX2 for fast copying of the base morton_key (128 bytes), then
    /// adds the shifted value in-place. This is faster than:
    ///   morton_key result = base; result.add_shifted_u64(v, shift);
    /// because it avoids the default constructor's zero-initialization.
    ///
    /// @param base The base morton_key to copy from
    /// @param v The uint64_t value to shift and add
    /// @param shift_bits Number of bits to shift left before adding
    /// @return New morton_key containing base + (v << shift_bits)
    static morton_key copy_and_add_shifted_u64(const morton_key &base, uint64_t v,
                                               size_t shift_bits) noexcept
    {
        morton_key r;

        // AVX2 fast copy: 4 blocks (256 bits) per iteration
        if constexpr (num_blocks >= 4) {
            size_t i = 0;
            for (; i + 4 <= num_blocks; i += 4) {
                __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&base.data_[i]));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&r.data_[i]), val);
            }
            // Handle remaining blocks (0-3)
            for (; i < num_blocks; ++i) {
                r.data_[i] = base.data_[i];
            }
        } else {
            // Scalar fallback for small keys
            for (size_t i = 0; i < num_blocks; ++i) {
                r.data_[i] = base.data_[i];
            }
        }
        r.null_ = base.null_;

        // Now add the shifted value in place
        r.add_shifted_u64(v, shift_bits);
        return r;
    }

    bool is_null() const noexcept { return null_; }

    // construct from a single uint64_t value
    explicit morton_key(block_t v) noexcept
    {
        zero_blocks(data_);
        data_[0] = v;
        mask_top(data_);
    }

    // access raw blocks pointer
    const block_t *blocks() const noexcept { return data_; }
    block_t *blocks() noexcept { return data_; }

    // get bit at position i (0 = least significant bit).
    // Doesn't perform bounds checks.
    bool get_bit_unsafe(size_t i) const noexcept
    {
        const size_t b = i / bits_per_block;
        const size_t off = i % bits_per_block;
        return (data_[b] >> off) & block_t(1);
    }

    // get bit at position i (0 = least significant bit)
    bool get_bit(size_t i) const
    {
        if (i >= bits)
            throw std::out_of_range("bit index out of range");
        get_bit_unsafe(i);
    }

    // set bit at position i. Doesn't provide bounds checking.
    inline void set_bit_unsafe(size_t i, bool v) noexcept
    {
        const size_t b = i / bits_per_block;
        const size_t off = i % bits_per_block;
        if (v)
            data_[b] |= (block_t(1) << off);
        else
            data_[b] &= ~(block_t(1) << off);
    }
    // set bit at position i
    void set_bit(size_t i, bool v)
    {
        if (i >= bits)
            throw std::out_of_range("bit index out of range");
        set_bit_unsafe(i, v);
    }

    // clear all bits (and clear null flag)
    void clear() noexcept
    {
        zero_blocks(data_);
        null_ = false;
    }

    // bitwise operators
    morton_key operator~() const noexcept
    {
        morton_key r;
        for (size_t i = 0; i < num_blocks; ++i)
            r.data_[i] = ~data_[i];
        mask_top(r.data_);
        r.null_ = null_;
        return r;
    }

    morton_key &operator&=(const morton_key &o) noexcept
    {
        for (size_t i = 0; i < num_blocks; ++i)
            data_[i] &= o.data_[i];
        // & never introduces 1s in masked bits, so no need to mask, but safe:
        mask_top(data_);
        null_ = null_ && o.null_;
        return *this;
    }

    morton_key &operator|=(const morton_key &o) noexcept
    {
        for (size_t i = 0; i < num_blocks; ++i)
            data_[i] |= o.data_[i];
        mask_top(data_);
        null_ = null_ || o.null_;
        return *this;
    }

    morton_key &operator^=(const morton_key &o) noexcept
    {
        for (size_t i = 0; i < num_blocks; ++i)
            data_[i] ^= o.data_[i];
        mask_top(data_);
        null_ = (null_ != o.null_); // xor the null flags
        return *this;
    }

    friend morton_key operator&(morton_key a, const morton_key &b) noexcept
    {
        a &= b;
        return a;
    }
    friend morton_key operator|(morton_key a, const morton_key &b) noexcept
    {
        a |= b;
        return a;
    }
    friend morton_key operator^(morton_key a, const morton_key &b) noexcept
    {
        a ^= b;
        return a;
    }

    /// @brief Fused XNOR: computes ~(a ^ b) directly with AVX2
    ///
    /// This is the hotpath for: morton_t bound_magic = ~(start_symbol ^ end_symbol)
    /// Avoids creating temporary morton_t objects from separate ^ and ~ operations.
    static morton_key xnor(const morton_key &a, const morton_key &b) noexcept
    {
        assert(!a.null_ && !b.null_);
        morton_key r;

        if constexpr (num_blocks >= 4) {
            // AVX2 path: process 4 blocks (256 bits) per iteration
            const __m256i all_ones = _mm256_set1_epi64x(-1LL);
            size_t i = 0;
            for (; i + 4 <= num_blocks; i += 4) {
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a.data_[i]));
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b.data_[i]));
                // XNOR = NOT(XOR) = XOR with all-ones after XOR
                __m256i vxor = _mm256_xor_si256(va, vb);
                __m256i vxnor = _mm256_xor_si256(vxor, all_ones);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&r.data_[i]), vxnor);
            }
            // Handle remaining blocks (0-3)
            for (; i < num_blocks; ++i) {
                r.data_[i] = ~(a.data_[i] ^ b.data_[i]);
            }
        } else {
            // Scalar fallback for small keys
            for (size_t i = 0; i < num_blocks; ++i) {
                r.data_[i] = ~(a.data_[i] ^ b.data_[i]);
            }
        }

        mask_top(r.data_);
        r.null_ = false;
        return r;
    }

    // comparisons (lexicographic by most-significant block), also compare null_
    // first
    friend bool operator==(const morton_key &a, const morton_key &b) noexcept
    {
        // assert(!a.null_ && !b.null_);

        // memcmp is fine here because we always mask the top block
        return std::memcmp(a.data_, b.data_, sizeof(block_t) * num_blocks) == 0;
    }

    /// @brief Fused masked inequality: checks if (a & mask) != (b & mask)
    ///
    /// This is equivalent to ((a ^ b) & mask) != 0, but avoids creating temporaries.
    /// Use this for the hot path: `(start_symbol & bound_magic) != (current_symbol & bound_magic)`
    ///
    /// @return true if differ in any bit position where mask is set
    static bool masked_not_equal(const morton_key &a, const morton_key &b,
                                 const morton_key &mask) noexcept
    {
        assert(!a.null_ && !b.null_ && !mask.null_);

        // AVX2 optimized path: fused (a ^ b) & mask, check if any bit is set
        if constexpr (num_blocks >= 4) {
            size_t i = 0;
            for (; i + 4 <= num_blocks; i += 4) {
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a.data_[i]));
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b.data_[i]));
                __m256i vm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&mask.data_[i]));

                // Compute (a ^ b) & mask
                __m256i diff = _mm256_and_si256(_mm256_xor_si256(va, vb), vm);

                // If any bit is set, they differ (early exit)
                if (!_mm256_testz_si256(diff, diff))
                    return true;
            }
            // Handle remaining blocks (0-3)
            for (; i < num_blocks; ++i) {
                if ((a.data_[i] ^ b.data_[i]) & mask.data_[i])
                    return true;
            }
            return false;
        } else {
            // Scalar fallback for small keys
            for (size_t i = 0; i < num_blocks; ++i) {
                if ((a.data_[i] ^ b.data_[i]) & mask.data_[i])
                    return true;
            }
            return false;
        }
    }

    /// @brief Alternative: accumulate all diffs, check once at end (no early exit)
    ///
    /// This removes branches per iteration but loses early termination.
    /// May be faster when most comparisons go through all blocks without finding a difference.
    static bool masked_not_equal_v2(const morton_key &a, const morton_key &b,
                                    const morton_key &mask) noexcept
    {
        assert(!a.null_ && !b.null_ && !mask.null_);

        if constexpr (num_blocks >= 4) {
            __m256i acc = _mm256_setzero_si256();
            size_t i = 0;
            for (; i + 4 <= num_blocks; i += 4) {
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a.data_[i]));
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b.data_[i]));
                __m256i vm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&mask.data_[i]));

                // Compute (a ^ b) & mask and accumulate
                __m256i diff = _mm256_and_si256(_mm256_xor_si256(va, vb), vm);
                acc = _mm256_or_si256(acc, diff);
            }
            // Handle remaining blocks (0-3) - accumulate into scalar
            uint64_t tail_acc = 0;
            for (; i < num_blocks; ++i) {
                tail_acc |= (a.data_[i] ^ b.data_[i]) & mask.data_[i];
            }
            // Check if any bit is set in either accumulator
            return !_mm256_testz_si256(acc, acc) || (tail_acc != 0);
        } else {
            uint64_t acc = 0;
            for (size_t i = 0; i < num_blocks; ++i) {
                acc |= (a.data_[i] ^ b.data_[i]) & mask.data_[i];
            }
            return acc != 0;
        }
    }

    /// @brief AVX-512 version: processes 8 blocks (512 bits) per iteration
    ///
    /// For 1024 dims (16 blocks): only 2 iterations vs 4 with AVX2
    __attribute__((target("avx512f"))) static bool
    masked_not_equal_avx512(const morton_key &a, const morton_key &b,
                            const morton_key &mask) noexcept
    {
        assert(!a.null_ && !b.null_ && !mask.null_);

        if constexpr (num_blocks >= 8) {
            size_t i = 0;
            for (; i + 8 <= num_blocks; i += 8) {
                __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&a.data_[i]));
                __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&b.data_[i]));
                __m512i vm = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&mask.data_[i]));

                // Compute (a ^ b) & mask
                __m512i diff = _mm512_and_si512(_mm512_xor_si512(va, vb), vm);

                // If any bit is set, they differ (early exit)
                // _mm512_test_epi64_mask returns 0 if all lanes are zero
                if (_mm512_test_epi64_mask(diff, diff) != 0)
                    return true;
            }
            // Handle remaining 4 blocks with AVX2 if needed
            if constexpr (num_blocks % 8 >= 4) {
                if (i + 4 <= num_blocks) {
                    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a.data_[i]));
                    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b.data_[i]));
                    __m256i vm =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&mask.data_[i]));
                    __m256i diff = _mm256_and_si256(_mm256_xor_si256(va, vb), vm);
                    if (!_mm256_testz_si256(diff, diff))
                        return true;
                    i += 4;
                }
            }
            // Handle remaining blocks (0-3) scalar
            for (; i < num_blocks; ++i) {
                if ((a.data_[i] ^ b.data_[i]) & mask.data_[i])
                    return true;
            }
            return false;
        } else if constexpr (num_blocks >= 4) {
            // Fall back to AVX2 for smaller keys
            return masked_not_equal(a, b, mask);
        } else {
            // Scalar fallback
            for (size_t i = 0; i < num_blocks; ++i) {
                if ((a.data_[i] ^ b.data_[i]) & mask.data_[i])
                    return true;
            }
            return false;
        }
    }

    friend bool operator!=(const morton_key &a, const morton_key &b) noexcept { return !(a == b); }

    friend bool operator<(const morton_key &a, const morton_key &b) noexcept
    {
        assert(!a.null_ && !b.null_);

        // Compare MSB block first
        for (size_t i = num_blocks; i-- > 0;) {
            if (a.data_[i] < b.data_[i])
                return true;
            if (a.data_[i] > b.data_[i])
                return false;
        }
        return false;
    }

    friend bool operator>(const morton_key &a, const morton_key &b) noexcept { return b < a; }
    friend bool operator<=(const morton_key &a, const morton_key &b) noexcept { return !(b < a); }
    friend bool operator>=(const morton_key &a, const morton_key &b) noexcept { return !(a < b); }

    // right shift by s bits
    morton_key operator>>(size_t s) const noexcept
    {
        if (s == 0)
            return *this;
        morton_key r;
        const size_t block_shift = s / bits_per_block;
        const size_t bit_shift = s % bits_per_block;

        for (size_t i = 0; i < num_blocks; ++i) {
            block_t v = 0;
            if (i + block_shift < num_blocks) {
                v = data_[i + block_shift] >> bit_shift;
                if (bit_shift != 0 && (i + block_shift + 1) < num_blocks) {
                    v |= data_[i + block_shift + 1] << (bits_per_block - bit_shift);
                }
            }
            r.data_[i] = v;
        }
        mask_top(r.data_);
        r.null_ = null_;
        return r;
    }

    // for debugging
    // to hex string (msb-first). Always prints 16 hex chars per 64-bit block.
    std::string to_hex() const
    {
        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (size_t i = num_blocks; i-- > 0;) {
            ss << std::setw(16) << static_cast<unsigned long long>(data_[i]);
        }
        return ss.str();
    }

    // and with uint64_t (masking only the lowest 64 bits)
    morton_key &operator&=(uint64_t v) noexcept
    {
        if constexpr (num_blocks > 0) {
            data_[0] &= v;
        }
        // all higher blocks are unaffected; if you want full zeroing beyond
        // first block, uncomment below:
        for (size_t i = 1; i < num_blocks; ++i)
            data_[i] = 0;
        mask_top(data_);
        return *this;
    }

    friend morton_key operator&(morton_key a, uint64_t v) noexcept
    {
        a &= v;
        return a;
    }

    friend morton_key operator&(uint64_t v, morton_key a) noexcept
    {
        a &= v;
        return a;
    }

    // assign from a single uint64_t (replaces all blocks)
    morton_key &operator=(uint64_t v) noexcept
    {
        // zero all blocks first
        for (size_t i = 0; i < num_blocks; ++i)
            data_[i] = 0;

        // put v into the lowest block
        if constexpr (num_blocks > 0)
            data_[0] = v;

        // ensure the top block is masked
        mask_top(data_);
        null_ = false;
        return *this;
    }

    // comparison against integral
    constexpr bool operator>(uint64_t v) const noexcept
    {
        // compare only against lower 64 bits; higher blocks must be zero for a
        // meaningful compare
        if constexpr (num_blocks == 1)
            return data_[0] > v;
        else {
            for (size_t i = num_blocks; i-- > 1;)
                if (data_[i] != 0)
                    return true;
            return data_[0] > v;
        }
    }

    // division by integral value (only safe for small denominators like 2)
    constexpr morton_key &operator/=(uint64_t v) noexcept
    {
        if (v == 0)
            return *this; // undefined, but guard
        uint64_t rem = 0;
        for (size_t i = num_blocks; i-- > 0;) {
            uint128 cur = (static_cast<uint128>(rem) << 64) | data_[i];
            data_[i] = static_cast<uint64_t>(cur / v);
            rem = static_cast<uint64_t>(cur % v);
        }
        mask_top(data_);
        return *this;
    }

    constexpr morton_key operator/(uint64_t v) const noexcept
    {
        morton_key r = *this;
        r /= v;
        return r;
    }

    // ---- helper: compare numeric value to a uint64_t ----
    constexpr int compare_u64(uint64_t v) const noexcept
    {
        if constexpr (num_blocks == 1) {
            uint64_t a = data_[0];
            if constexpr (bits < 64) {
                a &= (uint64_t(1) << bits) - 1; // mask unused high bits in single-block keys
            }
            if (a < v)
                return -1;
            if (a > v)
                return 1;
            return 0;
        } else {
            // If any higher block (beyond the lowest) is non-zero, the value
            // exceeds 64 bits.
            for (size_t i = num_blocks; i-- > 1;) {
                if (data_[i] != 0)
                    return 1; // definitely greater than any uint64_t
            }
            const uint64_t a = data_[0];
            if (a < v)
                return -1;
            if (a > v)
                return 1;
            return 0;
        }
    }

    // ---- morton_key op uint64_t (rhs) ----
    friend constexpr bool operator==(const morton_key &k, uint64_t v) noexcept
    {
        return k.compare_u64(v) == 0;
    }
    friend constexpr bool operator!=(const morton_key &k, uint64_t v) noexcept
    {
        return k.compare_u64(v) != 0;
    }
    friend constexpr bool operator<(const morton_key &k, uint64_t v) noexcept
    {
        return k.compare_u64(v) < 0;
    }
    friend constexpr bool operator<=(const morton_key &k, uint64_t v) noexcept
    {
        return k.compare_u64(v) <= 0;
    }
    friend constexpr bool operator>(const morton_key &k, uint64_t v) noexcept
    {
        return k.compare_u64(v) > 0;
    }
    friend constexpr bool operator>=(const morton_key &k, uint64_t v) noexcept
    {
        return k.compare_u64(v) >= 0;
    }

    // ---- uint64_t (lhs) op morton_key ----
    friend constexpr bool operator==(uint64_t v, const morton_key &k) noexcept { return k == v; }
    friend constexpr bool operator!=(uint64_t v, const morton_key &k) noexcept { return k != v; }
    friend constexpr bool operator<(uint64_t v, const morton_key &k) noexcept
    {
        // v < k  <=>  k > v
        return k.compare_u64(v) > 0;
    }
    friend constexpr bool operator<=(uint64_t v, const morton_key &k) noexcept
    {
        return k.compare_u64(v) >= 0;
    }
    friend constexpr bool operator>(uint64_t v, const morton_key &k) noexcept
    {
        return k.compare_u64(v) < 0;
    }
    friend constexpr bool operator>=(uint64_t v, const morton_key &k) noexcept
    {
        return k.compare_u64(v) <= 0;
    }

    // Returns the least-significant 64 bits of the key
    constexpr uint64_t lsb64() const noexcept
    {
        assert(*this < uint64_t(-1));

        if constexpr (num_blocks == 0)
            return 0ULL;
        else if constexpr (bits < 64)
            return data_[0] & ((uint64_t(1) << bits) - 1);
        else
            return data_[0];
    }

    /// Efficintly read up to `count` bits starting at `start_bit`.
    ///
    /// Performance: O(1) - accesses at most 2 blocks regardless of key size
    ///
    /// @param start_bit Starting bit position (0 = LSB)
    /// @param count Number of bits to extract (must be <= 64)
    /// @return Extracted bits as uint64_t, right-aligned (LSBs)
    constexpr uint64_t get_bits_at(size_t start_bit, size_t count) const noexcept
    {
        assert(count <= 64);
        assert(start_bit + count <= bits);

        if (count == 0)
            return 0;

        const size_t block_idx = start_bit / bits_per_block;
        const size_t bit_offset = start_bit % bits_per_block;

        // Extract from first block
        uint64_t result = data_[block_idx] >> bit_offset;

        // If bits span into next block, OR in the high bits
        const size_t bits_from_first = bits_per_block - bit_offset;
        if (count > bits_from_first && block_idx + 1 < num_blocks) {
            result |= data_[block_idx + 1] << bits_from_first;
        }

        // Mask to exact bit count (handles count < 64)
        if (count < 64) {
            result &= (uint64_t(1) << count) - 1;
        }

        return result;
    }

    // OR with uint64_t (affects only the lowest 64 bits)
    // Note: No mask_top needed - we only add 1-bits to block 0, top block unchanged
    morton_key &operator|=(uint64_t v) noexcept
    {
        if constexpr (num_blocks > 0) {
            if constexpr (bits < 64) {
                const uint64_t m = (bits == 0) ? 0ULL : ((uint64_t(1) << bits) - 1ULL);
                data_[0] |= (v & m);
            } else {
                data_[0] |= v;
            }
        }
        // No mask_top needed - OR into block 0 cannot affect top block
        return *this;
    }

    // morton_key | uint64_t
    friend morton_key operator|(morton_key a, uint64_t v) noexcept
    {
        a |= v;
        return a;
    }

    // uint64_t | morton_key
    friend morton_key operator|(uint64_t v, morton_key a) noexcept
    {
        a |= v;
        return a;
    }

    /// @brief Fused copy + OR: returns base | v efficiently for small v
    ///
    /// This is more efficient than `base | v` because:
    /// - `base | v` takes base by value (copies), then ORs
    /// - This uses AVX2 for fast copy, then direct block 0 OR
    ///
    /// Use for pattern: `parent_morton | only_last_level_symbol` where
    /// only_last_level_symbol fits in 64 bits.
    ///
    /// @param base The morton_key to copy from
    /// @param v The uint64_t value to OR into the low bits
    /// @return New morton_key containing base | v
    static morton_key copy_and_or_low_u64(const morton_key &base, uint64_t v) noexcept
    {
        morton_key r;

        // AVX2 fast copy: 4 blocks (256 bits) per iteration
        if constexpr (num_blocks >= 4) {
            size_t i = 0;
            for (; i + 4 <= num_blocks; i += 4) {
                __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&base.data_[i]));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&r.data_[i]), val);
            }
            // Handle remaining blocks (0-3)
            for (; i < num_blocks; ++i) {
                r.data_[i] = base.data_[i];
            }
        } else {
            // Scalar fallback for small keys
            for (size_t i = 0; i < num_blocks; ++i) {
                r.data_[i] = base.data_[i];
            }
        }
        r.null_ = base.null_;

        // Direct OR into block 0 - no mask_top needed
        if constexpr (num_blocks > 0) {
            r.data_[0] |= v;
        }

        return r;
    }

    // in-place left shift by 's' bits
    morton_key &operator<<=(uint64_t s) noexcept
    {
        if (s == 0)
            return *this;

        const size_t block_shift = s / bits_per_block;
        const size_t bit_shift = s % bits_per_block;

        if (block_shift >= num_blocks) {
            zero_blocks(data_);
            null_ = false;
            return *this;
        }

        // Shift whole blocks upward first
        if (block_shift > 0) {
            for (size_t i = num_blocks; i-- > 0;) {
                if (i >= block_shift)
                    data_[i] = data_[i - block_shift];
                else
                    data_[i] = 0;
            }
        }

        // Shift within each 64-bit block if needed
        if (bit_shift != 0) {
            for (size_t i = num_blocks; i-- > 0;) {
                uint64_t next = (i > 0) ? data_[i - 1] : 0ULL;
                data_[i] = (data_[i] << bit_shift) | (next >> (bits_per_block - bit_shift));
            }
        }

        mask_top(data_);
        return *this;
    }
    // non-mutating left shift
    friend morton_key operator<<(morton_key a, uint64_t s) noexcept
    {
        a <<= s;
        return a;
    }

    // pre-increment
    constexpr morton_key &operator++() noexcept
    {
        *this += uint64_t{1};
        return *this;
    }

    // post-increment, current becomes null on overflow
    constexpr morton_key operator++(int) noexcept
    {
        morton_key tmp = *this;
        *this += uint64_t{1};
        return tmp;
    }

    // += with uint64_t, set null_ on overflow
    constexpr morton_key &operator+=(uint64_t v) noexcept
    {
        if (v == 0)
            return *this;

        // add into block 0
        uint128 sum = static_cast<uint128>(data_[0]) + v;
        data_[0] = static_cast<uint64_t>(sum);
        uint64_t carry = static_cast<uint64_t>(sum >> 64);

        // propagate carry
        for (size_t i = 1; i < num_blocks && carry != 0; ++i) {
            sum = static_cast<uint128>(data_[i]) + carry;
            data_[i] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }

        // detect any overflow BEFORE masking
        bool overflow = (carry != 0);
        overflow |= ((data_[num_blocks - 1] & top_excess_mask()) != 0);

        // enforce bit-width
        mask_top(data_);

        if (overflow)
            null_ = true;
        return *this;
    }

    // morton_key + uint64_t
    friend constexpr morton_key operator+(morton_key a, uint64_t v) noexcept
    {
        a += v; // overflow handled inside
        return a;
    }

    // uint64_t + morton_key
    friend constexpr morton_key operator+(uint64_t v, morton_key a) noexcept
    {
        a += v; // overflow handled inside
        return a;
    }

    // in-place add: this += o, set null_ on overflow
    morton_key &operator+=(const morton_key &o) noexcept
    {
        uint128 carry = 0;

        for (size_t i = 0; i < num_blocks; ++i) {
            uint128 sum = static_cast<uint128>(data_[i]) + static_cast<uint128>(o.data_[i]) + carry;

            data_[i] = static_cast<uint64_t>(sum);
            carry = (sum >> 64);
        }

        // detect overflow BEFORE masking
        bool overflow = (carry != 0);
        overflow |= ((data_[num_blocks - 1] & top_excess_mask()) != 0);

        // respect DIM_BITS
        mask_top(data_);

        // your prior null semantics + overflow-to-null
        null_ = (null_ || o.null_) || overflow;

        return *this;
    }

    // non-mutating add reuses the in-place version
    friend morton_key operator+(morton_key a, const morton_key &b) noexcept
    {
        a += b; // overflow handled inside
        return a;
    }
};

// std::hash
namespace std
{
template <size_t DIM_BITS> struct hash<morton_key<DIM_BITS>> {
    size_t operator()(morton_key<DIM_BITS> const &k) const noexcept
    {
        constexpr size_t nb = morton_key<DIM_BITS>::num_blocks;
        const uint64_t *blocks = k.blocks();
        uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis

        for (size_t i = 0; i < nb; ++i) {
            h ^= blocks[i];
            h *= 1099511628211ULL;
        }

        // Mix in the null flag so null()/non-null collide less
        h ^= static_cast<uint64_t>(k.is_null());
        h *= 1099511628211ULL;

        // Finalize to size_t
        return static_cast<size_t>(h ^ (h >> 32));
    }
};
} // namespace std
