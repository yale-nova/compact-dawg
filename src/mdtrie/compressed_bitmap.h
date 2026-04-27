#ifndef BITMAP_COMPRESSED_BITMAP_H_
#define BITMAP_COMPRESSED_BITMAP_H_

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <x86intrin.h>

#include "bitmap_utils.h"
#include "debug.h"
#include "defs.h"
#include "morton_key.h"
#include "ordered_types.h"
#include "profiling_points.h"

struct level_info {
    node_bitmap_pos_t
        data_pos;   // location of current node in the bitmap, the actual node (skip collapsed bit)
    bool collapsed; // if current node is collapsed
    uint64_t word;  // how much of current node we have processed (1)
    node_bitmap_pos_t bit_base;  // how much of current node we have processed (2)
    node_bitmap_pos_t remaining; // remaining bits to process
};

namespace compressed_bitmap
{

// TODO: H_LEVEL has nothing to do with trie_level_t.
//       In fact since we routinely use H_LEVEL > 256 it's likely to cause issue in the future.
//       Please replace this with a sensible type like size_t or something.
template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION>
class compressed_bitmap
{
    static_assert(std::__has_single_bit(CHUNK_WIDTH), "CHUNK_WIDTH must be a power of 2");
    static_assert(DIMENSION > constexpr_log2(CHUNK_WIDTH) * H_LEVEL,
                  "DIMENSION must be greater than CHUNK_WIDTH * H_LEVEL");
    // todo: remove branching in is_on_data
public:
    typedef uint64_t size_type;
    typedef uint64_t data_type;
    typedef uint64_t width_type;

    // H_LEVEL := levels of _compression_.
    // 0 => no chunk compression is allowed.
    // 1 => each hier_node has depth _2_; levels 0 and 1.
    // 0 is the top level. H_LEVEL is the deepest layer (lsb).
    using chunk_val_t = uint64_t;

    // Morton key type for this compressed_bitmap's dimension
    using morton_type = morton_key<DIMENSION>;
    static constexpr uint64_t mask_low64(size_t n) noexcept
    {
        return (n >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1ULL);
    }

    inline data_type *select_buffer(bool is_on_data) noexcept { return is_on_data ? data_ : flag_; }

    inline size_type buffer_size_bits(bool is_on_data) const noexcept
    {
        return is_on_data ? data_size_ : flag_size_;
    }

    // Hierarchical-encoding related variables.
    // Hierarchical encoding structure:
    //
    // ("Top Level Chunk")_("Chunk")_("Chunk")_..._("Chunk")...
    //
    // "Top Level Chunk" control where the chunks should be placed in the next level of the
    // encoding. The chunks themselves are encoded in a DFS order, as in the regular MDTrie.

    // 10000
    // 5
    // Log2(CHUNK_WIDTH);
    static constexpr node_pos_t CHUNK_WIDTH_SHIFT = constexpr_log2(CHUNK_WIDTH);
    // Apply this to a `chunk_val_t` to limit its range to our chunk size.
    static constexpr chunk_val_t CHUNK_MASK = ((chunk_val_t)(CHUNK_WIDTH - 1));

    // When reading a hierachically encoded symbol, we read in multiples of the chunk size.
    // Unfortunately, the top level chunk isn't necessarily a multiple of the ordinary chunk size.
    // This var describes the log_2(top level chunk size field).
    static constexpr node_pos_t TOP_LEVEL_CHUNK_WIDTH_SHIFT =
        DIMENSION - H_LEVEL * CHUNK_WIDTH_SHIFT;

    // Makes code more readable.
    static constexpr node_pos_t TOP_LEVEL_COLLAPSED_CHUNK_WIDTH = TOP_LEVEL_CHUNK_WIDTH_SHIFT;
    static constexpr node_pos_t COLLAPSED_CHUNK_WIDTH = CHUNK_WIDTH_SHIFT;

    // When reading a hierachically encoded symbol, we read in multiples of the chunk size.
    // Unfortunately, the top level "Active Bit" region isn't necessarily a
    // multiple of the chunk size. This var describes the real length of the field.
    static constexpr node_pos_t TOP_LEVEL_CHUNK_WIDTH = 1UL << TOP_LEVEL_CHUNK_WIDTH_SHIFT;

    static_assert(TOP_LEVEL_CHUNK_WIDTH > 0, "TOP_LEVEL_CHUNK_WIDTH must be positive");

    // Ensure chunk-level values (after masking with CHUNK_MASK or shifting) fit in 64 bits.
    // This allows us to use uint64_t instead of morton_type for temporary storage.
    static_assert(CHUNK_WIDTH_SHIFT <= 64, "CHUNK_WIDTH_SHIFT must fit in 64 bits");
    static_assert(TOP_LEVEL_CHUNK_WIDTH_SHIFT <= 64,
                  "TOP_LEVEL_CHUNK_WIDTH_SHIFT must fit in 64 bits");

    // destructor cannot exist, please see destroy() method Note
    // ~compressed_bitmap() { destroy(); }

    // default constructor, everything null or zero
    explicit compressed_bitmap() : data_(nullptr), flag_(nullptr), data_size_(0), flag_size_(0) {}

    // construct a compressed bitmap with given sizes, nothing set
    explicit compressed_bitmap(width_type flag_size, width_type data_size)
    {
        assert((flag_size > 0 && data_size > 0));

        data_ = (data_type *)calloc(BITS2BLOCKS(data_size), sizeof(data_type));
        flag_ = (data_type *)calloc(BITS2BLOCKS(flag_size), sizeof(data_type));
        if (data_ == nullptr || flag_ == nullptr) {
            std::cerr << "Memory allocation failed in compressed_bitmap constructor" << std::endl;
            exit(EXIT_FAILURE);
        }
        data_size_ = data_size;
        flag_size_ = flag_size;
    }

    // Custom Destructor
    // (needed for when assigning cb1 = cb2:
    // Note: cb2 is dropped with inbuilt destructor, if free there, double free)
    void destroy()
    {
        if (data_size_) {
            free(data_);
            data_ = nullptr;
        }
        if (flag_size_) {
            free(flag_);
            flag_ = nullptr;
        }
        data_size_ = 0;
        flag_size_ = 0;
    }

    // returns the number of set bits between (pos, pos + width) on either data or flag.
    //
    // TODO(yash): can we rewrite this to avoid all the branching?
    //             Lot of confusion around this boolean arg.
    //             Also it's taking most of our CPU time, so speeding it up should help a lot!
    inline uint64_t popcount(node_bitmap_pos_t pos, width_type width, bool is_on_data = true) const
    {
        if (!width)
            return 0;

        if (width <= 64) {
            return (uint64_t)__builtin_popcountll(GetValPosU64(pos, width, is_on_data));
        }

        node_bitmap_pos_t s_off = pos % 64;
        node_bitmap_pos_t s_idx = pos / 64;
        uint64_t count = (uint64_t)__builtin_popcountll(GetValPosU64(pos, 64 - s_off, is_on_data));
        width -= 64 - s_off;
        s_idx += 1;

        if (is_on_data) {
            while (width > 64) {
                count += (uint64_t)__builtin_popcountll(data_[s_idx]);
                width -= 64;
                s_idx += 1;
            }
        } else {
            while (width > 64) {
                count += (uint64_t)__builtin_popcountll(flag_[s_idx]);
                width -= 64;
                s_idx += 1;
            }
        }
        if (width > 0) {
            count += (uint64_t)__builtin_popcountll(GetValPosU64(s_idx * 64, width, is_on_data));
        }
        return count;
    }

    // morton_key variant: copy `num_bits` LSBs from morton_type into [pos,
    // pos+num_bits)
    inline void SetValPos(node_bitmap_pos_t pos, const morton_type &val, width_type num_bits,
                          bool is_on_data)
    {
        assert(pos + num_bits <= buffer_size_bits(is_on_data));

        data_type *dst = select_buffer(is_on_data);
        const uint64_t *src = val.blocks();
        size_t done = 0;

        while (done < static_cast<size_t>(num_bits)) {
            // destination window
            const size_t d_bit = static_cast<size_t>(pos) + done;
            const size_t d_idx = d_bit / 64;
            const size_t d_off = d_bit % 64;
            const size_t w = std::min<size_t>(64 - d_off, static_cast<size_t>(num_bits) - done);

            // source window from morton_type (packed LSB-first)
            const size_t s_bit = done;
            const size_t s_idx = s_bit / 64;
            const size_t s_off = s_bit % 64;

            const uint64_t lo = (s_idx < morton_type::num_blocks) ? src[s_idx] : 0ULL;
            const uint64_t hi = (s_idx + 1 < morton_type::num_blocks) ? src[s_idx + 1] : 0ULL;

            uint64_t chunk = (lo >> s_off);
            if (s_off)
                chunk |= (hi << (64 - s_off));
            chunk &= mask_low64(w);

            const uint64_t m = mask_low64(w) << d_off;
            dst[d_idx] = (dst[d_idx] & ~m) | ((chunk << d_off) & m);

            done += w;
        }
    }

    // fast path for legacy 0..64-bit writes (keeps existing callers working)
    inline void SetValPos(node_bitmap_pos_t pos, uint64_t v, width_type num_bits, bool is_on_data)
    {
        assert(num_bits > 0 && num_bits <= 64);
        assert(pos + num_bits <= buffer_size_bits(is_on_data));

        data_type *dst = select_buffer(is_on_data);

        const size_t d_idx = static_cast<size_t>(pos) / 64;
        const size_t d_off = static_cast<size_t>(pos) % 64;

        if (d_off + static_cast<size_t>(num_bits) <= 64) {
            const uint64_t m = mask_low64(num_bits) << d_off;
            dst[d_idx] = (dst[d_idx] & ~m) | ((v << d_off) & m);
        } else {
            const size_t lo_bits = 64 - d_off;
            const size_t hi_bits = static_cast<size_t>(num_bits) - lo_bits;

            const uint64_t m0 = mask_low64(lo_bits) << d_off;
            const uint64_t m1 = mask_low64(hi_bits);

            dst[d_idx] = (dst[d_idx] & ~m0) | ((v << d_off) & m0);
            dst[d_idx + 1] = (dst[d_idx + 1] & ~m1) | ((v >> lo_bits) & m1);
        }
    }

    // Extract `num_bits` starting at `pos` into a morton_type (packed into its LSBs)
    inline morton_type GetValPos(node_bitmap_pos_t pos, width_type num_bits, bool is_on_data) const
    {
        // allow 0 for convenience; return zero key
        if (num_bits == 0) {
            return morton_type::zero();
        }

        assert(num_bits > 0);
        const size_type total_bits = buffer_size_bits(is_on_data);
        assert(static_cast<size_type>(pos) + static_cast<size_type>(num_bits) <= total_bits);

        morton_type out; // default-constructed => zeroed blocks
        uint64_t *dst = out.blocks();

        // Check dest pointer alignment (which is crucial for performance when using AVX512
        // instructions)
        // assert(reinterpret_cast<std::uintptr_t>(dst) % 64 == 0);

        const data_type *src = is_on_data ? data_ : flag_;
        size_t remaining = static_cast<size_t>(num_bits);
        size_t dst_bit = 0;

        while (remaining > 0) {
            // read up to 64 bits starting at current source bit
            const size_t s_bit =
                static_cast<size_t>(pos) + (static_cast<size_t>(num_bits) - remaining);
            const size_t s_idx = s_bit / 64;
            const size_t s_off = s_bit % 64;

            const size_t take = (remaining >= 64) ? size_t{64} : remaining;

            // load up to two source words (guarding bounds)
            const uint64_t lo = (s_idx < BITS2BLOCKS(total_bits)) ? src[s_idx] : 0ULL;
            const uint64_t hi = (s_idx + 1 < BITS2BLOCKS(total_bits)) ? src[s_idx + 1] : 0ULL;

            // mirror the known-good single/two-block extraction
            uint64_t chunk;
            if (s_off + take <= 64) {
                chunk = (lo >> s_off) & low_bits_set[take];
            } else {
                chunk = ((lo >> s_off) | (hi << (64 - s_off))) & low_bits_set[take];
            }

            // write `chunk` into dst at bit offset `dst_bit`, possibly straddling two words
            const size_t d_idx = dst_bit / 64;
            const size_t d_off = dst_bit % 64;

            if (d_off + take <= 64) {
                const uint64_t mask = (low_bits_set[take] << d_off);
                dst[d_idx] = (dst[d_idx] & ~mask) | ((chunk << d_off) & mask);
            } else {
                const size_t first = 64 - d_off;    // bits into current dst word
                const size_t second = take - first; // bits into next dst word

                const uint64_t m0 = (low_bits_set[first] << d_off);
                const uint64_t m1 = low_bits_set[second];

                dst[d_idx] = (dst[d_idx] & ~m0) | (((chunk & low_bits_set[first]) << d_off) & m0);
                dst[d_idx + 1] = (dst[d_idx + 1] & ~m1) | ((chunk >> first) & m1);
            }

            dst_bit += take;
            remaining -= take;
        }

        // `out` is already zeroed above written range; high blocks/bits remain 0.
        return out;
    }

    inline uint64_t GetValPosU64(node_bitmap_pos_t pos, width_type num_bits, bool is_on_data) const
    {
        assert(num_bits > 0 && num_bits <= 64);
        const size_type total_bits = buffer_size_bits(is_on_data);
        assert(pos + num_bits <= total_bits);

        const data_type *src = is_on_data ? data_ : flag_;

        const size_t s_idx = static_cast<size_t>(pos) / 64;
        const size_t s_off = static_cast<size_t>(pos) % 64;

        const uint64_t lo = (s_idx < BITS2BLOCKS(total_bits)) ? src[s_idx] : 0ULL;
        const uint64_t hi = (s_idx + 1 < BITS2BLOCKS(total_bits)) ? src[s_idx + 1] : 0ULL;

        uint64_t out = (lo >> s_off);
        if (s_off)
            out |= (hi << (64 - s_off));
        return out & mask_low64(num_bits);
    }

    // create 0-ed out holes at position (pos, pos + width) on either data
    // or flag
    inline void ClearWidth(node_bitmap_pos_t pos, width_type width, bool is_on_data)
    {
        assert(width != 0);

        if (is_on_data) {
            assert(pos + width <= data_size_);
        } else {
            assert(pos + width <= flag_size_);
        }

        if (width <= 64) {
            SetValPos(pos, 0, width, is_on_data);
            return;
        }
        node_bitmap_pos_t s_off = 64 - pos % 64;
        node_bitmap_pos_t s_idx = pos / 64;
        SetValPos(pos, 0, s_off, is_on_data);

        width -= s_off;
        s_idx += 1;
        while (width > 64) {
            if (is_on_data)
                data_[s_idx] = 0;
            else
                flag_[s_idx] = 0;
            width -= 64;
            s_idx += 1;
        }
        SetValPos(s_idx * 64, 0, width, is_on_data);
    }

public:
    // Copies bits from [from, from+bits) to [destination, destination+bits)
    // in forward direction. Safe for overlapping regions only if 'from >
    // destination' (i.e., destination is before from). The assert enforces
    // this to prevent undefined behavior if regions overlap in the wrong
    // direction. If the assertion fails, the copy may corrupt data.
    inline void bulkcopy_forward(node_bitmap_pos_t from, node_bitmap_pos_t destination,
                                 width_type bits, bool is_on_data)
    {
        assert(from > destination); // Only safe if copying forward and regions
                                    // do not overlap in the wrong direction
        // Bounds assertions to prevent out-of-bounds access
        if (is_on_data) {
            assert(from + bits <= data_size_);
            assert(destination + bits <= data_size_);
        } else {
            assert(from + bits <= flag_size_);
            assert(destination + bits <= flag_size_);
        }

        data_type *buf = is_on_data ? data_ : flag_;

        // Calculate alignment boundaries for destination range [destination, destination + bits)
        node_bitmap_pos_t dest_start = destination;
        node_bitmap_pos_t dest_end = destination + bits;

        // Find the first 64-bit aligned position >= dest_start
        node_bitmap_pos_t aligned_start = ((dest_start + 63) / 64) * 64;
        // Find the last 64-bit aligned position <= dest_end
        node_bitmap_pos_t aligned_end = (dest_end / 64) * 64;

        width_type middle_bits = (aligned_end > aligned_start) ? (aligned_end - aligned_start) : 0;

        // Only use memmove if middle chunk is significant (>= 256 bits = 4 words)
        // and source alignment matches destination alignment (required for correctness)
        node_bitmap_pos_t src_aligned_start = from + (aligned_start - dest_start);
        bool same_alignment = (src_aligned_start % 64 == 0);

        if (middle_bits >= 256 && same_alignment) {
            // Copy head (dest_start to aligned_start) with bit ops
            if (aligned_start > dest_start) {
                width_type head_bits = aligned_start - dest_start;
                SetValPos(dest_start, GetValPosU64(from, head_bits, is_on_data), head_bits,
                          is_on_data);
            }

            // Copy middle with memmove (aligned, fast!)
            // Source and destination are both 64-bit aligned here
            std::memmove(buf + aligned_start / 64, buf + src_aligned_start / 64, middle_bits / 8);

            // Copy tail (aligned_end to dest_end) with bit ops
            if (dest_end > aligned_end) {
                width_type tail_bits = dest_end - aligned_end;
                node_bitmap_pos_t src_tail = from + bits - tail_bits;
                SetValPos(aligned_end, GetValPosU64(src_tail, tail_bits, is_on_data), tail_bits,
                          is_on_data);
            }
            return;
        }

        // Fallback to existing loop for small copies or misaligned source
        while (bits > 64) {
            SetValPos(destination, GetValPosU64(from, 64, is_on_data), 64, is_on_data);
            from += 64;
            destination += 64;
            bits -= 64;
        }

        if (bits)
            SetValPos(destination, GetValPosU64(from, bits, is_on_data), bits, is_on_data);
    }

    // Copies bits from [`from - bits`, from) to [`destination - bits`, destination)
    // in backward direction. Safe for overlapping regions only if
    // 'destination > from' (i.e., destination is after from). The assert
    // enforces this to prevent undefined behavior if regions overlap in the
    // wrong direction. If the assertion fails, the copy may corrupt data.
    //
    //
    // YASH: For some reason, this function doesn't copy from
    // [start, start+size) to [end, end+size). It actually copies from
    // `[start - size, start)` to `[end - size, end)`...what a footgun.
    // Micah, see if you can make this more intuitive in the GPU version (if it's needed).
    inline void bulkcopy_backward(node_bitmap_pos_t from, node_bitmap_pos_t destination,
                                  width_type size, bool is_on_data = true)
    {
        assert(destination > from); // Only safe if copying backward and regions
                                    // do not overlap in the wrong direction
        if (size == 0)
            return;
        // Bounds assertions to prevent out-of-bounds access
        if (is_on_data) {
            assert(from <= data_size_);
            assert(destination <= data_size_);
            assert(from >= size);
            assert(destination >= size);
        } else {
            assert(from <= flag_size_);
            assert(destination <= flag_size_);
            assert(from >= size);
            assert(destination >= size);
        }

        data_type *buf = is_on_data ? data_ : flag_;

        // Calculate alignment boundaries for destination range [destination - bits, destination)
        node_bitmap_pos_t dest_start = destination - size;
        node_bitmap_pos_t dest_end = destination;

        // Find the first 64-bit aligned position >= dest_start
        node_bitmap_pos_t aligned_start = ((dest_start + 63) / 64) * 64;
        // Find the last 64-bit aligned position <= dest_end
        node_bitmap_pos_t aligned_end = (dest_end / 64) * 64;

        width_type middle_bits = (aligned_end > aligned_start) ? (aligned_end - aligned_start) : 0;

        // Only use memmove if middle chunk is significant (>= 256 bits = 4 words)
        // and source alignment matches destination alignment (required for correctness)
        node_bitmap_pos_t src_start = from - size;
        node_bitmap_pos_t src_aligned_start = src_start + (aligned_start - dest_start);
        bool same_alignment = (src_aligned_start % 64 == 0);

        if (middle_bits >= 256 && same_alignment) {
            // Copy tail (aligned_end to dest_end) with bit ops - do this first for backward copy
            if (dest_end > aligned_end) {
                width_type tail_bits = dest_end - aligned_end;
                node_bitmap_pos_t src_tail = from - tail_bits;
                SetValPos(aligned_end, GetValPosU64(src_tail, tail_bits, is_on_data), tail_bits,
                          is_on_data);
            }

            // Copy middle with memmove (aligned, fast!)
            // Source and destination are both 64-bit aligned here
            std::memmove(buf + aligned_start / 64, buf + src_aligned_start / 64, middle_bits / 8);

            // Copy head (dest_start to aligned_start) with bit ops
            if (aligned_start > dest_start) {
                width_type head_bits = aligned_start - dest_start;
                SetValPos(dest_start, GetValPosU64(src_start, head_bits, is_on_data), head_bits,
                          is_on_data);
            }
            return;
        }

        // Fallback to existing loop for small copies or misaligned source
        while (size > 64) {
            SetValPos(destination - 64, GetValPosU64(from - 64, 64, is_on_data), 64, is_on_data);
            size -= 64;
            destination -= 64;
            from -= 64;
        }

        if (size)
            SetValPos(destination - size, GetValPosU64(from - size, size, is_on_data), size,
                      is_on_data);
    }

    // create 0-ed out holes from node/node_pos of len flag_bits/data_bits
    inline void shift_backward(node_pos_t node, node_bitmap_pos_t node_pos, width_type flag_bits,
                               width_type data_bits)
    {
        size_type orig_data_size = data_size_;
        size_type orig_flag_size = flag_size_;

        increase_bitmap_size(data_bits);
        increase_flagmap_size(flag_bits);

        assert(node_pos <= orig_data_size);
        assert(node <= orig_flag_size);
        bulkcopy_backward(orig_data_size, data_size_, orig_data_size - node_pos, true);
        bulkcopy_backward(orig_flag_size, flag_size_, orig_flag_size - node, false);

        clear_bitmap_pos(node_pos, data_bits);
        clear_flagmap_pos(node, flag_bits);
    }

    // Shift data backward to make room at node_pos.
    // This function doesn't zero the newly created space.
    // @param data_bits: bits to shift (creates a gap of this size at node_pos)
    // @param extra_alloc_bits: additional bits to allocate at the end (not shifted, just empty
    // space)
    inline void shift_backward_data_only(node_bitmap_pos_t node_pos, width_type data_bits,
                                         width_type extra_alloc_bits = 0)
    {
        if (data_bits == 0) {
            if (extra_alloc_bits != 0)
                increase_bitmap_size(extra_alloc_bits);
            return;
        }
        size_type orig_data_size = data_size_;

        // Allocate space for both the shift gap AND the extra space
        increase_bitmap_size(data_bits + extra_alloc_bits);

        assert(node_pos <= orig_data_size);
        // Only shift by data_bits (the gap we're creating at node_pos)
        // The extra_alloc_bits remain as empty space at the end
        bulkcopy_backward(orig_data_size, orig_data_size + data_bits, orig_data_size - node_pos,
                          true);

        // clear_bitmap_pos(node_pos, data_bits);
    }

    // shift node, node_pos forward (to earlier address),
    // used to remove nodes
    inline void shift_forward(node_pos_t from_node, node_bitmap_pos_t from_node_pos,
                              node_pos_t to_node, node_bitmap_pos_t to_node_pos)

    {
        if (from_node == to_node) {
            assert(from_node_pos == to_node_pos);
            return;
        }

        if (data_size_ - from_node_pos != 0)
            bulkcopy_forward(from_node_pos, to_node_pos, data_size_ - from_node_pos, true);

        if (flag_size_ - from_node != 0)
            bulkcopy_forward(from_node, to_node, flag_size_ - from_node, false);

        decrease_bitmap_size(from_node_pos - to_node_pos);
        decrease_flagmap_size(from_node - to_node);
    }

    /// Return the width (in bits) of the trie node at `node_pos`, `node_bitmap_pos`.
    ///
    /// Trie nodes have variable width depending on their encoding. This function
    /// "parses" a symbol and returns its length.
    inline width_type get_symbol_width(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos) const
    {
        if (node_is_collapsed(node_pos)) {
            return DIMENSION;
        }

        const node_pos_t original_bitmap_pos = node_bitmap_pos;

        // Node uses hierarchical encoding. Count the number of children in the hierarchical tree.
        auto [num_child_subtrees, chunk_width] = count_toplevel_chunk_children(node_bitmap_pos);
        node_bitmap_pos += chunk_width;

        // Count the child nodes in each subtree of the root chunk.
        while (num_child_subtrees > 0) {
            assert(H_LEVEL > 0);
            node_bitmap_pos = skip_hier_subtree(node_bitmap_pos, 1);
            num_child_subtrees--;
        }

        return node_bitmap_pos - original_bitmap_pos;
    }

    // check if the symbol is set in the node at node_bitmap_pos
    inline bool has_child(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                          const morton_type &symbol) const
    {
        if (node_pos >= flag_size_)
            return false;

        if (node_is_collapsed(node_pos))
            return symbol == read_from_bitmap_pos(node_bitmap_pos, DIMENSION);

        // Use fast hierarchical check that can short-circuit at level 0.
        return has_child_hier(node_bitmap_pos, symbol);
    }

private:
    // A helper function used when dealing with hierarchically encoded nodes.
    //
    // Skips the "hier-node subtree" at a given bitmap position.
    // Eg: if you're dealing with a hierarchical node and want the _second_ child,
    //     use this function to skip past the first child (by pointing it at the first chunk).
    //
    // Returns: (new position, count)
    inline std::pair<node_bitmap_pos_t, node_pos_t> count_hier_subtree(node_bitmap_pos_t bitmap_pos,
                                                                       size_t hier_level) const
    {
        auto [children_in_chunk, chunk_width] = count_chunk_children(bitmap_pos, hier_level);
        bitmap_pos += chunk_width;

        if (hier_level == H_LEVEL)
            return {bitmap_pos, children_in_chunk};

        node_pos_t children_in_subtree = 0;
        while (children_in_chunk > 0) {
            auto [new_bitmap_pos, children] = count_hier_subtree(bitmap_pos, hier_level + 1);
            bitmap_pos = new_bitmap_pos;
            children_in_subtree += children;
            children_in_chunk--;
        }

        return {bitmap_pos, children_in_subtree};
    }

    // A helper function used when dealing with hierarchically encoded nodes.
    //
    // Skips the "hier-node subtree" at a given bitmap position.
    // Eg: if you're dealing with a hierarchical node and want the _second_ child,
    //     use this function to skip past the first child (by pointing it at the first chunk).
    //
    // Don't pass this function a top-level chunk! Those are treated differently.
    inline node_bitmap_pos_t skip_hier_subtree(node_bitmap_pos_t bitmap_pos,
                                               size_t hier_level) const
    {
        if (hier_level == H_LEVEL) {
            return bitmap_pos + get_chunk_width(bitmap_pos);
        }

        auto [children_in_chunk, chunk_width] = count_chunk_children(bitmap_pos, hier_level);
        bitmap_pos += chunk_width;

        while (children_in_chunk > 0) {
            bitmap_pos = skip_hier_subtree(bitmap_pos, hier_level + 1);
            children_in_chunk--;
        }
        return bitmap_pos;
    }

private:
    // Internal helper: Perform a _fast_ check whether a child symbol exists within a node.
    //
    // It takes advantage of the hierarchical bitmap representation to "short-circuit" and
    // return zero early, if the relevant chunk is not found.
    //
    //
    // Returns true if symbol found, false otherwise.
    // TODO(yash): Why couldn't I just integrate this into the `get_child()` method?
    //             Right now, this is a duplicate traversal since it needs to `popcount()` all
    //             non-level-0 chunks. Great optimization opportunity.
    inline bool has_child_hier(node_bitmap_pos_t node_bitmap_pos,
                               const morton_type &child_sym) const
    {
        // Each level of the hierarchy "reveals" new bits in the child morton symbol.
        // These bits fit within a `uint64_t`. And conveniently, they serve as an
        // index into each chunk! (think about it carefully...checking whether
        // the `extract_chunk()` bit is set will tell us whether our child is
        // present within the tree).
        //
        // This strategy informs our overall algorithm.
        // 1. Read the top level chunk. Are the bits for our child set? If not, the child can't
        //    be present; return false.
        // 2. Read the relevant next level chunk. Are the bits for our child
        //    set? If not, the child can't be present; return false.
        // 3. ...
        // 4. Check the lowest level chunk. If the child is set, return true :)

        // Print all hier bits for the symbol before starting logic
        debugf("[has_child_hier] All hier bits for child_sym:\n");
        debugf("  Level 0 (top): %u\n", (unsigned int)get_hier_top_level_bits(child_sym));
        for (size_t i = 1; i <= H_LEVEL; i++) {
            debugf("  Level %zu: %u\n", i, (unsigned int)get_hier_level_bits(child_sym, i));
        }

        chunk_val_t top_level_symbol_bits = get_hier_top_level_bits(child_sym);
        debugf("[has_child_hier] top_level_symbol_bits=%lu\n", top_level_symbol_bits);

        size_t cur_hier_level = 0;
        node_pos_t children_to_skip;

        if (chunk_is_collapsed(node_bitmap_pos)) {
            // Top level chunk is collapsed. If it doesn't encode our symbol, return false.
            chunk_val_t collapsed_value =
                read_u64_from_bitmap_pos(node_bitmap_pos + 1, TOP_LEVEL_CHUNK_WIDTH_SHIFT);

            debugf("[has_child_hier] Collapsed Top Level value=%lu, expected=%lu\n",
                   collapsed_value, top_level_symbol_bits);

            if (top_level_symbol_bits != collapsed_value) {
                debugf("[has_child_hier] Top level mismatch, returning false\n");
                return false;
            }

            // The top level chunk does encode our symbol. Move on to the next level chunks.
            cur_hier_level++;
            children_to_skip = 0;
            node_bitmap_pos += TOP_LEVEL_CHUNK_WIDTH_SHIFT + 1; // width + "collapsed bit"

            debugf("[has_child_hier] Top level matched, advancing to level %zu at pos %zu\n",
                   cur_hier_level, node_bitmap_pos);
        } else {
            // uncollapsed top level chunk.
            if (!uncollapsed_chunk_contains_child(node_bitmap_pos, top_level_symbol_bits)) {
                debugf("[has_child_hier] Top level bit (pos %lu) not set, returning false\n",
                       top_level_symbol_bits);
                return false;
            }

            // Our child is there. Recurse.
            cur_hier_level++;
            children_to_skip = popcount(node_bitmap_pos + 1, top_level_symbol_bits, true);
            node_bitmap_pos += TOP_LEVEL_CHUNK_WIDTH + 1; // width + "collapsed bit"

            debugf("[has_child_hier] Top level bit set, children_to_skip=%zu, advancing to level "
                   "%zu at pos %zu\n",
                   children_to_skip, cur_hier_level, node_bitmap_pos);
        }

        while (cur_hier_level < H_LEVEL) {
            debugf("[has_child_hier] === Level %zu ===\n", cur_hier_level);

            // Skip past all the irrelevant child subtrees in our level.
            while (children_to_skip > 0) {
                debugf("[has_child_hier] Skipping subtree %zu at pos %zu\n", children_to_skip,
                       node_bitmap_pos);
                node_bitmap_pos = skip_hier_subtree(node_bitmap_pos, cur_hier_level);
                children_to_skip--;
            }
            debugf("[has_child_hier] Subtree skips complete. Now at %zu\n", node_bitmap_pos);

            chunk_val_t child_bits = get_hier_level_bits(child_sym, cur_hier_level);

            debugf("[has_child_hier] At pos %zu, extracted child_bits=%lu for level %zu\n",
                   node_bitmap_pos, child_bits, cur_hier_level);

            // Check whether this chunk contains our symbol. If it doesn't, return false.
            // If it does, check the next level.
            if (chunk_is_collapsed(node_bitmap_pos)) {
                chunk_val_t collapsed_value =
                    read_u64_from_bitmap_pos(node_bitmap_pos + 1, CHUNK_WIDTH_SHIFT);

                debugf("[has_child_hier] Collapsed value=%lu, expected=%lu\n", collapsed_value,
                       child_bits);

                // Chunk doesn't encode our symbol, return false.
                if (child_bits != collapsed_value) {
                    debugf("[has_child_hier] Collapsed value=%lu, expected=%lu\n", collapsed_value,
                           child_bits);
                    return false;
                }

                // Chunk does encode our symbol, need to check lower levels to see if symbol is
                // present.
                cur_hier_level++;
                children_to_skip = 0;
                node_bitmap_pos += CHUNK_WIDTH_SHIFT + 1;

                debugf("[has_child_hier] Matched, advancing to level %zu at pos %zu\n",
                       cur_hier_level, node_bitmap_pos);
            } else {
                // uncollapsed chunk
                if (!uncollapsed_chunk_contains_child(node_bitmap_pos, child_bits)) {
                    debugf("[has_child_hier] Bit not set at level %zu, returning false\n",
                           cur_hier_level);
                    return false;
                }

                // Chunk does encode our symbol. Continue recursing.
                cur_hier_level++;
                children_to_skip = popcount(node_bitmap_pos + 1, child_bits, true);
                node_bitmap_pos += CHUNK_WIDTH + 1;
                debugf("[has_child_hier] uncollapsed Bit set, children_to_skip=%zu, advancing to "
                       "level %zu at "
                       "pos %zu\n",
                       children_to_skip, cur_hier_level, node_bitmap_pos);
            }
        }

        debugf("[has_child_hier] === Level %zu ===\n", cur_hier_level);
        debugf("[has_child_hier] Lowest level. Skipping %lu subtrees.\n", children_to_skip);
        while (children_to_skip > 0) {
            debugf("[has_child_hier] Skipping subtree %zu at pos %zu\n", children_to_skip,
                   node_bitmap_pos);
            node_bitmap_pos = skip_hier_subtree(node_bitmap_pos, H_LEVEL);
            children_to_skip--;
        }

        // We are at the lowest level. Check if the bit is set.
        assert(cur_hier_level == H_LEVEL);
        chunk_val_t child_bits = get_hier_level_bits(child_sym, cur_hier_level);
        bool child_found = chunk_contains_child(node_bitmap_pos, child_bits);
        debugf("[has_child_hier] Lowest level: found child = %d\n", child_found);
        return child_found;
    }

public:
    // Gather miscellaneous data about a child node.
    //
    // Given a parent node, check whether the node has a given child.
    //
    // Also counts the number of children less than the provided child, AND the width of
    // the symbol.
    //
    // TODO(yash): We have no need for this function. It isn't even faster than `get_child_info()`.
    // Can we just replace it?
    inline std::tuple<bool, node_pos_t, node_pos_t>
    get_child_info_lite(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                        const morton_type &child_symbol) const
    {
        constexpr width_type num_children = DIMENSION;
        if (node_pos >= flag_size_)
            return {false, 0, 0};

        if (node_is_collapsed(node_pos)) {
            bool found = (child_symbol == read_from_bitmap_pos(node_bitmap_pos, num_children));
            return {found, 0, num_children};
        }

        // TODO(yash): don't count all children.
        node_pos_t skip_count, total_child_count, total_bits;
        bool found = get_child_info_hier(node_bitmap_pos, child_symbol, skip_count,
                                         total_child_count, total_bits);
        return {found, skip_count, total_bits};
    }

    // Check if the symbol is set and return skip count, total children, and symbol width.
    // Returns: (has_symbol, num_nodes_before_symbol, total_children, symbol_width)
    // Use this for range search where all four values are needed.
    inline std::tuple<bool, node_pos_t, node_pos_t, node_pos_t>
    get_child_info(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                   const morton_type &symbol) const
    {
        assert(node_pos < flag_size_);
        assert(node_bitmap_pos < data_size_);

        if (node_is_collapsed(node_pos)) {
            constexpr node_bitmap_pos_t collapsed_node_width = DIMENSION;
            bool found = (symbol == read_from_bitmap_pos(node_bitmap_pos, collapsed_node_width));
            // For collapsed: skip=0, total_children=1, width=DIMENSION
            return {found, 0, 1, collapsed_node_width};
        }

        node_pos_t skip_count = 0, total_children = 0, total_bits = 0;
        bool found =
            get_child_info_hier(node_bitmap_pos, symbol, skip_count, total_children, total_bits);
        return {found, skip_count, total_children, total_bits};
    }

    // Get skip count and symbol width in one traversal.
    // Returns: (num_nodes_before_symbol, symbol_width)
    // Delegates to get_child_skip_and_num_children, discarding num_children.
    //
    // TODO: What's the benefit of this function compared to `get_child_info()`?
    //       Do we really need so many wrappers of this same function?
    //       Just use the relevant field of the returned tuple (eg: `auto [_, x, _, _]`).
    //       ALSO, this performs `popcount` on many more bits than necessary!
    inline std::pair<node_pos_t, node_pos_t>
    get_skip_count_and_symbol_width(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                    const morton_type &symbol) const
    {
        auto [skip, num_children, width] =
            get_num_children_to_skip_and_child_count(node_pos, node_bitmap_pos, symbol);
        (void)num_children; // unused
        return {skip, width};
    }

private:
    /// Recursive helper for `get_child_info_hier()`.
    inline std::pair<bool, node_bitmap_pos_t>
    get_child_info_hier_(node_bitmap_pos_t cur_bitmap_pos, const morton_type &target_child,
                         node_pos_t &children_before_target, node_pos_t &children_after_target,
                         size_t hier_level) const
    {
        assert(hier_level <= H_LEVEL);
        chunk_val_t target_bits = get_hier_level_bits(target_child, hier_level);
        if (!chunk_contains_child(cur_bitmap_pos, target_bits, hier_level)) {
            return {false, 0};
        }

        auto [before, after] =
            count_chunk_children_around_target(cur_bitmap_pos, target_bits, hier_level);

        node_bitmap_pos_t chunk_width = get_chunk_width(cur_bitmap_pos, hier_level);
        cur_bitmap_pos += chunk_width;

        if (hier_level == H_LEVEL) {
            children_before_target += before;
            children_after_target += after;
            return {true, cur_bitmap_pos};
        }

        // Count the subtrees before the target.
        while (before > 0) {
            auto [new_bitmap_pos, children] = count_hier_subtree(cur_bitmap_pos, hier_level + 1);
            cur_bitmap_pos = new_bitmap_pos;
            children_before_target += children;
            before--;
        }

        // Recurse into the target tree.
        auto [target_found, new_bitmap_pos] =
            get_child_info_hier_(cur_bitmap_pos, target_child, children_before_target,
                                 children_after_target, hier_level + 1);
        if (!target_found) {
            return {false, 0};
        }
        cur_bitmap_pos = new_bitmap_pos;

        // Count the subtrees after the target.
        while (after > 0) {
            node_pos_t children_in_subtree;
            std::tie(cur_bitmap_pos, children_in_subtree) =
                count_hier_subtree(cur_bitmap_pos, hier_level + 1);
            children_after_target += children_in_subtree;
            after--;
        }
        return {true, cur_bitmap_pos};
    }

    /// Internal helper. Return many statistics about a hierarchically encoded node, in one pass!
    ///
    /// If the `target_child` doesn't exist in the bitmap, then all other output parameters
    /// are undefined.
    ///
    /// @param node_bitmap_pos position of parent node
    /// @param target_child a child symbol (affects returned statistics)
    /// @param children_before_target output param. Counts the children less than
    /// `target_child`.
    /// @param total_num_children output param. Counts the total number of children this node has.
    /// @param symbol_width output param. Returns the width of this parent symbol (in bits).
    /// @return True if `target_child` is present, false otherwise.
    inline bool get_child_info_hier(node_bitmap_pos_t node_bitmap_pos,
                                    const morton_type &target_child,
                                    node_pos_t &children_before_target, node_pos_t &total_children,
                                    node_pos_t &symbol_width) const
    {
        debugf("[get_child_info_hier] BEGIN: node_bitmap_pos=%lu, target_child=%s\n",
               (unsigned long)node_bitmap_pos, target_child.to_hex().c_str());

        // Set up recursive accumulator params
        children_before_target = 0;
        node_pos_t children_after_target = 0;

        auto [target_found, final_bitmap_pos] = get_child_info_hier_(
            node_bitmap_pos, target_child, children_before_target, children_after_target, 0);
        if (!target_found) {
            return false;
        }

        // Transform accumulator params to output params
        total_children = children_before_target + 1 + children_after_target;
        symbol_width = final_bitmap_pos - node_bitmap_pos;
        return true;
    }

public:
    // return the number of children owned by the node at node_pos
    inline node_pos_t get_num_children(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos) const
    {
        if (node_is_collapsed(node_pos))
            return 1;

        // Node uses hierarchical encoding. Count the number of children in the hierarchical tree.
        auto [num_child_subtrees, chunk_width] = count_toplevel_chunk_children(node_bitmap_pos);
        node_bitmap_pos += chunk_width;

        // Count the child nodes in each subtree of the root chunk.
        node_pos_t num_children = 0;
        while (num_child_subtrees > 0) {
            auto [new_bitmap_pos, children] = count_hier_subtree(node_bitmap_pos, 1);
            node_bitmap_pos = new_bitmap_pos;
            num_children += children;

            num_child_subtrees--;
        }
        return num_children;
    }

    /// Fused Operator: Helper Function:
    ///
    /// In one pass, computes:
    /// 1. The width of the node symbol at `node_bitmap_pos` (like `get_symbol_width()`)
    /// 2. The number of children from this node (like `get_num_children()`)
    ///
    /// @return (symbol width, num children)
    inline std::pair<node_pos_t, node_pos_t>
    get_symbol_width_and_num_children(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos) const
    {
        if (node_is_collapsed(node_pos)) {
            return {DIMENSION, 1};
        }

        const node_pos_t original_bitmap_pos = node_bitmap_pos;

        // Node uses hierarchical encoding. Count the number of children in the hierarchical tree.
        auto [num_child_subtrees, chunk_width] = count_toplevel_chunk_children(node_bitmap_pos);
        node_bitmap_pos += chunk_width;

        // Count the child nodes in each subtree of the root chunk.
        node_pos_t num_children = 0;
        while (num_child_subtrees > 0) {
            auto [new_bitmap_pos, children] = count_hier_subtree(node_bitmap_pos, 1);
            node_bitmap_pos = new_bitmap_pos;
            num_children += children;

            num_child_subtrees--;
        }

        return {(node_bitmap_pos - original_bitmap_pos), num_children};
    }

    /// Given a node position in an existing bitmap, set the symbol at node/node_pos.
    ///
    /// This function shifts the remaining bits in the bitmap as needed, to make room for the new
    /// symbol.
    ///
    /// @param preallocate: The number of extra data bits to pre-allocate after this symbol.
    //                      This is a performance optimization for when the caller needs to write
    //                      something immediately after this symbol, after calling this fn.
    //                      extend_treeblock)
    /// @return True if the bit was newly set, false if the child bit already existed.
    inline bool
    set_child_in_node(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                      const morton_type &child_symbol, bool node_is_collapsed,
                      node_pos_t &num_nodes_before_symbol, // essentially `n_children_to_skip`
                                                           // in get_child_node_unsafe
                      node_pos_t &symbol_width_after_operation, width_type preallocate = 0)
    {
        constexpr width_type num_children = DIMENSION;
        if (node_is_collapsed) {
            // If the symbol at `node` already encodes our `symbol`, then we
            // don't need to do anything.
            morton_type prior_symbol = read_from_bitmap_pos(node_bitmap_pos, num_children);
            if (child_symbol == prior_symbol) {
                return false;
            }

            set_child_in_collapsed_node(node_pos, node_bitmap_pos, prior_symbol, child_symbol,
                                        num_nodes_before_symbol, symbol_width_after_operation,
                                        preallocate);
            return true;
        }

        // already uncollapsed
        bool inserted =
            set_child_in_hier_node(node_bitmap_pos, child_symbol, num_nodes_before_symbol,
                                   symbol_width_after_operation, preallocate);
        return inserted;
    }

    // Helper: Find the first set bit in a chunked bitmap, handling >64 bit chunks
    // Returns the bit index of the first set bit, and updates word/bit_base/chunk_bits in level
    inline node_pos_t find_first_set_bit_in_chunk(node_bitmap_pos_t chunk_data_pos,
                                                  const node_pos_t chunk_width,
                                                  level_info &level) const
    {
        level.bit_base = 0;

        width_type width = chunk_width > 64 ? 64 : static_cast<width_type>(chunk_width);
        level.word = read_u64_from_bitmap_pos(chunk_data_pos, width);

        level.remaining = chunk_width - width;

        while (!level.word) {
            assert(level.remaining > 0); // guaranteed to have a single set bit

            width = level.remaining > 64 ? 64 : static_cast<width_type>(level.remaining);
            level.bit_base += width;
            level.remaining -= width;
            level.word = read_u64_from_bitmap_pos(chunk_data_pos + level.bit_base, width);
        }

        unsigned tz = static_cast<unsigned>(__builtin_ctzll(level.word));
        return level.bit_base + tz;
    }

    inline std::tuple<morton_type, node_pos_t>
    next_symbol_reuse_iter(const morton_type &start_symbol_range,
                           const morton_type &end_symbol_range, const bool collapsed_encoding,
                           level_info (&stack)[H_LEVEL + 1], int &top, bool init,
                           node_bitmap_pos_t &next_node_bitmap_pos, morton_type &base_symbol)
    {

        node_pos_t skip_count = 0;

        if (start_symbol_range.is_null())
            return {morton_type::null(), 0};

        // if outer collapsed, don't need to create stack
        if (collapsed_encoding) {
            morton_type only_symbol = read_from_bitmap_pos(next_node_bitmap_pos, DIMENSION);
            if (start_symbol_range <= only_symbol && only_symbol <= end_symbol_range) {
                return {only_symbol, 0};
            }
            return {morton_type::null(), 0};
        }

        // Optimization: Once base_symbol >= start_symbol_range, all subsequent symbols
        // in DFS order are also >= start (DFS visits in sorted order). So we only need
        // to compare with start until we pass it, then never again.
        bool passed_start = false;

        // Lambda to descend from current level to leaf (H_LEVEL)
        // Handles level 0 with different chunk widths
        // Assumes base_symbol is zero when starting from top=-1
        auto descend_to_leaf = [&]() {
            while (top < H_LEVEL) {
                top++;

                const node_pos_t c_width = (top == 0) ? TOP_LEVEL_CHUNK_WIDTH : CHUNK_WIDTH;
                const node_pos_t c_collapsed_width =
                    (top == 0) ? TOP_LEVEL_COLLAPSED_CHUNK_WIDTH : COLLAPSED_CHUNK_WIDTH;

                stack[top].data_pos = next_node_bitmap_pos + 1;
                stack[top].collapsed = read_from_bitmap_pos(next_node_bitmap_pos, 1) == 0;

                if (!stack[top].collapsed) {
                    node_pos_t child_idx =
                        find_first_set_bit_in_chunk(stack[top].data_pos, c_width, stack[top]);
                    base_symbol.add_shifted_u64(child_idx, CHUNK_WIDTH_SHIFT * (H_LEVEL - top));
                    next_node_bitmap_pos += 1 + c_width;
                } else {
                    chunk_val_t only_sym =
                        read_u64_from_bitmap_pos(stack[top].data_pos, c_collapsed_width);
                    base_symbol.add_shifted_u64(only_sym, CHUNK_WIDTH_SHIFT * (H_LEVEL - top));
                    next_node_bitmap_pos += 1 + c_collapsed_width;
                }
            }
        };

        if (init) {
            assert(top == -1);
            descend_to_leaf();

            if (base_symbol > end_symbol_range) {
                return {morton_type::null(), 0};
            }

            if (base_symbol >= start_symbol_range) {
                // First symbol is already in range
                return {base_symbol, 0};
            }
            // First symbol is < start, will search in loop
        } else {
            // Not init - check if already in range
            if (base_symbol >= start_symbol_range && base_symbol <= end_symbol_range) {
                return {base_symbol, 0};
            }
        }

        // currently we are at the last level
        assert(top == H_LEVEL);

        // Traverse the DFS tree to find first symbol in range
        while (top >= 0) {
            if (base_symbol > end_symbol_range) {
                return {morton_type::null(), 0};
            }

            if (top == H_LEVEL) {
                if (!passed_start) {
                    // Still searching for start
                    if (base_symbol >= start_symbol_range) {
                        passed_start = true;
                        if (base_symbol <= end_symbol_range) {
                            return {base_symbol, skip_count};
                        }
                    } else {
                        skip_count++;
                    }
                } else {
                    // Already passed start, only check end
                    if (base_symbol <= end_symbol_range) {
                        return {base_symbol, skip_count};
                    }
                }
            }

            const node_pos_t c_shift = (top == 0) ? TOP_LEVEL_CHUNK_WIDTH_SHIFT : CHUNK_WIDTH_SHIFT;

            if (!stack[top].collapsed) {
                node_pos_t &remaining = stack[top].remaining;
                uint64_t &word = stack[top].word;
                node_pos_t &bit_base = stack[top].bit_base;

                // Clear current bit, move to next sibling
                word &= word - 1;

                // Find next set bit in this level
                while (!word && remaining) {
                    width_type width = remaining > 64 ? 64 : static_cast<width_type>(remaining);
                    remaining -= width;
                    bit_base += width;
                    word = read_u64_from_bitmap_pos(stack[top].data_pos + bit_base, width);
                }

                if (!word) {
                    // No more siblings at this level, pop up
                    base_symbol.clear_bits_at(CHUNK_WIDTH_SHIFT * (H_LEVEL - top), c_shift);
                    top--;
                    continue;
                }

                // Found next sibling
                unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                node_pos_t child_idx = bit_base + tz;

                base_symbol.replace_shifted_u64(child_idx, CHUNK_WIDTH_SHIFT * (H_LEVEL - top),
                                                c_shift);

                // Descend to leaf from this sibling
                descend_to_leaf();

            } else {
                // Collapsed node has no siblings, pop up
                base_symbol.clear_bits_at(CHUNK_WIDTH_SHIFT * (H_LEVEL - top), c_shift);
                top--;
            }
        }

        return {morton_type::null(), 0};
    }

#ifndef NDEBUG
    // return the next symbol starting from the node/node_pos that is >= symbol
    // and <= end_symbol_range, along with n_children_to_skip (children before the symbol)
    // Returns: (next_symbol, n_children_to_skip)
    inline std::pair<morton_type, node_pos_t>
    next_symbol(const morton_type &symbol, node_bitmap_pos_t node_bitmap_pos,
                const morton_type &end_symbol_range, const bool collapsed_encoding,
                const bool inner_top_level_collapse,
                node_bitmap_pos_t (&levels_pos_iter)[H_LEVEL + 1],
                node_bitmap_pos_t (&bit_bases)[H_LEVEL + 1],
                node_bitmap_pos_t (&words)[H_LEVEL + 1], node_bitmap_pos_t bitmap_width) const
    {
        // DFS implementation using cumulative count tracking.
        // O(N * depth) - re-traverses on each call to compute total count.
        //
        // We use words[0] to store cumulative count from previous calls.
        // Each call computes the total count up to the found symbol, then
        // returns the increment.
        //
        // Note: True O(N + depth) for DFS requires complex state management
        // for resumption through collapsed paths, which is not implemented.
        (void)levels_pos_iter;
        (void)bit_bases;
        (void)bitmap_width;

        node_pos_t previous_cumulative = words[0];

        if (symbol.is_null())
            return {symbol, 0};

        // Collapsed encoding - single child
        if (collapsed_encoding) {
            morton_type only_symbol = read_from_bitmap_pos(node_bitmap_pos, DIMENSION);
            if (symbol <= only_symbol && only_symbol <= end_symbol_range) {
                words[0] = 1;
                return {only_symbol, 0};
            }
            return {morton_type::null(), 0};
        }

        // Hierarchical encoding - traverse to find next symbol
        node_pos_t total_before_found = 0;
        morton_type found_sym = morton_type::null();

        if (inner_top_level_collapse) {
            chunk_val_t top_level_symbol =
                read_u64_from_bitmap_pos(node_bitmap_pos + 1, TOP_LEVEL_COLLAPSED_CHUNK_WIDTH);
            morton_type parent_morton =
                morton_type::from_shifted_u64(top_level_symbol, CHUNK_WIDTH_SHIFT * H_LEVEL);

            if (parent_morton > end_symbol_range)
                return {morton_type::null(), 0};

            node_bitmap_pos_t next_pos = node_bitmap_pos + 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH;
            std::tie(found_sym, total_before_found, std::ignore) =
                find_next_symbol_dfs_total(next_pos, parent_morton, 1, symbol, end_symbol_range);
        } else {
            node_bitmap_pos_t chunk_data_pos = node_bitmap_pos + 1;
            node_bitmap_pos_t next_pos = node_bitmap_pos + 1 + TOP_LEVEL_CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = TOP_LEVEL_CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : static_cast<width_type>(remaining);
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type parent_morton =
                        morton_type::from_shifted_u64(child_idx, CHUNK_WIDTH_SHIFT * H_LEVEL);

                    if (parent_morton > end_symbol_range) {
                        found_sym = morton_type::null();
                        goto done;
                    }

                    auto [sym, skip, end_pos] = find_next_symbol_dfs_total(
                        next_pos, parent_morton, 1, symbol, end_symbol_range);

                    if (!sym.is_null()) {
                        found_sym = sym;
                        total_before_found += skip;
                        goto done;
                    }

                    total_before_found += skip;
                    next_pos = end_pos;
                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
        }

    done:
        if (found_sym.is_null()) {
            return {morton_type::null(), 0};
        }

        node_pos_t increment = total_before_found - previous_cumulative;
        words[0] = total_before_found + 1;

        return {found_sym, increment};
    }
#endif

private:
#ifndef NDEBUG
    // DFS helper that returns TOTAL count of symbols before the found symbol (from beginning of
    // hierarchy) Returns (found_symbol, total_symbols_before_found, next_bitmap_pos)
    std::tuple<morton_type, node_pos_t, node_bitmap_pos_t>
    find_next_symbol_dfs_total(node_bitmap_pos_t bitmap_pos, const morton_type &parent_morton,
                               size_t hier_level, const morton_type &start_symbol,
                               const morton_type &end_symbol) const
    {
        node_pos_t total_before = 0;

        // Last level (H_LEVEL): individual symbols
        if (hier_level == H_LEVEL) {
            if (chunk_is_collapsed(bitmap_pos)) {
                chunk_val_t only_sym =
                    read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
                morton_type target = morton_type::copy_and_or_low_u64(parent_morton, only_sym);

                node_bitmap_pos_t end_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
                if (target > end_symbol)
                    return {morton_type::null(), 1, end_pos}; // 1 symbol exists but > end
                if (target >= start_symbol)
                    return {target, 0, end_pos}; // Found; 0 symbols before it in this chunk
                // target < start_symbol: 1 symbol skipped
                return {morton_type::null(), 1, end_pos};
            } else {
                node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
                node_pos_t bit_base = 0;
                node_pos_t remaining = CHUNK_WIDTH;

                while (remaining > 0) {
                    width_type width = (remaining > 64) ? 64 : remaining;
                    uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                    while (word) {
                        unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                        node_pos_t i = bit_base + tz;
                        morton_type target = parent_morton | i;

                        if (target > end_symbol) {
                            // No symbol in range; count remaining as skipped
                            total_before += __builtin_popcountll(word);
                            // Continue counting rest of chunk
                            word = 0;
                            continue;
                        }
                        if (target >= start_symbol) {
                            return {target, total_before, bitmap_pos + 1 + CHUNK_WIDTH};
                        }

                        total_before++;
                        word &= word - 1;
                    }
                    bit_base += width;
                    remaining -= width;
                }
                // No symbol found in range
                return {morton_type::null(), total_before, bitmap_pos + 1 + CHUNK_WIDTH};
            }
        }

        // Non-leaf levels
        const int shift = CHUNK_WIDTH_SHIFT * (H_LEVEL - hier_level);

        if (chunk_is_collapsed(bitmap_pos)) {
            chunk_val_t only_sym = read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
            morton_type new_parent = parent_morton;
            new_parent.add_shifted_u64(only_sym, shift);

            node_bitmap_pos_t next_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
            return find_next_symbol_dfs_total(next_pos, new_parent, hier_level + 1, start_symbol,
                                              end_symbol);
        } else {
            node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
            node_bitmap_pos_t next_pos = bitmap_pos + 1 + CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : remaining;
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type new_parent = parent_morton;
                    new_parent.add_shifted_u64(child_idx, shift);

                    auto [found_sym, child_count, end_pos] = find_next_symbol_dfs_total(
                        next_pos, new_parent, hier_level + 1, start_symbol, end_symbol);
                    next_pos = end_pos;

                    if (!found_sym.is_null()) {
                        return {found_sym, total_before + child_count, next_pos};
                    }

                    total_before += child_count;
                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
            return {morton_type::null(), total_before, next_pos};
        }
    }

    /* for debug purpose only, when asserting (reuse == debug)*/
    // DFS helper to find the first symbol >= start_symbol and <= end_symbol
    // Returns (found_symbol, skip_count, next_bitmap_pos)
    // If not found, returns (null, skip_count, next_bitmap_pos)
    //
    // Important: skip_count represents symbols we passed over BETWEEN start_symbol and
    // found_symbol. This is computed as (position of found) - (position of start), where position
    // is the count of symbols < that value.
    std::tuple<morton_type, node_pos_t, node_bitmap_pos_t>
    find_next_symbol_dfs_debug(node_bitmap_pos_t bitmap_pos, const morton_type &parent_morton,
                               size_t hier_level, const morton_type &start_symbol,
                               const morton_type &end_symbol) const
    {
        node_pos_t symbols_before_start = 0; // Symbols < start_symbol
        node_pos_t symbols_before_found = 0; // Symbols < found_symbol (includes those < start)

        // Last level (H_LEVEL): individual symbols
        if (hier_level == H_LEVEL) {
            if (chunk_is_collapsed(bitmap_pos)) {
                chunk_val_t only_sym =
                    read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
                morton_type target = morton_type::copy_and_or_low_u64(parent_morton, only_sym);

                node_bitmap_pos_t end_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
                if (target > end_symbol)
                    return {morton_type::null(), 0, end_pos};
                if (target >= start_symbol)
                    return {target, 0, end_pos}; // Found at position 0 (no skips)
                // target < start_symbol: this symbol is skipped
                return {morton_type::null(), 1, end_pos};
            } else {
                node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
                node_pos_t bit_base = 0;
                node_pos_t remaining = CHUNK_WIDTH;

                while (remaining > 0) {
                    width_type width = (remaining > 64) ? 64 : remaining;
                    uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                    while (word) {
                        unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                        node_pos_t i = bit_base + tz;
                        morton_type target = parent_morton | i;

                        if (target > end_symbol) {
                            // No symbol in range; return total skips (before start) for position
                            // tracking
                            return {morton_type::null(), symbols_before_start,
                                    bitmap_pos + 1 + CHUNK_WIDTH};
                        }
                        if (target >= start_symbol) {
                            // Found! Return increment = symbols_before_found - symbols_before_start
                            // = 0 (since found is the first >= start_symbol we see, no symbols
                            // between)
                            return {target, symbols_before_found - symbols_before_start,
                                    bitmap_pos + 1 + CHUNK_WIDTH};
                        }

                        // target < start_symbol
                        symbols_before_start++;
                        symbols_before_found++;
                        word &= word - 1;
                    }
                    bit_base += width;
                    remaining -= width;
                }
                // No symbol found in range
                return {morton_type::null(), symbols_before_start, bitmap_pos + 1 + CHUNK_WIDTH};
            }
        }

        // Non-leaf levels
        const int shift = CHUNK_WIDTH_SHIFT * (H_LEVEL - hier_level);

        if (chunk_is_collapsed(bitmap_pos)) {
            chunk_val_t only_sym = read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
            morton_type new_parent = parent_morton;
            new_parent.add_shifted_u64(only_sym, shift);

            node_bitmap_pos_t next_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
            return find_next_symbol_dfs_debug(next_pos, new_parent, hier_level + 1, start_symbol,
                                              end_symbol);
        } else {
            node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
            node_bitmap_pos_t next_pos = bitmap_pos + 1 + CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : remaining;
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type new_parent = parent_morton;
                    new_parent.add_shifted_u64(child_idx, shift);

                    auto [found_sym, found_skip, end_pos] = find_next_symbol_dfs_debug(
                        next_pos, new_parent, hier_level + 1, start_symbol, end_symbol);
                    next_pos = end_pos;

                    if (!found_sym.is_null()) {
                        // Found a symbol in range
                        return {found_sym, symbols_before_start + found_skip, next_pos};
                    }

                    // No symbol found in this subtree; found_skip = symbols < start in subtree
                    symbols_before_start += found_skip;
                    symbols_before_found += found_skip;

                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
            return {morton_type::null(), symbols_before_start, next_pos};
        }
    }
#endif

public:
    // Write a morton symbol to a location in the bitmap.
    //
    // This function assumes preallocated memory. It overwrites what's already there.
    inline void create_collapsed_node_unsafe(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                             const morton_type &symbol)
    {
        constexpr width_type collapsed_node_width = DIMENSION;
        assert(node_bitmap_pos + collapsed_node_width <= data_size_);
        assert(read_from_bitmap_pos(node_bitmap_pos, collapsed_node_width) == 0);

        // Write the node data to the bitmap.
        write_to_bitmap_pos(node_bitmap_pos, symbol, collapsed_node_width);

        // We can assume the new node will always be a "collapsed" node.
        // Since the "default" value of the "flags_" array is 0 (=> "collapsed"),
        (void)node_pos;
        assert(node_is_collapsed(node_pos));
    }

    // Copy x bits from a starting bit location in this bitmap to another bitmap, starting at bit 0
    void copy_bits_to(compressed_bitmap &dest, node_bitmap_pos_t src_start, width_type x,
                      bool is_on_data) const
    {
        node_bitmap_pos_t dest_pos = 0;
        while (x > 0) {
            width_type chunk = (x > 64) ? 64 : x;
            data_type val = GetValPosU64(src_start, chunk, is_on_data);
            dest.SetValPos(dest_pos, val, chunk, is_on_data);
            src_start += chunk;
            dest_pos += chunk;
            x -= chunk;
        }
    }

    // recursive helper for get_num_children_to_skip
    // counts how many children before the desired symbol, breaks early if the symbol already passes
    node_pos_t count_all_children_less_than_symbol(node_bitmap_pos_t (&levels_start)[H_LEVEL + 1],
                                                   const morton_type &parent_morton,
                                                   trie_level_t level,
                                                   const morton_type &symbol_morton,
                                                   bool &found) const
    {
        (void)levels_start;
        (void)parent_morton;
        (void)level;
        (void)symbol_morton;
        (void)found;
        assert(false);
        return 0;
    }

    // DFS recursive helper: in one traversal compute how many symbols appear before
    // `symbol_morton` (skip count) and the total number of symbols in this subtree.
    // Returns (skip, total, next_bitmap_pos) since DFS ordering means children follow parents.
    //
    // @param bitmap_pos: Current position in the bitmap
    // @param parent_morton: Morton prefix built up from parent levels
    // @param hier_level: Current hierarchical level (1 to H_LEVEL, not 0 which is top-level)
    // @param symbol_morton: The symbol we're comparing against
    // @return Tuple of (skip, total, next_bitmap_pos)
    std::tuple<node_pos_t, node_pos_t, node_bitmap_pos_t>
    count_skip_and_total_symbols_dfs(node_bitmap_pos_t bitmap_pos, const morton_type &parent_morton,
                                     size_t hier_level, const morton_type &symbol_morton) const
    {
        node_pos_t skip = 0;
        node_pos_t total = 0;

        // Last level (H_LEVEL): individual symbols
        if (hier_level == H_LEVEL) {
            if (chunk_is_collapsed(bitmap_pos)) {
                // Collapsed: single child
                chunk_val_t only_symbol =
                    read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
                // todo: is this construction pattern optimal?
                morton_type target_morton =
                    morton_type::copy_and_or_low_u64(parent_morton, only_symbol);

                total = 1;
                if (target_morton < symbol_morton) {
                    skip = 1;
                }
                return {skip, total, bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH};
            } else {
                // Uncollapsed: scan CHUNK_WIDTH bits
                node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
                node_pos_t bit_base = 0;
                node_pos_t remaining = CHUNK_WIDTH;

                while (remaining > 0) {
                    width_type width = (remaining > 64) ? 64 : remaining;
                    uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                    while (word) {
                        unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                        node_pos_t i = bit_base + tz;
                        morton_type target_morton = parent_morton | i;

                        total++;
                        if (target_morton < symbol_morton) {
                            skip++;
                        }

                        word &= word - 1;
                    }
                    bit_base += width;
                    remaining -= width;
                }
                return {skip, total, bitmap_pos + 1 + CHUNK_WIDTH};
            }
        }

        // Non-leaf levels: recurse into children in DFS order
        const int shift = CHUNK_WIDTH_SHIFT * (H_LEVEL - hier_level);

        if (chunk_is_collapsed(bitmap_pos)) {
            // Collapsed: single child, recurse immediately
            chunk_val_t only_symbol =
                read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
            morton_type new_parent = parent_morton;
            new_parent.add_shifted_u64(only_symbol, shift);

            node_bitmap_pos_t next_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
            auto [child_skip, child_total, end_pos] = count_skip_and_total_symbols_dfs(
                next_pos, new_parent, hier_level + 1, symbol_morton);
            return {child_skip, child_total, end_pos};
        } else {
            // Uncollapsed: iterate through set bits and recurse into each child
            node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
            node_bitmap_pos_t next_pos = bitmap_pos + 1 + CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : remaining;
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type new_parent = parent_morton;
                    new_parent.add_shifted_u64(child_idx, shift);

                    // DFS: recurse into child subtree immediately
                    auto [child_skip, child_total, end_pos] = count_skip_and_total_symbols_dfs(
                        next_pos, new_parent, hier_level + 1, symbol_morton);
                    skip += child_skip;
                    total += child_total;
                    next_pos = end_pos;

                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
            return {skip, total, next_pos};
        }
    }

    /// Helper function for get_child(). Collect stats about subtree rooted on node.
    ///
    /// Combined helper: computes get_num_children_to_skip (how many children before `symbol`),
    /// get_num_children (total children under node), and get_symbol_width (total bits)
    /// in a single traversal to avoid redundant scanning.
    ///
    /// @return Tuple of (children_to_skip, total_children, symbol_width).
    inline std::tuple<node_pos_t, node_pos_t, node_pos_t>
    get_num_children_to_skip_and_child_count(node_pos_t parent_node_pos,
                                             node_bitmap_pos_t parent_node_bitmap_pos,
                                             const morton_type &child_symbol) const
    {
        constexpr width_type num_children = DIMENSION;
        node_pos_t out_children_to_skip = 0;
        node_pos_t out_child_count = 0;

        if (node_is_collapsed(parent_node_pos)) {
            out_child_count = 1;
            morton_type only_symbol = read_from_bitmap_pos(parent_node_bitmap_pos, num_children);
            out_children_to_skip = (child_symbol > only_symbol) ? 1 : 0;
            return {out_children_to_skip, out_child_count, num_children}; // width = DIMENSION
        }

        // Hierarchical node: handle top-level chunk specially, then use DFS traversal
        node_bitmap_pos_t next_pos;
        node_bitmap_pos_t symbol_width;

        if (chunk_is_collapsed(parent_node_bitmap_pos)) {
            // Collapsed top-level: single subtree
            chunk_val_t top_level_symbol = read_u64_from_bitmap_pos(
                parent_node_bitmap_pos + 1, TOP_LEVEL_COLLAPSED_CHUNK_WIDTH);
            morton_type parent_morton =
                morton_type::from_shifted_u64(top_level_symbol, CHUNK_WIDTH_SHIFT * H_LEVEL);

            next_pos = parent_node_bitmap_pos + 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH;
            auto [skip, total, end_pos] =
                count_skip_and_total_symbols_dfs(next_pos, parent_morton, 1, child_symbol);
            out_children_to_skip = skip;
            out_child_count = total;
            symbol_width = end_pos - parent_node_bitmap_pos;
        } else {
            // Uncollapsed top-level: iterate active top-level children
            node_bitmap_pos_t chunk_data_pos = parent_node_bitmap_pos + 1;
            next_pos = parent_node_bitmap_pos + 1 + TOP_LEVEL_CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = TOP_LEVEL_CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : remaining;
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type parent_morton =
                        morton_type::from_shifted_u64(child_idx, CHUNK_WIDTH_SHIFT * H_LEVEL);

                    // DFS: recurse into child subtree
                    auto [skip, total, end_pos] =
                        count_skip_and_total_symbols_dfs(next_pos, parent_morton, 1, child_symbol);
                    out_children_to_skip += skip;
                    out_child_count += total;
                    next_pos = end_pos;

                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
            symbol_width = next_pos - parent_node_bitmap_pos;
        }

        return {out_children_to_skip, out_child_count, symbol_width};
    }

    // ----------------------------
    // API : SIMPLE BIT OPERATIONS
    // ----------------------------

    inline data_type *get_bitmap() const { return data_; }
    inline data_type *get_flagmap() const { return flag_; }
    inline size_type get_bitmap_size() const { return data_size_; }
    inline size_type get_flagmap_size() const { return flag_size_; }
    inline void set_bitmap(uint64_t data) { data_ = (data_type *)data; }
    inline void set_flagmap(uint64_t flag) { flag_ = (data_type *)flag; }

    // Increase the size of the array that holds the main data bits!
    inline void increase_bitmap_size(width_type increase_size)
    {
        grow_array(increase_size, this->data_size_, this->data_);
    }

    // Increase the size of the array that holds the "flags" bits!
    inline void increase_flagmap_size(width_type increase_size)
    {
        grow_array(increase_size, this->flag_size_, this->flag_);
    }

    // Decrease the size of the array that holds the main data bits!
    inline void decrease_bitmap_size(width_type decrease_size)
    {
        shrink_array(decrease_size, true);
    }

    // Decrease the size of the array that holds the "flags" bits!
    inline void decrease_flagmap_size(width_type decrease_size)
    {
        shrink_array(decrease_size, false);
    }

    // Extract `num_bits` starting at `pos` into a morton_type (packed into its LSBs)
    inline morton_type read_from_bitmap_pos(node_bitmap_pos_t bitmap_pos, width_type num_bits) const
    {
        return this->GetValPos(bitmap_pos, num_bits, true);
    }

    // Extract `num_bits` starting at `pos` into a morton_type from the "flag" bitmask.
    inline morton_type read_from_flagmap_pos(node_pos_t flagmap_pos, width_type num_bits) const
    {
        return this->GetValPos(flagmap_pos, num_bits, false);
    }

    inline uint64_t read_u64_from_bitmap_pos(node_bitmap_pos_t bitmap_pos,
                                             width_type num_bits) const
    {
        return this->GetValPosU64(bitmap_pos, num_bits, true);
    }

    // Get 64 bits from this compressed bitmap's "Flag" bit mask.
    inline uint64_t read_u64_from_flagmap_pos(node_bitmap_pos_t bitmap_pos,
                                              width_type num_bits) const
    {
        return this->GetValPosU64(bitmap_pos, num_bits, false);
    }

    // Write arbitrary data to a poisition.
    inline void write_to_bitmap_pos(node_bitmap_pos_t pos, const morton_type &val,
                                    width_type num_bits)
    {
        this->SetValPos(pos, val, num_bits, true);
    }

    // Fast-path for writing a single machine word of data to a position.
    inline void write_to_bitmap_pos(node_bitmap_pos_t pos, uint64_t val, width_type num_bits)
    {
        this->SetValPos(pos, val, num_bits, true);
    }

    inline void clear_bitmap_pos(node_bitmap_pos_t bitmap_pos, width_type num_bits)
    {
        this->clear_array(bitmap_pos, num_bits, true);
    }

    inline void clear_flagmap_pos(node_bitmap_pos_t bitmap_pos, width_type num_bits)
    {
        this->clear_array(bitmap_pos, num_bits, false);
    }

    inline void count_bitmap_bits_in_range(node_bitmap_pos_t bitmap_pos, width_type width)
    {
        this->popcount(bitmap_pos, width, true);
    }

    inline void count_flagmap_bits_in_range(node_bitmap_pos_t bitmap_pos, width_type width)
    {
        this->popcount(bitmap_pos, width, false);
    }

    // the current start is `current_offset`, just need to write there.
    // added to return the data_offset and flag_offset for setting in parent
    // tree_block
    //
    // writing data_ and flag_ to file, returning the corresponding offsets
    void serialize(FILE *file, uint64_t &data_offset_on_file, uint64_t &flag_offset_on_file) const
    {
        // note: not a good idea to write in bit-granularity, makes other accesses too slow

        // there's no way the data_ and flag_ are already in the pointers_to_offsets_map
        if (data_)
            assert(pointers_to_offsets_map.find((uint64_t)data_) == pointers_to_offsets_map.end());

        if (flag_)
            assert(pointers_to_offsets_map.find((uint64_t)flag_) == pointers_to_offsets_map.end());

        // assert that data_ won't be null when data_size_ > 0 and vice
        // versa for flag_ and flaog_size_
        assert((data_size_ == 0) == (data_ == nullptr));
        assert((flag_size_ == 0) == (flag_ == nullptr));

        // updated: since now a struct inside tree_block:
        // I am writing contiguous any way
        //     | tree_block | data_ | flag_ |

        // serialize the data
        if (data_size_ > 0) {
            pointers_to_offsets_map.insert({(uint64_t)data_, current_offset});

            data_offset_on_file = current_offset;
            current_offset += BITS2BLOCKS(data_size_) * sizeof(data_type);
            fwrite(data_, sizeof(data_type), BITS2BLOCKS(data_size_), file);
        }

        // serialize the flag
        if (flag_size_ > 0) {
            // assert(flag_len == flag_size_);
            pointers_to_offsets_map.insert({(uint64_t)flag_, current_offset});

            flag_offset_on_file = current_offset;
            current_offset += BITS2BLOCKS(flag_size_) * sizeof(data_type);
            fwrite(flag_, sizeof(data_type), BITS2BLOCKS(flag_size_), file);
        }
    }

    void deserialize(uint64_t base_addr)
    {
        assert((data_size_ == 0) == (data_ == nullptr));
        assert((flag_size_ == 0) == (flag_ == nullptr));

        if (data_size_ > 0) {
            data_ = (data_type *)(base_addr + (uint64_t)data_);
        }

        if (flag_size_ > 0) {
            flag_ = (data_type *)(base_addr + (uint64_t)flag_);
        }
    }

    /// Quickly "seek" to the position in a hier node where a child should be inserted.
    ///
    /// A helper function for set_child_node_hier. It operates on hierarchically-encoded symbols
    /// only. It traverses a hierarchical node in search of a given child, until
    /// it hits a level that doesn't contain the child's bits.
    /// Callers can use this function to quickly "seek" to the point where an
    /// insertion should occur.
    ///
    /// Assumes the node there is already in hier format.
    ///
    /// @param parent_bitmap_pos: Bitmap position where the hierarchically-encoded parent node
    ///                           starts. This is the node we're adding a child to.
    /// @param child_sym: The morton-encoded symbol we're searching for
    /// @param num_nodes_before_child: [OUTPUT] Count of children before the target
    /// @param num_subtrees_before_child: [OUTPUT] Subtrees before target at the level where we stop
    /// @param num_subtrees_after_child: [OUTPUT] Subtrees after target at each level (size H_LEVEL)
    /// @param cur_bitmap_pos: [OUTPUT] Current position in bitmap when we stop.
    ///                        Stops _at_ the chunk that requires child insertion (ie: to parent
    ///                        chunk)
    /// @param cur_hier_level: [OUTPUT] Current hierarchical level when we stop
    /// @return True if the child was already found. False otherwise.
    bool traverse_hier_until_missing(const node_bitmap_pos_t parent_bitmap_pos,
                                     const morton_type &child_sym,
                                     node_pos_t &num_nodes_before_child,
                                     node_pos_t &num_subtrees_before_child,
                                     node_pos_t num_subtrees_after_child[H_LEVEL],
                                     node_bitmap_pos_t &cur_bitmap_pos,
                                     size_t &cur_hier_level) const
    {
        debugf("[traverse_hier_until_missing] Starting at bitmap_pos=%lu\n", parent_bitmap_pos);

        // Initialize output parameters.
        num_nodes_before_child = 0;
        num_subtrees_before_child = 0;
        memset(num_subtrees_after_child, 0, H_LEVEL * sizeof(node_pos_t));
        cur_bitmap_pos = parent_bitmap_pos;
        cur_hier_level = 0;

        // PHASE 1: ANALYZE TOP LEVEL CHUNK

        chunk_val_t target_symbol_bits = get_hier_top_level_bits(child_sym);
        debugf("[traverse_hier_until_missing] PHASE 1: target_symbol_bits=%lu\n",
               target_symbol_bits);

        // Break down the top chunk: count the number of subtrees _before_ and _after_ our target.
        if (chunk_is_collapsed(cur_bitmap_pos)) {
            // Top level chunk is collapsed. Check if it encodes our symbol.
            node_pos_t child_symbol_bits =
                read_u64_from_bitmap_pos(cur_bitmap_pos + 1, TOP_LEVEL_CHUNK_WIDTH_SHIFT);

            debugf("[traverse_hier_until_missing] Top level collapsed\n");

            if (child_symbol_bits < target_symbol_bits) {
                // Chunk doesn't encode our symbol - need to insert here.
                // The children of this parent lie _before_ our new insertion point.
                num_subtrees_before_child = 1;
                num_subtrees_after_child[0] = 0;
                debugf(
                    "[traverse_hier_until_missing] Child missing (collapsed < target), returning "
                    "false\n");
                return false;
            } else if (child_symbol_bits > target_symbol_bits) {
                // Chunk doesn't encode our symbol - need to insert after the current child.
                // The children of this parent lie _after_ our new insertion point.
                num_subtrees_before_child = 0;
                num_subtrees_after_child[0] = 1;
                debugf(
                    "[traverse_hier_until_missing] Child missing (collapsed > target), returning "
                    "false\n");
                return false;
            }

            // Top level chunk _does_ encode our symbol. Advance to the next symbol in the trie.
            assert(num_subtrees_before_child == 0);
            assert(num_subtrees_after_child[cur_hier_level] == 0);
            cur_hier_level++;
            cur_bitmap_pos += 1 + TOP_LEVEL_CHUNK_WIDTH_SHIFT; // collapsed bit + width
            debugf("[traverse_hier_until_missing] Top level matched, advancing to level %zu at pos "
                   "%lu\n",
                   cur_hier_level, cur_bitmap_pos);
        } else { // Uncollapsed top level chunk.
            auto [before, after] = count_uncollapsed_toplevel_chunk_children_around_target(
                cur_bitmap_pos, target_symbol_bits);
            num_subtrees_before_child = before;
            num_subtrees_after_child[0] = after;

            debugf(
                "[traverse_hier_until_missing] Top level was uncollapsed: before=%lu, after=%lu\n",
                before, after);

            // Check if the bit corresponding to our child symbol is set.
            // If not, great! Start on the top level.
            if (!uncollapsed_chunk_contains_child(cur_bitmap_pos, target_symbol_bits)) {
                debugf(
                    "[traverse_hier_until_missing] Child missing (bit not set), returning false\n");
                return false;
            }
            // Advance to the next symbol in the trie.
            cur_hier_level++;
            cur_bitmap_pos += TOP_LEVEL_CHUNK_WIDTH + 1; // width + "collapsed bit"
            debugf("[traverse_hier_until_missing] Top level bit set, advancing to level %zu at pos "
                   "%lu\n",
                   cur_hier_level, cur_bitmap_pos);
        }

        // PHASE 2: Seek through our DFS tree until we hit our "target" or find it missing.

        debugf("[traverse_hier_until_missing] PHASE 2: Seeking through DFS tree\n");

        // PHASE 2.1: Count the children in all subtrees BEFORE the one our target resides in.

        while (num_subtrees_before_child > 0) {
            auto [new_bitmap_pos, child_count] = count_hier_subtree(cur_bitmap_pos, cur_hier_level);
            cur_bitmap_pos = new_bitmap_pos;
            num_nodes_before_child += child_count;

            num_subtrees_before_child--;
        }

        // Recurse into each hierarchical level, tallying the children before and after our target
        // child.
        assert(cur_hier_level == 1);
        while (cur_hier_level < H_LEVEL) {

            debugf("[traverse_hier_until_missing] === Level %zu ===\n", cur_hier_level);

            // PHASE 2.2: Check the chunk that includes our target.

            // Parse the chunk that includes our target child.
            target_symbol_bits = get_hier_level_bits(child_sym, cur_hier_level);

            debugf("[traverse_hier_until_missing] At pos %lu, target_symbol_bits=%lu\n",
                   cur_bitmap_pos, target_symbol_bits);

            // Check whether this chunk contains our symbol. If it doesn't, break.
            if (chunk_is_collapsed(cur_bitmap_pos)) {
                node_pos_t child_symbol_bits =
                    read_u64_from_bitmap_pos(cur_bitmap_pos + 1, CHUNK_WIDTH_SHIFT);

                debugf("[traverse_hier_until_missing] Collapsed\n");

                if (child_symbol_bits < target_symbol_bits) {
                    // Chunk doesn't encode our symbol - need to insert after the current child.
                    // The children of this parent lie _before_ our new insertion point.
                    num_subtrees_before_child = 1;
                    num_subtrees_after_child[cur_hier_level] = 0;
                    debugf("[traverse_hier_until_missing] Child missing (collapsed < target), "
                           "returning false\n");
                    debugf("[traverse_hier_until_missing] SET num_subtrees_after_child[%zu] = 0\n",
                           cur_hier_level);
                    return false;
                } else if (child_symbol_bits > target_symbol_bits) {
                    // Chunk doesn't encode our symbol - need to insert here.
                    // The children of this parent lie _after_ our new insertion point.
                    num_subtrees_before_child = 0;
                    num_subtrees_after_child[cur_hier_level] = 1;
                    debugf("[traverse_hier_until_missing] Child missing (collapsed > target), "
                           "returning false\n");
                    debugf("[traverse_hier_until_missing] SET num_subtrees_after_child[%zu] = 1\n",
                           cur_hier_level);
                    return false;
                }

                // Symbol _does_ encode our child. Recurse and continue to next level.
                assert(num_subtrees_before_child == 0);
                assert(num_subtrees_after_child[cur_hier_level] == 0);

                // Advance to the next level.
                cur_hier_level++;
                cur_bitmap_pos += CHUNK_WIDTH_SHIFT + 1;
                debugf("[traverse_hier_until_missing] Matched, advancing to level %zu at pos %lu\n",
                       cur_hier_level, cur_bitmap_pos);
                continue;
            } else {
                // Chunk is uncollapsed.

                // Count the children before and after our "target" child.
                auto [before, after] = count_uncollapsed_chunk_children_around_target(
                    cur_bitmap_pos, target_symbol_bits);
                num_subtrees_before_child = before;
                num_subtrees_after_child[cur_hier_level] = after;

                debugf("[traverse_hier_until_missing] Uncollapsed: before=%lu, after=%lu\n", before,
                       after);
                debugf("[traverse_hier_until_missing] SET num_subtrees_after_child[%zu] = %lu\n",
                       cur_hier_level, after);

                if (!uncollapsed_chunk_contains_child(cur_bitmap_pos, target_symbol_bits)) {
                    debugf("[traverse_hier_until_missing] Child missing (bit not set), returning "
                           "false\n");
                    return false;
                }

                // Advance to the next symbol in the trie.
                cur_hier_level++;
                cur_bitmap_pos += CHUNK_WIDTH + 1; // width + "collapsed bit"
                debugf("[traverse_hier_until_missing] Bit set, advancing to level %zu at pos %lu\n",
                       cur_hier_level, cur_bitmap_pos);
                debugf("[traverse_hier_until_missing] After advancing, "
                       "num_subtrees_after_child[%zu] was left at %lu\n",
                       cur_hier_level - 1, num_subtrees_after_child[cur_hier_level - 1]);
            }

            // PHASE 2.1 AGAIN: (for the loop): Count the children in all subtrees BEFORE the one
            // our target resides in.

            while (num_subtrees_before_child > 0) {
                auto [new_bitmap_pos, child_count] =
                    count_hier_subtree(cur_bitmap_pos, cur_hier_level);
                cur_bitmap_pos = new_bitmap_pos;
                num_nodes_before_child += child_count;

                num_subtrees_before_child--;
            }
        }

        // PHASE 3: Reach the lowest level chunk!
        // Check if our child is present (and how many children lie before it).
        assert(num_subtrees_before_child == 0);
        assert(cur_hier_level == H_LEVEL);

        debugf("[traverse_hier_until_missing] PHASE 3: Lowest level chunk (%lu)\n", cur_bitmap_pos);

        target_symbol_bits = get_hier_level_bits(child_sym, H_LEVEL);

        debugf("[traverse_hier_until_missing] Lowest level target_symbol_bits=%lu\n",
               target_symbol_bits);

        if (chunk_is_collapsed(cur_bitmap_pos)) {
            // Check whether this collapsed chunk contains our symbol.
            node_pos_t child_symbol_bits =
                read_u64_from_bitmap_pos(cur_bitmap_pos + 1, CHUNK_WIDTH_SHIFT);

            debugf("[traverse_hier_until_missing] Lowest level collapsed\n");

            // Update `num_nodes_before_child`
            if (child_symbol_bits < target_symbol_bits) {
                num_nodes_before_child++; // One child present, before our target child
            }

            bool child_already_set = child_symbol_bits == target_symbol_bits;
            debugf("[traverse_hier_until_missing] Returning %s (child_already_set=%d)\n",
                   child_already_set ? "true" : "false", child_already_set);
            return child_already_set;
        }

        // Uncollapsed node case: count the children less than our position and exit.
        num_nodes_before_child += popcount(cur_bitmap_pos + 1, target_symbol_bits);
        bool child_already_set =
            uncollapsed_chunk_contains_child(cur_bitmap_pos, target_symbol_bits);

        debugf("[traverse_hier_until_missing] Lowest level uncollapsed: child_already_set=%d, "
               "num_nodes_before_child=%lu\n",
               child_already_set, num_nodes_before_child);
        debugf("[traverse_hier_until_missing] Returning %s\n",
               child_already_set ? "true" : "false");
        return child_already_set;
    }

    /// Helper function for `set_child_in_hier_node`. Finds the insertion position by skipping
    /// past irrelevant child subtrees.
    ///
    /// This implements step 1 of the algorithm described in `set_child_in_hier_node`:
    /// Skip past the irrelevant child subtrees (counted in `num_subtrees_before_child`)
    /// and return the position of the new insertion.
    ///
    /// @param parent_chunk_bitmap_pos: Position of the parent chunk that will get the new child.
    /// @param parent_chunk_hier_level: Hierarchical level of the parent chunk.
    /// @param num_subtrees_before_child: Number of child subtrees to skip past.
    /// @return Pair of (insertion_position, num_children_in_irrelevant_subtrees).
    ///         The insertion position is where the new child subtree should be inserted.
    ///         The child count is the total number of leaf nodes in all skipped subtrees.
    inline std::pair<node_bitmap_pos_t, node_pos_t>
    set_child_in_hier_node_find_insertion_position(node_bitmap_pos_t parent_chunk_bitmap_pos,
                                                   size_t parent_chunk_hier_level,
                                                   node_pos_t &num_subtrees_before_child) const
    {
        debugf("[set_child_in_hier_node_find_insertion_position] Entering at bitmap_pos=%lu, "
               "hier_level=%zu, num_subtrees_before_child=%lu\n",
               parent_chunk_bitmap_pos, parent_chunk_hier_level, num_subtrees_before_child);

        // Skip past the parent chunk itself to get to its first child subtree.
        // We assume the chunk will be uncollapsed later.
        node_bitmap_pos_t parent_chunk_width =
            get_chunk_width(parent_chunk_bitmap_pos, parent_chunk_hier_level);

        // Start position: right after the parent chunk
        node_bitmap_pos_t cur_bitmap_pos = parent_chunk_bitmap_pos + parent_chunk_width;

        debugf("[set_child_in_hier_node_find_insertion_position] Parent chunk "
               "width=%lu, first child at pos=%lu\n",
               parent_chunk_width, cur_bitmap_pos);

        // Find the length of the irrelevant subtrees. Count children as we go.

        const size_t child_subtree_level = parent_chunk_hier_level + 1;
        node_pos_t total_children_in_irrelevant_subtrees = 0;

        while (num_subtrees_before_child > 0) {
            debugf("[set_child_in_hier_node_find_insertion_position] Skipping subtree at pos=%lu "
                   "(level=%zu), %lu remaining\n",
                   cur_bitmap_pos, child_subtree_level, num_subtrees_before_child);

            auto [new_bitmap_pos, child_count] =
                count_hier_subtree(cur_bitmap_pos, child_subtree_level);
            cur_bitmap_pos = new_bitmap_pos;
            total_children_in_irrelevant_subtrees += child_count;
            num_subtrees_before_child--;
        }
        const node_bitmap_pos_t irrelevant_subtrees_end = cur_bitmap_pos;

        debugf("[set_child_in_hier_node_find_insertion_position] Found insertion position: %lu, "
               "counted %lu children in irrelevant subtrees\n",
               irrelevant_subtrees_end, total_children_in_irrelevant_subtrees);

        return {irrelevant_subtrees_end, total_children_in_irrelevant_subtrees};
    }

    /// Helper function for `set_child_node_hier`. Calculates how many extra bits
    /// we need to allocate to insert a new child symbol, and performs the allocation.
    ///
    /// Before calling this function:
    /// `parent_chunk, later_chunks, node_1, node_2...`
    ///
    /// After calling this function:
    /// `parent_chunk, free space, later_chunks, node_1, node_2`.
    /// ie: the new data is inserted _after_ the parent chunk.
    ///
    /// @param cur_bitmap_pos: The bitmap position of the chunk where we need to insert.
    /// @param cur_hier_level: The hierarchical level of the chunk being pointed at (0 = top level).
    /// @return The number of bits allocated for the new child's chunks (excludes
    ///         extra_bits_to_allocate), AND whether we'll need to uncollapse
    ///         the parent chunk when inserting.
    inline std::pair<node_pos_t, bool>
    set_child_in_hier_node_calc_alloc_size(node_bitmap_pos_t cur_bitmap_pos, size_t cur_hier_level)
    {
        debugf("[set_child_in_hier_node_calc_alloc_size] Starting at bitmap_pos=%lu, "
               "hier_level=%zu\n",
               cur_bitmap_pos, cur_hier_level);

        // PHASE 1: CALCULATE ALLOCATION SIZE (new bits needed after insert)

        node_pos_t alloc_size = 0;
        bool need_to_uncollapse_parent_chunk;

        // 1. If the current chunk is collapsed, we need to uncollapse it.
        //    This adds (CHUNK_WIDTH - CHUNK_WIDTH_SHIFT) bits for inner chunks,
        //    or (TOP_LEVEL_CHUNK_WIDTH - TOP_LEVEL_CHUNK_WIDTH_SHIFT) for top level.
        if (chunk_is_collapsed(cur_bitmap_pos)) {
            need_to_uncollapse_parent_chunk = true;

            if (cur_hier_level == 0) {
                alloc_size += TOP_LEVEL_CHUNK_WIDTH - TOP_LEVEL_CHUNK_WIDTH_SHIFT;
                debugf("[set_child_in_hier_node_calc_alloc_size] Uncollapsing top level chunk: "
                       "adding %lu "
                       "bits\n",
                       (unsigned long)(TOP_LEVEL_CHUNK_WIDTH - TOP_LEVEL_CHUNK_WIDTH_SHIFT));
            } else {
                alloc_size += CHUNK_WIDTH - COLLAPSED_CHUNK_WIDTH;
                debugf(
                    "[set_child_in_hier_node_calc_alloc_size] Uncollapsing inner chunk: adding %lu "
                    "bits\n",
                    (unsigned long)(CHUNK_WIDTH - COLLAPSED_CHUNK_WIDTH));
            }
        } else {
            // uncollapsed chunks stay uncollapsed => no change in size.
            need_to_uncollapse_parent_chunk = false;
            debugf("[set_child_in_hier_node_calc_alloc_size] Chunk already uncollapsed, no size "
                   "change\n");
        }

        // 2. Allocate space for the remaining collapsed chunks we need to insert.
        //    Starting from (cur_hier_level + 1) down to H_LEVEL.
        //    Each collapsed chunk takes (1 + CHUNK_WIDTH_SHIFT) bits.
        size_t num_remaining_levels = H_LEVEL - cur_hier_level;
        alloc_size += num_remaining_levels * (1 + COLLAPSED_CHUNK_WIDTH);
        debugf(
            "[set_child_in_hier_node_calc_alloc_size] Allocating %zu remaining levels: adding %lu "
            "bits\n",
            num_remaining_levels, (unsigned long)(num_remaining_levels * (1 + CHUNK_WIDTH_SHIFT)));

        debugf("[set_child_in_hier_node_calc_alloc_size] final allocation size: %lu\n", alloc_size);
        return {alloc_size, need_to_uncollapse_parent_chunk};
    }

    /// Helper function to uncollapse a collapsed chunk.
    ///
    /// Converts a collapsed chunk (1 bit flag + collapsed_width bits for single child index)
    /// into an uncollapsed chunk (1 bit flag + full chunk_width bits with one bit set).
    ///
    /// Assumes the bitmap space for the larger "uncollapsed" chunk is allocated (but not
    /// necessarily zeroed).
    ///
    /// @param chunk_bitmap_pos: Position of the chunk to uncollapse
    /// @param hier_level: Hierarchical level (0 = top level, affects chunk width)
    inline void uncollapse_chunk(node_bitmap_pos_t chunk_bitmap_pos, size_t hier_level)
    {
        assert(chunk_is_collapsed(chunk_bitmap_pos));

        const node_bitmap_pos_t collapsed_chunk_width =
            hier_level == 0 ? TOP_LEVEL_COLLAPSED_CHUNK_WIDTH : COLLAPSED_CHUNK_WIDTH;
        const node_bitmap_pos_t uncollapsed_chunk_width =
            hier_level == 0 ? TOP_LEVEL_CHUNK_WIDTH : CHUNK_WIDTH;

        // Read the existing collapsed value (the index of the single set bit)
        chunk_val_t existing_child =
            read_u64_from_bitmap_pos(chunk_bitmap_pos + 1, collapsed_chunk_width);

        debugf("[uncollapse_chunk] Uncollapsing chunk at pos=%lu, level=%zu, existing_child=%u\n",
               chunk_bitmap_pos, hier_level, (unsigned int)existing_child);

        clear_bitmap_pos(chunk_bitmap_pos + 1, uncollapsed_chunk_width);
        set_chunk_as_uncollapsed(chunk_bitmap_pos);
        set_child_in_uncollapsed_chunk(chunk_bitmap_pos, existing_child);

        debugf("[uncollapse_chunk] Set existing child bit at position %lu\n",
               chunk_bitmap_pos + 1 + existing_child);
    }

    /// Helper function for `set_child_node_hier`. Writes the new child chunks
    /// into allocated (and zeroed) space.
    ///
    /// @param cur_bitmap_pos: The bitmap position of the chunk where we're inserting.
    /// @param cur_hier_level: The hierarchical level of that chunk (0 = top level).
    /// @param child_sym: The morton-encoded symbol for the child being inserted.
    /// @return the bitmap position immediately after the insert.
    inline node_bitmap_pos_t set_child_in_hier_node_write_chunks(node_bitmap_pos_t cur_bitmap_pos,
                                                                 size_t cur_hier_level,
                                                                 const morton_type &child_sym)
    {
        debugf("[set_child_in_hier_node_write_chunks] Starting at bitmap_pos=%lu, hier_level=%zu\n",
               cur_bitmap_pos, cur_hier_level);

        while (cur_hier_level <= H_LEVEL) {
            // Get the relevant bits for this level
            chunk_val_t level_bits = get_hier_level_bits(child_sym, cur_hier_level);

            debugf("[set_child_in_hier_node_write_chunks] Level %zu: writing collapsed chunk with "
                   "level_bits=%u at pos %lu\n",
                   cur_hier_level, (unsigned int)level_bits, cur_bitmap_pos);

            // Write collapsed chunk: flag bit is 0 (already cleared), just write the index
            assert(chunk_is_collapsed(cur_bitmap_pos));
            write_to_bitmap_pos(cur_bitmap_pos + 1, level_bits, CHUNK_WIDTH_SHIFT);

            // Advance to the next chunk
            cur_bitmap_pos += 1 + CHUNK_WIDTH_SHIFT;
            cur_hier_level++;
        }

        debugf("[set_child_in_hier_node_write_chunks] DONE: final pos=%lu\n", cur_bitmap_pos);
        return cur_bitmap_pos; // just past the end of the insert
    }

    /// A helper function for set_child_node(). Inserts a child symbol into a hierarchical node.
    ///
    /// Assumes the node there is already in hier format.
    ///
    /// TODO(yash): Make the child-counting toggleable. We don't need to count children at the
    /// lowest level. This will reduce popcount time for the lowest chunk level.
    /// A helper function for set_child_node(). Inserts a child symbol into a hierarchical node.
    ///
    /// @param child_sym: The morton-encoded symbol for the child being inserted. Contains
    ///                   DIMENSION bits that define the child's path through the hierarchy.
    /// @param num_nodes_before_child: [OUTPUT] Number of sibling child nodes that come before
    ///                                 the newly inserted child in the hierarchical ordering.
    ///                                 Used by caller to determine the child's index.
    ///                                 This is useful when inserting: suppose we insert a child
    ///                                 into the parent node. Now, we must seek to the next level
    ///                                 of the MDTrie, to continue insertion! But this requires
    ///                                 counting how many child subtrees to skip. => need to count
    ///                                 the number of children that come before our child node.
    /// @param total_num_bits_after_operation: [OUTPUT] Total width (in bits) of the parent node's
    ///                                         representation after insertion. Includes all
    ///                                         hierarchical encoding overhead.
    /// @param extra_bits_to_allocate: Additional data bits to pre-allocate beyond what's needed
    ///                                for this insertion (for combining with extend_treeblock).
    ///                                Default is 0. These bits are allocated at the end of the
    ///                                treeblock. This saves one call to `realloc()`.
    ///                                IS THERE ANY EVIDENCE THAT REALLOC WAS A PROBLEM TO BEGIN
    ///                                WITH?
    /// @return true if child was inserted, false if child_sym already existed in the tree.
    bool set_child_in_hier_node(node_bitmap_pos_t parent_bitmap_pos, const morton_type &child_sym,
                                node_pos_t &num_nodes_before_child,
                                node_pos_t &parent_symbol_width_after_insert,
                                width_type extra_bits_to_allocate = 0)
    {
        // Overall algorithm:
        //
        // 1. Iterate through the DFS tree. At each level, count the subtrees before the target
        //    child. Continue until we hit a trie level that DOESN'T encode our child node.
        // 2. Allocate enough space to encode a new chunk subtree corresponding to our new child.
        // 3. Write the new child to the empty space.

        // initialize output vars (we update them throughout the operation)
        num_nodes_before_child = 0;
        parent_symbol_width_after_insert = 0;

        // PHASE 1: ITERATE THROUGH THE TREE AS FAR AS WE CAN (FOLLOW EXISTING NODES!)

        size_t parent_chunk_hier_level;
        // The number of current-level subtrees to skip before we find the desired child node.
        node_pos_t num_subtrees_before_target = 0;
        // num_children_after_target[parent_level] := # children _after_ the target subtree.
        node_pos_t num_subtrees_after_target[H_LEVEL] = {};
        node_bitmap_pos_t cur_bitmap_pos;

        // Traverse the hierarchical tree to find where to insert (or if child already exists)
        bool child_already_set = traverse_hier_until_missing(
            parent_bitmap_pos, child_sym, num_nodes_before_child, num_subtrees_before_target,
            num_subtrees_after_target, cur_bitmap_pos, parent_chunk_hier_level);

        if (!child_already_set) {
            // PHASE 2: ALLOCATE ENOUGH FREE SPACE FOR OUR NEWLY INSERTED CHILD'S CHUNKS!

            // End goal: bitmap should look like:
            // (...) (parent chunk), (irrelevant child subtrees), (new child subtree), (...)
            //
            // Current state:
            // (...) (parent chunk), (irrelevant child subtrees), (...)
            //
            // Algorithm:
            // 1. Allocate new memory _after_ the irrelevant child subtrees.
            // 2. _if_ the parent chunk needs to be expanded, shift the irrelevant child subtree
            //    data to the right as needed.
            // 3. Write the new child subtree after our irrelevant child subtree.

            const node_bitmap_pos_t parent_chunk_bitmap_pos = cur_bitmap_pos;

            // PHASE 2.1.1: Find out how much space we need to allocate, and where to allocate it.

            auto [alloc_size, need_to_uncollapse_parent_chunk] =
                set_child_in_hier_node_calc_alloc_size(parent_chunk_bitmap_pos,
                                                       parent_chunk_hier_level);

            auto [alloc_bitmap_pos, children_in_irrelevant_subtrees] =
                set_child_in_hier_node_find_insertion_position(
                    parent_chunk_bitmap_pos, parent_chunk_hier_level, num_subtrees_before_target);
            num_nodes_before_child += children_in_irrelevant_subtrees;

            // PHASE 2.1.3: Allocate new memory for our new chunks! (and for expansion of old
            // chunks)

            shift_backward_data_only(alloc_bitmap_pos, alloc_size, extra_bits_to_allocate);
            clear_bitmap_pos(alloc_bitmap_pos, alloc_size);

            // PHASE 2.2: Shift irrelevant subtrees if parent chunk needs expansion
            //
            // If the parent chunk is collapsed and needs to be uncollapsed, it will expand from
            // collapsed to uncollapsed format. We need to shift the irrelevant child subtrees to
            // the right to make room for this expansion.
            if (need_to_uncollapse_parent_chunk) {
                node_bitmap_pos_t parent_chunk_width = (parent_chunk_hier_level == 0)
                                                           ? 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH
                                                           : 1 + COLLAPSED_CHUNK_WIDTH;
                node_bitmap_pos_t parent_expansion_size =
                    (parent_chunk_hier_level == 0)
                        ? TOP_LEVEL_CHUNK_WIDTH - TOP_LEVEL_COLLAPSED_CHUNK_WIDTH
                        : CHUNK_WIDTH - COLLAPSED_CHUNK_WIDTH;

                node_bitmap_pos_t irrelevant_subtrees_start =
                    parent_chunk_bitmap_pos + parent_chunk_width;
                node_bitmap_pos_t irrelevant_subtrees_size =
                    alloc_bitmap_pos - irrelevant_subtrees_start;

                // Shift irrelevant subtrees to the right by parent_expansion_size

                // For some reason, this function doesn't copy from [start, start+size) to [end,
                // end+size). It actually copies from `[start - size, start)` to `[end - size,
                // end)`...what a footgun. This is equivalent to:
                //     memcpy(irrelevant_subtrees_start,
                //              irrelevant_subtrees_start + parent_expansion_size,
                //              irrelevant_subtrees_size);
                bulkcopy_backward(irrelevant_subtrees_start + irrelevant_subtrees_size,
                                  (irrelevant_subtrees_start + parent_expansion_size) +
                                      irrelevant_subtrees_size,
                                  irrelevant_subtrees_size);

                // The "empty" bitmap space is now shifted to the right.
                alloc_bitmap_pos += parent_expansion_size;

                uncollapse_chunk(parent_chunk_bitmap_pos, parent_chunk_hier_level);
            }

            // PHASE 3: WRITE THE DATA FOR THE NEW CHILD SUBTREE!

            // Set our child subtree in the parent chunk.
            chunk_val_t child_bits = (parent_chunk_hier_level == 0)
                                         ? get_hier_top_level_bits(child_sym)
                                         : get_hier_level_bits(child_sym, parent_chunk_hier_level);
            set_child_in_uncollapsed_chunk(parent_chunk_bitmap_pos, child_bits);

            // Write the child subtree data to the allocated space.
            cur_bitmap_pos = set_child_in_hier_node_write_chunks(
                alloc_bitmap_pos, parent_chunk_hier_level + 1, child_sym);

            // TODO(yash): skip the _rest_ of the subtrees in this level so we can move on.

            // Debug statement; assert we've used all of the allocated space.
            node_bitmap_pos_t parent_chunk_expansion_size = 0;
            if (need_to_uncollapse_parent_chunk) {
                parent_chunk_expansion_size =
                    (parent_chunk_hier_level == 0)
                        ? TOP_LEVEL_CHUNK_WIDTH - TOP_LEVEL_COLLAPSED_CHUNK_WIDTH
                        : CHUNK_WIDTH - COLLAPSED_CHUNK_WIDTH;
            }
            (void)parent_chunk_expansion_size;
            assert(cur_bitmap_pos == alloc_bitmap_pos + alloc_size - parent_chunk_expansion_size);
        } else {
            // The child is already there. So skip the subtrees until we reach its position.
            // Skip prior subtrees.
            while (num_subtrees_before_target > 0) {
                assert(parent_chunk_hier_level < H_LEVEL);
                cur_bitmap_pos = skip_hier_subtree(cur_bitmap_pos, parent_chunk_hier_level + 1);
                num_subtrees_before_target--;
            }
            // Skip the already-existing child subtree too, so the cursor is one
            // after the inserted position.
            cur_bitmap_pos = skip_hier_subtree(
                cur_bitmap_pos, std::min((size_t)H_LEVEL, parent_chunk_hier_level + 1));
        }

        // PHASE 4: TRAVERSE TO THE END OF THE HIERARCHICAL SYMBOL.
        //
        // We've inserted a new child successfully. But the caller still needs
        // `parent_symbol_width_after_insert`, requiring us to traverse until the end.
        // Skip through all remaining sibling subtrees after our inserted child.
        // We traverse from parent_chunk_hier_level back down to 0, skipping sibling
        // subtrees at each level as we unwind back to the root.

        assert(num_subtrees_before_target == 0);

        // We inserted at the lowest level => start at H_LEVEL-1 parent.
        size_t cur_chunk_hier_level = H_LEVEL - 1;
        while (cur_chunk_hier_level > 0) {
            while (num_subtrees_after_target[cur_chunk_hier_level] > 0) {
                cur_bitmap_pos = skip_hier_subtree(cur_bitmap_pos, cur_chunk_hier_level + 1);
                num_subtrees_after_target[cur_chunk_hier_level]--;
            }
            cur_chunk_hier_level--;
        }
        while (num_subtrees_after_target[0] > 0) {
            cur_bitmap_pos = skip_hier_subtree(cur_bitmap_pos, 1);
            num_subtrees_after_target[0]--;
        }

        // Now cur_bitmap_pos is at the end of the symbol
        parent_symbol_width_after_insert = cur_bitmap_pos - parent_bitmap_pos;

        return !child_already_set;
    }

    /// A helper function for set_child_node(). Inserts a child symbol into a collapsed node.
    ///
    /// Assumes the node there is already in collapsed format.
    ///
    /// TODO(yash): Make the child-counting toggleable. We don't need to count children at the
    /// lowest level. This will reduce popcount time for the lowest chunk level.
    /// A helper function for set_child_node(). Inserts a child symbol into a hierarchical node.
    ///
    /// @param child_sym: The morton-encoded symbol for the child being inserted. Contains
    ///                   DIMENSION bits that define the child's path through the hierarchy.
    /// @param num_nodes_before_child: [OUTPUT] Number of sibling child nodes that come before
    ///                                 the newly inserted child in the hierarchical ordering.
    ///                                 Used by caller to determine the child's index.
    ///                                 This is useful when inserting: suppose we insert a child
    ///                                 into the parent node. Now, we must seek to the next level
    ///                                 of the MDTrie, to continue insertion! But this requires
    ///                                 counting how many child subtrees to skip. => need to count
    ///                                 the number of children that come before our child node.
    /// @param parent_sym_width_after_operation: [OUTPUT] Total width (in bits) of the parent node's
    ///                                           representation after insertion. Includes all
    ///                                          hierarchical encoding overhead.
    /// @param extra_bits_to_allocate: Additional data bits to pre-allocate beyond what's needed
    ///                                for this insertion (for combining with extend_treeblock).
    ///                                Default is 0. These bits are allocated at the end of the
    ///                                treeblock. This saves one call to `realloc()`.
    ///                                IS THERE ANY EVIDENCE THAT REALLOC WAS A PROBLEM TO BEGIN
    ///                                WITH?
    void set_child_in_collapsed_node(node_pos_t node_pos, node_bitmap_pos_t bitmap_pos,
                                     const morton_type &old_child, const morton_type &new_child,
                                     node_pos_t &num_nodes_before_symbol,
                                     node_pos_t &parent_width_after_operation,
                                     width_type extra_data_bits = 0)
    {
        // There are 2 children in this node, so the final hierarchical node will look like:
        //
        // (collapsed chunk) (collapsed chunk) (uncollapsed chunk) (collapsed subtree 1) (collapsed
        // subtree)
        //
        // We'll call the (hier_chunk) level the "divergence level", because
        // it's where the childrens' encoding paths diverge.

        const node_bitmap_pos_t original_bitmap_pos = bitmap_pos;

        // PHASE 1: CALCULATE THE DIVERGENCE LEVEL

        size_t divergence_level = 0;

        chunk_val_t old_top = get_hier_top_level_bits(old_child);
        chunk_val_t new_top = get_hier_top_level_bits(new_child);

        chunk_val_t old_divergent_bits = old_top;
        chunk_val_t new_divergent_bits = new_top;

        if (old_top == new_top) {
            [[maybe_unused]] bool divergence_found = false;

            for (size_t hier_level = 1; hier_level <= H_LEVEL; hier_level++) {
                old_divergent_bits = get_hier_level_bits(old_child, hier_level);
                new_divergent_bits = get_hier_level_bits(new_child, hier_level);

                if (old_divergent_bits != new_divergent_bits) {
                    divergence_level = hier_level;
                    divergence_found = true;
                    break;
                }
            }
            assert(divergence_found);
        }

        // PHASE 2: CALCULATE HOW MUCH MEMORY TO ALLOCATE FOR OUR NEW HIER NODE

        // Calculate final size:
        // - Levels Above Divergence: 1 collapsed chunk each (which will generally be
        //   `1 + CHUNK_WIDTH_SHIFT` bits)
        // - Divergence Level: 1 uncollapsed chunk (generally 1 + CHUNK_WIDTH)
        // - Levels below divergence: 2 collapsed chunks each (one per child path)

        width_type new_symbol_size = 0;
        if (divergence_level == 0) {
            // Top level chunk (with collapsed bit)
            new_symbol_size += 1 + TOP_LEVEL_CHUNK_WIDTH;
            // Below top level: 2 paths, each with H_LEVEL collapsed chunks
            new_symbol_size += H_LEVEL * 2 * (1 + COLLAPSED_CHUNK_WIDTH);
        } else {
            // Top level chunk
            new_symbol_size += 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH;
            // All chunks up to the divergence level are collapsed
            new_symbol_size += (1 + COLLAPSED_CHUNK_WIDTH) * (divergence_level - 1);
            // The divergence itself is just one uncollapsed chunk
            new_symbol_size += 1 + CHUNK_WIDTH;
            // And lastly come all the levels _after_ the divergence level which are collapsed.
            // (we multiply by two because there will be two child subtrees from here on).
            new_symbol_size += 2 * (H_LEVEL - divergence_level) * (1 + COLLAPSED_CHUNK_WIDTH);
        }

        // PHASE 3: ALLOCATE MEMORY FOR OUR NEW HIERARCHICAL NODE

        // Expand bitmap from DIMENSION to total_bits, plus any extra bits for extend_treeblock
        // extra_data_bits is passed separately so it's allocated but not shifted into
        const node_bitmap_pos_t old_symbol_size = DIMENSION;
        shift_backward_data_only(original_bitmap_pos, new_symbol_size - old_symbol_size,
                                 extra_data_bits);
        clear_bitmap_pos(original_bitmap_pos, new_symbol_size);

        // PHASE 4: WRITE THE NEW HIER SYMBOL DATA

        // Write the components of the hierarchical structure that are shared by both subtrees.
        if (divergence_level == 0) {
            // Set child bits in the top level (divergent) chunk.
            set_chunk_as_uncollapsed(bitmap_pos);
            set_child_in_uncollapsed_chunk(bitmap_pos, old_top);
            set_child_in_uncollapsed_chunk(bitmap_pos, new_top);

            bitmap_pos += 1 + TOP_LEVEL_CHUNK_WIDTH;
        } else {
            // Write bits in the top level chunk
            assert(chunk_is_collapsed(bitmap_pos));
            write_to_bitmap_pos(bitmap_pos + 1, old_top, TOP_LEVEL_CHUNK_WIDTH_SHIFT);
            bitmap_pos += 1 + TOP_LEVEL_CHUNK_WIDTH_SHIFT;

            // Continue to write the shared portion of the tree until we hit divergence chunk.
            for (size_t hier_level = 1; hier_level < divergence_level; hier_level++) {
                chunk_val_t level_bits = get_hier_level_bits(old_child, hier_level);
                assert(level_bits == get_hier_level_bits(new_child, hier_level));
                assert(chunk_is_collapsed(bitmap_pos));
                write_to_bitmap_pos(bitmap_pos + 1, level_bits, CHUNK_WIDTH_SHIFT);

                bitmap_pos += 1 + CHUNK_WIDTH_SHIFT;
            }

            // Set bits in the divergent chunk.
            set_chunk_as_uncollapsed(bitmap_pos);
            SETBITVAL(data_, bitmap_pos + 1 + old_divergent_bits);
            SETBITVAL(data_, bitmap_pos + 1 + new_divergent_bits);

            bitmap_pos += 1 + CHUNK_WIDTH;
        }

        // We are just past the divergence chunk. Figure out which (smaller) subtree to write first.

        bool old_first = old_divergent_bits < new_divergent_bits;
        const morton_type &first_sym = old_first ? old_child : new_child;
        const morton_type &second_sym = old_first ? new_child : old_child;
        assert(first_sym <= second_sym);

        // Write the smallest subtree.

        for (size_t hier_level = divergence_level + 1; hier_level <= H_LEVEL; hier_level++) {
            assert(chunk_is_collapsed(bitmap_pos));

            chunk_val_t level_bits = get_hier_level_bits(first_sym, hier_level);
            write_to_bitmap_pos(bitmap_pos + 1, level_bits, COLLAPSED_CHUNK_WIDTH);
            bitmap_pos += 1 + CHUNK_WIDTH_SHIFT;
        }

        // write the second (smallest) subtree
        for (size_t hier_level = divergence_level + 1; hier_level <= H_LEVEL; hier_level++) {
            chunk_val_t level_bits = get_hier_level_bits(second_sym, hier_level);
            assert(chunk_is_collapsed(bitmap_pos));
            write_to_bitmap_pos(bitmap_pos + 1, level_bits, CHUNK_WIDTH_SHIFT);

            bitmap_pos += 1 + CHUNK_WIDTH_SHIFT;
        }

        // We should have exactly used the allocated space.
        assert(bitmap_pos == original_bitmap_pos + new_symbol_size);

        // Update output parameters before returning
        num_nodes_before_symbol = (old_first) ? 1 : 0;
        parent_width_after_operation = new_symbol_size;

        set_node_as_uncollapsed(node_pos);

        return;
    }

    morton_type get_next_morton_in_range(node_bitmap_pos_t (&levels_start)[H_LEVEL + 1],
                                         const morton_type &parent_morton, trie_level_t level,
                                         const morton_type &start_symbol,
                                         const morton_type &end_symbol, bool &found) const
    {
        // TODO(yash): Convert range search to DFS representation. This fn isn't needed.
        (void)levels_start;
        (void)parent_morton;
        (void)level;
        (void)start_symbol;
        (void)end_symbol;
        (void)found;
        assert(false);
        return 0;
    }

    /// recursive helper for next_symbol
    /// @param found: set when the smallest morton that satisifes the specified
    ///                 function terminates early when found is set
    /// @return     : (found = true, next_symbol)     actual next symbol
    ///               (found = false, 0)              continue the search,
    ///                                               currently smaller than the
    ///                                               start range
    ///               (found = false, end_symbol+1)   out of range, return false
    ///                                               and early terminate
    morton_type get_next_morton_in_range_reuse(node_bitmap_pos_t (&levels_start)[H_LEVEL + 1],
                                               const morton_type &parent_morton, trie_level_t level,
                                               const morton_type &start_symbol,
                                               const morton_type &end_symbol, bool &found,
                                               node_pos_t (&bit_bases)[H_LEVEL + 1],
                                               uint64_t (&words)[H_LEVEL + 1], bool &advance,
                                               node_pos_t &skip_count) const
    {
        // TODO(yash): Convert range search to DFS representation. This fn isn't needed.
        (void)levels_start;
        (void)parent_morton;
        (void)level;
        (void)start_symbol;
        (void)end_symbol;
        (void)found;
        (void)bit_bases;
        (void)words;
        (void)advance;
        (void)skip_count;
        assert(false);
        return morton_type::null();
    }

private:
    /// Parse a single chunk at the given position in the hierarchical encoding.
    ///
    /// @param bitmap_pos The starting position of the chunk in the bitmap
    /// @param hier_level The hierarchical level of the chunk (default: H_LEVEL)
    ///                   When hier_level is 0, this delegates to count_toplevel_chunk_children
    /// @return Pair of (num_children, pos_after_chunk)
    ///         - num_children: number of set bits in this chunk
    ///         - width: the width of this chunk (in bits)
    inline std::pair<node_pos_t, node_bitmap_pos_t>
    count_chunk_children(node_bitmap_pos_t bitmap_pos, size_t hier_level = H_LEVEL) const
    {
        if (hier_level == 0) {
            return count_toplevel_chunk_children(bitmap_pos);
        }

        if (chunk_is_collapsed(bitmap_pos)) {
            // Collapsed: 1 bit (flag) + CHUNK_WIDTH_shift bits (index of single set bit)
            node_pos_t num_children = 1;
            return {num_children, 1 + COLLAPSED_CHUNK_WIDTH};
        }
        // Uncollapsed: 1 bit (flag) + CHUNK_WIDTH bits (data)
        node_pos_t num_children = popcount(bitmap_pos + 1, CHUNK_WIDTH, true);
        return {num_children, 1 + CHUNK_WIDTH};
    }

    /// Return the width of the hier chunk at the given position.
    ///
    /// Don't give it a "top-level" chunk, which requires special handling!
    ///
    /// @param bitmap_pos The starting position of the chunk in the bitmap
    /// @return width of the chunk in bits
    inline node_bitmap_pos_t get_chunk_width(node_bitmap_pos_t bitmap_pos,
                                             size_t hier_level = H_LEVEL) const
    {
        if (hier_level == 0) {
            if (chunk_is_collapsed(bitmap_pos))
                return 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH;
            else
                return 1 + TOP_LEVEL_CHUNK_WIDTH;
        } else {
            if (chunk_is_collapsed(bitmap_pos))
                return 1 + COLLAPSED_CHUNK_WIDTH;
            else
                return 1 + CHUNK_WIDTH;
        }
    }

public:
    /// Parse a single top-level chunk at the given position in the hierarchical encoding.
    ///
    /// @param bitmap_pos The starting position of the chunk in the bitmap
    /// @return Pair of (num_children, width of chunk)
    ///         - num_children: number of set bits in this chunk
    ///         - width: the width of this chunk (in bits)
    inline std::pair<node_pos_t, node_bitmap_pos_t>
    count_toplevel_chunk_children(node_bitmap_pos_t bitmap_pos) const
    {
        bool chunk_is_uncollapsed = GETBITVAL(data_, bitmap_pos);

        if (chunk_is_uncollapsed) {
            // Uncollapsed: 1 bit (flag) + CHUNK_WIDTH bits (data)
            node_pos_t num_children = popcount(bitmap_pos + 1, TOP_LEVEL_CHUNK_WIDTH);
            return {num_children, 1 + TOP_LEVEL_CHUNK_WIDTH};
        }

        // Collapsed: 1 bit (flag) + CHUNK_WIDTH_shift bits (index of single set bit)
        node_pos_t num_children = 1;
        return {num_children, 1 + TOP_LEVEL_CHUNK_WIDTH_SHIFT};
    }

    inline bool chunk_contains_child(node_bitmap_pos_t bitmap_pos, chunk_val_t child_bits,
                                     size_t hier_level = H_LEVEL) const noexcept
    {
        if (hier_level == 0)
            return toplevel_chunk_contains_child(bitmap_pos, child_bits);

        if (chunk_is_collapsed(bitmap_pos))
            return collapsed_chunk_contains_child(bitmap_pos, child_bits);
        return uncollapsed_chunk_contains_child(bitmap_pos, child_bits);
    }

    inline bool toplevel_chunk_contains_child(node_bitmap_pos_t bitmap_pos,
                                              chunk_val_t child_bits) const noexcept
    {
        if (chunk_is_collapsed(bitmap_pos))
            return collapsed_toplevel_chunk_contains_child(bitmap_pos, child_bits);
        return uncollapsed_chunk_contains_child(bitmap_pos, child_bits);
    }

    inline chunk_val_t get_child_from_collapsed_chunk(node_bitmap_pos_t bitmap_pos,
                                                      bool top_level = false) const noexcept
    {
        if (top_level)
            return read_u64_from_bitmap_pos(bitmap_pos + 1, TOP_LEVEL_COLLAPSED_CHUNK_WIDTH);
        return read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
    }

    inline bool collapsed_chunk_contains_child(node_bitmap_pos_t bitmap_pos,
                                               chunk_val_t child_bits) const noexcept
    {
        chunk_val_t collapsed_value =
            read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
        return collapsed_value == child_bits;
    }

    inline bool collapsed_toplevel_chunk_contains_child(node_bitmap_pos_t bitmap_pos,
                                                        chunk_val_t child_bits) const noexcept
    {
        chunk_val_t collapsed_value =
            read_u64_from_bitmap_pos(bitmap_pos + 1, TOP_LEVEL_COLLAPSED_CHUNK_WIDTH);
        return collapsed_value == child_bits;
    }

    /// Given an uncollapsed chunk's position, returns true if the chunk
    /// encodes the child given by `child_bits()`.
    ///
    /// Applicable to top level _and_ normal chunks.
    inline bool uncollapsed_chunk_contains_child(node_bitmap_pos_t bitmap_pos,
                                                 chunk_val_t child_bits) const noexcept
    {
        // skip the collapsed bit => add 1
        assert(!chunk_is_collapsed(bitmap_pos));
        return GETBITVAL(data_, bitmap_pos + 1 + child_bits);
    }

    /// Given an uncollapsed chunk's position, sets the child given by `child_bits()`.
    ///
    /// Applicable to top level _and_ normal chunks.
    inline void set_child_in_uncollapsed_chunk(node_bitmap_pos_t bitmap_pos,
                                               chunk_val_t child_bits) const noexcept
    {
        // skip the collapsed bit => add 1
        assert(!chunk_is_collapsed(bitmap_pos));
        SETBITVAL(data_, bitmap_pos + 1 + child_bits);
    }

    /// Count children before and after a target index within an chunk.
    ///
    /// @param chunk_data_pos: Bitmap position of the chunk
    /// @param chunk_width: Width of the chunk (e.g., TOP_LEVEL_CHUNK_WIDTH or CHUNK_WIDTH).
    /// @param target: The target index within the chunk.
    /// @return (children_before_target, children_after_target)
    inline std::pair<node_pos_t, node_pos_t>
    count_toplevel_chunk_children_around_target(node_bitmap_pos_t chunk_data_pos,
                                                chunk_val_t target) const
    {
        if (!chunk_is_collapsed(chunk_data_pos)) {
            return count_uncollapsed_toplevel_chunk_children_around_target(chunk_data_pos, target);
        }

        chunk_val_t current_bits = get_child_from_collapsed_chunk(chunk_data_pos, true);
        chunk_val_t children_before_target = 0;
        chunk_val_t children_after_target = 0;
        if (current_bits < target) {
            children_before_target = 1;
            children_after_target = 0;
        } else if (current_bits > target) {
            children_before_target = 1;
            children_after_target = 0;
        }
        return {children_before_target, children_after_target};
    }

    /// Count children before and after a target index within an chunk.
    ///
    /// @param chunk_data_pos: Bitmap position of the chunk
    /// @param target: The target index within the chunk.
    /// @param hier_level: The hierarchical level of the chunk (default: H_LEVEL)
    ///                    When hier_level is 0, this delegates to
    ///                    count_toplevel_chunk_children_around_target
    /// @return (children_before_target, children_after_target)
    inline std::pair<node_pos_t, node_pos_t>
    count_chunk_children_around_target(node_bitmap_pos_t chunk_data_pos, chunk_val_t target,
                                       size_t hier_level = H_LEVEL) const
    {
        if (hier_level == 0) {
            return count_toplevel_chunk_children_around_target(chunk_data_pos, target);
        }

        if (!chunk_is_collapsed(chunk_data_pos)) {
            return count_uncollapsed_chunk_children_around_target(chunk_data_pos, target);
        }
        chunk_val_t current_bits = get_child_from_collapsed_chunk(chunk_data_pos);
        chunk_val_t children_before_target = 0;
        chunk_val_t children_after_target = 0;
        if (current_bits < target) {
            children_before_target = 1;
            children_after_target = 0;
        } else if (current_bits > target) {
            children_before_target = 1;
            children_after_target = 0;
        }
        return {children_before_target, children_after_target};
    }

    /// Count children before and after a target index within an uncollapsed chunk.
    ///
    /// Don't pass a top-level chunk here.
    ///
    /// @param chunk_data_pos: Bitmap position of the chunk
    /// @param chunk_width: Width of the chunk (e.g., TOP_LEVEL_CHUNK_WIDTH or CHUNK_WIDTH).
    /// @param target: The target index within the chunk.
    /// @return (children_before_target, children_after_target)
    inline std::pair<node_pos_t, node_pos_t>
    count_uncollapsed_chunk_children_around_target(node_bitmap_pos_t chunk_data_pos,
                                                   chunk_val_t target) const
    {
        assert(!chunk_is_collapsed(chunk_data_pos));
        node_pos_t children_before = popcount(chunk_data_pos + 1, target, true);
        node_pos_t children_after =
            popcount(chunk_data_pos + 1 + target + 1, CHUNK_WIDTH - (target + 1), true);
        return {children_before, children_after};
    }

public:
    /// Count children before and after a target index within an uncollapsed chunk.
    ///
    /// Don't pass a top-level chunk here.
    ///
    /// @param chunk_data_pos: Bitmap position of the chunk
    /// @param chunk_width: Width of the chunk (e.g., TOP_LEVEL_CHUNK_WIDTH or CHUNK_WIDTH).
    /// @param target: The target index within the chunk.
    /// @return (children_before_target, children_after_target)
    inline std::pair<node_pos_t, node_pos_t>
    count_uncollapsed_toplevel_chunk_children_around_target(node_bitmap_pos_t chunk_data_pos,
                                                            chunk_val_t target) const
    {
        assert(!chunk_is_collapsed(chunk_data_pos));
        chunk_data_pos += 1; // (skip the collapsed bit)
        node_pos_t children_before = popcount(chunk_data_pos, target, true);
        node_pos_t children_after =
            popcount(chunk_data_pos + target + 1, TOP_LEVEL_CHUNK_WIDTH - (target + 1), true);
        return {children_before, children_after};
    }

    // Private helper for increase_bitmap_size and increase_flagmap_size.
    inline void grow_array(width_type increase_width, size_type &size_field,
                           data_type *&bitmap_field)
    {
        size_type old_size = size_field;
        size_field += increase_width;
        size_t old_blocks = BITS2BLOCKS(old_size);
        size_t new_blocks = BITS2BLOCKS(size_field);
        if (bitmap_field == nullptr) {
            bitmap_field = (data_type *)calloc(1, new_blocks * sizeof(data_type));
            if (bitmap_field == nullptr) {
                std::cerr << "Memory allocation failed in " << __func__ << std::endl;
                exit(EXIT_FAILURE);
                return;
            }
        }

        bitmap_field = (data_type *)realloc(bitmap_field, new_blocks * sizeof(data_type));
        if (bitmap_field == nullptr) {
            std::cerr << "Memory allocation failed in " << __func__ << std::endl;
            exit(EXIT_FAILURE);
            return;
        }

        // Zero new region if blocks increased
        if (new_blocks > old_blocks) {
            std::memset(bitmap_field + old_blocks, 0,
                        (new_blocks - old_blocks) * sizeof(data_type));
        }
    }

    // simply trimming the size of the bitmap, nothing else
    inline void shrink_array(width_type decrease_width, bool is_on_data)
    {
        // assertions
        assert(data_size_ > 0 && flag_size_ > 0 && data_ != nullptr && flag_ != nullptr);
        if (is_on_data) {
            assert(data_size_ >= decrease_width);
        } else {
            assert(flag_size_ >= decrease_width);
        }

        if (is_on_data) {
            data_size_ -= decrease_width;
            data_ = (data_type *)realloc(data_, BITS2BLOCKS(data_size_) * sizeof(data_type));
            if (data_ == nullptr) {
                std::cerr << "Memory allocation failed in "
                             "compressed_bitmap::decrease_bits"
                          << std::endl;
                exit(EXIT_FAILURE);
                return;
            }
        } else {
            flag_size_ -= decrease_width;
            flag_ = (data_type *)realloc(flag_, BITS2BLOCKS(flag_size_) * sizeof(data_type));
            if (flag_ == nullptr) {
                std::cerr << "Memory allocation failed in "
                             "compressed_bitmap::decrease_bits"
                          << std::endl;
                exit(EXIT_FAILURE);
                return;
            }
        }
    }

    // create 0-ed out holes at position (pos, pos + width) on either data
    // or flag
    inline void clear_array(node_bitmap_pos_t pos, width_type width, bool is_on_data)
    {
        if (width == 0)
            return;

        if (is_on_data) {
            assert(pos + width <= data_size_);
        } else {
            assert(pos + width <= flag_size_);
        }

        if (width <= 64) {
            SetValPos(pos, 0, width, is_on_data);
            return;
        }
        node_bitmap_pos_t s_off = 64 - pos % 64;
        node_bitmap_pos_t s_idx = pos / 64;
        SetValPos(pos, 0, s_off, is_on_data);

        width -= s_off;
        s_idx += 1;
        while (width > 64) {
            if (is_on_data)
                data_[s_idx] = 0;
            else
                flag_[s_idx] = 0;
            width -= 64;
            s_idx += 1;
        }
        SetValPos(s_idx * 64, 0, width, is_on_data);
    }

    /// Retrieve the subset of morton_key bits encoded by a top-level hierarchical chunk.
    ///
    /// Each level of the hierarchical encoding reveals a few bits of the morton code.
    /// This snippet returns the relevant bits from the top level.
    ///
    /// Replaces: `((symbol >> (H_LEVEL * CHUNK_WIDTH_SHIFT))).lsb64()`.
public: // hack
    constexpr uint64_t get_hier_top_level_bits(const morton_type &symbol) const noexcept
    {
        return symbol.get_bits_at(H_LEVEL * CHUNK_WIDTH_SHIFT, TOP_LEVEL_CHUNK_WIDTH_SHIFT);
    }

    /// Retrieve the subset of morton_key bits encoded by a hierarchical chunk.
    ///
    /// Don't call this on the top level chunk, which is of different width.
    ///
    /// Each level of the hierarchical encoding reveals a few bits of the morton code.
    /// This snippet returns the relevant bits from the top level.
    ///
    /// Replaces: `((morton_key >> (level * CHUNK_WIDTH_SHIFT)) & CHUNK_MASK).lsb64()`
    ///
    /// @param level Hierarchical trie level (0 to H_LEVEL), 0 is top and H_LEVEL is bottom.
    /// @param chunk_width_shift log2(chunk_width) - typically 16 for CHUNK_WIDTH=65536
    /// @return Chunk value as a uint64_t
public: // hack
    constexpr uint64_t get_hier_level_bits(const morton_type &symbol, size_t level) const noexcept
    {
        if (level == 0)
            return get_hier_top_level_bits(symbol);
        return symbol.get_bits_at((H_LEVEL - level) * CHUNK_WIDTH_SHIFT, CHUNK_WIDTH_SHIFT);
    }

    // DFS recursive helper for get_all_symbols: collects all symbols under a subtree
    // Returns the bitmap position after this subtree.
    //
    // @param bitmap_pos: Current position in the bitmap
    // @param parent_morton: Morton prefix built up from parent levels
    // @param hier_level: Current hierarchical level (1 to H_LEVEL, not 0 which is top-level)
    // @param out_symbols: Output vector to append symbols to
    // @return Position after this subtree in the bitmap
    node_bitmap_pos_t collect_all_symbols_recursive_dfs(node_bitmap_pos_t bitmap_pos,
                                                        const morton_type &parent_morton,
                                                        size_t hier_level,
                                                        std::vector<morton_type> &out_symbols) const
    {
        // Last level (H_LEVEL): these are the leaf chunks, collect individual symbols
        if (hier_level == H_LEVEL) {
            if (chunk_is_collapsed(bitmap_pos)) {
                // Collapsed: single child
                chunk_val_t only_symbol =
                    read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
                morton_type target_morton =
                    morton_type::copy_and_or_low_u64(parent_morton, only_symbol);
                out_symbols.push_back(target_morton);
                return bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
            } else {
                // Uncollapsed: scan CHUNK_WIDTH bits
                node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
                node_pos_t bit_base = 0;
                node_pos_t remaining = CHUNK_WIDTH;

                while (remaining > 0) {
                    width_type width = (remaining > 64) ? 64 : remaining;
                    uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                    while (word) {
                        unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                        node_pos_t i = bit_base + tz;
                        morton_type target_morton = parent_morton | i;
                        out_symbols.push_back(target_morton);
                        word &= word - 1;
                    }
                    bit_base += width;
                    remaining -= width;
                }
                return bitmap_pos + 1 + CHUNK_WIDTH;
            }
        }

        // Non-leaf levels: recurse into children in DFS order
        const int shift = CHUNK_WIDTH_SHIFT * (H_LEVEL - hier_level);

        if (chunk_is_collapsed(bitmap_pos)) {
            // Collapsed: single child, recurse immediately
            chunk_val_t only_symbol =
                read_u64_from_bitmap_pos(bitmap_pos + 1, COLLAPSED_CHUNK_WIDTH);
            morton_type new_parent = parent_morton;
            new_parent.add_shifted_u64(only_symbol, shift);

            node_bitmap_pos_t next_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
            return collect_all_symbols_recursive_dfs(next_pos, new_parent, hier_level + 1,
                                                     out_symbols);
        } else {
            // Uncollapsed: iterate through set bits and recurse into each child
            node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1;
            node_bitmap_pos_t next_pos = bitmap_pos + 1 + CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : remaining;
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type new_parent = parent_morton;
                    new_parent.add_shifted_u64(child_idx, shift);

                    // DFS: recurse into child subtree immediately
                    next_pos = collect_all_symbols_recursive_dfs(next_pos, new_parent,
                                                                 hier_level + 1, out_symbols);

                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
            return next_pos;
        }
    }

public:
    // Return all symbols stored in the bitmap starting at node/node_pos.
    //
    // This traverses the hierarchical structure efficiently (O(num_symbols)) instead of
    // brute-force iteration over 2^d possibilities.
    //
    // This function isn't performance critical. It's just used for testing.
    inline std::vector<morton_type> get_all_children(node_pos_t node_pos,
                                                     node_bitmap_pos_t node_bitmap_pos) const
    {
        std::vector<morton_type> symbols;

        if (node_is_collapsed(node_pos)) {
            morton_type only_symbol = read_from_bitmap_pos(node_bitmap_pos, DIMENSION);
            symbols.push_back(only_symbol);
            return symbols;
        }

        // Hierarchical node: handle top-level chunk specially, then use DFS traversal
        node_bitmap_pos_t next_pos;

        if (chunk_is_collapsed(node_bitmap_pos)) {
            // Collapsed top-level: single subtree
            chunk_val_t top_level_symbol =
                read_u64_from_bitmap_pos(node_bitmap_pos + 1, TOP_LEVEL_COLLAPSED_CHUNK_WIDTH);
            morton_type parent_morton =
                morton_type::from_shifted_u64(top_level_symbol, CHUNK_WIDTH_SHIFT * H_LEVEL);

            next_pos = node_bitmap_pos + 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH;
            collect_all_symbols_recursive_dfs(next_pos, parent_morton, 1, symbols);
        } else {
            // Uncollapsed top-level: iterate active top-level children
            node_bitmap_pos_t chunk_data_pos = node_bitmap_pos + 1;
            next_pos = node_bitmap_pos + 1 + TOP_LEVEL_CHUNK_WIDTH;
            node_pos_t bit_base = 0;
            node_pos_t remaining = TOP_LEVEL_CHUNK_WIDTH;

            while (remaining > 0) {
                width_type width = (remaining > 64) ? 64 : remaining;
                uint64_t word = read_u64_from_bitmap_pos(chunk_data_pos + bit_base, width);

                while (word) {
                    unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                    node_pos_t child_idx = bit_base + tz;

                    morton_type parent_morton =
                        morton_type::from_shifted_u64(child_idx, CHUNK_WIDTH_SHIFT * H_LEVEL);

                    // DFS: recurse into child subtree
                    next_pos =
                        collect_all_symbols_recursive_dfs(next_pos, parent_morton, 1, symbols);

                    word &= word - 1;
                }
                bit_base += width;
                remaining -= width;
            }
        }

        return symbols;
    }

    // Variant that returns both symbol width and total children count.
    // This avoids separate traversals when both values are needed.
    // Returns: (symbol_width, total_children)
    std::pair<node_pos_t, node_pos_t>
    get_hier_encoding_offsets_width_and_children(node_bitmap_pos_t node_bitmap_pos) const
    {
        auto [end_pos, total_children] = compute_symbol_width_and_children_dfs(node_bitmap_pos);
        return {end_pos - node_bitmap_pos, total_children};
    }

private:
    // DFS helper to compute both width and total leaf children
    // Returns (end_pos, total_children)
    std::pair<node_bitmap_pos_t, node_pos_t>
    compute_symbol_width_and_children_dfs(node_bitmap_pos_t node_bitmap_pos) const
    {
        // Handle top level first
        node_bitmap_pos_t next_pos;
        node_pos_t num_top_children;

        if (chunk_is_collapsed(node_bitmap_pos)) {
            next_pos = node_bitmap_pos + 1 + TOP_LEVEL_COLLAPSED_CHUNK_WIDTH;
            num_top_children = 1;
        } else {
            next_pos = node_bitmap_pos + 1 + TOP_LEVEL_CHUNK_WIDTH;
            num_top_children = popcount(node_bitmap_pos + 1, TOP_LEVEL_CHUNK_WIDTH, true);
        }

        // DFS traverse all top-level children
        node_pos_t total_children = 0;
        for (node_pos_t i = 0; i < num_top_children; ++i) {
            auto [end_pos, subtree_children] = compute_subtree_width_and_children_dfs(next_pos, 1);
            next_pos = end_pos;
            total_children += subtree_children;
        }

        return {next_pos, total_children};
    }

    // DFS helper to compute width and children of a subtree at given hier_level
    // Returns (end_pos, total_children)
    std::pair<node_bitmap_pos_t, node_pos_t>
    compute_subtree_width_and_children_dfs(node_bitmap_pos_t bitmap_pos, size_t hier_level) const
    {
        if (hier_level == H_LEVEL) {
            // Leaf level - count children in this chunk
            if (chunk_is_collapsed(bitmap_pos)) {
                return {bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH, 1};
            } else {
                node_pos_t children = popcount(bitmap_pos + 1, CHUNK_WIDTH, true);
                return {bitmap_pos + 1 + CHUNK_WIDTH, children};
            }
        }

        // Non-leaf: get children and recurse
        node_pos_t num_children;
        node_bitmap_pos_t next_pos;

        if (chunk_is_collapsed(bitmap_pos)) {
            next_pos = bitmap_pos + 1 + COLLAPSED_CHUNK_WIDTH;
            num_children = 1;
        } else {
            next_pos = bitmap_pos + 1 + CHUNK_WIDTH;
            num_children = popcount(bitmap_pos + 1, CHUNK_WIDTH, true);
        }

        node_pos_t total_children = 0;
        for (node_pos_t i = 0; i < num_children; ++i) {
            auto [end_pos, subtree_children] =
                compute_subtree_width_and_children_dfs(next_pos, hier_level + 1);
            next_pos = end_pos;
            total_children += subtree_children;
        }

        return {next_pos, total_children};
    }

public:
    // add the data and flag storage to the size calculation
    void update_size(uint64_t &size) const
    {
        if (data_size_ > 0) {
            size += BITS2BLOCKS(data_size_) * sizeof(data_type);
        }

        if (flag_size_ > 0) {
            size += BITS2BLOCKS(flag_size_) * sizeof(data_type);
        }
    }

    void print_data()
    {
#ifdef DEBUGF_ENABLED
        std::cout << "Data bits: ";
        for (node_pos_t i = 0; i < data_size_; ++i) {
            std::cout << GETBITVAL(data_, i);
        }
        std::cout << std::endl;
#endif
    }

    // Applicable for both top level and inner chunks.
    inline bool chunk_is_collapsed(node_bitmap_pos_t bitmap_pos) const
    {
        assert(bitmap_pos < data_size_);
        // 1 => collapsed. 0 => uncollapsed.
        return !GETBITVAL(data_, bitmap_pos);
    }

    inline void set_chunk_as_collapsed(node_bitmap_pos_t bitmap_pos) const
    {
        assert(bitmap_pos < data_size_);
        // 1 => collapsed. 0 => uncollapsed. TODO(yash): did I do this right
        CLRBITVAL(data_, bitmap_pos);
    }

    inline void set_chunk_as_uncollapsed(node_bitmap_pos_t bitmap_pos) const
    {
        assert(bitmap_pos < data_size_);
        SETBITVAL(data_, bitmap_pos);
    }

    // check if the node is collapsed
    inline bool node_is_collapsed(node_pos_t node_pos) const
    {
        assert(flag_size_ > 0 && flag_ != nullptr);
        assert(data_size_ > 0 && data_ != nullptr);
        assert(node_pos < flag_size_);

        // if the bit is set then the node is uncollapsed
        // 1 is uncollapsed, 0 is collapsed
        return !GETBITVAL(flag_, node_pos);
    }

    // set node
    inline void set_node_as_collapsed(node_pos_t node_pos) const
    {
        assert(flag_size_ > 0 && flag_ != nullptr);
        assert(data_size_ > 0 && data_ != nullptr);
        assert(node_pos < flag_size_);

        // 1 is uncollapsed, 0 is collapsed
        CLRBITVAL(flag_, node_pos);
    }

    // set node
    inline void set_node_as_uncollapsed(node_pos_t node_pos) const
    {
        assert(flag_size_ > 0 && flag_ != nullptr);
        assert(data_size_ > 0 && data_ != nullptr);
        assert(node_pos < flag_size_);

        // 1 is uncollapsed, 0 is collapsed
        SETBITVAL(flag_, node_pos);
    }

    /// Debug helper: dump chunk information at the given bitmap position.
    /// Returns a string describing the chunk's type and children.
    /// Format: "C <child_index>" for collapsed chunks
    ///         "U <child1>,<child2>,..." for uncollapsed chunks
    /// @param bitmap_pos: Position of the chunk in the bitmap
    /// @param is_top_level: True if this is a top-level chunk, false for inner chunks
    inline std::string dump_chunk(node_bitmap_pos_t bitmap_pos, bool is_top_level = false) const
    {
        if (bitmap_pos >= data_size_) {
            return std::string("EOF");
        }

        std::ostringstream oss;

        if (chunk_is_collapsed(bitmap_pos)) {
            // Collapsed chunk: read the index of the single child
            chunk_val_t child_index;
            if (is_top_level) {
                if (bitmap_pos + 1 + TOP_LEVEL_CHUNK_WIDTH_SHIFT >= data_size_) {
                    return std::string("EOF: Overrun");
                }
                child_index = read_u64_from_bitmap_pos(bitmap_pos + 1, TOP_LEVEL_CHUNK_WIDTH_SHIFT);
            } else {
                if (bitmap_pos + 1 + CHUNK_WIDTH_SHIFT > data_size_)
                    return std::string("EOF: Overrun");
                child_index = read_u64_from_bitmap_pos(bitmap_pos + 1, CHUNK_WIDTH_SHIFT);
            }
            oss << "C " << child_index;
        } else {
            // Uncollapsed chunk: read all set bits from the chunk bitmap
            oss << "U ";
            node_bitmap_pos_t chunk_data_pos = bitmap_pos + 1; // Skip collapsed flag bit

            node_pos_t chunk_width = is_top_level ? TOP_LEVEL_CHUNK_WIDTH : CHUNK_WIDTH;
            bool first = true;
            if (bitmap_pos + 1 + chunk_width >= data_size_)
                return std::string("EOF: Overrun");

            // Iterate through all positions in the chunk and collect set bits
            for (chunk_val_t i = 0; i < chunk_width; i++) {
                if (GETBITVAL(data_, chunk_data_pos + i)) {
                    if (!first) {
                        oss << ",";
                    }
                    oss << i;
                    first = false;
                }
            }

            // Handle empty uncollapsed chunk case
            if (first) {
                oss << "(empty)";
            }
        }

        return oss.str();
    }

    /// Dumps all chunks in a hierarchical node in DFS order.
    /// Iterates through the entire hierarchical node structure and prints each chunk.
    ///
    /// Don't trust this function. (see internal comment).
    ///
    /// @param node_bitmap_pos The starting position of the hierarchical node
    inline void dump_hier_node(node_bitmap_pos_t node_bitmap_pos) const
    {
        (void)node_bitmap_pos; // silence compile warning
#ifdef DEBUGF_ENABLED
        // Helper function: operates on non top level nodes.
        std::function<node_bitmap_pos_t(node_bitmap_pos_t, size_t)> dump_hier_subtree =
            [this, &dump_hier_subtree](node_bitmap_pos_t bitmap_pos,
                                       size_t level) -> node_bitmap_pos_t {
            printf("L%zu @ pos=%lu: %s\n", level, bitmap_pos, dump_chunk(bitmap_pos).c_str());

            if (level == H_LEVEL)
                return bitmap_pos + get_chunk_width(bitmap_pos, level);

            auto [children_in_chunk, chunk_width] = count_chunk_children(bitmap_pos);
            bitmap_pos += chunk_width;
            while (children_in_chunk > 0) {
                bitmap_pos = dump_hier_subtree(bitmap_pos, level + 1);
                children_in_chunk--;
            }
            return bitmap_pos;
        };

        printf("=== Hierarchical Node Dump (starting at bitmap_pos=%lu) ===\n", node_bitmap_pos);

        node_pos_t cur_bitmap_pos = node_bitmap_pos;
        printf("L0 @ pos=%lu: %s\n", cur_bitmap_pos, dump_chunk(cur_bitmap_pos, true).c_str());
        auto [children_in_chunk, chunk_width] = count_toplevel_chunk_children(node_bitmap_pos);
        cur_bitmap_pos += chunk_width;

        while (children_in_chunk > 0) {
            cur_bitmap_pos = dump_hier_subtree(cur_bitmap_pos, 1);
            children_in_chunk--;
        }

        printf("=== End of Hierarchical Node (final bitmap_pos=%lu) ===\n",
               cur_bitmap_pos - node_bitmap_pos);
#endif
    }

protected:
    // Data members
    data_type *data_ = nullptr;
    data_type *flag_ = nullptr;
    size_type data_size_ = 0;
    size_type flag_size_ = 0;

    template <n_dimensions_t D> friend class tree_block;
};
} // namespace compressed_bitmap

#endif
