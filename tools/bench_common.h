#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <vector>

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

inline std::string comma_fmt(size_t n)
{
    std::string s = std::to_string(n);
    if (s.size() <= 3)
        return s;

    std::string out;
    out.reserve(s.size() + (s.size() - 1) / 3);

    size_t lead = s.size() % 3;
    if (lead == 0)
        lead = 3;

    out.append(s, 0, lead);
    for (size_t pos = lead; pos < s.size(); pos += 3) {
        out.push_back(',');
        out.append(s, pos, 3);
    }
    return out;
}

inline std::string size_fmt(size_t bytes)
{
    char buf[32];
    if (bytes < 1024)
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    else if (bytes < 1024 * 1024)
        snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024ULL * 1024 * 1024)
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
    else
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

inline std::string time_fmt(double s)
{
    char buf[32];
    if (s < 10.0)
        snprintf(buf, sizeof(buf), "%.4f", s);
    else if (s < 100.0)
        snprintf(buf, sizeof(buf), "%.2f", s);
    else
        snprintf(buf, sizeof(buf), "%.1f", s);
    return buf;
}

// ---------------------------------------------------------------------------
// CLI parsing helpers
// ---------------------------------------------------------------------------

inline std::vector<size_t> parse_csv_sizes(const char *s)
{
    std::vector<size_t> out;
    std::string token;
    for (const char *p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!token.empty()) {
                out.push_back(std::stoul(token));
                token.clear();
            }
            if (*p == '\0')
                break;
        } else {
            token += *p;
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline std::vector<uint32_t> parse_csv_uint32s(const char *s)
{
    std::vector<uint32_t> out;
    std::string token;
    for (const char *p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!token.empty()) {
                out.push_back(static_cast<uint32_t>(std::stoul(token)));
                token.clear();
            }
            if (*p == '\0')
                break;
        } else {
            token += *p;
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---------------------------------------------------------------------------
// System helpers
// ---------------------------------------------------------------------------

inline void ensure_large_stack(size_t mb = 256)
{
    struct rlimit rl;
    getrlimit(RLIMIT_STACK, &rl);
    rlim_t needed = static_cast<rlim_t>(mb) * 1024 * 1024;
    if (rl.rlim_cur < needed) {
        rl.rlim_cur = std::min(needed, rl.rlim_max);
        setrlimit(RLIMIT_STACK, &rl);
    }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

inline bool load_float_file(const char *path, std::vector<float> &out, size_t num_floats)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return false;
    }
    f.seekg(0, std::ios::end);
    auto file_bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    size_t needed = num_floats * sizeof(float);
    if (file_bytes < needed) {
        fprintf(stderr, "Error: %s too small (%zu bytes, need %zu)\n", path, file_bytes, needed);
        return false;
    }
    out.resize(num_floats);
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(needed));
    return true;
}

// ---------------------------------------------------------------------------
// Morton encoding (requires NUM_DIMENSIONS + data_point.h)
// ---------------------------------------------------------------------------

#ifdef NUM_DIMENSIONS
#include "mdtrie/data_point.h"

inline std::string encode_morton_bitstring(const data_point<NUM_DIMENSIONS> &pt)
{
    std::string bits;
    bits.reserve(static_cast<size_t>(NUM_DIMENSIONS) * MAX_TRIE_DEPTH);
    for (trie_level_t level = 0; level < MAX_TRIE_DEPTH; ++level) {
        morton_t sym = pt.leaf_to_symbol(level);
        for (size_t i = 0; i < NUM_DIMENSIONS; ++i) {
            bits += sym.get_bit_unsafe(NUM_DIMENSIONS - 1 - i) ? '1' : '0';
        }
    }
    return bits;
}
#endif
#include <cstring>
#include <chrono>

static inline uint32_t float_to_ordered_u32(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

static inline float half_to_float(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1F;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | (static_cast<uint32_t>(exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

static inline uint16_t float_to_ordered_u16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    
    uint16_t h;
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t u_abs = u & 0x7FFFFFFFu;
    
    if (u_abs == 0) {
        h = sign;
    } else if (u_abs >= 0x47800000u) { // >= 65536.0 (inf)
        h = sign | 0x7C00u;
    } else if (u_abs < 0x33000000u) { // < 5.96e-8 (subnormal)
        h = sign; 
    } else {
        uint32_t exp = ((u_abs >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = (u_abs & 0x7FFFFFu) >> 13;
        h = sign | (exp << 10) | mant;
    }
    
    return (h & 0x8000u) ? ~h : (h ^ 0x8000u);
}

static inline std::string morton_encode(const float *data, size_t n_dims, float shift, bool is_fp16)
{
    std::string bits;
    if (is_fp16) {
        std::vector<uint16_t> ordered(n_dims);
        for (size_t i = 0; i < n_dims; i++)
            ordered[i] = float_to_ordered_u16(data[i] + shift);

        bits.reserve(n_dims * 16);
        for (int level = 0; level < 16; level++) {
            uint16_t bit_offset = 15 - static_cast<uint16_t>(level);
            for (size_t d = 0; d < n_dims; d++)
                bits += ((ordered[d] >> bit_offset) & 1) ? '1' : '0';
        }
    } else {
        std::vector<uint32_t> ordered(n_dims);
        for (size_t i = 0; i < n_dims; i++)
            ordered[i] = float_to_ordered_u32(data[i] + shift);

        bits.reserve(n_dims * 32);
        for (int level = 0; level < 32; level++) {
            uint32_t bit_offset = 31 - static_cast<uint32_t>(level);
            for (size_t d = 0; d < n_dims; d++)
                bits += ((ordered[d] >> bit_offset) & 1) ? '1' : '0';
        }
    }
    return bits;
}

static inline size_t file_size_bytes(const char *path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return 0;
    return static_cast<size_t>(f.tellg());
}

struct EncodedDataset {
    std::vector<std::string> keys;
    size_t vectors_read = 0;
    double encode_s = 0;
    double sort_dedup_s = 0;
};

static inline EncodedDataset load_encode_dedup(const char *path, size_t n_dims,
                                        bool is_fp16, size_t target_unique,
                                        float shift)
{
    EncodedDataset res;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return res;
    }
    size_t fsize = file_size_bytes(path);
    size_t elem_bytes = is_fp16 ? sizeof(uint16_t) : sizeof(float);
    size_t row_bytes = n_dims * elem_bytes;
    size_t total_in_file = fsize / row_bytes;
    if (total_in_file == 0)
        return res;

    std::vector<float> fbuf(n_dims);
    std::vector<uint16_t> hbuf;
    if (is_fp16)
        hbuf.resize(n_dims);

    auto read_and_encode_one = [&]() -> bool {
        if (is_fp16) {
            f.read(reinterpret_cast<char *>(hbuf.data()),
                   static_cast<std::streamsize>(row_bytes));
            if (!f)
                return false;
            for (size_t d = 0; d < n_dims; d++)
                fbuf[d] = half_to_float(hbuf[d]);
        } else {
            f.read(reinterpret_cast<char *>(fbuf.data()),
                   static_cast<std::streamsize>(row_bytes));
            if (!f)
                return false;
        }
        res.keys.push_back(morton_encode(fbuf.data(), n_dims, shift, is_fp16));
        res.vectors_read++;
        return true;
    };

    // Phase 1: read target * 1.1 vectors (10% overshoot for duplicates).
    size_t initial_read = std::min(total_in_file,
                                   static_cast<size_t>(target_unique * 1.1) + 1000);
    res.keys.reserve(initial_read);

    auto t_enc = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < initial_read; i++) {
        if (!read_and_encode_one())
            break;
    }
    auto t_enc_end = std::chrono::high_resolution_clock::now();
    res.encode_s = std::chrono::duration<double>(t_enc_end - t_enc).count();

    auto t_sort = std::chrono::high_resolution_clock::now();
    std::sort(res.keys.begin(), res.keys.end());
    res.keys.erase(std::unique(res.keys.begin(), res.keys.end()), res.keys.end());
    auto t_sort_end = std::chrono::high_resolution_clock::now();
    res.sort_dedup_s = std::chrono::duration<double>(t_sort_end - t_sort).count();

    // Phase 2: if initial read wasn't enough, keep reading in batches and
    // merge into the already-sorted result until we hit the target.
    while (res.keys.size() < target_unique && res.vectors_read < total_in_file) {
        size_t deficit = target_unique - res.keys.size();
        size_t batch = std::min(total_in_file - res.vectors_read,
                                std::max(deficit * 2, static_cast<size_t>(10000)));

        std::vector<std::string> extra;
        extra.reserve(batch);
        size_t prev_read = res.vectors_read;

        auto t2_enc = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch; i++) {
            if (!read_and_encode_one())
                break;
        }
        auto t2_enc_end = std::chrono::high_resolution_clock::now();
        res.encode_s += std::chrono::duration<double>(t2_enc_end - t2_enc).count();

        // The new keys were appended to res.keys starting at the old sorted size.
        // Split them off, sort, dedup the batch, then merge.
        size_t old_size = res.keys.size() - (res.vectors_read - prev_read);
        extra.assign(std::make_move_iterator(res.keys.begin() + static_cast<std::ptrdiff_t>(old_size)),
                     std::make_move_iterator(res.keys.end()));
        res.keys.resize(old_size);

        auto t2_sort = std::chrono::high_resolution_clock::now();
        std::sort(extra.begin(), extra.end());
        extra.erase(std::unique(extra.begin(), extra.end()), extra.end());

        std::vector<std::string> merged;
        merged.reserve(res.keys.size() + extra.size());
        std::merge(res.keys.begin(), res.keys.end(),
                   extra.begin(), extra.end(),
                   std::back_inserter(merged));
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
        res.keys = std::move(merged);
        auto t2_sort_end = std::chrono::high_resolution_clock::now();
        res.sort_dedup_s += std::chrono::duration<double>(t2_sort_end - t2_sort).count();

        fflush(stdout);
        fprintf(stderr, "  [extend] read %s vectors so far -> %s unique keys (target: %s)\n",
                comma_fmt(res.vectors_read).c_str(),
                comma_fmt(res.keys.size()).c_str(),
                comma_fmt(target_unique).c_str());
    }

    return res;
}
