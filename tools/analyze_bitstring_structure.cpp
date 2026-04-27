/**
 * analyze_bitstring_structure.cpp
 *
 * Pure data analysis on Morton-encoded bitstrings (no DAWG): for each power-of-2 group width,
 * measures symbol cardinality at aligned chunk positions; also runs a greedy segmentation that
 * picks a group width per segment subject to a saturation (K/min(N,2^g)) threshold.
 *
 * This tool:
 *   1) Reads --input (embedding binary), Morton-encodes up to --n keys for --dims / --dtype
 *   2) Emits a main CSV (--output or stdout) with per-(group_width, chunk_idx) cardinality metrics
 *      (including cardinality_over_kmax = K/min(N,2^g))
 *   3) Writes a companion {output_stem}_segmentation.csv when segmentation runs
 *
 * Example:
 *   cmake --build . --target analyze_bitstring_structure
 *   ./analyze_bitstring_structure \
 *       --input data/embeddings/qwen3-embedding-0.6b/msmarco_v2/msmarco_v2_corpus_256d_float32.bin \
 *       --dims 256 --dtype float32 --n 100000 \
 *       --group-widths 1,2,4,8,16,32,64,128,256 \
 *       --seg-thresholds 0.5,0.7,0.9 \
 *       --output results/bitstring_analysis_256d.csv
 */

#include "bench_common.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    const char *input_path = nullptr;
    size_t dims = 0;
    bool is_fp16 = false;
    size_t n_keys = 100000;
    float shift = 0.5f;
    std::vector<uint32_t> group_widths;
    std::vector<double> seg_thresholds;
    bool run_segmentation = true;
    const char *output_path = nullptr; // main CSV
};

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --input <path> --dims <D> --dtype float32|float16\n"
            "       [--n <keys>] [--shift <f>]\n"
            "       [--group-widths <csv>] [--seg-thresholds <csv>] [--no-segmentation]\n"
            "       [--output <csv>]\n"
            "\n"
            "Defaults:\n"
            "  --n           100000\n"
            "  --shift       0.5\n"
            "  --group-widths 1,2,4,8,16,32,64,128,256,512,1024\n"
            "  --seg-thresholds 0.3,0.5,0.7,0.8,0.9,0.95\n"
            "  --output      stdout\n",
            prog);
}

static std::vector<double> parse_csv_doubles(const char *s)
{
    std::vector<double> out;
    std::string token;
    for (const char *p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!token.empty()) {
                out.push_back(std::stod(token));
                token.clear();
            }
            if (*p == '\0')
                break;
        } else {
            token += *p;
        }
    }
    return out;
}

static Config parse_args(int argc, char **argv)
{
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input") && i + 1 < argc)
            cfg.input_path = argv[++i];
        else if (!strcmp(argv[i], "--dims") && i + 1 < argc)
            cfg.dims = std::stoul(argv[++i]);
        else if (!strcmp(argv[i], "--dtype") && i + 1 < argc) {
            cfg.is_fp16 = (strcmp(argv[++i], "float16") == 0);
        } else if (!strcmp(argv[i], "--n") && i + 1 < argc)
            cfg.n_keys = std::stoul(argv[++i]);
        else if (!strcmp(argv[i], "--shift") && i + 1 < argc)
            cfg.shift = std::stof(argv[++i]);
        else if (!strcmp(argv[i], "--group-widths") && i + 1 < argc) {
            cfg.group_widths = parse_csv_uint32s(argv[++i]);
        } else if (!strcmp(argv[i], "--seg-thresholds") && i + 1 < argc) {
            cfg.seg_thresholds = parse_csv_doubles(argv[++i]);
        } else if (!strcmp(argv[i], "--no-segmentation")) {
            cfg.run_segmentation = false;
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            cfg.output_path = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            exit(1);
        }
    }
    if (!cfg.input_path || cfg.dims == 0) {
        fprintf(stderr, "Error: --input and --dims are required.\n");
        usage(argv[0]);
        exit(1);
    }
    // Default group widths: powers of 2 from 1 to 1024
    if (cfg.group_widths.empty()) {
        for (uint32_t g = 1; g <= 1024; g *= 2)
            cfg.group_widths.push_back(g);
    }
    if (cfg.seg_thresholds.empty()) {
        cfg.seg_thresholds = {0.3, 0.5, 0.7, 0.8, 0.9, 0.95};
    }
    for (double t : cfg.seg_thresholds) {
        if (t < 0.0 || t > 1.0) {
            fprintf(stderr, "Error: --seg-thresholds values must be in [0,1].\n");
            exit(1);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Cardinality computation
// ---------------------------------------------------------------------------

struct ChunkStats {
    uint32_t group_width;
    uint32_t chunk_idx;
    uint32_t chunk_start_bit;
    size_t cardinality;
};

struct CardinalityCount {
    size_t cardinality;
    bool exact;
};

// K_max = min(N, 2^g). For g >= 64, 2^g is not representable in uint64_t; any realistic N
// satisfies N < 2^64, so min(N, 2^g) = N.
static size_t symbol_capacity(size_t n_keys, uint32_t g)
{
    if (g >= 64u)
        return n_keys;
    const uint64_t two_g = 1ULL << g;
    if (two_g >= n_keys)
        return n_keys;
    return static_cast<size_t>(two_g);
}

// Count unique symbols, optionally stopping once the count exceeds stop_after_unique.
// When exact capacity is reached (e.g. both symbols for width=1), the count is exact
// without scanning the remaining keys.
static CardinalityCount count_unique_symbols_until(const std::vector<std::string> &keys,
                                                   uint32_t start_bit,
                                                   uint32_t width,
                                                   size_t stop_after_unique)
{
    const size_t kmax = symbol_capacity(keys.size(), width);
    const size_t reserve_n =
        (stop_after_unique == std::numeric_limits<size_t>::max())
            ? std::min(keys.size(), kmax)
            : std::min(keys.size(), std::min(kmax, stop_after_unique + 1));

    std::unordered_set<std::string_view> seen;
    seen.reserve(reserve_n);
    for (const auto &k : keys) {
        std::string_view sv(k.data() + start_bit, width);
        auto inserted = seen.insert(sv);
        if (!inserted.second)
            continue;

        if (seen.size() > stop_after_unique)
            return {seen.size(), false};
        if (seen.size() == kmax)
            return {seen.size(), true};
    }
    return {seen.size(), true};
}

static size_t count_unique_symbols_exact(const std::vector<std::string> &keys,
                                         uint32_t start_bit,
                                         uint32_t width)
{
    return count_unique_symbols_until(
               keys, start_bit, width, std::numeric_limits<size_t>::max())
        .cardinality;
}

// ---------------------------------------------------------------------------
// Greedy segmentation
// ---------------------------------------------------------------------------

struct Segment {
    uint32_t start_bit;
    uint32_t width;
    size_t cardinality;
    double cardinality_over_n;
    double cardinality_over_kmax;
};

// Greedy segmentation: at each position, pick the LARGEST group width whose saturation
// ρ = K / min(N, 2^g) is ≤ threshold (so small g is not auto-favored via tiny K/N).
// If no group width meets the threshold, fall back to the smallest candidate width tried.
static std::vector<Segment> greedy_segment(
    const std::vector<std::string> &keys,
    uint32_t total_bits,
    const std::vector<uint32_t> &group_widths_desc, // sorted DESCENDING
    double threshold)
{
    std::vector<Segment> segs;
    uint32_t pos = 0;
    size_t n = keys.size();

    while (pos < total_bits) {
        Segment best;
        best.start_bit = pos;
        best.width = 0;
        best.cardinality = 0;
        best.cardinality_over_n = 1.0;
        best.cardinality_over_kmax = 1.0;

        bool found = false;
        uint32_t fallback_width = 0;
        for (uint32_t g : group_widths_desc) {
            if (pos + g > total_bits)
                continue;
            size_t kmax = symbol_capacity(n, g);
            const size_t max_allowed =
                static_cast<size_t>(threshold * static_cast<double>(kmax));
            CardinalityCount count = count_unique_symbols_until(keys, pos, g, max_allowed);
            size_t card = count.cardinality;
            double rho = static_cast<double>(card) / static_cast<double>(kmax);
            if (count.exact && rho <= threshold) {
                double ratio = static_cast<double>(card) / static_cast<double>(n);
                best.width = g;
                best.cardinality = card;
                best.cardinality_over_n = ratio;
                best.cardinality_over_kmax = rho;
                found = true;
                break; // largest g that fits threshold
            }
            // Track the smallest g tried as fallback
            if (fallback_width == 0 || g < fallback_width)
                fallback_width = g;
        }

        // If nothing fit the threshold, fallback width is already set to the
        // smallest g that was tried. But we still need to handle the case where
        // even width=1 didn't pass. Use it anyway.
        if (!found) {
            best.width = (fallback_width > 0) ? fallback_width : 1;
            best.cardinality = count_unique_symbols_exact(keys, pos, best.width);
            best.cardinality_over_n = static_cast<double>(best.cardinality) / static_cast<double>(n);
            size_t kmax = symbol_capacity(n, best.width);
            best.cardinality_over_kmax =
                static_cast<double>(best.cardinality) / static_cast<double>(kmax);
        }

        segs.push_back(best);
        pos += best.width;
        if (segs.size() % 1024 == 0 || pos == total_bits) {
            fprintf(stderr, "    progress: %u/%u bits, %zu segments\r",
                    pos, total_bits, segs.size());
        }
    }
    fprintf(stderr, "\n");
    return segs;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);

    uint32_t bits_per_dim = cfg.is_fp16 ? 16u : 32u;
    uint32_t total_bits = static_cast<uint32_t>(cfg.dims) * bits_per_dim;

    fprintf(stderr, "=== Bitstring Structure Analysis ===\n");
    fprintf(stderr, "  input:   %s\n", cfg.input_path);
    fprintf(stderr, "  dims:    %zu\n", cfg.dims);
    fprintf(stderr, "  dtype:   %s\n", cfg.is_fp16 ? "float16" : "float32");
    fprintf(stderr, "  bits:    %u total (%u dims × %u bits/dim)\n",
            total_bits, static_cast<uint32_t>(cfg.dims), bits_per_dim);
    fprintf(stderr, "  n_keys:  %s (target unique)\n", comma_fmt(cfg.n_keys).c_str());
    fprintf(stderr, "  shift:   %.2f\n", cfg.shift);
    fprintf(stderr, "  groups:  ");
    for (size_t i = 0; i < cfg.group_widths.size(); i++) {
        if (i > 0)
            fprintf(stderr, ",");
        fprintf(stderr, "%u", cfg.group_widths[i]);
    }
    fprintf(stderr, "\n\n");
    if (cfg.run_segmentation) {
        fprintf(stderr, "  seg thresholds: ");
        for (size_t i = 0; i < cfg.seg_thresholds.size(); i++) {
            if (i > 0)
                fprintf(stderr, ",");
            fprintf(stderr, "%.2f", cfg.seg_thresholds[i]);
        }
        fprintf(stderr, "\n\n");
    } else {
        fprintf(stderr, "  segmentation: disabled\n\n");
    }

    // --- Load & encode ---
    fprintf(stderr, "Loading and encoding...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    EncodedDataset ds = load_encode_dedup(cfg.input_path, cfg.dims,
                                          cfg.is_fp16, cfg.n_keys, cfg.shift);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();

    if (ds.keys.empty()) {
        fprintf(stderr, "Error: no keys loaded.\n");
        return 1;
    }

    size_t n = ds.keys.size();
    fprintf(stderr, "  Loaded %s unique keys (from %s vectors) in %.2fs\n",
            comma_fmt(n).c_str(), comma_fmt(ds.vectors_read).c_str(), load_s);
    fprintf(stderr, "  Key length: %zu chars (expected %u)\n",
            ds.keys[0].size(), total_bits);

    if (ds.keys[0].size() != total_bits) {
        fprintf(stderr, "Error: key length mismatch!\n");
        return 1;
    }

    // Filter group widths: skip any that don't divide evenly OR exceed total_bits.
    // Actually, we want to handle non-divisible group widths gracefully by just
    // skipping the trailing partial chunk. But let's also skip widths > total_bits.
    std::vector<uint32_t> valid_widths;
    for (uint32_t g : cfg.group_widths) {
        if (g <= total_bits)
            valid_widths.push_back(g);
        else
            fprintf(stderr, "  Warning: group_width %u > total_bits %u, skipping\n", g, total_bits);
    }

    // --- Compute per-chunk cardinality ---
    // Open output
    FILE *out = stdout;
    if (cfg.output_path) {
        out = fopen(cfg.output_path, "w");
        if (!out) {
            fprintf(stderr, "Error: cannot open %s for writing\n", cfg.output_path);
            return 1;
        }
    }

    // CSV header
    fprintf(out,
            "dim,n_keys,group_width,chunk_idx,chunk_start_bit,total_bits,cardinality,cardinality_over_n,"
            "cardinality_over_kmax\n");

    for (uint32_t g : valid_widths) {
        uint32_t n_chunks = total_bits / g;
        fprintf(stderr, "  group_width=%u: %u chunks...\n", g, n_chunks);

        auto t_g0 = std::chrono::high_resolution_clock::now();
        for (uint32_t ci = 0; ci < n_chunks; ci++) {
            uint32_t start = ci * g;
            size_t card = count_unique_symbols_exact(ds.keys, start, g);
            double ratio = static_cast<double>(card) / static_cast<double>(n);
            size_t kmax = symbol_capacity(n, g);
            double rho = static_cast<double>(card) / static_cast<double>(kmax);
            fprintf(out, "%zu,%zu,%u,%u,%u,%u,%zu,%.8f,%.8f\n",
                    cfg.dims, n, g, ci, start, total_bits, card, ratio, rho);
        }
        auto t_g1 = std::chrono::high_resolution_clock::now();
        double g_s = std::chrono::duration<double>(t_g1 - t_g0).count();
        fprintf(stderr, "    done in %.2fs\n", g_s);
    }

    if (cfg.output_path)
        fclose(out);

    if (!cfg.run_segmentation) {
        fprintf(stderr, "\nDone.\n");
        return 0;
    }

    // --- Greedy segmentation ---
    // Build group widths in descending order for greedy algorithm
    std::vector<uint32_t> gw_desc(valid_widths.rbegin(), valid_widths.rend());

    // Segmentation CSV: same base name with _segmentation suffix
    std::string seg_path;
    if (cfg.output_path) {
        seg_path = cfg.output_path;
        auto dot = seg_path.rfind('.');
        if (dot != std::string::npos)
            seg_path.insert(dot, "_segmentation");
        else
            seg_path += "_segmentation";
    }

    FILE *seg_out = nullptr;
    if (!seg_path.empty()) {
        seg_out = fopen(seg_path.c_str(), "w");
        if (!seg_out) {
            fprintf(stderr, "Error: cannot open %s for writing\n", seg_path.c_str());
            return 1;
        }
    } else {
        seg_out = stdout;
        fprintf(seg_out, "\n--- Greedy Segmentation ---\n");
    }

    fprintf(seg_out,
            "dim,n_keys,total_bits,threshold,segment_idx,segment_start_bit,segment_width,cardinality,"
            "cardinality_over_n,cardinality_over_kmax\n");

    for (double thr : cfg.seg_thresholds) {
        fprintf(stderr, "  greedy segmentation (ρ threshold=%.2f)...\n", thr);
        auto t_s0 = std::chrono::high_resolution_clock::now();
        auto segs = greedy_segment(ds.keys, total_bits, gw_desc, thr);
        auto t_s1 = std::chrono::high_resolution_clock::now();
        double seg_s = std::chrono::duration<double>(t_s1 - t_s0).count();

        for (size_t si = 0; si < segs.size(); si++) {
            const auto &s = segs[si];
            fprintf(seg_out, "%zu,%zu,%u,%.2f,%zu,%u,%u,%zu,%.8f,%.8f\n",
                    cfg.dims, n, total_bits, thr,
                    si, s.start_bit, s.width, s.cardinality, s.cardinality_over_n,
                    s.cardinality_over_kmax);
        }

        fprintf(stderr, "    %zu segments in %.2fs\n", segs.size(), seg_s);
    }

    if (!seg_path.empty() && seg_out)
        fclose(seg_out);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
