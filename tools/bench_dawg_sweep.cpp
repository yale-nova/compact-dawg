/**
 * bench_dawg_sweep.cpp
 *
 * Multi-axis benchmark for CompactDawg: storage (total bytes, bytes/key, edges) and build
 * timings (Morton encode, sort+dedup, insert, finish) across dimension, dtype, unique key
 * count N, and GROUP_BITS (CD-* and PC-* templates).
 * CD rows also include a suffix-collapse storage comparison: an equivalent packed trie
 * estimate before memoized suffix deduplication vs. the final CompactDawg size.
 *
 * This tool:
 *   1) Reads embedding binaries from --data-dir using msmarco_v2_corpus_{dim}d_{dtype}.bin
 *   2) Morton-encodes once per (dim, dtype), up to the maximum requested N (after sort+dedup)
 *   3) For each N and valid GROUP_BITS (must divide dim*16 for float16 or dim*32 for float32),
 *      benchmarks each CD/GB and PC/GB variant and optionally dawgdic (--dawgdic, N <= 10000
 *      only; larger N risks 32-bit index overflow in dawgdic)
 *   4) Writes --output-csv (default dawg_sweep_results.csv) and prints a table on stdout;
 *      progress and warnings go to stderr
 *
 * Plot sweep CSVs with:
 *   python3 scripts/plot_dawg_sweep.py -i <csv> -o plots/sweep/
 *
 * Example:
 *   cmake --build . --target bench_dawg_sweep
 *   ./bench_dawg_sweep --data-dir data/embeddings/qwen3-embedding-0.6b/msmarco_v2 \
 *       --dims 256,1024 --dtypes float32 --n-keys 10000,100000 \
 *       --group-bits 32,64,128 --output-csv results.csv
 */

#include "bench_common.h"
#include "compact_dawg.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include <dawgdic/dawg-builder.h>
#include <dawgdic/dictionary-builder.h>
#include <dawgdic/dictionary.h>

using hrc = std::chrono::high_resolution_clock;

// dawgdic uses 32-bit state indices; large N at high dimension overflows silently.
// Only benchmark dawgdic for small N (same spirit as local sanity checks).
static constexpr size_t DAWGDIC_MAX_KEYS = 10000;

static std::string bpk_fmt(double b)
{
    char buf[32];
    if (b < 1024.0)
        snprintf(buf, sizeof(buf), "%.1f B", b);
    else if (b < 1024.0 * 1024.0)
        snprintf(buf, sizeof(buf), "%.1f KB", b / 1024.0);
    else
        snprintf(buf, sizeof(buf), "%.2f MB", b / (1024.0 * 1024.0));
    return buf;
}


// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct SweepResult {
    uint32_t dim;
    std::string dtype;
    size_t n_keys;
    size_t n_unique_keys;
    uint32_t group_bits;
    std::string method;
    size_t total_bytes;
    double bytes_per_key;
    size_t edges;
    size_t trie_edges_before_suffix_collapse;
    size_t dawg_edges_after_suffix_collapse;
    size_t pre_suffix_total_bytes;
    double pre_suffix_bytes_per_key;
    double pre_suffix_normalized_bpk;
    double post_suffix_normalized_bpk;
    double suffix_collapse_saved_bytes_per_key;
    double suffix_collapse_saved_normalized_bpk;
    double suffix_collapse_saving_pct;
    double morton_encode_s;
    double sort_dedup_s;
    double insert_s;
    double finish_s;
    double total_build_s;
};

template <uint32_t BITS>
static SweepResult bench_cd(const std::vector<std::string> &keys, size_t total_bits)
{
    SweepResult r{};
    char name[16];
    snprintf(name, sizeof(name), "CD-%u", BITS);
    r.method = name;
    r.group_bits = BITS;

    auto t0 = hrc::now();
    CompactDawg<BITS, false, true> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    auto t1 = hrc::now();
    dawg.Finish();
    auto t2 = hrc::now();

    const auto &ss = dawg.GetSharingStats();
    const double n = static_cast<double>(keys.size());
    const double key_bytes = static_cast<double>(total_bits) / 8.0;

    r.total_bytes = dawg.size_in_bytes();
    r.bytes_per_key = static_cast<double>(r.total_bytes) / n;
    r.edges = dawg.get_total_edges();
    r.trie_edges_before_suffix_collapse = ss.trie_edges;
    r.dawg_edges_after_suffix_collapse = ss.dawg_edges;
    r.pre_suffix_total_bytes =
        CompactDawg<BITS>::PackedFixedWidthBytesForEdgeCount(r.trie_edges_before_suffix_collapse);
    r.pre_suffix_bytes_per_key = static_cast<double>(r.pre_suffix_total_bytes) / n;
    r.pre_suffix_normalized_bpk = key_bytes > 0.0 ? r.pre_suffix_bytes_per_key / key_bytes : 0.0;
    r.post_suffix_normalized_bpk = key_bytes > 0.0 ? r.bytes_per_key / key_bytes : 0.0;
    r.suffix_collapse_saved_bytes_per_key = r.pre_suffix_bytes_per_key - r.bytes_per_key;
    r.suffix_collapse_saved_normalized_bpk =
        r.pre_suffix_normalized_bpk - r.post_suffix_normalized_bpk;
    r.suffix_collapse_saving_pct =
        r.pre_suffix_bytes_per_key > 0.0
            ? 1.0 - (r.bytes_per_key / r.pre_suffix_bytes_per_key)
            : 0.0;
    r.insert_s = std::chrono::duration<double>(t1 - t0).count();
    r.finish_s = std::chrono::duration<double>(t2 - t1).count();
    r.total_build_s = std::chrono::duration<double>(t2 - t0).count();
    return r;
}

template <uint32_t BITS>
static SweepResult bench_pc(const std::vector<std::string> &keys)
{
    SweepResult r{};
    char name[16];
    snprintf(name, sizeof(name), "PC-%u", BITS);
    r.method = name;
    r.group_bits = BITS;

    auto t0 = hrc::now();
    CompactDawg<BITS, true> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    auto t1 = hrc::now();
    dawg.Finish();
    auto t2 = hrc::now();

    r.total_bytes = dawg.size_in_bytes();
    r.bytes_per_key = static_cast<double>(r.total_bytes) / static_cast<double>(keys.size());
    r.edges = dawg.get_total_edges();
    r.insert_s = std::chrono::duration<double>(t1 - t0).count();
    r.finish_s = std::chrono::duration<double>(t2 - t1).count();
    r.total_build_s = std::chrono::duration<double>(t2 - t0).count();
    return r;
}

// dawgdic baseline: standard 8-bit char encoding (GROUP_BITS=1 effectively).
// Uses 32-bit state indices internally, so it will silently overflow on large
// high-dimensional datasets. We detect this via the bytes_per_key sanity check.
static SweepResult bench_dawgdic(const std::vector<std::string> &keys)
{
    SweepResult r{};
    r.method = "dawgdic";
    r.group_bits = 0;

    auto t0 = hrc::now();
    dawgdic::DawgBuilder builder;
    bool insert_ok = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!builder.Insert(keys[i].c_str(),
                            static_cast<dawgdic::SizeType>(keys[i].length()), 0)) {
            fprintf(stderr, "  [dawgdic] Insert failed at key %s\n", comma_fmt(i).c_str());
            insert_ok = false;
            break;
        }
    }
    auto t1 = hrc::now();

    dawgdic::Dawg dawg;
    bool finish_ok = insert_ok && builder.Finish(&dawg);
    dawgdic::Dictionary dic;
    bool build_ok = finish_ok && dawgdic::DictionaryBuilder::Build(dawg, &dic);
    auto t2 = hrc::now();

    r.insert_s = std::chrono::duration<double>(t1 - t0).count();
    r.finish_s = std::chrono::duration<double>(t2 - t1).count();
    r.total_build_s = std::chrono::duration<double>(t2 - t0).count();

    if (!build_ok) {
        r.total_bytes = 0;
        r.bytes_per_key = 0;
        r.edges = 0;
        return r;
    }

    r.total_bytes = dic.total_size();
    r.bytes_per_key = static_cast<double>(r.total_bytes) / static_cast<double>(keys.size());
    r.edges = dic.size();

    if (r.bytes_per_key < 1.0) {
        fprintf(stderr, "  [dawgdic] Likely 32-bit overflow: %.1f B/key is implausibly small\n",
                r.bytes_per_key);
        r.total_bytes = 0;
        r.bytes_per_key = 0;
        r.edges = 0;
    }

    return r;
}

// Compile-time dispatch: run bench_cd and bench_pc for a single GROUP_BITS value,
// only if it divides the total key length.
template <uint32_t BITS>
static void maybe_bench_one_gb(std::vector<SweepResult> &out,
                               const std::vector<std::string> &keys,
                               size_t total_bits)
{
    if (total_bits % BITS != 0)
        return;
    out.push_back(bench_cd<BITS>(keys, total_bits));
    out.push_back(bench_pc<BITS>(keys));
}

// Dispatch across all supported GROUP_BITS values
static void bench_all_group_bits(std::vector<SweepResult> &out,
                                 const std::vector<std::string> &keys,
                                 size_t total_bits,
                                 const std::vector<uint32_t> &requested_gb)
{
    auto want = [&](uint32_t gb) {
        for (uint32_t v : requested_gb)
            if (v == gb)
                return true;
        return false;
    };

    if (want(16))
        maybe_bench_one_gb<16>(out, keys, total_bits);
    if (want(32))
        maybe_bench_one_gb<32>(out, keys, total_bits);
    if (want(64))
        maybe_bench_one_gb<64>(out, keys, total_bits);
    if (want(128))
        maybe_bench_one_gb<128>(out, keys, total_bits);
    if (want(256))
        maybe_bench_one_gb<256>(out, keys, total_bits);
    if (want(512))
        maybe_bench_one_gb<512>(out, keys, total_bits);
    if (want(1024))
        maybe_bench_one_gb<1024>(out, keys, total_bits);
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

static void write_csv_header(FILE *fp)
{
    fprintf(fp, "timestamp,dim,dtype,n_keys,n_unique_keys,group_bits,method,"
                "total_bytes,bytes_per_key,edges,"
                "trie_edges_before_suffix_collapse,dawg_edges_after_suffix_collapse,"
                "pre_suffix_total_bytes,pre_suffix_bytes_per_key,"
                "pre_suffix_normalized_bpk,post_suffix_normalized_bpk,"
                "suffix_collapse_saved_bytes_per_key,"
                "suffix_collapse_saved_normalized_bpk,suffix_collapse_saving_pct,"
                "morton_encode_s,sort_dedup_s,insert_s,finish_s,total_build_s\n");
}

static void write_csv_row(FILE *fp, const SweepResult &r, const char *timestamp)
{
    fprintf(fp, "%s,%u,%s,%zu,%zu,%u,%s,%zu,%.6f,%zu,"
                "%zu,%zu,%zu,%.6f,%.8f,%.8f,%.6f,%.8f,%.8f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f\n",
            timestamp, r.dim, r.dtype.c_str(), r.n_keys, r.n_unique_keys, r.group_bits,
            r.method.c_str(), r.total_bytes, r.bytes_per_key, r.edges,
            r.trie_edges_before_suffix_collapse, r.dawg_edges_after_suffix_collapse,
            r.pre_suffix_total_bytes, r.pre_suffix_bytes_per_key,
            r.pre_suffix_normalized_bpk, r.post_suffix_normalized_bpk,
            r.suffix_collapse_saved_bytes_per_key,
            r.suffix_collapse_saved_normalized_bpk, r.suffix_collapse_saving_pct,
            r.morton_encode_s, r.sort_dedup_s, r.insert_s, r.finish_s, r.total_build_s);
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> parse_csv_strings(const char *s)
{
    std::vector<std::string> out;
    std::string token;
    for (const char *p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!token.empty()) {
                out.push_back(token);
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

static std::string build_data_path(const std::string &data_dir, uint32_t dim,
                                   const std::string &dtype)
{
    char buf[512];
    snprintf(buf, sizeof(buf), "%s/msmarco_v2_corpus_%ud_%s.bin",
             data_dir.c_str(), dim, dtype.c_str());
    return buf;
}

static void print_usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s --data-dir <path> [options]\n"
            "\n"
            "Options:\n"
            "  --data-dir <path>       Embeddings directory (contains msmarco_v2_corpus_*d_*.bin)\n"
            "  --dims <csv>            Dimensions to sweep (default: 16,32,64,128,256,512,768,1024)\n"
            "  --dtypes <csv>          Data types to sweep (default: float32)\n"
            "  --n-keys <csv>          Dataset sizes to sweep (default: 1000,10000,100000,1000000)\n"
            "  --group-bits <csv>      GROUP_BITS values (default: 32,64,128,256,512,1024)\n"
            "  --float-shift <val>     Shift added to each coordinate (default: 0.5)\n"
            "  --output-csv <path>     Output CSV file (default: dawg_sweep_results.csv)\n"
            "  --dawgdic               Include dawgdic baseline for N <= 10000 only (32-bit overflow)\n"
            "\n",
            argv0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::string data_dir;
    std::vector<uint32_t> dims = {16, 32, 64, 128, 256, 512, 768, 1024};
    std::vector<std::string> dtypes = {"float32"};
    std::vector<size_t> n_keys_list = {1000, 10000, 100000, 1000000};
    std::vector<uint32_t> group_bits = {32, 64, 128, 256, 512, 1024};
    float float_shift = 0.5f;
    std::string output_csv = "dawg_sweep_results.csv";
    bool run_dawgdic = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--dims" && i + 1 < argc) {
            dims = parse_csv_uint32s(argv[++i]);
        } else if (arg == "--dtypes" && i + 1 < argc) {
            dtypes = parse_csv_strings(argv[++i]);
        } else if (arg == "--n-keys" && i + 1 < argc) {
            n_keys_list = parse_csv_sizes(argv[++i]);
        } else if (arg == "--group-bits" && i + 1 < argc) {
            group_bits = parse_csv_uint32s(argv[++i]);
        } else if (arg == "--float-shift" && i + 1 < argc) {
            float_shift = std::stof(argv[++i]);
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--dawgdic") {
            run_dawgdic = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (data_dir.empty()) {
        fprintf(stderr, "Error: --data-dir is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    ensure_large_stack();

    time_t now_t = time(nullptr);
    struct tm *tm_info = localtime(&now_t);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", tm_info);

    FILE *csv_fp = fopen(output_csv.c_str(), "w");
    if (!csv_fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", output_csv.c_str());
        return 1;
    }
    write_csv_header(csv_fp);

    const int hdr_w = 100;
    printf("\n%s\n", std::string(static_cast<size_t>(hdr_w), '=').c_str());
    printf("  CompactDawg Multi-Axis Storage & Insertion Sweep\n");
    printf("  %s\n", timestamp);
    printf("  Data dir: %s\n", data_dir.c_str());
    printf("  Float shift: %.4f\n", float_shift);
    printf("  Output CSV: %s\n", output_csv.c_str());
    printf("%s\n", std::string(static_cast<size_t>(hdr_w), '=').c_str());

    size_t max_n = *std::max_element(n_keys_list.begin(), n_keys_list.end());

    for (uint32_t dim : dims) {
        for (const auto &dtype : dtypes) {
            std::string path = build_data_path(data_dir, dim, dtype);
            bool is_fp16 = (dtype == "float16");
            size_t total_bits = static_cast<size_t>(dim) * (is_fp16 ? 16 : 32);

            printf("\n  === dim=%u  dtype=%s  key_length=%s bits ===\n",
                   dim, dtype.c_str(), comma_fmt(total_bits).c_str());

            EncodedDataset ds = load_encode_dedup(path.c_str(), dim, is_fp16,
                                                  max_n, float_shift);
            if (ds.keys.empty()) {
                fprintf(stderr, "  [skip] Cannot load %s (run scripts/truncate_embeddings.py)\n",
                        path.c_str());
                continue;
            }

            printf("  Read %s vectors -> %s unique keys  (encode: %s s, sort+dedup: %s s)\n",
                   comma_fmt(ds.vectors_read).c_str(), comma_fmt(ds.keys.size()).c_str(),
                   time_fmt(ds.encode_s).c_str(), time_fmt(ds.sort_dedup_s).c_str());

            printf("  Valid GROUP_BITS:");
            for (uint32_t gb : group_bits) {
                if (total_bits % gb == 0)
                    printf(" %u", gb);
            }
            printf("\n");

            for (size_t n : n_keys_list) {
                size_t actual_n = std::min(n, ds.keys.size());
                if (actual_n == 0)
                    continue;
                if (actual_n < n) {
                    fflush(stdout);
                    fprintf(stderr, "  [warn] Requested N=%s but only %s unique keys in file, using %s\n",
                            comma_fmt(n).c_str(), comma_fmt(ds.keys.size()).c_str(),
                            comma_fmt(actual_n).c_str());
                }

                std::vector<std::string> keys(
                    ds.keys.begin(),
                    ds.keys.begin() + static_cast<std::ptrdiff_t>(actual_n));

                fflush(stdout);
                fprintf(stderr, "  [bench] dim=%u dtype=%s N=%s ...\n",
                        dim, dtype.c_str(), comma_fmt(actual_n).c_str());

                std::vector<SweepResult> results;
                results.reserve(16);
                if (run_dawgdic && actual_n <= DAWGDIC_MAX_KEYS)
                    results.push_back(bench_dawgdic(keys));
                else if (run_dawgdic && actual_n > DAWGDIC_MAX_KEYS) {
                    fflush(stdout);
                    fprintf(stderr,
                            "  [dawgdic] Skipping N=%s (dawgdic only runs for N <= %s; larger N overflows "
                            "32-bit indices at high dim)\n",
                            comma_fmt(actual_n).c_str(),
                            comma_fmt(DAWGDIC_MAX_KEYS).c_str());
                }
                bench_all_group_bits(results, keys, total_bits, group_bits);

                const char *row_fmt = "    %-10s %11s %9s %13s %10s %10s %10s\n";
                printf("\n    N = %s keys\n", comma_fmt(actual_n).c_str());
                printf("    %s\n", std::string(84, '-').c_str());
                printf(row_fmt, "Method", "Total Size", "B/Key", "Edges",
                       "Insert(s)", "Finish(s)", "Total(s)");
                printf("    %s\n", std::string(84, '-').c_str());

                for (auto &r : results) {
                    r.dim = dim;
                    r.dtype = dtype;
                    r.n_keys = n;
                    r.n_unique_keys = actual_n;
                    r.morton_encode_s = ds.encode_s;
                    r.sort_dedup_s = ds.sort_dedup_s;

                    if (r.method == "dawgdic" && r.total_bytes == 0) {
                        printf(row_fmt, r.method.c_str(), "OVERFLOW", "--",
                               "--", time_fmt(r.insert_s).c_str(),
                               time_fmt(r.finish_s).c_str(),
                               time_fmt(r.total_build_s).c_str());
                    } else {
                        printf(row_fmt, r.method.c_str(),
                               size_fmt(r.total_bytes).c_str(),
                               bpk_fmt(r.bytes_per_key).c_str(),
                               comma_fmt(r.edges).c_str(),
                               time_fmt(r.insert_s).c_str(),
                               time_fmt(r.finish_s).c_str(),
                               time_fmt(r.total_build_s).c_str());
                    }

                    write_csv_row(csv_fp, r, timestamp);
                    fflush(stdout);
                }
                fflush(csv_fp);

                printf("    %s\n", std::string(84, '-').c_str());
                fflush(stdout);
            }
        }
    }

    fclose(csv_fp);
    printf("\n  Results written to %s\n\n", output_csv.c_str());
    return 0;
}
