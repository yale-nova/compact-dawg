#include "bench_common.h"
#include "compact_dawg.h"

#include <chrono>
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

struct BenchResult {
    std::string method;
    size_t n_keys;
    size_t total_bytes;
    double bytes_per_key;
    size_t edges;
    double insert_s;
    double finish_s;
    double total_s;
    bool failed = false;
};

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

static BenchResult bench_dawgdic(const std::vector<std::string> &keys)
{
    BenchResult r;
    r.method = "dawgdic";
    r.n_keys = keys.size();

    auto t0 = hrc::now();
    dawgdic::DawgBuilder builder;
    bool insert_ok = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!builder.Insert(keys[i].c_str(), static_cast<dawgdic::SizeType>(keys[i].length()), 0)) {
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
    r.total_s = std::chrono::duration<double>(t2 - t0).count();

    if (!build_ok) {
        r.failed = true;
        return r;
    }

    r.total_bytes = dic.total_size();
    r.bytes_per_key = static_cast<double>(r.total_bytes) / static_cast<double>(r.n_keys);
    r.edges = dic.size();

    if (r.bytes_per_key < 1.0) {
        fprintf(stderr, "  [dawgdic] Likely 32-bit overflow: %.1f B/key is implausibly small\n",
                r.bytes_per_key);
        r.failed = true;
    }

    return r;
}

template <uint32_t BITS> static BenchResult bench_compact(const std::vector<std::string> &keys)
{
    BenchResult r;
    char name[16];
    snprintf(name, sizeof(name), "CD-%u", BITS);
    r.method = name;
    r.n_keys = keys.size();

    auto t0 = hrc::now();
    CompactDawg<BITS> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    auto t1 = hrc::now();
    dawg.Finish();
    auto t2 = hrc::now();

    r.total_bytes = dawg.size_in_bytes();
    r.bytes_per_key = static_cast<double>(r.total_bytes) / static_cast<double>(r.n_keys);
    r.edges = dawg.get_total_edges();
    r.insert_s = std::chrono::duration<double>(t1 - t0).count();
    r.finish_s = std::chrono::duration<double>(t2 - t1).count();
    r.total_s = std::chrono::duration<double>(t2 - t0).count();
    return r;
}

template <uint32_t... Bits>
static void collect_compact(std::vector<BenchResult> &out, const std::vector<std::string> &keys)
{
    (out.push_back(bench_compact<Bits>(keys)), ...);
}

template <uint32_t BITS> static BenchResult bench_compact_pc(const std::vector<std::string> &keys)
{
    BenchResult r;
    char name[16];
    snprintf(name, sizeof(name), "PC-%u", BITS);
    r.method = name;
    r.n_keys = keys.size();

    auto t0 = hrc::now();
    CompactDawg<BITS, true> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    auto t1 = hrc::now();
    dawg.Finish();
    auto t2 = hrc::now();

    r.total_bytes = dawg.size_in_bytes();
    r.bytes_per_key = static_cast<double>(r.total_bytes) / static_cast<double>(r.n_keys);
    r.edges = dawg.get_total_edges();
    r.insert_s = std::chrono::duration<double>(t1 - t0).count();
    r.finish_s = std::chrono::duration<double>(t2 - t1).count();
    r.total_s = std::chrono::duration<double>(t2 - t0).count();
    return r;
}

template <uint32_t... Bits>
static void collect_compact_pc(std::vector<BenchResult> &out, const std::vector<std::string> &keys)
{
    (out.push_back(bench_compact_pc<Bits>(keys)), ...);
}

static const char *basename_of(const char *path)
{
    const char *p = strrchr(path, '/');
    return p ? p + 1 : path;
}

static int print_usage(const char *prog, int exit_code)
{
    fprintf(stderr, "Usage: %s <input_keys_bin> [--no-dawgdic] [N [N2 ...]]\n", prog);
    fprintf(stderr, "\n  Loads all keys from the binary file, sorts and deduplicates.\n");
    fprintf(stderr, "  With no N arguments, benchmarks once using every unique key.\n");
    fprintf(stderr, "  With N arguments, runs once per N (subset of first N keys).\n");
    fprintf(stderr, "  CD-N = CompactDawg (GROUP_BITS=N), PC-N = Path-Compressed\n");
    fprintf(stderr, "  --no-dawgdic  Skip dawgdic benchmark\n");
    return exit_code;
}

static void print_section(size_t n, const std::vector<BenchResult> &results)
{
    size_t dawgdic_bytes = 0;
    bool dawgdic_ok = false;
    for (const auto &r : results) {
        if (r.method == "dawgdic" && !r.failed) {
            dawgdic_bytes = r.total_bytes;
            dawgdic_ok = true;
        }
    }

    const char *fmt = "  %-10s %11s %9s %13s %10s %10s %10s %10s\n";
    const int line_w = 96;

    printf("\n  N = %s keys\n", comma_fmt(n).c_str());
    printf("  %s\n", std::string(static_cast<size_t>(line_w), '-').c_str());
    printf(fmt, "Method", "Total Size", "B/Key", "Edges", "Insert(s)", "Finish(s)", "Total(s)",
           "Size Ratio");
    printf("  %s\n", std::string(static_cast<size_t>(line_w), '-').c_str());

    for (const auto &r : results) {
        if (r.failed) {
            printf("  %-10s %11s %9s %13s %10s %10s %10s %10s\n", r.method.c_str(), "FAILED", "--",
                   "--", time_fmt(r.insert_s).c_str(), time_fmt(r.finish_s).c_str(),
                   time_fmt(r.total_s).c_str(), "--");
            continue;
        }

        const char *ratio_str = "--";
        char ratio_buf[16];
        if (dawgdic_ok && dawgdic_bytes > 0) {
            double ratio = static_cast<double>(r.total_bytes) / static_cast<double>(dawgdic_bytes);
            snprintf(ratio_buf, sizeof(ratio_buf), "%.2fx", ratio);
            ratio_str = ratio_buf;
        }

        printf(fmt, r.method.c_str(), size_fmt(r.total_bytes).c_str(),
               bpk_fmt(r.bytes_per_key).c_str(), comma_fmt(r.edges).c_str(),
               time_fmt(r.insert_s).c_str(), time_fmt(r.finish_s).c_str(),
               time_fmt(r.total_s).c_str(), ratio_str);
    }

    printf("  %s\n", std::string(static_cast<size_t>(line_w), '-').c_str());
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        return print_usage(argv[0], 1);
    }

    if (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        return print_usage(argv[0], 0);
    }

    const char *input_path = argv[1];

    std::vector<size_t> entry_counts;
    bool skip_dawgdic = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-dawgdic") {
            skip_dawgdic = true;
        } else {
            entry_counts.push_back(std::stoul(argv[i]));
        }
    }

    std::ifstream in_file(input_path, std::ios::binary);
    if (!in_file) {
        fprintf(stderr, "Error: Could not open %s\n", input_path);
        return 1;
    }

    static constexpr int NUM_DIMS = 1024;
    static constexpr int TRIE_DEPTH = 32;
    const size_t total_bits = static_cast<size_t>(NUM_DIMS) * TRIE_DEPTH;
    const size_t bytes_per_point = (total_bits + 7) / 8;
    std::vector<uint8_t> buffer(bytes_per_point);

    std::vector<std::string> all_keys;
    auto read_start = hrc::now();
    while (in_file.read(reinterpret_cast<char *>(buffer.data()),
                        static_cast<std::streamsize>(bytes_per_point))) {
        std::string bitstring;
        bitstring.reserve(total_bits);
        for (size_t i = 0; i < total_bits; ++i) {
            bool bit = (buffer[i / 8] >> (7 - (i % 8))) & 1;
            bitstring += (bit ? '1' : '0');
        }
        all_keys.push_back(std::move(bitstring));
    }
    in_file.close();
    auto read_end = hrc::now();
    double read_s = std::chrono::duration<double>(read_end - read_start).count();

    if (all_keys.empty()) {
        fprintf(stderr, "Error: No keys read from file.\n");
        return 1;
    }

    if (entry_counts.empty()) {
        entry_counts.push_back(all_keys.size());
    } else {
        std::sort(entry_counts.begin(), entry_counts.end());
    }

    auto sort_start = hrc::now();
    std::sort(all_keys.begin(), all_keys.end());
    all_keys.erase(std::unique(all_keys.begin(), all_keys.end()), all_keys.end());
    auto sort_end = hrc::now();
    double sort_s = std::chrono::duration<double>(sort_end - sort_start).count();

    time_t now = time(nullptr);
    struct tm *tm_info = localtime(&now);
    char date_buf[32];
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%d %H:%M", tm_info);

    const int header_w = 98;
    printf("\n%s\n", std::string(static_cast<size_t>(header_w), '=').c_str());
    printf("  CompactDawg Storage & Performance Benchmark\n");
    printf("  %s\n", date_buf);
    printf("  Dataset: %s\n", basename_of(input_path));
    printf("  Dimensions: %s  |  Depth: %d bits  |  Key length: %s bits\n",
           comma_fmt(NUM_DIMS).c_str(), TRIE_DEPTH, comma_fmt(total_bits).c_str());
    printf("  Unique keys loaded: %s  |  Read: %s s  |  Sort+dedup: %s s\n",
           comma_fmt(all_keys.size()).c_str(), time_fmt(read_s).c_str(), time_fmt(sort_s).c_str());
    printf("  CD-N = CompactDawg (GROUP_BITS=N)  |  PC-N = Path-Compressed CD-N\n");
    printf("  Size Ratio = size / dawgdic size\n");
    printf("%s\n", std::string(static_cast<size_t>(header_w), '=').c_str());

    for (size_t n : entry_counts) {
        if (n > all_keys.size()) {
            fprintf(stderr, "  [warn] Requested N=%s but only %s unique keys available, using %s\n",
                    comma_fmt(n).c_str(), comma_fmt(all_keys.size()).c_str(),
                    comma_fmt(all_keys.size()).c_str());
            n = all_keys.size();
        }

        fflush(stdout);
        fprintf(stderr, "  [bench] N=%s ...\n", comma_fmt(n).c_str());

        std::vector<std::string> keys(all_keys.begin(),
                                      all_keys.begin() + static_cast<std::ptrdiff_t>(n));

        std::vector<BenchResult> results;
        results.reserve(16);
        if (!skip_dawgdic)
            results.push_back(bench_dawgdic(keys));
        collect_compact<16, 32, 64, 128, 256, 512, 1024>(results, keys);
        collect_compact_pc<16, 32, 64, 128, 256, 512, 1024>(results, keys);

        print_section(n, results);
    }

    printf("\n");
    return 0;
}
