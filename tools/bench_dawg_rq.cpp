#ifndef NUM_DIMENSIONS
#define NUM_DIMENSIONS 1024
#endif

#include "bench_common.h"
#include "compact_dawg.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

using hrc = std::chrono::high_resolution_clock;

struct RQResult {
    size_t n_keys;
    /// Range queries timed for this row (slice length = queries_per_step).
    size_t queries_run = 0;
    double sort_s;
    double build_s;
    size_t size_bytes;
    double avg_query_ms;
    /// Ground-truth verification: if `gt_queries == 0`, column shows n/a.
    size_t gt_queries = 0;
    size_t gt_mismatches = 0;
};

template <uint32_t BITS>
static bool matches_equal_ground_truth(const std::vector<std::string> &matches,
                                       const float *gt_base, size_t matches_per_query)
{
    std::vector<std::string> expected;
    expected.reserve(matches_per_query);
    for (size_t m = 0; m < matches_per_query; ++m) {
        data_point<NUM_DIMENSIONS> pt;
        const float *row = gt_base + m * NUM_DIMENSIONS;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            pt.set_float_coordinate(d, row[d]);
        std::string bits = encode_morton_bitstring(pt);
        const size_t rem = bits.size() % BITS;
        if (rem != 0)
            bits.append(BITS - rem, '0');
        expected.push_back(std::move(bits));
    }

    std::vector<std::string> got = matches;
    for (auto &g : got) {
        const size_t rem = g.size() % BITS;
        if (rem != 0)
            g.append(BITS - rem, '0');
    }

    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());
    return expected == got;
}

template <uint32_t BITS>
static std::vector<RQResult>
bench_one_group_bits(const std::vector<std::string> &all_keys_fileorder,
                     const std::vector<float> &lower_data, const std::vector<float> &upper_data,
                     const std::vector<float> *ground_truth,
                     const std::vector<size_t> &key_counts, size_t key_step,
                     size_t queries_per_step, size_t matches_per_query)
{
    std::vector<RQResult> results;

    for (size_t n : key_counts) {
        if (n > all_keys_fileorder.size()) {
            fprintf(stderr, "  [warn] N=%s exceeds available keys %s, skipping\n",
                    comma_fmt(n).c_str(), comma_fmt(all_keys_fileorder.size()).c_str());
            continue;
        }

        fflush(stdout);
        fprintf(stderr, "  [bench] GB=%u  N=%s  sorting...\n", BITS, comma_fmt(n).c_str());

        std::vector<std::string> keys(all_keys_fileorder.begin(),
                                      all_keys_fileorder.begin() + static_cast<ptrdiff_t>(n));
        auto ts0 = hrc::now();
        std::sort(keys.begin(), keys.end());
        auto ts1 = hrc::now();

        fprintf(stderr, "  [bench] GB=%u  N=%s  building...\n", BITS, comma_fmt(n).c_str());

        auto tb0 = hrc::now();
        CompactDawg<BITS> dawg;
        for (const auto &k : keys)
            dawg.Insert(k);
        dawg.Finish();
        auto tb1 = hrc::now();

        size_t step_idx = n / key_step - 1;
        size_t q_start = step_idx * queries_per_step;
        size_t q_end = q_start + queries_per_step;

        double total_query_ms = 0;
        size_t total_matches = 0;
        size_t gt_mismatch_queries = 0;

        fprintf(stderr, "  [bench] GB=%u  N=%s  querying %zu-%zu...\n", BITS, comma_fmt(n).c_str(),
                q_start, q_end - 1);

        for (size_t q = q_start; q < q_end; ++q) {
            data_point<NUM_DIMENSIONS> lo, hi;
            const float *lo_ptr = lower_data.data() + q * NUM_DIMENSIONS;
            const float *hi_ptr = upper_data.data() + q * NUM_DIMENSIONS;
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                lo.set_float_coordinate(d, lo_ptr[d]);
                hi.set_float_coordinate(d, hi_ptr[d]);
            }

            std::vector<std::string> matches;
            auto qt0 = hrc::now();
            dawg.SpatialRangeSearch(lo, hi, &matches);
            auto qt1 = hrc::now();

            total_query_ms += std::chrono::duration<double, std::milli>(qt1 - qt0).count();
            total_matches += matches.size();

            if (ground_truth) {
                const float *gt_base =
                    ground_truth->data() + q * matches_per_query * NUM_DIMENSIONS;
                if (!matches_equal_ground_truth<BITS>(matches, gt_base, matches_per_query)) {
                    gt_mismatch_queries++;
                    fprintf(stderr,
                            "  [GT mismatch] GB=%u N=%s global_query=%zu (local batch offset %zu)\n",
                            BITS, comma_fmt(n).c_str(), q, q - q_start);
                }
            }
        }

        double avg_results =
            static_cast<double>(total_matches) / static_cast<double>(queries_per_step);
        if (avg_results < 1.0) {
            fprintf(stderr,
                    "  [WARN] GB=%u N=%s: avg %.1f results/query — queries may be mismatched!\n",
                    BITS, comma_fmt(n).c_str(), avg_results);
        }

        RQResult r;
        r.n_keys = n;
        r.queries_run = queries_per_step;
        r.sort_s = std::chrono::duration<double>(ts1 - ts0).count();
        r.build_s = std::chrono::duration<double>(tb1 - tb0).count();
        r.size_bytes = dawg.size_in_bytes();
        r.avg_query_ms = total_query_ms / static_cast<double>(queries_per_step);
        if (ground_truth) {
            r.gt_queries = queries_per_step;
            r.gt_mismatches = gt_mismatch_queries;
        }
        results.push_back(r);
    }

    return results;
}

static void print_table(uint32_t group_bits, const std::vector<RQResult> &results)
{
    const char *fmt = "  %10s %10s %10s %10s %10s %14s %16s\n";
    const int line_w = 92;

    printf("\n  GROUP_BITS = %u\n", group_bits);
    printf("  %s\n", std::string(static_cast<size_t>(line_w), '-').c_str());
    printf(fmt, "N (keys)", "Queries", "Sort(s)", "Build(s)", "Size", "Avg Query(ms)", "GT check");
    printf("  %s\n", std::string(static_cast<size_t>(line_w), '-').c_str());

    for (const auto &r : results) {
        char query_ms[16];
        snprintf(query_ms, sizeof(query_ms), "%.2f", r.avg_query_ms);
        char gt_col[64];
        if (r.gt_queries == 0) {
            snprintf(gt_col, sizeof(gt_col), "n/a");
        } else if (r.gt_mismatches == 0) {
            snprintf(gt_col, sizeof(gt_col), "OK");
        } else {
            snprintf(gt_col, sizeof(gt_col), "FAIL %zu/%zu", r.gt_mismatches, r.gt_queries);
        }
        printf(fmt, comma_fmt(r.n_keys).c_str(), comma_fmt(r.queries_run).c_str(),
               time_fmt(r.sort_s).c_str(), time_fmt(r.build_s).c_str(),
               size_fmt(r.size_bytes).c_str(), query_ms, gt_col);
    }

    printf("  %s\n", std::string(static_cast<size_t>(line_w), '-').c_str());
}

/// Write one CSV field (quote only if needed).
static void fputc_csv_field(FILE *fp, const char *s)
{
    if (!s) {
        return;
    }
    bool need_quote = false;
    for (const char *p = s; *p; ++p) {
        if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
            need_quote = true;
            break;
        }
    }
    if (!need_quote) {
        fputs(s, fp);
        return;
    }
    fputc('"', fp);
    for (const char *p = s; *p; ++p) {
        if (*p == '"')
            fputs("\"\"", fp);
        else
            fputc(*p, fp);
    }
    fputc('"', fp);
}

static int write_rq_results_csv(
    const char *path, const char *timestamp, const char *data_path, const char *lower_path,
    const char *upper_path, const char *ground_truth_path, int dimensions, size_t key_step,
    size_t queries_per_step, size_t matches_per_query, size_t total_points, double encode_s,
    const std::vector<std::pair<uint32_t, std::vector<RQResult>>> &rows_by_gb)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open --output-csv %s\n", path);
        return 1;
    }
    fprintf(fp,
            "timestamp,dimensions,total_points,key_step,queries_per_step,matches_per_query,encode_"
            "s,group_bits,n_keys,queries_run,sort_s,build_s,size_bytes,avg_query_ms,gt_queries,gt_"
            "mismatches,data_file,lower_file,upper_file,ground_truth_file\n");
    for (const auto &gb_row : rows_by_gb) {
        uint32_t gb = gb_row.first;
        for (const auto &r : gb_row.second) {
            fprintf(fp, "%s,%d,%zu,%zu,%zu,%zu,%.9g,%u,%zu,%zu,%.9g,%.9g,%zu,%.9g,%zu,%zu,", timestamp,
                    dimensions, total_points, key_step, queries_per_step, matches_per_query, encode_s,
                    gb, r.n_keys, r.queries_run, r.sort_s, r.build_s, r.size_bytes, r.avg_query_ms,
                    r.gt_queries, r.gt_mismatches);
            fputc_csv_field(fp, data_path);
            fputc(',', fp);
            fputc_csv_field(fp, lower_path);
            fputc(',', fp);
            fputc_csv_field(fp, upper_path);
            fputc(',', fp);
            fputc_csv_field(fp, ground_truth_path ? ground_truth_path : "");
            fputc('\n', fp);
        }
    }
    fclose(fp);
    fprintf(stderr, "  [csv] Wrote %s\n", path);
    return 0;
}

typedef std::vector<RQResult> (*bench_fn_t)(const std::vector<std::string> &,
                                            const std::vector<float> &, const std::vector<float> &,
                                            const std::vector<float> *,
                                            const std::vector<size_t> &, size_t, size_t, size_t);

template <uint32_t BITS>
static std::vector<RQResult>
bench_dispatch(const std::vector<std::string> &keys, const std::vector<float> &lo,
               const std::vector<float> &hi, const std::vector<float> *gt,
               const std::vector<size_t> &kc, size_t ks, size_t qps, size_t mpq)
{
    return bench_one_group_bits<BITS>(keys, lo, hi, gt, kc, ks, qps, mpq);
}

static bench_fn_t get_bench_fn(uint32_t gb)
{
    switch (gb) {
        case 16:
            return bench_dispatch<16>;
        case 32:
            return bench_dispatch<32>;
        case 64:
            return bench_dispatch<64>;
        case 128:
            return bench_dispatch<128>;
        case 256:
            return bench_dispatch<256>;
        case 512:
            return bench_dispatch<512>;
        case 1024:
            return bench_dispatch<1024>;
        default:
            return nullptr;
    }
}

static size_t auto_detect_queries_per_step(const char *lower_path, size_t max_n, size_t key_step)
{
    std::ifstream f(lower_path, std::ios::binary);
    if (!f)
        return 0;
    f.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(f.tellg());
    size_t floats_in_file = file_bytes / sizeof(float);
    size_t queries_in_file = floats_in_file / NUM_DIMENSIONS;
    size_t num_steps = max_n / key_step;
    if (num_steps == 0)
        return 0;
    return queries_in_file / num_steps;
}

static int print_usage(const char *prog, int exit_code)
{
    fprintf(
        stderr,
        "Usage: %s <data_file> <lower_file> <upper_file> [options]\n"
        "\n"
        "  <data_file>    Raw float32 dataset (e.g. 1024d-uniq-100k.bin)\n"
        "  <lower_file>   Query lower bounds (from gen_rq_queries_exact10_1024d)\n"
        "  <upper_file>   Query upper bounds (from gen_rq_queries_exact10_1024d)\n"
        "\n"
        "Options:\n"
        "  --group-bits <a,b,...>    GROUP_BITS to benchmark (default: 32)\n"
        "  --n-keys <a,b,c,...>      Key counts to test (default: 1000,10000,50000,100000)\n"
        "  --key-step <N>            Key step used in query generation (default: 1000)\n"
        "  --queries-per-step <N>    Queries per key count (default: auto-detect from file)\n"
        "  --matches-per-query <N>   Matches per query / GT width (default: 10)\n"
        "  --total-points <N>        Points to load from data file (default: 100000)\n"
        "  --ground-truth <path>     Binary float32 [Q][M][1024] from gen_rq_queries_exact10_1024d\n"
        "                            (M = --matches-per-query). Adds GT check column; exit 1 if any mismatch.\n"
        "  --output-csv <path>       Write one row per (GROUP_BITS, N) for plotting (truncates file)\n",
        prog);
    return exit_code;
}

int main(int argc, char **argv)
{
    ensure_large_stack();

    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        return print_usage(argv[0], 0);
    }

    if (argc < 4) {
        return print_usage(argv[0], 1);
    }

    const char *data_path = argv[1];
    const char *lower_path = argv[2];
    const char *upper_path = argv[3];

    std::vector<uint32_t> group_bits_list = {32};
    std::vector<size_t> key_counts = {1000, 10000, 50000, 100000};
    size_t key_step = 1000;
    size_t queries_per_step = 0;
    size_t matches_per_query = 10;
    size_t total_points = 100000;
    bool qps_explicit = false;
    const char *ground_truth_path = nullptr;
    const char *output_csv_path = nullptr;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--group-bits" && i + 1 < argc) {
            group_bits_list = parse_csv_uint32s(argv[++i]);
        } else if (arg == "--n-keys" && i + 1 < argc) {
            key_counts = parse_csv_sizes(argv[++i]);
        } else if (arg == "--key-step" && i + 1 < argc) {
            key_step = std::stoul(argv[++i]);
        } else if (arg == "--queries-per-step" && i + 1 < argc) {
            queries_per_step = std::stoul(argv[++i]);
            qps_explicit = true;
        } else if (arg == "--matches-per-query" && i + 1 < argc) {
            matches_per_query = std::stoul(argv[++i]);
        } else if (arg == "--total-points" && i + 1 < argc) {
            total_points = std::stoul(argv[++i]);
        } else if (arg == "--ground-truth" && i + 1 < argc) {
            ground_truth_path = argv[++i];
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    size_t max_n = *std::max_element(key_counts.begin(), key_counts.end());

    if (!qps_explicit) {
        queries_per_step = auto_detect_queries_per_step(lower_path, max_n, key_step);
        if (queries_per_step == 0) {
            fprintf(stderr,
                    "Error: Could not auto-detect queries-per-step from %s.\n"
                    "  Specify --queries-per-step explicitly.\n",
                    lower_path);
            return 1;
        }
        fprintf(stderr, "  [auto] Detected %zu queries per step from fixture file size\n",
                queries_per_step);
    }

    size_t total_queries = (max_n / key_step) * queries_per_step;

    fprintf(stderr, "  [load] Reading %s points from %s...\n", comma_fmt(total_points).c_str(),
            data_path);
    std::vector<float> dataset;
    if (!load_float_file(data_path, dataset, total_points * NUM_DIMENSIONS))
        return 1;

    fprintf(stderr, "  [load] Reading %s queries from bounds files...\n",
            comma_fmt(total_queries).c_str());
    std::vector<float> lower_data, upper_data;
    if (!load_float_file(lower_path, lower_data, total_queries * NUM_DIMENSIONS))
        return 1;
    if (!load_float_file(upper_path, upper_data, total_queries * NUM_DIMENSIONS))
        return 1;

    std::vector<float> ground_truth_data;
    const std::vector<float> *ground_truth_ptr = nullptr;
    if (ground_truth_path) {
        size_t gt_floats = total_queries * matches_per_query * NUM_DIMENSIONS;
        fprintf(stderr, "  [load] Reading ground truth %s floats from %s...\n",
                comma_fmt(gt_floats).c_str(), ground_truth_path);
        if (!load_float_file(ground_truth_path, ground_truth_data, gt_floats))
            return 1;
        ground_truth_ptr = &ground_truth_data;
    }

    fprintf(stderr, "  [load] Encoding %s points to morton...\n", comma_fmt(total_points).c_str());
    auto enc_start = hrc::now();
    std::vector<std::string> all_keys_fileorder;
    all_keys_fileorder.reserve(total_points);
    for (size_t i = 0; i < total_points; ++i) {
        data_point<NUM_DIMENSIONS> pt;
        const float *row = dataset.data() + i * NUM_DIMENSIONS;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            pt.set_float_coordinate(d, row[d]);
        all_keys_fileorder.push_back(encode_morton_bitstring(pt));
    }
    auto enc_end = hrc::now();
    double enc_s = std::chrono::duration<double>(enc_end - enc_start).count();

    time_t now = time(nullptr);
    struct tm *tm_info = localtime(&now);
    char date_buf[32];
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%d %H:%M", tm_info);

    const int header_w = 78;
    printf("\n%s\n", std::string(static_cast<size_t>(header_w), '=').c_str());
    printf("  CompactDawg Range Query Benchmark\n");
    printf("  %s\n", date_buf);
    printf("  Dataset: %s (%s points, %d dimensions)\n", data_path, comma_fmt(total_points).c_str(),
           NUM_DIMENSIONS);
    printf("  Queries: %zu per key count, %zu expected matches each\n", queries_per_step,
           matches_per_query);
    if (ground_truth_path)
        printf("  Ground truth: %s\n", ground_truth_path);
    printf("  Encode: %s s\n", time_fmt(enc_s).c_str());
    printf("%s\n", std::string(static_cast<size_t>(header_w), '=').c_str());

    int verify_exit_code = 0;
    std::vector<std::pair<uint32_t, std::vector<RQResult>>> csv_accum;
    if (output_csv_path)
        csv_accum.reserve(group_bits_list.size());

    for (uint32_t gb : group_bits_list) {
        bench_fn_t fn = get_bench_fn(gb);
        if (!fn) {
            fprintf(stderr, "Error: Unsupported --group-bits %u\n", gb);
            return 1;
        }
        auto rq_results =
            fn(all_keys_fileorder, lower_data, upper_data, ground_truth_ptr, key_counts, key_step,
               queries_per_step, matches_per_query);
        print_table(gb, rq_results);
        if (ground_truth_ptr) {
            for (const auto &r : rq_results) {
                if (r.gt_mismatches > 0)
                    verify_exit_code = 1;
            }
        }
        if (output_csv_path)
            csv_accum.emplace_back(gb, std::move(rq_results));
    }
    printf("\n");

    if (output_csv_path) {
        if (write_rq_results_csv(output_csv_path, date_buf, data_path, lower_path, upper_path,
                                 ground_truth_path, NUM_DIMENSIONS, key_step, queries_per_step,
                                 matches_per_query, total_points, enc_s, csv_accum) != 0)
            return 1;
    }

    if (verify_exit_code != 0)
        fprintf(stderr, "Error: One or more GT checks failed (see table and stderr).\n");

    return verify_exit_code;
}
