/**
 * gen_rq_queries_exact10_1024d.cpp
 *
 * Generate range-query fixtures for 1024D float32 datasets.
 *
 * This tool creates:
 *   1) lower bounds binary file  (float32, shape [Q][1024])
 *   2) upper bounds binary file  (float32, shape [Q][1024])
 *   3) ground truth binary file  (float32, shape [Q][M][1024])
 *
 * Where:
 *   - Q = num_tries * queries_per_trie
 *   - M = matches_per_query
 *
 * Query ordering is fixed and deterministic:
 *   - trie-size-major, then query-minor
 *   - global_query_id = trie_idx * queries_per_trie + local_query_idx
 *   - trie size n = (trie_idx + 1) * trie_step
 *
 * For each query at trie size n, this generator repeatedly samples M seed
 * points from the first n rows, builds a bounding box from those seed points,
 * then validates that the query returns exactly M matches over the same prefix
 * [0, n). If exact-M is not found within max_attempts_per_query, the tool
 * exits with non-zero status and prints failure diagnostics.
 *
 * Example:
 *   ./build/gen_rq_queries_exact10_1024d \
 *       --data-file ./1024d-uniq-100k.bin \
 *       --output-lower ./tests/testdata/1024d_rq_lower_1000_to_100000_q10.bin \
 *       --output-upper ./tests/testdata/1024d_rq_upper_1000_to_100000_q10.bin \
 *       --output-ground-truth ./tests/testdata/1024d_rq_ground_truth_1000_to_100000_q10.bin \
 *       --seed 42
 *
 * Smoke test:
 *   ./build/gen_rq_queries_exact10_1024d \
 *       --data-file /tmp/smoke_1024d.bin \
 *       --total-points 300 \
 *       --trie-step 100 \
 *       --num-tries 3 \
 *       --queries-per-trie 2 \
 *       --matches-per-query 3 \
 *       --max-attempts-per-query 5000 \
 *       --seed 7
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::size_t kDimensions = 1024;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr const char *kDefaultDataFile = "./1024d-uniq-100k.bin";
constexpr const char *kDefaultOutputLower =
    "./tests/testdata/1024d_rq_lower_1000_to_100000_q10.bin";
constexpr const char *kDefaultOutputUpper =
    "./tests/testdata/1024d_rq_upper_1000_to_100000_q10.bin";
constexpr const char *kDefaultOutputGroundTruth =
    "./tests/testdata/1024d_rq_ground_truth_1000_to_100000_q10.bin";

struct Config {
    std::string data_file = kDefaultDataFile;
    std::string output_lower = kDefaultOutputLower;
    std::string output_upper = kDefaultOutputUpper;
    std::string output_ground_truth = kDefaultOutputGroundTruth;

    std::size_t total_points = 100000;
    std::size_t trie_step = 1000;
    std::size_t num_tries = 100;
    std::size_t queries_per_trie = 10;
    std::size_t matches_per_query = 10;
    std::size_t max_attempts_per_query = 50000;

    bool seed_provided = false;
    uint64_t seed = 0;
};

void print_usage(const char *prog)
{
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --data-file <path>               Default: " << kDefaultDataFile << "\n"
        << "  --output-lower <path>            Default: tests/testdata/"
           "1024d_rq_lower_1000_to_100000_q10.bin\n"
        << "  --output-upper <path>            Default: tests/testdata/"
           "1024d_rq_upper_1000_to_100000_q10.bin\n"
        << "  --output-ground-truth <path>     Default: tests/testdata/"
           "1024d_rq_ground_truth_1000_to_100000_q10.bin\n"
        << "  --total-points <N>               Default: 100000\n"
        << "  --trie-step <N>                  Default: 1000\n"
        << "  --num-tries <N>                  Default: 100\n"
        << "  --queries-per-trie <N>           Default: 10\n"
        << "  --matches-per-query <N>          Default: 10\n"
        << "  --max-attempts-per-query <N>     Default: 50000\n"
        << "  --seed <u64>                     Optional fixed seed\n"
        << "  --help                           Show this help\n";
}

bool parse_u64(const std::string &text, uint64_t &value)
{
    try {
        std::size_t parsed = 0;
        const unsigned long long raw = std::stoull(text, &parsed, 10);
        if (parsed != text.size()) {
            return false;
        }
        value = static_cast<uint64_t>(raw);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_size_t_arg(const std::string &arg_name, const std::string &arg_value, std::size_t &out)
{
    uint64_t parsed = 0;
    if (!parse_u64(arg_value, parsed)) {
        std::cerr << "Error: Invalid value for " << arg_name << ": " << arg_value << std::endl;
        return false;
    }
    if (parsed == 0) {
        std::cerr << "Error: " << arg_name << " must be > 0" << std::endl;
        return false;
    }
    out = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_args(int argc, char **argv, Config &cfg)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--data-file") {
            const char *v = need_value(arg);
            if (v == nullptr)
                return false;
            cfg.data_file = v;
        } else if (arg == "--output-lower") {
            const char *v = need_value(arg);
            if (v == nullptr)
                return false;
            cfg.output_lower = v;
        } else if (arg == "--output-upper") {
            const char *v = need_value(arg);
            if (v == nullptr)
                return false;
            cfg.output_upper = v;
        } else if (arg == "--output-ground-truth") {
            const char *v = need_value(arg);
            if (v == nullptr)
                return false;
            cfg.output_ground_truth = v;
        } else if (arg == "--total-points") {
            const char *v = need_value(arg);
            if (v == nullptr || !parse_size_t_arg(arg, v, cfg.total_points))
                return false;
        } else if (arg == "--trie-step") {
            const char *v = need_value(arg);
            if (v == nullptr || !parse_size_t_arg(arg, v, cfg.trie_step))
                return false;
        } else if (arg == "--num-tries") {
            const char *v = need_value(arg);
            if (v == nullptr || !parse_size_t_arg(arg, v, cfg.num_tries))
                return false;
        } else if (arg == "--queries-per-trie") {
            const char *v = need_value(arg);
            if (v == nullptr || !parse_size_t_arg(arg, v, cfg.queries_per_trie))
                return false;
        } else if (arg == "--matches-per-query") {
            const char *v = need_value(arg);
            if (v == nullptr || !parse_size_t_arg(arg, v, cfg.matches_per_query))
                return false;
        } else if (arg == "--max-attempts-per-query") {
            const char *v = need_value(arg);
            if (v == nullptr || !parse_size_t_arg(arg, v, cfg.max_attempts_per_query))
                return false;
        } else if (arg == "--seed") {
            const char *v = need_value(arg);
            if (v == nullptr)
                return false;
            uint64_t parsed_seed = 0;
            if (!parse_u64(v, parsed_seed)) {
                std::cerr << "Error: Invalid value for --seed: " << v << std::endl;
                return false;
            }
            cfg.seed_provided = true;
            cfg.seed = parsed_seed;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

std::string format_seconds(double seconds)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << "s";
    return out.str();
}

double elapsed_seconds(const std::chrono::steady_clock::time_point &start)
{
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

bool validate_config(const Config &cfg)
{
    if (kDimensions != 1024) {
        std::cerr << "Error: NUM_DIMENSIONS must be 1024." << std::endl;
        return false;
    }
    if (cfg.matches_per_query > cfg.trie_step) {
        std::cerr << "Error: matches-per-query (" << cfg.matches_per_query
                  << ") must be <= trie-step (" << cfg.trie_step << ")." << std::endl;
        return false;
    }
    const uint64_t max_prefix = static_cast<uint64_t>(cfg.num_tries) * cfg.trie_step;
    if (max_prefix > cfg.total_points) {
        std::cerr << "Error: num-tries * trie-step = " << max_prefix
                  << " exceeds total-points = " << cfg.total_points << std::endl;
        return false;
    }
    return true;
}

bool load_dataset(const Config &cfg, std::vector<float> &dataset, std::size_t &available_points)
{
    std::ifstream in(cfg.data_file, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Unable to open data file: " << cfg.data_file << std::endl;
        return false;
    }

    in.seekg(0, std::ios::end);
    const auto file_bytes = static_cast<uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    const uint64_t record_bytes = kDimensions * kFloatBytes;
    if (file_bytes % record_bytes != 0) {
        std::cerr << "Error: Data file size " << file_bytes
                  << " is not divisible by one 1024D float32 record (" << record_bytes << " bytes)."
                  << std::endl;
        return false;
    }

    available_points = static_cast<std::size_t>(file_bytes / record_bytes);
    if (available_points < cfg.total_points) {
        std::cerr << "Error: Requested total-points=" << cfg.total_points
                  << " but data file contains only " << available_points << " points." << std::endl;
        return false;
    }

    const std::size_t floats_to_read = cfg.total_points * kDimensions;
    dataset.resize(floats_to_read);

    const auto bytes_to_read = static_cast<std::streamsize>(floats_to_read * sizeof(float));
    in.read(reinterpret_cast<char *>(dataset.data()), bytes_to_read);
    if (in.gcount() != bytes_to_read) {
        std::cerr << "Error: Short read from data file. Expected " << bytes_to_read << " bytes, got "
                  << in.gcount() << " bytes." << std::endl;
        return false;
    }

    return true;
}

std::vector<std::size_t>
sample_unique_indices(std::size_t n, std::size_t k, std::mt19937_64 &rng)
{
    std::unordered_set<std::size_t> sampled;
    sampled.reserve(k * 2);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    while (sampled.size() < k) {
        sampled.insert(dist(rng));
    }
    return std::vector<std::size_t>(sampled.begin(), sampled.end());
}

void build_bounds(const std::vector<float> &dataset, const std::vector<std::size_t> &seed_indices,
                  std::vector<float> &lower, std::vector<float> &upper)
{
    std::fill(lower.begin(), lower.end(), std::numeric_limits<float>::max());
    std::fill(upper.begin(), upper.end(), std::numeric_limits<float>::lowest());

    for (std::size_t i = 0; i < seed_indices.size(); ++i) {
        const std::size_t row_idx = seed_indices[i];
        const float *row = dataset.data() + row_idx * kDimensions;
        for (std::size_t d = 0; d < kDimensions; ++d) {
            lower[d] = std::min(lower[d], row[d]);
            upper[d] = std::max(upper[d], row[d]);
        }
    }
}

void build_dimension_order(const std::vector<float> &lower, const std::vector<float> &upper,
                           std::vector<std::size_t> &dim_order)
{
    std::iota(dim_order.begin(), dim_order.end(), 0);
    std::sort(dim_order.begin(), dim_order.end(),
              [&](std::size_t lhs, std::size_t rhs) -> bool {
                  const float w_lhs = upper[lhs] - lower[lhs];
                  const float w_rhs = upper[rhs] - lower[rhs];
                  if (w_lhs == w_rhs) {
                      return lhs < rhs;
                  }
                  return w_lhs < w_rhs;
              });
}

std::size_t count_matches_in_prefix(const std::vector<float> &dataset, std::size_t n,
                                    const std::vector<float> &lower, const std::vector<float> &upper,
                                    const std::vector<std::size_t> &dim_order,
                                    std::size_t stop_after_count,
                                    std::vector<std::size_t> *match_indices)
{
    if (match_indices != nullptr) {
        match_indices->clear();
    }

    std::size_t count = 0;
    for (std::size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float *row = dataset.data() + row_idx * kDimensions;

        bool in_range = true;
        for (std::size_t k = 0; k < dim_order.size(); ++k) {
            const std::size_t d = dim_order[k];
            const float v = row[d];
            if (v < lower[d] || v > upper[d]) {
                in_range = false;
                break;
            }
        }

        if (!in_range) {
            continue;
        }

        ++count;
        if (match_indices != nullptr && count <= stop_after_count) {
            match_indices->push_back(row_idx);
        }
        if (count > stop_after_count) {
            break;
        }
    }

    return count;
}

bool write_binary_f32(const std::string &path, const std::vector<float> &payload)
{
    const std::filesystem::path out_path(path);
    std::error_code ec;
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            std::cerr << "Error: Failed to create parent directories for " << path << ": "
                      << ec.message() << std::endl;
            return false;
        }
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Error: Unable to open output file for writing: " << path << std::endl;
        return false;
    }

    const std::streamsize bytes = static_cast<std::streamsize>(payload.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(payload.data()), bytes);
    if (!out) {
        std::cerr << "Error: Failed while writing output file: " << path << std::endl;
        return false;
    }

    return true;
}

} // namespace

int main(int argc, char **argv)
{
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) ? 0
                                                                                                  : 1;
    }
    if (!validate_config(cfg)) {
        return 1;
    }

    const std::size_t total_queries = cfg.num_tries * cfg.queries_per_trie;
    const uint64_t expected_lower_bytes = static_cast<uint64_t>(total_queries) * kDimensions * kFloatBytes;
    const uint64_t expected_upper_bytes = expected_lower_bytes;
    const uint64_t expected_gt_bytes =
        static_cast<uint64_t>(total_queries) * cfg.matches_per_query * kDimensions * kFloatBytes;

    uint64_t seed_value = cfg.seed;
    if (!cfg.seed_provided) {
        seed_value = std::random_device{}();
    }
    std::mt19937_64 rng(seed_value);

    std::cout << "[START] gen_rq_queries_exact10_1024d" << std::endl;
    std::cout << "[START] config:"
              << " data_file=" << cfg.data_file << " total_points=" << cfg.total_points
              << " trie_step=" << cfg.trie_step << " num_tries=" << cfg.num_tries
              << " queries_per_trie=" << cfg.queries_per_trie
              << " matches_per_query=" << cfg.matches_per_query
              << " max_attempts_per_query=" << cfg.max_attempts_per_query << std::endl;
    std::cout << "[START] outputs:"
              << " lower=" << cfg.output_lower << " upper=" << cfg.output_upper
              << " ground_truth=" << cfg.output_ground_truth << std::endl;
    std::cout << "[START] seed=" << seed_value << " expected_bytes(lower)=" << expected_lower_bytes
              << " expected_bytes(upper)=" << expected_upper_bytes
              << " expected_bytes(ground_truth)=" << expected_gt_bytes << std::endl;

    const auto global_start = std::chrono::steady_clock::now();

    std::vector<float> dataset;
    std::size_t available_points = 0;
    if (!load_dataset(cfg, dataset, available_points)) {
        return 1;
    }
    std::cout << "[START] loaded_points=" << cfg.total_points
              << " available_points_in_file=" << available_points << std::endl;

    std::vector<float> lower_out(total_queries * kDimensions);
    std::vector<float> upper_out(total_queries * kDimensions);
    std::vector<float> ground_truth_out(total_queries * cfg.matches_per_query * kDimensions);

    std::vector<float> lower(kDimensions);
    std::vector<float> upper(kDimensions);
    std::vector<std::size_t> dim_order(kDimensions);
    std::vector<std::size_t> matching_indices;
    matching_indices.reserve(cfg.matches_per_query + 1);

    std::map<std::size_t, std::size_t> retries_histogram;
    std::vector<std::size_t> per_trie_success(cfg.num_tries, 0);
    std::size_t completed_queries = 0;
    std::size_t total_attempts = 0;
    constexpr std::size_t kRateLogInterval = 25;

    for (std::size_t trie_idx = 0; trie_idx < cfg.num_tries; ++trie_idx) {
        const auto trie_start = std::chrono::steady_clock::now();
        const std::size_t n = (trie_idx + 1) * cfg.trie_step;

        std::cout << "[TRIE] start index=" << (trie_idx + 1) << "/" << cfg.num_tries << " n=" << n
                  << std::endl;

        for (std::size_t local_q = 0; local_q < cfg.queries_per_trie; ++local_q) {
            const std::size_t global_q = trie_idx * cfg.queries_per_trie + local_q;
            std::size_t best_count = 0;
            std::size_t attempts_used = 0;
            bool accepted = false;

            std::cout << "[QUERY] start global_id=" << global_q << " trie_index=" << (trie_idx + 1)
                      << "/" << cfg.num_tries << " local_query=" << (local_q + 1) << "/"
                      << cfg.queries_per_trie << " n=" << n << std::endl;

            for (std::size_t attempt = 1; attempt <= cfg.max_attempts_per_query; ++attempt) {
                ++total_attempts;

                std::vector<std::size_t> seed_indices =
                    sample_unique_indices(n, cfg.matches_per_query, rng);

                build_bounds(dataset, seed_indices, lower, upper);
                build_dimension_order(lower, upper, dim_order);

                const std::size_t count = count_matches_in_prefix(
                    dataset, n, lower, upper, dim_order, cfg.matches_per_query, &matching_indices);

                if (count == cfg.matches_per_query) {
                    accepted = true;
                    attempts_used = attempt;
                    break;
                }

                best_count = std::max(best_count, count);
                if (attempt <= 3 || (attempt % 1000 == 0)) {
                    std::cout << "[QUERY] rejected global_id=" << global_q << " attempt=" << attempt
                              << " count=" << count << " target=" << cfg.matches_per_query
                              << std::endl;
                }
            }

            if (!accepted) {
                std::cerr << "[FAIL] Unable to generate exact-" << cfg.matches_per_query
                          << " query within max attempts."
                          << " trie_index=" << (trie_idx + 1) << "/" << cfg.num_tries << " n=" << n
                          << " local_query=" << (local_q + 1) << "/" << cfg.queries_per_trie
                          << " global_query_id=" << global_q
                          << " attempts_used=" << cfg.max_attempts_per_query
                          << " best_observed_count=" << best_count << std::endl;
                return 1;
            }

            if (matching_indices.size() != cfg.matches_per_query) {
                std::cerr << "[FAIL] Internal error: accepted query has " << matching_indices.size()
                          << " collected matches, expected " << cfg.matches_per_query << std::endl;
                return 1;
            }

            const std::size_t lower_offset = global_q * kDimensions;
            const std::size_t upper_offset = global_q * kDimensions;
            std::copy(lower.begin(), lower.end(), lower_out.begin() + lower_offset);
            std::copy(upper.begin(), upper.end(), upper_out.begin() + upper_offset);

            const std::size_t gt_base = global_q * cfg.matches_per_query * kDimensions;
            for (std::size_t m = 0; m < cfg.matches_per_query; ++m) {
                const std::size_t row_idx = matching_indices[m];
                const float *src = dataset.data() + row_idx * kDimensions;
                float *dst = ground_truth_out.data() + gt_base + m * kDimensions;
                std::copy(src, src + kDimensions, dst);
            }

            const std::size_t retries = attempts_used - 1;
            ++retries_histogram[retries];
            ++per_trie_success[trie_idx];
            ++completed_queries;

            std::cout << "[QUERY] accepted global_id=" << global_q << " attempts=" << attempts_used
                      << " retries=" << retries << " matches=" << matching_indices.size()
                      << std::endl;

            if (completed_queries % kRateLogInterval == 0 || completed_queries == total_queries) {
                const double elapsed_s = elapsed_seconds(global_start);
                const double qps = (elapsed_s > 0.0) ? (completed_queries / elapsed_s) : 0.0;
                const std::size_t remaining = total_queries - completed_queries;
                const double eta_s = (qps > 0.0) ? (remaining / qps) : 0.0;

                std::cout << "[RATE] completed=" << completed_queries << "/" << total_queries
                          << " elapsed=" << format_seconds(elapsed_s)
                          << " throughput_qps=" << std::fixed << std::setprecision(2) << qps
                          << " eta=" << format_seconds(eta_s) << std::endl;
            }
        }

        const double trie_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - trie_start).count();
        std::cout << "[TRIE] done index=" << (trie_idx + 1) << "/" << cfg.num_tries << " n=" << n
                  << " success=" << per_trie_success[trie_idx] << "/" << cfg.queries_per_trie
                  << " elapsed=" << format_seconds(trie_elapsed) << std::endl;
    }

    std::cout << "[WRITE] writing output files..." << std::endl;
    if (!write_binary_f32(cfg.output_lower, lower_out))
        return 1;
    if (!write_binary_f32(cfg.output_upper, upper_out))
        return 1;
    if (!write_binary_f32(cfg.output_ground_truth, ground_truth_out))
        return 1;

    std::error_code ec;
    const uint64_t lower_bytes = static_cast<uint64_t>(std::filesystem::file_size(cfg.output_lower, ec));
    if (ec) {
        std::cerr << "Error: Unable to stat output file " << cfg.output_lower << ": " << ec.message()
                  << std::endl;
        return 1;
    }
    const uint64_t upper_bytes = static_cast<uint64_t>(std::filesystem::file_size(cfg.output_upper, ec));
    if (ec) {
        std::cerr << "Error: Unable to stat output file " << cfg.output_upper << ": " << ec.message()
                  << std::endl;
        return 1;
    }
    const uint64_t gt_bytes =
        static_cast<uint64_t>(std::filesystem::file_size(cfg.output_ground_truth, ec));
    if (ec) {
        std::cerr << "Error: Unable to stat output file " << cfg.output_ground_truth << ": "
                  << ec.message() << std::endl;
        return 1;
    }

    const double total_elapsed = elapsed_seconds(global_start);
    std::cout << "[DONE] completed_queries=" << completed_queries << "/" << total_queries
              << " total_attempts=" << total_attempts << " avg_attempts_per_query=" << std::fixed
              << std::setprecision(3)
              << ((completed_queries > 0) ? (static_cast<double>(total_attempts) / completed_queries)
                                          : 0.0)
              << " total_elapsed=" << format_seconds(total_elapsed) << std::endl;

    std::cout << "[DONE] retries_histogram (retries -> query_count):" << std::endl;
    for (const auto &entry : retries_histogram) {
        std::cout << "  " << entry.first << " -> " << entry.second << std::endl;
    }

    std::cout << "[DONE] per_trie_success (n -> success/queries_per_trie):" << std::endl;
    for (std::size_t trie_idx = 0; trie_idx < cfg.num_tries; ++trie_idx) {
        const std::size_t n = (trie_idx + 1) * cfg.trie_step;
        std::cout << "  n=" << n << " -> " << per_trie_success[trie_idx] << "/"
                  << cfg.queries_per_trie << std::endl;
    }

    std::cout << "[DONE] output_sizes_bytes:"
              << " lower=" << lower_bytes << " upper=" << upper_bytes
              << " ground_truth=" << gt_bytes << std::endl;

    if (lower_bytes != expected_lower_bytes || upper_bytes != expected_upper_bytes ||
        gt_bytes != expected_gt_bytes) {
        std::cerr << "[FAIL] Output size mismatch."
                  << " expected(lower,upper,gt)=(" << expected_lower_bytes << ","
                  << expected_upper_bytes << "," << expected_gt_bytes << ")"
                  << " actual=(" << lower_bytes << "," << upper_bytes << "," << gt_bytes << ")"
                  << std::endl;
        return 1;
    }

    std::cout << "[DONE] success" << std::endl;
    return 0;
}
