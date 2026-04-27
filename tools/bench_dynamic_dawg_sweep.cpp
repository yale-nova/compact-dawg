#include "bench_common.h"
#include "compact_dawg.h"
#include "dynamic_dawg.h"
#include "dawg_segmentation.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hrc = std::chrono::high_resolution_clock;

struct FixedWidthPcResult {
    uint32_t group_bits = 0;
    size_t total_bytes = 0;
    double bytes_per_key = 0.0;
    size_t edges = 0;
    double insert_s = 0.0;
    double finish_s = 0.0;
    double total_build_s = 0.0;
    bool valid = false;
};

struct DynamicDawgResult {
    double rho_threshold = 0.0;
    std::string segmentation_method;
    size_t n_keys = 0;
    size_t n_unique_keys = 0;
    size_t n_segments = 0;
    size_t total_bytes = 0;
    double bytes_per_key = 0.0;
    double normalized_bpk = 0.0;
    size_t edges = 0;
    double plan_s = 0.0;
    double insert_s = 0.0;
    double finish_s = 0.0;
    double total_build_s = 0.0;

    uint32_t best_pc_storage_group_bits = 0;
    double best_pc_bytes_per_key = 0.0;
    size_t best_pc_edges = 0;
    uint32_t best_pc_build_group_bits = 0;
    double best_pc_total_build_s = 0.0;
    double delta_vs_best_pc_bpk = 0.0;
    double delta_vs_best_pc_build_s = 0.0;
    double edge_reduction_vs_best_pc_pct = 0.0;

    uint32_t best_cd_storage_group_bits = 0;
    double best_cd_bytes_per_key = 0.0;
    size_t best_cd_edges = 0;
    uint32_t best_cd_build_group_bits = 0;
    double best_cd_total_build_s = 0.0;
    double delta_vs_best_cd_bpk = 0.0;
    double delta_vs_best_cd_build_s = 0.0;
    double edge_reduction_vs_best_cd_pct = 0.0;

    size_t label_bits = 0;
    size_t label_offset_bits = 0;
    size_t label_length_bits = 0;
    size_t plan_bytes = 0;
    size_t non_label_metadata_bits = 0;
    std::string selected_sample_method;
    double sample_normalized_bpk = 0.0;

    double avg_label_bits = 0.0;
    double label_metadata_share = 0.0;
    std::string width_histogram;
};

static std::vector<int> parse_csv_ints(const std::string &s)
{
    std::vector<int> out;
    std::string token;
    std::stringstream ss(s);
    while (std::getline(ss, token, ',')) {
        if (!token.empty())
            out.push_back(std::stoi(token));
    }
    return out;
}

static std::vector<double> parse_csv_doubles(const std::string &s)
{
    std::vector<double> out;
    std::string token;
    std::stringstream ss(s);
    while (std::getline(ss, token, ',')) {
        if (!token.empty())
            out.push_back(std::stod(token));
    }
    return out;
}

static std::vector<std::string> parse_csv_strings(const std::string &s)
{
    std::vector<std::string> out;
    std::string token;
    std::stringstream ss(s);
    while (std::getline(ss, token, ',')) {
        if (!token.empty())
            out.push_back(token);
    }
    return out;
}

static bool contains_string(const std::vector<std::string> &values, const std::string &needle)
{
    return std::find(values.begin(), values.end(), needle) != values.end();
}

static std::vector<std::string> make_even_sample(const std::vector<std::string> &keys,
                                                 size_t sample_size)
{
    if (sample_size == 0 || sample_size >= keys.size())
        return keys;
    if (sample_size == 1)
        return {keys.front()};

    std::vector<std::string> sample;
    sample.reserve(sample_size);
    const size_t last = keys.size() - 1;
    const size_t denom = sample_size - 1;
    for (size_t i = 0; i < sample_size; ++i) {
        const size_t idx = (i * last + denom / 2) / denom;
        sample.push_back(keys[idx]);
    }
    return sample;
}

template <uint32_t BITS>
static FixedWidthPcResult bench_pc_one(const std::vector<std::string> &keys, uint32_t total_bits)
{
    FixedWidthPcResult out;
    out.group_bits = BITS;
    if (total_bits % BITS != 0)
        return out;

    auto t0 = hrc::now();
    CompactDawg<BITS, true> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    auto t1 = hrc::now();
    dawg.Finish();
    auto t2 = hrc::now();

    out.total_bytes = dawg.size_in_bytes();
    out.bytes_per_key = static_cast<double>(out.total_bytes) / static_cast<double>(keys.size());
    out.edges = dawg.get_total_edges();
    out.insert_s = std::chrono::duration<double>(t1 - t0).count();
    out.finish_s = std::chrono::duration<double>(t2 - t1).count();
    out.total_build_s = std::chrono::duration<double>(t2 - t0).count();
    out.valid = true;
    return out;
}

template <uint32_t BITS>
static FixedWidthPcResult bench_cd_one(const std::vector<std::string> &keys, uint32_t total_bits)
{
    FixedWidthPcResult out;
    out.group_bits = BITS;
    if (total_bits % BITS != 0)
        return out;

    auto t0 = hrc::now();
    CompactDawg<BITS> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    auto t1 = hrc::now();
    dawg.Finish();
    auto t2 = hrc::now();

    out.total_bytes = dawg.size_in_bytes();
    out.bytes_per_key = static_cast<double>(out.total_bytes) / static_cast<double>(keys.size());
    out.edges = dawg.get_total_edges();
    out.insert_s = std::chrono::duration<double>(t1 - t0).count();
    out.finish_s = std::chrono::duration<double>(t2 - t1).count();
    out.total_build_s = std::chrono::duration<double>(t2 - t0).count();
    out.valid = true;
    return out;
}

static std::vector<FixedWidthPcResult> bench_pc_baselines(const std::vector<std::string> &keys,
                                                          uint32_t total_bits)
{
    std::vector<FixedWidthPcResult> out;
    for (const auto &r : {bench_pc_one<16>(keys, total_bits), bench_pc_one<32>(keys, total_bits),
                          bench_pc_one<64>(keys, total_bits), bench_pc_one<128>(keys, total_bits),
                          bench_pc_one<256>(keys, total_bits),
                          bench_pc_one<512>(keys, total_bits),
                          bench_pc_one<1024>(keys, total_bits)}) {
        if (r.valid)
            out.push_back(r);
    }
    return out;
}

using DynamicMethodRunner = std::function<DynamicDawgResult(double)>;

static std::vector<FixedWidthPcResult> bench_cd_baselines(const std::vector<std::string> &keys,
                                                          uint32_t total_bits)
{
    std::vector<FixedWidthPcResult> out;
    for (const auto &r : {bench_cd_one<16>(keys, total_bits), bench_cd_one<32>(keys, total_bits),
                          bench_cd_one<64>(keys, total_bits), bench_cd_one<128>(keys, total_bits),
                          bench_cd_one<256>(keys, total_bits),
                          bench_cd_one<512>(keys, total_bits),
                          bench_cd_one<1024>(keys, total_bits)}) {
        if (r.valid)
            out.push_back(r);
    }
    return out;
}

static const FixedWidthPcResult *best_by_storage(const std::vector<FixedWidthPcResult> &pcs)
{
    if (pcs.empty())
        return nullptr;
    return &*std::min_element(
        pcs.begin(), pcs.end(), [](const FixedWidthPcResult &a, const FixedWidthPcResult &b) {
            if (a.bytes_per_key != b.bytes_per_key)
                return a.bytes_per_key < b.bytes_per_key;
            return a.group_bits < b.group_bits;
        });
}

static const FixedWidthPcResult *best_by_build(const std::vector<FixedWidthPcResult> &pcs)
{
    if (pcs.empty())
        return nullptr;
    return &*std::min_element(
        pcs.begin(), pcs.end(), [](const FixedWidthPcResult &a, const FixedWidthPcResult &b) {
            if (a.total_build_s != b.total_build_s)
                return a.total_build_s < b.total_build_s;
            return a.group_bits < b.group_bits;
        });
}

static double key_bytes_for_dim(int dim) { return static_cast<double>(dim) * 32.0 / 8.0; }

static std::string summarize_width_histogram(const dawg_seg::SegmentPlan &plan)
{
    std::map<uint32_t, uint32_t> hist;
    for (uint32_t w : plan.widths)
        hist[w]++;

    std::ostringstream oss;
    bool first = true;
    for (const auto &[width, count] : hist) {
        if (!first)
            oss << ";";
        first = false;
        oss << width << ":" << count;
    }
    return oss.str();
}

static void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --data-dir PATH         Path to dataset directory\n"
              << "  --dims DIMS             Comma-separated list of dimensions (e.g. 16,64,256)\n"
              << "  --n-keys NUMBER         Number of unique keys to load\n"
              << "  --planner-sample-size N Number of evenly spaced keys used for DynamicDawg planning "
                 "(0 = all, default 10000)\n"
              << "  --methods METHODS       Comma-separated DynamicDawg rows to run. Choices: greedy_rho,"
                 "greedy_cost_aware,greedy_cost_min4,greedy_cost_min8,phase_1024_16_128,"
                 "sampled_best\n"
              << "  --rho-thresholds VALUES Comma-separated rho thresholds (e.g. 0.3,0.5,0.7)\n"
              << "  --output-csv PATH       Path to output CSV file\n";
}

static void write_csv_header(std::ofstream &out)
{
    out << "timestamp,dim,dtype,n_keys,n_unique_keys,rho_threshold,segmentation_method,n_segments,"
           "total_bytes,bytes_per_key,normalized_bpk,edges,plan_s,insert_s,finish_s,total_build_s,"
           "best_pc_storage_group_bits,best_pc_bytes_per_key,best_pc_edges,"
           "best_pc_build_group_bits,best_pc_total_build_s,"
           "delta_vs_best_pc_bpk,delta_vs_best_pc_build_s,edge_reduction_vs_best_pc_pct,"
           "best_cd_storage_group_bits,best_cd_bytes_per_key,best_cd_edges,"
           "best_cd_build_group_bits,best_cd_total_build_s,"
           "delta_vs_best_cd_bpk,delta_vs_best_cd_build_s,edge_reduction_vs_best_cd_pct,"
           "label_bits,label_offset_bits,label_length_bits,plan_bytes,non_label_metadata_bits,"
           "selected_sample_method,sample_normalized_bpk,"
           "avg_label_bits,label_metadata_share,width_histogram\n";
}

static void write_csv_row(std::ofstream &out, const char *timestamp, int dim,
                          const DynamicDawgResult &r)
{
    out << timestamp << "," << dim << ",float32,"
        << r.n_keys << "," << r.n_unique_keys << ","
        << r.rho_threshold << "," << r.segmentation_method << "," << r.n_segments << ","
        << r.total_bytes << "," << r.bytes_per_key << "," << r.normalized_bpk << ","
        << r.edges << "," << r.plan_s << "," << r.insert_s << "," << r.finish_s << ","
        << r.total_build_s << ","
        << r.best_pc_storage_group_bits << "," << r.best_pc_bytes_per_key << ","
        << r.best_pc_edges << ","
        << r.best_pc_build_group_bits << "," << r.best_pc_total_build_s << ","
        << r.delta_vs_best_pc_bpk << "," << r.delta_vs_best_pc_build_s << ","
        << r.edge_reduction_vs_best_pc_pct << ","
        << r.best_cd_storage_group_bits << "," << r.best_cd_bytes_per_key << ","
        << r.best_cd_edges << ","
        << r.best_cd_build_group_bits << "," << r.best_cd_total_build_s << ","
        << r.delta_vs_best_cd_bpk << "," << r.delta_vs_best_cd_build_s << ","
        << r.edge_reduction_vs_best_cd_pct << ","
        << r.label_bits << "," << r.label_offset_bits << "," << r.label_length_bits << ","
        << r.plan_bytes << "," << r.non_label_metadata_bits << ","
        << r.selected_sample_method << "," << r.sample_normalized_bpk << ","
        << r.avg_label_bits << "," << r.label_metadata_share << ","
        << r.width_histogram << "\n";
}

static double measure_dynamic_dawg_normalized_bpk(const std::vector<std::string> &keys,
                                                  const dawg_seg::SegmentPlan &plan, int dim)
{
    DynamicDawg dawg(plan, true);
    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();
    return (static_cast<double>(dawg.size_in_bytes()) / static_cast<double>(keys.size())) /
           key_bytes_for_dim(dim);
}

static DynamicDawgResult bench_dynamic_dawg_with_plan(
    const std::vector<std::string> &keys, const dawg_seg::SegmentPlan &plan, int dim,
    double rho, const std::string &method_name, double plan_s,
    const FixedWidthPcResult *best_pc_storage, const FixedWidthPcResult *best_pc_build,
    const FixedWidthPcResult *best_cd_storage, const FixedWidthPcResult *best_cd_build,
    const std::string &selected_sample_method = "", double sample_normalized_bpk = 0.0)
{
    DynamicDawg dawg(plan, true);

    auto start_insert = hrc::now();
    for (const auto &k : keys)
        dawg.Insert(k);
    auto end_insert = hrc::now();

    auto start_finish = hrc::now();
    dawg.Finish();
    auto end_finish = hrc::now();

    DynamicDawgResult out;
    out.rho_threshold = rho;
    out.segmentation_method = method_name;
    out.n_keys = keys.size();
    out.n_unique_keys = keys.size();
    out.n_segments = plan.depth();
    out.total_bytes = dawg.size_in_bytes();
    out.bytes_per_key = static_cast<double>(out.total_bytes) / static_cast<double>(keys.size());
    out.normalized_bpk = out.bytes_per_key / key_bytes_for_dim(dim);
    out.edges = dawg.get_total_edges();
    out.plan_s = plan_s;
    out.insert_s = std::chrono::duration<double>(end_insert - start_insert).count();
    out.finish_s = std::chrono::duration<double>(end_finish - start_finish).count();
    out.total_build_s = out.plan_s + out.insert_s + out.finish_s;
    out.label_bits = dawg.get_total_label_bits();
    out.label_offset_bits = dawg.get_label_offsets_bits();
    out.label_length_bits = dawg.get_label_lengths_bits();
    out.plan_bytes = dawg.get_plan_bytes();
    out.non_label_metadata_bits = dawg.get_non_label_metadata_bits();
    out.selected_sample_method = selected_sample_method;
    out.sample_normalized_bpk = sample_normalized_bpk;
    out.avg_label_bits = dawg.get_average_label_bits();
    out.label_metadata_share = dawg.get_label_metadata_share();
    out.width_histogram = summarize_width_histogram(plan);

    if (best_pc_storage) {
        out.best_pc_storage_group_bits = best_pc_storage->group_bits;
        out.best_pc_bytes_per_key = best_pc_storage->bytes_per_key;
        out.best_pc_edges = best_pc_storage->edges;
        if (best_pc_storage->bytes_per_key > 0.0) {
            out.delta_vs_best_pc_bpk =
                (out.bytes_per_key - best_pc_storage->bytes_per_key) /
                best_pc_storage->bytes_per_key;
        }
        if (best_pc_storage->edges > 0) {
            out.edge_reduction_vs_best_pc_pct =
                100.0 * (static_cast<double>(best_pc_storage->edges) - static_cast<double>(out.edges)) /
                static_cast<double>(best_pc_storage->edges);
        }
    }

    if (best_pc_build) {
        out.best_pc_build_group_bits = best_pc_build->group_bits;
        out.best_pc_total_build_s = best_pc_build->total_build_s;
        if (best_pc_build->total_build_s > 0.0) {
            out.delta_vs_best_pc_build_s =
                (out.total_build_s - best_pc_build->total_build_s) /
                best_pc_build->total_build_s;
        }
    }

    if (best_cd_storage) {
        out.best_cd_storage_group_bits = best_cd_storage->group_bits;
        out.best_cd_bytes_per_key = best_cd_storage->bytes_per_key;
        out.best_cd_edges = best_cd_storage->edges;
        if (best_cd_storage->bytes_per_key > 0.0) {
            out.delta_vs_best_cd_bpk =
                (out.bytes_per_key - best_cd_storage->bytes_per_key) /
                best_cd_storage->bytes_per_key;
        }
        if (best_cd_storage->edges > 0) {
            out.edge_reduction_vs_best_cd_pct =
                100.0 * (static_cast<double>(best_cd_storage->edges) -
                         static_cast<double>(out.edges)) /
                static_cast<double>(best_cd_storage->edges);
        }
    }

    if (best_cd_build) {
        out.best_cd_build_group_bits = best_cd_build->group_bits;
        out.best_cd_total_build_s = best_cd_build->total_build_s;
        if (best_cd_build->total_build_s > 0.0) {
            out.delta_vs_best_cd_build_s =
                (out.total_build_s - best_cd_build->total_build_s) /
                best_cd_build->total_build_s;
        }
    }

    return out;
}

template <typename PlannerFn>
static DynamicDawgResult bench_dynamic_dawg_plan(
    const std::vector<std::string> &keys, const std::vector<std::string> &planning_keys,
    uint32_t total_bits, int dim, double rho, const std::string &method_name, PlannerFn planner,
    const FixedWidthPcResult *best_pc_storage, const FixedWidthPcResult *best_pc_build,
    const FixedWidthPcResult *best_cd_storage, const FixedWidthPcResult *best_cd_build)
{
    auto start_plan = hrc::now();
    dawg_seg::SegmentPlan plan = planner(planning_keys, total_bits, rho);
    auto end_plan = hrc::now();
    double plan_s = std::chrono::duration<double>(end_plan - start_plan).count();

    return bench_dynamic_dawg_with_plan(keys, plan, dim, rho, method_name, plan_s,
                                        best_pc_storage, best_pc_build, best_cd_storage,
                                        best_cd_build);
}

static DynamicDawgResult bench_dynamic_dawg_sampled_best(
    const std::vector<std::string> &keys, uint32_t total_bits, int dim, double rho,
    const std::vector<std::string> &planning_keys, const FixedWidthPcResult *best_pc_storage,
    const FixedWidthPcResult *best_pc_build, const FixedWidthPcResult *best_cd_storage,
    const FixedWidthPcResult *best_cd_build)
{
    auto start_plan = hrc::now();
    size_t sample_n = std::min<size_t>(5000, planning_keys.size());
    std::vector<std::string> sample = make_even_sample(planning_keys, sample_n);

    struct Candidate {
        std::string name;
        dawg_seg::SegmentPlan plan;
        double sample_normalized_bpk = 0.0;
    };

    std::vector<Candidate> candidates;
    candidates.push_back({"sample_greedy_rho", dawg_seg::greedy(sample, total_bits, rho), 0.0});
    candidates.push_back(
        {"sample_greedy_cost_aware", dawg_seg::greedy_cost_aware(sample, total_bits, rho), 0.0});
    candidates.push_back({"sample_greedy_cost_min4",
                          dawg_seg::greedy_cost_aware_min_width(sample, total_bits, rho, 4), 0.0});
    candidates.push_back({"sample_greedy_cost_min8",
                          dawg_seg::greedy_cost_aware_min_width(sample, total_bits, rho, 8), 0.0});
    candidates.push_back({"phase_1024_16_128",
                          dawg_seg::phase_aware(total_bits, 0.20, 0.70, 1024, 16, 128), 0.0});
    candidates.push_back({"phase_1024_8_256",
                          dawg_seg::phase_aware(total_bits, 0.20, 0.70, 1024, 8, 256), 0.0});
    candidates.push_back({"phase_512_16_256",
                          dawg_seg::phase_aware(total_bits, 0.20, 0.70, 512, 16, 256), 0.0});

    Candidate *best = nullptr;
    for (auto &candidate : candidates) {
        if (!candidate.plan.valid())
            continue;
        candidate.sample_normalized_bpk =
            measure_dynamic_dawg_normalized_bpk(sample, candidate.plan, dim);
        if (!best || candidate.sample_normalized_bpk < best->sample_normalized_bpk)
            best = &candidate;
    }

    auto end_plan = hrc::now();
    double plan_s = std::chrono::duration<double>(end_plan - start_plan).count();

    if (!best) {
        dawg_seg::SegmentPlan fallback = dawg_seg::greedy_cost_aware(planning_keys, total_bits, rho);
        return bench_dynamic_dawg_with_plan(keys, fallback, dim, rho, "sampled_best", plan_s,
                                            best_pc_storage, best_pc_build, best_cd_storage,
                                            best_cd_build, "fallback_greedy_cost_aware", 0.0);
    }

    return bench_dynamic_dawg_with_plan(keys, best->plan, dim, rho, "sampled_best", plan_s,
                                        best_pc_storage, best_pc_build, best_cd_storage,
                                        best_cd_build, best->name,
                                        best->sample_normalized_bpk);
}

} // namespace

int main(int argc, char **argv)
{
    std::string data_dir = "data/embeddings/qwen3-embedding-0.6b/msmarco_v2";
    std::vector<int> dims = {16, 64, 256, 1024};
    size_t n_keys = 100000;
    size_t planner_sample_size = 10000;
    std::vector<double> rho_thresholds = {0.3, 0.5, 0.7, 0.9};
    std::vector<std::string> methods = {"greedy_rho", "greedy_cost_aware", "greedy_cost_min4",
                                        "greedy_cost_min8", "phase_1024_16_128",
                                        "sampled_best"};
    std::string output_csv = "results/dynamic_dawg_sweep.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--dims" && i + 1 < argc) {
            dims = parse_csv_ints(argv[++i]);
        } else if (arg == "--n-keys" && i + 1 < argc) {
            n_keys = std::stoull(argv[++i]);
        } else if (arg == "--planner-sample-size" && i + 1 < argc) {
            planner_sample_size = std::stoull(argv[++i]);
        } else if (arg == "--methods" && i + 1 < argc) {
            methods = parse_csv_strings(argv[++i]);
        } else if (arg == "--rho-thresholds" && i + 1 < argc) {
            rho_thresholds = parse_csv_doubles(argv[++i]);
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::vector<std::string> valid_methods = {"greedy_rho", "greedy_cost_aware",
                                                    "greedy_cost_min4", "greedy_cost_min8",
                                                    "phase_1024_16_128", "sampled_best"};
    for (const auto &method : methods) {
        if (!contains_string(valid_methods, method)) {
            std::cerr << "Unknown method: " << method << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    time_t now_t = time(nullptr);
    struct tm *tm_info = localtime(&now_t);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", tm_info);

    std::ofstream out(output_csv);
    if (!out) {
        std::cerr << "Failed to open " << output_csv << " for writing\n";
        return 1;
    }
    write_csv_header(out);

    for (int dim : dims) {
        std::cout << "\n=== Dimension " << dim << " ===\n";

        std::string filename =
            data_dir + "/msmarco_v2_corpus_" + std::to_string(dim) + "d_float32.bin";
        bool is_fp16 = false;
        float float_shift = 0.5f;
        EncodedDataset ds = load_encode_dedup(filename.c_str(), dim, is_fp16, n_keys, float_shift);
        std::vector<std::string> keys = std::move(ds.keys);

        if (keys.empty()) {
            std::cerr << "Failed to load keys for dim " << dim << " (file: " << filename << ")\n";
            continue;
        }
        const std::vector<std::string> planning_keys = make_even_sample(keys, planner_sample_size);
        std::cout << "  Loaded " << keys.size() << " unique keys";
        if (planning_keys.size() != keys.size()) {
            std::cout << "  (DynamicDawg planning sample=" << planning_keys.size() << ")";
        }
        std::cout << "\n";

        const uint32_t total_bits = static_cast<uint32_t>(dim * 32);
        const std::vector<FixedWidthPcResult> pc_results = bench_pc_baselines(keys, total_bits);
        const std::vector<FixedWidthPcResult> cd_results = bench_cd_baselines(keys, total_bits);
        const FixedWidthPcResult *best_pc_storage = best_by_storage(pc_results);
        const FixedWidthPcResult *best_pc_build = best_by_build(pc_results);
        const FixedWidthPcResult *best_cd_storage = best_by_storage(cd_results);
        const FixedWidthPcResult *best_cd_build = best_by_build(cd_results);

        if (!best_pc_storage || !best_pc_build || !best_cd_storage || !best_cd_build) {
            std::cerr << "No valid fixed-width baseline for dim " << dim << "\n";
            continue;
        }

        std::cout << "  Best PC storage: GB=" << best_pc_storage->group_bits << "  "
                  << std::fixed << std::setprecision(2) << best_pc_storage->bytes_per_key
                  << " B/key  edges=" << best_pc_storage->edges << "\n";
        std::cout << "  Best PC build:   GB=" << best_pc_build->group_bits << "  "
                  << best_pc_build->total_build_s << " s\n";
        std::cout << "  Best CD storage: GB=" << best_cd_storage->group_bits << "  "
                  << best_cd_storage->bytes_per_key << " B/key  edges=" << best_cd_storage->edges
                  << "\n";
        std::cout << "  Best CD build:   GB=" << best_cd_build->group_bits << "  "
                  << best_cd_build->total_build_s << " s\n";

        std::vector<std::pair<std::string, DynamicMethodRunner>> method_runners;
        method_runners.reserve(methods.size());
        for (const auto &method : methods) {
            if (method == "greedy_rho") {
                method_runners.push_back({method, [&](double rho) {
                    return bench_dynamic_dawg_plan(
                        keys, planning_keys, total_bits, dim, rho, "greedy_rho",
                        [](const std::vector<std::string> &k, uint32_t bits, double thr) {
                            return dawg_seg::greedy(k, bits, thr);
                        },
                        best_pc_storage, best_pc_build, best_cd_storage, best_cd_build);
                }});
            } else if (method == "greedy_cost_aware") {
                method_runners.push_back({method, [&](double rho) {
                    return bench_dynamic_dawg_plan(
                        keys, planning_keys, total_bits, dim, rho, "greedy_cost_aware",
                        [](const std::vector<std::string> &k, uint32_t bits, double thr) {
                            return dawg_seg::greedy_cost_aware(k, bits, thr);
                        },
                        best_pc_storage, best_pc_build, best_cd_storage, best_cd_build);
                }});
            } else if (method == "greedy_cost_min4") {
                method_runners.push_back({method, [&](double rho) {
                    return bench_dynamic_dawg_plan(
                        keys, planning_keys, total_bits, dim, rho, "greedy_cost_min4",
                        [](const std::vector<std::string> &k, uint32_t bits, double thr) {
                            return dawg_seg::greedy_cost_aware_min_width(k, bits, thr, 4);
                        },
                        best_pc_storage, best_pc_build, best_cd_storage, best_cd_build);
                }});
            } else if (method == "greedy_cost_min8") {
                method_runners.push_back({method, [&](double rho) {
                    return bench_dynamic_dawg_plan(
                        keys, planning_keys, total_bits, dim, rho, "greedy_cost_min8",
                        [](const std::vector<std::string> &k, uint32_t bits, double thr) {
                            return dawg_seg::greedy_cost_aware_min_width(k, bits, thr, 8);
                        },
                        best_pc_storage, best_pc_build, best_cd_storage, best_cd_build);
                }});
            } else if (method == "phase_1024_16_128") {
                method_runners.push_back({method, [&](double rho) {
                    return bench_dynamic_dawg_plan(
                        keys, planning_keys, total_bits, dim, rho, "phase_1024_16_128",
                        [](const std::vector<std::string> &, uint32_t bits, double) {
                            return dawg_seg::phase_aware(bits, 0.20, 0.70, 1024, 16, 128);
                        },
                        best_pc_storage, best_pc_build, best_cd_storage, best_cd_build);
                }});
            } else if (method == "sampled_best") {
                method_runners.push_back({method, [&](double rho) {
                    return bench_dynamic_dawg_sampled_best(
                        keys, total_bits, dim, rho, planning_keys, best_pc_storage, best_pc_build,
                        best_cd_storage, best_cd_build);
                }});
            }
        }

        for (double rho : rho_thresholds) {
            std::cout << "  Threshold: " << rho << "\n";
            std::vector<DynamicDawgResult> results;
            for (const auto &[label, run_method] : method_runners) {
                std::cout << "    running " << label << "..." << std::flush;
                results.push_back(run_method(rho));
                std::cout << " done\n";
            }

            for (const auto &r : results) {
                std::cout << "    " << r.segmentation_method
                          << ": segments=" << r.n_segments
                          << " edges=" << r.edges
                          << " size=" << std::fixed << std::setprecision(2) << r.bytes_per_key
                          << " B/key"
                          << " delta_vs_best_pc_bpk=" << std::showpos
                          << (100.0 * r.delta_vs_best_pc_bpk) << "%"
                          << " delta_vs_best_pc_build=" << (100.0 * r.delta_vs_best_pc_build_s)
                          << "%"
                          << " delta_vs_best_cd_bpk=" << (100.0 * r.delta_vs_best_cd_bpk)
                          << "%";
                if (!r.selected_sample_method.empty()) {
                    std::cout << std::noshowpos
                              << " selected=" << r.selected_sample_method
                              << " sample_norm_bpk=" << r.sample_normalized_bpk;
                }
                std::cout << std::noshowpos << "\n";
                write_csv_row(out, timestamp, dim, r);
                out.flush();
            }
        }
    }

    std::cout << "\nResults written to " << output_csv << "\n";
    return 0;
}
