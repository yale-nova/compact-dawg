/**
 * bench_dawg_sharing.cpp
 *
 * Suffix-sharing analysis for CompactDawg built with TRACK_SHARING: aggregate memo/trie/DAWG
 * statistics, per-depth sharing_rate (with normalized_depth), and in-degree histograms.
 * One dtype per run (--dtype); sweeps --dims, --n-keys, and --group-bits.
 *
 * This tool:
 *   1) Loads msmarco_v2_corpus_{dim}d_{dtype}.bin from --data-dir (same layout as bench_dawg_sweep)
 *   2) For each (dim, N, GROUP_BITS), builds CompactDawg<GB, false, true> and collects stats
 *   3) Writes three CSVs under --output-dir (default sharing_results/):
 *        sharing_summary.csv   — per-(dim, N, GB) aggregates
 *        sharing_depth.csv     — per-depth finalize/memo hits and sharing_rate
 *        sharing_indegree.csv  — in-degree distribution
 *
 * Plot results with:
 *   python3 scripts/plot_dawg_sharing.py --input-dir sharing_results/ --output-dir plots/sharing/
 *
 * Example:
 *   cmake --build . --target bench_dawg_sharing
 *   ./bench_dawg_sharing --data-dir data/embeddings/qwen3-embedding-0.6b/msmarco_v2 \
 *       --dims 16,64,256,1024 --n-keys 1000,10000,100000 --group-bits 32,64,128 \
 *       --output-dir sharing_results/
 */

#include "bench_common.h"
#include "compact_dawg.h"
#include "dawg_sharing_analysis.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using hrc = std::chrono::high_resolution_clock;


// ---------------------------------------------------------------------------
// Sharing analysis: build CompactDawg<GB, false, true> and collect stats
// ---------------------------------------------------------------------------

struct SharingResult {
    uint32_t dim;
    size_t n_keys;
    size_t n_unique_keys;
    uint32_t group_bits;
    size_t total_bits;
    size_t finalize_calls;
    size_t memo_hits;
    size_t unique_nodes;
    size_t trie_edges;
    size_t dawg_edges;
    double sharing_ratio;
    double node_reduction;
    double edge_saving_pct;
    size_t total_bytes;
    double bytes_per_key;
    size_t node_count;
    size_t shared_nodes;

    std::vector<size_t> per_depth_finalize;
    std::vector<size_t> per_depth_hits;
    std::vector<size_t> indegree_hist;
};

template <uint32_t BITS>
static SharingResult analyze(const std::vector<std::string> &keys, size_t total_bits)
{
    SharingResult r{};
    r.group_bits = BITS;
    r.total_bits = total_bits;

    CompactDawg<BITS, false, true> dawg;
    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    const auto &ss = dawg.GetSharingStats();
    r.finalize_calls = ss.finalize_calls;
    r.memo_hits = ss.memo_hits;
    r.unique_nodes = ss.unique_nodes;
    r.trie_edges = ss.trie_edges;
    r.dawg_edges = ss.dawg_edges;

    r.sharing_ratio = ss.finalize_calls > 0
                          ? static_cast<double>(ss.memo_hits) / static_cast<double>(ss.finalize_calls)
                          : 0.0;
    r.node_reduction = ss.finalize_calls > 0
                           ? static_cast<double>(ss.unique_nodes) / static_cast<double>(ss.finalize_calls)
                           : 1.0;
    r.edge_saving_pct = ss.trie_edges > 0
                            ? 1.0 - static_cast<double>(ss.dawg_edges) / static_cast<double>(ss.trie_edges)
                            : 0.0;

    r.total_bytes = dawg.size_in_bytes();
    r.bytes_per_key = keys.empty() ? 0.0
                                   : static_cast<double>(r.total_bytes) / static_cast<double>(keys.size());
    r.node_count = dawg.get_node_count();
    r.shared_nodes = CountSharedNodes(dawg);

    r.per_depth_finalize = dawg.GetPerDepthFinalize();
    r.per_depth_hits = dawg.GetPerDepthHits();
    r.indegree_hist = ComputeInDegreeHistogram(dawg);

    return r;
}

// When total_bits / GROUP_BITS is tiny, the trie has almost no depth in
// GROUP_BITS units (often memo_hits == 0). Skip those configs so CSVs and
// plots do not show a misleading "0% sharing" for cases that are not
// comparable to deeper tries.
static constexpr size_t kMinKeySymbolLevels = 3;

// Dispatch across supported GROUP_BITS values.
template <uint32_t BITS>
static void maybe_analyze(std::vector<SharingResult> &out,
                          const std::vector<std::string> &keys,
                          size_t total_bits)
{
    if (total_bits % BITS != 0)
        return;
    if (total_bits / static_cast<size_t>(BITS) < kMinKeySymbolLevels)
        return;
    out.push_back(analyze<BITS>(keys, total_bits));
}

static void analyze_all_gb(std::vector<SharingResult> &out,
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
        maybe_analyze<16>(out, keys, total_bits);
    if (want(32))
        maybe_analyze<32>(out, keys, total_bits);
    if (want(64))
        maybe_analyze<64>(out, keys, total_bits);
    if (want(128))
        maybe_analyze<128>(out, keys, total_bits);
    if (want(256))
        maybe_analyze<256>(out, keys, total_bits);
    if (want(512))
        maybe_analyze<512>(out, keys, total_bits);
    if (want(1024))
        maybe_analyze<1024>(out, keys, total_bits);
}

// ---------------------------------------------------------------------------
// CSV writers
// ---------------------------------------------------------------------------

static void write_summary_header(FILE *fp)
{
    fprintf(fp, "dim,n_keys,n_unique_keys,group_bits,"
                "finalize_calls,memo_hits,unique_nodes,trie_edges,dawg_edges,"
                "sharing_ratio,node_reduction,edge_saving_pct,"
                "total_bytes,bytes_per_key,node_count,shared_nodes\n");
}

static void write_summary_row(FILE *fp, const SharingResult &r)
{
    fprintf(fp, "%u,%zu,%zu,%u,"
                "%zu,%zu,%zu,%zu,%zu,"
                "%.8f,%.8f,%.8f,"
                "%zu,%.6f,%zu,%zu\n",
            r.dim, r.n_keys, r.n_unique_keys, r.group_bits,
            r.finalize_calls, r.memo_hits, r.unique_nodes, r.trie_edges, r.dawg_edges,
            r.sharing_ratio, r.node_reduction, r.edge_saving_pct,
            r.total_bytes, r.bytes_per_key, r.node_count, r.shared_nodes);
}

static void write_depth_header(FILE *fp)
{
    fprintf(fp, "dim,n_keys,group_bits,depth,normalized_depth,finalize_calls,memo_hits,sharing_rate\n");
}

static void write_depth_rows(FILE *fp, const SharingResult &r)
{
    size_t max_depth = std::max(r.per_depth_finalize.size(), r.per_depth_hits.size());
    for (size_t d = 0; d < max_depth; d++) {
        size_t fc = d < r.per_depth_finalize.size() ? r.per_depth_finalize[d] : 0;
        size_t mh = d < r.per_depth_hits.size() ? r.per_depth_hits[d] : 0;
        double rate = fc > 0 ? static_cast<double>(mh) / static_cast<double>(fc) : 0.0;
        double norm_d = r.total_bits > 0 ? static_cast<double>(d * r.group_bits) / static_cast<double>(r.total_bits) : 0.0;
        fprintf(fp, "%u,%zu,%u,%zu,%.6f,%zu,%zu,%.8f\n",
                r.dim, r.n_keys, r.group_bits, d, norm_d, fc, mh, rate);
    }
}

static void write_indegree_header(FILE *fp)
{
    fprintf(fp, "dim,n_keys,group_bits,in_degree,node_count\n");
}

static void write_indegree_rows(FILE *fp, const SharingResult &r)
{
    for (size_t deg = 0; deg < r.indegree_hist.size(); deg++) {
        if (r.indegree_hist[deg] == 0)
            continue;
        fprintf(fp, "%u,%zu,%u,%zu,%zu\n",
                r.dim, r.n_keys, r.group_bits, deg, r.indegree_hist[deg]);
    }
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

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
            "Analyze suffix sharing in CompactDawg across dimensions,\n"
            "dataset sizes, and GROUP_BITS values.\n"
            "\n"
            "Options:\n"
            "  --data-dir <path>       Embeddings directory (msmarco_v2_corpus_*d_*.bin)\n"
            "  --dims <csv>            Dimensions to sweep (default: 16,32,64,128,256,512,1024)\n"
            "  --dtype <str>           Data type (default: float32)\n"
            "  --n-keys <csv>          Dataset sizes (default: 1000,10000,100000)\n"
            "  --group-bits <csv>      GROUP_BITS values (default: 32,64,128,256)\n"
            "  --float-shift <val>     Shift added to each coordinate (default: 0.5)\n"
            "  --output-dir <path>     Output directory for CSVs (default: sharing_results)\n"
            "\n"
            "Note: a (dim, GROUP_BITS) pair is skipped unless GROUP_BITS divides the Morton\n"
            "      key length in bits and the key spans at least %zu GROUP_BITS-wide\n"
            "      symbols (total_bits / GROUP_BITS).\n"
            "\n"
            "Outputs:\n"
            "  sharing_summary.csv     Per-(dim,N,GB) aggregate sharing metrics\n"
            "  sharing_depth.csv       Per-depth sharing rates\n"
            "  sharing_indegree.csv    In-degree distribution\n"
            "\n",
            argv0, kMinKeySymbolLevels);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::string data_dir;
    std::vector<uint32_t> dims = {16, 32, 64, 128, 256, 512, 1024};
    std::string dtype = "float32";
    std::vector<size_t> n_keys_list = {1000, 10000, 100000};
    std::vector<uint32_t> group_bits = {32, 64, 128, 256};
    float float_shift = 0.5f;
    std::string output_dir = "sharing_results";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--dims" && i + 1 < argc) {
            dims = parse_csv_uint32s(argv[++i]);
        } else if (arg == "--dtype" && i + 1 < argc) {
            dtype = argv[++i];
        } else if (arg == "--n-keys" && i + 1 < argc) {
            n_keys_list = parse_csv_sizes(argv[++i]);
        } else if (arg == "--group-bits" && i + 1 < argc) {
            group_bits = parse_csv_uint32s(argv[++i]);
        } else if (arg == "--float-shift" && i + 1 < argc) {
            float_shift = std::stof(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
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

    std::error_code fs_error;
    if (!std::filesystem::create_directories(output_dir, fs_error) && fs_error) {
        fprintf(stderr, "Error: Cannot create output directory %s (%s)\n",
                output_dir.c_str(), fs_error.message().c_str());
        return 1;
    }

    std::string summary_path = output_dir + "/sharing_summary.csv";
    std::string depth_path = output_dir + "/sharing_depth.csv";
    std::string indegree_path = output_dir + "/sharing_indegree.csv";

    FILE *fp_summary = fopen(summary_path.c_str(), "w");
    FILE *fp_depth = fopen(depth_path.c_str(), "w");
    FILE *fp_indegree = fopen(indegree_path.c_str(), "w");
    if (!fp_summary || !fp_depth || !fp_indegree) {
        fprintf(stderr, "Error: Cannot open output CSVs in %s\n", output_dir.c_str());
        return 1;
    }

    write_summary_header(fp_summary);
    write_depth_header(fp_depth);
    write_indegree_header(fp_indegree);

    bool is_fp16 = (dtype == "float16");
    size_t max_n = *std::max_element(n_keys_list.begin(), n_keys_list.end());

    printf("\n");
    printf("========================================\n");
    printf("  CompactDawg Suffix Sharing Analysis\n");
    printf("  Data dir: %s\n", data_dir.c_str());
    printf("  Output:   %s/\n", output_dir.c_str());
    printf("========================================\n");

    for (uint32_t dim : dims) {
        std::string path = build_data_path(data_dir, dim, dtype);
        size_t total_bits = static_cast<size_t>(dim) * (is_fp16 ? 16 : 32);

        printf("\n  === dim=%u  key_length=%s bits ===\n", dim, comma_fmt(total_bits).c_str());

        EncodedDataset ds = load_encode_dedup(path.c_str(), dim, is_fp16, max_n, float_shift);
        if (ds.keys.empty()) {
            fprintf(stderr, "  [skip] Cannot load %s\n", path.c_str());
            continue;
        }

        printf("  Read %s vectors -> %s unique keys\n",
               comma_fmt(ds.vectors_read).c_str(), comma_fmt(ds.keys.size()).c_str());

        for (size_t n : n_keys_list) {
            size_t actual_n = std::min(n, ds.keys.size());
            if (actual_n == 0)
                continue;

            std::vector<std::string> keys(
                ds.keys.begin(),
                ds.keys.begin() + static_cast<std::ptrdiff_t>(actual_n));

            fprintf(stderr, "  [analyze] dim=%u N=%s ...\n", dim, comma_fmt(actual_n).c_str());

            std::vector<SharingResult> results;
            analyze_all_gb(results, keys, total_bits, group_bits);

            const char *hdr = "    %-8s %12s %12s %12s %12s %10s %10s %10s\n";
            const char *row = "    %-8s %12s %12s %12s %12s %10.6f %10.6f %10.6f\n";
            printf("\n    N = %s keys\n", comma_fmt(actual_n).c_str());
            printf("    %s\n", std::string(96, '-').c_str());
            printf(hdr, "GB", "Finalize", "MemoHits", "UniqueNodes", "TrieEdges",
                   "ShrRatio", "NodeRed", "EdgeSave%");
            printf("    %s\n", std::string(96, '-').c_str());

            for (auto &r : results) {
                r.dim = dim;
                r.n_keys = n;
                r.n_unique_keys = actual_n;

                char gb_str[16];
                snprintf(gb_str, sizeof(gb_str), "GB-%u", r.group_bits);
                printf(row, gb_str,
                       comma_fmt(r.finalize_calls).c_str(),
                       comma_fmt(r.memo_hits).c_str(),
                       comma_fmt(r.unique_nodes).c_str(),
                       comma_fmt(r.trie_edges).c_str(),
                       r.sharing_ratio, r.node_reduction, r.edge_saving_pct);

                write_summary_row(fp_summary, r);
                write_depth_rows(fp_depth, r);
                write_indegree_rows(fp_indegree, r);
            }

            printf("    %s\n", std::string(96, '-').c_str());
            fflush(stdout);
            fflush(fp_summary);
            fflush(fp_depth);
            fflush(fp_indegree);
        }
    }

    fclose(fp_summary);
    fclose(fp_depth);
    fclose(fp_indegree);

    printf("\n  Results written to:\n");
    printf("    %s\n", summary_path.c_str());
    printf("    %s\n", depth_path.c_str());
    printf("    %s\n", indegree_path.c_str());
    printf("\n");

    return 0;
}
