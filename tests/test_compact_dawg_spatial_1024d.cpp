#ifndef NUM_DIMENSIONS
#define NUM_DIMENSIONS 1024
#endif

#include "bench_common.h"
#include "compact_dawg.h"
#include "mdtrie/trie.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

template <uint32_t GROUP_BITS> static std::string canonicalize_for_group_bits(std::string bits)
{
    const size_t rem = bits.size() % GROUP_BITS;
    if (rem != 0)
        bits.append(GROUP_BITS - rem, '0');
    return bits;
}

static bool point_in_box(const data_point<NUM_DIMENSIONS> &point,
                         const data_point<NUM_DIMENSIONS> &start,
                         const data_point<NUM_DIMENSIONS> &end)
{
    for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
        ordered_coordinate_t v = point.get_ordered_coordinate(d);
        if (v < start.get_ordered_coordinate(d) || v > end.get_ordered_coordinate(d))
            return false;
    }
    return true;
}

// GROUP_BITS = 32 divides 1024 evenly (32 DAWG edges per trie level).
// This is the most natural grouping for 1024D and the one used in benchmarks.
static constexpr uint32_t GB = 32;

void test_1024d_fuzz_oracle()
{
    std::cout << "Running test_1024d_fuzz_oracle (GB=" << GB << ", 80 points, 60 queries)...\n";

    CompactDawg<GB> dawg;
    std::mt19937 rng(314159);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    constexpr int N_POINTS = 80;
    constexpr int N_QUERIES = 60;

    std::vector<data_point<NUM_DIMENSIONS>> points;
    std::vector<std::string> keys;
    points.reserve(N_POINTS);
    keys.reserve(N_POINTS);

    for (int i = 0; i < N_POINTS; ++i) {
        data_point<NUM_DIMENSIONS> p;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            p.set_float_coordinate(d, val_dist(rng));
        points.push_back(p);
        keys.push_back(encode_morton_bitstring(p));
    }

    std::vector<size_t> idx(keys.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });

    std::vector<std::string> unique_keys;
    std::vector<data_point<NUM_DIMENSIONS>> unique_points;
    for (size_t i : idx) {
        if (!unique_keys.empty() && keys[i] == unique_keys.back())
            continue;
        unique_keys.push_back(keys[i]);
        unique_points.push_back(points[i]);
    }

    std::vector<std::string> canonical_keys;
    canonical_keys.reserve(unique_keys.size());
    for (const auto &k : unique_keys)
        canonical_keys.push_back(canonicalize_for_group_bits<GB>(k));

    std::vector<std::string> sorted_keys = unique_keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    for (const auto &k : sorted_keys)
        dawg.Insert(k);
    dawg.Finish();

    std::uniform_int_distribution<size_t> pick(0, unique_points.size() - 1);
    std::uniform_real_distribution<float> radius_dist(0.05f, 0.5f);

    for (int q = 0; q < N_QUERIES; ++q) {
        data_point<NUM_DIMENSIONS> start, end;

        if (q % 2 == 0) {
            const auto &a = unique_points[pick(rng)];
            const auto &b = unique_points[pick(rng)];
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                ordered_coordinate_t av = a.get_ordered_coordinate(d);
                ordered_coordinate_t bv = b.get_ordered_coordinate(d);
                start.set_ordered_coordinate(d, std::min(av, bv));
                end.set_ordered_coordinate(d, std::max(av, bv));
            }
        } else {
            const auto &center = unique_points[pick(rng)];
            float r = radius_dist(rng);
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                float c = center.get_float_coordinate(d);
                start.set_float_coordinate(d, c - r);
                end.set_float_coordinate(d, c + r);
            }
        }

        std::vector<std::string> expected;
        for (size_t i = 0; i < unique_points.size(); ++i) {
            if (point_in_box(unique_points[i], start, end))
                expected.push_back(canonical_keys[i]);
        }
        std::sort(expected.begin(), expected.end());

        std::vector<std::string> got;
        dawg.SpatialRangeSearch(start, end, &got);
        std::sort(got.begin(), got.end());

        if (got != expected) {
            std::cerr << "Fuzz oracle FAILED at query " << q << "\n";
            std::cerr << "  Expected " << expected.size() << " results, got " << got.size() << "\n";
            std::exit(1);
        }
    }

    std::cout << "test_1024d_fuzz_oracle PASSED\n";
}

void test_1024d_vs_mdtrie_differential()
{
    std::cout << "Running test_1024d_vs_mdtrie_differential (GB=" << GB
              << ", 50 points, 40 queries)...\n";

    using mdtrie_t = md_trie<63, 65536, NUM_DIMENSIONS>;
    mdtrie_t mdtrie;
    CompactDawg<GB> dawg;

    std::mt19937 rng(271828);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> radius_dist(0.1f, 0.5f);

    constexpr int N_POINTS = 50;
    constexpr int N_QUERIES = 40;

    std::vector<data_point<NUM_DIMENSIONS>> points;
    std::vector<std::string> keys;
    points.reserve(N_POINTS);
    keys.reserve(N_POINTS);

    for (int i = 0; i < N_POINTS; ++i) {
        data_point<NUM_DIMENSIONS> p;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            p.set_float_coordinate(d, val_dist(rng));
        points.push_back(p);
        keys.push_back(encode_morton_bitstring(p));
    }

    std::vector<size_t> idx(keys.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });

    std::vector<data_point<NUM_DIMENSIONS>> unique_points;
    std::vector<std::string> unique_keys;
    for (size_t i : idx) {
        if (!unique_keys.empty() && keys[i] == unique_keys.back())
            continue;
        unique_points.push_back(points[i]);
        unique_keys.push_back(keys[i]);
    }

    for (auto &p : unique_points)
        mdtrie.insert_trie(&p);

    std::vector<std::string> sorted_keys = unique_keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    for (const auto &k : sorted_keys)
        dawg.Insert(k);
    dawg.Finish();

    std::uniform_int_distribution<size_t> pick(0, unique_points.size() - 1);

    for (int q = 0; q < N_QUERIES; ++q) {
        data_point<NUM_DIMENSIONS> start, end;

        if (q % 2 == 0) {
            const auto &a = unique_points[pick(rng)];
            const auto &b = unique_points[pick(rng)];
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                ordered_coordinate_t av = a.get_ordered_coordinate(d);
                ordered_coordinate_t bv = b.get_ordered_coordinate(d);
                start.set_ordered_coordinate(d, std::min(av, bv));
                end.set_ordered_coordinate(d, std::max(av, bv));
            }
        } else {
            const auto &center = unique_points[pick(rng)];
            float r = radius_dist(rng);
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                float c = center.get_float_coordinate(d);
                start.set_float_coordinate(d, c - r);
                end.set_float_coordinate(d, c + r);
            }
        }

        std::vector<std::string> dawg_results;
        dawg.SpatialRangeSearch(start, end, &dawg_results);
        std::sort(dawg_results.begin(), dawg_results.end());
        dawg_results.erase(std::unique(dawg_results.begin(), dawg_results.end()),
                           dawg_results.end());

        std::vector<ordered_coordinate_t> md_raw;
        data_point<NUM_DIMENSIONS> md_start = start;
        data_point<NUM_DIMENSIONS> md_end = end;
        mdtrie.range_search_trie(&md_start, &md_end, md_raw);

        std::vector<std::string> md_keys;
        md_keys.reserve(md_raw.size() / NUM_DIMENSIONS);
        for (size_t i = 0; i + NUM_DIMENSIONS <= md_raw.size(); i += NUM_DIMENSIONS) {
            data_point<NUM_DIMENSIONS> p;
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
                p.set_ordered_coordinate(d, md_raw[i + d]);
            md_keys.push_back(canonicalize_for_group_bits<GB>(encode_morton_bitstring(p)));
        }
        std::sort(md_keys.begin(), md_keys.end());
        md_keys.erase(std::unique(md_keys.begin(), md_keys.end()), md_keys.end());

        if (dawg_results != md_keys) {
            std::cerr << "Differential mismatch at query " << q << "\n";
            std::cerr << "  CompactDawg: " << dawg_results.size()
                      << " results, MDTrie: " << md_keys.size() << " results\n";
            std::exit(1);
        }
    }

    std::cout << "test_1024d_vs_mdtrie_differential PASSED\n";
}

void test_1024d_full_range_returns_all()
{
    std::cout << "Running test_1024d_full_range_returns_all (GB=" << GB << ", 30 points)...\n";

    CompactDawg<GB> dawg;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> val_dist(-5.0f, 5.0f);

    constexpr int N = 30;
    std::vector<std::string> keys;
    keys.reserve(N);

    for (int i = 0; i < N; ++i) {
        data_point<NUM_DIMENSIONS> p;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            p.set_float_coordinate(d, val_dist(rng));
        keys.push_back(encode_morton_bitstring(p));
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    std::vector<std::string> canonical;
    for (const auto &k : keys)
        canonical.push_back(canonicalize_for_group_bits<GB>(k));
    std::sort(canonical.begin(), canonical.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    data_point<NUM_DIMENSIONS> lo, hi;
    for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
        lo.set_ordered_coordinate(d, 0U);
        hi.set_ordered_coordinate(d, 0xFFFFFFFFU);
    }

    std::vector<std::string> results;
    dawg.SpatialRangeSearch(lo, hi, &results);
    std::sort(results.begin(), results.end());

    assert(results == canonical);

    std::cout << "test_1024d_full_range_returns_all PASSED\n";
}

void test_1024d_empty_range()
{
    std::cout << "Running test_1024d_empty_range (GB=" << GB << ")...\n";

    CompactDawg<GB> dawg;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> val_dist(0.0f, 0.5f);

    for (int i = 0; i < 20; ++i) {
        data_point<NUM_DIMENSIONS> p;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            p.set_float_coordinate(d, val_dist(rng));
        std::string k = encode_morton_bitstring(p);
        dawg.Insert(k);
    }
    dawg.Finish();

    data_point<NUM_DIMENSIONS> lo, hi;
    for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
        lo.set_float_coordinate(d, 100.0f);
        hi.set_float_coordinate(d, 101.0f);
    }

    std::vector<std::string> results;
    dawg.SpatialRangeSearch(lo, hi, &results);
    assert(results.empty());

    std::cout << "test_1024d_empty_range PASSED\n";
}

void test_1024d_pc_cross_validation()
{
    std::cout << "Running test_1024d_pc_cross_validation (GB=" << GB
              << ", 80 points, 60 queries)...\n";

    CompactDawg<GB, false> dawg_ref;
    CompactDawg<GB, true> dawg_pc;
    std::mt19937 rng(314159);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    constexpr int N_POINTS = 80;
    constexpr int N_QUERIES = 60;

    std::vector<data_point<NUM_DIMENSIONS>> points;
    std::vector<std::string> keys;
    points.reserve(N_POINTS);
    keys.reserve(N_POINTS);

    for (int i = 0; i < N_POINTS; ++i) {
        data_point<NUM_DIMENSIONS> p;
        for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d)
            p.set_float_coordinate(d, val_dist(rng));
        points.push_back(p);
        keys.push_back(encode_morton_bitstring(p));
    }

    std::vector<size_t> idx(keys.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });

    std::vector<std::string> unique_keys;
    std::vector<data_point<NUM_DIMENSIONS>> unique_points;
    for (size_t i : idx) {
        if (!unique_keys.empty() && keys[i] == unique_keys.back())
            continue;
        unique_keys.push_back(keys[i]);
        unique_points.push_back(points[i]);
    }

    std::vector<std::string> sorted_keys = unique_keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    for (const auto &k : sorted_keys) {
        dawg_ref.Insert(k);
        dawg_pc.Insert(k);
    }
    dawg_ref.Finish();
    dawg_pc.Finish();

    std::uniform_int_distribution<size_t> pick(0, unique_points.size() - 1);
    std::uniform_real_distribution<float> radius_dist(0.05f, 0.5f);

    for (int q = 0; q < N_QUERIES; ++q) {
        data_point<NUM_DIMENSIONS> start, end;

        if (q % 2 == 0) {
            const auto &a = unique_points[pick(rng)];
            const auto &b = unique_points[pick(rng)];
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                ordered_coordinate_t av = a.get_ordered_coordinate(d);
                ordered_coordinate_t bv = b.get_ordered_coordinate(d);
                start.set_ordered_coordinate(d, std::min(av, bv));
                end.set_ordered_coordinate(d, std::max(av, bv));
            }
        } else {
            const auto &center = unique_points[pick(rng)];
            float r = radius_dist(rng);
            for (n_dimensions_t d = 0; d < NUM_DIMENSIONS; ++d) {
                float c = center.get_float_coordinate(d);
                start.set_float_coordinate(d, c - r);
                end.set_float_coordinate(d, c + r);
            }
        }

        std::vector<std::string> ref_results, pc_results;
        dawg_ref.SpatialRangeSearch(start, end, &ref_results);
        dawg_pc.SpatialRangeSearch(start, end, &pc_results);

        std::sort(ref_results.begin(), ref_results.end());
        std::sort(pc_results.begin(), pc_results.end());

        if (ref_results != pc_results) {
            std::cerr << "PC cross-validation FAILED at query " << q << "\n";
            std::cerr << "  ref=" << ref_results.size() << " results, pc=" << pc_results.size()
                      << " results\n";
            std::exit(1);
        }
    }

    std::cout << "  ref edges=" << dawg_ref.get_total_edges()
              << "  pc edges=" << dawg_pc.get_total_edges()
              << "  ref size=" << dawg_ref.size_in_bytes()
              << "B  pc size=" << dawg_pc.size_in_bytes() << "B\n";
    std::cout << "test_1024d_pc_cross_validation PASSED\n";
}

int main()
{
    ensure_large_stack();

    std::cout << "=== CompactDawg 1024D Spatial Test Suite ===\n";
    std::cout << "NUM_DIMENSIONS=" << NUM_DIMENSIONS << "  GROUP_BITS=" << GB << "\n\n";

    test_1024d_full_range_returns_all();
    test_1024d_empty_range();
    test_1024d_fuzz_oracle();
    test_1024d_vs_mdtrie_differential();
    test_1024d_pc_cross_validation();

    std::cout << "\n=== 1024D SPATIAL TESTS COMPLETED SUCCESSFULLY ===\n";
    return 0;
}
