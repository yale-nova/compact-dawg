#include "dynamic_dawg.h"
#include "compact_dawg.h"
#include "dawg_segmentation.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// Test helpers
// -----------------------------------------------------------------------

static std::vector<std::string> generate_random_keys(int count, int bit_length, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> bit_dist(0, 1);

    std::vector<std::string> keys;
    keys.reserve(count);
    for (int i = 0; i < count; ++i) {
        std::string key;
        key.reserve(bit_length);
        for (int j = 0; j < bit_length; ++j) {
            key += bit_dist(rng) ? '1' : '0';
        }
        keys.push_back(std::move(key));
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

void test_dynamic_dawg_basic_insertion_uniform()
{
    std::cout << "Running test_dynamic_dawg_basic_insertion_uniform...\n";

    // Use a uniform plan (equivalent to CompactDawg<4>)
    auto plan = dawg_seg::uniform(4, 16);

    DynamicDawg dawg(plan);

    std::vector<std::string> keys = {"0001001000110100", "0001001010001001", "0101011001111000"};
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : keys)
        assert(dawg.Contains(k) == true);

    assert(dawg.Contains("1001100110011001") == false);
    assert(dawg.Contains("000100100011") == false);
    assert(dawg.Contains("00010010001101000101") == false);
    assert(dawg.Contains("0001001000110101") == false);

    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_basic_insertion_uniform PASSED\n";
}

void test_dynamic_dawg_variable_width_plan()
{
    std::cout << "Running test_dynamic_dawg_variable_width_plan...\n";

    // Variable-width plan: [8, 4, 4, 8, 8] = 32 bits
    auto plan = dawg_seg::from_widths({8, 4, 4, 8, 8}, 32);
    assert(plan.valid());

    DynamicDawg dawg(plan);

    std::vector<std::string> keys = {
        "00001010000101000001111000101000",
        "01100011010110000001111000101000",
    };
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : keys)
        assert(dawg.Contains(k) == true);

    assert(dawg.Contains("11111111111111111111111111111111") == false);

    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_variable_width_plan PASSED\n";
}

void test_dynamic_dawg_suffix_merging()
{
    std::cout << "Running test_dynamic_dawg_suffix_merging...\n";

    // Two keys sharing a common suffix — suffix sharing should still work
    auto plan = dawg_seg::uniform(8, 32);
    DynamicDawg dawg(plan);

    std::vector<std::string> keys = {
        "00001010000101000001111000101000",
        "01100011010110000001111000101000",
    };
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    assert(dawg.Contains("00001010000101000001111000101000") == true);
    assert(dawg.Contains("01100011010110000001111000101000") == true);

    // With suffix sharing, edges should be < 8 (4 chunks * 2 keys)
    std::cout << "  edges=" << dawg.get_total_edges() << " (expecting < 8 with sharing)\n";
    assert(dawg.get_total_edges() < 8);

    std::cout << "test_dynamic_dawg_suffix_merging PASSED\n";
}

void test_dynamic_dawg_variable_width_suffix_merging()
{
    std::cout << "Running test_dynamic_dawg_variable_width_suffix_merging...\n";

    // Same suffix-sharing test but with a non-uniform plan
    auto plan = dawg_seg::from_widths({4, 12, 8, 8}, 32);
    assert(plan.valid());
    DynamicDawg dawg(plan);

    std::vector<std::string> keys = {
        "00001010000101000001111000101000",
        "01100011010110000001111000101000",
    };
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : keys)
        assert(dawg.Contains(k) == true);

    assert(dawg.Contains("11111111111111111111111111111111") == false);

    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_variable_width_suffix_merging PASSED\n";
}

void test_dynamic_dawg_range_search_uniform()
{
    std::cout << "Running test_dynamic_dawg_range_search_uniform...\n";

    auto plan = dawg_seg::uniform(4, 16);
    DynamicDawg dawg(plan);

    std::vector<std::string> keys = {
        "0000000000000000", "0000000000000100",
        "0000000000001000", "0000000000001100",
        "0000000000010000",
    };
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    std::vector<std::string> results;
    dawg.LexicographicSearch("0000000000000010", "0000000000001110", &results);

    assert(results.size() == 3);
    assert(results[0] == "0000000000000100");
    assert(results[1] == "0000000000001000");
    assert(results[2] == "0000000000001100");

    results.clear();
    dawg.LexicographicSearch("0000000000000000", "1111111111111111", &results);
    assert(results.size() == 5);

    std::cout << "test_dynamic_dawg_range_search_uniform PASSED\n";
}

void test_dynamic_dawg_range_search_variable()
{
    std::cout << "Running test_dynamic_dawg_range_search_variable...\n";

    // Variable plan: [8, 4, 4] = 16 bits
    auto plan = dawg_seg::from_widths({8, 4, 4}, 16);
    DynamicDawg dawg(plan);

    std::vector<std::string> keys = {
        "0000000000000000", "0000000000000100",
        "0000000000001000", "0000000000001100",
        "0000000000010000",
    };
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    std::vector<std::string> results;
    dawg.LexicographicSearch("0000000000000010", "0000000000001110", &results);

    std::sort(results.begin(), results.end());
    assert(results.size() == 3);
    assert(results[0] == "0000000000000100");
    assert(results[1] == "0000000000001000");
    assert(results[2] == "0000000000001100");

    results.clear();
    dawg.LexicographicSearch("0000000000000000", "1111111111111111", &results);
    assert(results.size() == 5);

    std::cout << "test_dynamic_dawg_range_search_variable PASSED\n";
}

void test_dynamic_dawg_cross_check_contains_vs_v1()
{
    std::cout << "Running test_dynamic_dawg_cross_check_contains_vs_v1...\n";

    constexpr int BIT_LENGTH = 512;
    constexpr int NUM_KEYS = 500;

    auto all_keys = generate_random_keys(NUM_KEYS, BIT_LENGTH, 42);
    size_t split = all_keys.size() / 2;
    std::vector<std::string> inserted(all_keys.begin(), all_keys.begin() + split);
    std::vector<std::string> not_inserted(all_keys.begin() + split, all_keys.end());

    // V1: CompactDawg<16>
    CompactDawg<16> v1;
    for (const auto &k : inserted)
        v1.Insert(k);
    v1.Finish();

    // DynamicDawg: uniform plan equivalent to GB=16
    auto plan = dawg_seg::uniform(16, BIT_LENGTH);
    DynamicDawg dynamic_dawg(plan);
    for (const auto &k : inserted)
        dynamic_dawg.Insert(k);
    dynamic_dawg.Finish();

    // Cross-check: inserted keys
    for (const auto &k : inserted) {
        assert(v1.Contains(k));
        if (!dynamic_dawg.Contains(k)) {
            std::cerr << "DynamicDawg Contains FAILED for inserted key!\n";
            std::exit(1);
        }
    }

    // Cross-check: non-inserted keys
    for (const auto &k : not_inserted) {
        assert(!v1.Contains(k));
        if (dynamic_dawg.Contains(k)) {
            std::cerr << "DynamicDawg Contains FALSE POSITIVE for non-inserted key!\n";
            std::exit(1);
        }
    }

    std::cout << "  V1 edges=" << v1.get_total_edges() << "  size=" << v1.size_in_bytes() << "B\n";
    std::cout << "  DynamicDawg edges=" << dynamic_dawg.get_total_edges()
              << "  size=" << dynamic_dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_cross_check_contains_vs_v1 PASSED\n";
}

void test_dynamic_dawg_cross_check_lexsearch_vs_v1()
{
    std::cout << "Running test_dynamic_dawg_cross_check_lexsearch_vs_v1...\n";

    constexpr int BIT_LENGTH = 256;
    constexpr int NUM_KEYS = 300;

    auto all_keys = generate_random_keys(NUM_KEYS, BIT_LENGTH, 99);
    size_t split = all_keys.size() / 2;
    std::vector<std::string> inserted(all_keys.begin(), all_keys.begin() + split);

    // V1
    CompactDawg<16> v1;
    for (const auto &k : inserted)
        v1.Insert(k);
    v1.Finish();

    // DynamicDawg uniform
    auto plan = dawg_seg::uniform(16, BIT_LENGTH);
    DynamicDawg dynamic_dawg(plan);
    for (const auto &k : inserted)
        dynamic_dawg.Insert(k);
    dynamic_dawg.Finish();

    // Cross-check LexicographicSearch with 20 random ranges
    std::mt19937 rng(12345);
    for (int q = 0; q < 20; ++q) {
        size_t a = rng() % inserted.size();
        size_t b = rng() % inserted.size();
        if (a > b)
            std::swap(a, b);

        const std::string &lo = inserted[a];
        const std::string &hi = inserted[b];

        std::vector<std::string> v1_results, dynamic_dawg_results;
        v1.LexicographicSearch(lo, hi, &v1_results);
        dynamic_dawg.LexicographicSearch(lo, hi, &dynamic_dawg_results);

        std::sort(v1_results.begin(), v1_results.end());
        std::sort(dynamic_dawg_results.begin(), dynamic_dawg_results.end());

        if (v1_results != dynamic_dawg_results) {
            std::cerr << "LexSearch MISMATCH at query " << q
                      << ": v1=" << v1_results.size()
                      << " dynamic=" << dynamic_dawg_results.size() << "\n";
            std::exit(1);
        }
    }

    std::cout << "test_dynamic_dawg_cross_check_lexsearch_vs_v1 PASSED\n";
}

void test_dynamic_dawg_variable_width_fuzzing()
{
    std::cout << "Running test_dynamic_dawg_variable_width_fuzzing...\n";

    constexpr int BIT_LENGTH = 256;
    constexpr int NUM_KEYS = 500;

    auto all_keys = generate_random_keys(NUM_KEYS, BIT_LENGTH, 777);
    size_t split = all_keys.size() / 2;
    std::vector<std::string> inserted(all_keys.begin(), all_keys.begin() + split);
    std::vector<std::string> not_inserted(all_keys.begin() + split, all_keys.end());

    // Variable-width plan: mix of sizes that sum to 256
    // [64, 32, 16, 16, 8, 8, 8, 8, 16, 16, 32, 32]
    std::vector<uint32_t> widths = {64, 32, 16, 16, 8, 8, 8, 8, 16, 16, 32, 32};
    uint32_t sum = 0;
    for (auto w : widths)
        sum += w;
    assert(sum == BIT_LENGTH);

    auto plan = dawg_seg::from_widths(widths, BIT_LENGTH);
    DynamicDawg dawg(plan);

    for (const auto &k : inserted)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : inserted) {
        if (!dawg.Contains(k)) {
            std::cerr << "Fuzz: inserted key not found!\n";
            std::exit(1);
        }
    }

    for (const auto &k : not_inserted) {
        if (dawg.Contains(k)) {
            std::cerr << "Fuzz: non-inserted key returned true!\n";
            std::exit(1);
        }
    }

    // Also verify LexicographicSearch returns all inserted keys
    std::string all_zeros(BIT_LENGTH, '0');
    std::string all_ones(BIT_LENGTH, '1');
    std::vector<std::string> results;
    dawg.LexicographicSearch(all_zeros, all_ones, &results);
    std::sort(results.begin(), results.end());
    assert(results == inserted);

    std::cout << "  " << inserted.size() << " keys inserted OK, "
              << not_inserted.size() << " keys rejected OK\n";
    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_variable_width_fuzzing PASSED\n";
}

void test_dynamic_dawg_single_giant_segment()
{
    std::cout << "Running test_dynamic_dawg_single_giant_segment...\n";

    // Edge case: entire key is one segment
    constexpr int BIT_LENGTH = 64;
    auto plan = dawg_seg::from_widths({64}, 64);
    assert(plan.valid());

    DynamicDawg dawg(plan);

    auto keys = generate_random_keys(50, BIT_LENGTH, 101);
    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : keys)
        assert(dawg.Contains(k) == true);

    // DAWG depth should be 1 (each key is a single edge)
    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_single_giant_segment PASSED\n";
}

void test_dynamic_dawg_all_1bit_segments()
{
    std::cout << "Running test_dynamic_dawg_all_1bit_segments...\n";

    // Edge case: every bit is its own segment
    constexpr int BIT_LENGTH = 32;
    auto plan = dawg_seg::uniform(1, BIT_LENGTH);
    assert(plan.valid());
    assert(plan.depth() == 32);

    DynamicDawg dawg(plan);

    auto keys = generate_random_keys(20, BIT_LENGTH, 202);
    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : keys)
        assert(dawg.Contains(k) == true);

    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_all_1bit_segments PASSED\n";
}

void test_dynamic_dawg_path_compression()
{
    std::cout << "Running test_dynamic_dawg_path_compression...\n";

    constexpr int BIT_LENGTH = 256;
    constexpr int NUM_KEYS = 200;

    auto inserted = generate_random_keys(NUM_KEYS, BIT_LENGTH, 333);
    size_t split = inserted.size() / 2;
    std::vector<std::string> test_keys(inserted.begin(), inserted.begin() + split);
    std::vector<std::string> negative_keys(inserted.begin() + split, inserted.end());

    std::vector<uint32_t> widths = {64, 32, 16, 16, 16, 16, 32, 64};
    auto plan = dawg_seg::from_widths(widths, BIT_LENGTH);

    // Without path compression
    DynamicDawg ref(plan, false);
    for (const auto &k : test_keys)
        ref.Insert(k);
    ref.Finish();

    // With path compression
    DynamicDawg pc(plan, true);
    for (const auto &k : test_keys)
        pc.Insert(k);
    pc.Finish();

    // Cross-check
    for (const auto &k : test_keys) {
        assert(ref.Contains(k));
        if (!pc.Contains(k)) {
            std::cerr << "PC Contains FAILED for inserted key!\n";
            std::exit(1);
        }
    }
    for (const auto &k : negative_keys) {
        assert(!ref.Contains(k));
        if (pc.Contains(k)) {
            std::cerr << "PC Contains FALSE POSITIVE!\n";
            std::exit(1);
        }
    }

    // PC should have fewer or equal edges
    assert(pc.get_total_edges() <= ref.get_total_edges());

    // LexSearch cross-check
    std::mt19937 rng(55555);
    for (int q = 0; q < 20; ++q) {
        size_t a = rng() % test_keys.size();
        size_t b = rng() % test_keys.size();
        if (a > b)
            std::swap(a, b);

        std::vector<std::string> ref_results, pc_results;
        ref.LexicographicSearch(test_keys[a], test_keys[b], &ref_results);
        pc.LexicographicSearch(test_keys[a], test_keys[b], &pc_results);

        std::sort(ref_results.begin(), ref_results.end());
        std::sort(pc_results.begin(), pc_results.end());

        if (ref_results != pc_results) {
            std::cerr << "Path compression LexSearch MISMATCH at query " << q << "\n";
            std::exit(1);
        }
    }

    std::cout << "  ref edges=" << ref.get_total_edges() << "  size=" << ref.size_in_bytes() << "B"
              << "  pc edges=" << pc.get_total_edges() << "  size=" << pc.size_in_bytes() << "B"
              << "  ratio=" << std::fixed << std::setprecision(1)
              << (100.0 * pc.size_in_bytes() / ref.size_in_bytes()) << "%\n";
    std::cout << "test_dynamic_dawg_path_compression PASSED\n";
}

void test_dynamic_dawg_greedy_segmentation()
{
    std::cout << "Running test_dynamic_dawg_greedy_segmentation...\n";

    constexpr int BIT_LENGTH = 128;
    constexpr int NUM_KEYS = 200;

    auto keys = generate_random_keys(NUM_KEYS, BIT_LENGTH, 444);

    // Use greedy segmentation (Strategy B)
    auto plan = dawg_seg::greedy(keys, BIT_LENGTH, 0.5);
    assert(plan.valid());

    std::cout << "  greedy plan: " << plan.depth() << " segments, widths = [";
    for (size_t i = 0; i < plan.widths.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << plan.widths[i];
    }
    std::cout << "]\n";

    DynamicDawg dawg(plan);
    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    // Verify all keys present
    for (const auto &k : keys)
        assert(dawg.Contains(k));

    // Verify a wrong key is absent
    std::string wrong = keys[0];
    wrong[BIT_LENGTH / 2] = (wrong[BIT_LENGTH / 2] == '1') ? '0' : '1';
    // Might still be in the set by chance; re-sort to check
    bool actually_in = std::binary_search(keys.begin(), keys.end(), wrong);
    assert(dawg.Contains(wrong) == actually_in);

    std::cout << "  edges=" << dawg.get_total_edges()
              << "  size=" << dawg.size_in_bytes() << "B\n";
    std::cout << "test_dynamic_dawg_greedy_segmentation PASSED\n";
}

void test_dynamic_dawg_cost_aware_segmentation()
{
    std::cout << "Running test_dynamic_dawg_cost_aware_segmentation...\n";

    constexpr int BIT_LENGTH = 128;
    std::vector<std::string> keys;
    keys.reserve(64);

    for (int i = 0; i < 64; ++i) {
        std::string key(32, '0');
        for (int bit = 0; bit < 32; ++bit) {
            key += ((i >> (bit % 6)) & 1) ? '1' : '0';
        }
        key += (i % 4 == 0 || i % 4 == 1) ? std::string(64, '0') : std::string(64, '1');
        keys.push_back(std::move(key));
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    auto plan = dawg_seg::greedy_cost_aware(keys, BIT_LENGTH, 0.5);
    assert(plan.valid());
    assert(!plan.widths.empty());
    const uint32_t min_width = *std::min_element(plan.widths.begin(), plan.widths.end());
    const uint32_t max_width = *std::max_element(plan.widths.begin(), plan.widths.end());
    assert(plan.depth() > 1);
    assert(max_width >= 16);
    assert(min_width <= 8);

    DynamicDawg dawg(plan, true);
    for (const auto &k : keys)
        dawg.Insert(k);
    dawg.Finish();

    for (const auto &k : keys)
        assert(dawg.Contains(k));

    std::cout << "  cost-aware plan depth=" << plan.depth()
              << " avg_label_bits=" << std::fixed << std::setprecision(1)
              << dawg.get_average_label_bits()
              << " metadata_share=" << dawg.get_label_metadata_share() << "\n";
    std::cout << "test_dynamic_dawg_cost_aware_segmentation PASSED\n";
}

void test_segment_plan_utilities()
{
    std::cout << "Running test_segment_plan_utilities...\n";

    // Test from_widths
    auto plan1 = dawg_seg::from_widths({8, 16, 8}, 32);
    assert(plan1.valid());
    assert(plan1.depth() == 3);
    assert(plan1.bit_offsets[0] == 0);
    assert(plan1.bit_offsets[1] == 8);
    assert(plan1.bit_offsets[2] == 24);

    // Test uniform
    auto plan2 = dawg_seg::uniform(16, 64);
    assert(plan2.valid());
    assert(plan2.depth() == 4);
    for (auto w : plan2.widths)
        assert(w == 16);

    // Test chunk_key
    std::string key = "01234567890123456789012345678901";
    auto chunks = plan1.chunk_key(key);
    assert(chunks.size() == 3);
    assert(chunks[0] == "01234567");
    assert(chunks[1] == "8901234567890123");
    assert(chunks[2] == "45678901");

    // Test invalid plan
    auto plan3 = dawg_seg::from_widths({8, 8}, 32); // sum=16 != 32
    assert(!plan3.valid());

    // Cost-aware planner should still produce a valid covering plan
    auto planner_keys = std::vector<std::string>{
        "00000000000000001111111111111111",
        "00000000000000000000111100001111",
        "00000000000000001111000011110000",
        "00000000000000000000000000000000",
    };
    std::sort(planner_keys.begin(), planner_keys.end());
    auto plan4 = dawg_seg::greedy_cost_aware(planner_keys, 32, 0.5);
    assert(plan4.valid());
    assert(!plan4.widths.empty());
    for (auto w : plan4.widths)
        assert(w > 0);

    auto plan5 = dawg_seg::greedy_cost_aware_min_width(planner_keys, 32, 0.5, 4);
    assert(plan5.valid());
    for (auto w : plan5.widths)
        assert(w >= 4 || w == plan5.widths.back());

    auto plan6 = dawg_seg::phase_aware(128, 0.20, 0.70, 64, 8, 32);
    assert(plan6.valid());
    assert(plan6.total_bits == 128);
    uint32_t phase_sum = 0;
    for (auto w : plan6.widths)
        phase_sum += w;
    assert(phase_sum == 128);

    std::cout << "test_segment_plan_utilities PASSED\n";
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main()
{
    std::cout << "=== DynamicDawg Test Suite ===\n\n";

    std::cout << "--- Segmentation Utilities ---\n";
    test_segment_plan_utilities();

    std::cout << "\n--- Basic DynamicDawg Tests ---\n";
    test_dynamic_dawg_basic_insertion_uniform();
    test_dynamic_dawg_variable_width_plan();
    test_dynamic_dawg_suffix_merging();
    test_dynamic_dawg_variable_width_suffix_merging();

    std::cout << "\n--- Range Search Tests ---\n";
    test_dynamic_dawg_range_search_uniform();
    test_dynamic_dawg_range_search_variable();

    std::cout << "\n--- Cross-Validation vs V1 ---\n";
    test_dynamic_dawg_cross_check_contains_vs_v1();
    test_dynamic_dawg_cross_check_lexsearch_vs_v1();

    std::cout << "\n--- Fuzzing & Edge Cases ---\n";
    test_dynamic_dawg_variable_width_fuzzing();
    test_dynamic_dawg_single_giant_segment();
    test_dynamic_dawg_all_1bit_segments();

    std::cout << "\n--- Path Compression ---\n";
    test_dynamic_dawg_path_compression();

    std::cout << "\n--- Greedy Segmentation (Strategy B) ---\n";
    test_dynamic_dawg_greedy_segmentation();
    test_dynamic_dawg_cost_aware_segmentation();

    std::cout << "\n=== ALL DYNAMICDAWG TESTS COMPLETED SUCCESSFULLY ===\n";
    return 0;
}
