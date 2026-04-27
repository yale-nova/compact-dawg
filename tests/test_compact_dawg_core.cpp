#include "compact_dawg.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

void test_basic_insertion_and_query()
{
    std::cout << "Running test_basic_insertion_and_query...\n";

    CompactDawg<4> dawg;

    std::vector<std::string> keys = {"0001001000110100", "0001001010001001", "0101011001111000"};

    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys) {
        dawg.Insert(k);
    }
    dawg.Finish();

    for (const auto &k : keys) {
        assert(dawg.Contains(k) == true);
    }

    assert(dawg.Contains("1001100110011001") == false);
    assert(dawg.Contains("000100100011") == false);
    assert(dawg.Contains("001000110100") == false);
    assert(dawg.Contains("00010010001101000101") == false);
    assert(dawg.Contains("0001001000110101") == false);

    std::cout << "test_basic_insertion_and_query PASSED\n";
}

void test_suffix_merging()
{
    std::cout << "Running test_suffix_merging...\n";

    CompactDawg<8> dawg;
    std::vector<std::string> keys = {"00001010000101000001111000101000",
                                     "01100011010110000001111000101000"};

    std::sort(keys.begin(), keys.end());
    for (const auto &k : keys) {
        dawg.Insert(k);
    }
    dawg.Finish();

    assert(dawg.Contains("00001010000101000001111000101000") == true);
    assert(dawg.Contains("01100011010110000001111000101000") == true);
    assert(dawg.get_total_edges() == 6);

    std::cout << "test_suffix_merging PASSED (Graph strictly optimal)\n";
}

void test_suffix_collapse_storage_estimate()
{
    std::cout << "Running test_suffix_collapse_storage_estimate...\n";

    assert(CompactDawg<8>::OffsetBitsForEdgeCount(0) == 1);
    assert(CompactDawg<8>::OffsetBitsForEdgeCount(1) == 1);
    assert(CompactDawg<8>::OffsetBitsForEdgeCount(2) == 2);
    assert(CompactDawg<8>::OffsetBitsForEdgeCount(3) == 2);
    assert(CompactDawg<8>::OffsetBitsForEdgeCount(4) == 3);
    assert(CompactDawg<8>::OffsetBitsForEdgeCount(255) == 8);
    assert(CompactDawg<8>::OffsetBitsForEdgeCount(256) == 9);

    CompactDawg<8, false, true> shared;
    std::vector<std::string> shared_keys = {"00001010000101000001111000101000",
                                            "01100011010110000001111000101000"};
    std::sort(shared_keys.begin(), shared_keys.end());
    for (const auto &k : shared_keys) {
        shared.Insert(k);
    }
    shared.Finish();

    const auto &shared_stats = shared.GetSharingStats();
    const size_t shared_pre_bytes =
        CompactDawg<8>::PackedFixedWidthBytesForEdgeCount(shared_stats.trie_edges);
    assert(shared_stats.memo_hits > 0);
    assert(shared_pre_bytes >= shared.size_in_bytes());

    CompactDawg<4, false, true> unshared;
    std::vector<std::string> unshared_keys = {"0000", "1111"};
    for (const auto &k : unshared_keys) {
        unshared.Insert(k);
    }
    unshared.Finish();

    const auto &unshared_stats = unshared.GetSharingStats();
    const size_t unshared_pre_bytes =
        CompactDawg<4>::PackedFixedWidthBytesForEdgeCount(unshared_stats.trie_edges);
    assert(unshared_stats.memo_hits == 0);
    assert(unshared_pre_bytes == unshared.size_in_bytes());

    std::cout << "test_suffix_collapse_storage_estimate PASSED\n";
}

void test_random_fuzzing()
{
    std::cout << "Running test_random_fuzzing...\n";

    CompactDawg<16> dawg;

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 65535);

    const int NUM_KEYS = 1000;
    const int BIT_LENGTH = 512;

    std::vector<std::string> all_keys;
    for (int i = 0; i < NUM_KEYS; ++i) {
        std::string key;
        key.reserve(BIT_LENGTH);
        for (int j = 0; j < BIT_LENGTH; ++j) {
            key += (dist(rng) % 2 == 0) ? '0' : '1';
        }
        all_keys.push_back(std::move(key));
    }

    std::sort(all_keys.begin(), all_keys.end());
    all_keys.erase(std::unique(all_keys.begin(), all_keys.end()), all_keys.end());

    size_t split_index = all_keys.size() / 2;
    for (size_t i = 0; i < split_index; ++i) {
        dawg.Insert(all_keys[i]);
    }
    dawg.Finish();

    for (size_t i = 0; i < split_index; ++i) {
        if (!dawg.Contains(all_keys[i])) {
            std::cerr << "Fuzz failed: Inserted key not found!\n";
            std::exit(1);
        }
    }

    for (size_t i = split_index; i < all_keys.size(); ++i) {
        if (dawg.Contains(all_keys[i])) {
            std::cerr << "Fuzz failed: Uninserted key returned true!\n";
            std::exit(1);
        }
    }

    std::cout << "test_random_fuzzing PASSED\n";
}

void test_range_search()
{
    std::cout << "Running test_range_search...\n";

    CompactDawg<4> dawg;
    std::vector<std::string> keys = {"0000000000000000", "0000000000000100", "0000000000001000",
                                     "0000000000001100", "0000000000010000"};

    std::sort(keys.begin(), keys.end());
    for (const auto &k : keys) {
        dawg.Insert(k);
    }
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

    std::cout << "test_range_search PASSED\n";
}

void test_dynamic_resizing()
{
    std::cout << "Running test_dynamic_resizing...\n";

    CompactDawg<4> small_dawg;
    std::vector<std::string> small_keys = {"00000000", "00000001"};
    for (const auto &k : small_keys) {
        small_dawg.Insert(k);
    }
    small_dawg.Finish();

    CompactDawg<4> large_dawg;
    std::vector<std::string> large_keys;
    for (int i = 0; i < 1000; ++i) {
        std::string key;
        key.reserve(16);
        int val = i;
        for (int j = 0; j < 16; ++j) {
            key = ((val & 1) ? "1" : "0") + key;
            val >>= 1;
        }
        large_keys.push_back(std::move(key));
    }
    std::sort(large_keys.begin(), large_keys.end());
    for (const auto &k : large_keys) {
        large_dawg.Insert(k);
    }
    large_dawg.Finish();

    std::cout << "Small DAWG offset bits: " << (int)small_dawg.get_offset_bits() << "\n";
    std::cout << "Large DAWG offset bits: " << (int)large_dawg.get_offset_bits() << "\n";
    assert(small_dawg.get_offset_bits() < large_dawg.get_offset_bits());

    std::cout << "test_dynamic_resizing PASSED\n";
}

void test_large_group_bits()
{
    std::cout << "Running test_large_group_bits...\n";

    CompactDawg<1024> dawg;

    std::string key1;
    std::string key2;
    key1.reserve(2048);
    key2.reserve(2048);
    for (int i = 0; i < 2048; i++) {
        key1 += (i % 3 == 0) ? '1' : '0';
        key2 += (i % 5 == 0) ? '1' : '0';
    }

    std::vector<std::string> keys = {key1, key2};
    std::sort(keys.begin(), keys.end());

    for (const auto &k : keys) {
        dawg.Insert(k);
    }
    dawg.Finish();

    assert(dawg.Contains(key1) == true);
    assert(dawg.Contains(key2) == true);

    std::string key3 = key1;
    key3[1023] = (key3[1023] == '1') ? '0' : '1';
    assert(dawg.Contains(key3) == false);

    std::string key4 = key2;
    key4[2047] = (key4[2047] == '1') ? '0' : '1';
    assert(dawg.Contains(key4) == false);

    std::cout << "test_large_group_bits PASSED\n";
}

// --- Path Compression Cross-Validation Tests ---

template <uint32_t GB>
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

void test_pc_single_key()
{
    std::cout << "Running test_pc_single_key...\n";

    std::string key = "00110100010111000011010001011100";
    CompactDawg<8, false> ref;
    CompactDawg<8, true> pc;

    ref.Insert(key);
    ref.Finish();
    pc.Insert(key);
    pc.Finish();

    assert(ref.Contains(key));
    assert(pc.Contains(key));

    // The entire path is a single chain with no branching, in-degree 1 everywhere.
    // With path compression, this should collapse to 1 edge.
    assert(pc.get_total_edges() == 1);
    assert(pc.get_total_edges() < ref.get_total_edges());
    assert(pc.size_in_bytes() <= ref.size_in_bytes());

    std::string wrong = key;
    wrong[0] = (wrong[0] == '1') ? '0' : '1';
    assert(!ref.Contains(wrong));
    assert(!pc.Contains(wrong));

    std::cout << "  ref edges=" << ref.get_total_edges()
              << "  pc edges=" << pc.get_total_edges()
              << "  ref size=" << ref.size_in_bytes()
              << "  pc size=" << pc.size_in_bytes() << "\n";
    std::cout << "test_pc_single_key PASSED\n";
}

void test_pc_shared_suffix()
{
    std::cout << "Running test_pc_shared_suffix...\n";

    // Two keys sharing a suffix: the shared node has in-degree 2, must NOT be compressed.
    CompactDawg<8, false> ref;
    CompactDawg<8, true> pc;
    std::vector<std::string> keys = {"00001010000101000001111000101000",
                                     "01100011010110000001111000101000"};
    std::sort(keys.begin(), keys.end());
    for (const auto &k : keys) {
        ref.Insert(k);
        pc.Insert(k);
    }
    ref.Finish();
    pc.Finish();

    for (const auto &k : keys) {
        assert(ref.Contains(k));
        assert(pc.Contains(k));
    }

    assert(!pc.Contains("11111111111111111111111111111111"));
    assert(pc.get_total_edges() < ref.get_total_edges());

    std::cout << "  ref edges=" << ref.get_total_edges()
              << "  pc edges=" << pc.get_total_edges() << "\n";
    std::cout << "test_pc_shared_suffix PASSED\n";
}

void test_pc_all_same_prefix()
{
    std::cout << "Running test_pc_all_same_prefix...\n";

    // Keys share a long common prefix, then diverge. The prefix chain compresses.
    CompactDawg<4, false> ref;
    CompactDawg<4, true> pc;
    std::vector<std::string> keys = {
        "0000000000000000",
        "0000000000000001",
        "0000000000000010",
        "0000000000000011",
    };
    std::sort(keys.begin(), keys.end());
    for (const auto &k : keys) {
        ref.Insert(k);
        pc.Insert(k);
    }
    ref.Finish();
    pc.Finish();

    for (const auto &k : keys) {
        assert(ref.Contains(k));
        assert(pc.Contains(k));
    }
    assert(!ref.Contains("1111111111111111"));
    assert(!pc.Contains("1111111111111111"));
    assert(pc.get_total_edges() < ref.get_total_edges());

    std::cout << "  ref edges=" << ref.get_total_edges()
              << "  pc edges=" << pc.get_total_edges() << "\n";
    std::cout << "test_pc_all_same_prefix PASSED\n";
}

void test_path_compression_validation()
{
    std::cout << "Running test_path_compression_validation (Contains + LexSearch)...\n";

    constexpr uint32_t GB = 16;
    constexpr int BIT_LENGTH = 512;
    constexpr int NUM_KEYS = 500;

    auto all_keys = generate_random_keys<GB>(NUM_KEYS, BIT_LENGTH, 12345);
    size_t split = all_keys.size() / 2;
    std::vector<std::string> inserted(all_keys.begin(), all_keys.begin() + split);
    std::vector<std::string> not_inserted(all_keys.begin() + split, all_keys.end());

    CompactDawg<GB, false> ref;
    CompactDawg<GB, true> pc;
    for (const auto &k : inserted) {
        ref.Insert(k);
        pc.Insert(k);
    }
    ref.Finish();
    pc.Finish();

    // 1. Cross-check Contains() for inserted keys
    for (const auto &k : inserted) {
        assert(ref.Contains(k));
        if (!pc.Contains(k)) {
            std::cerr << "PC Contains FAILED for inserted key!\n";
            std::exit(1);
        }
    }

    // 2. Cross-check Contains() for non-inserted keys
    for (const auto &k : not_inserted) {
        assert(!ref.Contains(k));
        if (pc.Contains(k)) {
            std::cerr << "PC Contains FALSE POSITIVE for non-inserted key!\n";
            std::exit(1);
        }
    }

    // 3. Cross-check LexicographicSearch() with 20 random ranges
    std::mt19937 rng(99999);
    for (int q = 0; q < 20; ++q) {
        size_t a = rng() % inserted.size();
        size_t b = rng() % inserted.size();
        if (a > b) std::swap(a, b);

        const std::string &lo = inserted[a];
        const std::string &hi = inserted[b];

        std::vector<std::string> ref_results, pc_results;
        ref.LexicographicSearch(lo, hi, &ref_results);
        pc.LexicographicSearch(lo, hi, &pc_results);

        std::sort(ref_results.begin(), ref_results.end());
        std::sort(pc_results.begin(), pc_results.end());

        if (ref_results != pc_results) {
            std::cerr << "LexicographicSearch MISMATCH at query " << q
                      << ": ref=" << ref_results.size() << " pc=" << pc_results.size() << "\n";
            std::exit(1);
        }
    }

    // 4. Verify compression
    assert(pc.get_total_edges() <= ref.get_total_edges());
    assert(pc.size_in_bytes() <= ref.size_in_bytes());

    std::cout << "  ref edges=" << ref.get_total_edges() << "  size=" << ref.size_in_bytes()
              << "B  pc edges=" << pc.get_total_edges() << "  size=" << pc.size_in_bytes() << "B"
              << "  ratio=" << std::fixed << std::setprecision(2)
              << (100.0 * pc.size_in_bytes() / ref.size_in_bytes()) << "%\n";
    std::cout << "test_path_compression_validation PASSED\n";
}

int main()
{
    std::cout << "=== CompactDawg Core Test Suite ===\n";

    test_basic_insertion_and_query();
    test_suffix_merging();
    test_suffix_collapse_storage_estimate();
    test_range_search();
    test_random_fuzzing();
    test_dynamic_resizing();
    test_large_group_bits();

    std::cout << "\n=== Path Compression Tests ===\n";
    test_pc_single_key();
    test_pc_shared_suffix();
    test_pc_all_same_prefix();
    test_path_compression_validation();

    std::cout << "=== ALL CORE TESTS COMPLETED SUCCESSFULLY ===\n";
    return 0;
}
