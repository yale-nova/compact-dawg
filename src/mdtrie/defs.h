#ifndef MD_TRIE_DEFS_H
#define MD_TRIE_DEFS_H

#include <assert.h>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "morton_key.h"
#include "ordered_types.h"
#include <map>

#define DATA_FOLDER_PATH "./data/"
#define PROJECT_ROOT_PATH "./"

// Please define NUM_DIMENSIONS to the number of dimensions you want _BEFORE_ including this header
// file!
//
// AKA:
// #define NUM_DIMENSIONS 1024
// #include "defs.h"
// ...rest of your C++ code...
#ifndef NUM_DIMENSIONS
// Default to the benchmarked 1024d configuration when the header is parsed
// outside a translation unit-specific build context. Files that need another
// dimensionality should still define NUM_DIMENSIONS explicitly before including
// this header.
#define NUM_DIMENSIONS 1024
#endif

// TYPES USED FOR TRIE'S IN-MEMORY REPRESENTATION:

// Counts the number of data points in a trie (and number of tree blocks, etc). Like size_t.
typedef uint64_t n_leaves_t;

// Represents a bit position within the compressed bitmap arrays. An offset type for bit ops!
typedef uint64_t node_bitmap_pos_t;
// Represents trie depth (aka: number of trie levels). Saves memory!
typedef uint8_t trie_level_t;
// Tracks the number of active dimensions at any given level. We support 2^{64} dims...
typedef uint64_t n_dimensions_t;

// This is just a convenience type to avoid writing "morton_key<128>" all the
// time.
//
// Represents a single morton-encoded symbol, at a single level of the trie!
// AKA: if an edge has morton_t of `0100` and there are four active dimensions,
//      then that edge reveals one bit for every dimension.
typedef morton_key<NUM_DIMENSIONS> morton_t;
// Indexes into a treeblock in pre-order traversal numbering.
// Acts as an index for individual nodes. In terms of types:
// `treeblock_node_pos_traversal_order[node_pos_t] = trie_node`
typedef uint64_t node_pos_t;
constexpr node_pos_t null_node = (node_pos_t)-1;

template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION> class tree_block;
template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION>
class disk_tree_block;

template <n_dimensions_t DIMENSION> class data_point;

/**
 * node_info and subtree info are used when splitting the treeblock to create a
 * new frontier node node_info stores the additional information of number of
 * children each node has subtree_info stores the additional information of the
 * subtree size of that node
 */

template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION>
struct frontier_node {
    node_pos_t preorder_;
    tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *child_treeblock;
};

template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION>
struct disk_frontier_node {
    node_pos_t preorder_;
    disk_tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *child_treeblock;
};

struct node_split_info {
    node_pos_t node;
    node_pos_t node_count;
    node_pos_t node_pos;
    node_pos_t node_bits;
    node_pos_t frontier_node_pos;
    node_pos_t frontier_count;
    trie_level_t node_depth;
};

struct subtree_info {
    node_pos_t node_count;
    node_pos_t node_bits;
    node_pos_t frontier_count;
};

struct IndexPtrPair {
    morton_t index;
    uint64_t ptr;
};

// TRIE-STATE RELATED GLOBAL VARIABLES.

// Total number of points in the dataset. TODO remove.
n_leaves_t total_points_count = 0;

// Maximum depth (# bits) of the whole MDTrie data structure.
constexpr trie_level_t MAX_TRIE_DEPTH = 32;

// The maximum depth of the upper (hashmap-based) layers of the MDTrie.
constexpr trie_level_t MAX_TRIE_HASHMAP_DEPTH = 1;
// The maximum depth of the lower (treeblock based) segment of the MDTrie.
constexpr trie_level_t MAX_TRIE_TREEBLOCK_DEPTH = MAX_TRIE_DEPTH - MAX_TRIE_HASHMAP_DEPTH;
// TODO: rename this to MAX_TRIE_NODES for consistency.
constexpr n_leaves_t MAX_TREE_NODES = 512;

std::map<void *, void *> old_ptr_to_new_ptr;
std::map<void *, size_t> ptr_to_file_offset;
size_t current_file_offset = 0;

// PERFORMANCE RELATED GLOBAL VARS

// Cache primary key -> treeblock mappings.
bool enable_client_cache_pkey_mapping = false;

// DEBUGGING RELATED GLOBAL VARS

n_leaves_t treeblock_ctr = 0;
int current_dataset_idx = 0;
int lookup_scanned_nodes = 0;
uint64_t bare_minimum_count = 0;
uint64_t checked_points_count = 0;

// DISK SERIALIZATION-RELATED GLOBAL VARIABLES

// storing the mapping between pointer address in memory and the proposed disk
// storage offset. Used in disk serialization.
// This var is a hack, so we don't have to pass arguments around...
std::unordered_map<uint64_t, uint64_t> pointers_to_offsets_map;

// the last write offset of the file
uint64_t current_offset = 0;

constexpr node_pos_t constexpr_log2(node_pos_t n)
{
    node_pos_t result = 0;
    while (n > 1) {
        n /= 2;
        ++result;
    }
    return result;
}

// state management. Set as true when the current (global) mdtrie has already been
// deserialized (=> has become read-only).
bool DESERIALIZED_MDTRIE = false;

// The current serialization / deserialization implementation uses global variables in many places.
// (pointers_to_offsets_map, current_offset, DESERIALIZED_MDTRIE). They don't clean themselves up
// properly...so calling md_trie.serialize() twice will fail with assertion errors >:P
//
// Per-layer storage tracking
struct LayerStorageStats {
    // For sorted array levels (level < MAX_TRIE_HASHMAP_DEPTH):
    uint64_t num_entries = 0; // number of child trie_nodes
    uint64_t entry_bytes = 0; // sizeof(IndexPtrPair) * entries + sizeof(size_t)

    // For treeblock levels (level >= MAX_TRIE_HASHMAP_DEPTH):
    // Data bits (track in bits for accuracy):
    uint64_t data_bits_collapsed = 0;   // data bits from collapsed nodes
    uint64_t data_bits_uncollapsed = 0; // data bits from uncollapsed nodes
    uint64_t data_padding_bits = 0;     // padding for data_ array (only at treeblock hosting level)
    // Flag bits:
    uint64_t flag_bits = 0;         // flag bits (one per node at this level)
    uint64_t flag_padding_bits = 0; // padding for flag_ array (only at treeblock hosting level)
    // Node distribution:
    uint64_t collapsed_nodes = 0;   // nodes where node_is_collapsed() == true
    uint64_t uncollapsed_nodes = 0; // nodes where node_is_collapsed() == false
    uint64_t frontier_nodes = 0;    // frontier boundary nodes
    uint64_t num_treeblocks = 0;    // number of treeblocks hosted at this level

    // Metadata (struct overhead, frontiers):
    uint64_t metadata_bytes = 0; // trie_node/tree_block/frontier structs
};

inline std::vector<LayerStorageStats> layer_stats(MAX_TRIE_DEPTH + 1);

// Statistics collection for charting
inline std::vector<uint64_t> uncollapsed_children_counts; // children count per uncollapsed node
inline std::vector<uint64_t> uncollapsed_node_bits;       // bits per uncollapsed node
inline std::vector<uint64_t> treeblock_node_counts;       // node count per treeblock
inline std::vector<uint64_t> treeblock_data_bits;         // data bits per treeblock

inline void reset_layer_stats()
{
    for (auto &s : layer_stats) {
        s = LayerStorageStats{};
    }
    uncollapsed_children_counts.clear();
    uncollapsed_node_bits.clear();
    treeblock_node_counts.clear();
    treeblock_data_bits.clear();
}

// Export uncollapsed node children counts to file (one count per line)
inline void export_uncollapsed_children_stats(const char *filename)
{
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    for (uint64_t count : uncollapsed_children_counts) {
        out << count << "\n";
    }
    out.close();
    std::cout << "Exported " << uncollapsed_children_counts.size()
              << " uncollapsed node children counts to " << filename << std::endl;
}

// Export uncollapsed node bits to file (one count per line)
inline void export_uncollapsed_bits_stats(const char *filename)
{
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    for (uint64_t bits : uncollapsed_node_bits) {
        out << bits << "\n";
    }
    out.close();
    std::cout << "Exported " << uncollapsed_node_bits.size() << " uncollapsed node bits to "
              << filename << std::endl;
}

// Export treeblock node counts to file (one count per line)
inline void export_treeblock_node_stats(const char *filename)
{
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    for (uint64_t count : treeblock_node_counts) {
        out << count << "\n";
    }
    out.close();
    std::cout << "Exported " << treeblock_node_counts.size() << " treeblock node counts to "
              << filename << std::endl;
}

// Export treeblock data bits to file (one count per line)
inline void export_treeblock_data_bits_stats(const char *filename)
{
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    for (uint64_t bits : treeblock_data_bits) {
        out << bits << "\n";
    }
    out.close();
    std::cout << "Exported " << treeblock_data_bits.size() << " treeblock data bits to " << filename
              << std::endl;
}

// This function is a hack to quickly reset the vars and permit re-serialization.
//
// TODO(yash): refactor these so they clean themselves up properly.
void reset_mdtrie_serialization_vars()
{
    pointers_to_offsets_map.clear();
    current_offset = 0;
    reset_layer_stats();
}

inline void print_layer_stats(uint64_t total_storage)
{
    std::cout << "\n=== Sorted Array Levels (0 to " << (int)(MAX_TRIE_HASHMAP_DEPTH - 1)
              << ") ===" << std::endl;
    std::cout << "Level\tTotal_Bytes\t%\tEntries\tEntry_Bytes\tMetadata" << std::endl;
    for (trie_level_t l = 0; l < MAX_TRIE_HASHMAP_DEPTH; l++) {
        const auto &s = layer_stats[l];
        if (s.num_entries > 0 || s.metadata_bytes > 0) {
            uint64_t level_total = s.entry_bytes + s.metadata_bytes;
            double pct = total_storage > 0 ? (100.0 * level_total / total_storage) : 0.0;
            std::cout << (int)l << "\t" << level_total << "\t" << std::fixed << std::setprecision(2)
                      << pct << "%\t" << s.num_entries << "\t" << s.entry_bytes << "\t"
                      << s.metadata_bytes << std::endl;
        }
    }

    std::cout << "\n=== Treeblock Levels (" << (int)MAX_TRIE_HASHMAP_DEPTH << " to "
              << (int)(MAX_TRIE_DEPTH - 1) << ") ===" << std::endl;
    std::cout << "Level\tTotal\t%\tCollapsed\tUncollapsed\tFrontier\tDataB_C\tDataB_U\tFlagB\t#"
                 "TB\tPadB\tMetaB"
              << std::endl;
    for (trie_level_t l = MAX_TRIE_HASHMAP_DEPTH; l < MAX_TRIE_DEPTH; l++) {
        const auto &s = layer_stats[l];
        uint64_t data_bits = s.data_bits_collapsed + s.data_bits_uncollapsed;
        if (data_bits > 0 || s.flag_bits > 0 || s.metadata_bytes > 0) {
            // Per-level assertion: flag_bits == collapsed + uncollapsed + frontier
            assert(s.flag_bits == s.collapsed_nodes + s.uncollapsed_nodes + s.frontier_nodes);

            // Compute bytes from bits (padding only at hosting levels)
            uint64_t all_bits = data_bits + s.flag_bits + s.data_padding_bits + s.flag_padding_bits;
            uint64_t level_bytes = all_bits / 8 + s.metadata_bytes;

            // Compute individual byte components for display
            uint64_t data_bytes_c = s.data_bits_collapsed / 8;
            uint64_t data_bytes_u = s.data_bits_uncollapsed / 8;
            uint64_t flag_bytes = s.flag_bits / 8;
            uint64_t padding_bytes = (s.data_padding_bits + s.flag_padding_bits) / 8;

            double pct = total_storage > 0 ? (100.0 * level_bytes / total_storage) : 0.0;
            std::cout << (int)l << "\t" << level_bytes << "\t" << std::fixed << std::setprecision(2)
                      << pct << "%\t" << s.collapsed_nodes << "\t" << s.uncollapsed_nodes << "\t"
                      << s.frontier_nodes << "\t" << data_bytes_c << "\t" << data_bytes_u << "\t"
                      << flag_bytes << "\t" << s.num_treeblocks << "\t" << padding_bytes << "\t"
                      << s.metadata_bytes << std::endl;
        }
    }

    // Aggregate summary
    uint64_t sum_entry_bytes = 0;
    uint64_t sum_sa_metadata = 0;
    for (trie_level_t l = 0; l < MAX_TRIE_HASHMAP_DEPTH; l++) {
        sum_entry_bytes += layer_stats[l].entry_bytes;
        sum_sa_metadata += layer_stats[l].metadata_bytes;
    }

    uint64_t sum_data_bits_c = 0, sum_data_bits_u = 0, sum_flag_bits = 0;
    uint64_t sum_padding_bits = 0, sum_tb_metadata = 0;
    for (trie_level_t l = MAX_TRIE_HASHMAP_DEPTH; l < MAX_TRIE_DEPTH; l++) {
        const auto &s = layer_stats[l];
        sum_data_bits_c += s.data_bits_collapsed;
        sum_data_bits_u += s.data_bits_uncollapsed;
        sum_flag_bits += s.flag_bits;
        sum_padding_bits += s.data_padding_bits + s.flag_padding_bits;
        sum_tb_metadata += s.metadata_bytes;
    }

    uint64_t sum_data_bytes_c = sum_data_bits_c / 8;
    uint64_t sum_data_bytes_u = sum_data_bits_u / 8;
    uint64_t sum_flag_bytes = sum_flag_bits / 8;
    uint64_t sum_padding_bytes = sum_padding_bits / 8;

    uint64_t grand_total = sum_entry_bytes + sum_sa_metadata + sum_data_bytes_c + sum_data_bytes_u +
                           sum_flag_bytes + sum_padding_bytes + sum_tb_metadata;

    auto pct_of = [&](uint64_t val) { return grand_total > 0 ? (100.0 * val / grand_total) : 0.0; };

    std::cout << "\n=== Storage Summary ===" << std::endl;
    std::cout << "Category\tEntry_B\tSA_Meta\tDataB_C\tDataB_U\tFlagB\tPadB\tTB_Meta\tTotal"
              << std::endl;
    std::cout << "%\t" << std::fixed << std::setprecision(2) << pct_of(sum_entry_bytes) << "%\t"
              << pct_of(sum_sa_metadata) << "%\t" << pct_of(sum_data_bytes_c) << "%\t"
              << pct_of(sum_data_bytes_u) << "%\t" << pct_of(sum_flag_bytes) << "%\t"
              << pct_of(sum_padding_bytes) << "%\t" << pct_of(sum_tb_metadata) << "%\t100%"
              << std::endl;
    std::cout << "Bytes\t" << sum_entry_bytes << "\t" << sum_sa_metadata << "\t" << sum_data_bytes_c
              << "\t" << sum_data_bytes_u << "\t" << sum_flag_bytes << "\t" << sum_padding_bytes
              << "\t" << sum_tb_metadata << "\t" << grand_total << std::endl;
}

inline uint64_t verify_layer_stats(uint64_t expected_total)
{
    uint64_t total = 0;

    // Sorted array levels: entry_bytes + metadata_bytes
    for (trie_level_t l = 0; l < MAX_TRIE_HASHMAP_DEPTH; l++) {
        total += layer_stats[l].entry_bytes;
        total += layer_stats[l].metadata_bytes;
    }

    // Treeblock levels - track all bits
    uint64_t total_data_bits_collapsed = 0;
    uint64_t total_data_bits_uncollapsed = 0;
    uint64_t total_data_padding_bits = 0;
    uint64_t total_flag_bits = 0;
    uint64_t total_flag_padding_bits = 0;
    uint64_t total_tb_metadata = 0;
    uint64_t total_collapsed = 0;
    uint64_t total_uncollapsed = 0;
    uint64_t total_frontier = 0;

    for (trie_level_t l = MAX_TRIE_HASHMAP_DEPTH; l < MAX_TRIE_DEPTH; l++) {
        const auto &s = layer_stats[l];
        total_data_bits_collapsed += s.data_bits_collapsed;
        total_data_bits_uncollapsed += s.data_bits_uncollapsed;
        total_data_padding_bits += s.data_padding_bits;
        total_flag_bits += s.flag_bits;
        total_flag_padding_bits += s.flag_padding_bits;
        total_tb_metadata += s.metadata_bytes;
        total_collapsed += s.collapsed_nodes;
        total_uncollapsed += s.uncollapsed_nodes;
        total_frontier += s.frontier_nodes;
    }

    uint64_t total_data_bits = total_data_bits_collapsed + total_data_bits_uncollapsed;
    uint64_t all_bits =
        total_data_bits + total_flag_bits + total_data_padding_bits + total_flag_padding_bits;
    uint64_t treeblock_bytes = all_bits / 8 + total_tb_metadata;
    total += treeblock_bytes;

    // Verify: flag_bits == collapsed + uncollapsed + frontier
    bool node_count_match =
        (total_flag_bits == total_collapsed + total_uncollapsed + total_frontier);

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Sorted array bytes: "
              << (layer_stats[0].entry_bytes + layer_stats[0].metadata_bytes) << std::endl;
    std::cout << "Treeblock bits: data=" << total_data_bits
              << " (collapsed=" << total_data_bits_collapsed
              << " uncollapsed=" << total_data_bits_uncollapsed << ")"
              << " flag=" << total_flag_bits
              << " padding=" << (total_data_padding_bits + total_flag_padding_bits) << std::endl;
    std::cout << "Treeblock bytes: (" << all_bits << " bits) / 8 + " << total_tb_metadata
              << " metadata = " << treeblock_bytes << std::endl;
    std::cout << "Node counts: collapsed=" << total_collapsed
              << " uncollapsed=" << total_uncollapsed << " frontier=" << total_frontier
              << " total=" << (total_collapsed + total_uncollapsed + total_frontier)
              << " vs flag_bits=" << total_flag_bits << " -> "
              << (node_count_match ? "MATCH" : "MISMATCH") << std::endl;
    std::cout << "Computed total: " << total << std::endl;
    std::cout << "Expected total: " << expected_total << std::endl;
    std::cout << "Match: " << (total == expected_total ? "YES" : "NO") << std::endl;

    return total;
}

#endif // MD_TRIE_DEFS_H
