#ifndef MD_TRIE_MD_TRIE_H
#define MD_TRIE_MD_TRIE_H

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <queue>
#include <sstream>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include "data_point.h"
#include "debug.h"
#include "defs.h"
#include "ordered_types.h"
#include "profiling_points.h"
#include "tree_block.h"
#include "trie_node.h"

// ---------------------------------------
// DEBUG VARIABLES, USED FOR ANALYZING TREE PERFORMANCE.
// ---------------------------------------

// HELPER FUNCTIONS
// These define profile points for every tree _level_.

static std::vector<std::pair<std::vector<profile_point> *, std::string>> profile_points;

[[maybe_unused]]
static inline bool register_pp_vec(std::vector<profile_point> *ppv, std::string name)
{
    profile_points.push_back(std::make_pair(ppv, name));
    return true;
}
#define generate_pp_vec(name)                                                                      \
    std::vector<profile_point> name(MAX_TRIE_DEPTH + 1);                                           \
    static bool name##_registered = register_pp_vec(&name, std::string(#name))
#define define_pp_vec(name) extern std::vector<profile_point> name

#ifdef CONFIG_PROFILING_POINTS
void clear_and_activate_all_pps()
{
    for (auto [ppv, name] : profile_points) {
        size_t i = 0;
        for (auto &pp : *ppv) {
            pp.clear();
            pp.enable(true);
            std::ostringstream pp_name;
            pp_name << "level_" << i << "_" << name;
            pp.pp_name = pp_name.str();
            i++;
        }
    }
    clear_pps(); // global PPs.
    enable_pps(true);
}

// Returns a JSON array.
nlohmann::json jsonize_all_pps()
{
    // Collect all per-level PPs
    nlohmann::json all_pps = nlohmann::json::array();
    for (auto [ppv, _] : profile_points) {
        for (auto &pp : *ppv) {
            all_pps.push_back(pp.toJson());
        }
    }

    // Append global PPs
    nlohmann::json global_pps = dump_pp_json();
    for (auto &pp : global_pps) {
        all_pps.push_back(pp);
    }

    return all_pps;
}

void print_all_pps()
{
    // Output as JSON object
    std::cout << jsonize_all_pps().dump() << std::endl;
}
#else // !CONFIG_PROFILING_POINTS
void clear_and_activate_all_pps() {}
nlohmann::json jsonize_all_pps() { return nlohmann::json::array(); }
void print_all_pps() {}
#endif

// DEBUG VARIABLES :: RANGE QUERY PROFILING POINTS

DEFINE_PP(range_query_total);

// total time in each pp level
generate_pp_vec(pp_rq_total);
generate_pp_vec(pp_rq_total_base);
generate_pp_vec(pp_rq_total_boundary);
generate_pp_vec(pp_rq_total_frontier);
generate_pp_vec(pp_rq_total_null_sym);
generate_pp_vec(pp_rq_total_large_sym);
generate_pp_vec(pp_rq_total_good);

// Breakdowns
generate_pp_vec(pp_rq_stack_alloc);
generate_pp_vec(pp_rq_loop);
// time taken to generate morton_t bounds
generate_pp_vec(pp_rq_define_bounds);
generate_pp_vec(pp_rq_adjust_bounds_a);
generate_pp_vec(pp_rq_adjust_bounds_b);
generate_pp_vec(pp_rq_get_next_symbol);
// time taken to check whether a subtree is within the datapoint range
generate_pp_vec(pp_rq_test_subtree_range);
// time taken to navigate to the child node via range_search_get_child_node() etc
generate_pp_vec(pp_rq_get_child_node);
// time taken to check whether a subtree is within the datapoint range
generate_pp_vec(pp_rq_recurse_in_range);
// true when the next_symbol isn't even in the query range.
generate_pp_vec(pp_rq_symbol_not_in_range_);
// The number of points returned by the avg query on this level
generate_pp_vec(pp_rq_points_returned_);
// The number of treeblock recursions required.
generate_pp_vec(pp_rq_treeblocks_recursed_);

// ---------------------------------------
// MAIN TRIE CODE
// ---------------------------------------

template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION> class md_trie
{
private:
    using node_t = trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>;
    using treeblock_t = tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>;
    using point_t = data_point<DIMENSION>;

    // member variables
    node_t *root_ = nullptr;
    static constexpr coordinate_t KNN_RADIUS_SCALING_FACTOR = 2;
    size_t num_points_in_trie_ = 0;

    // MDTrie is allowed to "train" on representative input data to optimize its own index
    // (as per the `faiss::Index` spec). These variables optimize KNN performance.
    bool is_trained_ = false;
    point_t min_training_point_;
    point_t max_training_point_;
    float dataset_radius_;

    // Debug vars
    std::vector<debug_knn_rq> debug_knn_log_;

public:
    // API functions
    explicit md_trie() { root_ = new node_t(MAX_TRIE_HASHMAP_DEPTH == 0); }

    ~md_trie()
    {
        if (DESERIALIZED_MDTRIE) {
            // should not delete anything if deserialized
            return;
        }

        int current_level = 0;
        // free all trie nodes, layers by layer
        std::queue<node_t *> trie_node_queue;
        trie_node_queue.push(root_);
        while (!trie_node_queue.empty()) {

            unsigned long long queue_size = trie_node_queue.size();

            for (unsigned long long s = 0; s < queue_size; s++) {

                node_t *current_node = trie_node_queue.front();
                trie_node_queue.pop();

                if (current_level != MAX_TRIE_HASHMAP_DEPTH) {
                    auto trie_ptr = (std::map<morton_t, node_t *> *)current_node->get_treeblock();
                    for (auto &kv : *trie_ptr) {
                        if (kv.second) {
                            trie_node_queue.push(kv.second);
                        }
                    }
                    current_node->delete_non_leaf_node();
                } else {
                    delete current_node->get_treeblock();
                }
                delete current_node;
            }
            current_level++;
        }
    }

    inline trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *root() { return root_; }

    // Insert a datapoint into the MDTrie.
    void insert_trie(data_point<DIMENSION> *leaf_point)
    {
        assert(!DESERIALIZED_MDTRIE && "Error: cannot insert into a deserialized MDTrie!");

        debugf("[INSERT] Inserting point (first 2 dims as float): [%f, %f]\n",
               leaf_point->get_float_coordinate(0), leaf_point->get_float_coordinate(1));

        // Traverse the the sparse trie structure (implemented w/ hashmaps keyed by morton code
        // symbols). Traverse from depth 0 to MAX_TRIE_DEPTH until we get the treeblock we need!
        // (creating if necessary).
        trie_level_t level = 0, treeblock_level;
        node_t *current_trie_node = root_;
        tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *current_treeblock =
            walk_upper_trie(current_trie_node, leaf_point, level);

        assert(level == MAX_TRIE_HASHMAP_DEPTH);
        treeblock_level = level;

        // Insert into the "lower" trie (encoded in a treeblock).
        current_treeblock->insert_into_treeblock_layer(leaf_point, level, treeblock_level);

        // Note: this is actually incorrect when duplicates are inserted (don't care for now)
        num_points_in_trie_++;
    }

    /// @brief Retrieve all the points in [start_point, end_point] of the MDTrie.
    ///
    ///
    /// @param start_point Lower bound of query range. Coordinates should be in ordered format!
    /// @param end_point Upper bound of query range. Coordinates should be in ordered format!
    /// @param found_points Output vector of matching point coordinates (in ordered format)
    void range_search_trie(data_point<DIMENSION> *start_point, data_point<DIMENSION> *end_point,
                           std::vector<ordered_coordinate_t> &found_points)
    {
        PP_STATE(range_query_total);
        PP_ENTER(range_query_total);
        range_search(start_point, end_point, this->root_, 0, found_points);
        PP_EXIT(range_query_total);
    }

    /// @brief Optimize the MDTrie indices, given a representative dataset.
    ///
    /// MDTrie is allowed to "train" on representative input data to optimize its own index
    /// (as per the `faiss::Index` spec).
    void train_trie(std::vector<data_point<DIMENSION>> training_data)
    {
        using namespace ordered_types;

        is_trained_ = true;

        if (training_data.size() == 0)
            return;

        dataset_radius_ = 0;
        for (size_t d = 0; d < NUM_DIMENSIONS; d++) {
            auto cmp = [d](const auto &v1, const auto &v2) {
                return v1.get_ordered_coordinate(d) < v2.get_ordered_coordinate(d);
            };

            ordered_coordinate_t min_val =
                (*std::min_element(training_data.begin(), training_data.end(), cmp))
                    .get_ordered_coordinate(d);
            ordered_coordinate_t max_val =
                (*std::max_element(training_data.begin(), training_data.end(), cmp))
                    .get_ordered_coordinate(d);

            min_training_point_.set_ordered_coordinate(d, min_val);
            max_training_point_.set_ordered_coordinate(d, max_val);

            float dim_radius =
                (ordered_u32_to_float(max_val) - ordered_u32_to_float(min_val)) / 2.0f;
            dataset_radius_ = std::max(dataset_radius_, dim_radius);
        }
    }

    /// @brief Return the `k` nearest neighbor points of the specified input point.
    ///
    /// Uses Euclidean distance.
    ///
    /// @param search_point This function returns `n` points nearest to here! Underlying
    ///                     data representation should be in ordered format.
    /// @param k The number of neighboring points to return.
    /// @param radius The search radius around the query point.
    /// @param distance_fn Comparison function for sorting by distance (defaults to Euclidean).
    /// @param found_points The output vector. Will be populated with the points found.
    /// @return List of K nearest data points (already encoded in ordered
    ///         format, just retrieve them as floats when you need).
    void knn_search_trie(data_point<DIMENSION> &search_point, node_pos_t k,
                         std::vector<data_point<DIMENSION>> &out)
    {
        debugf("[KNN] Starting KNN search for k=%lu\n", (unsigned long)k);
        debugf("[KNN] Total points in trie: %lu\n", (unsigned long)this->num_points_in_trie_);

        float query_radius = 0;
        this->debug_knn_log_.clear();

        if (k > this->num_points_in_trie_) {
            debugf("[KNN] k > num_points_in_trie, returning empty\n");
            // TODO(yash): should we return _all_ points here instead?
            return;
        }

        while (true) { // retry queries with larger radii until we get enough points
            // PHASE 1: Perform a range-query on the data.

            // Retry range query until we've gathered enough candidate points.
            std::vector<ordered_coordinate_t> candidate_data;
            candidate_data.reserve(k);
            size_t num_candidates = 0;
            while (true) {
                query_radius =
                    expand_knn_query_radius(search_point, k, query_radius, num_candidates);
                auto [start_point, end_point] =
                    knn_radius_to_query_bounds(search_point, query_radius);

                debugf("[KNN] Query radius: %f\n", query_radius);
                debugf("[KNN] Query bounds (first 2 dims): [%f, %f] to [%f, %f]\n",
                       start_point.get_float_coordinate(0), start_point.get_float_coordinate(1),
                       end_point.get_float_coordinate(0), end_point.get_float_coordinate(1));

                // Log information about the component range queries.
                debug_knn_rq rq_info;
                rq_info.radius = query_radius;
                PP_SAVE_NEW_TIMESTAMP(rq_latency_start);
                rq_info.latency = rq_latency_start;

                range_search(&start_point, &end_point, this->root_, 0, candidate_data);
                num_candidates = candidate_data.size() / DIMENSION;

                PP_SAVE_NEW_TIMESTAMP(rq_latency_end);
                rq_info.latency = rq_latency_end - rq_info.latency;
                rq_info.points_returned = num_candidates;
                this->debug_knn_log_.push_back(rq_info);

                assert(candidate_data.size() % DIMENSION == 0);
                debugf("[KNN] Found %zu candidates\n", num_candidates);

                if (num_candidates >= k)
                    break;
                candidate_data.clear();
            }

            // PHASE 2: Sort the retrieved points by distance.

            std::vector<point_t> candidates;
            std::vector<std::pair<point_t *, float>> points_with_distance_sq;
            candidates.reserve(num_candidates);
            points_with_distance_sq.reserve(num_candidates);

            for (size_t i = 0; i < candidate_data.size(); i += DIMENSION) {
                point_t candidate;
                for (n_dimensions_t j = 0; j < DIMENSION; j++)
                    candidate.set_ordered_coordinate(j, candidate_data[i + j]);
                candidates.push_back(candidate);
                debugf("[KNN] Candidate %zu (first 2 dims as float): [%f, %f]\n", i / DIMENSION,
                       candidate.get_float_coordinate(0), candidate.get_float_coordinate(1));

                double distance_sq =
                    md_trie::euclidean_distance_sq(candidates.back(), search_point);
                debugf("[KNN] Candidate %zu (distance from center): %f\n", i, distance_sq);
                points_with_distance_sq.emplace_back(&candidates.back(), distance_sq);
            }

            // Sort points by distance.
            // TODO just filter through the points in one pass instead.
            std::sort(points_with_distance_sq.begin(), points_with_distance_sq.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });

            // PHASE 3: Perform conventional KNN on the retrieved points

            // Check whether we have enough valid points to return.
            // Our query may return points outside of our radius too (since it searches in an
            // N-cube, not an N-sphere). We can't use these points, since we can't guarantee there
            // isn't a _closer_ point just outside our query interval.
            const float query_radius_sq = query_radius * query_radius;
            const bool found_k_valid_points =
                points_with_distance_sq[k - 1].second <= query_radius_sq;
            debugf("Checking whether K valid points are in radius^2 %f: %d\n", query_radius_sq,
                   found_k_valid_points);

            if (!found_k_valid_points) { // Retry the query
                // Our query has returned the full dataset. We know no more points are coming,
                // so we can return everything. No need to retry w/ larger query radius.
                if (candidates.size() == this->num_points_in_trie_) {
                    debugf("[KNN] Retrieved all points in trie. Not enough points. Giving up.\n");
                    break;
                }

                debugf("[KNN] k-th point outside radius, retrying with larger radius\n");
                // Retry w/ larger query radius.
                continue;
            }

            debugf("[KNN] Returning %lu valid points\n", (unsigned long)k);

            // Return the valid points
            out.clear();
            out.reserve(k);
            for (size_t i = 0; i < k; i++) {
                point_t *point = points_with_distance_sq[i].first;
                out.push_back(*point);
            }
            return;
        }
    }

    void debug_print_knn_log()
    {
        std::cout << "KNN RQ BEGIN: " << this->debug_knn_log_.size() << std::endl;
        for (auto &e : this->debug_knn_log_)
            std::cout << e.toString() << std::endl;
        std::cout << "KNN RQ END" << std::endl;
    }

    /// @brief starts to serialize the trie at the `current_offset`,
    ///        it is caller's responsibility to align the file cursor
    ///
    /// Uses the `current_offset` global variable to determine...something. TODO(yash):
    ///
    /// @param file the file to write to.
    void serialize(FILE *file)
    {
        assert(!DESERIALIZED_MDTRIE && "Error: should not serialize a deserialized MDTrie!");

        // asserting that the trie is not already inserted, each data structure should only be
        // inserted once
        assert(pointers_to_offsets_map.find((uint64_t)this) == pointers_to_offsets_map.end());

        // current_offset should be the location where the trie is written
        pointers_to_offsets_map.insert({(uint64_t)this, current_offset});

        // create a buffer for easy modification of the trie copy
        md_trie *temp_trie = (md_trie *)calloc(1, sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>));
        temp_trie->root_ = this->root_;

        // create root node offset, after where the trie would live
        uint64_t root_offset = current_offset + sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
        if (this->root_)
            temp_trie->root_ = (node_t *)(root_offset);
        else {
            fwrite(temp_trie, sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);
            free(temp_trie);
            return;
        }

        // perform write for normal case
        // current_offset should always be the next write offset
        fwrite(temp_trie, sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);
        free(temp_trie);

        current_offset += sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>);

        // create a temp buffer and write empty bytes in place of the root node to be written later
        node_t *temp_root = (node_t *)calloc(1, sizeof(node_t));
        fwrite(temp_root, sizeof(node_t), 1, file);

        // now match `current_offset` with FILE CURSOR
        current_offset += sizeof(node_t);

        // this is fine, the next node is guaranteed to not have been created
        root_->serialize(file, 0, root_offset, temp_root);
    }

    void deserialize(uint64_t base_addr)
    {
        assert(!DESERIALIZED_MDTRIE && "Error: trie already deserialized!");
        DESERIALIZED_MDTRIE = true;
        if (root_) {
            root_ = (node_t *)(base_addr + (uint64_t)root_);
            root_->deserialize(0, base_addr);
            // Count points after deserialization
            num_points_in_trie_ = root_->count_points_recur(0);
        }
    }

    // compute total storage used by the md_trie in bytes
    uint64_t get_total_storage()
    {
        uint64_t size = 0;

        size += sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
        size += sizeof(node_t);

        root_->update_size_recur(0, size);

        return size;
    }

    /// @brief Get storage breakdown per trie level.
    /// Populates layer_stats vector with per-level stats.
    void get_storage_per_layer()
    {
        reset_layer_stats();

        // Account for md_trie struct and root node as metadata at level 0
        layer_stats[0].metadata_bytes += sizeof(md_trie<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
        layer_stats[0].metadata_bytes += sizeof(node_t);

        root_->get_size_per_level_recur(0);
    }

    size_t get_num_points_in_trie() { return num_points_in_trie_; }

private:
    /// @brief Helper function; returns the square of the euclidean distance between 2 points.
    ///
    /// Assumes the input data's original format was a floating point number
    /// (ie: the data point contains an ordered representation). This should always be true in
    /// the current codebase.
    static double euclidean_distance_sq(const data_point<DIMENSION> &p1,
                                        const data_point<DIMENSION> &p2)
    {
        debugf("[EUCLIDEAN_DISTANCE_SQ] Computing distance\n");
        double sum_sq = 0;
        for (n_dimensions_t i = 0; i < DIMENSION; i++) {
            double diff = p1.get_float_coordinate(i) - p2.get_float_coordinate(i);
            sum_sq += diff * diff;
            if (i < 2) {
                debugf("[EUCLIDEAN_DISTANCE_SQ] diff[%ld]=%lf\n", i, diff);
            }
        }
        return static_cast<float>(sum_sq);
    }

    /// @brief Helper function. Retrieves all the points in [start_point, end_point] of the MDTrie.
    ///
    /// Helper function for `range_search_trie()`. Searches the upper (`std::map` based) levels
    /// of the mdtrie. `range_search_treeblock()` handles the search inside of each treeblock.
    ///
    /// @param start_point Lower bound of query range (modified during recursion)
    /// @param end_point Upper bound of query range (modified during recursion)
    /// @param current_trie_node Current node in sparse trie
    /// @param level Current trie level (0 to MAX_TRIE_DEPTH)
    /// @param found_points Output vector of matching point coordinates
    ///
    void range_search(data_point<DIMENSION> *start_point, data_point<DIMENSION> *end_point,
                      trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *current_trie_node,
                      trie_level_t cur_trie_level, std::vector<ordered_coordinate_t> &found_points)
    {
        PP_SAVE_NEW_TIMESTAMP(ts_enter);
        PP_NEW_TIMESTAMP(ts_a);
        PP_NEW_TIMESTAMP(ts_b);

        // Reached MAX_TRIE_DEPTH: switch to tree_block layer
        if (cur_trie_level == MAX_TRIE_HASHMAP_DEPTH) {
            assert(current_trie_node);
            treeblock_t *current_treeblock = current_trie_node->get_treeblock();
            current_treeblock->range_search_treeblock(start_point, end_point, current_treeblock,
                                                      cur_trie_level, 0, 0, 0, found_points);
            // Any PP time in this layer will already be counted by the treeblock layer.
            return;
        }

        // Convert our bounds in "Input Space" into equivalent bounds in the "Morton Space".
        PP_SAVE_TIMESTAMP(ts_b);

        morton_t start_symbol = start_point->leaf_to_symbol(cur_trie_level);
        morton_t end_symbol = end_point->leaf_to_symbol(cur_trie_level);
        // bound_magic masks bits that must match for a point to be in range.
        morton_t bound_magic = ~(start_symbol ^ end_symbol);

        // Store original ranges for restoration after each child recursion
        data_point<DIMENSION> original_start_point = (*start_point);
        data_point<DIMENSION> original_end_point = (*end_point);

        PP_SAVE_TIMESTAMP(ts_a);
        pp_rq_define_bounds[cur_trie_level].record(ts_a - ts_b);

        // Iterate over all children of the current node that lie within our interval.
        auto trie_ptr = current_trie_node->get_children();
        auto start_it = trie_ptr.lower_bound(start_symbol);
        auto end_it = trie_ptr.upper_bound(end_symbol);
        PP_SAVE_TIMESTAMP(ts_b);
        pp_rq_get_next_symbol[cur_trie_level].record(ts_b - ts_a);

        TimeStamp ts_loop_exit;
        PP_SAVE_TIMESTAMP(ts_loop_exit); // set before entering!
        for (auto it = start_it; it != end_it; ++it) {
            const morton_t current_symbol = it.key();
            node_t *child_node = it.value();

            // requires ts_loop_exit to be set!
            PP_SAVE_NEW_TIMESTAMP(ts_loop_enter);
            pp_rq_get_next_symbol[cur_trie_level].record(ts_loop_enter - ts_loop_exit);

            if (!child_node) {
                pp_rq_loop[cur_trie_level].exit(ts_loop_enter);
                ts_loop_exit = ts_loop_enter; // set before continue
                continue;
            }

            // PHASE: Test if symbol is in range
            PP_SAVE_TIMESTAMP(ts_a);
            if (morton_t::masked_not_equal(start_symbol, current_symbol, bound_magic)) {
                PP_SAVE_TIMESTAMP(ts_loop_exit);
                pp_rq_test_subtree_range[cur_trie_level].record(ts_loop_exit - ts_a);
                pp_rq_loop[cur_trie_level].record(ts_loop_exit - ts_loop_enter);
                pp_rq_symbol_not_in_range_[cur_trie_level].increment();
                continue;
            }
            PP_SAVE_TIMESTAMP(ts_b);
            pp_rq_test_subtree_range[cur_trie_level].record(ts_b - ts_a);

            // PHASE: Adjust bounds before recursion
            // Each subtree maps to a small N-cube in our input space.
            // If we're going to recurse into a subtree, we can narrow down the bounds of our query
            // to the part that intersects with this N-cube. This is just so the above
            // bitmap logic continues to work.
            data_point<DIMENSION>::shrink_query_bounds(start_point, end_point, current_symbol,
                                                       cur_trie_level);
            PP_SAVE_TIMESTAMP(ts_a);
            pp_rq_adjust_bounds_a[cur_trie_level].record(ts_a - ts_b);

            // PHASE: Recurse into child
            size_t points_returned = found_points.size();
            range_search(start_point, end_point, child_node, cur_trie_level + 1, found_points);
            PP_SAVE_TIMESTAMP(ts_b);
            pp_rq_recurse_in_range[cur_trie_level].record(ts_b - ts_a);

            // PHASE: Restore bounds after recursion
            // Restore original query range boundaries
            (*start_point) = original_start_point;
            (*end_point) = original_end_point;

            PP_SAVE_TIMESTAMP(ts_loop_exit);
            pp_rq_adjust_bounds_b[cur_trie_level].record(ts_loop_exit - ts_b);
            points_returned = found_points.size() - points_returned;
            pp_rq_points_returned_[cur_trie_level].record(points_returned);
            pp_rq_loop[cur_trie_level].record(ts_loop_exit - ts_loop_enter);
        }

        PP_SAVE_NEW_TIMESTAMP(ts_exit);
        pp_rq_total_good[cur_trie_level].record(ts_exit - ts_enter);
        pp_rq_total[cur_trie_level].record(ts_exit - ts_enter);
    }

    /// @brief Navigate through the sparse upper trie to find/create the tree_block
    ///
    /// This function implements the first part of the insertion process:
    /// it walks down the trie_node hierarchy using Morton code symbols,
    ///    - creating nodes as needed
    ///    - until it reaches MAX_TRIE_DEPTH where tree_blocks live.
    ///
    /// @param current_trie_node Starting trie node (typically root)
    /// @param leaf_point Data point being inserted
    /// @param cur_trie_level Current level in trie (modified to MAX_TRIE_DEPTH on return)
    /// @return Pointer to the tree_block where insertion should continue
    tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *
    walk_upper_trie(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *current_trie_node,
                    data_point<DIMENSION> *leaf_point, trie_level_t &level) const
    {
        morton_t current_symbol;

        // Follow the trie as far as we can (or until we encounter a missing child).
        //
        // TODO(yash). Hey we run through the same method twice here for no reason.
        // Note(Leo): apply commit 46a3003a8d3acf6dbee2a875edd869e5e28fb891 later
        while (level < MAX_TRIE_HASHMAP_DEPTH &&
               current_trie_node->get_child(leaf_point->leaf_to_symbol(level))) {
            current_trie_node = current_trie_node->get_child(leaf_point->leaf_to_symbol(level));
            level++;
        }

        // Create missing trie_nodes if needed. Mark the last node as a leaf
        // (will hold a tree_block_pointer).
        while (level < MAX_TRIE_HASHMAP_DEPTH) {
            current_symbol = leaf_point->leaf_to_symbol(level);
            bool is_leaf_node = (level == MAX_TRIE_HASHMAP_DEPTH - 1);
            auto new_child = new node_t(is_leaf_node);
            current_trie_node->set_child(current_symbol, new_child);

            // TODO(yash): can I just reuse new_child var?
            // Note(Leo): apply commit 46a3003a8d3acf6dbee2a875edd869e5e28fb891 later
            current_trie_node = current_trie_node->get_child(current_symbol);
            level++;
        }

        // Get (or create!) a tree_block for our leaf node.
        tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *current_treeblock = nullptr;
        if (current_trie_node->get_treeblock() == nullptr) {
            // create a new empty treeblock, no preallocation
            current_treeblock = new tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>();
            current_trie_node->set_treeblock(current_treeblock);
        } else {
            // tree_block already exists, retrieve it
            current_treeblock =
                (tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)current_trie_node->get_treeblock();
        }
        return current_treeblock;
    }

private:
    /// @brief Guess an initial range-query radius for a KNN query.
    ///
    /// Too large, and our performance will be low because we return too much.
    /// Too small, and we'll have to retry with a larger radius.
    ///
    /// TODO(yash): implement something smarter here!
    float estimate_knn_query_radius([[maybe_unused]] point_t search_point,
                                    [[maybe_unused]] size_t k)
    {
        if (!is_trained_) {
            return 1.0; // initial guess
        }
        // Assume points are uniformly distributed in the radius. Also, assume pi is three.
        // TODO(yash): this code is nonsense. Fix it.
        // 1. In high dimensions, the volume of a ball isn't Pi * Radius^2.
        // 2. Shouldn't I be including the whole dataset's radius here, for the
        //    initial radius of the dataset-ball?
        return std::sqrt(static_cast<float>(num_points_in_trie_) / (3 * k));
    }

    /// @brief Update our guess for a range-query radius for a KNN query.
    ///
    /// Eventually, we should implement something smarter here.
    ///
    /// @param prior_guess Our prior guess for KNN query radius
    /// @param prior_num_results The number of points the prior KNN query request found
    static float expand_knn_query_radius([[maybe_unused]] point_t search_point,
                                         [[maybe_unused]] size_t k,
                                         [[maybe_unused]] float prior_guess,
                                         [[maybe_unused]] size_t prior_num_results)
    {
        if (prior_guess == 0.0f)
            return 1;
        return prior_guess * 2.0f;
    }

    /// @brief Helper function for knn_search_trie.
    ///
    /// @param search_point the starting point of our KNN search
    /// @param query_radius the  query radius (in euclidean distance)
    static std::pair<point_t, point_t> knn_radius_to_query_bounds(point_t search_point,
                                                                  float radius)
    {
        point_t start_point, end_point;
        start_point.clear();
        end_point.clear();

        // TODO(yash): check for overflow!!! Or INT_MAX or FLOAT_MAX or whatever.
        for (n_dimensions_t i = 0; i < DIMENSION; ++i) {
            float middle = search_point.get_float_coordinate(i);
            if (i < 2) {
                debugf("[KNN_BOUNDS] dim %lu: middle=%f, radius=%f, start=%f, end=%f\n", i, middle,
                       radius, middle - radius, middle + radius);
            }
            start_point.set_float_coordinate(i, middle - radius);
            end_point.set_float_coordinate(i, middle + radius);
        }
        debugf("[KNN_BOUNDS] start point (first 2 dims as uint32): [%u, %u]\n",
               start_point.get_ordered_coordinate(0), start_point.get_ordered_coordinate(1));
        debugf("[KNN_BOUNDS] end point (first 2 dims as uint32): [%u, %u]\n",
               end_point.get_ordered_coordinate(0), end_point.get_ordered_coordinate(1));
        return std::pair(start_point, end_point);
    }
};

#endif // MD_TRIE_MD_TRIE_H
