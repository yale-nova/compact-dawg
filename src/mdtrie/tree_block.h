#ifndef MD_TRIE_TREE_BLOCK_H
#define MD_TRIE_TREE_BLOCK_H

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/time.h>

#include "compressed_bitmap.h"
#include "defs.h"
#include "profiling_points.h"
#include "trie_node.h"

#define declare_pp_vec(name) extern std::vector<profile_point> name

DECLARE_PP(range_query_total);

// total time in each pp level
declare_pp_vec(pp_rq_total);
declare_pp_vec(pp_rq_total_base);
declare_pp_vec(pp_rq_total_boundary);
declare_pp_vec(pp_rq_total_frontier);
declare_pp_vec(pp_rq_total_null_sym);
declare_pp_vec(pp_rq_total_large_sym);
declare_pp_vec(pp_rq_total_good);

// Breakdowns
declare_pp_vec(pp_rq_stack_alloc);
declare_pp_vec(pp_rq_loop);
// time taken to declare morton_t bounds
declare_pp_vec(pp_rq_define_bounds);
declare_pp_vec(pp_rq_adjust_bounds_a);
declare_pp_vec(pp_rq_adjust_bounds_b);
declare_pp_vec(pp_rq_get_next_symbol);
// time taken to check whether a subtree is within the datapoint range
declare_pp_vec(pp_rq_test_subtree_range);
// time taken to navigate to the child node via range_search_get_child_node() etc
declare_pp_vec(pp_rq_get_child_node);
// time taken to check whether a subtree is within the datapoint range
declare_pp_vec(pp_rq_recurse_in_range);
// true when the next_symbol isn't even in the query range.
declare_pp_vec(pp_rq_symbol_not_in_range_);
// The number of points returned by the avg query on this level
declare_pp_vec(pp_rq_points_returned_);
declare_pp_vec(pp_rq_treeblocks_recursed_);

template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION> class tree_block
{
    using point_t = data_point<DIMENSION>;

public:
    explicit tree_block(node_pos_t num_nodes, node_bitmap_pos_t total_nodes_bits)
    {
        bitmap_ = compressed_bitmap::compressed_bitmap<H_LEVEL, CHUNK_WIDTH, DIMENSION>(
            num_nodes, total_nodes_bits);
    }

    explicit tree_block(compressed_bitmap::compressed_bitmap<H_LEVEL, CHUNK_WIDTH, DIMENSION> dfuds)
    {
        bitmap_ = std::move(dfuds);
    }

    explicit tree_block()
    {
        bitmap_ =
            std::move(compressed_bitmap::compressed_bitmap<H_LEVEL, CHUNK_WIDTH, DIMENSION>());
    }

    ~tree_block()
    {
        for (node_pos_t i = 0; i < num_frontiers_; i++) {
            auto &frontier = frontiers_[i];
            if (frontier.child_treeblock)
                delete frontier.child_treeblock;
        }
        if (num_frontiers_ > 0)
            free(frontiers_);
        bitmap_.destroy();
    }

    inline node_pos_t get_num_frontiers() const { return num_frontiers_; }

    inline tree_block *get_child_treeblock(node_pos_t current_frontier) const
    {

        return frontiers_[current_frontier].child_treeblock;
    }

    inline void set_child_treeblock(node_pos_t current_frontier, tree_block *child)
    {
        frontiers_[current_frontier].child_treeblock = child;
    }

    inline node_pos_t get_frontier_node_pos(node_pos_t current_frontier) const
    {
        return frontiers_[current_frontier].preorder_;
    }

    inline void set_frontier_node_pos(node_pos_t current_frontier, node_pos_t preorder)
    {
        frontiers_[current_frontier].preorder_ = preorder;
    }

    // Recursive helper function for `select_node_to_split_`.
    //
    // node_level is the level of this parent_node.
    // this is first invoked in on the root parent_node (0, 0)
    subtree_info select_node_to_split_(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                       node_pos_t frontier, trie_level_t node_level,
                                       node_split_info &best) const
    {
        // Frontier parent_node: return immediately
        if (frontier < num_frontiers_ && get_frontier_node_pos(frontier) == node_pos) {
            return {1, 0, 1};
        }

        // Leaf parent_node: return its info
        if (node_level == MAX_TRIE_DEPTH - 1) {
            return {1, bitmap_.get_symbol_width(node_pos, node_bitmap_pos), 0};
        }

        // Use the optimized pair-returning function

        auto [node_bits, num_children] =
            bitmap_.get_symbol_width_and_num_children(node_pos, node_bitmap_pos);
        assert(node_bits == bitmap_.get_symbol_width(node_pos, node_bitmap_pos));
        assert(num_children == bitmap_.get_num_children(node_pos, node_bitmap_pos));

        subtree_info current_subtree = {1, node_bits, 0};

        node_pos_t child_node = node_pos + 1;
        node_pos_t child_pos = node_bitmap_pos + node_bits;

        // Minimize recursion depth by using a local best
        node_split_info local_best = best;

        for (node_pos_t i = 0; i < num_children; i++) {
            subtree_info child_info =
                select_node_to_split_(child_node, child_pos, frontier, node_level + 1, local_best);

            // Only update best if child is better
            if (is_better(local_best, child_info, node_level + 1)) {
                local_best = {.node = child_node,
                              .node_count = child_info.node_count,
                              .node_pos = child_pos,
                              .node_bits = child_info.node_bits,
                              .frontier_node_pos = frontier,
                              .frontier_count = child_info.frontier_count,
                              .node_depth = static_cast<trie_level_t>(node_level + 1)};
            }

            // Advance to next child
            child_node += child_info.node_count;
            child_pos += child_info.node_bits;
            frontier += child_info.frontier_count;

            current_subtree.node_count += child_info.node_count;
            current_subtree.node_bits += child_info.node_bits;
            current_subtree.frontier_count += child_info.frontier_count;
        }

        // Write back the best found in this subtree
        best = local_best;
        return current_subtree;
    }

    // return if the queried "now" subtree is the better than the "best"
    bool is_better(const node_split_info &best, const subtree_info &now,
                   const trie_level_t now_depth) const
    {
        if (now.node_bits == 0) {
            return false;
        }

        auto best_nc_diff = std::llabs(static_cast<long long>(best.node_count) -
                                       static_cast<long long>(size_in_nodes() / 2));
        auto now_nc_diff = std::llabs(static_cast<long long>(now.node_count) -
                                      static_cast<long long>(size_in_nodes() / 2));
        if (best_nc_diff > now_nc_diff) {
            return true;
        } else if (best_nc_diff < now_nc_diff) {
            return false;
        }

        auto best_bits_diff = std::llabs(static_cast<long long>(best.node_bits) -
                                         static_cast<long long>(this->size_in_bits() / 2));
        auto now_bits_diff = std::llabs(static_cast<long long>(now.node_bits) -
                                        static_cast<long long>(this->size_in_bits() / 2));
        if (best_bits_diff > now_bits_diff) {
            return true;
        } else if (best_bits_diff < now_bits_diff) {
            return false;
        }

        auto best_frontier_diff = std::llabs(static_cast<long long>(best.frontier_count) -
                                             static_cast<long long>(get_num_frontiers() / 2));
        auto now_frontier_diff = std::llabs(static_cast<long long>(now.frontier_count) -
                                            static_cast<long long>(get_num_frontiers() / 2));
        if (best_frontier_diff > now_frontier_diff) {
            return true;
        } else if (best_frontier_diff < now_frontier_diff) {
            return false;
        }

        // else the deeper the better
        return now_depth > best.node_depth;
    }

    // returns the best node to be selected for splitting,
    // should only split when there is at least one non-frontier node
    inline node_split_info select_node_to_split(trie_level_t root_depth) const
    {
        assert(size_in_nodes() > num_frontiers_);

        node_split_info best = {};
        // Just call get_subtree_info at the root; it will update 'best'
        // recursively
        select_node_to_split_(0, 0, 0, root_depth, best);

        return best;
    }

    subtree_info print_subtree_info(node_pos_t node, node_pos_t node_pos, node_pos_t frontier,
                                    trie_level_t node_level) const
    {
        std::cout << "node: " << node << ", pos: " << node_pos << ", frontier index: " << frontier
                  << ", frontier_preorder: " << get_frontier_node_pos(frontier)
                  << ", trie_depth: " << node_level << std::endl;

        // this is the subtree of the frontier node
        if (frontier < num_frontiers_ && get_frontier_node_pos(frontier) == node) {
            // simply return this frontier node as the single subtree

            // note that a single frontier node is 0 bits
            return {1, 0, 1};
        }

        if (node_level == MAX_TRIE_DEPTH - 1) {
            return {1, bitmap_.get_symbol_width(node, node_pos), 0};
        }

        auto [node_bits, num_children] = bitmap_.get_symbol_width_and_num_children(node, node_pos);
        subtree_info current_subtree = {1, node_bits, 0};

        // Start with the first child position
        node_pos_t child_node = node + 1;
        node_pos_t child_pos = node_pos + node_bits;

        for (node_pos_t i = 0; i < num_children; i++) {
            subtree_info child_info =
                print_subtree_info(child_node, child_pos, frontier, node_level + 1);

            // Move to the next child by advancing through the current child's
            // subtree
            child_node += child_info.node_count;
            child_pos += child_info.node_bits;
            frontier += child_info.frontier_count;

            current_subtree.node_count += child_info.node_count;
            current_subtree.node_bits += child_info.node_bits;
            current_subtree.frontier_count += child_info.frontier_count;
        }

        return current_subtree;
    }

private:
    /// @brief Get a given child of a parent node.
    ///
    /// This function takes in a node (in preorder) and a symbol (branch index)
    /// It returns the child node (in preorder) designated by that symbol.
    /// Same as `get_child_node_unsafe()`, but it checks if the child exists
    /// first.
    ///
    /// Side effects! updates parent_node_bitmap_pos to the child's position
    /// too.
    ///
    /// @param p unknown
    /// @param parent_node_pos Parent node (preorder position)
    /// @param parent_node_bitmap_pos Bit position (this will be updated to
    /// child's position)
    /// @param child_symbol Morton code symbol of the child (branch to follow)
    /// @param current_trie_level Current trie level
    /// @param current_frontier Frontier index (in/out parameter)
    /// @return Child node preorder position
    node_pos_t get_child_node(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *&p,
                              node_pos_t parent_node_pos, node_pos_t &parent_node_bitmap_pos,
                              const morton_t &child_symbol, trie_level_t current_trie_level,
                              node_pos_t &current_frontier_pos)
    {
        if (parent_node_pos >= size_in_nodes())
            return null_node;

        // Check frontier condition first to decide which path to take
        bool is_frontier = frontiers_ != nullptr && current_frontier_pos < num_frontiers_ &&
                           parent_node_pos == get_frontier_node_pos(current_frontier_pos);

        if (is_frontier) {
            // Frontier case: use fast has_symbol (can short-circuit early)
            // TODO(yash): this short circuit is not as fast as we expected. I
            //             should instead create a proper function that
            //             retrieves the required info; aborting early if the
            //             child doesn't exist.
            bool has_child =
                bitmap_.has_child(parent_node_pos, parent_node_bitmap_pos, child_symbol);
            if (!has_child)
                return null_node;

            if (current_trie_level == MAX_TRIE_DEPTH - 1)
                return parent_node_pos;

            // Switch to child treeblock and compute fresh
            p = get_child_treeblock(current_frontier_pos);
            current_frontier_pos = 0;
            node_pos_t temp_node_pos = 0;
            node_pos_t temp_node_bitmap_pos = 0;
            node_pos_t current_node =
                p->get_child_node_unsafe(temp_node_pos, temp_node_bitmap_pos, child_symbol,
                                         current_trie_level, current_frontier_pos);
            parent_node_bitmap_pos = temp_node_bitmap_pos;
            return current_node;
        }

        // Non-frontier case: use has_symbol_with_info to get skip count and width
        auto [has_child, n_children_to_skip, parent_node_width] =
            bitmap_.get_child_info_lite(parent_node_pos, parent_node_bitmap_pos, child_symbol);
        if (!has_child)
            return null_node;

        if (current_trie_level == MAX_TRIE_DEPTH - 1)
            return parent_node_pos;

        // Verify pre-computed values match the original functions
        assert(parent_node_width ==
               bitmap_.get_symbol_width(parent_node_pos, parent_node_bitmap_pos));

        return get_child_node_unsafe_pre_calc(parent_node_pos, parent_node_bitmap_pos,
                                              current_trie_level, current_frontier_pos,
                                              n_children_to_skip, parent_node_width);
    }

    node_pos_t get_child_node_pre_calc(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *&p,
                                       node_pos_t parent_node_pos,
                                       node_pos_t &parent_node_bitmap_pos,
                                       const morton_t &child_symbol,
                                       trie_level_t current_trie_level,
                                       node_pos_t &current_frontier_pos,
                                       node_pos_t n_children_to_skip, node_pos_t parent_node_width)
    {
        if (parent_node_pos >= size_in_nodes())
            return null_node;

        // Check frontier condition first to decide which path to take
        bool is_frontier = frontiers_ != nullptr && current_frontier_pos < num_frontiers_ &&
                           parent_node_pos == get_frontier_node_pos(current_frontier_pos);

        if (is_frontier) {
            // Frontier case: use fast has_symbol (can short-circuit early)
            assert(bitmap_.has_child(parent_node_pos, parent_node_bitmap_pos, child_symbol));

            if (current_trie_level == MAX_TRIE_DEPTH - 1)
                return parent_node_pos;

            // Switch to child treeblock and compute fresh
            p = get_child_treeblock(current_frontier_pos);
            current_frontier_pos = 0;
            node_pos_t temp_node_pos = 0;
            node_pos_t temp_node_bitmap_pos = 0;
            node_pos_t current_node =
                p->get_child_node_unsafe(temp_node_pos, temp_node_bitmap_pos, child_symbol,
                                         current_trie_level, current_frontier_pos);
            parent_node_bitmap_pos = temp_node_bitmap_pos;
            return current_node;
        } else {
            // Non-frontier case: use has_symbol_with_info to get skip count and width
            if (current_trie_level == MAX_TRIE_DEPTH - 1)
                return parent_node_pos;

            // Verify pre-computed values match the original functions
            assert(bitmap_.has_child(parent_node_pos, parent_node_bitmap_pos, child_symbol));
            assert(parent_node_width ==
                   bitmap_.get_symbol_width(parent_node_pos, parent_node_bitmap_pos));

            return get_child_node_unsafe_pre_calc(parent_node_pos, parent_node_bitmap_pos,
                                                  current_trie_level, current_frontier_pos,
                                                  n_children_to_skip, parent_node_width);
        }
    }

    /// @brief Get a given child of a parent node. Doesn't check if child
    /// exists.
    ///
    /// This function takes in a node (in preorder) and a symbol (branch index)
    /// It returns the location of the child node designated by that symbol.
    ///
    /// Same as to get_child_node(), but doesn't check if the child exists.
    ///
    /// SIDE EFFECTS! Changes the parent_node_bitmap_pos, and frontier pos!
    ///
    /// @param parent_node_pos Parent node (preorder position)
    /// @param parent_node_bitmap_pos Bit position (SIDE EFFECT! Updated to child's position)
    /// @param child_symbol Morton code symbol of the child (branch to follow)
    /// @param current_trie_level Current trie level
    /// @param current_frontier_pos Frontier index (in/out parameter)
    /// @return Child node preorder position AND bitmap position (via parent_node_bitmap_pos).
    node_pos_t get_child_node_unsafe(node_pos_t parent_node_pos, node_pos_t &parent_node_bitmap_pos,
                                     const morton_t &child_symbol, trie_level_t current_trie_level,
                                     node_pos_t &current_frontier_pos)
    {
        // Fast-path checks
        assert(this->size_in_nodes() != 0);
        if (current_trie_level == MAX_TRIE_DEPTH)
            return parent_node_pos;

        // Get both skip count and symbol width in one traversal

        // Never hit in 1000 point test
        // std::cerr << "A this: " << this << " | "
        //           << "cur_node_bitmap_pos: " << parent_node_bitmap_pos << std::endl;

        auto [n_children_to_skip, parent_node_width] = bitmap_.get_skip_count_and_symbol_width(
            parent_node_pos, parent_node_bitmap_pos, child_symbol);

        // We're recursing into the first child. No tree traversal is required.
        if (n_children_to_skip == 0) {
            parent_node_bitmap_pos += parent_node_width;
            return parent_node_pos + 1;
        }

        // Records how many nodes there are left to "skip" at a given level.
        // "stack[3] = 5" => need to iterate through 5 more children before we can return
        //                   from the current child in trie level 3.
        node_pos_t nodes_to_skip[MAX_TRIE_TREEBLOCK_DEPTH + 1] = {};
        nodes_to_skip[0] = n_children_to_skip;
        int dfs_level = 0; // signed var for easy checks

        // Initialize traversal cursor to first child
        node_pos_t cur_node_pos = parent_node_pos + 1;
        node_pos_t cur_node_bitmap_pos = parent_node_bitmap_pos + parent_node_width;

        // Initialize frontier tracking with cached pointers and avoid calls in loop
        constexpr node_pos_t max_frontier_pos = (node_pos_t)-1;
        node_pos_t next_frontier_pos = max_frontier_pos;
        if (frontiers_ != nullptr) {
            if (current_frontier_pos < num_frontiers_ &&
                cur_node_pos > get_frontier_node_pos(current_frontier_pos))
                ++current_frontier_pos;
            if (current_frontier_pos < num_frontiers_)
                next_frontier_pos = get_frontier_node_pos(current_frontier_pos);
        }

        // descend one level before starting loop
        ++current_trie_level;

        // Iterative DFS-like traversal: stop when we've accounted for enough
        // children (i.e., child_stop_count >= stack[0]) or when we've exhausted nodes
        const node_pos_t treeblock_size_nodes = this->size_in_nodes();
        while (cur_node_pos < treeblock_size_nodes && dfs_level >= 0 && nodes_to_skip[0] > 0) {

            // Skip frontier nodes.
            if (frontiers_ != nullptr && cur_node_pos == next_frontier_pos) {
                // skip frontier nodes by advancing frontier index and reduce
                // current-trie_depth counter
                ++current_frontier_pos;
                if (current_frontier_pos >= num_frontiers_)
                    next_frontier_pos = max_frontier_pos;
                else
                    next_frontier_pos = get_frontier_node_pos(current_frontier_pos);
                nodes_to_skip[dfs_level]--;
            } else if (current_trie_level < MAX_TRIE_DEPTH - 1) {
                // push children count of this node to the stack.
                ++dfs_level;

                // Never hit in 1000 point test
                // std::cerr << "B this: " << this << " | "
                //           << "cur_node_bitmap_pos: " << cur_node_bitmap_pos << std::endl;
                auto [node_width, node_num_children] =
                    bitmap_.get_symbol_width_and_num_children(cur_node_pos, cur_node_bitmap_pos);
                nodes_to_skip[dfs_level] = node_num_children;
                cur_node_bitmap_pos += node_width;
                ++current_trie_level;
            } else {
                // We've encountered a leaf of this subtree at level MAX_TRIE_DEPTH - 1. Consume
                // one child.
                --nodes_to_skip[dfs_level];

                // Never hit in 1000 point test
                // std::cerr << "C this: " << this << " | "
                //           << "cur_node_bitmap_pos: " << cur_node_bitmap_pos << std::endl;
                cur_node_bitmap_pos += bitmap_.get_symbol_width(cur_node_pos, cur_node_bitmap_pos);
            }

            ++cur_node_pos;

            // collapse empty levels
            while (dfs_level >= 0 && nodes_to_skip[dfs_level] == 0) {
                dfs_level--;
                current_trie_level--;
                if (dfs_level >= 0) // child subtree traversed => subtract from parent.
                    --nodes_to_skip[dfs_level];
            }
        }

        parent_node_bitmap_pos = cur_node_bitmap_pos;

        return cur_node_pos;
    }

    /// @brief Navigate to a child node using pre-computed skip count and width.
    ///
    /// Optimized version of get_child_node_unsafe that accepts pre-computed values
    /// from has_symbol_with_info, avoiding redundant traversal of the hierarchical
    /// encoding. Use this when the caller has already computed n_children_to_skip
    /// and parent_node_width.
    ///
    /// SIDE EFFECTS! Changes the parent_node_bitmap_pos, and frontier pos!
    ///
    /// @param parent_node_pos Parent node (preorder position)
    /// @param parent_node_bitmap_pos Bit position (SIDE EFFECT! Updated to child's position)
    /// @param current_trie_level Current trie level
    /// @param current_frontier_pos Frontier index (in/out parameter)
    /// @param n_children_to_skip Pre-computed: number of children before target symbol
    /// @param parent_node_width Pre-computed: total bits of parent node's symbol encoding
    /// @return Child node preorder position AND bitmap position (via parent_node_bitmap_pos).
    node_pos_t get_child_node_unsafe_pre_calc(node_pos_t parent_node_pos,
                                              node_pos_t &parent_node_bitmap_pos,
                                              trie_level_t current_trie_level,
                                              node_pos_t &current_frontier_pos,
                                              node_pos_t n_children_to_skip,
                                              node_pos_t parent_node_width)
    {
        // Fast-path checks
        assert(this->size_in_nodes() != 0);
        if (current_trie_level == MAX_TRIE_DEPTH)
            return parent_node_pos;

        // We're recursing into the first child. No tree traversal is required.
        if (n_children_to_skip == 0) {
            parent_node_bitmap_pos += parent_node_width;
            return parent_node_pos + 1;
        }

        // Records how many nodes there are left to "skip" at a given level.
        // "stack[3] = 5" => need to iterate through 5 more children before we can return
        //                   from the current child in trie level 3.
        node_pos_t nodes_to_skip[MAX_TRIE_TREEBLOCK_DEPTH + 1] = {};
        nodes_to_skip[0] = n_children_to_skip;
        int dfs_level = 0; // signed var for easy checks

        // Initialize traversal cursor to first child
        node_pos_t cur_node_pos = parent_node_pos + 1;
        node_pos_t cur_node_bitmap_pos = parent_node_bitmap_pos + parent_node_width;

        // Initialize frontier tracking with cached pointers and avoid calls in loop
        constexpr node_pos_t max_frontier_pos = (node_pos_t)-1;
        node_pos_t next_frontier_pos = max_frontier_pos;
        if (frontiers_ != nullptr) {
            if (current_frontier_pos < num_frontiers_ &&
                cur_node_pos > get_frontier_node_pos(current_frontier_pos))
                ++current_frontier_pos;
            if (current_frontier_pos < num_frontiers_)
                next_frontier_pos = get_frontier_node_pos(current_frontier_pos);
        }

        // descend one level before starting loop
        ++current_trie_level;

        // Iterative DFS-like traversal: stop when we've accounted for enough
        // children (i.e., child_stop_count >= stack[0]) or when we've exhausted nodes
        const node_pos_t treeblock_size_nodes = this->size_in_nodes();
        while (cur_node_pos < treeblock_size_nodes && dfs_level >= 0 && nodes_to_skip[0] > 0) {

            // Skip frontier nodes.
            if (frontiers_ != nullptr && cur_node_pos == next_frontier_pos) {
                // skip frontier nodes by advancing frontier index and reduce
                // current-trie_depth counter
                ++current_frontier_pos;
                if (current_frontier_pos >= num_frontiers_)
                    next_frontier_pos = max_frontier_pos;
                else
                    next_frontier_pos = get_frontier_node_pos(current_frontier_pos);
                nodes_to_skip[dfs_level]--;
            } else if (current_trie_level < MAX_TRIE_DEPTH - 1) {
                // push children count of this node to the stack.
                ++dfs_level;

                // upto 190 times (1000 datapoints)
                // std::cerr << "D this: " << this << " | "
                //           << "cur_node_bitmap_pos: " << cur_node_bitmap_pos << std::endl;

                // Compute two values in one "decoding" of a symbol.
                auto [node_width, node_num_children] =
                    bitmap_.get_symbol_width_and_num_children(cur_node_pos, cur_node_bitmap_pos);

                nodes_to_skip[dfs_level] = node_num_children;
                cur_node_bitmap_pos += node_width;
                ++current_trie_level;
            } else {
                // We've encountered a leaf of this subtree at level MAX_TRIE_DEPTH - 1. Consume
                // one child.
                --nodes_to_skip[dfs_level];

                // upto 190 times (1000 datapoints)
                // std::cerr << "E this: " << this << " | "
                //           << "cur_node_bitmap_pos: " << cur_node_bitmap_pos << std::endl;
                cur_node_bitmap_pos += bitmap_.get_symbol_width(cur_node_pos, cur_node_bitmap_pos);
            }

            ++cur_node_pos;

            // collapse empty levels
            while (dfs_level >= 0 && nodes_to_skip[dfs_level] == 0) {
                dfs_level--;
                current_trie_level--;
                if (dfs_level >= 0) // child subtree traversed => subtract from parent.
                    --nodes_to_skip[dfs_level];
            }
        }

        parent_node_bitmap_pos = cur_node_bitmap_pos;

        return cur_node_pos;
    }

    /// @brief Helper function for range search. Gets child node via symbol if
    /// it exists.
    ///
    /// Uses has_symbol_with_full_info to get existence check, skip count,
    /// total children, and symbol width in one traversal.
    ///
    /// @return Child node preorder position, or null_node if child doesn't
    /// exist
    node_pos_t range_search_get_child_node_pre_calc(
        node_pos_t node_pos, node_bitmap_pos_t &node_bitmap_pos, trie_level_t current_level,
        node_pos_t &current_frontier, node_pos_t stack[MAX_TRIE_DEPTH + 1], int &sTop,
        node_pos_t &current_node_bitmap_pos, node_pos_t &current_node_pos,
        node_pos_t &next_frontier_pos, node_pos_t &current_frontier_cont,
        node_bitmap_pos_t symbol_width, node_bitmap_pos_t n_children_total,
        node_bitmap_pos_t n_children_to_skip)
    {
        if (node_pos >= size_in_nodes())
            return null_node;

        if (current_level == MAX_TRIE_DEPTH - 1)
            return node_pos;

        return range_search_get_child_node_unsafe_pre_calc(
            node_pos, node_bitmap_pos, current_level, current_frontier, stack, sTop,
            current_node_bitmap_pos, current_node_pos, next_frontier_pos, current_frontier_cont,
            n_children_to_skip, n_children_total, symbol_width);
    }

    /// @brief Helper function for range search. Navigates to child node via
    /// symbol using pre-computed values.
    ///
    /// Optimized version that accepts pre-computed skip count, total children,
    /// and symbol width from has_symbol_with_full_info.
    ///
    /// @param node_pos Parent node (preorder position)
    /// @param node_bitmap_pos Bit position (updated to child's position)
    /// @param current_level Current trie level
    /// @param current_frontier Frontier index (in/out parameter)
    /// @param stack Shared stack for DFS traversal (reused across calls)
    /// @param stack_top Stack top pointer (in/out, -1 = first call)
    /// @param n_children_to_skip Pre-computed: children before target symbol
    /// @param n_children_total Pre-computed: total children under this node
    /// @param symbol_width Pre-computed: width of parent node's symbol encoding
    /// @return Child node preorder position
    node_pos_t range_search_get_child_node_unsafe_pre_calc(
        node_pos_t node_pos, node_bitmap_pos_t &node_bitmap_pos, trie_level_t current_level,
        node_pos_t &current_frontier, node_pos_t stack[MAX_TRIE_DEPTH + 1], int &stack_top,
        node_bitmap_pos_t &current_node_bitmap_pos, node_pos_t &current_node_pos,
        node_pos_t &next_frontier_pos, node_pos_t &current_frontier_cont,
        node_pos_t n_children_to_skip, node_pos_t n_children_total, node_pos_t symbol_width)
    {
        if (current_level == MAX_TRIE_DEPTH) {
            return node_pos;
        }

        node_pos_t child_stop_count = n_children_total - n_children_to_skip;

        bool first_time = false;
        if (stack_top == -1) {
            stack_top++;
            stack[stack_top] = n_children_total;
            first_time = true;
        }

        if (first_time)
            current_node_bitmap_pos = node_bitmap_pos + symbol_width;
        if (first_time)
            current_node_pos = node_pos + 1;

        if (first_time) {
            if (frontiers_ != nullptr && current_frontier < num_frontiers_ &&
                current_node_pos > get_frontier_node_pos(current_frontier))
                ++current_frontier;
        } else {
            current_frontier = current_frontier_cont;
        }

        if (first_time) {
            if (num_frontiers_ == 0 || current_frontier >= num_frontiers_)
                next_frontier_pos = (node_pos_t)-1;
            else
                next_frontier_pos = get_frontier_node_pos(current_frontier);
        }

        current_level++;

        while (
            (current_node_pos < size_in_nodes() && stack_top >= 0 && child_stop_count < stack[0]) ||
            !first_time) {

            // First time needs to go down first.
            first_time = true;
            if (current_node_pos == next_frontier_pos) {
                current_frontier++;
                if (num_frontiers_ == 0 || current_frontier >= num_frontiers_)
                    next_frontier_pos = (node_pos_t)-1;
                else
                    next_frontier_pos = get_frontier_node_pos(current_frontier);
                stack[stack_top]--;
            }
            // It is "-1" because current_level is 0th indexed.
            else if (current_level < MAX_TRIE_DEPTH - 1) {
                stack_top++;

                // upto 2 times (1000 datapoints)
                // std::cerr << "F this: " << this << " | "
                //           << "cur_node_bitmap_pos: " << current_node_bitmap_pos << std::endl;
                auto [node_bits2, node_children2] = bitmap_.get_symbol_width_and_num_children(
                    current_node_pos, current_node_bitmap_pos);
                assert(node_bits2 ==
                       bitmap_.get_symbol_width(current_node_pos, current_node_bitmap_pos));
                assert(node_children2 ==
                       bitmap_.get_num_children(current_node_pos, current_node_bitmap_pos));

                stack[stack_top] = node_children2;
                current_node_bitmap_pos += node_bits2;
                current_level++;
            } else {
                stack[stack_top]--;

                // upto 2 times (1000 datapoints)
                // std::cerr << "G this: " << this << " | "
                //           << "cur_node_bitmap_pos: " << current_node_bitmap_pos << std::endl;
                current_node_bitmap_pos +=
                    bitmap_.get_symbol_width(current_node_pos, current_node_bitmap_pos);
            }
            current_node_pos++;

            while (stack_top >= 0 && stack[stack_top] == 0) {
                stack_top--;
                current_level--;
                if (stack_top >= 0)
                    stack[stack_top]--;
            }
        }
        node_bitmap_pos = current_node_bitmap_pos;
        current_frontier_cont = current_frontier;
        return current_node_pos;
    }

    // Helper function for extend_treeblock_and_insert() (called from
    // tree_block::insert()). Appends a symbol to the end of a treeblock. Called
    // when we're trying to
    //  1) insert into an empty treeblock -- OR
    //  2) our new node is at the end of the DFUDS tree
    // => can just be appended to the treeblock.
    void append_to_treeblock(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                             data_point<DIMENSION> *leaf_point, trie_level_t trie_depth,
                             node_pos_t current_num_children)
    {
        // Calculate required additional space
        node_pos_t increased_nodes = MAX_TRIE_DEPTH - trie_depth;
        // Calculate total bits needed for all levels
        // Each level needs bits to represent its children
        node_pos_t increased_bits = DIMENSION * (MAX_TRIE_DEPTH - trie_depth);

        // Make room for new data bits and flag (node existence) bits.
        bitmap_.increase_bitmap_size(increased_bits);
        bitmap_.increase_flagmap_size(increased_nodes);
        bitmap_.clear_bitmap_pos(node_bitmap_pos, increased_bits);
        bitmap_.clear_flagmap_pos(node_pos, increased_nodes);

        // Write Morton symbols for each level down to MAX_TRIE_DEPTH
        // Everything is collapsed (no branching yet in this new path)
        for (trie_level_t current_level = trie_depth; current_level < MAX_TRIE_DEPTH;
             current_level++) {
            current_num_children = DIMENSION;
            bitmap_.create_collapsed_node_unsafe(node_pos, node_bitmap_pos,
                                                 leaf_point->leaf_to_symbol(current_level));
            node_bitmap_pos += current_num_children;
            node_pos++;
        }
    }

    // Helper function for extend_treeblock_and_insert().
    // Given a node to insert into the tree, this function expands the treeblock
    // to clear enough room for the new node (taking care of details like
    // "shifting the rest of the treeblock over to make room").
    void extend_treeblock(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                          trie_level_t trie_depth, node_pos_t current_frontier)
    {
        // Calculate space needed for remaining levels
        node_pos_t increased_nodes = MAX_TRIE_DEPTH - trie_depth;
        node_pos_t increased_bits = DIMENSION * (MAX_TRIE_DEPTH - trie_depth);

        // Insert space into the compressed_bitmap
        if (node_pos == size_in_nodes()) {
            // Appending at the end: simple allocation
            bitmap_.increase_bitmap_size(increased_bits);
            bitmap_.increase_flagmap_size(increased_nodes);
            bitmap_.clear_bitmap_pos(node_bitmap_pos, increased_bits);
            bitmap_.clear_flagmap_pos(node_pos, increased_nodes);
        } else {
            // Inserting in middle: shift existing data to make room
            bitmap_.shift_backward(node_pos, node_bitmap_pos, increased_nodes, increased_bits);

            // Update frontier_node preorder numbers (they shifted right too)
            for (node_pos_t j = current_frontier; j < get_num_frontiers(); j++) {
                if (get_frontier_node_pos(j) >= node_pos)
                    set_frontier_node_pos(j, get_frontier_node_pos(j) + increased_nodes);
            }
        }
    }

    // Helper function for extend_treeblock_and_insert() when data bits were pre-allocated.
    // Data buffer already has enough space (allocated via extra_data_bits in set_symbol),
    // so we only allocate flag bits and shift both data and flags within their buffers.
    void extend_treeblock_flags_only(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                     trie_level_t trie_depth, node_pos_t current_frontier)
    {
        // Calculate space needed for remaining levels
        node_pos_t increased_nodes = MAX_TRIE_DEPTH - trie_depth;
        node_pos_t increased_bits = DIMENSION * (MAX_TRIE_DEPTH - trie_depth);

        // Insert space into the flag buffer only (data already has space)
        if (node_pos == size_in_nodes()) {
            // Appending at the end: simple allocation for flags only
            bitmap_.increase_flagmap_size(increased_nodes);
            bitmap_.clear_bitmap_pos(node_bitmap_pos, increased_bits);
            bitmap_.clear_flagmap_pos(node_pos, increased_nodes);
        } else {
            // Inserting in middle:
            // - Data buffer already has space, but need to shift data to create gap at
            // node_bitmap_pos
            // - Flag buffer needs allocation AND shift

            // Shift data within already-allocated space to create gap
            // Data from [node_bitmap_pos, data_size - increased_bits) needs to move to
            // [node_bitmap_pos + increased_bits, data_size)
            node_pos_t data_size = bitmap_.get_bitmap_size();
            node_pos_t bits_to_copy = data_size - increased_bits - node_bitmap_pos;
            if (bits_to_copy > 0) {
                bitmap_.bulkcopy_backward(data_size - increased_bits, data_size, bits_to_copy,
                                          true);
            }
            bitmap_.clear_bitmap_pos(node_bitmap_pos, increased_bits);

            // Allocate and shift flag buffer
            bitmap_.increase_flagmap_size(increased_nodes);
            node_pos_t orig_flag_size = bitmap_.get_flagmap_size() - increased_nodes;
            bitmap_.bulkcopy_backward(orig_flag_size, bitmap_.get_flagmap_size(),
                                      orig_flag_size - node_pos, false);
            bitmap_.clear_flagmap_pos(node_pos, increased_nodes);

            // Update frontier_node preorder numbers (they shifted right too)
            for (node_pos_t j = current_frontier; j < get_num_frontiers(); j++) {
                if (get_frontier_node_pos(j) >= node_pos)
                    set_frontier_node_pos(j, get_frontier_node_pos(j) + increased_nodes);
            }
        }
    }

    // Helper function for tree_block::insert_at_pos().
    // Adds a symbol into a block. Assumes the treeblock has enough room!
    void extend_treeblock_and_insert(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                     data_point<DIMENSION> *leaf_point, trie_level_t trie_depth,
                                     node_pos_t current_frontier)
    {
        node_pos_t current_num_children = DIMENSION;

        // If we're inserting into an empty block (or at the end of the block),
        // we can just append the new path to the end of the treeblock.
        if (size_in_nodes() == 0 || node_pos == size_in_nodes()) {
            append_to_treeblock(node_pos, node_bitmap_pos, leaf_point, trie_depth,
                                current_num_children);

            return;
        }

        // Add the first element at the position of node/node_pos.
        morton_t symbol = leaf_point->leaf_to_symbol(trie_depth);

        // PHASE 1: follow existng path in the treeblock.

        /*
         *  Optimization: When set_symbol allocates, we pre-allocate extra bits
         *  for the subsequent extend_treeblock operation. This combines two
         *  realloc calls into one.
         *
         *  Each set_symbol call gets extra_data_bits based on the levels that
         *  would remain if THIS call is the one that allocates:
         *  - If set_symbol at depth D allocates, we still need to insert
         *    (MAX_TRIE_DEPTH - D - 1) collapsed symbols, each needing DIMENSION bits.
         */

        node_pos_t n_children_to_skip = 0;
        node_pos_t parent_node_width = 0;

        // Calculate extra bits to allocate for this depth level: remaining levels after navigating
        // to child After set_symbol at depth D succeeds, we call get_child_node_unsafe_pre_calc
        // (D+1) Then extend_treeblock at (D+1) for levels (D+1) to MAX_TRIE_DEPTH-1
        node_pos_t extra_data_bits = DIMENSION * (MAX_TRIE_DEPTH - trie_depth - 1);

        bool child_bit_already_set = !bitmap_.set_child_in_node(
            node_pos, node_bitmap_pos, symbol, bitmap_.node_is_collapsed(node_pos),
            n_children_to_skip, parent_node_width, extra_data_bits);

        // Follow existing path as far as possible, setting bits along the way
        while (trie_depth < MAX_TRIE_DEPTH && child_bit_already_set) {
            node_pos = get_child_node_unsafe_pre_calc(node_pos, node_bitmap_pos, trie_depth,
                                                      current_frontier, n_children_to_skip,
                                                      parent_node_width);

            symbol = leaf_point->leaf_to_symbol(++trie_depth);
            current_num_children = DIMENSION;

            // Recalculate extra bits for the new depth level
            extra_data_bits = DIMENSION * (MAX_TRIE_DEPTH - trie_depth - 1);

            child_bit_already_set = !bitmap_.set_child_in_node(
                node_pos, node_bitmap_pos, symbol, bitmap_.node_is_collapsed(node_pos),
                n_children_to_skip, parent_node_width, extra_data_bits);
        }
        assert(trie_depth < MAX_TRIE_DEPTH);

        // this also implies child_bit_already_set is true, we've found a duplicate
        if (trie_depth == MAX_TRIE_DEPTH - 1) {
            return;
        }

        // The remaining bits aren't encoded in the treeblock.
        // Now add the remaining levels all at once.

        // Navigate to where the new nodes should be inserted
        node_pos =
            get_child_node_unsafe_pre_calc(node_pos, node_bitmap_pos, trie_depth, current_frontier,
                                           n_children_to_skip, parent_node_width);
        trie_depth++;

        // Data bits were pre-allocated by set_symbol via extra_data_bits parameter.
        // Only need to allocate/shift flag bits now.
        extend_treeblock_flags_only(node_pos, node_bitmap_pos, trie_depth, current_frontier);

        // Write the remaining Morton symbols into our newly cleared bitmap region.
        for (trie_level_t current_level = trie_depth; current_level < MAX_TRIE_DEPTH;
             current_level++) {
            node_pos_t current_level_children = DIMENSION;
            bitmap_.create_collapsed_node_unsafe(node_pos, node_bitmap_pos,
                                                 leaf_point->leaf_to_symbol(current_level));
            node_bitmap_pos += current_level_children;
            node_pos++;
        }
    }

    // Helper function for tree_block::insert().
    // Splits a treeblock into two, then inserts a symbol into one of the
    // treeblocks.
    void split_treeblock_and_insert(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                    data_point<DIMENSION> *leaf_point, trie_level_t trie_depth,
                                    node_pos_t current_frontier, trie_level_t cur_treeblock_depth)
    {
        // All nodes are already frontier pointers, can't split more.
        if (size_in_nodes() == num_frontiers_) {
            extend_treeblock_and_insert(node_pos, node_bitmap_pos, leaf_point, trie_depth,
                                        current_frontier);
            return;
        }

        // PHASE 1: select a good subtree to split off.
        node_split_info best_split = select_node_to_split(cur_treeblock_depth);

        // Sanity check: can't split if subtree is too small
        if (best_split.node_count <= 1 || best_split.node_bits == 0 || best_split.node == 0) {
            extend_treeblock_and_insert(node_pos, node_bitmap_pos, leaf_point, trie_depth,
                                        current_frontier);
            return;
        }

        // PHASE 2: create a new treeblock for selected subtree.
        compressed_bitmap::compressed_bitmap<H_LEVEL, CHUNK_WIDTH, DIMENSION> new_dfuds(
            best_split.node_count, best_split.node_bits);

        // TODO(yash): why calculate this here? Is there some side effect later?
        // Note(Leo): decides which block to insert into
        bool insert_in_new_block =
            (node_pos >= best_split.node && node_pos < best_split.node + best_split.node_count);

        bitmap_.copy_bits_to(new_dfuds, best_split.node_pos, best_split.node_bits, true);
        bitmap_.copy_bits_to(new_dfuds, best_split.node, best_split.node_count, false);
        tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *new_treeblock =
            new tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>(new_dfuds);

        assert(best_split.frontier_count + best_split.frontier_node_pos <= num_frontiers_);

        // PHASE 3: Migrate frontier_nodes from old to new tree_block.
        //
        // If the subtree contained any frontier_nodes, they belong to the new
        // tree_block now. Move them over and adjust the indices.

        if (best_split.frontier_count > 0) {
            // allocate space for the frontiers in the new treeblock
            new_treeblock->frontiers_ = (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                best_split.frontier_count, sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));
            new_treeblock->num_frontiers_ = best_split.frontier_count;

            // copy the frontiers from the old treeblock to the new one
            for (node_pos_t i = 0; i < best_split.frontier_count; i++) {
                new_treeblock->frontiers_[i].preorder_ =
                    frontiers_[i + best_split.frontier_node_pos].preorder_ - best_split.node;
                new_treeblock->frontiers_[i].child_treeblock =
                    frontiers_[i + best_split.frontier_node_pos].child_treeblock;
            }

            // remove the frontiers from the old treeblock
            for (node_pos_t i = best_split.frontier_node_pos;
                 i < num_frontiers_ - best_split.frontier_count; i++) {
                frontiers_[i] = frontiers_[i + best_split.frontier_count];
            }

            num_frontiers_ -= best_split.frontier_count;
            frontiers_ = (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)realloc(
                frontiers_,
                num_frontiers_ * sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));
        }

        assert(best_split.node + best_split.node_count <= size_in_nodes());

        // PHASE 4: Replace subtree with frontier_node in old tree_block

        // remove all bits, so that the current node doesn't take space anymore,
        // keeping the flag bit.
        bitmap_.shift_forward(best_split.node + best_split.node_count,
                              best_split.node_pos + best_split.node_bits, best_split.node + 1,
                              best_split.node_pos); // shift the bits to the left

        // now the current node will be a frontier node, and need to insert that
        if (num_frontiers_ > 0) {
            frontiers_ = (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)realloc(
                frontiers_,
                (num_frontiers_ + 1) * sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));

            // shift entire array from best.frontier_node_pos to end to right
            for (node_pos_t i = num_frontiers_; i > best_split.frontier_node_pos; i--) {
                frontiers_[i] = frontiers_[i - 1];
                frontiers_[i].preorder_ -= best_split.node_count - 1;
            }
            frontiers_[best_split.frontier_node_pos].preorder_ = best_split.node;
            frontiers_[best_split.frontier_node_pos].child_treeblock = new_treeblock;
        } else {
            frontiers_ = (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                1, sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));
            frontiers_[0].preorder_ = best_split.node;
            frontiers_[0].child_treeblock = new_treeblock;
        }
        num_frontiers_++;

        // PHASE 5: Recursively insert into the appropriate block.

        // Current node should be placed in new treeblock.
        if (insert_in_new_block) {
            new_treeblock->insert_into_treeblock(
                node_pos - best_split.node, node_bitmap_pos - best_split.node_pos, leaf_point,
                trie_depth, current_frontier - best_split.frontier_node_pos, best_split.node_depth);
            return;
        }

        // Recurse, placing the current node on the old treeblock.
        if (node_pos >= best_split.node + best_split.node_count) {
            // in the second half of the split, still on to the next node. one
            // node denote the frontier.
            node_pos -= best_split.node_count - 1;
            // specific nodes don't take any space
            node_bitmap_pos -= best_split.node_bits;
            // One from the added node
            current_frontier -= best_split.frontier_count - 1;
        }
        insert_into_treeblock(node_pos, node_bitmap_pos, leaf_point, trie_depth, current_frontier,
                              cur_treeblock_depth);
        return;
    }

public:
    /// @brief Insert a datapoint into this tree block at a given position (don't chase frontier
    /// nodes!)
    ///
    /// @param node_pos Preorder position of the _parent_ of the current node.
    /// @param node_bitmap_pos Bit position in compressed_bitmap of the _parent_ of the current
    /// node.
    /// @param leaf_point Data point being inserted
    /// @param trie_depth Current level in trie
    /// @param current_frontier Current frontier index
    /// @param treeblock_depth Root depth of this tree_block
    void insert_into_treeblock(node_pos_t parent_node_pos, node_bitmap_pos_t parent_node_bitmap_pos,
                               data_point<DIMENSION> *leaf_point, trie_level_t trie_depth,
                               node_pos_t current_frontier, trie_level_t treeblock_depth)
    {
        // Base case: this case should not insert anything. it means that a
        // datapoint was already found (already traversed by insert_into_treeblock_layer).
        if (trie_depth == MAX_TRIE_DEPTH) {
            return;
        }

        // If we're inserting a leaf node: just set a bit in the relevant,
        // morton symbol, and we're done!
        if (trie_depth + 1 == MAX_TRIE_DEPTH) {
            morton_t next_symbol = leaf_point->leaf_to_symbol(trie_depth);

            node_pos_t dummy1;
            node_pos_t dummy2;
            // TODO(yash): don't calculate the expensive dummy vars!
            bitmap_.set_child_in_node(parent_node_pos, parent_node_bitmap_pos, next_symbol,
                                      bitmap_.node_is_collapsed(parent_node_pos), dummy1, dummy2);
            return;
        }

        // Does the new symbol fit in the block? (ie: can we extend this block?).
        if (size_in_nodes() + (MAX_TRIE_DEPTH - trie_depth) - 1 <= get_size_class()) {
            extend_treeblock_and_insert(parent_node_pos, parent_node_bitmap_pos, leaf_point,
                                        trie_depth, current_frontier);
            return;
        }
        split_treeblock_and_insert(parent_node_pos, parent_node_bitmap_pos, leaf_point, trie_depth,
                                   current_frontier, treeblock_depth);
    }

    /// @brief Traverse the tree_block hierarchy to find insertion point
    ///
    /// This function navigates through the treeblock, following Morton code
    /// symbols and recursing into frontier_nodes (which point to
    /// sub-tree_blocks) as needed. Once it can't navigate further, it calls
    /// insert() to perform the actual insertion.
    ///
    /// @param leaf_point Data point being inserted
    /// @param level Current level in the overall trie
    /// @param treeblock_depth Root level of the current tree_block (for
    /// splitting decisions)
    void insert_into_treeblock_layer(data_point<DIMENSION> *leaf_point, trie_level_t cur_level,
                                     trie_level_t &treeblock_level)
    {
        node_pos_t node_pos = 0;               // preorder node position in treeblock
        node_bitmap_pos_t node_bitmap_pos = 0; // bit position in the compressed_bitmap
        node_pos_t frontier = 0;               // index into frontiers_[] array

        node_pos_t tmp_node_pos = 0;
        node_bitmap_pos_t tmp_node_bitmap_pos = 0;

        // Traverse the tree block's encoding as far as possible (without
        // modifying treeblock). Recurses into frontier nodes if needed.
        while (cur_level < MAX_TRIE_DEPTH) {
            tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *current_treeblock = this;

            tmp_node_pos =
                get_child_node(current_treeblock, node_pos, tmp_node_bitmap_pos,
                               leaf_point->leaf_to_symbol(cur_level), cur_level, frontier);
            if (tmp_node_pos == null_node)
                break; // Child doesn't exist, need to insert here.

            node_pos = tmp_node_pos;
            node_bitmap_pos = tmp_node_bitmap_pos;
            if (node_pos == size_in_nodes()) {
                break; // Reached end of tree_block
            }

            // If the current node is a frontier node, recurse into it.
            if (get_num_frontiers() > 0 && frontier < get_num_frontiers() &&
                node_pos == get_frontier_node_pos(frontier)) {
                tree_block *next_block = get_child_treeblock(frontier);
                treeblock_level = cur_level + 1;
                next_block->insert_into_treeblock_layer(leaf_point, cur_level + 1, treeblock_level);
                return;
            }
            cur_level++;
        }

        insert_into_treeblock(node_pos, node_bitmap_pos, leaf_point, cur_level, frontier,
                              treeblock_level);
        return;
    }

    /// @brief Recursively search dense tree_block for all points in
    /// [start_point, end_point]
    ///
    /// Traverses the DFUDS-encoded tree to find all points within the query
    /// range. Uses next_symbol() to efficiently iterate only symbols present in
    /// the current node. This algorithm is basically the same as
    /// `md_trie::range_search()`, but operates on treeblock's compressed tree
    /// instead of `std::map`s.
    ///
    /// Algorithm:
    ///   1. At MAX_TRIE_DEPTH: found a matching point, add to results
    ///   2. If frontier node: recurse into child tree_block
    ///   3. Otherwise: iterate symbols in [start_symbol, end_symbol] using
    ///   next_symbol()
    ///   4. For each valid symbol: navigate to child and recurse
    ///
    /// @param start_point Lower bound (modified during recursion)
    /// @param end_point Upper bound (modified during recursion)
    /// @param current_treeblock Current tree_block being searched
    /// @param current_trie_level Current depth in the trie
    /// @param current_node_pos Preorder node position in DFUDS tree
    /// @param current_node_bitmap_pos Bit position in compressed_bitmap
    /// @param current_frontier Index into frontiers_[] array
    /// @param found_points Output vector of matching coordinates.
    ///                     Will be returned in the "ordered" representation!
    void range_search_treeblock(const data_point<DIMENSION> *start_point,
                                const data_point<DIMENSION> *end_point,
                                tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *current_treeblock,
                                trie_level_t current_trie_level, node_pos_t current_node_pos,
                                node_pos_t current_node_bitmap_pos, node_pos_t current_frontier,
                                std::vector<ordered_coordinate_t> &found_points)
    {
        PP_SAVE_NEW_TIMESTAMP(ts_enter);

        // Base case: reached leaf level, found a matching point
        if (current_trie_level == MAX_TRIE_DEPTH) {
            for (n_dimensions_t j = 0; j < DIMENSION; j++) {
                found_points.push_back(start_point->get_ordered_coordinate(j));
            }
            pp_rq_points_returned_[current_trie_level].increment();

            PP_SAVE_NEW_TIMESTAMP(ts_exit);
            pp_rq_total_base[current_trie_level].record(ts_exit - ts_enter);
            pp_rq_total[current_trie_level].record(ts_exit - ts_enter);
            return;
        }

        // Boundary check: traversed beyond tree
        if (current_node_pos >= size_in_nodes()) {
            PP_SAVE_NEW_TIMESTAMP(ts_exit);
            pp_rq_total_boundary[current_trie_level].record(ts_exit - ts_enter);
            pp_rq_total[current_trie_level].record(ts_exit - ts_enter);
            return;
        }

        // If current node is a frontier node, recurse into its child tree_block
        if (get_num_frontiers() > 0 && current_frontier < get_num_frontiers() &&
            current_node_pos == get_frontier_node_pos(current_frontier)) {
            tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *new_current_block =
                get_child_treeblock(current_frontier);
            node_pos_t new_current_frontier = 0;
            new_current_block->range_search_treeblock(start_point, end_point, new_current_block,
                                                      current_trie_level, 0, 0,
                                                      new_current_frontier, found_points);

            pp_rq_treeblocks_recursed_[current_trie_level].increment();
            PP_SAVE_NEW_TIMESTAMP(ts_exit);
            pp_rq_total_frontier[current_trie_level].record(ts_exit - ts_enter);
            pp_rq_total[current_trie_level].record(ts_exit - ts_enter);
            return;
        }

        // PHASE 1: GENERATE MORTON_T BOUNDS
        PP_SAVE_NEW_TIMESTAMP(ts_b);

        // Transform range query bounds into their morton-space equivalents.
        morton_t start_range_symbol = start_point->leaf_to_symbol(current_trie_level);
        morton_t end_range_symbol = end_point->leaf_to_symbol(current_trie_level);

        // Compute bit masks (used later for for checking which symbols are
        // within our bounds).
        morton_t bound_magic = ~(start_range_symbol ^ end_range_symbol);

        // Store mutable copies of our query bounds.
        data_point<DIMENSION> tmp_start_point = (*start_point);
        data_point<DIMENSION> tmp_end_point = (*end_point);

        // PHASE 2.0: ITERATE TO FIRST CHILD

        PP_SAVE_NEW_TIMESTAMP(ts_a);
        pp_rq_define_bounds[current_trie_level].record(ts_a - ts_b);

        // Find the first symbol >= start_range_symbol that exists in this
        // treeblock. (next_symbol() efficiently skips missing symbols using the
        // DFUDS bitmap). This code doesn't _navigate_ to the child node! It just
        // uses the information in the parent node to learn all the children that
        // exist in the tree (and what morton_t choice they represent).

        bool collapsed_encoding = bitmap_.node_is_collapsed(current_node_pos);

#ifndef NDEBUG
        bool inner_top_level_collapse = false;
#endif
        node_bitmap_pos_t symbol_width = DIMENSION; // Default for collapsed encoding
        node_bitmap_pos_t symbol_num_children = 1;

        // if not collapsed, fill the iteration helpers.
        if (!collapsed_encoding) {
#ifndef NDEBUG
            inner_top_level_collapse = bitmap_.chunk_is_collapsed(current_node_bitmap_pos);
#endif
            std::tie(symbol_width, symbol_num_children) =
                bitmap_.get_hier_encoding_offsets_width_and_children(current_node_bitmap_pos);
        }

        node_pos_t n_children_to_skip = 0;

#ifndef NDEBUG
        node_bitmap_pos_t h_levels_pos_iter[H_LEVEL + 1] = {};
        node_bitmap_pos_t h_levels_bit_bases[H_LEVEL + 1] = {};
        node_bitmap_pos_t h_levels_words[H_LEVEL + 1] = {};
#endif

        level_info ns_stack[H_LEVEL + 1];
        int ns_stack_top = -1;
        morton_t base_symbol = morton_t::zero();

        node_pos_t arg_current_node_bitmap_pos = current_node_bitmap_pos;

        // init the first call
        auto [current_symbol, initial_skip] = bitmap_.next_symbol_reuse_iter(
            start_range_symbol, end_range_symbol, collapsed_encoding, ns_stack, ns_stack_top, true,
            arg_current_node_bitmap_pos, base_symbol);

#ifndef NDEBUG
        auto [debug_current_symbol, debug_initial_skip] =
            bitmap_.next_symbol(start_range_symbol, current_node_bitmap_pos, end_range_symbol,
                                collapsed_encoding, inner_top_level_collapse, h_levels_pos_iter,
                                h_levels_bit_bases, h_levels_words, symbol_width);

        assert(debug_current_symbol == current_symbol);
        assert(debug_initial_skip == initial_skip);
#endif

        n_children_to_skip = initial_skip;

        PP_SAVE_TIMESTAMP(ts_b);
        pp_rq_get_next_symbol[current_trie_level].record(ts_b - ts_a);

        if (current_symbol.is_null()) {
            PP_SAVE_NEW_TIMESTAMP(ts_exit);
            pp_rq_total_null_sym[current_trie_level].record(ts_exit - ts_enter);
            pp_rq_total[current_trie_level].record(ts_exit - ts_enter);
            return;
        }
        if (current_symbol > end_range_symbol) {
            PP_SAVE_NEW_TIMESTAMP(ts_exit);
            pp_rq_total_large_sym[current_trie_level].record(ts_exit - ts_enter);
            pp_rq_total[current_trie_level].record(ts_exit - ts_enter);
            return;
        }

        // Stack for optimized child navigation (reused across iterations)
        //
        // TODO(yash): When we are recursing through our tree in a DFS manner, the stack
        //             counts how many nodes are still in this level that we have yet to
        //             process. This is important for an unknown reason. Leo believes it's
        //             cause we need to know when to "stop traversing".
        //             "This is a replacement for the GPU stack because GPU stacks suck".
        //             Go talk to Ziming and understand what's happening here.
        node_pos_t stack_range_search[MAX_TRIE_DEPTH + 1];
        int sTop_range_search = -1;
        node_pos_t current_node_pos_range_search = 0;
        node_pos_t current_node_range_search = 0;
        node_pos_t next_frontier_preorder_range_search = 0;
        node_pos_t current_frontier_cont = 0;

        PP_SAVE_TIMESTAMP(ts_a);
        pp_rq_stack_alloc[current_trie_level].record(ts_a - ts_b);

        do {
            PP_SAVE_NEW_TIMESTAMP(ts_loop_enter);
            assert(bitmap_.has_child(current_node_pos, current_node_bitmap_pos, current_symbol));

            // If symbol isn't within our query bounds, move onto the next symbol.
            if (morton_t::masked_not_equal(start_range_symbol, current_symbol, bound_magic)) {
                PP_SAVE_TIMESTAMP(ts_a);

                auto [next_sym, next_skip] = bitmap_.next_symbol_reuse_iter(
                    current_symbol + 1, end_range_symbol, collapsed_encoding, ns_stack,
                    ns_stack_top, false, arg_current_node_bitmap_pos, base_symbol);

#ifndef NDEBUG
                auto [debug_next_sym, debug_next_skip] = bitmap_.next_symbol(
                    current_symbol + 1, current_node_bitmap_pos, end_range_symbol,
                    collapsed_encoding, inner_top_level_collapse, h_levels_pos_iter,
                    h_levels_bit_bases, h_levels_words, symbol_width);

                assert(debug_next_sym == next_sym);
                if (!debug_next_sym.is_null())
                    assert(debug_next_skip + 1 == next_skip);
#endif

                current_symbol = next_sym;
                n_children_to_skip += next_skip;

                PP_SAVE_NEW_TIMESTAMP(ts_loop_exit);

                pp_rq_get_next_symbol[current_trie_level].record(ts_loop_exit - ts_a);
                pp_rq_test_subtree_range[current_trie_level].record(ts_a - ts_loop_enter);
                pp_rq_loop[current_trie_level].record(ts_loop_exit - ts_loop_enter);
                pp_rq_symbol_not_in_range_[current_trie_level].increment();
                continue;
            }

            PP_SAVE_TIMESTAMP(ts_a);
            pp_rq_test_subtree_range[current_trie_level].record(ts_a - ts_loop_enter);

            // PHASE 3: GET CHILD NODE

            tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *new_current_block = current_treeblock;
            node_pos_t new_current_frontier = current_frontier;
            node_pos_t new_current_node_bitmap_pos = current_node_bitmap_pos;

            // Navigate to child node.
            node_pos_t new_current_node_pos;

            // two things that next_symbol_reuse should account for
            //      1. Next iteration should continue from where we left off
            //      2. Get information for these

            // what are the parameters needed here:
            //      1. symbol_width: get to next node of interest
            //      2. n_children_total: stateful iteration
            //      3. n_children_to_skip: current skip number

            if (current_trie_level < MAX_TRIE_DEPTH - 1) { /* At least 1 levels to the bottom */
                // SIDE EFFECT! Changes the passed in frontier, etc!
                new_current_node_pos = current_treeblock->range_search_get_child_node_pre_calc(
                    current_node_pos, new_current_node_bitmap_pos, current_trie_level,
                    new_current_frontier, stack_range_search, sTop_range_search,
                    current_node_pos_range_search, current_node_range_search,
                    next_frontier_preorder_range_search, current_frontier_cont, symbol_width,
                    symbol_num_children, n_children_to_skip);
            } else {
                new_current_node_pos = current_treeblock->get_child_node_pre_calc(
                    new_current_block, current_node_pos, new_current_node_bitmap_pos,
                    current_symbol, current_trie_level, new_current_frontier, n_children_to_skip,
                    symbol_width);
            }

            PP_SAVE_TIMESTAMP(ts_b);
            pp_rq_get_child_node[current_trie_level].record(ts_b - ts_a);

            // PHASE 4: RECURSE INTO RANGE

            // Update query ranges before recursion.
            point_t::shrink_query_bounds(&tmp_start_point, &tmp_end_point, current_symbol,
                                         current_trie_level);

            size_t points_returned = found_points.size();

            PP_SAVE_TIMESTAMP(ts_a);
            pp_rq_adjust_bounds_a[current_trie_level].record(ts_a - ts_b);

            current_treeblock->range_search_treeblock(
                &tmp_start_point, &tmp_end_point, current_treeblock, current_trie_level + 1,
                new_current_node_pos, new_current_node_bitmap_pos, new_current_frontier,
                found_points);

            PP_SAVE_TIMESTAMP(ts_b);
            pp_rq_recurse_in_range[current_trie_level].record(ts_b - ts_a);

            // Restore original ranges for next iteration
            tmp_start_point = *start_point;
            tmp_end_point = *end_point;

            PP_SAVE_TIMESTAMP(ts_a);
            pp_rq_adjust_bounds_b[current_trie_level].record(ts_a - ts_b);
            points_returned = found_points.size() - points_returned;
            pp_rq_points_returned_[current_trie_level].record(points_returned);

            {
                auto [next_sym, next_skip] = bitmap_.next_symbol_reuse_iter(
                    current_symbol + 1, end_range_symbol, collapsed_encoding, ns_stack,
                    ns_stack_top, false, arg_current_node_bitmap_pos, base_symbol);

#ifndef NDEBUG
                auto [debug_next_sym, debug_next_skip] = bitmap_.next_symbol(
                    current_symbol + 1, current_node_bitmap_pos, end_range_symbol,
                    collapsed_encoding, inner_top_level_collapse, h_levels_pos_iter,
                    h_levels_bit_bases, h_levels_words, symbol_width);

                assert(debug_next_sym == next_sym);
                if (!debug_next_sym.is_null())
                    assert(debug_next_skip + 1 == next_skip);
#endif

                current_symbol = next_sym;
                n_children_to_skip += next_skip;
            }

            PP_SAVE_TIMESTAMP(ts_b);
            pp_rq_get_next_symbol[current_trie_level].record(ts_b - ts_a);
            pp_rq_loop[current_trie_level].record(ts_b - ts_loop_enter);
        } while (!current_symbol.is_null() && current_symbol <= end_range_symbol);

        PP_SAVE_NEW_TIMESTAMP(ts_exit);
        pp_rq_total_good[current_trie_level].record(ts_exit - ts_enter);
        pp_rq_total[current_trie_level].record(ts_exit - ts_enter);
    }

    /// @brief Writes the treeblock to a file.
    /// @param file
    /// @param treeblock_offset where the tree should be written
    /// @param temp_treeblock a fully zeroed-out tree_block for reuse, it's
    ///                       this function's responsibility to free it.
    void serialize(FILE *file, uint64_t treeblock_offset,
                   tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *temp_treeblock, trie_level_t level)
    {

        /* For the purpose of logging stats

                int node_pos = 0;
                for (int i = 0; i < size_in_nodes(); i++) {
                if (bitmap_.node_is_collapsed(i))
                {
                nc_count[1]++;
                collapsed++;
                }
                else
                {
                nc_count[bitmap_.get_num_children(i, node_pos,
                level_to_num_children[level])]++; uncollapsed++;
                }
                    node_pos += bitmap_.get_symbol_width(i, level);
                }
        */

        // this treeblock should not be here
        assert(pointers_to_offsets_map.find((uint64_t)(this)) == pointers_to_offsets_map.end());

        // good old start, current_offset is at the next write location
        pointers_to_offsets_map.insert({(uint64_t)(this), treeblock_offset});

        // populate the previous dummy tree_block with the correct data
        memcpy(temp_treeblock, this, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>));

        // deal with pointers
        // 1) parent_combined_ptr_ (no more)
        // 2) bitmap_ (converted into struct within)
        // 3) frontiers_

        // todo:
        /* 1) parent_combined_ptr_ (no more) */

        /* 2) bitmap_ (converted into struct within) */

        uint64_t data_offset = 0;
        uint64_t flags_offset = 0;

        bitmap_.serialize(file, data_offset, flags_offset);
        temp_treeblock->bitmap_.set_bitmap(data_offset);
        temp_treeblock->bitmap_.set_flagmap(flags_offset);

        // current_offset is still at the next writing location of the file

        /* 3) frontiers_ */
        // need to perform for each frontier if exists
        if (this->num_frontiers_ != 0) {
            assert(pointers_to_offsets_map.find((uint64_t)(this->frontiers_)) ==
                   pointers_to_offsets_map.end());
            // frontiers_ not found, need to create, a list of frontiers of size
            // num_frontiers_

            // writing to current_offset, so can assign
            temp_treeblock->frontiers_ =
                (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)current_offset;

            // storing where the frontiers_ are started to be stored
            uint64_t frontiers_start_offset = current_offset;

            // create wipeout space for frontier_nodes to be inserted later,
            // increment current_offset accordingly
            frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *temp_frontiers_ =
                (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                    num_frontiers_, sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));

            fwrite(temp_frontiers_, sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>),
                   num_frontiers_, file);
            pointers_to_offsets_map.insert({(uint64_t)(this->frontiers_), current_offset});

            current_offset +=
                sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>) * num_frontiers_;

            memcpy(temp_frontiers_, this->frontiers_,
                   sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>) * num_frontiers_);

            // again, current_offset is at the next writing location of the file
            for (node_pos_t i = 0; i < this->num_frontiers_; i++) {
                // check and process the tree_block pointed by the frontier node
                assert(pointers_to_offsets_map.find(
                           (uint64_t)(this->frontiers_[i].child_treeblock)) == // bug:
                       pointers_to_offsets_map.end());

                temp_frontiers_[i].child_treeblock =
                    (tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)current_offset;

                // serialize the block

                // create a empty tree_block
                tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *wipe_out_treeblock =
                    (tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                        1, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>));
                fwrite(wipe_out_treeblock, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1,
                       file);

                uint64_t wipe_out_treeblock_offset = current_offset;
                current_offset += sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>);

                this->frontiers_[i].child_treeblock->serialize(file, wipe_out_treeblock_offset,
                                                               wipe_out_treeblock, level + 1);
            }
            // write the list of frontiers
            if ((uint64_t)ftell(file) != frontiers_start_offset) {
                fseek(file, (long)frontiers_start_offset, SEEK_SET);
                fwrite(temp_frontiers_, sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>),
                       num_frontiers_, file);
                fseek(file, 0, SEEK_END);
            } else {
                fwrite(temp_frontiers_, sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>),
                       num_frontiers_, file);
            }
            free(temp_frontiers_);
        }

        // before operation, just
        assert((uint64_t)ftell(file) == current_offset);

        // modified: don't need to deal with primary_key_list no more

        // finally I can write my current block
        if ((uint64_t)ftell(file) != treeblock_offset) {
            fseek(file, (long)treeblock_offset, SEEK_SET);
            fwrite(temp_treeblock, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);
            fseek(file, 0, SEEK_END);
        } else {
            fwrite(temp_treeblock, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);
        }

        free(temp_treeblock);
    }

    void deserialize(uint64_t base_addr)
    {
        // no more parent_combined_ptr_

        if (bitmap_.get_flagmap_size() != 0) {
            assert(bitmap_.get_bitmap_size() != 0);
            assert(bitmap_.get_bitmap() != nullptr);
            assert(bitmap_.get_flagmap() != nullptr);

            bitmap_.deserialize(base_addr);
        }

        if (num_frontiers_) {
            // update the pointer to the correct location
            frontiers_ = (frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)(base_addr +
                                                                            (uint64_t)frontiers_);

            for (node_pos_t i = 0; i < num_frontiers_; i++) {
                assert(frontiers_[i].child_treeblock);
                frontiers_[i].child_treeblock =
                    (tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>
                         *)(base_addr + (uint64_t)frontiers_[i].child_treeblock);
                frontiers_[i].child_treeblock->deserialize(base_addr);
            }
        }
    }

    inline node_pos_t size_in_nodes() const { return bitmap_.get_flagmap_size(); }

    inline node_pos_t size_in_bits() const { return bitmap_.get_bitmap_size(); }

    // Return the smallest power of 2 that's greater than or equal to the number
    // of nodes in this treeblock.
    inline node_pos_t get_size_class() const
    {
        node_pos_t n = size_in_nodes();
        node_pos_t implicit_size_class = 1;
        while (implicit_size_class < n) {
            implicit_size_class <<= 1;
        }
        return std::max(implicit_size_class, MAX_TREE_NODES);
    }

    // get the bytes that this treeblock and its children would take when
    // serialized
    void update_size_recur(trie_level_t level, uint64_t &size)
    {

        bitmap_.update_size(size);

        /* frontiers_ */
        // need to perform for each frontier if exists
        if (this->num_frontiers_ != 0) {

            size += sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>) * num_frontiers_;

            for (n_leaves_t i = 0; i < this->num_frontiers_; i++) {
                size += sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>);

                this->frontiers_[i].child_treeblock->update_size_recur(level + 1, size);
            }
        }
    }

    /// @brief Track storage per trie level for this treeblock and children.
    void get_size_per_level_recur(trie_level_t hosting_level)
    {
        std::vector<uint64_t> bits_per_level(MAX_TRIE_DEPTH + 1, 0);
        std::vector<uint64_t> flags_per_level(MAX_TRIE_DEPTH + 1, 0);
        std::vector<trie_level_t>
            frontier_levels; // Track level at which each frontier is encountered
        uint64_t node_count = traverse_dfuds_for_size(0, 0, 0, hosting_level, bits_per_level,
                                                      flags_per_level, frontier_levels)
                                  .node_count;

        // Record treeblock node count for statistics
        treeblock_node_counts.push_back(node_count);

        uint64_t data_size_bits = bitmap_.get_bitmap_size();
        uint64_t flag_size_bits = bitmap_.get_flagmap_size();

        // Record treeblock data bits for statistics
        treeblock_data_bits.push_back(data_size_bits);

        // Track this treeblock at hosting level
        layer_stats[hosting_level].num_treeblocks += 1;

        // Calculate padding bits for this treeblock (stored at hosting level)
        uint64_t data_padding_bits = data_size_bits > 0 ? (64 - (data_size_bits % 64)) % 64 : 0;
        uint64_t flag_padding_bits = flag_size_bits > 0 ? (64 - (flag_size_bits % 64)) % 64 : 0;
        layer_stats[hosting_level].data_padding_bits += data_padding_bits;
        layer_stats[hosting_level].flag_padding_bits += flag_padding_bits;

        // Frontiers and child treeblocks
        if (num_frontiers_ != 0) {
            // Verify we tracked the correct number of frontier levels
            assert(frontier_levels.size() == num_frontiers_ &&
                   "frontier_levels size should match num_frontiers_");
            layer_stats[hosting_level].metadata_bytes +=
                sizeof(frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>) * num_frontiers_;
            for (node_pos_t i = 0; i < num_frontiers_; i++) {
                layer_stats[hosting_level].metadata_bytes +=
                    sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
                // Child treeblock starts at the level where the frontier was encountered
                trie_level_t child_hosting_level = frontier_levels[i];
                frontiers_[i].child_treeblock->get_size_per_level_recur(child_hosting_level);
            }
        }
    }

    /// @brief Count the number of points (leaf nodes) in this treeblock and its children.
    /// @param base_level The starting level of this treeblock.
    /// @return Number of leaf nodes (points) in this treeblock subtree.
    uint64_t count_points_recur(trie_level_t base_level)
    {
        return traverse_dfuds_for_point_count(0, 0, 0, base_level).point_count;
    }

private:
    /// @brief Traverse DFUDS and count leaf nodes (points at MAX_TRIE_DEPTH - 1).
    /// Returns {node_count, node_bits, frontier_count, point_count}.
    struct point_count_info {
        node_pos_t node_count;
        node_pos_t node_bits;
        node_pos_t frontier_count;
        uint64_t point_count;
    };

    point_count_info traverse_dfuds_for_point_count(node_pos_t node_pos,
                                                    node_bitmap_pos_t node_bitmap_pos,
                                                    node_pos_t frontier, trie_level_t level)
    {
        // If this is a frontier, recurse into child treeblock
        if (frontier < num_frontiers_ && get_frontier_node_pos(frontier) == node_pos) {
            uint64_t child_points = frontiers_[frontier].child_treeblock->count_points_recur(level);
            return {1, 0, 1, child_points};
        }
        // If at leaf level, this is one point
        if (level == MAX_TRIE_DEPTH - 1) {
            node_pos_t bits = bitmap_.get_symbol_width(node_pos, node_bitmap_pos);
            return {1, bits, 0, 1};
        }
        // Get bits and children for traversal
        auto [node_bits, num_children] =
            bitmap_.get_symbol_width_and_num_children(node_pos, node_bitmap_pos);
        point_count_info current = {1, node_bits, 0, 0};
        node_pos_t child_pos = node_pos + 1;
        node_pos_t child_bitmap_pos = node_bitmap_pos + node_bits;
        for (node_pos_t i = 0; i < num_children; i++) {
            auto child_info =
                traverse_dfuds_for_point_count(child_pos, child_bitmap_pos, frontier, level + 1);
            child_pos += child_info.node_count;
            child_bitmap_pos += child_info.node_bits;
            frontier += child_info.frontier_count;
            current.node_count += child_info.node_count;
            current.node_bits += child_info.node_bits;
            current.frontier_count += child_info.frontier_count;
            current.point_count += child_info.point_count;
        }
        return current;
    }

    subtree_info traverse_dfuds_for_size(node_pos_t node_pos, node_bitmap_pos_t node_bitmap_pos,
                                         node_pos_t frontier, trie_level_t level,
                                         std::vector<uint64_t> &bits_per_level,
                                         std::vector<uint64_t> &flags_per_level,
                                         std::vector<trie_level_t> &frontier_levels)
    {
        // Frontier node: counts as 1 flag bit but not collapsed/uncollapsed
        if (frontier < num_frontiers_ && get_frontier_node_pos(frontier) == node_pos) {
            layer_stats[level].frontier_nodes += 1;
            layer_stats[level].flag_bits += 1;
            flags_per_level[level] += 1;
            // Record the level at which this frontier is encountered
            frontier_levels.push_back(level);
            return {1, 0, 1};
        }

        // Every non-frontier node contributes 1 flag bit
        layer_stats[level].flag_bits += 1;
        flags_per_level[level] += 1;

        // Check if collapsed or uncollapsed
        bool node_is_collapsed = bitmap_.node_is_collapsed(node_pos);

        if (level == MAX_TRIE_DEPTH - 1) {
            // Leaf level
            node_pos_t bits = bitmap_.get_symbol_width(node_pos, node_bitmap_pos);
            bits_per_level[level] += bits;
            if (node_is_collapsed) {
                layer_stats[level].collapsed_nodes += 1;
                layer_stats[level].data_bits_collapsed += bits;
            } else {
                layer_stats[level].uncollapsed_nodes += 1;
                layer_stats[level].data_bits_uncollapsed += bits;
            }
            return {1, bits, 0};
        }

        // Get bits and children for this node (works for both collapsed and uncollapsed)
        auto [node_bits, num_children] =
            bitmap_.get_symbol_width_and_num_children(node_pos, node_bitmap_pos);
        bits_per_level[level] += node_bits;

        if (node_is_collapsed) {
            layer_stats[level].collapsed_nodes += 1;
            layer_stats[level].data_bits_collapsed += node_bits;
        } else {
            layer_stats[level].uncollapsed_nodes += 1;
            layer_stats[level].data_bits_uncollapsed += node_bits;
            // Record statistics for uncollapsed nodes only
            uncollapsed_children_counts.push_back(num_children);
            uncollapsed_node_bits.push_back(node_bits);
        }

        subtree_info current_subtree = {1, node_bits, 0};
        node_pos_t child_pos = node_pos + 1;
        node_pos_t child_bitmap_pos = node_bitmap_pos + node_bits;
        for (node_pos_t i = 0; i < num_children; i++) {
            subtree_info child_info =
                traverse_dfuds_for_size(child_pos, child_bitmap_pos, frontier, level + 1,
                                        bits_per_level, flags_per_level, frontier_levels);
            child_pos += child_info.node_count;
            child_bitmap_pos += child_info.node_bits;
            frontier += child_info.frontier_count;
            current_subtree.node_count += child_info.node_count;
            current_subtree.node_bits += child_info.node_bits;
            current_subtree.frontier_count += child_info.frontier_count;
        }
        return current_subtree;
    }

    compressed_bitmap::compressed_bitmap<H_LEVEL, CHUNK_WIDTH, DIMENSION> bitmap_;
    frontier_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *frontiers_ = nullptr;
    node_pos_t num_frontiers_ = 0;
};

#endif // MD_TRIE_TREE_BLOCK_H
