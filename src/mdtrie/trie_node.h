#ifndef MD_TRIE_TRIE_NODE_H
#define MD_TRIE_TRIE_NODE_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sys/mman.h>

#include "defs.h"

template <typename Node> struct ChildIter {
    enum class Backend { MAP, ARRAY };

    Backend backend;

    union {
        typename std::map<morton_t, Node *>::const_iterator map_it;
        const IndexPtrPair *array_it;
    };

    // --- Constructors ---

    // Constructor for MAP backend
    explicit ChildIter(typename std::map<morton_t, Node *>::const_iterator it)
        : backend(Backend::MAP), map_it(it)
    {
    }

    // Constructor for ARRAY backend
    explicit ChildIter(const IndexPtrPair *it) : backend(Backend::ARRAY), array_it(it) {}

    // Default constructor (for e.g., end() construction)
    ChildIter() : backend(Backend::ARRAY), array_it(nullptr) {}

    // --- Comparison ---
    bool operator!=(const ChildIter &other) const
    {
        if (backend != other.backend)
            return true;
        return (backend == Backend::MAP) ? (map_it != other.map_it) : (array_it != other.array_it);
    }

    // --- Increment ---
    ChildIter &operator++()
    {
        if (backend == Backend::MAP)
            ++map_it;
        else
            ++array_it;
        return *this;
    }

    // --- Accessors ---
    morton_t key() const { return (backend == Backend::MAP) ? map_it->first : array_it->index; }

    Node *value() const
    {
        return (backend == Backend::MAP) ? map_it->second : reinterpret_cast<Node *>(array_it->ptr);
    }
};

template <typename Node> struct ChildView {
    using iterator = ChildIter<Node>;

    typename ChildIter<Node>::Backend backend;
    const void *ptr;
    size_t count = 0;

    // MAP accessor
    const std::map<morton_t, Node *> &map_ref() const
    {
        return *static_cast<const std::map<morton_t, Node *> *>(ptr);
    }

    // ARRAY accessor
    const IndexPtrPair *array_start() const { return static_cast<const IndexPtrPair *>(ptr); }

    // begin()
    iterator begin() const
    {
        if (backend == iterator::Backend::MAP)
            return iterator(map_ref().begin());
        else
            return iterator(array_start());
    }

    // end()
    iterator end() const
    {
        if (backend == iterator::Backend::MAP)
            return iterator(map_ref().end());
        else
            return iterator(array_start() + count);
    }

    // lower_bound()
    iterator lower_bound(const morton_t &key) const
    {
        if (backend == iterator::Backend::MAP) {
            return iterator(map_ref().lower_bound(key));
        } else {
            auto start = array_start();
            auto end = start + count;

            auto it =
                std::lower_bound(start, end, key, [](const IndexPtrPair &p, const morton_t &k) {
                    return p.index < k;
                });
            return iterator(it);
        }
    }

    // upper_bound()
    iterator upper_bound(const morton_t &key) const
    {
        if (backend == iterator::Backend::MAP) {
            return iterator(map_ref().upper_bound(key));
        } else {
            auto start = array_start();
            auto end = start + count;

            auto it =
                std::upper_bound(start, end, key, [](const morton_t &k, const IndexPtrPair &p) {
                    return k < p.index;
                });
            return iterator(it);
        }
    }

    // operator[] = always key lookup
    Node *operator[](const morton_t &key) const
    {
        auto it = lower_bound(key);
        if (it == end())
            return nullptr;
        return (it.key() == key) ? it.value() : nullptr;
    }
};

// forward define tree_block
template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION> class tree_block;

template <trie_level_t H_LEVEL, node_pos_t CHUNK_WIDTH, n_dimensions_t DIMENSION> class trie_node
{

public:
    explicit trie_node(bool is_leaf)
    {
        if (!is_leaf) {
            trie_or_treeblock_ptr_ =
                new std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *>();
        }
    }

    void delete_non_leaf_node()
    {
        if (trie_or_treeblock_ptr_) {
            delete (std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *)
                trie_or_treeblock_ptr_;
        }
    }

    inline trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *get_child(const morton_t &symbol)
    {
        if (!DESERIALIZED_MDTRIE) {
            auto m = static_cast<
                const std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *>(
                trie_or_treeblock_ptr_);

            auto it = m->find(symbol);
            return (it == m->end()) ? nullptr : it->second;
        }

        // DESERIALIZED: array + size encoded before array
        auto arr_bytes = static_cast<const uint8_t *>(trie_or_treeblock_ptr_);
        size_t n;
        std::memcpy(&n, arr_bytes - sizeof(size_t), sizeof(size_t));

        const IndexPtrPair *arr = reinterpret_cast<const IndexPtrPair *>(arr_bytes);

        // binary search using std::lower_bound
        auto it =
            std::lower_bound(arr, arr + n, symbol, [](const IndexPtrPair &p, const morton_t &key) {
                return p.index < key;
            });

        if (it == arr + n || it->index != symbol)
            return nullptr;

        return reinterpret_cast<trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *>(it->ptr);
    }

    inline void set_child(const morton_t &symbol, trie_node *node)
    {
        auto trie_ptr = (std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *)
            trie_or_treeblock_ptr_;
        (*trie_ptr)[symbol] = node;
    }

    inline ChildView<trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>> get_children() const
    {
        using Node = trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>;
        ChildView<Node> v;

        if (!DESERIALIZED_MDTRIE) {
            v.backend = ChildView<Node>::iterator::Backend::MAP;
            v.ptr = trie_or_treeblock_ptr_;
            return v;
        }

        // DESERIALIZED_MDTRIE: count stored 8 bytes before array
        const uint8_t *arr_bytes = static_cast<const uint8_t *>(trie_or_treeblock_ptr_);

        size_t count;
        std::memcpy(&count, arr_bytes - sizeof(size_t), sizeof(size_t));

        v.backend = ChildView<Node>::iterator::Backend::ARRAY;
        v.ptr = arr_bytes;
        v.count = count;

        return v;
    }

    // Only use this when you're sure this trie_node doesn't point directly to
    // a treeblock. Should really use a union here instead of a `void *`...
    void set_children(std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *map)
    {
        trie_or_treeblock_ptr_ = map;
    }

    inline tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *get_treeblock() const
    {
        return (tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)trie_or_treeblock_ptr_;
    }

    inline void set_treeblock(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *block)
    {
        trie_or_treeblock_ptr_ = block;
    }

    /// @brief Write a the MDTrie into a file.
    ///
    /// @param file
    /// @param level
    /// @param node_offset the offset to file at which the node should be
    ///                    inserted
    /// @param temp_node a previously wiped out trie_node<DIMENSION> that could
    ///                  be reused, it is this function's responsibility to free
    void serialize(FILE *file, trie_level_t level, uint64_t node_offset,
                   trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *temp_node)
    {

        // ensure we have allocated space for this node
        assert(current_offset > node_offset);

        // each node should only be serialized once
        assert(pointers_to_offsets_map.find((uint64_t)this) == pointers_to_offsets_map.end());

        // the location of insertion is known at this point, store (this ->
        // offset) in the map and update the offset
        pointers_to_offsets_map.insert({(uint64_t)this, node_offset});

        // again, reuse the buffer for easy modification of the node copy
        memcpy(temp_node, this, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));

        // no more parent_trie_node_

        // populate temp_node for insertion, but trie_node's nature differs with
        // level:
        //      1) leaf -> tree_block
        //      2) internal -> list of trie_node<DIMENSION> *
        if (level == MAX_TRIE_HASHMAP_DEPTH) {
            // at this point, the trie_node is probably at the end of the insertion queue, meaning
            // it can be inserted right away
            assert((current_offset - sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>)) ==
                   node_offset);

            // this is a leaf node and should have a tree_block that needs
            // serializing
            assert(this->get_treeblock());
            // this block should not have been allocated yet
            assert(pointers_to_offsets_map.find((uint64_t)(this->get_treeblock())) ==
                   pointers_to_offsets_map.end());

            // we need to create the block on file, and
            // `current_offset` is the next writing position
            temp_node->trie_or_treeblock_ptr_ = (void *)current_offset;

            // node has all the necessary info, write the node at the position `node_offset`
            if ((uint64_t)ftell(file) != node_offset) {
                fseek(file, (long)node_offset, SEEK_SET);
                fwrite(temp_node, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);
                fseek(file, 0, SEEK_END);
            } else
                fwrite(temp_node, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);

            free(temp_node);

            // write empty space for treeblock
            tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *wipe_out_treeblock =
                (tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                    1, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>));
            fwrite(wipe_out_treeblock, sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1,
                   file);
            uint64_t treeblock_offset = current_offset;
            current_offset += sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>);

            this->get_treeblock()->serialize(file, treeblock_offset, wipe_out_treeblock, level);
        } else {
            // this is not a leafnode, but a map of
            //      trie_node<DIMENSION> *
            // of the map's size

            auto *m = (std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *)this
                          ->trie_or_treeblock_ptr_;
            size_t map_size = m->size();

            // write the size of the map first
            fwrite(&map_size, sizeof(size_t), 1, file);
            current_offset += sizeof(size_t);

            // create the big chunk
            IndexPtrPair *temp_children = (IndexPtrPair *)calloc(map_size, sizeof(IndexPtrPair));

            // write to file
            fwrite(temp_children, sizeof(IndexPtrPair), map_size, file);
            uint64_t temp_children_offset = current_offset;
            assert(pointers_to_offsets_map.find((uint64_t)temp_children) ==
                   pointers_to_offsets_map.end());
            pointers_to_offsets_map.insert(
                {(uint64_t)this->trie_or_treeblock_ptr_, temp_children_offset});
            current_offset += sizeof(IndexPtrPair) * (map_size);

            // create dummy node to write empty space
            trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *wipe_out_trie_node =
                (trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                    1, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>));

            uint64_t iteration = 0;

            // iterate the entire map
            for (const auto &pair : *m) {
                morton_t i = pair.first; // key

                // the child node should not have been created yet
                assert(pointers_to_offsets_map.find((uint64_t)(this->get_child(i))) ==
                       pointers_to_offsets_map.end());

                temp_children[iteration].index = i;
                temp_children[iteration].ptr = current_offset;

                uint64_t child_offset = current_offset;

                fwrite(wipe_out_trie_node, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1,
                       file);
                current_offset += sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>);

                pair.second->serialize(file, level + 1, child_offset,
                                       (trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)calloc(
                                           1, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>)));

                iteration++;
            }

            free(wipe_out_trie_node);

            // write the big chunk of list of trie_node<DIMENSION> *'s at `temp_children_offset`
            if ((uint64_t)ftell(file) != temp_children_offset) {
                fseek(file, (long)temp_children_offset, SEEK_SET);
                fwrite(temp_children, sizeof(IndexPtrPair), map_size, file);
                fseek(file, 0, SEEK_END);
            } else
                fwrite(temp_children, sizeof(IndexPtrPair), map_size, file);

            free(temp_children);

            // populate the trie_node with the list of trie_node<DIMENSION> *
            temp_node->trie_or_treeblock_ptr_ = (void *)temp_children_offset;

            // write the node at the position node_offset
            if ((uint64_t)ftell(file) != node_offset) {
                fseek(file, (long)node_offset, SEEK_SET);
                fwrite(temp_node, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);
                fseek(file, 0, SEEK_END);
            } else
                fwrite(temp_node, sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>), 1, file);

            free(temp_node);
        }
    }

    void deserialize(trie_level_t level, uint64_t base_addr)
    {
        // no more parent_trie_node_
        if (trie_or_treeblock_ptr_ != NULL) {
            // update the pointer to the correct location
            trie_or_treeblock_ptr_ = (void *)(base_addr + (uint64_t)trie_or_treeblock_ptr_);

            if (level == MAX_TRIE_HASHMAP_DEPTH) {
                // if leaf, then should be a block
                get_treeblock()->deserialize(base_addr);
            } else {
                auto [ptr, count] = children_span_deserialized(trie_or_treeblock_ptr_);
                for (uint64_t i = 0; i < count; i++) {
                    ptr[i].ptr = (uint64_t)(base_addr + ptr[i].ptr);
                    ((trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *)(ptr[i].ptr))
                        ->deserialize(level + 1, base_addr);
                }
            }
        }
    }

    void update_size_recur(trie_level_t level, uint64_t &size)
    {
        if (level == MAX_TRIE_HASHMAP_DEPTH) {
            size += sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
            this->get_treeblock()->update_size_recur(level, size);
        } else if (!DESERIALIZED_MDTRIE) {
            auto *m = (std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *)
                trie_or_treeblock_ptr_;
            size_t map_size = m->size();
            size += sizeof(size_t);
            size += sizeof(IndexPtrPair) * map_size;
            for (const auto &pair : *m) {
                size += sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
                pair.second->update_size_recur(level + 1, size);
            }
        } else {
            auto [ptr, count] = children_span_deserialized(trie_or_treeblock_ptr_);
            size += sizeof(size_t);
            size += sizeof(IndexPtrPair) * count;
            for (size_t i = 0; i < count; i++) {
                size += sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
                auto *child =
                    reinterpret_cast<trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *>(ptr[i].ptr);
                child->update_size_recur(level + 1, size);
            }
        }
    }

    /// @brief Track storage per trie level for this node and children.
    void get_size_per_level_recur(trie_level_t level)
    {
        if (level == MAX_TRIE_HASHMAP_DEPTH) {
            // Treeblock struct is metadata at hosting level
            layer_stats[level].metadata_bytes +=
                sizeof(tree_block<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
            get_treeblock()->get_size_per_level_recur(level);
        } else if (!DESERIALIZED_MDTRIE) {
            auto *m = (std::map<morton_t, trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *> *)
                trie_or_treeblock_ptr_;
            size_t count = m->size();
            layer_stats[level].num_entries += count;
            layer_stats[level].entry_bytes += sizeof(size_t) + sizeof(IndexPtrPair) * count;
            for (const auto &pair : *m) {
                layer_stats[level].metadata_bytes +=
                    sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
                pair.second->get_size_per_level_recur(level + 1);
            }
        } else {
            auto [ptr, count] = children_span_deserialized(trie_or_treeblock_ptr_);
            layer_stats[level].num_entries += count;
            layer_stats[level].entry_bytes += sizeof(size_t) + sizeof(IndexPtrPair) * count;
            for (size_t i = 0; i < count; i++) {
                layer_stats[level].metadata_bytes +=
                    sizeof(trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION>);
                auto *child =
                    reinterpret_cast<trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *>(ptr[i].ptr);
                child->get_size_per_level_recur(level + 1);
            }
        }
    }

    /// @brief Count the number of points (leaf nodes) reachable from this trie_node.
    uint64_t count_points_recur(trie_level_t level)
    {
        if (level == MAX_TRIE_HASHMAP_DEPTH) {
            // At treeblock level, delegate to treeblock
            return this->get_treeblock()->count_points_recur(level);
        } else {
            // Recursively count from all children
            uint64_t count = 0;
            auto [ptr, child_count] = children_span_deserialized(trie_or_treeblock_ptr_);
            for (size_t i = 0; i < child_count; i++) {
                auto *child =
                    reinterpret_cast<trie_node<H_LEVEL, CHUNK_WIDTH, DIMENSION> *>(ptr[i].ptr);
                count += child->count_points_recur(level + 1);
            }
            return count;
        }
    }

    inline std::pair<IndexPtrPair *, size_t> children_span_deserialized(void *addr)
    {
        uint8_t *arr = static_cast<uint8_t *>(addr);

        size_t count;
        std::memcpy(&count, arr - sizeof(size_t), sizeof(size_t));

        IndexPtrPair *ptr = reinterpret_cast<IndexPtrPair *>(arr);
        return {ptr, count};
    }

private:
    void *trie_or_treeblock_ptr_ = NULL;
    // in DESERIALIZED_MDTRIE mode, this is a sorted array
    // not in DESERIALIZED_MDTRIE mode, this is a hashmap
};

#endif // MD_TRIE_TRIE_NODE_H
