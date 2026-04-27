# External libraries and baselines

Index: [AGENTS.md](../AGENTS.md)

---

## 3. External Libraries & Baselines

### `dawgdic`

[dawgdic](https://github.com/s-yata/dawgdic) is a well-known, highly optimized C++ library for building DAWGs. We use it as our primary baseline for storage size and build time.

- **How it works:** It builds a standard DAWG using a `DawgBuilder` (which hashes and merges nodes), then compacts it into a double-array trie structure (`DictionaryBuilder`).
- **Limitations for our use case:**
  - It operates on 8-bit `char` alphabets. It cannot group bits arbitrarily (e.g., `GROUP_BITS=32`).
  - It uses 32-bit integers for state indices. When storing 100k+ high-dimensional Morton codes, the number of states easily exceeds $2^{32} - 1$, causing **silent integer overflow and data corruption** in `dawgdic`.
  - It does not support spatial range queries.

### `MDTrie` Reference

The core `MDTrie` implementation (`src/mdtrie/trie.h`, `src/mdtrie/tree_block.h`) serves as our mathematical reference for spatial queries. `CompactDawg::SpatialRangeSearch` is directly modeled after `MDTrie::range_search_trie`, utilizing the same `bound_magic` bitwise masking and `shrink_query_bounds` logic.

### `ds-lib`

The repository also vendors [`lib/ds-lib/`](../lib/ds-lib), an auxiliary low-level data-structure library used by the DAWG implementations for packed bit-vector storage. In practice, the main touchpoint here is [`bit_vector.h`](../lib/ds-lib/bit_vector.h), which backs the packed arrays written by `CompactDawg` and `DynamicDawg`. Upstream source: `https://github.com/anuragkh/ds-lib`
