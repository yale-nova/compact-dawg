# `CompactDawg` architecture

Index: [AGENTS.md](../AGENTS.md)

`CompactDawg` is the fixed-width DAWG implementation in [`src/compact_dawg.h`](../src/compact_dawg.h):

```cpp
template <uint32_t GROUP_BITS, bool PATH_COMPRESS = false, bool TRACK_SHARING = false>
class CompactDawg;
```

This document explains the stable design and the main edit points.

## Dependency note

The packed storage layer in `CompactDawg` relies on the vendored auxiliary library under [`lib/ds-lib/`](../lib/ds-lib), especially [`bit_vector.h`](../lib/ds-lib/bit_vector.h). That library provides the growable packed bit-vector used for `labels_`, `is_last_`, `offsets_`, and related metadata. Upstream source: `https://github.com/anuragkh/ds-lib`

## Template parameters

- `GROUP_BITS`: width of each logical edge label before path compression
- `PATH_COMPRESS`: whether fanout-1 / indegree-1 chains are merged during `Finish()`
- `TRACK_SHARING`: whether construction-time suffix-sharing instrumentation is collected

These controls are intentionally orthogonal:

- `GROUP_BITS` changes traversal granularity
- suffix sharing comes from DAWG minimization
- path compression is an extra post-build compaction step

## Build pipeline

### 1. Sorted insertion

Insertion assumes keys arrive in lexicographic order. The builder keeps an `active_path_` for the most recent key and uses Daciuk-style incremental minimization:

- shared prefixes stay on the active path
- when a new key diverges, the no-longer-mutable suffix of the previous key is finalized
- finalized nodes are memoized by structure so identical suffixes can be reused immediately

### 2. Finalization and suffix sharing

Suffix sharing is decided during node finalization, not only at the end of the build:

- `Insert()` finalizes diverging suffixes
- `Finish()` drains the remaining active path, including the root

When `TRACK_SHARING = true`, this stage also records the structural counters used by the sharing benchmarks.

### 3. Packed runtime layout

After all insertions, `Finish()` writes the runtime representation into bit-packed vectors.

Fixed-width mode stores:

- `labels_`: edge labels
- `is_last_`: one bit per edge marking node boundaries
- `offsets_`: packed target-node indices

The offset width is computed from the actual node/edge count instead of hard-coding 32-bit pointers.

## Path compression

When `PATH_COMPRESS = true`, `Finish()` runs an additional pass that:

1. computes node starts and degrees
2. keeps shared or branching nodes intact
3. collapses linear chains into longer labels
4. remaps targets into the compressed edge array

This adds variable-length label metadata:

- `label_offsets_`
- `label_lengths_`

Path compression changes the final edge layout, but it does not change key semantics.

## Query semantics

The implementation currently supports three query families:

1. `Contains(bitstring)`
2. `LexicographicSearch(start, end)`
3. `SpatialRangeSearch(start_point, end_point)`

Important contributor note: `SpatialRangeSearch` is not a plain lexicographic interval walk. It mirrors the Morton-aware pruning rules used by the MDTrie reference path.

## Public observability

When `TRACK_SHARING = true`, the builder exposes structural counters such as:

- `finalize_calls`
- `memo_hits`
- `unique_nodes`
- `trie_edges`
- `dawg_edges`

Regardless of instrumentation mode, the packed DAWG also exposes enough accessors for post-build structural analysis in [`src/dawg_sharing_analysis.h`](../src/dawg_sharing_analysis.h).

## Files to edit

- main implementation: [`src/compact_dawg.h`](../src/compact_dawg.h)
- post-build structural analysis: [`src/dawg_sharing_analysis.h`](../src/dawg_sharing_analysis.h)
- fixed-width tests: [`tests/test_compact_dawg_core.cpp`](../tests/test_compact_dawg_core.cpp)
- spatial validation: [`tests/test_compact_dawg_spatial_1024d.cpp`](../tests/test_compact_dawg_spatial_1024d.cpp)

## Safe editing checklist

When modifying CompactDawg:

1. preserve sorted-insertion assumptions, or document and test any change
2. keep exact lookup and spatial traversal semantics aligned with tests
3. treat path compression as a storage optimization, not a semantic change
4. update metric docs if instrumentation fields or CSV semantics change
