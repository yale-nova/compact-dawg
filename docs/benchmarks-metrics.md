# Benchmark metric glossary

Index: [AGENTS.md](../AGENTS.md) · Workflow guide: [dawg-benchmarks.md](dawg-benchmarks.md)

This file defines the benchmark fields that are meant to stay stable across tools and plots. It is a glossary, not a record of any specific run.

## Common sweep fields

These appear in the fixed-width sweep CSV or are derived directly from it.

- `dim`: embedding dimensionality
- `dtype`: coordinate type used by the benchmark input, typically `float32` or `float16`
- `n_keys`: requested key count for the run
- `n_unique_keys`: actual number of unique Morton keys after deduplication
- `group_bits`: fixed label width used by the build
- `method`: benchmarked variant such as `CD-32`, `PC-32`, or an optional baseline name
- `total_bytes`: final packed structure size
- `bytes_per_key`: `total_bytes / n_unique_keys`
- `edges`: final structural edge count
- `morton_encode_s`: time spent encoding vectors into Morton keys
- `sort_dedup_s`: time spent sorting and deduplicating encoded keys
- `insert_s`: time spent inserting keys into the builder
- `finish_s`: time spent finalizing and packing the structure
- `total_build_s`: `insert_s + finish_s`

### Derived storage terms

- `key_bytes`: raw Morton-key size in bytes for the active `(dim, dtype)`
- `normalized_bpk`: `bytes_per_key / key_bytes`

`normalized_bpk` is the main cross-dimension storage metric because it scales the packed index size against the size of the underlying key representation.

## Sharing metrics

These fields come from `TRACK_SHARING` instrumentation and the sharing benchmark.

- `finalize_calls`: number of finalized builder nodes in the equivalent grouped trie
- `memo_hits`: finalizations that reused an already-known suffix node
- `unique_nodes`: `finalize_calls - memo_hits`
- `trie_edges`: equivalent grouped-trie edge count before suffix sharing
- `dawg_edges`: final edge count after suffix sharing
- `sharing_ratio`: `memo_hits / finalize_calls`
- `node_reduction`: `unique_nodes / finalize_calls`
- `edge_saving_pct`: `1 - dawg_edges / trie_edges`
- `shared_nodes`: packed nodes with indegree greater than 1

### Depth-normalized sharing fields

- `depth`: grouped depth index in a particular run
- `normalized_depth`: `depth * group_bits / total_bits`
- `sharing_rate`: per-depth sharing ratio

`normalized_depth` allows contributors to compare where sharing appears along keys of very different total lengths.

## Suffix-collapse storage fields

Some `CD-*` sweep rows also include a before/after packed-trie estimate so storage plots can isolate the effect of suffix sharing.

- `trie_edges_before_suffix_collapse`: equivalent fixed-width trie edge count before suffix deduplication
- `dawg_edges_after_suffix_collapse`: final edge count after suffix deduplication
- `pre_suffix_total_bytes`: estimated packed fixed-width trie size before suffix sharing
- `pre_suffix_bytes_per_key`: `pre_suffix_total_bytes / n_unique_keys`
- `pre_suffix_normalized_bpk`: pre-sharing normalized bytes per key
- `post_suffix_normalized_bpk`: final normalized bytes per key after suffix sharing
- `suffix_collapse_saved_bytes_per_key`: pre-sharing minus post-sharing bytes per key
- `suffix_collapse_saved_normalized_bpk`: pre-sharing minus post-sharing normalized bytes per key
- `suffix_collapse_saving_pct`: fraction of estimated packed-trie bytes removed by suffix sharing

These fields are meant to isolate suffix sharing. They are not a measurement of temporary builder memory.

## DynamicDawg fields

The DynamicDawg sweep adds plan-level and comparator fields. The most important ones are:

- `segmentation_method`: planner or profile used to build the segment plan
- `plan_s`: time spent computing the plan
- `n_segments`: number of plan segments
- `avg_label_bits`: average packed edge-label width
- `label_metadata_share`: fraction of packed storage used by label metadata
- `width_histogram`: summary of segment widths chosen by the planner

Comparator-oriented fields may include:

- best fixed-width storage comparator
- best fixed-width build-time comparator
- storage delta versus the best comparator
- build-time delta versus the best comparator

When those columns change, update this glossary and the DynamicDawg workflow docs together.

## Interpretation guidelines

- Use `sharing_ratio` and `edge_saving_pct` for structural questions.
- Use `normalized_bpk` for cross-dimension storage comparisons.
- Use `total_build_s` for end-to-end build tradeoffs unless you explicitly want to separate insertion from finalization.
- Treat missing rows in plots as filtered or invalid configurations, not implicit zeros.
