# Repository guide

This repository contains the DAWG-based indexing work and experimental tooling for a senior thesis on high-dimensional Morton-key indexing.

## What is in this repo

1. `CompactDawg`
   A fixed-width grouped DAWG for sorted Morton-encoded bitstrings.
2. `DynamicDawg`
   A variable-width grouped DAWG driven by runtime `SegmentPlan`s.
3. Benchmark and plotting tools
   Storage, build-time, suffix-sharing, bitstring-analysis, and range-query workflows.
4. Spatial-query support code
   The Morton/data-point helpers and the MDTrie-derived reference path used by CompactDawg spatial validation, organized under `src/mdtrie/`.

This repo is still centered on the DAWG line, but it now carries the additional support code required by the full CompactDawg workflow.

## Where to start

| Task | Read first |
|------|------------|
| Understand the repo | [`README.md`](README.md) |
| Work on CompactDawg | [`docs/compact-dawg-architecture.md`](docs/compact-dawg-architecture.md) |
| Work on DynamicDawg | [`docs/dynamic-dawg-architecture.md`](docs/dynamic-dawg-architecture.md) |
| Run or extend benchmarks | [`docs/dawg-benchmarks.md`](docs/dawg-benchmarks.md) |
| Interpret CSV columns | [`docs/benchmarks-metrics.md`](docs/benchmarks-metrics.md) |
| Understand grouping analysis | [`docs/bitstring-analysis.md`](docs/bitstring-analysis.md) |
| Dataset naming/layout | [`docs/embeddings-datasets.md`](docs/embeddings-datasets.md) |
| Understand baselines and reference context | [`docs/compact-dawg-baselines.md`](docs/compact-dawg-baselines.md) |

## Key invariants

- Morton encoding is a shared contract across CompactDawg, DynamicDawg, and the benchmark tools.
- Query correctness matters more than a local storage win.
- `GROUP_BITS` changes traversal shape, storage layout, and benchmark behavior.
- Planner logic in `dawg_segmentation.h` should stay conceptually separate from DynamicDawg core correctness.
- `CompactDawg::SpatialRangeSearch` should stay aligned with the Morton-aware pruning logic used by the reference path.

## Tests

Main validation entry points:

- [`tests/test_compact_dawg_core.cpp`](tests/test_compact_dawg_core.cpp)
- [`tests/test_compact_dawg_spatial_1024d.cpp`](tests/test_compact_dawg_spatial_1024d.cpp)
- [`tests/test_dynamic_dawg.cpp`](tests/test_dynamic_dawg.cpp)
- [`tests/test_dawgdic_lexicographic.cpp`](tests/test_dawgdic_lexicographic.cpp)

If query semantics change, keep exact lookup, lexicographic search, and spatial/range-query coverage aligned with the implementation.

## Documentation rules

When behavior, CLI semantics, dataset naming assumptions, or CSV columns change, update the matching docs in the same change.

Prefer docs that capture:

- architecture
- invariants
- edit locations
- workflows
- metric meaning

Avoid docs that hard-code one-off benchmark results or report-specific conclusions.
