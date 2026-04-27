# DynamicDawg architecture

Index: [AGENTS.md](../AGENTS.md)

`DynamicDawg` is the variable-width grouped DAWG in [`src/dynamic_dawg.h`](../src/dynamic_dawg.h).

Its goal is simple:

> keep the same Morton-key indexing idea as `CompactDawg`, but allow edge widths to vary by depth according to a runtime plan.

## Dependency note

Like `CompactDawg`, `DynamicDawg` uses the vendored auxiliary library under [`lib/ds-lib/`](../lib/ds-lib), especially [`bit_vector.h`](../lib/ds-lib/bit_vector.h), for its packed runtime storage. Upstream source: `https://github.com/anuragkh/ds-lib`

## Why it exists

`CompactDawg<GROUP_BITS>` uses one width for the entire key. That is easy to reason about, but it can waste depth in low-entropy regions and waste branching budget in high-entropy regions.

DynamicDawg introduces a `SegmentPlan` so the build can say:

- use a wide segment here
- use a narrow segment there
- then widen again later if the tail becomes redundant

## Main pieces

- [`src/dynamic_dawg.h`](../src/dynamic_dawg.h): variable-width DAWG implementation
- [`src/dawg_segmentation.h`](../src/dawg_segmentation.h): plan construction utilities
- [`tests/test_dynamic_dawg.cpp`](../tests/test_dynamic_dawg.cpp): correctness and cross-validation coverage
- [`tools/bench_dynamic_dawg_sweep.cpp`](../tools/bench_dynamic_dawg_sweep.cpp): benchmark driver

## Segment plans

A `dawg_seg::SegmentPlan` defines the width consumed at each logical depth.

It typically stores:

- `widths`: width of each segment
- `bit_offsets`: starting bit position of each segment

Every key in one DynamicDawg build uses the same plan. The plan is runtime-configurable, but it does not adapt per key after the build starts.

## How plans are created

The segmentation helpers support both hand-authored and data-driven plans:

- `from_widths(...)`: explicit offline plan
- `greedy(...)`: width choice driven by local saturation
- `greedy_cost_aware(...)`: width choice driven by a simple storage-oriented heuristic
- min-width variants: cost-aware planners that avoid very tiny segments
- `phase_aware(...)`: fixed profile shaped by the common low-entropy / high-entropy / low-entropy pattern

These planners are inputs to the build, not alternative DAWG structures.

## Build pipeline

1. compute or load a `SegmentPlan`
2. split each Morton key according to that plan
3. run Daciuk-style incremental minimization over the resulting variable-width segments
4. pack the minimized graph into bit-vectors
5. optionally run path compression

Compared with fixed-width CompactDawg, the packed layout needs more label metadata because labels are no longer uniform.

## Query support

Current supported query families:

- `Contains(key)`
- `LexicographicSearch(...)`

The code can also path-compress variable-width chains.

## Current limitation

`SpatialRangeSearch` for DynamicDawg is still deferred. The hard part is translating variable-width segment positions back into the Morton-aware spatial pruning logic used by MDTrie and CompactDawg.

Treat DynamicDawg as an experimental storage/indexing path until that spatial story is complete.

## How to work on it safely

When editing DynamicDawg:

1. keep the fixed-width CompactDawg path as the semantic reference
2. validate that a uniform `SegmentPlan` behaves like the equivalent fixed-width grouping
3. keep planner logic separate from core build correctness
4. update metric docs if benchmark output columns change
