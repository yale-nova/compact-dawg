# DAWG benchmarks, tooling, and tests

Index: [AGENTS.md](../AGENTS.md)

See also:

- [benchmarks-metrics.md](benchmarks-metrics.md) for CSV field definitions
- [bitstring-analysis.md](bitstring-analysis.md) for the standalone segmentation analysis workflow
- [embeddings-datasets.md](embeddings-datasets.md) for dataset layout and naming

This document is a workflow guide. It explains which tool to use for each job and where its outputs go. For the full current CLI, prefer the tool's `--help` output over hard-coding every flag here.

## Tool map

### Storage and build sweeps

- [`tools/bench_dawg_storage.cpp`](../tools/bench_dawg_storage.cpp)
  Runs a single-file storage/build benchmark on Morton bitstring inputs that are already encoded.
- [`tools/bench_dawg_sweep.cpp`](../tools/bench_dawg_sweep.cpp)
  Runs fixed-width CompactDawg sweeps across dimensions, dataset sizes, dtypes, and `GROUP_BITS`.
- [`scripts/plot_dawg_sweep.py`](../scripts/plot_dawg_sweep.py)
  Turns sweep CSVs into plots and Markdown tables.

Use `bench_dawg_storage` when you already have a raw Morton-key binary and want a quick local comparison. Use the sweep workflow when you want the full dataset/dimension experiment matrix.

### Suffix-sharing analysis

- [`tools/bench_dawg_sharing.cpp`](../tools/bench_dawg_sharing.cpp)
  Builds `CompactDawg<GB, false, true>` and records structural sharing metrics.
- [`scripts/plot_dawg_sharing.py`](../scripts/plot_dawg_sharing.py)
  Plots aggregate sharing, depth profiles, per-depth DAWG edge counts, and indegree summaries.

Use this pair when the question is structural: how much sharing happened, and where in the key.

### Bitstring-only analysis

- [`tools/analyze_bitstring_structure.cpp`](../tools/analyze_bitstring_structure.cpp)
  Studies Morton bitstrings without building a DAWG.
- [`scripts/plot_bitstring_analysis.py`](../scripts/plot_bitstring_analysis.py)
  Visualizes cardinality, saturation, and optional segmentation summaries.

Use this pair when exploring whether variable-width grouping might help before you change the index itself.

### DynamicDawg experiments

- [`tools/bench_dynamic_dawg_sweep.cpp`](../tools/bench_dynamic_dawg_sweep.cpp)
  Evaluates variable-width segmentation plans against fixed-width baselines.
- [`scripts/plot_dynamic_dawg_sweep.py`](../scripts/plot_dynamic_dawg_sweep.py)
  Writes comparison plots and companion tables for DynamicDawg runs.

Use this path for experimentation with `dawg_segmentation.h` and `dynamic_dawg.h`.

### Range-query evaluation

- [`tools/gen_rq_queries_exact10_1024d.cpp`](../tools/gen_rq_queries_exact10_1024d.cpp)
  Generates aligned range-query fixtures.
- [`tools/bench_dawg_rq.cpp`](../tools/bench_dawg_rq.cpp)
  Benchmarks `SpatialRangeSearch`.
- [`tools/linear_scan_rq.cpp`](../tools/linear_scan_rq.cpp)
  Legacy linear-scan baseline helper for CSV-formatted range-query experiments.
- [`scripts/plot_rq_bench.py`](../scripts/plot_rq_bench.py)
  Plots range-query CSV output.
- [`scripts/run_rq_gen_and_bench.sh`](../scripts/run_rq_gen_and_bench.sh)
  Convenience wrapper around the generator and bench.

Use this path when changing spatial query semantics or evaluating query-time tradeoffs.

## Common workflows

### 1. Fixed-width storage sweep

Build the tool:

```bash
cmake --build build --target bench_dawg_sweep
```

Run a sweep against the embedding corpus:

```bash
./build/bench_dawg_sweep \
  --data-dir data/embeddings/qwen3-embedding-0.6b/msmarco_v2 \
  --output-csv results/dawg_sweep.csv
```

Plot the results:

```bash
python3 scripts/plot_dawg_sweep.py \
  -i results/dawg_sweep.csv \
  -o plots/sweep \
  --plot-type all \
  --export-tables
```

### 2. Sharing analysis

```bash
cmake --build build --target bench_dawg_sharing

./build/bench_dawg_sharing \
  --data-dir data/embeddings/qwen3-embedding-0.6b/msmarco_v2 \
  --output-dir results/sharing

python3 scripts/plot_dawg_sharing.py \
  --input-dir results/sharing \
  --output-dir plots/sharing
```

### 3. Bitstring analysis

```bash
cmake --build build --target analyze_bitstring_structure

./build/analyze_bitstring_structure \
  --input data/embeddings/qwen3-embedding-0.6b/msmarco_v2/msmarco_v2_corpus_1024d_float32.bin \
  --dims 1024 \
  --dtype float32 \
  --output results/bitstring_1024d.csv

python3 scripts/plot_bitstring_analysis.py \
  --input results/bitstring_1024d.csv \
  --outdir plots/bitstring_analysis
```

### 4. DynamicDawg sweep

```bash
cmake --build build --target bench_dynamic_dawg_sweep

./build/bench_dynamic_dawg_sweep \
  --data-dir data/embeddings/qwen3-embedding-0.6b/msmarco_v2 \
  --output-csv results/dynamic_dawg.csv

python3 scripts/plot_dynamic_dawg_sweep.py \
  -i results/dynamic_dawg.csv \
  -o plots/dynamic_dawg \
  --export-tables
```

### 5. Range-query pipeline

```bash
cmake --build build --target gen_rq_queries_exact10_1024d bench_dawg_rq

scripts/run_rq_gen_and_bench.sh --build-dir build
```

The default range-query workflow expects the root-level fixture `./1024d-uniq-100k.bin` together with generated bounds and ground-truth files under `tests/testdata/`.

## Output conventions

- `results/` is for generated CSVs and intermediate artifacts
- `plots/` is for generated figures and Markdown tables
- both directories are intentionally untracked by Git

If a tool changes its CSV columns or plot/table naming, update [benchmarks-metrics.md](benchmarks-metrics.md) and any doc that refers to the workflow.

### Plot styling for accessibility

The plotting scripts share [`scripts/_plot_style.py`](../scripts/_plot_style.py),
which exposes ordered marker, linestyle, and hatch cycles. Series in any new
plot must use a non-color cue in addition to color so the figures stay readable
without color (color-deficient viewers, B&W prints):

- **Line plots:** vary marker shape per series via `marker_for(idx)`. Use
  `markevery` to thin markers when a series has many points.
- **Multi-axis plots (e.g. dim x planner):** combine `marker_for(method_idx)`
  with `line_style_for(dim_idx)` so each axis is encoded redundantly.
- **Bar charts:** combine `hatch_for(idx)` with the bar color and add a thin
  black edge so the hatch is visible on top of the fill.

## Tests to run after code changes

Main correctness coverage for the DAWG work lives in:

- [`tests/test_compact_dawg_core.cpp`](../tests/test_compact_dawg_core.cpp)
- [`tests/test_compact_dawg_spatial_1024d.cpp`](../tests/test_compact_dawg_spatial_1024d.cpp)
- [`tests/test_dynamic_dawg.cpp`](../tests/test_dynamic_dawg.cpp)
- [`tests/test_dawgdic_lexicographic.cpp`](../tests/test_dawgdic_lexicographic.cpp)

If you modify spatial logic, favor the spatial test suite and any range-query fixture workflow that exercises the changed path.

## Contributor rules of thumb

- Keep generated artifacts out of Git.
- Prefer adding a clear CSV field or plot label over embedding conclusions in docs.
- If the question is about **structure**, use the sharing tool.
- If the question is about **storage/build tradeoffs**, use the sweep tool.
- If the question is about **whether variable-width grouping might help at all**, start with the bitstring analysis.
