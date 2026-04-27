# CompactDawg

`CompactDawg` was developed as a senior thesis project for the Computer Science major at Yale College. It investigates DAWG-style indexing over Morton-encoded high-dimensional vectors.


The repository contains two main indexes:

- `CompactDawg`: fixed-width grouped DAWG
- `DynamicDawg`: variable-width grouped DAWG driven by runtime segment plans

It also includes benchmarks, plotting scripts, range-query tooling, and the
MDTrie-derived support code needed for spatial validation.

Please contact [Wesley Andrade](mailto:wesleyantonioaguiar@gmail.com) if you have any questions!

## Repository Layout

- `src/`: core DAWG implementations
- `src/mdtrie/`: Morton/data-point helpers and MDTrie reference code kept for spatial validation
- `tools/`: benchmark and analysis executables
- `tests/`: correctness tests
- `scripts/`: plotting and workflow helpers
- `docs/`: architecture notes and workflow docs
- `lib/`: third-party dependencies used by the current code

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Data

The benchmark workflows expect untracked embedding binaries under:

```text
data/embeddings/qwen3-embedding-0.6b/msmarco_v2/
```

Some range-query tooling also expects the root-level fixture:

```text
./1024d-uniq-100k.bin
```

Dataset naming and binary layout are documented in
[docs/embeddings-datasets.md](docs/embeddings-datasets.md).

## Start Here

- [docs/compact-dawg-architecture.md](docs/compact-dawg-architecture.md)
- [docs/dynamic-dawg-architecture.md](docs/dynamic-dawg-architecture.md)
- [docs/dawg-benchmarks.md](docs/dawg-benchmarks.md)
- [docs/benchmarks-metrics.md](docs/benchmarks-metrics.md)
- [docs/compact-dawg-baselines.md](docs/compact-dawg-baselines.md)

## Agent Instructions

- [AGENTS.md](AGENTS.md)
- [CLAUDE.md](CLAUDE.md)
- [GEMNI.md](GEMNI.md)
