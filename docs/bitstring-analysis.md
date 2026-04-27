# Bitstring analysis

Index: [AGENTS.md](../AGENTS.md) · Workflow examples: [dawg-benchmarks.md](dawg-benchmarks.md)

This document explains the standalone Morton-bitstring analysis used to reason about grouping choices before building a DAWG.

It is intentionally separate from `CompactDawg` and `DynamicDawg`. The goal is to understand the **input keys**, not the packed index.

## Why this tool exists

Fixed-width grouping forces one `GROUP_BITS` choice for the entire key. That is often a poor fit for Morton-encoded floating-point data, where different parts of the key can have very different entropy and branching behavior.

The bitstring analysis tools answer questions such as:

- where are the keys nearly identical?
- where do they branch rapidly?
- where might larger or smaller segment widths be reasonable?

Those answers motivate experiments in `dawg_segmentation.h` and `dynamic_dawg.h`.

## Core terms

- **Chunk**: an aligned bit slice of width `group_width`
- **Symbol**: the integer value stored in that chunk
- **Cardinality (`K`)**: number of distinct chunk symbols observed at one aligned position
- **Saturation (`rho`)**: `K / min(N, 2^g)`, where `g` is chunk width
- **Path-length proxy**: number of grouped steps induced by a fixed width

### Why both cardinality and saturation matter

`K / N` alone is not enough. For small widths, the alphabet can be tiny even when local branching is already maximal. Saturation fixes that by comparing observed symbols to the number of symbols that could possibly appear at that width.

In practice:

- use **cardinality** to see raw distinctness
- use **saturation** to judge whether a width is locally "full" or still compressible

## Typical qualitative pattern

For Morton-encoded floating-point embeddings, contributors often observe three broad regions:

1. a low-entropy prefix where many keys agree
2. a higher-entropy middle where keys diverge quickly
3. a lower-entropy tail where some redundancy returns

That pattern is the main reason variable-width grouping is interesting. The exact boundaries are dataset-dependent, so the docs do not hard-code percentages.

## What the tool writes

`analyze_bitstring_structure` writes:

1. a main CSV with aligned chunk cardinality statistics
2. optionally, a `_segmentation.csv` companion describing greedy segment choices

The plotting script can then render:

- per-dimension cardinality curves
- per-dimension saturation curves
- cross-dimension comparisons
- segmentation summaries

## How to use the output

Use bitstring analysis to guide:

- which `GROUP_BITS` values are worth sweeping
- whether a variable-width planner is justified
- where a planner should prefer wide versus narrow segments

Do not use it as a substitute for actual index measurements. Once a grouping idea looks promising, validate it with the DAWG benchmarks.

## Relationship to DynamicDawg

`dawg_segmentation.h` uses the same underlying intuition:

- wide segments are attractive in low-entropy regions because they reduce depth
- narrow segments are safer in high-entropy regions because they avoid exploding local branching

The analysis tool is therefore a design aid for segmentation logic, not a proof that a given planner will win on storage or latency.
