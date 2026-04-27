#pragma once

#include "compact_dawg.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Post-construction structural analysis of a CompactDawg.
// All functions operate on the packed bit-vector representation via public
// accessors (after Finish). The third template argument is TRACK_SHARING and
// does not affect the packed layout used here.

template <uint32_t GB, bool PC, bool TS = false>
std::vector<size_t> ComputeInDegreeHistogram(const CompactDawg<GB, PC, TS> &dawg)
{
    size_t total_edges = dawg.get_total_edges();
    if (total_edges == 0)
        return {};

    uint32_t terminal = (1u << dawg.get_offset_bits()) - 1;

    std::unordered_map<uint32_t, uint32_t> in_deg;
    for (size_t e = 0; e < total_edges; ++e) {
        uint32_t tgt = dawg.get_target(static_cast<uint32_t>(e));
        if (tgt != terminal)
            in_deg[tgt]++;
    }

    size_t max_deg = 0;
    for (auto &[_, d] : in_deg) {
        if (d > max_deg)
            max_deg = d;
    }

    // Build histogram: index = in-degree, value = number of nodes
    std::vector<size_t> hist(max_deg + 1, 0);
    for (auto &[_, d] : in_deg)
        hist[d]++;

    // Nodes not targeted by any edge have in-degree 0 (root, or unreachable).
    // We don't include them since we can't enumerate them from offsets alone.

    return hist;
}

template <uint32_t GB, bool PC, bool TS = false>
size_t CountSharedNodes(const CompactDawg<GB, PC, TS> &dawg)
{
    size_t total_edges = dawg.get_total_edges();
    if (total_edges == 0)
        return 0;

    uint32_t terminal = (1u << dawg.get_offset_bits()) - 1;

    std::unordered_map<uint32_t, uint32_t> in_deg;
    for (size_t e = 0; e < total_edges; ++e) {
        uint32_t tgt = dawg.get_target(static_cast<uint32_t>(e));
        if (tgt != terminal)
            in_deg[tgt]++;
    }

    size_t shared = 0;
    for (auto &[_, d] : in_deg) {
        if (d > 1)
            shared++;
    }
    return shared;
}
