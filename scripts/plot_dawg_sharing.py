#!/usr/bin/env python3
"""Plot CompactDawg suffix sharing analysis results.

Metric definitions
------------------
finalize_calls
    Number of times Finalize() is called during DAWG construction — once for
    every non-empty node encountered while processing the active_path_.  This
    equals the total number of nodes in the equivalent uncompressed trie (minus
    leaves, which are represented as BUILD_TERMINAL_NODE and not finalized).

memo_hits
    Of those finalize_calls, the number where an identical node was already
    present in the memo table.  A memo hit means that suffix is *shared*: two
    or more keys map to the same DAWG sub-graph.  No new node is stored.

unique_nodes  (= finalize_calls - memo_hits)
    Distinct nodes actually inserted into the DAWG structure.

trie_edges
    Cumulative edge count across all Finalize() calls — edges the equivalent
    trie would have.

dawg_edges
    Edges in the packed DAWG after deduplication (len(temp_edges_) at Finish()).

sharing_ratio  = memo_hits / finalize_calls
    Fraction of node-finalization events that found an already-known suffix.
    This is the primary suffix sharing signal.

node_reduction  = unique_nodes / finalize_calls
    Fraction of trie nodes that survived deduplication (1.0 → no sharing).

edge_saving_pct  = 1 - dawg_edges / trie_edges
    Fraction of trie edges eliminated by DAWG sharing.

normalized_depth  = depth * group_bits / total_bits
    Depth level mapped to [0, 1] across all (dim, GB) configurations so that
    sharing-rate curves are comparable between 16-dim and 1024-dim keys.
    Computed by bench_dawg_sharing and stored in sharing_depth.csv.

shared_nodes
    Nodes with in-degree > 1 in the packed edge table — nodes reachable via
    multiple paths, i.e. concretely shared suffixes.

Usage:
    python scripts/plot_dawg_sharing.py --input-dir sharing_results/ --output-dir plots/sharing/
    python scripts/plot_dawg_sharing.py --input-dir sharing_results/ --plot-type sharing_ratio_vs_dim
    python scripts/plot_dawg_sharing.py ... --coord-bits 16   # fp16 keys (dim×16 must divide GB)
    python scripts/plot_dawg_sharing.py ... --min-key-levels 4  # stricter omit for wide GB / low dim
    python scripts/plot_dawg_sharing.py --input-dir sharing_results/ --plot-type all --show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import hatch_for, marker_for  # noqa: E402

PLOT_TYPES = [
    "sharing_ratio_vs_dim",
    "sharing_ratio_vs_n",
    "finalize_breakdown",
    "edge_saving_heatmap",
    "depth_profile",
    "depth_edges",
    "depth_heatmap",
    "indegree_histogram",
]


def load_summary(input_dir):
    path = os.path.join(input_dir, "sharing_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_depth(input_dir):
    path = os.path.join(input_dir, "sharing_depth.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_indegree(input_dir):
    path = os.path.join(input_dir, "sharing_indegree.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def filter_df(df, filter_dims, filter_n, filter_gb):
    if filter_dims:
        df = df[df["dim"].isin(filter_dims)]
    if filter_n:
        df = df[df["n_keys"].isin(filter_n)]
    if filter_gb:
        df = df[df["group_bits"].isin(filter_gb)]
    return df


def _apply_light_grid(ax, *, axis="both"):
    """Use a quiet major grid so plot lines stay visually dominant."""
    ax.grid(
        True,
        which="major",
        axis=axis,
        linestyle="-",
        linewidth=0.5,
        alpha=0.18,
        color="0.45",
    )
    ax.grid(False, which="minor", axis=axis)


def _mask_valid_dim_group_bits(df, coord_bits):
    """True where GROUP_BITS divides dim * coord_bits (matches bench_dawg_sharing)."""
    if df.empty:
        return np.zeros(len(df), dtype=bool)
    cb = int(coord_bits) if coord_bits and int(coord_bits) > 0 else 32
    dim = df["dim"].to_numpy(dtype=np.int64, copy=False)
    gb = df["group_bits"].to_numpy(dtype=np.int64, copy=False)
    return (dim * cb % gb) == 0


def _mask_min_key_symbol_levels(df, coord_bits, min_levels):
    """True where (dim * coord_bits) / group_bits >= min_levels (meaningful trie depth).

    When the Morton key spans only one or two GROUP_BITS-wide symbols, runs are
    degenerate and ``sharing_ratio`` is often 0% for structural reasons — omit
    those points rather than plotting them as real zeros (matches post-fix
    ``bench_dawg_sharing`` skipping the same configs).
    """
    if df.empty:
        return np.zeros(len(df), dtype=bool)
    cb = int(coord_bits) if coord_bits and int(coord_bits) > 0 else 32
    ml = int(min_levels) if min_levels and int(min_levels) > 0 else 3
    dim = df["dim"].to_numpy(dtype=np.int64, copy=False)
    gb = df["group_bits"].to_numpy(dtype=np.int64, copy=False)
    total_bits = dim * cb
    levels = total_bits // gb
    return levels >= ml


def plot_sharing_ratio_vs_dim(summary, _depth, _indeg, output_dir, show, **kwargs):
    """Sharing ratio (memo_hits / finalize_calls) vs dimension.

    One line per GROUP_BITS value, using the largest N in the dataset.
    Lower dimensionality → higher sharing because Morton-encoded suffixes
    have lower per-bit entropy and collide more often in the memo table.

    Rows where ``dim * coord_bits`` is not divisible by ``group_bits`` are
    dropped (bench never runs those configurations); optional ``coord_bits``
    (default 32) matches fp32 keys. Rows with fewer than ``min_key_levels``
    Morton symbols along the key (``(dim * coord_bits) / group_bits``) are
    dropped so degenerate wide-GB runs are not drawn as false zeros.
    NaN / non-finite ``sharing_ratio`` or zero ``finalize_calls`` rows are
    also omitted.
    """
    if summary.empty:
        return
    max_n = summary["n_unique_keys"].max()
    sub = summary[summary["n_unique_keys"] == max_n].copy()
    if sub.empty:
        return

    coord_bits = int(kwargs.get("coord_bits", 32) or 32)
    min_key_levels = int(kwargs.get("min_key_levels", 3) or 3)
    mask = _mask_valid_dim_group_bits(sub, coord_bits)
    mask &= _mask_min_key_symbol_levels(sub, coord_bits, min_key_levels)
    sr = pd.to_numeric(sub["sharing_ratio"], errors="coerce")
    mask &= np.isfinite(sr.to_numpy(dtype=float, copy=False))
    if "finalize_calls" in sub.columns:
        fc = pd.to_numeric(sub["finalize_calls"], errors="coerce")
        fcv = fc.to_numpy(dtype=float, copy=False)
        mask &= np.isfinite(fcv) & (fcv > 0)
    sub = sub.loc[mask]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, gb in enumerate(sorted(sub["group_bits"].unique())):
        g = sub[sub["group_bits"] == gb].sort_values("dim")
        ax.plot(
            g["dim"],
            g["sharing_ratio"] * 100,
            marker=marker_for(i),
            markersize=6,
            linewidth=1.4,
            label=f"GB={gb}",
        )

    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Sharing Ratio = memo_hits / finalize_calls (%)")
    ax.set_title(f"Suffix Sharing Ratio vs Dimension (N={max_n:,})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["dim"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(title="GROUP_BITS")
    _apply_light_grid(ax)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "sharing_ratio_vs_dim.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_sharing_ratio_vs_n(summary, _depth, _indeg, output_dir, show, **kwargs):
    """Sharing ratio (memo_hits / finalize_calls) vs dataset size N.

    One subplot per dimension, one line per GROUP_BITS.  Sharing ratio
    typically increases with N because a larger dataset provides more
    opportunities for suffix collisions in the memo table.
    """
    if summary.empty:
        return
    n_values = sorted(summary["n_unique_keys"].dropna().unique())
    if len(n_values) < 2:
        only = _comma_fmt(n_values[0]) if n_values else "none"
        print(
            "Skipping sharing_ratio_vs_n: need at least two N values after "
            f"filtering; found {only}.",
            file=sys.stderr,
        )
        return

    dims = sorted(summary["dim"].unique())
    n_dims = len(dims)
    cols = min(n_dims, 3)
    rows = (n_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, dim in enumerate(dims):
        ax = axes[idx // cols][idx % cols]
        sub = summary[summary["dim"] == dim]
        for i, gb in enumerate(sorted(sub["group_bits"].unique())):
            g = sub[sub["group_bits"] == gb].sort_values("n_unique_keys")
            ax.plot(
                g["n_unique_keys"],
                g["sharing_ratio"] * 100,
                marker=marker_for(i),
                markersize=5,
                linewidth=1.3,
                label=f"GB={gb}",
            )
        ax.set_xlabel("N (unique keys inserted)")
        ax.set_ylabel("memo_hits / finalize_calls (%)")
        ax.set_title(f"dim={dim}")
        ax.set_xscale("log")
        ax.legend(fontsize=7, title="GB")
        _apply_light_grid(ax)
        ax.set_ylim(bottom=0)

    for idx in range(n_dims, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Suffix Sharing Ratio (memo_hits / finalize_calls) vs Dataset Size",
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "sharing_ratio_vs_n.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_finalize_breakdown(summary, _depth, _indeg, output_dir, show, **kwargs):
    """Stacked bar: finalize_calls broken into memo_hits vs unique_nodes.

    finalize_calls = unique_nodes + memo_hits.
    - unique_nodes: new nodes written into the DAWG.
    - memo_hits:    nodes reused from the memo table (suffix sharing events).

    One chart per GROUP_BITS value, at the largest N.
    """
    if summary.empty:
        return
    max_n = summary["n_unique_keys"].max()
    sub = summary[summary["n_unique_keys"] == max_n].copy()
    if sub.empty:
        return

    for gb in sorted(sub["group_bits"].unique()):
        g = sub[sub["group_bits"] == gb].sort_values("dim")
        if g.empty:
            continue

        dims = list(g["dim"])
        x = np.arange(len(dims))
        unique = g["unique_nodes"].values
        hits = g["memo_hits"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x,
            unique,
            label="unique_nodes (new)",
            color="steelblue",
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch_for(0),
        )
        ax.bar(
            x,
            hits,
            bottom=unique,
            label="memo_hits (shared)",
            color="coral",
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch_for(1),
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("finalize_calls  (unique_nodes + memo_hits)")
        ax.set_title(
            f"Finalize Breakdown: unique vs shared nodes\n"
            f"GB={gb}, N={max_n:,}  —  sharing_ratio = memo_hits / finalize_calls"
        )
        ax.legend()
        _apply_light_grid(ax, axis="y")

        fig.tight_layout()
        path = os.path.join(output_dir, f"finalize_breakdown_gb{gb}.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
        if show:
            plt.show()
        plt.close(fig)


def plot_edge_saving_heatmap(summary, _depth, _indeg, output_dir, show, **kwargs):
    """Heatmap of edge_saving_pct = 1 - dawg_edges / trie_edges.

    Rows = Dimensions, columns = GROUP_BITS, at the largest N.
    Each cell shows what fraction of trie edges were eliminated by DAWG
    suffix sharing.  This is a structural consequence of sharing_ratio.
    """
    if summary.empty:
        return
    max_n = summary["n_unique_keys"].max()
    sub = summary[summary["n_unique_keys"] == max_n]
    if sub.empty:
        return

    pivot = sub.pivot_table(values="edge_saving_pct", index="dim",
                            columns="group_bits", aggfunc="mean")
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values * 100, aspect="auto",
                   cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index])
    ax.set_xlabel("GROUP_BITS")
    ax.set_ylabel("Dimensions")
    ax.set_title(
        f"Edge Saving % = 1 - dawg_edges / trie_edges  (N={max_n:,})\n"
        "Fraction of trie edges eliminated by suffix sharing"
    )

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=8)

    fig.colorbar(im, ax=ax, label="Edge saving (%)")
    fig.tight_layout()
    path = os.path.join(output_dir, "edge_saving_heatmap.png")
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_depth_profile(_summary, depth, _indeg, output_dir, show, **kwargs):
    """Per-dim line plot of sharing rate vs normalized_depth.

    Uses the precomputed `normalized_depth` column from sharing_depth.csv:
        normalized_depth = depth * group_bits / total_bits
    This maps all (dim, GB) configurations onto [0, 1] uniformly so that
    curves from e.g. 16-dim (512-bit keys) and 1024-dim (32768-bit keys)
    are aligned.  A value of 0.0 = root, 1.0 = deepest level.

    sharing_rate at depth d = memo_hits[d] / finalize_calls[d]:
    the fraction of Finalize() calls at that depth that found an already-known
    node in the memo table.
    """
    if depth.empty:
        return

    max_n = depth["n_keys"].max()
    sub = depth[depth["n_keys"] == max_n]
    if sub.empty:
        return

    # normalized_depth must be present (written by bench_dawg_sharing).
    if "normalized_depth" not in sub.columns:
        print("Warning: normalized_depth column missing from depth CSV — skipping depth_profile")
        return

    gb_values = sorted(sub["group_bits"].unique())
    for gb in gb_values:
        gs = sub[sub["group_bits"] == gb]
        dims = sorted(gs["dim"].unique())

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, dim in enumerate(dims):
            dd = gs[(gs["dim"] == dim) & (gs["finalize_calls"] > 0)].sort_values("normalized_depth")
            if dd.empty:
                continue
            n_points = len(dd)
            markevery = max(1, n_points // 18) if n_points > 30 else 1
            ax.plot(
                dd["normalized_depth"],
                dd["sharing_rate"] * 100,
                marker=marker_for(i),
                markersize=5,
                markevery=markevery,
                linewidth=1.2,
                label=f"dim={dim}",
                alpha=0.85,
            )

        ax.set_xlabel(
            "normalized_depth = depth × group_bits / total_bits\n"
            "(0 = root, 1 = deepest level; comparable across all dim/GB)"
        )
        ax.set_ylabel("sharing_rate = memo_hits / finalize_calls at depth (%)")
        ax.set_title(
            f"Per-Depth Sharing Rate vs Normalized Depth\n"
            f"GB={gb}, N={max_n:,}  —  curves aligned across dimensionalities"
        )
        ax.legend(fontsize=7, ncol=2, title="dim")
        _apply_light_grid(ax)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 1)

        fig.tight_layout()
        path = os.path.join(output_dir, f"depth_profile_gb{gb}.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
        if show:
            plt.show()
        plt.close(fig)


def plot_depth_edges(_summary, depth, _indeg, output_dir, show, **kwargs):
    """Per-dim line plot of DAWG edges vs normalized_depth.

    `dawg_edges` counts edges emitted for newly materialized DAWG nodes at each
    grouped depth. Memo hits are not counted here because they reuse an existing
    suffix node instead of adding new packed edges.
    """
    if depth.empty:
        return

    max_n = depth["n_keys"].max()
    sub = depth[depth["n_keys"] == max_n]
    if sub.empty:
        return

    need = {"normalized_depth", "dawg_edges"}
    if not need.issubset(sub.columns):
        print("Warning: sharing_depth.csv missing dawg_edges — rerun bench_dawg_sharing for depth_edges")
        return

    gb_values = sorted(sub["group_bits"].unique())
    for gb in gb_values:
        gs = sub[sub["group_bits"] == gb]
        dims = sorted(gs["dim"].unique())

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, dim in enumerate(dims):
            dd = gs[gs["dim"] == dim].sort_values("normalized_depth")
            if dd.empty:
                continue
            n_points = len(dd)
            markevery = max(1, n_points // 18) if n_points > 30 else 1
            ax.plot(
                dd["normalized_depth"],
                dd["dawg_edges"],
                marker=marker_for(i),
                markersize=5,
                markevery=markevery,
                linewidth=1.2,
                label=f"dim={dim}",
                alpha=0.85,
            )

        ax.set_xlabel(
            "normalized_depth = depth × group_bits / total_bits\n"
            "(0 = root, 1 = deepest level; comparable across all dim/GB)"
        )
        ax.set_ylabel("DAWG edges added at depth")
        ax.set_title(f"DAWG Edge Count vs Normalized Depth\nGB={gb}, N={max_n:,}")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2, title="dim")
        _apply_light_grid(ax)
        ax.set_xlim(0, 1)

        fig.tight_layout()
        path = os.path.join(output_dir, f"depth_edges_gb{gb}.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
        if show:
            plt.show()
        plt.close(fig)


def plot_depth_heatmap(_summary, depth, _indeg, output_dir, show, **kwargs):
    """Heatmap: normalized_depth (rows) vs dimension (columns), color = sharing_rate.

    Uses the precomputed `normalized_depth` column:
        normalized_depth = depth * group_bits / total_bits  ∈ [0, 1]
    Rows are binned into n_depth_bins buckets so the image is a fixed size
    regardless of the absolute key length.  Within each bin, finalize_calls
    and memo_hits are summed before computing sharing_rate, so the aggregated
    rate is properly weighted by node count.
    """
    if depth.empty:
        return

    max_n = depth["n_keys"].max()
    sub = depth[depth["n_keys"] == max_n]
    if sub.empty:
        return

    if "normalized_depth" not in sub.columns:
        print("Warning: normalized_depth column missing — skipping depth_heatmap")
        return

    gb_values = sorted(sub["group_bits"].unique())
    n_depth_bins = 20

    for gb in gb_values:
        gs = sub[sub["group_bits"] == gb]
        dims = sorted(gs["dim"].unique())

        all_rows = []
        for dim in dims:
            dd = gs[(gs["dim"] == dim) & (gs["finalize_calls"] > 0)].copy()
            if dd.empty:
                continue
            # Bin by normalized_depth (already in [0,1]) — no recomputation needed.
            dd["depth_bin"] = (
                dd["normalized_depth"] * (n_depth_bins - 1)
            ).round().astype(int)
            binned = dd.groupby("depth_bin").agg(
                total_finalize=("finalize_calls", "sum"),
                total_hits=("memo_hits", "sum")
            ).reset_index()
            binned["sharing_rate"] = np.where(
                binned["total_finalize"] > 0,
                binned["total_hits"] / binned["total_finalize"],
                0.0
            )
            for _, row in binned.iterrows():
                all_rows.append({
                    "dim": dim,
                    "depth_bin": int(row["depth_bin"]),
                    "sharing_rate": row["sharing_rate"]
                })

        if not all_rows:
            continue

        pdf = pd.DataFrame(all_rows)
        pivot = pdf.pivot_table(values="sharing_rate", index="depth_bin",
                                columns="dim", aggfunc="mean")
        pivot = pivot.sort_index(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values * 100, aspect="auto",
                       cmap="YlOrRd", interpolation="nearest", origin="upper")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        y_labels = [f"{b / (n_depth_bins - 1):.1f}" for b in pivot.index]
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Dimensions")
        ax.set_ylabel(
            "normalized_depth = depth × group_bits / total_bits\n"
            "(0 = root, 1 = deepest level)"
        )
        ax.set_title(
            f"sharing_rate = memo_hits / finalize_calls by Depth & Dimension\n"
            f"GB={gb}, N={max_n:,}  —  aggregated within {n_depth_bins} normalized-depth bins"
        )

        fig.colorbar(im, ax=ax, label="sharing_rate (%)")
        fig.tight_layout()
        path = os.path.join(output_dir, f"depth_heatmap_gb{gb}.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
        if show:
            plt.show()
        plt.close(fig)


def plot_indegree_histogram(_summary, _depth, indeg, output_dir, show, **kwargs):
    """Bar chart of in-degree distribution, one panel per dim.

    In-degree of a node = number of edges in the packed DAWG that target it.
    - in-degree 1 (blue): nodes reached by exactly one parent — not shared.
    - in-degree > 1 (coral): nodes reached by multiple parents — concretely
      shared suffixes.  These are the nodes that make a DAWG smaller than a trie.

    Note: in-degree-0 nodes (root, or isolated nodes) are NOT included because
    they cannot be enumerated from the edge table alone.

    Every subplot uses fixed x positions (in-degrees 1..cap) so empty bins are
    visible as gaps.  y-axis is log-scaled; zero bars are omitted (NaN).
    """
    if indeg.empty:
        return

    IN_DEGREE_DISPLAY_CAP = 20

    max_n = indeg["n_keys"].max()
    sub = indeg[indeg["n_keys"] == max_n]
    if sub.empty:
        return

    gb_values = sorted(sub["group_bits"].unique())
    cap = IN_DEGREE_DISPLAY_CAP

    for gb in gb_values:
        gs = sub[sub["group_bits"] == gb]

        dims = sorted(gs["dim"].unique())
        n_dims = len(dims)
        cols = min(n_dims, 3)
        rows = (n_dims + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                                 squeeze=False)

        for idx, dim in enumerate(dims):
            ax = axes[idx // cols][idx % cols]
            dd = gs[gs["dim"] == dim].sort_values("in_degree")
            if dd.empty:
                continue

            dd = dd.groupby("in_degree", as_index=False)["node_count"].sum().sort_values(
                "in_degree"
            )

            total_nodes = int(dd["node_count"].sum())
            shared = dd[dd["in_degree"] > 1]
            shared_total = int(shared["node_count"].sum())

            cnt = dict(zip(dd["in_degree"].astype(int), dd["node_count"].astype(int)))
            heights = [int(cnt.get(d, 0)) for d in range(1, cap + 1)]

            x_positions = list(range(1, cap + 1))
            plot_heights = list(heights)
            colors = ["steelblue"] + ["coral"] * (cap - 1)
            hatches = [hatch_for(0)] + [hatch_for(1)] * (cap - 1)

            # Log y: use NaN where count is 0 so matplotlib omits the bar (empty slot).
            bar_heights = [float(h) if h > 0 else np.nan for h in plot_heights]

            bars = ax.bar(
                x_positions,
                bar_heights,
                width=0.75,
                align="center",
                color=colors,
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            tick_labels = [str(x) for x in range(1, cap + 1)]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

            ax.set_xlim(0.5, cap + 0.5)

            ax.set_xlabel("In-degree")
            ax.set_ylabel("Node count")
            ax.set_title(f"dim={dim} ({shared_total}/{total_nodes} shared)")
            ax.set_yscale("log")
            # Only decades on y (10^n): no minor ticks, no minor grid — cleaner background.
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_axisbelow(True)
            ax.grid(False, axis="x")
            ax.grid(
                True,
                which="major",
                axis="y",
                linestyle="-",
                linewidth=0.5,
                alpha=0.2,
                color="0.55",
            )

        for idx in range(n_dims, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        legend_handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="steelblue", edgecolor="black",
                linewidth=0.6, hatch=hatch_for(0),
                label="in-degree = 1 (not shared)",
            ),
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="coral", edgecolor="black",
                linewidth=0.6, hatch=hatch_for(1),
                label="in-degree > 1 (shared)",
            ),
        ]
        # Use the first subplot's legend so suptitle has room and the meaning
        # of the colors/hatches stays close to the bars.
        axes[0][0].legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

        fig.suptitle(f"In-Degree Distribution (GB={gb}, N={max_n:,})", fontsize=14)
        fig.tight_layout()
        path = os.path.join(output_dir, f"indegree_histogram_gb{gb}.png")
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
        if show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Table export helpers (formatting aligned with bench_dawg_storage / plot_dawg_sweep)
# ---------------------------------------------------------------------------


def _pct_fmt(v: object) -> str:
    """Format a [0, 1] fraction as a percentage string, e.g. 0.239 -> '23.90%'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{float(v) * 100:.2f}%"


def _comma_fmt(n: object) -> str:
    """Format an integer with thousands separators, e.g. 1234567 -> '1,234,567'."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "--"
    return f"{int(round(float(n))):,}"


def _size_fmt(bytes_val: object) -> str:
    """Human-readable byte size string (B / KB / MB / GB)."""
    if bytes_val is None or (isinstance(bytes_val, float) and np.isnan(bytes_val)):
        return "--"
    b = int(round(float(bytes_val)))
    kb, mb, gb = 1024, 1024 ** 2, 1024 ** 3
    if b < kb:
        return f"{b} B"
    if b < mb:
        return f"{b / kb:.1f} KB"
    if b < gb:
        return f"{b / mb:.2f} MB"
    return f"{b / gb:.2f} GB"


def _bpk_fmt(b: object) -> str:
    """Bytes-per-key with unit suffix."""
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return "--"
    x = float(b)
    if x < 1024.0:
        return f"{x:.1f} B"
    if x < 1024.0 ** 2:
        return f"{x / 1024.0:.1f} KB"
    return f"{x / 1024.0 ** 2:.2f} MB"


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a plain Markdown table string."""
    h = "| " + " | ".join(_md_escape(x) for x in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_md_escape(c) for c in row) + " |" for row in rows]
    return "\n".join([h, sep] + body)


def _write_md(path: str, title: str, blurb: str, sections: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{blurb}\n\n")
        f.write("".join(sections))
    print(f"Saved {path}")


def _pivot_to_md(
    pivot: pd.DataFrame,
    row_label: str,
    fmt: object,  # callable(value) -> str
    col_labels: list[str] | None = None,
) -> str:
    """Render a pivoted DataFrame as a Markdown table string."""
    if pivot.empty:
        return ""
    cols = col_labels if col_labels is not None else [str(c) for c in pivot.columns]
    rows: list[list[str]] = []
    for idx, row in pivot.iterrows():
        cells = [str(idx)] + [fmt(row[c]) for c in pivot.columns]
        rows.append(cells)
    return _markdown_table([row_label] + cols, rows) + "\n\n"


def export_tables(summary: pd.DataFrame, depth: pd.DataFrame,
                  indeg: pd.DataFrame, output_dir: str) -> None:
    """Write Markdown companion tables for all three sharing CSVs.

    Outputs (in output_dir/):
    - table_sharing_summary.md  sharing_ratio, edge_saving_pct, node_reduction
                                 dim × GB, one section per N
    - table_sharing_vs_n.md     sharing_ratio vs N
                                 GB × N, one section per dim
    - table_storage.md          bytes/key and total size
                                 dim × GB, one section per N
    - table_depth_profile.md    sharing_rate at ~5 normalized-depth breakpoints
                                 dim × breakpoint, one section per GB
    - table_indegree_summary.md shared_nodes count and % of total nodes
                                 dim × GB, one section per N
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. table_sharing_summary.md
    #    sharing_ratio (%), edge_saving_pct (%), node_reduction (%)
    #    rows = dim, columns = GB, one section per N
    # ------------------------------------------------------------------
    def _sharing_section(sub: pd.DataFrame, n: int) -> str:
        """Three side-by-side pivot blocks for one N value."""
        out = f"## N = {_comma_fmt(n)} unique keys\n\n"
        for metric, label, fmt in [
            ("sharing_ratio",  "sharing_ratio = memo_hits / finalize_calls", _pct_fmt),
            ("edge_saving_pct", "edge_saving_pct = 1 − dawg_edges / trie_edges", _pct_fmt),
            ("node_reduction",  "node_reduction = unique_nodes / finalize_calls", _pct_fmt),
        ]:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(
                index="dim", columns="group_bits", values=metric, aggfunc="mean"
            )
            if pivot.empty:
                continue
            pivot = pivot.reindex(index=sorted(int(i) for i in pivot.index))
            pivot = pivot.reindex(columns=sorted(int(c) for c in pivot.columns))
            col_lbl = [f"GB={int(c)}" for c in pivot.columns]
            out += f"### {label}\n\n"
            out += _pivot_to_md(pivot, "dim", fmt, col_lbl)
        return out

    if not summary.empty:
        ss_sections: list[str] = []
        for n in sorted(summary["n_unique_keys"].unique()):
            nsub = summary[summary["n_unique_keys"] == n]
            if not nsub.empty:
                ss_sections.append(_sharing_section(nsub, int(n)))
        _write_md(
            os.path.join(output_dir, "table_sharing_summary.md"),
            "Suffix sharing summary",
            "Three metrics across **dim × GROUP_BITS** for each dataset size N.\n"
            "- **sharing_ratio** = `memo_hits / finalize_calls`: fraction of trie nodes "
            "that were memo hits (suffix shared).\n"
            "- **edge_saving_pct** = `1 - dawg_edges / trie_edges`: fraction of edges "
            "eliminated by sharing.\n"
            "- **node_reduction** = `unique_nodes / finalize_calls`: complement of "
            "sharing_ratio (fraction of nodes written as new).",
            ss_sections,
        )

    # ------------------------------------------------------------------
    # 2. table_sharing_vs_n.md
    #    sharing_ratio (%) — rows = GB, columns = N, one section per dim
    # ------------------------------------------------------------------
    if not summary.empty:
        svn_sections: list[str] = []
        for dim in sorted(summary["dim"].unique()):
            dsub = summary[summary["dim"] == dim]
            if dsub.empty:
                continue
            pivot = dsub.pivot_table(
                index="group_bits", columns="n_unique_keys",
                values="sharing_ratio", aggfunc="mean"
            )
            if pivot.empty:
                continue
            pivot = pivot.reindex(index=sorted(int(i) for i in pivot.index))
            pivot = pivot.reindex(columns=sorted(int(c) for c in pivot.columns))
            col_lbl = [_comma_fmt(int(c)) for c in pivot.columns]
            rows_m: list[list[str]] = []
            for gb, row in pivot.iterrows():
                rows_m.append([f"GB={int(gb)}"] + [_pct_fmt(row[c]) for c in pivot.columns])
            svn_sections.append(
                f"## dim={dim}\n\n"
                + _markdown_table(["GROUP_BITS"] + col_lbl, rows_m)
                + "\n\n"
            )
        _write_md(
            os.path.join(output_dir, "table_sharing_vs_n.md"),
            "Sharing ratio vs dataset size",
            "**sharing_ratio = memo_hits / finalize_calls (%)** for each "
            "_(dim, GROUP_BITS, N)_ combination. Rows: **GROUP_BITS**. "
            "Columns: **N** (unique keys). One section per dimension.",
            svn_sections,
        )

    # ------------------------------------------------------------------
    # 3. table_storage.md
    #    bytes/key and total size — rows = dim, columns = GB, one section per N
    # ------------------------------------------------------------------
    if not summary.empty:
        st_sections: list[str] = []
        for n in sorted(summary["n_unique_keys"].unique()):
            nsub = summary[summary["n_unique_keys"] == n]
            if nsub.empty:
                continue
            out = f"## N = {_comma_fmt(n)} unique keys\n\n"
            for metric, label, fmt in [
                ("bytes_per_key", "Bytes per key", _bpk_fmt),
                ("total_bytes",   "Total DAWG size", _size_fmt),
            ]:
                if metric not in nsub.columns:
                    continue
                pivot = nsub.pivot_table(
                    index="dim", columns="group_bits", values=metric, aggfunc="mean"
                )
                if pivot.empty:
                    continue
                pivot = pivot.reindex(index=sorted(int(i) for i in pivot.index))
                pivot = pivot.reindex(columns=sorted(int(c) for c in pivot.columns))
                out += f"### {label}\n\n"
                out += _pivot_to_md(pivot, "dim", fmt,
                                    [f"GB={int(c)}" for c in pivot.columns])
            st_sections.append(out)
        _write_md(
            os.path.join(output_dir, "table_storage.md"),
            "DAWG storage by dim and GROUP_BITS",
            "**Bytes/key** (`bpk_fmt`) and **total DAWG size** (`size_fmt`) "
            "for each _(dim, GROUP_BITS, N)_ combination. "
            "Rows: **dim**. Columns: **GROUP_BITS**. One section per N.",
            st_sections,
        )

    # ------------------------------------------------------------------
    # 4. table_depth_profile.md
    #    sharing_rate at ~5 normalized-depth breakpoints
    #    rows = dim, columns = breakpoint, one section per (GB, N)
    # ------------------------------------------------------------------
    DEPTH_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]

    if not depth.empty and "normalized_depth" in depth.columns:
        dp_sections: list[str] = []
        max_n = depth["n_keys"].max()
        sub = depth[depth["n_keys"] == max_n]
        for gb in sorted(sub["group_bits"].unique()):
            gs = sub[sub["group_bits"] == gb]
            dims = sorted(gs["dim"].unique())
            if not dims:
                continue

            bin_labels = [f"nd≈{b:.2f}" for b in DEPTH_BINS]
            rows_m: list[list[str]] = []
            for dim in dims:
                dd = gs[(gs["dim"] == dim) & (gs["finalize_calls"] > 0)].copy()
                if dd.empty:
                    cells = ["--"] * len(DEPTH_BINS)
                else:
                    cells: list[str] = []
                    for target in DEPTH_BINS:
                        # Find the row whose normalized_depth is closest to target
                        closest = dd.iloc[
                            (dd["normalized_depth"] - target).abs().argsort().iloc[0]
                        ]
                        fc = closest["finalize_calls"]
                        mh = closest["memo_hits"]
                        rate = mh / fc if fc > 0 else 0.0
                        nd = closest["normalized_depth"]
                        cells.append(f"{rate*100:.1f}% (nd={nd:.2f})")
                rows_m.append([str(int(dim))] + cells)

            dp_sections.append(
                f"## GB={gb}, N={_comma_fmt(max_n)} unique keys\n\n"
                + _markdown_table(["dim"] + bin_labels, rows_m)
                + "\n\n"
            )
        _write_md(
            os.path.join(output_dir, "table_depth_profile.md"),
            "Per-depth sharing rate (5 normalized-depth breakpoints)",
            "**sharing_rate = memo_hits / finalize_calls (%)** sampled at "
            "five `normalized_depth` (nd) levels: 0.0 (root), 0.25, 0.50, 0.75, 1.0 (deepest). "
            "Each cell shows the rate and the actual `nd` of the nearest data point. "
            "Rows: **dim**. Columns: **nd breakpoint**. "
            "One section per GROUP_BITS (at the largest N).",
            dp_sections,
        )

    # ------------------------------------------------------------------
    # 5. table_indegree_summary.md
    #    shared_nodes (in-degree > 1) count and % of total finalised nodes
    #    rows = dim, columns = GB, one section per N
    #    Uses summary.shared_nodes and summary.node_count
    # ------------------------------------------------------------------
    if not summary.empty and "shared_nodes" in summary.columns and "node_count" in summary.columns:
        id_sections: list[str] = []
        for n in sorted(summary["n_unique_keys"].unique()):
            nsub = summary[summary["n_unique_keys"] == n].copy()
            if nsub.empty:
                continue
            nsub = nsub.copy()
            nsub["shared_pct"] = np.where(
                nsub["node_count"] > 0,
                nsub["shared_nodes"] / nsub["node_count"],
                0.0,
            )

            out = f"## N = {_comma_fmt(n)} unique keys\n\n"
            # shared_nodes count
            pivot_cnt = nsub.pivot_table(
                index="dim", columns="group_bits", values="shared_nodes", aggfunc="mean"
            )
            if not pivot_cnt.empty:
                pivot_cnt = pivot_cnt.reindex(index=sorted(int(i) for i in pivot_cnt.index))
                pivot_cnt = pivot_cnt.reindex(columns=sorted(int(c) for c in pivot_cnt.columns))
                out += "### Shared nodes (in-degree > 1)\n\n"
                out += _pivot_to_md(pivot_cnt, "dim", _comma_fmt,
                                    [f"GB={int(c)}" for c in pivot_cnt.columns])
            # shared_pct
            pivot_pct = nsub.pivot_table(
                index="dim", columns="group_bits", values="shared_pct", aggfunc="mean"
            )
            if not pivot_pct.empty:
                pivot_pct = pivot_pct.reindex(index=sorted(int(i) for i in pivot_pct.index))
                pivot_pct = pivot_pct.reindex(columns=sorted(int(c) for c in pivot_pct.columns))
                out += "### Shared nodes as % of packed DAWG nodes\n\n"
                out += _pivot_to_md(pivot_pct, "dim", _pct_fmt,
                                    [f"GB={int(c)}" for c in pivot_pct.columns])
            id_sections.append(out)
        _write_md(
            os.path.join(output_dir, "table_indegree_summary.md"),
            "Shared-node summary (in-degree > 1)",
            "Nodes with **in-degree > 1** in the packed DAWG edge table — nodes "
            "reachable via multiple parents (concretely shared suffixes). "
            "Note: in-degree-0 nodes (root) are not included in `node_count`.\n\n"
            "- **Shared nodes**: absolute count (`comma_fmt`).\n"
            "- **%**: `shared_nodes / node_count` — fraction of packed nodes that are shared.",
            id_sections,
        )


PLOT_FUNCS = {
    "sharing_ratio_vs_dim": plot_sharing_ratio_vs_dim,
    "sharing_ratio_vs_n": plot_sharing_ratio_vs_n,
    "finalize_breakdown": plot_finalize_breakdown,
    "edge_saving_heatmap": plot_edge_saving_heatmap,
    "depth_profile": plot_depth_profile,
    "depth_edges": plot_depth_edges,
    "depth_heatmap": plot_depth_heatmap,
    "indegree_histogram": plot_indegree_histogram,
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot CompactDawg suffix sharing analysis results")
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Directory containing sharing CSVs")
    parser.add_argument("--output-dir", "-o", default="plots/sharing",
                        help="Output directory for plots and tables")
    parser.add_argument("--plot-type", "-p", default="all",
                        help="Plot type: " + ",".join(PLOT_TYPES)
                             + ",none or 'all' (use none with --export-tables to skip figures)")
    parser.add_argument("--filter-dim", default=None,
                        help="Comma-separated dimensions to include")
    parser.add_argument("--filter-n", default=None,
                        help="Comma-separated N values to include")
    parser.add_argument("--filter-gb", default=None,
                        help="Comma-separated GROUP_BITS to include")
    parser.add_argument("--export-tables", action="store_true",
                        help="Write Markdown companion tables to --output-dir: "
                             "table_sharing_summary.md, table_sharing_vs_n.md, "
                             "table_storage.md, table_depth_profile.md, "
                             "table_indegree_summary.md")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively")
    parser.add_argument(
        "--coord-bits",
        type=int,
        default=32,
        metavar="B",
        help="For sharing_ratio_vs_dim only: bits per coordinate when checking whether "
             "a (dim, GROUP_BITS) row is valid (same rule as bench_dawg_sharing: "
             "(dim × B) %% GROUP_BITS == 0). Default 32 (fp32); use 16 for fp16.",
    )
    parser.add_argument(
        "--min-key-levels",
        type=int,
        default=3,
        metavar="L",
        help="For sharing_ratio_vs_dim only: drop rows where the Morton key spans "
             "fewer than L symbols of width GROUP_BITS, i.e. (dim × coord_bits) / "
             "GROUP_BITS < L (default 3). Matches bench_dawg_sharing after it skips "
             "the same shallow-key configs.",
    )
    args = parser.parse_args()

    summary = load_summary(args.input_dir)
    depth = load_depth(args.input_dir)
    indeg = load_indegree(args.input_dir)

    if summary.empty and depth.empty and indeg.empty:
        print(f"Error: no data found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    filter_dims = (
        [int(x) for x in args.filter_dim.split(",")]
        if args.filter_dim else None
    )
    filter_n = (
        [int(x) for x in args.filter_n.split(",")]
        if args.filter_n else None
    )
    filter_gb = (
        [int(x) for x in args.filter_gb.split(",")]
        if args.filter_gb else None
    )

    if not summary.empty:
        summary = filter_df(summary, filter_dims, filter_n, filter_gb)
    if not depth.empty:
        depth = filter_df(depth, filter_dims, filter_n, filter_gb)
    if not indeg.empty:
        indeg = filter_df(indeg, filter_dims, filter_n, filter_gb)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_type != "none":
        if args.plot_type == "all":
            types_to_plot = PLOT_TYPES
        else:
            types_to_plot = [t.strip() for t in args.plot_type.split(",")]

        plot_kwargs = {"coord_bits": args.coord_bits, "min_key_levels": args.min_key_levels}
        for pt in types_to_plot:
            if pt not in PLOT_FUNCS:
                print(f"Warning: unknown plot type '{pt}', skipping")
                continue
            print(f"Generating {pt}...")
            PLOT_FUNCS[pt](summary, depth, indeg, args.output_dir, args.show,
                           **plot_kwargs)

    if args.export_tables:
        print("Exporting Markdown tables...")
        export_tables(summary, depth, indeg, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
