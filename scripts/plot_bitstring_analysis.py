#!/usr/bin/env python3
"""
plot_bitstring_analysis.py — Visualize variable group bits analysis.

Reads CSVs produced by analyze_bitstring_structure and generates XY line plots.
CSV may include `cardinality_over_kmax` (K/min(N,2^g)); if absent, it is derived
for backward compatibility with older analyzer outputs.
Empty auto-detected segmentation CSVs are skipped. Pass ``--clean-stale`` when
you want old segmentation PNGs removed if no segmentation data is available.

To change which ``group_width`` curves appear on per-dimension cardinality/saturation
plots, edit ``GROUP_WIDTHS_TO_PLOT`` below (only values present in the CSV are drawn).

Saturation curves use a centered moving average along chunk index; set
``SATURATION_SMOOTH_FRAC`` to ``0`` for raw per-chunk points.

Usage:
    python3 scripts/plot_bitstring_analysis.py \
        --input results/bitstring_analysis_256d.csv \
        [--input results/bitstring_analysis_1024d.csv ...] \
        [--seg-input results/bitstring_analysis_256d_segmentation.csv ...] \
        [--outdir plots/bitstring_analysis]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import hatch_for, interesting_marker_indices, marker_for  # noqa: E402


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

# Muted, visually distinct palette for group widths
GW_COLORS = {
    1: "#636EFA",
    2: "#EF553B",
    4: "#00CC96",
    8: "#AB63FA",
    16: "#FFA15A",
    32: "#19D3F3",
    64: "#FF6692",
    128: "#B6E880",
    256: "#FF97FF",
    512: "#FECB52",
    1024: "#1F77B4",
}

DIM_COLORS = {
    16: "#636EFA",
    32: "#EF553B",
    64: "#00CC96",
    128: "#AB63FA",
    256: "#FFA15A",
    512: "#19D3F3",
    768: "#FF6692",
    1024: "#B6E880",
}

THRESHOLD_COLORS = {
    0.30: "#636EFA",
    0.50: "#EF553B",
    0.70: "#00CC96",
    0.80: "#AB63FA",
    0.90: "#FFA15A",
    0.95: "#19D3F3",
}

# Explicit group_width (g) values for per-dimension cardinality/saturation plots.
# Order here is plot/legend order; widths not in the loaded CSV are skipped.
GROUP_WIDTHS_TO_PLOT = (1024, 512, 256, 64, 32, 8, 2)

# Saturation plot: centered moving average along chunk index; window length ~ ``frac * n_chunks`` (odd).
# Set to 0 for raw stepwise ρ. Increase for a smoother line; decrease to follow fine detail.
SATURATION_SMOOTH_FRAC = 0.004

# Subset of segmentation thresholds (match nearest value present in CSV).
KEY_SEGMENTATION_THRESHOLDS = (0.95, 0.90, 0.80, 0.70, 0.50, 0.30)


def _group_widths_for_plot(sorted_gws: list[int]) -> list[int]:
    """``GROUP_WIDTHS_TO_PLOT`` filtered to widths present in the data, order preserved."""
    s = set(sorted_gws)
    return [g for g in GROUP_WIDTHS_TO_PLOT if g in s]


def _centered_moving_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Odd-length centered moving average; ``window`` clamped to ``<= len(y)``, odd, >=1."""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    if n == 0:
        return y.copy()
    w = min(max(1, window), n)
    if w % 2 == 0:
        w = max(1, w - 1)
    if w <= 1:
        return y.copy()
    half = w // 2
    pad = np.pad(y, (half, half), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    out = np.convolve(pad, kernel, mode="valid")
    assert out.size == n
    return out


def _smooth_saturation_along_chunks(rho: np.ndarray, *, smooth_frac: float) -> np.ndarray:
    """Smooth ρ along increasing chunk index (along the Morton bitstring)."""
    rho = np.asarray(rho, dtype=np.float64)
    n = rho.size
    if n < 3 or smooth_frac <= 0:
        return rho.copy()
    # Odd window ~ smooth_frac * n, at least 3, at most n (odd)
    raw = max(3, int(n * smooth_frac))
    w = min(n, raw + (1 - raw % 2))  # bump to odd if needed
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return rho.copy()
    return np.clip(_centered_moving_mean(rho, w), 0.0, 1.0)


def _pick_key_thresholds(sorted_th: list[float], *, max_lines: int = 4) -> list[float]:
    if len(sorted_th) <= max_lines:
        return sorted_th

    def _close(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) < 1e-12

    out: list[float] = []
    for target in KEY_SEGMENTATION_THRESHOLDS:
        closest = min(sorted_th, key=lambda t: abs(float(t) - float(target)))
        if not any(_close(closest, x) for x in out):
            out.append(closest)
        if len(out) >= max_lines:
            return sorted(out)
    for t in sorted_th:
        if not any(_close(t, x) for x in out):
            out.append(t)
        if len(out) >= max_lines:
            break
    return sorted(out)


def _symbol_capacity(n: int, g: int) -> int:
    """K_max = min(N, 2^g); match analyze_bitstring_structure (g >= 64 => N)."""
    if g >= 64:
        return int(n)
    cap = 1 << g
    return int(n) if cap >= n else cap


def ensure_cardinality_over_kmax(df: pd.DataFrame) -> pd.DataFrame:
    """Add saturation column K/min(N,2^g) when missing."""
    if "cardinality_over_kmax" in df.columns:
        return df
    out = df.copy()
    caps = np.array(
        [_symbol_capacity(int(a), int(b)) for a, b in zip(out["n_keys"], out["group_width"])],
        dtype=np.float64,
    )
    out["cardinality_over_kmax"] = out["cardinality"].astype(np.float64) / caps
    return out


def ensure_seg_cardinality_over_kmax(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Add cardinality_over_kmax to segmentation rows when missing."""
    if "cardinality_over_kmax" in seg_df.columns:
        return seg_df
    out = seg_df.copy()
    caps = np.array(
        [
            _symbol_capacity(int(a), int(b))
            for a, b in zip(out["n_keys"], out["segment_width"])
        ],
        dtype=np.float64,
    )
    out["cardinality_over_kmax"] = out["cardinality"].astype(np.float64) / caps
    return out


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F9FA",
            "axes.edgecolor": "#CCCCCC",
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.color": "#CCCCCC",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )


def read_csv_if_nonempty(path: str | Path, *, label: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"  Skipping missing {label}: {p}", file=sys.stderr)
        return None
    if p.stat().st_size == 0:
        print(f"  Skipping empty {label}: {p}", file=sys.stderr)
        return None
    try:
        return pd.read_csv(p)
    except EmptyDataError:
        print(f"  Skipping empty {label}: {p}", file=sys.stderr)
        return None


def clean_stale_segmentation_plots(outdir: str) -> None:
    out = Path(outdir)
    patterns = ("segmentation_*d.png", "segment_count_summary.png")
    for pattern in patterns:
        for p in out.glob(pattern):
            p.unlink()
            print(f"  removed stale plot {p}")


# ---------------------------------------------------------------------------
# Plot 1: Cardinality/N vs normalized position (per dimension)
# ---------------------------------------------------------------------------


def plot_cardinality_per_dim(df, outdir):
    """One figure per dimensionality, lines for each group width."""
    dims = sorted(df["dim"].unique())

    for dim in dims:
        dfd = df[df["dim"] == dim]
        total_bits = dfd["total_bits"].iloc[0]
        n_keys = dfd["n_keys"].iloc[0]
        gws_all = sorted(dfd["group_width"].unique())
        gws = _group_widths_for_plot(gws_all)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
        ax.set_facecolor("white")

        for i, gw in enumerate(gws):
            sub = dfd[dfd["group_width"] == gw].sort_values("chunk_start_bit")
            # Normalized position: center of each chunk
            norm_pos = (sub["chunk_start_bit"].values + gw / 2.0) / total_bits
            card_n = sub["cardinality_over_n"].values
            color = GW_COLORS.get(gw, "#333333")
            # Skip markers on the ρ≈0 floor and the ρ≈1 plateau; they would
            # otherwise stack across every group width in the same flat region.
            markevery = interesting_marker_indices(card_n, floor=0.02, ceiling=0.98)
            ax.plot(
                norm_pos,
                card_n,
                label=f"g={gw}",
                color=color,
                linewidth=1.3,
                alpha=0.9,
                marker=marker_for(i),
                markersize=5,
                markevery=markevery,
            )

        ax.set_xlabel("Normalized bit position (0 = MSB, 1 = LSB)")
        ax.set_ylabel("Cardinality / N")
        ax.set_title(
            f"Symbol cardinality vs position — {dim}D (N={n_keys:,}, {total_bits} bits; "
            f"g={','.join(str(g) for g in gws)})"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(ncol=3, loc="upper left")
        fig.tight_layout()

        fname = os.path.join(outdir, f"cardinality_per_dim_{dim}d.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 1b: Saturation K/Kmax vs normalized position (per dimension)
# ---------------------------------------------------------------------------


def plot_saturation_per_dim(df, outdir):
    """Same as cardinality plot but Y = cardinality_over_kmax (rho)."""
    df = ensure_cardinality_over_kmax(df)
    dims = sorted(df["dim"].unique())

    for dim in dims:
        dfd = df[df["dim"] == dim]
        total_bits = dfd["total_bits"].iloc[0]
        n_keys = dfd["n_keys"].iloc[0]
        gws_all = sorted(dfd["group_width"].unique())
        gws = _group_widths_for_plot(gws_all)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
        ax.set_facecolor("white")

        ax.axhline(
            1.0,
            color="#888888",
            linestyle="--",
            linewidth=0.9,
            zorder=0,
            label=r"$\rho=1$",
        )

        for i, gw in enumerate(gws):
            sub = dfd[dfd["group_width"] == gw].sort_values("chunk_start_bit")
            norm_pos = (sub["chunk_start_bit"].values + gw / 2.0) / total_bits
            rho = sub["cardinality_over_kmax"].values.astype(np.float64)
            rho_plot = _smooth_saturation_along_chunks(rho, smooth_frac=SATURATION_SMOOTH_FRAC)
            color = GW_COLORS.get(gw, "#333333")
            # Skip markers on the ρ≈0 floor and the ρ≈1 plateau so curves
            # stop stacking markers vertically in the saturated region.
            markevery = interesting_marker_indices(rho_plot, floor=0.02, ceiling=0.98)
            ax.plot(
                norm_pos,
                rho_plot,
                label=f"g={gw}",
                color=color,
                linewidth=1.4,
                alpha=0.9,
                marker=marker_for(i),
                markersize=5,
                markevery=markevery,
                antialiased=True,
            )

        ax.set_xlabel("Normalized bit position (0 = MSB, 1 = LSB)")
        ax.set_ylabel(r"Saturation $\rho$")
        ax.set_title(f"Symbol saturation vs position — {dim}D")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        # Put reference line first, then g= curves
        ax.legend(handles, labels, ncol=2, loc="upper left", fontsize=8)
        fig.tight_layout()

        fname = os.path.join(outdir, f"saturation_per_dim_{dim}d.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2: Cross-dimension comparison (per group width)
# ---------------------------------------------------------------------------


def plot_cross_dim(df, outdir):
    """One figure per group width, lines for each dimensionality."""
    gws = sorted(df["group_width"].unique())
    dims = sorted(df["dim"].unique())

    if len(dims) < 2:
        print("  Skipping cross-dim plots (only 1 dimensionality)")
        return

    for gw in gws:
        dfg = df[df["group_width"] == gw]

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, dim in enumerate(dims):
            sub = dfg[dfg["dim"] == dim].sort_values("chunk_start_bit")
            if sub.empty:
                continue
            total_bits = sub["total_bits"].iloc[0]
            n_keys = sub["n_keys"].iloc[0]
            norm_pos = (sub["chunk_start_bit"].values + gw / 2.0) / total_bits
            card_n = sub["cardinality_over_n"].values
            color = DIM_COLORS.get(dim, "#333333")
            markevery = interesting_marker_indices(card_n, floor=0.02, ceiling=0.98)
            ax.plot(
                norm_pos,
                card_n,
                label=f"{dim}D (N={n_keys:,})",
                color=color,
                linewidth=1.4,
                alpha=0.9,
                marker=marker_for(i),
                markersize=5,
                markevery=markevery,
            )

        ax.set_xlabel("Normalized bit position (0 = MSB, 1 = LSB)")
        ax.set_ylabel("Cardinality / N")
        ax.set_title(f"Cross-dimension comparison — group_width={gw}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left")
        fig.tight_layout()

        fname = os.path.join(outdir, f"cross_dim_gw{gw}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2b: Cross-dimension saturation (per group width)
# ---------------------------------------------------------------------------


def plot_cross_dim_saturation(df, outdir):
    """One figure per group width: saturation vs position, one line per dim."""
    df = ensure_cardinality_over_kmax(df)
    gws = sorted(df["group_width"].unique())
    dims = sorted(df["dim"].unique())

    if len(dims) < 2:
        print("  Skipping cross-dim saturation plots (only 1 dimensionality)")
        return

    for gw in gws:
        dfg = df[df["group_width"] == gw]

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, dim in enumerate(dims):
            sub = dfg[dfg["dim"] == dim].sort_values("chunk_start_bit")
            if sub.empty:
                continue
            total_bits = sub["total_bits"].iloc[0]
            n_keys = sub["n_keys"].iloc[0]
            norm_pos = (sub["chunk_start_bit"].values + gw / 2.0) / total_bits
            rho = sub["cardinality_over_kmax"].values
            color = DIM_COLORS.get(dim, "#333333")
            markevery = interesting_marker_indices(rho, floor=0.02, ceiling=0.98)
            ax.plot(
                norm_pos,
                rho,
                label=f"{dim}D (N={n_keys:,})",
                color=color,
                linewidth=1.4,
                alpha=0.9,
                marker=marker_for(i),
                markersize=5,
                markevery=markevery,
            )

        ax.set_xlabel("Normalized bit position (0 = MSB, 1 = LSB)")
        ax.set_ylabel("Saturation K / min(N, 2^g)")
        ax.set_title(f"Cross-dimension saturation — group_width={gw}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left")
        fig.tight_layout()

        fname = os.path.join(outdir, f"cross_dim_saturation_gw{gw}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 3: Greedy segmentation visualization
# ---------------------------------------------------------------------------


def plot_segmentation(seg_df, outdir):
    """Step-function plot: group width chosen at each bit position, with
    ρ on secondary axis and horizontal ρ threshold (greedy gate)."""
    seg_df = ensure_seg_cardinality_over_kmax(seg_df)
    dims = sorted(seg_df["dim"].unique())
    thresholds_all = sorted(seg_df["threshold"].unique())

    for dim in dims:
        dfd = seg_df[seg_df["dim"] == dim]
        total_bits = dfd["total_bits"].iloc[0]
        n_keys = dfd["n_keys"].iloc[0]
        thresholds = _pick_key_thresholds(thresholds_all)

        fig, axes = plt.subplots(
            len(thresholds), 1,
            figsize=(14, 3 * len(thresholds)),
            sharex=True,
        )
        if len(thresholds) == 1:
            axes = [axes]

        for ax, thr in zip(axes, thresholds):
            sub = dfd[dfd["threshold"] == thr].sort_values("segment_start_bit")
            if sub.empty:
                continue

            starts = sub["segment_start_bit"].values
            widths = sub["segment_width"].values
            rho = sub["cardinality_over_kmax"].values

            # Build step-function arrays for width
            x_steps = []
            y_steps = []
            for s, w in zip(starts, widths):
                x_steps.extend([s, s + w])
                y_steps.extend([w, w])

            ax.plot(
                np.array(x_steps) / total_bits,
                y_steps,
                color="#636EFA",
                linewidth=1.5,
                drawstyle="steps-post",
                label="Group width",
            )
            ax.set_yscale("log", base=2)
            ax.set_ylabel("Group width")
            ax.set_title(
                f"Greedy segmentation — {dim}D, ρ ≤ {thr:.2f} "
                f"({len(starts)} segments)"
            )

            # ρ (greedy gate) on secondary axis
            ax2 = ax.twinx()
            mid_x = (starts + widths / 2.0) / total_bits
            ax2.axhline(
                thr,
                color="#2CA02C",
                linestyle=":",
                linewidth=1.0,
                alpha=0.55,
                label="ρ threshold",
            )
            ax2.plot(mid_x, rho, color="#2CA02C", linewidth=1.0,
                     alpha=0.75, marker="x", markersize=3, label="ρ=K/Kmax")
            ax2.set_ylabel(r"$\rho = K / K_{\max}$ (dotted = limit)")
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis="y", labelcolor="#2CA02C")

            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                      fontsize=8)

        axes[-1].set_xlabel("Normalized bit position")
        axes[-1].set_xlim(0, 1)
        fig.tight_layout()

        fname = os.path.join(outdir, f"segmentation_{dim}d.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 4: Summary — total segments vs dimension
# ---------------------------------------------------------------------------


def plot_segment_summary(seg_df, outdir):
    """Bar chart: number of segments per dim, grouped by threshold."""
    summary = (
        seg_df.groupby(["dim", "threshold"])
        .agg(n_segments=("segment_idx", "count"),
             total_bits=("total_bits", "first"),
             n_keys=("n_keys", "first"))
        .reset_index()
    )

    dims = sorted(summary["dim"].unique())
    thresholds = sorted(summary["threshold"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.8 / len(thresholds)
    x_base = np.arange(len(dims))

    for i, thr in enumerate(thresholds):
        sub = summary[summary["threshold"] == thr]
        heights = []
        for dim in dims:
            row = sub[sub["dim"] == dim]
            heights.append(row["n_segments"].values[0] if len(row) > 0 else 0)
        color = THRESHOLD_COLORS.get(thr, "#333333")
        ax.bar(
            x_base + i * bar_width,
            heights,
            bar_width,
            label=f"thr={thr:.2f}",
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch_for(i),
        )

    ax.set_xticks(x_base + bar_width * (len(thresholds) - 1) / 2)
    ax.set_xticklabels([f"{d}D" for d in dims])
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Number of segments (greedy)")
    ax.set_title("Greedy segmentation: segment count by dimensionality and threshold")
    ax.legend()
    fig.tight_layout()

    fname = os.path.join(outdir, "segment_count_summary.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot variable group bits analysis results"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more cardinality CSV files from analyze_bitstring_structure",
    )
    parser.add_argument(
        "--seg-input",
        nargs="*",
        default=[],
        help="One or more segmentation CSV files",
    )
    parser.add_argument(
        "--outdir",
        default="plots/bitstring_analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--clean-stale",
        action="store_true",
        help="Remove old segmentation PNGs when no segmentation CSV data is available",
    )
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load cardinality CSVs ---
    dfs = []
    for path in args.input:
        print(f"Loading {path}")
        df_part = read_csv_if_nonempty(path, label="cardinality CSV")
        if df_part is not None:
            dfs.append(df_part)
    if not dfs:
        raise SystemExit("No non-empty cardinality CSVs loaded.")
    df = pd.concat(dfs, ignore_index=True)
    df = ensure_cardinality_over_kmax(df)
    print(f"  Total rows: {len(df):,}")
    print(f"  Dims: {sorted(df['dim'].unique())}")
    print(f"  Group widths: {sorted(df['group_width'].unique())}")

    # --- Load segmentation CSVs ---
    seg_df = None
    if args.seg_input:
        seg_dfs = []
        for path in args.seg_input:
            print(f"Loading segmentation: {path}")
            seg_part = read_csv_if_nonempty(path, label="segmentation CSV")
            if seg_part is not None:
                seg_dfs.append(seg_part)
        if seg_dfs:
            seg_df = pd.concat(seg_dfs, ignore_index=True)
            seg_df = ensure_seg_cardinality_over_kmax(seg_df)
            print(f"  Total segmentation rows: {len(seg_df):,}")
    else:
        # Auto-detect segmentation files from input paths
        seg_dfs = []
        for path in args.input:
            p = Path(path)
            seg_path = p.parent / (p.stem + "_segmentation" + p.suffix)
            if seg_path.exists():
                print(f"Auto-detected segmentation: {seg_path}")
                seg_part = read_csv_if_nonempty(seg_path, label="segmentation CSV")
                if seg_part is not None:
                    seg_dfs.append(seg_part)
        if seg_dfs:
            seg_df = pd.concat(seg_dfs, ignore_index=True)
            seg_df = ensure_seg_cardinality_over_kmax(seg_df)
            print(f"  Total segmentation rows: {len(seg_df):,}")

    # --- Generate plots ---
    print("\nGenerating plots...")

    print("Plot 1: Cardinality per dimension")
    plot_cardinality_per_dim(df, args.outdir)

    print("Plot 1b: Saturation per dimension")
    plot_saturation_per_dim(df, args.outdir)

    print("Plot 2: Cross-dimension comparison")
    plot_cross_dim(df, args.outdir)

    print("Plot 2b: Cross-dimension saturation")
    plot_cross_dim_saturation(df, args.outdir)

    if seg_df is not None and not seg_df.empty:
        print("Plot 3: Segmentation visualization")
        plot_segmentation(seg_df, args.outdir)

        print("Plot 4: Segment count summary")
        plot_segment_summary(seg_df, args.outdir)
    else:
        print("Skipping segmentation plots (no segmentation data)")
        if args.clean_stale:
            clean_stale_segmentation_plots(args.outdir)

    print(f"\nAll plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
