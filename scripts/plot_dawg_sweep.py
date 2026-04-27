#!/usr/bin/env python3
"""Plot CompactDawg sweep benchmark results from CSV.

Usage:
    python scripts/plot_dawg_sweep.py -i dawg_sweep_results.csv -o plots/
    python scripts/plot_dawg_sweep.py -i results.csv -p storage_sweep --sweep-x-axis n,dim
    python scripts/plot_dawg_sweep.py -i results.csv -p all --export-tables

Merge multiple benchmark runs (e.g. small-N + large-N) into one plot; later rows win on duplicate keys:
    python scripts/plot_dawg_sweep.py -i part1.csv -i part2.csv --output-dir plots/

``storage_sweep`` / ``time_sweep``: y-axis vs ``--sweep-x-axis`` (``n``, ``dim``, ``group_bits``).
Storage uses **Normalized BPK** (``bytes_per_key / key_bytes``). **CD** and **PC** are on
separate figures (never mixed). Rows with ``method == dawgdic`` are **dropped** on load. Storage-oriented figures also drop
shallow ``(dim, GROUP_BITS)`` rows where the Morton key spans fewer than a configurable number of grouped symbols (default: 3),
because 1- or 2-level keys produce degenerate near-zero suffix-collapse behavior that is not scientifically comparable.
``suffix_collapse`` uses CD rows with suffix-collapse columns and plots estimated packed-trie normalized BPK
before suffix collapse against final CompactDawg normalized BPK after suffix collapse.
With multiple dtypes, filenames get a ``_float32`` / ``_float16`` suffix.
For ``storage_sweep`` with ``x=n``, the script also writes a compact ``best fixed scheme per dimension``
variant that keeps only one method per dimension: the method with the lowest **mean normalized BPK**
across the plotted ``N`` values (ties break on the largest-``N`` normalized BPK, then method order).
Whenever **x** is not ``n`` (i.e. ``dim`` or ``group_bits``), **y** uses only rows at one
``n_unique_keys`` value: the **largest** N such that **every** dimension in the filtered
data has at least one row at that N (if none exists, a legacy mixed-N fallback is used).
``x=n`` still sweeps all ``N``.

``--plot-type all`` runs ``storage_sweep`` and ``time_sweep``. Sweep axes come from ``--sweep-x-axis``
(default ``n`` only; pass ``n,dim,group_bits`` for more figures).

``--filter-dtype`` only affects matplotlib figures; tables use every dtype left after
``--filter-dim`` / ``--filter-n`` / ``--filter-gb`` (override with ``--export-tables-dtype``).
When both ``float32`` and ``float16`` are present, ``--export-tables`` also writes
direct normalized-BPK comparison tables for CD and PC.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PLOT_TYPES = [
    "storage_sweep",
    "time_sweep",
    "suffix_collapse",
    "none",  # skip figures (use with --export-tables)
]

# Rows that identify one benchmark measurement when merging CSVs from multiple runs.
DEDUPE_KEYS = ["dim", "dtype", "n_unique_keys", "group_bits", "method"]

SWEEP_X_ALIASES = {"n": "n", "dim": "dim", "group_bits": "group_bits", "gb": "group_bits"}

TIME_METRIC_CHOICES = ("total_build_s", "insert_s", "finish_s")
DEFAULT_MIN_KEY_LEVELS = 3


def _bits_per_coord(dtype: object) -> int:
    if dtype is None or (isinstance(dtype, float) and np.isnan(dtype)) or pd.isna(dtype):
        return 32
    s = str(dtype).lower().strip()
    if s == "float16":
        return 16
    return 32


def add_key_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bpc = out["dtype"].map(_bits_per_coord) if "dtype" in out.columns else 32
    if not isinstance(bpc, pd.Series):
        bpc = pd.Series([32] * len(out), index=out.index)
    out["bits_per_coord"] = bpc.astype(int)
    out["key_bytes"] = out["dim"].astype(float) * out["bits_per_coord"].astype(float) / 8.0
    kb = out["key_bytes"].replace(0, np.nan)
    out["normalized_bpk"] = out["bytes_per_key"].astype(float) / kb
    return out


def _annotate_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_pc"] = out["method"].str.startswith("PC-")
    out["variant"] = out["method"].apply(lambda m: "PC" if str(m).startswith("PC-") else "CD")
    return out


def load_csv(paths: list[str], dedupe: bool) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "dtype" in df.columns:
        st = df["dtype"].astype(str).str.strip()
        df["dtype"] = st.mask(st.str.lower().isin(("", "nan", "none")), pd.NA)
    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=DEDUPE_KEYS, keep="last").reset_index(drop=True)
        if before != len(df):
            print(f"Merged {len(paths)} file(s): dropped {before - len(df)} duplicate row(s) (keep=last).")
    df = _annotate_variants(df)
    if "method" in df.columns:
        before_d = len(df)
        df = df[df["method"].astype(str) != "dawgdic"].reset_index(drop=True)
        dropped_d = before_d - len(df)
        if dropped_d:
            print(f"Excluded {dropped_d} dawgdic row(s) (not plotted or tabulated).")
    if "bytes_per_key" in df.columns and "dim" in df.columns:
        df = add_key_metrics(df)
    return df


def filter_df(
    df: pd.DataFrame,
    filter_dims: list[int] | None,
    filter_n: list[int] | None,
    filter_gb: list[int] | None,
    filter_dtype: str | None,
) -> pd.DataFrame:
    if filter_dims:
        df = df[df["dim"].isin(filter_dims)]
    if filter_n:
        df = df[df["n_unique_keys"].isin(filter_n)]
    if filter_gb:
        df = df[df["group_bits"].isin(filter_gb)]
    if filter_dtype:
        df = df[df["dtype"] == filter_dtype]
    return df


def _storage_relevance_mask(df: pd.DataFrame, min_key_levels: int) -> pd.Series:
    """True where a storage row is structurally meaningful for cross-config plots.

    We only keep rows whose Morton key spans at least ``min_key_levels``
    GROUP_BITS-wide symbols. This mirrors the shallow-key guard already used by
    the sharing benchmark/plotter so wide GROUP_BITS on small dimensions do not
    show up as misleading zero-savings storage points.
    """
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)
    if min_key_levels <= 0:
        return pd.Series(True, index=df.index)
    if not {"dim", "group_bits"}.issubset(df.columns):
        return pd.Series(True, index=df.index)

    bpc = df["dtype"].map(_bits_per_coord) if "dtype" in df.columns else 32
    if not isinstance(bpc, pd.Series):
        bpc = pd.Series([32] * len(df), index=df.index)

    total_bits = pd.to_numeric(df["dim"], errors="coerce") * pd.to_numeric(bpc, errors="coerce")
    gb = pd.to_numeric(df["group_bits"], errors="coerce")

    valid = total_bits.notna() & gb.notna() & (gb > 0)
    valid &= np.isclose(np.mod(total_bits, gb), 0.0)
    levels = total_bits / gb
    valid &= levels >= float(min_key_levels)
    return valid.fillna(False)


def _filter_storage_relevant_rows(df: pd.DataFrame, min_key_levels: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[_storage_relevance_mask(df, min_key_levels)].copy()


def _per_dtype_plot_dfs(df: pd.DataFrame) -> list[tuple[str | None, pd.DataFrame]]:
    if df.empty or "dtype" not in df.columns:
        return [(None, df)]
    dtypes = sorted(str(x) for x in df["dtype"].dropna().unique())
    if len(dtypes) <= 1:
        return [(None, df)]
    return [(dt, df[df["dtype"] == dt]) for dt in dtypes]


def _plot_filename_stem(base: str, dtype_tag: str | None) -> str:
    return f"{base}_{dtype_tag}" if dtype_tag else base


def _parse_sweep_axes(s: str) -> list[str]:
    out: list[str] = []
    for part in s.split(","):
        key = part.strip().lower()
        if not key:
            continue
        canon = SWEEP_X_ALIASES.get(key)
        if canon is None:
            print(f"Warning: unknown sweep axis '{part}', expected n,dim,group_bits (or gb)", file=sys.stderr)
            continue
        if canon not in out:
            out.append(canon)
    return out if out else ["n"]


def _xaxis_file_token(x_kind: str) -> str:
    return {"n": "xn", "dim": "xdim", "group_bits": "xgb"}[x_kind]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _format_group_bits_tick(val: float, pos: object = None) -> str:
    """X-axis tick labels: show powers of two as 2^n; otherwise the integer value."""
    if not np.isfinite(val) or val <= 0:
        return ""
    v = int(round(val))
    if v <= 0:
        return ""
    if _is_power_of_two(v):
        p = v.bit_length() - 1
        return rf"$2^{{{p}}}$"
    return str(v)


def _configure_group_bits_xaxis(ax: plt.Axes, group_bits_values: pd.Series | np.ndarray) -> None:
    """Log₂ x-axis with ticks at observed GROUP_BITS; labels as 2^n when applicable."""
    vals = sorted({int(x) for x in pd.Series(group_bits_values).dropna().unique()})
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    if vmin <= 0:
        return
    ax.set_xscale("log", base=2)
    ax.set_xlim(vmin * (2**-0.25), vmax * (2**0.25))
    ax.xaxis.set_major_locator(mticker.FixedLocator([float(v) for v in vals]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_group_bits_tick))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("GROUP_BITS")
    ax.tick_params(axis="x", labelrotation=0)


def _methods_for_variant(sub: pd.DataFrame, variant: str) -> list[str]:
    if sub.empty:
        return []
    methods = [str(m) for m in sub["method"].unique()]
    pref = variant + "-"
    m2 = [m for m in methods if m.startswith(pref)]

    def gb_key(m: str) -> tuple[int, str]:
        rest = m[len(pref) :]
        try:
            return (int(rest), m)
        except ValueError:
            return (10**9, m)

    return sorted(m2, key=gb_key)


def _best_fixed_scheme_per_dim(
    sub: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Pick one fixed method per dimension for x=n sweeps.

    Selection rule:
    1) lowest mean ``value_col`` across available N values
    2) lowest ``value_col`` at the largest available N for that method
    3) stable method order via ``_method_sort_key``
    """
    if sub.empty or value_col not in sub.columns or "n_unique_keys" not in sub.columns:
        return sub.iloc[0:0].copy()

    picks: list[dict[str, object]] = []
    for dim in sorted(int(d) for d in sub["dim"].dropna().unique()):
        dsub = sub[sub["dim"] == dim]
        if dsub.empty:
            continue
        candidates: list[tuple[tuple[object, ...], dict[str, object]]] = []
        for method in sorted({str(m) for m in dsub["method"].dropna().unique()}, key=_method_sort_key):
            ms = dsub[dsub["method"] == method].sort_values("n_unique_keys")
            if ms.empty:
                continue
            mean_val = float(ms[value_col].astype(float).mean())
            max_n = int(ms["n_unique_keys"].max())
            max_n_val = float(
                ms.loc[ms["n_unique_keys"] == max_n, value_col].astype(float).iloc[-1]
            )
            record = {
                "dim": dim,
                "method": method,
                "mean_value": mean_val,
                "value_at_max_n": max_n_val,
                "max_n": max_n,
                "n_points": int(ms["n_unique_keys"].nunique()),
            }
            key = (mean_val, max_n_val, _method_sort_key(method))
            candidates.append((key, record))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            picks.append(candidates[0][1])
    return pd.DataFrame(picks)


def _slice_max_n(ddf: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Pick one N = n_unique_keys for dim / group_bits sweeps.

    Prefer the **largest** N such that **every** dimension in ``ddf`` has at least
    one row at that N (so cross-dimension curves are comparable). If no such N
    exists, fall back to the previous behavior (global max N, then per
    (dim, method) idxmax rows). The bool is True iff the common-N rule matched.
    """
    if ddf.empty:
        return ddf, 0, False
    dims = sorted({int(d) for d in ddf["dim"].dropna().unique()})
    if not dims:
        return ddf.iloc[0:0].copy(), 0, False
    n_sorted = sorted({int(x) for x in ddf["n_unique_keys"].dropna().unique()}, reverse=True)
    for n in n_sorted:
        if all(
            not ddf[(ddf["dim"] == d) & (ddf["n_unique_keys"] == n)].empty for d in dims
        ):
            sub = ddf[ddf["n_unique_keys"] == n].reset_index(drop=True)
            return sub, n, True
    max_n = int(ddf["n_unique_keys"].max())
    sub = ddf[ddf["n_unique_keys"] == max_n]
    if sub.empty:
        idx = ddf.groupby(["dim", "method"])["n_unique_keys"].idxmax()
        sub = ddf.loc[idx].reset_index(drop=True)
        max_n = int(sub["n_unique_keys"].max()) if not sub.empty else max_n
    return sub, max_n, False


def _plot_sweep_facets(
    ddf: pd.DataFrame,
    variant: str,
    x_kind: str,
    ycol: str,
    ylabel: str,
    title_bits: str,
    *,
    storage_y_log: bool = False,
    time_log_y: bool = True,
) -> plt.Figure | None:
    slice_max_n_for_title: int | None = None
    vs = ddf[ddf["variant"] == variant]
    if vs.empty or ycol not in vs.columns:
        return None
    methods = _methods_for_variant(vs, variant)
    if not methods:
        return None

    if x_kind == "n":
        xcol = "n_unique_keys"
        dims = sorted(vs["dim"].unique())
        n_dims = len(dims)
        cols = min(n_dims, 3)
        rows = (n_dims + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        for idx, dim in enumerate(dims):
            ax = axes[idx // cols][idx % cols]
            sub = vs[vs["dim"] == dim]
            for method in methods:
                ms = sub[sub["method"] == method].sort_values(xcol)
                if ms.empty:
                    continue
                ax.plot(ms[xcol], ms[ycol], marker="o", markersize=4, label=method)
            ax.set_xlabel("N (unique keys)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"dim={dim}")
            ax.set_xscale("log")
            if storage_y_log or (ycol != "normalized_bpk" and time_log_y and ycol.endswith("_s")):
                ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        for idx in range(n_dims, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

    elif x_kind == "dim":
        sub_all, max_n, n_all_dims = _slice_max_n(vs)
        if sub_all.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in methods:
            ms = sub_all[sub_all["method"] == method].sort_values("dim")
            if ms.empty:
                continue
            ax.plot(ms["dim"], ms[ycol], marker="o", markersize=5, label=method)
        ax.set_xlabel("Dimensions")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub_all["dim"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if storage_y_log or (ycol != "normalized_bpk" and time_log_y and ycol.endswith("_s")):
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        n_note = "largest N with all dimensions" if n_all_dims else "mixed N (no single N for all dims)"
        ax.set_title(f"N={max_n:,} ({n_note})")
        slice_max_n_for_title = max_n

    elif x_kind == "group_bits":
        xcol = "group_bits"
        plot_vs, gb_max_n, _ = _slice_max_n(vs)
        if plot_vs.empty:
            return None
        slice_max_n_for_title = gb_max_n
        # One axes: one line per dimension (distinct colors by dim).
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for i, dim in enumerate(sorted(plot_vs["dim"].unique())):
            sub_dim = plot_vs[plot_vs["dim"] == dim].sort_values(xcol)
            if sub_dim.empty:
                continue
            ax.plot(
                sub_dim[xcol],
                sub_dim[ycol],
                marker="o",
                markersize=5,
                label=f"{int(dim)}D",
                color=cmap(i % 10),
            )
        ax.set_ylabel(ylabel)
        _configure_group_bits_xaxis(ax, plot_vs["group_bits"])
        if storage_y_log or (ycol != "normalized_bpk" and time_log_y and ycol.endswith("_s")):
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2, title="Dimension")
    else:
        return None

    if slice_max_n_for_title is not None and x_kind in ("dim", "group_bits"):
        title_bits = f"{title_bits} (N={slice_max_n_for_title:,})"

    fig.suptitle(title_bits, fontsize=14)
    fig.tight_layout()
    return fig


def _plot_best_fixed_scheme_by_dim(
    ddf: pd.DataFrame,
    variant: str,
    ycol: str,
    ylabel: str,
    title_bits: str,
    *,
    storage_y_log: bool = False,
) -> plt.Figure | None:
    vs = ddf[ddf["variant"] == variant]
    if vs.empty or ycol not in vs.columns or "n_unique_keys" not in vs.columns:
        return None

    selected = _best_fixed_scheme_per_dim(vs, ycol)
    if selected.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, row in enumerate(selected.sort_values("dim").itertuples(index=False)):
        ms = vs[(vs["dim"] == row.dim) & (vs["method"] == row.method)].sort_values("n_unique_keys")
        if ms.empty:
            continue
        ax.plot(
            ms["n_unique_keys"],
            ms[ycol],
            marker="o",
            markersize=5,
            label=f"{int(row.dim)}D ({row.method})",
            color=cmap(i % 10),
        )

    ax.set_xlabel("N (unique keys)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if storage_y_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(
        fontsize=8,
        ncol=1,
        title="Chosen fixed scheme",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.suptitle(title_bits, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return fig


def plot_storage_sweep(
    df: pd.DataFrame,
    output_dir: str,
    show: bool,
    *,
    sweep_axes: list[str],
    storage_y_log: bool = False,
    min_key_levels: int = DEFAULT_MIN_KEY_LEVELS,
) -> None:
    if "normalized_bpk" not in df.columns:
        print("  [skip] storage_sweep: missing normalized_bpk (need bytes_per_key, dim, dtype)", file=sys.stderr)
        return
    for x_kind in sweep_axes:
        for dtype_tag, ddf in _per_dtype_plot_dfs(df):
            ddf = _filter_storage_relevant_rows(ddf, min_key_levels)
            if ddf.empty:
                continue
            for variant in ("CD", "PC"):
                title_extra = f" ({dtype_tag})" if dtype_tag else ""
                title = f"Normalized BPK, {variant}{title_extra} — x={x_kind}"
                fig = _plot_sweep_facets(
                    ddf,
                    variant,
                    x_kind,
                    "normalized_bpk",
                    "Normalized BPK",
                    title,
                    storage_y_log=storage_y_log,
                    time_log_y=False,
                )
                if fig is None:
                    continue
                stem = _plot_filename_stem(
                    f"storage_sweep_{_xaxis_file_token(x_kind)}_{variant.lower()}",
                    dtype_tag,
                )
                path = os.path.join(output_dir, f"{stem}.png")
                fig.savefig(path, dpi=200)
                print(f"Saved {path}")
                if show:
                    plt.show()
                plt.close(fig)

                if x_kind == "n":
                    best_title = f"Normalized BPK, best fixed {variant} per dimension{title_extra} — x=n"
                    best_fig = _plot_best_fixed_scheme_by_dim(
                        ddf,
                        variant,
                        "normalized_bpk",
                        "Normalized BPK",
                        best_title,
                        storage_y_log=storage_y_log,
                    )
                    if best_fig is not None:
                        best_stem = _plot_filename_stem(
                            f"storage_sweep_{_xaxis_file_token(x_kind)}_best_{variant.lower()}",
                            dtype_tag,
                        )
                        best_path = os.path.join(output_dir, f"{best_stem}.png")
                        best_fig.savefig(best_path, dpi=200, bbox_inches="tight")
                        print(f"Saved {best_path}")
                        if show:
                            plt.show()
                        plt.close(best_fig)


def plot_time_sweep(
    df: pd.DataFrame,
    output_dir: str,
    show: bool,
    *,
    sweep_axes: list[str],
    time_metric: str = "total_build_s",
) -> None:
    if time_metric not in TIME_METRIC_CHOICES:
        print(f"  [skip] time_sweep: bad time metric {time_metric!r}", file=sys.stderr)
        return
    if time_metric not in df.columns:
        print(f"  [skip] time_sweep: column {time_metric} missing", file=sys.stderr)
        return
    ylabel = {"total_build_s": "Total build (s)", "insert_s": "Insert (s)", "finish_s": "Finish (s)"}[
        time_metric
    ]
    for x_kind in sweep_axes:
        for dtype_tag, ddf in _per_dtype_plot_dfs(df):
            if ddf.empty:
                continue
            for variant in ("CD", "PC"):
                title_extra = f" ({dtype_tag})" if dtype_tag else ""
                title = f"{ylabel}, {variant}{title_extra} — x={x_kind}"
                fig = _plot_sweep_facets(
                    ddf,
                    variant,
                    x_kind,
                    time_metric,
                    ylabel,
                    title,
                    storage_y_log=False,
                    time_log_y=True,
                )
                if fig is None:
                    continue
                stem = _plot_filename_stem(
                    f"time_sweep_{_xaxis_file_token(x_kind)}_{variant.lower()}_{time_metric}",
                    dtype_tag,
                )
                path = os.path.join(output_dir, f"{stem}.png")
                fig.savefig(path, dpi=200)
                print(f"Saved {path}")
                if show:
                    plt.show()
                plt.close(fig)


def _suffix_ready_df(df: pd.DataFrame, min_key_levels: int = DEFAULT_MIN_KEY_LEVELS) -> pd.DataFrame:
    out = df.copy()
    if "post_suffix_normalized_bpk" not in out.columns and "normalized_bpk" in out.columns:
        out["post_suffix_normalized_bpk"] = out["normalized_bpk"]
    if "suffix_collapse_saving_pct" not in out.columns and {
        "pre_suffix_bytes_per_key",
        "bytes_per_key",
    }.issubset(out.columns):
        pre = out["pre_suffix_bytes_per_key"].astype(float).replace(0, np.nan)
        out["suffix_collapse_saving_pct"] = 1.0 - out["bytes_per_key"].astype(float) / pre
    need = {"pre_suffix_normalized_bpk", "post_suffix_normalized_bpk", "suffix_collapse_saving_pct"}
    if not need.issubset(out.columns):
        return out.iloc[0:0].copy()
    out = out[out["method"].astype(str).str.startswith("CD-")].copy()
    out = out[out["pre_suffix_normalized_bpk"].astype(float) > 0.0]
    out = _filter_storage_relevant_rows(out, min_key_levels)
    return out


def _plot_suffix_comparison(
    ddf: pd.DataFrame,
    x_kind: str,
    title_bits: str,
    *,
    min_key_levels: int = DEFAULT_MIN_KEY_LEVELS,
) -> plt.Figure | None:
    vs = _suffix_ready_df(ddf, min_key_levels=min_key_levels)
    if vs.empty:
        return None
    methods = _methods_for_variant(vs, "CD")
    if not methods:
        return None

    if x_kind == "n":
        xcol = "n_unique_keys"
        dims = sorted(vs["dim"].unique())
        cols = min(len(dims), 3)
        rows = (len(dims) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        for idx, dim in enumerate(dims):
            ax = axes[idx // cols][idx % cols]
            sub = vs[vs["dim"] == dim]
            cmap = plt.get_cmap("tab10")
            for i, method in enumerate(methods):
                ms = sub[sub["method"] == method].sort_values(xcol)
                if ms.empty:
                    continue
                color = cmap(i % 10)
                ax.plot(ms[xcol], ms["pre_suffix_normalized_bpk"], ls="--", marker="o",
                        markersize=4, color=color, label=f"{method} before")
                ax.plot(ms[xcol], ms["post_suffix_normalized_bpk"], ls="-", marker="o",
                        markersize=4, color=color, label=f"{method} after")
            ax.set_xlabel("N (unique keys)")
            ax.set_ylabel("Normalized BPK")
            ax.set_title(f"dim={dim}")
            ax.set_xscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        for idx in range(len(dims), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

    elif x_kind == "dim":
        sub_all, max_n, n_all_dims = _slice_max_n(vs)
        if sub_all.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for i, method in enumerate(methods):
            ms = sub_all[sub_all["method"] == method].sort_values("dim")
            if ms.empty:
                continue
            color = cmap(i % 10)
            ax.plot(ms["dim"], ms["pre_suffix_normalized_bpk"], ls="--", marker="o",
                    markersize=5, color=color, label=f"{method} before")
            ax.plot(ms["dim"], ms["post_suffix_normalized_bpk"], ls="-", marker="o",
                    markersize=5, color=color, label=f"{method} after")
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Normalized BPK")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub_all["dim"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        n_note = "largest N with all dimensions" if n_all_dims else "mixed N (no single N for all dims)"
        ax.set_title(f"N={max_n:,} ({n_note})")
        title_bits = f"{title_bits} (N={max_n:,})"

    elif x_kind == "group_bits":
        plot_vs, max_n, _ = _slice_max_n(vs)
        if plot_vs.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for i, dim in enumerate(sorted(plot_vs["dim"].unique())):
            ms = plot_vs[plot_vs["dim"] == dim].sort_values("group_bits")
            if ms.empty:
                continue
            color = cmap(i % 10)
            ax.plot(ms["group_bits"], ms["pre_suffix_normalized_bpk"], ls="--", marker="o",
                    markersize=5, color=color, label=f"{int(dim)}D before")
            ax.plot(ms["group_bits"], ms["post_suffix_normalized_bpk"], ls="-", marker="o",
                    markersize=5, color=color, label=f"{int(dim)}D after")
        ax.set_ylabel("Normalized BPK")
        _configure_group_bits_xaxis(ax, plot_vs["group_bits"])
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2, title="Dimension")
        title_bits = f"{title_bits} (N={max_n:,})"
    else:
        return None

    fig.suptitle(title_bits, fontsize=14)
    fig.tight_layout()
    return fig


def _plot_suffix_savings(
    ddf: pd.DataFrame,
    x_kind: str,
    title_bits: str,
    *,
    min_key_levels: int = DEFAULT_MIN_KEY_LEVELS,
) -> plt.Figure | None:
    vs = _suffix_ready_df(ddf, min_key_levels=min_key_levels)
    if vs.empty:
        return None
    methods = _methods_for_variant(vs, "CD")
    if not methods:
        return None
    vs = vs.copy()
    vs["suffix_collapse_saving_pct_plot"] = vs["suffix_collapse_saving_pct"].astype(float) * 100.0

    if x_kind == "n":
        xcol = "n_unique_keys"
        dims = sorted(vs["dim"].unique())
        cols = min(len(dims), 3)
        rows = (len(dims) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        for idx, dim in enumerate(dims):
            ax = axes[idx // cols][idx % cols]
            sub = vs[vs["dim"] == dim]
            for method in methods:
                ms = sub[sub["method"] == method].sort_values(xcol)
                if ms.empty:
                    continue
                ax.plot(ms[xcol], ms["suffix_collapse_saving_pct_plot"], marker="o",
                        markersize=4, label=method)
            ax.set_xlabel("N (unique keys)")
            ax.set_ylabel("Saved by suffix collapse (%)")
            ax.set_title(f"dim={dim}")
            ax.set_xscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        for idx in range(len(dims), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

    elif x_kind == "dim":
        sub_all, max_n, n_all_dims = _slice_max_n(vs)
        if sub_all.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in methods:
            ms = sub_all[sub_all["method"] == method].sort_values("dim")
            if ms.empty:
                continue
            ax.plot(ms["dim"], ms["suffix_collapse_saving_pct_plot"], marker="o",
                    markersize=5, label=method)
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Saved by suffix collapse (%)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub_all["dim"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        n_note = "largest N with all dimensions" if n_all_dims else "mixed N (no single N for all dims)"
        ax.set_title(f"N={max_n:,} ({n_note})")
        title_bits = f"{title_bits} (N={max_n:,})"

    elif x_kind == "group_bits":
        plot_vs, max_n, _ = _slice_max_n(vs)
        if plot_vs.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for i, dim in enumerate(sorted(plot_vs["dim"].unique())):
            ms = plot_vs[plot_vs["dim"] == dim].sort_values("group_bits")
            if ms.empty:
                continue
            ax.plot(ms["group_bits"], ms["suffix_collapse_saving_pct_plot"], marker="o",
                    markersize=5, label=f"{int(dim)}D", color=cmap(i % 10))
        ax.set_ylabel("Saved by suffix collapse (%)")
        _configure_group_bits_xaxis(ax, plot_vs["group_bits"])
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2, title="Dimension")
        title_bits = f"{title_bits} (N={max_n:,})"
    else:
        return None

    fig.suptitle(title_bits, fontsize=14)
    fig.tight_layout()
    return fig


def plot_suffix_collapse(
    df: pd.DataFrame,
    output_dir: str,
    show: bool,
    *,
    sweep_axes: list[str],
    min_key_levels: int = DEFAULT_MIN_KEY_LEVELS,
) -> None:
    for x_kind in sweep_axes:
        for dtype_tag, ddf in _per_dtype_plot_dfs(df):
            ready = _suffix_ready_df(ddf, min_key_levels=min_key_levels)
            if ready.empty:
                print(
                    "  [skip] suffix_collapse: missing CD suffix-collapse columns "
                    "or no populated CD rows",
                    file=sys.stderr,
                )
                continue
            title_extra = f" ({dtype_tag})" if dtype_tag else ""
            title = f"Before vs after suffix collapse, CD{title_extra} — x={x_kind}"
            fig = _plot_suffix_comparison(ready, x_kind, title, min_key_levels=min_key_levels)
            if fig is not None:
                stem = _plot_filename_stem(f"suffix_collapse_{_xaxis_file_token(x_kind)}_cd", dtype_tag)
                path = os.path.join(output_dir, f"{stem}.png")
                fig.savefig(path, dpi=200)
                print(f"Saved {path}")
                if show:
                    plt.show()
                plt.close(fig)

            savings_title = f"Suffix-collapse storage savings, CD{title_extra} — x={x_kind}"
            fig = _plot_suffix_savings(ready, x_kind, savings_title, min_key_levels=min_key_levels)
            if fig is not None:
                stem = _plot_filename_stem(f"suffix_collapse_savings_{_xaxis_file_token(x_kind)}_cd", dtype_tag)
                path = os.path.join(output_dir, f"{stem}.png")
                fig.savefig(path, dpi=200)
                print(f"Saved {path}")
                if show:
                    plt.show()
                plt.close(fig)


# ---------------------------------------------------------------------------
# Table export: Markdown only, formatting aligned with bench_common.h /
# bench_dawg_storage.cpp (comma_fmt, size_fmt, bpk_fmt, time_fmt).
# ---------------------------------------------------------------------------


def _comma_fmt_storage(n: float | int | None) -> str:
    if n is None or pd.isna(n):
        return "--"
    x = int(round(float(n)))
    return f"{x:,}"


def _size_fmt_storage(bytes_val: float | int | None) -> str:
    if bytes_val is None or pd.isna(bytes_val):
        return "--"
    b = int(round(float(bytes_val)))
    kb = 1024
    mb = kb * kb
    gb = mb * kb
    if b < kb:
        return f"{b} B"
    if b < mb:
        return f"{b / kb:.1f} KB"
    if b < gb:
        return f"{b / float(mb):.2f} MB"
    return f"{b / float(gb):.2f} GB"


def _bpk_fmt_storage(b: float | None) -> str:
    if b is None or pd.isna(b):
        return "--"
    x = float(b)
    if x < 1024.0:
        return f"{x:.1f} B"
    if x < 1024.0 * 1024.0:
        return f"{x / 1024.0:.1f} KB"
    return f"{x / (1024.0 * 1024.0):.2f} MB"


def _norm_bpk_fmt(v: object) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.4f}"


def _time_fmt_storage(s: float | None) -> str:
    if s is None or pd.isna(s):
        return "--"
    x = float(s)
    if x < 10.0:
        return f"{x:.4f}"
    if x < 100.0:
        return f"{x:.2f}"
    return f"{x:.1f}"


def _ratio_fmt(v: object) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.4f}x"


def _delta_fmt(v: object) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):+.4f}"


def _md_escape_cell(s: str) -> str:
    return str(s).replace("|", "\\|")


def _markdown_table_md(headers: list[str], rows: list[list[str]]) -> str:
    h = "| " + " | ".join(_md_escape_cell(x) for x in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_md_escape_cell(x) for x in row) + " |" for row in rows]
    return "\n".join([h, sep] + body)


def _pivot_df_to_table_md(
    pivot: pd.DataFrame,
    row_label: str,
    fmt_cell: Callable[[object], str],
    col_labels: list[str] | None = None,
) -> str:
    if pivot.empty:
        return ""
    if col_labels is None:
        col_labels = [str(c) for c in pivot.columns]
    headers = [row_label] + col_labels
    rows: list[list[str]] = []
    for idx, row in pivot.iterrows():
        cells = [str(idx)]
        for c in pivot.columns:
            cells.append(fmt_cell(row[c]))
        rows.append(cells)
    return _markdown_table_md(headers, rows) + "\n\n"


def _method_sort_key(method: object) -> tuple[int, str]:
    s = str(method)
    if "-" in s:
        pref, _, rest = s.partition("-")
        try:
            bits = int(rest)
            group = 0 if pref == "CD" else 1
            return (group, bits, s)
        except ValueError:
            pass
    return (2_000_000_000, s)


def _pivot_method_by_n(sub: pd.DataFrame, value_col: str) -> pd.DataFrame | None:
    if sub.empty or value_col not in sub.columns:
        return None
    pt = sub.pivot_table(
        index="method", columns="n_unique_keys", values=value_col, aggfunc="first"
    )
    if pt.empty:
        return None
    pt = pt.reindex(columns=sorted(int(c) for c in pt.columns))
    methods = sorted(pt.index, key=_method_sort_key)
    pt = pt.reindex(index=methods)
    return pt


def _pivot_first(sub: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame | None:
    if sub.empty or not all(c in sub.columns for c in (index, columns, values)):
        return None
    pt = sub.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    return None if pt.empty else pt


def _slice_max_n_complete_by_dtype_dim(ddf: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Pick one N such that every (dtype, dim) pair is present."""
    if ddf.empty or not {"dtype", "dim", "n_unique_keys"}.issubset(ddf.columns):
        return ddf.iloc[0:0].copy(), 0, False
    dims = sorted({int(d) for d in ddf["dim"].dropna().unique()})
    dtypes = sorted(str(dt) for dt in ddf["dtype"].dropna().unique())
    if not dims or not dtypes:
        return ddf.iloc[0:0].copy(), 0, False
    n_sorted = sorted({int(x) for x in ddf["n_unique_keys"].dropna().unique()}, reverse=True)
    for n in n_sorted:
        sub = ddf[ddf["n_unique_keys"] == n]
        if all(not sub[(sub["dtype"] == dt) & (sub["dim"] == dim)].empty for dt in dtypes for dim in dims):
            return sub.reset_index(drop=True), n, True
    max_n = int(ddf["n_unique_keys"].max())
    sub = ddf[ddf["n_unique_keys"] == max_n].reset_index(drop=True)
    return sub, max_n, False


def _write_graph_md(filename: str, title: str, blurb: str, body_parts: list[str], output_dir: str) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{blurb}\n\n")
        f.write("".join(body_parts))
    print(f"Saved {path}")


def _export_dtype_comparison_tables(df: pd.DataFrame, output_dir: str) -> None:
    """Write direct float16 vs float32 normalized-BPK comparison tables when both are present."""
    if "dtype" not in df.columns:
        return
    dtypes = {str(x) for x in df["dtype"].dropna().unique()}
    if not {"float16", "float32"}.issubset(dtypes):
        return
    need = {"dim", "method", "normalized_bpk", "variant", "n_unique_keys"}
    if not need.issubset(df.columns):
        return

    for variant in ("CD", "PC"):
        vsub = df[df["variant"] == variant].copy()
        if vsub.empty:
            continue
        slice_df, max_n, n_complete = _slice_max_n_complete_by_dtype_dim(vsub)
        if slice_df.empty:
            continue

        parts: list[str] = []
        n_note = "all dtype × dim pairs" if n_complete else "mixed-N fallback"

        best = (
            slice_df.sort_values("normalized_bpk")
            .groupby(["dtype", "dim"], as_index=False)
            .first()[["dtype", "dim", "method", "group_bits", "normalized_bpk"]]
        )
        best32 = (
            best[best["dtype"] == "float32"]
            .rename(
                columns={
                    "method": "best_method_float32",
                    "group_bits": "best_group_bits_float32",
                    "normalized_bpk": "best_norm_bpk_float32",
                }
            )
            .drop(columns=["dtype"])
        )
        best16 = (
            best[best["dtype"] == "float16"]
            .rename(
                columns={
                    "method": "best_method_float16",
                    "group_bits": "best_group_bits_float16",
                    "normalized_bpk": "best_norm_bpk_float16",
                }
            )
            .drop(columns=["dtype"])
        )
        best_join = best32.merge(best16, on="dim", how="inner")
        if not best_join.empty:
            best_join["f16_over_f32_ratio"] = (
                best_join["best_norm_bpk_float16"] / best_join["best_norm_bpk_float32"]
            )
            best_join["f16_minus_f32_delta"] = (
                best_join["best_norm_bpk_float16"] - best_join["best_norm_bpk_float32"]
            )
            rows = []
            for _, row in best_join.sort_values("dim").iterrows():
                rows.append(
                    [
                        str(int(row["dim"])),
                        str(row["best_method_float32"]),
                        _comma_fmt_storage(row["best_group_bits_float32"]),
                        _norm_bpk_fmt(row["best_norm_bpk_float32"]),
                        str(row["best_method_float16"]),
                        _comma_fmt_storage(row["best_group_bits_float16"]),
                        _norm_bpk_fmt(row["best_norm_bpk_float16"]),
                        _ratio_fmt(row["f16_over_f32_ratio"]),
                        _delta_fmt(row["f16_minus_f32_delta"]),
                    ]
                )
            parts.append(
                f"## Best Norm BPK By Dimension\n\n"
                f"N = {_comma_fmt_storage(max_n)} ({n_note}). Each dtype picks its own best "
                f"`GROUP_BITS` / method for that dimension.\n\n"
                + _markdown_table_md(
                    [
                        "dim",
                        "Best float32",
                        "GB32",
                        "Norm32",
                        "Best float16",
                        "GB16",
                        "Norm16",
                        "f16/f32",
                        "Delta",
                    ],
                    rows,
                )
                + "\n\n"
            )

        same = slice_df.pivot_table(
            index="dim", columns=["method", "dtype"], values="normalized_bpk", aggfunc="first"
        )
        if not same.empty:
            ratio_cols = []
            delta_cols = []
            ratio_map: dict[str, pd.Series] = {}
            delta_map: dict[str, pd.Series] = {}
            methods = sorted({str(m) for m, _dt in same.columns}, key=_method_sort_key)
            for method in methods:
                key32 = (method, "float32")
                key16 = (method, "float16")
                if key32 not in same.columns or key16 not in same.columns:
                    continue
                ratio_map[method] = same[key16] / same[key32]
                delta_map[method] = same[key16] - same[key32]
                ratio_cols.append(method)
                delta_cols.append(method)
            if ratio_cols:
                ratio_df = pd.DataFrame(ratio_map).reindex(index=sorted(int(i) for i in same.index))
                delta_df = pd.DataFrame(delta_map).reindex(index=sorted(int(i) for i in same.index))
                parts.append(
                    "## Same Method Norm BPK Ratio\n\n"
                    "Float16 divided by float32 at the same dimension and method.\n\n"
                    + _pivot_df_to_table_md(ratio_df, "dim", _ratio_fmt, ratio_cols)
                )
                parts.append(
                    "## Same Method Norm BPK Delta\n\n"
                    "Float16 minus float32 at the same dimension and method.\n\n"
                    + _pivot_df_to_table_md(delta_df, "dim", _delta_fmt, delta_cols)
                )

        if parts:
            _write_graph_md(
                f"table_dtype_compare_norm_xdim_{variant.lower()}.md",
                f"Dtype comparison (dim) — {variant}",
                "Direct `float16` vs `float32` normalized-BPK comparison. "
                "Values above `1.0000x` in the ratio table mean `float16` uses more "
                "normalized bytes per key than `float32`.",
                parts,
                output_dir,
            )


def export_tables(df: pd.DataFrame, output_dir: str, sweep_axes: list[str] | None = None) -> None:
    """Write Markdown tables using bench_dawg_storage-style formatters."""
    os.makedirs(output_dir, exist_ok=True)
    if sweep_axes is None:
        sweep_axes = ["n", "dim", "group_bits"]

    df = df.copy()
    if "variant" not in df.columns:
        df = _annotate_variants(df)
    if "normalized_bpk" not in df.columns and all(c in df.columns for c in ("bytes_per_key", "dim")):
        df = add_key_metrics(df)

    md_intro = (
        "# CompactDawg sweep tables\n\n"
        "Formatting matches `tools/bench_dawg_storage.cpp` / `bench_common.h`: "
        "**Total size** (`size_fmt`), **B/Key** (`bpk_fmt`), **Edges** (`comma_fmt`), "
        "times (`time_fmt`). **Norm BPK** is `bytes_per_key / key_bytes` (Morton key size). "
        "`dawgdic` rows are excluded from this export.\n\n"
    )

    grouper = ["dtype", "n_unique_keys", "dim"]
    if not all(c in df.columns for c in grouper):
        print("Warning: export_tables skipped long sections (missing dtype/n_unique_keys/dim).", file=sys.stderr)
        sections: list[str] = []
    else:
        extra_cols: list[tuple[str, str]] = []
        if "morton_encode_s" in df.columns:
            extra_cols.append(("Morton(s)", "morton_encode_s"))
        if "sort_dedup_s" in df.columns:
            extra_cols.append(("Sort(s)", "sort_dedup_s"))
        suffix_cols: list[tuple[str, str]] = []
        if "pre_suffix_normalized_bpk" in df.columns:
            suffix_cols.append(("Pre-suffix Norm", "pre_suffix_normalized_bpk"))
        if "post_suffix_normalized_bpk" in df.columns:
            suffix_cols.append(("Post-suffix Norm", "post_suffix_normalized_bpk"))
        if "suffix_collapse_saving_pct" in df.columns:
            suffix_cols.append(("Suffix Save %", "suffix_collapse_saving_pct"))

        headers = (
            ["Method", "Total Size", "B/Key", "Norm BPK", "Key B", "Edges"]
            + [lbl for lbl, _ in suffix_cols]
            + [lbl for lbl, _ in extra_cols]
            + ["Insert(s)", "Finish(s)", "Total(s)"]
        )

        sections = []
        for dtype in sorted(df["dtype"].dropna().unique()):
            dsub = df[df["dtype"] == dtype]
            for n in sorted(dsub["n_unique_keys"].unique()):
                nsub = dsub[dsub["n_unique_keys"] == n]
                for dim in sorted(nsub["dim"].unique()):
                    sub = nsub[nsub["dim"] == dim].sort_values("method")
                    if sub.empty:
                        continue
                    rows_m: list[list[str]] = []
                    for _, r in sub.iterrows():
                        row = [
                            str(r["method"]),
                            _size_fmt_storage(r.get("total_bytes")),
                            _bpk_fmt_storage(r.get("bytes_per_key")),
                            _norm_bpk_fmt(r.get("normalized_bpk")),
                            _comma_fmt_storage(r.get("key_bytes")),
                            _comma_fmt_storage(r.get("edges")),
                        ]
                        for lbl, col in suffix_cols:
                            if lbl == "Suffix Save %":
                                v = r.get(col)
                                row.append("--" if pd.isna(v) or float(v) == 0.0 else f"{float(v) * 100.0:.2f}%")
                            else:
                                v = r.get(col)
                                row.append("--" if pd.isna(v) or float(v) == 0.0 else _norm_bpk_fmt(v))
                        for _, col in extra_cols:
                            row.append(_time_fmt_storage(r.get(col)))
                        row.extend(
                            [
                                _time_fmt_storage(r.get("insert_s")),
                                _time_fmt_storage(r.get("finish_s")),
                                _time_fmt_storage(r.get("total_build_s")),
                            ]
                        )
                        rows_m.append(row)
                    title = (
                        f"{dtype} — N = {_comma_fmt_storage(n)} unique keys — dim = {dim}"
                    )
                    sections.append(f"## {title}\n\n{_markdown_table_md(headers, rows_m)}\n\n")

    long_path = os.path.join(output_dir, "table_sweep_long.md")
    with open(long_path, "w", encoding="utf-8") as out:
        out.write(md_intro)
        out.write("\n".join(sections))
    print(f"Saved {long_path}")

    def fmt_bpk_cell(v: object) -> str:
        if pd.isna(v):
            return "--"
        return _bpk_fmt_storage(float(v))

    def fmt_time_cell(v: object) -> str:
        if pd.isna(v):
            return "--"
        return _time_fmt_storage(float(v))

    def _method_by_n_markdown(
        sub: pd.DataFrame, value_col: str, fmt_cell: Callable[[object], str]
    ) -> str:
        pt = _pivot_method_by_n(sub, value_col)
        if pt is None or pt.empty:
            return ""
        col_headers = [_comma_fmt_storage(int(c)) for c in pt.columns]
        rows_m = []
        for method, row in pt.iterrows():
            rows_m.append([str(method)] + [fmt_cell(row[c]) for c in pt.columns])
        return _markdown_table_md(["Method"] + col_headers, rows_m) + "\n\n"

    def _filter_variant(sub: pd.DataFrame, variant: str) -> pd.DataFrame:
        if "variant" not in sub.columns:
            return sub.iloc[0:0]
        return sub[sub["variant"] == variant]

    # --- Per (sweep_x, variant) storage + time tables -----------------------------
    for x_kind in sweep_axes:
        tok = _xaxis_file_token(x_kind)
        for variant in ("CD", "PC"):
            st_parts: list[str] = []
            st_best_parts: list[str] = []
            tm_parts: list[str] = []
            need = ("dtype", "dim", "method", "n_unique_keys", "bytes_per_key", "variant")
            if not all(c in df.columns for c in need):
                continue
            for dtype in sorted(df["dtype"].dropna().unique()):
                dsub = df[df["dtype"] == dtype]
                vsub = _filter_variant(dsub, variant)
                if vsub.empty:
                    continue
                m_list = _methods_for_variant(vsub, variant)
                if not m_list:
                    continue

                if x_kind == "n":
                    for dim in sorted(vsub["dim"].unique()):
                        block_bpk = _method_by_n_markdown(
                            vsub[vsub["dim"] == dim], "bytes_per_key", fmt_bpk_cell
                        )
                        block_norm = _method_by_n_markdown(
                            vsub[vsub["dim"] == dim], "normalized_bpk", _norm_bpk_fmt
                        )
                        if block_bpk:
                            st_parts.append(
                                f"## {dtype} — dim={dim}\n\n"
                                f"### B/Key (`bpk_fmt`)\n\n{block_bpk}"
                                f"### Norm BPK (÷ key bytes)\n\n{block_norm}"
                            )
                        if "total_build_s" in vsub.columns:
                            tblk = _method_by_n_markdown(
                                vsub[vsub["dim"] == dim], "total_build_s", fmt_time_cell
                            )
                            iblk = (
                                _method_by_n_markdown(vsub[vsub["dim"] == dim], "insert_s", fmt_time_cell)
                                if "insert_s" in vsub.columns
                                else ""
                            )
                            fblk = (
                                _method_by_n_markdown(vsub[vsub["dim"] == dim], "finish_s", fmt_time_cell)
                                if "finish_s" in vsub.columns
                                else ""
                            )
                            if tblk:
                                tm_parts.append(
                                    f"## {dtype} — dim={dim}\n\n"
                                    f"### total_build_s\n\n{tblk}"
                                    + (f"### insert_s\n\n{iblk}" if iblk else "")
                                    + (f"### finish_s\n\n{fblk}" if fblk else "")
                                )

                    selected = _best_fixed_scheme_per_dim(vsub, "normalized_bpk")
                    if not selected.empty:
                        summary_rows: list[list[str]] = []
                        sel_rows: list[pd.DataFrame] = []
                        for row in selected.sort_values("dim").itertuples(index=False):
                            summary_rows.append(
                                [
                                    str(int(row.dim)),
                                    str(row.method),
                                    _norm_bpk_fmt(row.mean_value),
                                    _comma_fmt_storage(row.max_n),
                                    _norm_bpk_fmt(row.value_at_max_n),
                                ]
                            )
                            ms = vsub[(vsub["dim"] == row.dim) & (vsub["method"] == row.method)].copy()
                            if ms.empty:
                                continue
                            ms["dim_method"] = f"{int(row.dim)}D ({row.method})"
                            sel_rows.append(ms)
                        if sel_rows:
                            sel_df = pd.concat(sel_rows, ignore_index=True)
                            pivot_best_b = _pivot_first(sel_df, "dim_method", "n_unique_keys", "bytes_per_key")
                            pivot_best_n = _pivot_first(sel_df, "dim_method", "n_unique_keys", "normalized_bpk")
                            if pivot_best_b is not None and not pivot_best_b.empty:
                                pivot_best_b = pivot_best_b.reindex(
                                    index=[f"{int(r.dim)}D ({r.method})" for r in selected.sort_values("dim").itertuples(index=False)]
                                )
                                pivot_best_b = pivot_best_b.reindex(
                                    columns=sorted(int(c) for c in pivot_best_b.columns)
                                )
                            if pivot_best_n is not None and not pivot_best_n.empty:
                                pivot_best_n = pivot_best_n.reindex(
                                    index=[f"{int(r.dim)}D ({r.method})" for r in selected.sort_values("dim").itertuples(index=False)]
                                )
                                pivot_best_n = pivot_best_n.reindex(
                                    columns=sorted(int(c) for c in pivot_best_n.columns)
                                )
                            st_best_parts.append(
                                f"## {dtype}\n\n"
                                "### Chosen Fixed Scheme Per Dimension\n\n"
                                "Selection rule: lowest mean normalized BPK across available `N`; "
                                "ties break on the normalized BPK at the largest available `N`, "
                                "then method order.\n\n"
                                + _markdown_table_md(
                                    ["dim", "Best Method", "Mean Norm BPK", "Largest N", "Norm @ Largest N"],
                                    summary_rows,
                                )
                                + "\n\n"
                                + (
                                    "### B/Key (`bpk_fmt`)\n\n"
                                    + _pivot_df_to_table_md(
                                        pivot_best_b,
                                        "dim (method)",
                                        fmt_bpk_cell,
                                        [_comma_fmt_storage(int(c)) for c in pivot_best_b.columns],
                                    )
                                    if pivot_best_b is not None and not pivot_best_b.empty
                                    else ""
                                )
                                + (
                                    "### Norm BPK (÷ key bytes)\n\n"
                                    + _pivot_df_to_table_md(
                                        pivot_best_n,
                                        "dim (method)",
                                        _norm_bpk_fmt,
                                        [_comma_fmt_storage(int(c)) for c in pivot_best_n.columns],
                                    )
                                    if pivot_best_n is not None and not pivot_best_n.empty
                                    else ""
                                )
                            )

                elif x_kind == "dim":
                    slice_df, max_n, n_all_dims = _slice_max_n(vsub)
                    if slice_df.empty:
                        continue
                    pivot_b = slice_df.pivot_table(
                        index="dim", columns="method", values="bytes_per_key", aggfunc="first"
                    )
                    pivot_n = slice_df.pivot_table(
                        index="dim", columns="method", values="normalized_bpk", aggfunc="first"
                    )
                    pivot_b = pivot_b.reindex(index=sorted(int(i) for i in pivot_b.index))
                    pivot_n = pivot_n.reindex(index=sorted(int(i) for i in pivot_n.index))
                    m_order = [m for m in sorted(pivot_b.columns, key=_method_sort_key) if m in m_list]
                    pivot_b = pivot_b.reindex(columns=m_order)
                    pivot_n = pivot_n.reindex(columns=m_order)
                    if not pivot_b.empty:
                        n_note = "all dimensions" if n_all_dims else "mixed-N fallback"
                        st_parts.append(
                            f"## {dtype} — N = {_comma_fmt_storage(max_n)} ({n_note})\n\n"
                            f"### B/Key\n\n"
                            + _pivot_df_to_table_md(pivot_b, "dim", fmt_bpk_cell, [str(c) for c in pivot_b.columns])
                            + f"### Norm BPK\n\n"
                            + _pivot_df_to_table_md(
                                pivot_n, "dim", _norm_bpk_fmt, [str(c) for c in pivot_n.columns]
                            )
                        )
                    if "total_build_s" in slice_df.columns:
                        dim_time_chunks: list[str] = []
                        pivot_t = slice_df.pivot_table(
                            index="dim", columns="method", values="total_build_s", aggfunc="first"
                        )
                        pivot_t = pivot_t.reindex(index=sorted(int(i) for i in pivot_t.index))
                        pivot_t = pivot_t.reindex(columns=[m for m in m_order if m in pivot_t.columns])
                        if not pivot_t.empty:
                            dim_time_chunks.append(
                                "### total_build_s\n\n"
                                + _pivot_df_to_table_md(
                                    pivot_t, "dim", fmt_time_cell, [str(c) for c in pivot_t.columns]
                                )
                            )
                        if "insert_s" in slice_df.columns:
                            pi = slice_df.pivot_table(
                                index="dim", columns="method", values="insert_s", aggfunc="first"
                            )
                            pi = pi.reindex(index=sorted(int(i) for i in pi.index))
                            pi = pi.reindex(columns=[m for m in m_order if m in pi.columns])
                            if not pi.empty:
                                dim_time_chunks.append(
                                    "### insert_s\n\n"
                                    + _pivot_df_to_table_md(
                                        pi, "dim", fmt_time_cell, [str(c) for c in pi.columns]
                                    )
                                )
                        if "finish_s" in slice_df.columns:
                            pf = slice_df.pivot_table(
                                index="dim", columns="method", values="finish_s", aggfunc="first"
                            )
                            pf = pf.reindex(index=sorted(int(i) for i in pf.index))
                            pf = pf.reindex(columns=[m for m in m_order if m in pf.columns])
                            if not pf.empty:
                                dim_time_chunks.append(
                                    "### finish_s\n\n"
                                    + _pivot_df_to_table_md(
                                        pf, "dim", fmt_time_cell, [str(c) for c in pf.columns]
                                    )
                                )
                        if dim_time_chunks:
                            tm_parts.append(
                                f"## {dtype} — N = {_comma_fmt_storage(max_n)}\n\n"
                                + "".join(dim_time_chunks)
                            )

                elif x_kind == "group_bits":
                    for dim in sorted(vsub["dim"].unique()):
                        dvs = vsub[vsub["dim"] == dim]
                        for method in m_list:
                            ms = dvs[dvs["method"] == method]
                            if ms.empty:
                                continue
                            pivot_b = _pivot_first(ms, "group_bits", "n_unique_keys", "bytes_per_key")
                            pivot_n = _pivot_first(ms, "group_bits", "n_unique_keys", "normalized_bpk")
                            if pivot_b is None or pivot_b.empty:
                                continue
                            pivot_b = pivot_b.reindex(index=sorted(int(i) for i in pivot_b.index))
                            pivot_b = pivot_b.reindex(columns=sorted(int(c) for c in pivot_b.columns))
                            col_lbl = [_comma_fmt_storage(int(c)) for c in pivot_b.columns]
                            st_parts.append(
                                f"## {dtype} — dim={dim} — {method}\n\n"
                                f"### B/Key (rows=GROUP_BITS, cols=N)\n\n"
                                + _pivot_df_to_table_md(pivot_b, "group_bits", fmt_bpk_cell, col_lbl)
                            )
                            if pivot_n is not None and not pivot_n.empty:
                                pivot_n = pivot_n.reindex(index=sorted(int(i) for i in pivot_n.index))
                                pivot_n = pivot_n.reindex(columns=sorted(int(c) for c in pivot_n.columns))
                                st_parts.append(
                                    f"### Norm BPK\n\n"
                                    + _pivot_df_to_table_md(
                                        pivot_n, "group_bits", _norm_bpk_fmt, col_lbl
                                    )
                                )
                            if "total_build_s" in ms.columns:
                                pivot_t = _pivot_first(ms, "group_bits", "n_unique_keys", "total_build_s")
                                if pivot_t is not None and not pivot_t.empty:
                                    pivot_t = pivot_t.reindex(index=sorted(int(i) for i in pivot_t.index))
                                    pivot_t = pivot_t.reindex(columns=sorted(int(c) for c in pivot_t.columns))
                                    tm_parts.append(
                                        f"## {dtype} — dim={dim} — {method}\n\n"
                                        f"### total_build_s\n\n"
                                        + _pivot_df_to_table_md(
                                            pivot_t,
                                            "group_bits",
                                            fmt_time_cell,
                                            [_comma_fmt_storage(int(c)) for c in pivot_t.columns],
                                        )
                                    )
                                if "insert_s" in ms.columns:
                                    pi = _pivot_first(ms, "group_bits", "n_unique_keys", "insert_s")
                                    if pi is not None and not pi.empty:
                                        pi = pi.reindex(index=sorted(int(i) for i in pi.index))
                                        pi = pi.reindex(columns=sorted(int(c) for c in pi.columns))
                                        tm_parts.append(
                                            "### insert_s\n\n"
                                            + _pivot_df_to_table_md(
                                                pi,
                                                "group_bits",
                                                fmt_time_cell,
                                                [_comma_fmt_storage(int(c)) for c in pi.columns],
                                            )
                                        )
                                if "finish_s" in ms.columns:
                                    pf = _pivot_first(ms, "group_bits", "n_unique_keys", "finish_s")
                                    if pf is not None and not pf.empty:
                                        pf = pf.reindex(index=sorted(int(i) for i in pf.index))
                                        pf = pf.reindex(columns=sorted(int(c) for c in pf.columns))
                                        tm_parts.append(
                                            "### finish_s\n\n"
                                            + _pivot_df_to_table_md(
                                                pf,
                                                "group_bits",
                                                fmt_time_cell,
                                                [_comma_fmt_storage(int(c)) for c in pf.columns],
                                            )
                                        )

            if st_parts:
                _write_graph_md(
                    f"table_storage_sweep_{tok}_{variant.lower()}.md",
                    f"Storage sweep ({x_kind}) — {variant}",
                    f"Companion to **`storage_sweep_{tok}_{variant.lower()}.png`** "
                    f"(and dtype-suffixed variants). Raw **B/Key** and **Norm BPK**.",
                    st_parts,
                    output_dir,
                )
            if st_best_parts:
                _write_graph_md(
                    f"table_storage_sweep_{tok}_best_{variant.lower()}.md",
                    f"Storage sweep ({x_kind}, best fixed scheme) — {variant}",
                    f"Companion to **`storage_sweep_{tok}_best_{variant.lower()}.png`** "
                    f"(and dtype-suffixed variants). Keeps one fixed method per dimension: "
                    f"the method with the lowest mean **Norm BPK** across the plotted `N` values.",
                    st_best_parts,
                    output_dir,
                )
            if tm_parts:
                _write_graph_md(
                    f"table_time_sweep_{tok}_{variant.lower()}.md",
                    f"Build time sweep ({x_kind}) — {variant}",
                    f"Companion to **`time_sweep_{tok}_{variant.lower()}_*.png`**. "
                    "**total_build_s** / **insert_s** / **finish_s** (`time_fmt`).",
                    tm_parts,
                    output_dir,
                )

    _export_dtype_comparison_tables(df, output_dir)


PLOT_FUNCS = {
    "storage_sweep": plot_storage_sweep,
    "time_sweep": plot_time_sweep,
    "suffix_collapse": plot_suffix_collapse,
}


def main():
    parser = argparse.ArgumentParser(description="Plot CompactDawg sweep benchmark results")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="One or more sweep CSV paths (concatenated; overlapping keys use last row)",
    )
    parser.add_argument("--output-dir", "-o", default="plots/sweep", help="Output directory")
    parser.add_argument(
        "--plot-type",
        "-p",
        default="all",
        help=f"Plot type: {','.join(PLOT_TYPES[:-1])},none or 'all' (default: all); "
        "use none with --export-tables to skip figures",
    )
    parser.add_argument(
        "--sweep-x-axis",
        default="n",
        help="Comma-separated sweep axes for storage_sweep/time_sweep and table exports: "
        "n, dim, group_bits (alias: gb). Default: n only. Example: n,dim,group_bits",
    )
    parser.add_argument(
        "--storage-y-log",
        action="store_true",
        help="Use log scale for Normalized BPK (storage_sweep) y-axis",
    )
    parser.add_argument(
        "--time-metric",
        default="total_build_s",
        choices=list(TIME_METRIC_CHOICES),
        help="Y column for time_sweep plots (default: total_build_s)",
    )
    parser.add_argument(
        "--min-key-levels",
        type=int,
        default=DEFAULT_MIN_KEY_LEVELS,
        help="For storage_sweep and suffix_collapse only, drop rows where "
        "(dim x bits_per_coord) / GROUP_BITS is below this threshold. "
        "Default: 3. Use 0 to disable.",
    )
    parser.add_argument("--filter-dim", default=None, help="Comma-separated dimensions to include")
    parser.add_argument("--filter-n", default=None, help="Comma-separated N values to include")
    parser.add_argument("--filter-gb", default=None, help="Comma-separated GROUP_BITS to include")
    parser.add_argument(
        "--filter-dtype",
        default=None,
        help="Restrict matplotlib plots to this dtype (float32 or float16). "
        "Does not apply to --export-tables (tables include every dtype after dim/n/gb filters "
        "unless you also pass --export-tables-dtype).",
    )
    parser.add_argument(
        "--export-tables-dtype",
        default=None,
        help="If set, restrict Markdown table export to this dtype only. "
        "Default: export all dtypes present after --filter-dim/--filter-n/--filter-gb.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not drop duplicate (dim,dtype,n,gb,method) rows when merging CSVs",
    )
    parser.add_argument(
        "--export-tables",
        action="store_true",
        help="Write Markdown: table_sweep_long.md plus table_storage_sweep_<x>_<variant>.md "
        "and table_time_sweep_<x>_<variant>.md per sweep axis (see --sweep-x-axis). "
        "When both float32 and float16 are present, also writes "
        "table_dtype_compare_norm_xdim_<variant>.md.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    for p in args.input:
        if not os.path.exists(p):
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    df = load_csv(args.input, dedupe=not args.no_dedupe)
    if df.empty:
        print("Error: CSV is empty", file=sys.stderr)
        sys.exit(1)

    filter_dims = [int(x) for x in args.filter_dim.split(",")] if args.filter_dim else None
    filter_n = [int(x) for x in args.filter_n.split(",")] if args.filter_n else None
    filter_gb = [int(x) for x in args.filter_gb.split(",")] if args.filter_gb else None

    df_common = filter_df(df, filter_dims, filter_n, filter_gb, None)
    if df_common.empty:
        print("Error: no data after dim/n/gb filters", file=sys.stderr)
        sys.exit(1)

    df_plots = (
        filter_df(df_common.copy(), None, None, None, args.filter_dtype)
        if args.filter_dtype
        else df_common
    )

    df_tables = (
        filter_df(df_common.copy(), None, None, None, args.export_tables_dtype)
        if args.export_tables_dtype
        else df_common
    )
    if df_tables.empty:
        print("Error: no data for table export after --export-tables-dtype", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    sweep_axes_all = _parse_sweep_axes(args.sweep_x_axis)

    if args.plot_type == "all":
        types_to_plot = [t for t in PLOT_TYPES if t != "none"]
    else:
        types_to_plot = [t.strip() for t in args.plot_type.split(",")]

    will_plot = any(pt in PLOT_FUNCS for pt in types_to_plot if pt != "none")
    if will_plot and df_plots.empty:
        print("Error: no data for plots after --filter-dtype", file=sys.stderr)
        sys.exit(1)

    any_plot = False
    for pt in types_to_plot:
        if pt == "none":
            continue
        if pt not in PLOT_FUNCS:
            print(f"Warning: unknown plot type '{pt}', skipping")
            continue
        print(f"Generating {pt}...")
        if pt == "storage_sweep":
            plot_storage_sweep(
                df_plots,
                args.output_dir,
                args.show,
                sweep_axes=sweep_axes_all,
                storage_y_log=args.storage_y_log,
                min_key_levels=args.min_key_levels,
            )
        elif pt == "time_sweep":
            plot_time_sweep(
                df_plots,
                args.output_dir,
                args.show,
                sweep_axes=sweep_axes_all,
                time_metric=args.time_metric,
            )
        elif pt == "suffix_collapse":
            plot_suffix_collapse(
                df_plots,
                args.output_dir,
                args.show,
                sweep_axes=sweep_axes_all,
                min_key_levels=args.min_key_levels,
            )
        any_plot = True

    if not any_plot and not args.export_tables:
        print(
            "Error: nothing to do (use a plot type other than 'none', or add --export-tables).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.export_tables:
        print("Exporting tables...")
        export_tables(df_tables, args.output_dir, sweep_axes=sweep_axes_all)

    print("Done.")


if __name__ == "__main__":
    main()
