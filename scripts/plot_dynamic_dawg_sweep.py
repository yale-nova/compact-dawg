#!/usr/bin/env python3
"""
plot_dynamic_dawg_sweep.py

Plots DynamicDawg sweep output with direct comparisons against the best
fixed-width PC-* baseline from the same CSV row.
Figure titles include the CSV's unique-key count, and the output directory gets
a README.md noting that DynamicDawg plots are only directly comparable to
baselines from the same CSV. With ``--export-tables``, the script also writes
Markdown companions summarizing the DynamicDawg rows and the
best-by-dimension comparisons. The best-storage-by-dimension comparison uses a
log-scaled dimension axis so wide dimension ranges stay readable.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import hatch_for, line_style_for, marker_for  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DynamicDawg Storage Sweep")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file from bench_dynamic_dawg_sweep")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save plots")
    parser.add_argument(
        "--export-tables",
        action="store_true",
        help="Write Markdown companion tables for DynamicDawg rows, deltas, and best-by-dimension summaries",
    )
    return parser.parse_args()


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key_bytes = out["dim"].astype(float) * 4.0 if "dim" in out.columns else None
    if "normalized_bpk" not in out.columns and {"bytes_per_key", "dim"}.issubset(out.columns):
        out["normalized_bpk"] = out["bytes_per_key"].astype(float) / key_bytes
    if key_bytes is not None and "best_pc_bytes_per_key" in out.columns:
        out["best_pc_normalized_bpk"] = out["best_pc_bytes_per_key"].astype(float) / key_bytes
    if key_bytes is not None and "best_cd_bytes_per_key" in out.columns:
        out["best_cd_normalized_bpk"] = out["best_cd_bytes_per_key"].astype(float) / key_bytes
    if "delta_vs_best_pc_bpk" in out.columns:
        out["delta_vs_best_pc_bpk_pct"] = out["delta_vs_best_pc_bpk"].astype(float) * 100.0
    if "delta_vs_best_pc_build_s" in out.columns:
        out["delta_vs_best_pc_build_pct"] = out["delta_vs_best_pc_build_s"].astype(float) * 100.0
    if "delta_vs_best_cd_bpk" in out.columns:
        out["delta_vs_best_cd_bpk_pct"] = out["delta_vs_best_cd_bpk"].astype(float) * 100.0
    if "delta_vs_best_cd_build_s" in out.columns:
        out["delta_vs_best_cd_build_pct"] = out["delta_vs_best_cd_build_s"].astype(float) * 100.0
    return out


def run_context_label(df: pd.DataFrame) -> str:
    parts = []
    if "n_unique_keys" in df.columns:
        ns = sorted(int(x) for x in df["n_unique_keys"].dropna().unique())
        if len(ns) == 1:
            parts.append(f"N={ns[0]:,}")
        elif ns:
            parts.append(f"N={ns[0]:,}..{ns[-1]:,}")
    if "dtype" in df.columns:
        dtypes = sorted(str(x) for x in df["dtype"].dropna().unique())
        if len(dtypes) == 1:
            parts.append(dtypes[0])
        elif dtypes:
            parts.append("dtype=" + ",".join(dtypes))
    return ", ".join(parts)


def titled(title: str, df: pd.DataFrame) -> str:
    label = run_context_label(df)
    return f"{title}\n({label})" if label else title


def write_plot_notes(df: pd.DataFrame, outdir: str, input_path: str) -> None:
    n_values = sorted(int(x) for x in df.get("n_unique_keys", pd.Series(dtype=int)).dropna().unique())
    dims = sorted(int(x) for x in df.get("dim", pd.Series(dtype=int)).dropna().unique())
    methods = sorted(str(x) for x in df.get("segmentation_method", pd.Series(dtype=str)).dropna().unique())
    thresholds = sorted(float(x) for x in df.get("rho_threshold", pd.Series(dtype=float)).dropna().unique())

    lines = [
        "# DynamicDawg plot notes",
        "",
        f"- Source CSV: `{input_path}`",
    ]
    if n_values:
        lines.append("- Unique-key counts: " + ", ".join(f"{n:,}" for n in n_values))
    if dims:
        lines.append("- Dimensions: " + ", ".join(str(d) for d in dims))
    if methods:
        lines.append("- Planners: " + ", ".join(methods))
    if thresholds:
        lines.append("- Rho thresholds: " + ", ".join(f"{t:g}" for t in thresholds))
    lines.extend(
        [
            "",
            "These plots compare each DynamicDawg row only against the fixed-width CD/PC baselines "
            "stored in the same CSV. Do not compare them directly to 100k/500k sweep "
            "figures unless the unique-key counts match.",
            "The best-storage-by-dimension comparison uses a log-scaled dimension axis.",
            "",
        ]
    )

    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def add_legend() -> None:
    plt.legend(
        title="Dimension / planner",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize="small",
    )


def add_inside_legend(title: str = "") -> None:
    plt.legend(title=title, loc="best", frameon=True, fontsize="small")


def set_log_dimension_axis(dims: pd.Series) -> None:
    tick_dims = sorted({int(d) for d in dims.dropna() if float(d) > 0.0})
    if not tick_dims:
        return
    plt.xscale("log", base=2)
    plt.xticks(tick_dims, [str(d) for d in tick_dims])


def _comma_fmt(n: object) -> str:
    if n is None or pd.isna(n):
        return "--"
    return f"{int(round(float(n))):,}"


def _float_fmt(v: object, digits: int = 4) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):.{digits}f}"


def _pct_fmt(v: object) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v):+.2f}%"


def _time_fmt(v: object) -> str:
    if v is None or pd.isna(v):
        return "--"
    x = float(v)
    if x < 10.0:
        return f"{x:.4f}"
    if x < 100.0:
        return f"{x:.2f}"
    return f"{x:.1f}"


def _md_escape_cell(s: object) -> str:
    return str(s).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(_md_escape_cell(x) for x in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(_md_escape_cell(x) for x in row) + " |"
        for row in rows
    ]
    return "\n".join([head, sep] + body) + "\n\n"


def _write_md(path: str, title: str, blurb: str, body: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{blurb}\n\n{body}")


def _pivot_method_threshold(
    sub: pd.DataFrame,
    value_col: str,
    methods: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    pt = sub.pivot_table(
        index="segmentation_method",
        columns="rho_threshold",
        values=value_col,
        aggfunc="first",
    )
    if pt.empty:
        return pt
    pt = pt.reindex(index=[m for m in methods if m in pt.index])
    pt = pt.reindex(columns=[t for t in thresholds if t in pt.columns])
    return pt


def export_tables(df: pd.DataFrame, outdir: str, input_path: str) -> None:
    methods = sorted(str(x) for x in df.get("segmentation_method", pd.Series(dtype=str)).dropna().unique())
    dims = sorted(int(x) for x in df.get("dim", pd.Series(dtype=int)).dropna().unique())
    thresholds = sorted(float(x) for x in df.get("rho_threshold", pd.Series(dtype=float)).dropna().unique())
    label = run_context_label(df)
    label_text = f" ({label})" if label else ""

    # Long-form DynamicDawg rows
    headers = [
        "dim",
        "planner",
        "rho",
        "segments",
        "Norm BPK",
        "Build (s)",
        "Plan (s)",
        "Best CD Norm",
        "Best PC Norm",
        "d vs CD",
        "d vs PC",
    ]
    rows: list[list[object]] = []
    long_cols = {
        "dim", "segmentation_method", "rho_threshold", "n_segments", "normalized_bpk",
        "total_build_s", "plan_s", "best_cd_normalized_bpk", "best_pc_normalized_bpk",
        "delta_vs_best_cd_bpk_pct", "delta_vs_best_pc_bpk_pct",
    }
    long_df = df.copy()
    for _, row in long_df.sort_values(["dim", "segmentation_method", "rho_threshold"]).iterrows():
        rows.append([
            _comma_fmt(row.get("dim")),
            row.get("segmentation_method", "--"),
            _float_fmt(row.get("rho_threshold"), 2),
            _comma_fmt(row.get("n_segments")),
            _float_fmt(row.get("normalized_bpk")),
            _time_fmt(row.get("total_build_s")),
            _time_fmt(row.get("plan_s")),
            _float_fmt(row.get("best_cd_normalized_bpk")),
            _float_fmt(row.get("best_pc_normalized_bpk")),
            _pct_fmt(row.get("delta_vs_best_cd_bpk_pct")),
            _pct_fmt(row.get("delta_vs_best_pc_bpk_pct")),
        ])
    _write_md(
        os.path.join(outdir, "table_dynamic_dawg_long.md"),
        "DynamicDawg Rows",
        f"All DynamicDawg sweep rows from `{input_path}`{label_text}.",
        _markdown_table(headers, rows),
    )

    # Delta tables vs best CD/PC
    delta_specs = [
        ("delta_vs_best_pc_bpk_pct", "table_dynamic_dawg_storage_delta_vs_best_pc.md", "Storage delta vs best PC-*", "Storage delta vs best PC-* (%)"),
        ("delta_vs_best_cd_bpk_pct", "table_dynamic_dawg_storage_delta_vs_best_cd.md", "Storage delta vs best CD-*", "Storage delta vs best CD-* (%)"),
        ("delta_vs_best_pc_build_pct", "table_dynamic_dawg_build_delta_vs_best_pc.md", "Build-time delta vs fastest PC-*", "Build-time delta vs fastest PC-* (%)"),
        ("delta_vs_best_cd_build_pct", "table_dynamic_dawg_build_delta_vs_best_cd.md", "Build-time delta vs fastest CD-*", "Build-time delta vs fastest CD-* (%)"),
    ]
    for value_col, filename, title, value_desc in delta_specs:
        if value_col not in df.columns:
            continue
        parts: list[str] = []
        for dim in dims:
            sub = df[df["dim"] == dim]
            pt = _pivot_method_threshold(sub, value_col, methods, thresholds)
            if pt.empty:
                continue
            table_rows = []
            for method, prow in pt.iterrows():
                table_rows.append([method] + [_pct_fmt(prow[t]) for t in pt.columns])
            parts.append(
                f"## {dim}D\n\n"
                + _markdown_table(
                    ["Planner"] + [f"{t:g}" for t in pt.columns],
                    table_rows,
                )
            )
        if parts:
            _write_md(
                os.path.join(outdir, filename),
                title,
                f"{value_desc}{label_text}. Rows are planners; columns are rho thresholds.",
                "".join(parts),
            )

    # Best storage/build/plan summaries by dimension
    if {"dim", "normalized_bpk", "total_build_s"}.issubset(df.columns):
        best_storage = (
            df.sort_values(["dim", "normalized_bpk", "total_build_s"])
            .groupby("dim", as_index=False)
            .first()
            .sort_values("dim")
        )
        base = (
            df.sort_values("dim")
            .groupby("dim", as_index=False)
            .first()
            .sort_values("dim")
        )
        rows = []
        for _, row in best_storage.iterrows():
            brow = base[base["dim"] == row["dim"]].iloc[0]
            rows.append([
                _comma_fmt(row["dim"]),
                _float_fmt(brow.get("best_cd_normalized_bpk")),
                _float_fmt(brow.get("best_pc_normalized_bpk")),
                row.get("segmentation_method", "--"),
                _float_fmt(row.get("rho_threshold"), 2),
                _float_fmt(row.get("normalized_bpk")),
                _time_fmt(row.get("total_build_s")),
                _time_fmt(row.get("plan_s")),
            ])
        _write_md(
            os.path.join(outdir, "table_dynamic_dawg_best_storage_by_dim.md"),
            "Best DynamicDawg Storage By Dimension",
            f"Lowest normalized-BPK DynamicDawg row per dimension{label_text}, compared against fixed-width CD/PC baselines from the same CSV.",
            _markdown_table(
                ["dim", "Best CD Norm", "Best PC Norm", "Best DynamicDawg Planner", "rho", "Best DynamicDawg Norm", "Build (s)", "Plan (s)"],
                rows,
            ),
        )

    if {"dim", "total_build_s", "normalized_bpk"}.issubset(df.columns):
        best_build = (
            df.sort_values(["dim", "total_build_s", "normalized_bpk"])
            .groupby("dim", as_index=False)
            .first()
            .sort_values("dim")
        )
        base = (
            df.sort_values("dim")
            .groupby("dim", as_index=False)
            .first()
            .sort_values("dim")
        )
        rows = []
        for _, row in best_build.iterrows():
            brow = base[base["dim"] == row["dim"]].iloc[0]
            rows.append([
                _comma_fmt(row["dim"]),
                _time_fmt(brow.get("best_cd_total_build_s")),
                _time_fmt(brow.get("best_pc_total_build_s")),
                row.get("segmentation_method", "--"),
                _float_fmt(row.get("rho_threshold"), 2),
                _time_fmt(row.get("total_build_s")),
                _float_fmt(row.get("normalized_bpk")),
            ])
        _write_md(
            os.path.join(outdir, "table_dynamic_dawg_best_build_time_by_dim.md"),
            "Best DynamicDawg Build Time By Dimension",
            f"Fastest DynamicDawg row per dimension{label_text}, compared against fixed-width CD/PC baselines from the same CSV.",
            _markdown_table(
                ["dim", "Fastest CD (s)", "Fastest PC (s)", "Fastest DynamicDawg Planner", "rho", "Fastest DynamicDawg Build (s)", "Norm BPK"],
                rows,
            ),
        )

    if {"dim", "normalized_bpk", "total_build_s", "plan_s"}.issubset(df.columns):
        best_storage = (
            df.sort_values(["dim", "normalized_bpk", "total_build_s"])
            .groupby("dim", as_index=False)
            .first()
            .sort_values("dim")
        )
        rows = []
        for _, row in best_storage.iterrows():
            rows.append([
                _comma_fmt(row["dim"]),
                row.get("segmentation_method", "--"),
                _float_fmt(row.get("rho_threshold"), 2),
                _time_fmt(row.get("plan_s")),
                _float_fmt(row.get("normalized_bpk")),
                _time_fmt(row.get("total_build_s")),
            ])
        _write_md(
            os.path.join(outdir, "table_dynamic_dawg_best_plan_time_by_dim.md"),
            "Plan Time Of Best-Storage DynamicDawg Row By Dimension",
            f"Planning time for the same storage-best DynamicDawg row used in the best-storage summary{label_text}.",
            _markdown_table(
                ["dim", "Planner", "rho", "Plan (s)", "Norm BPK", "Build (s)"],
                rows,
            ),
        )


def plot_delta_lines(df: pd.DataFrame, outdir: str, ycol: str, title: str, ylabel: str,
                     filename: str) -> None:
    plt.figure(figsize=(10, 6))
    dims = sorted(int(x) for x in df["dim"].dropna().unique())
    methods = sorted(str(x) for x in df["segmentation_method"].dropna().unique())
    for (dim, method), sub in df.groupby(["dim", "segmentation_method"], sort=True):
        sub = sub.sort_values("rho_threshold")
        method_idx = methods.index(method)
        dim_idx = dims.index(int(dim))
        plt.plot(
            sub["rho_threshold"],
            sub[ycol],
            marker=marker_for(method_idx),
            markersize=6,
            linewidth=1.3,
            linestyle=line_style_for(dim_idx),
            label=f"{dim}d {method}",
        )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title(titled(title, df))
    plt.xlabel("Segmentation Threshold ($\\rho$)")
    plt.ylabel(ylabel)
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    add_legend()
    plt.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()


def plot_best_storage_by_dim(df: pd.DataFrame, outdir: str) -> None:
    required = {"dim", "normalized_bpk", "best_pc_normalized_bpk", "best_cd_normalized_bpk"}
    if not required.issubset(df.columns):
        return

    best_dynamic_dawg = (
        df.sort_values(["dim", "normalized_bpk", "total_build_s"])
        .groupby("dim", as_index=False)
        .first()
        .sort_values("dim")
    )

    baseline = (
        df.sort_values("dim")
        .groupby("dim", as_index=False)
        .first()
        .sort_values("dim")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline["dim"],
        baseline["best_cd_normalized_bpk"],
        marker="o",
        linewidth=2,
        label="Best CD-*",
    )
    plt.plot(
        baseline["dim"],
        baseline["best_pc_normalized_bpk"],
        marker="s",
        linewidth=2,
        label="Best PC-*",
    )
    plt.plot(
        best_dynamic_dawg["dim"],
        best_dynamic_dawg["normalized_bpk"],
        marker="^",
        linewidth=2,
        label="Best DynamicDawg threshold/planner",
    )

    for _, row in best_dynamic_dawg.iterrows():
        label = f"{row['segmentation_method']}\\n$\\rho$={row['rho_threshold']}"
        plt.annotate(
            label,
            (row["dim"], row["normalized_bpk"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    plt.title(titled("Best Storage By Dimension", df))
    set_log_dimension_axis(baseline["dim"])
    plt.xlabel("Dimension (log2 scale)")
    plt.ylabel("Normalized BPK")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    add_inside_legend("Series")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dynamic_dawg_best_storage_by_dim.png"), dpi=200)
    plt.close()


def plot_best_build_time_by_dim(df: pd.DataFrame, outdir: str) -> None:
    required = {"dim", "total_build_s", "best_pc_total_build_s", "best_cd_total_build_s"}
    if not required.issubset(df.columns):
        return

    best_dynamic_dawg = (
        df.sort_values(["dim", "total_build_s", "normalized_bpk"])
        .groupby("dim", as_index=False)
        .first()
        .sort_values("dim")
    )

    baseline = (
        df.sort_values("dim")
        .groupby("dim", as_index=False)
        .first()
        .sort_values("dim")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline["dim"],
        baseline["best_cd_total_build_s"],
        marker="o",
        linewidth=2,
        label="Fastest CD-*",
    )
    plt.plot(
        baseline["dim"],
        baseline["best_pc_total_build_s"],
        marker="s",
        linewidth=2,
        label="Fastest PC-*",
    )
    plt.plot(
        best_dynamic_dawg["dim"],
        best_dynamic_dawg["total_build_s"],
        marker="^",
        linewidth=2,
        label="Fastest DynamicDawg threshold/planner",
    )

    for _, row in best_dynamic_dawg.iterrows():
        label = f"{row['segmentation_method']}\\n$\\rho$={row['rho_threshold']}"
        plt.annotate(
            label,
            (row["dim"], row["total_build_s"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    plt.title(titled("Best Build Time By Dimension", df))
    plt.xlabel("Dimension")
    plt.ylabel("Total Build Time (s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    add_inside_legend("Series")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dynamic_dawg_best_build_time_by_dim.png"), dpi=200)
    plt.close()


def plot_best_plan_time_by_dim(df: pd.DataFrame, outdir: str) -> None:
    required = {"dim", "normalized_bpk", "total_build_s", "plan_s"}
    if not required.issubset(df.columns):
        return

    best_dynamic_dawg = (
        df.sort_values(["dim", "normalized_bpk", "total_build_s"])
        .groupby("dim", as_index=False)
        .first()
        .sort_values("dim")
    )

    x = range(len(best_dynamic_dawg))
    plt.figure(figsize=(10, 6))
    plt.bar(
        x,
        best_dynamic_dawg["plan_s"],
        label="Plan time",
        edgecolor="black",
        linewidth=0.8,
        hatch=hatch_for(1),
    )

    for pos, (_, row) in enumerate(best_dynamic_dawg.iterrows()):
        label = f"{row['segmentation_method']}\n$\\rho$={row['rho_threshold']}"
        plt.annotate(
            label,
            (pos, row["plan_s"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    plt.title(titled("Plan Time Of Best Storage Plan By Dimension", df))
    plt.xlabel("Dimension")
    plt.ylabel("Plan Time (s)")
    plt.xticks(list(x), [f"{dim}d" for dim in best_dynamic_dawg["dim"]])
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    add_inside_legend("Series")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dynamic_dawg_best_plan_time_by_dim.png"), dpi=200)
    plt.close()


def plot_storage_delta_pc(df: pd.DataFrame, outdir: str) -> None:
    plot_delta_lines(
        df,
        outdir,
        "delta_vs_best_pc_bpk_pct",
        "DynamicDawg Storage Delta vs Best PC-*",
        "Storage Delta vs Best PC-* (%)",
        "dynamic_dawg_storage_delta_vs_best_pc.png",
    )


def plot_build_delta_pc(df: pd.DataFrame, outdir: str) -> None:
    plot_delta_lines(
        df,
        outdir,
        "delta_vs_best_pc_build_pct",
        "DynamicDawg Build-Time Delta vs Fastest PC-*",
        "Build-Time Delta vs Best PC-* (%)",
        "dynamic_dawg_build_delta_vs_best_pc.png",
    )


def plot_storage_delta_cd(df: pd.DataFrame, outdir: str) -> None:
    if "delta_vs_best_cd_bpk_pct" not in df.columns:
        return
    plot_delta_lines(
        df,
        outdir,
        "delta_vs_best_cd_bpk_pct",
        "DynamicDawg Storage Delta vs Best CD-*",
        "Storage Delta vs Best CD-* (%)",
        "dynamic_dawg_storage_delta_vs_best_cd.png",
    )


def plot_build_delta_cd(df: pd.DataFrame, outdir: str) -> None:
    if "delta_vs_best_cd_build_pct" not in df.columns:
        return
    plot_delta_lines(
        df,
        outdir,
        "delta_vs_best_cd_build_pct",
        "DynamicDawg Build-Time Delta vs Fastest CD-*",
        "Build-Time Delta vs Best CD-* (%)",
        "dynamic_dawg_build_delta_vs_best_cd.png",
    )


def plot_segments_vs_storage(df: pd.DataFrame, outdir: str) -> None:
    plt.figure(figsize=(10, 6))
    methods = sorted(str(x) for x in df["segmentation_method"].dropna().unique())
    for (dim, method), sub in df.groupby(["dim", "segmentation_method"], sort=True):
        method_idx = methods.index(method)
        plt.scatter(
            sub["n_segments"],
            sub["normalized_bpk"],
            s=110,
            marker=marker_for(method_idx),
            edgecolor="black",
            linewidth=0.6,
            label=f"{dim}d {method}",
        )
    plt.title(titled("DynamicDawg Segments vs Normalized Storage", df))
    plt.xlabel("Number of Variable-Width Segments")
    plt.ylabel("Normalized BPK")
    plt.grid(True, linestyle="--", alpha=0.6)
    add_legend()
    plt.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(os.path.join(outdir, "dynamic_dawg_segments_vs_normalized_bpk.png"), dpi=200)
    plt.close()


def plot_threshold_context(df: pd.DataFrame, outdir: str) -> None:
    plt.figure(figsize=(10, 6))
    dims = sorted(int(x) for x in df["dim"].dropna().unique())
    methods = sorted(str(x) for x in df["segmentation_method"].dropna().unique())
    for (dim, method), sub in df.groupby(["dim", "segmentation_method"], sort=True):
        sub = sub.sort_values("rho_threshold")
        method_idx = methods.index(method)
        dim_idx = dims.index(int(dim))
        plt.plot(
            sub["rho_threshold"],
            sub["n_segments"],
            marker=marker_for(method_idx),
            markersize=6,
            linewidth=1.3,
            linestyle=line_style_for(dim_idx),
            label=f"{dim}d {method}",
        )
    plt.title(titled("Variable-Width Segments vs Greedy Threshold", df))
    plt.xlabel("Segmentation Threshold ($\\rho$)")
    plt.ylabel("Number of Variable-Width Segments")
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    add_legend()
    plt.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(os.path.join(outdir, "dynamic_dawg_segments_vs_threshold.png"), dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = ensure_derived_columns(pd.read_csv(args.input))

    write_plot_notes(df, args.output_dir, args.input)

    plot_storage_delta_pc(df, args.output_dir)
    plot_build_delta_pc(df, args.output_dir)
    plot_storage_delta_cd(df, args.output_dir)
    plot_build_delta_cd(df, args.output_dir)
    plot_best_storage_by_dim(df, args.output_dir)
    plot_best_build_time_by_dim(df, args.output_dir)
    plot_best_plan_time_by_dim(df, args.output_dir)
    plot_segments_vs_storage(df, args.output_dir)
    plot_threshold_context(df, args.output_dir)
    if args.export_tables:
        export_tables(df, args.output_dir, args.input)


if __name__ == "__main__":
    main()
