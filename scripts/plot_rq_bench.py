#!/usr/bin/env python3
"""Plot CompactDawg range-query bench CSVs from bench_dawg_rq --output-csv.

Produces two panels (one figure by default):
  1) X = group_bits, Y = avg_query_ms — one point per GROUP_BITS using the row
     with the largest n_keys (max N) within that group.
  2) X = n_keys, Y = avg_query_ms — one line per GROUP_BITS (all N in the CSV).
The default figure title includes the per-row query count when that metadata is
present, so small query-count smoke runs are visibly labeled.

Optional Markdown tables (--export-md) mirror the per-GROUP_BITS ASCII tables printed by
bench_dawg_rq.cpp (same columns and the same formatting rules as bench_common.h).

Usage:
    python3 scripts/plot_rq_bench.py -i results/rq_bench_1000_to_100000_q10.csv -o plots/rq_bench.png
    python3 scripts/plot_rq_bench.py -i run1.csv -i run2.csv -o plots/merged.png --logy
    python3 scripts/plot_rq_bench.py -i results/rq.csv -o plots/rq_bench.png --export-md
    python3 scripts/plot_rq_bench.py -i results/rq.csv -o plots/rq_bench.png --export-md plots/custom_tables.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLS = (
    "group_bits",
    "n_keys",
    "avg_query_ms",
)

# Columns needed for Markdown tables (always present in bench_dawg_rq --output-csv).
TABLE_COLS = (
    "queries_run",
    "sort_s",
    "build_s",
    "size_bytes",
    "gt_queries",
    "gt_mismatches",
)


def comma_fmt(n: int) -> str:
    """Match tools/bench_common.h comma_fmt (thousands separators)."""
    return f"{int(n):,}"


def size_fmt(num_bytes: float) -> str:
    """Match tools/bench_common.h size_fmt."""
    b = int(num_bytes)
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024.0:.1f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024.0 * 1024.0):.2f} MB"
    return f"{b / (1024.0 * 1024.0 * 1024.0):.2f} GB"


def time_fmt(s: float) -> str:
    """Match tools/bench_common.h time_fmt."""
    if s < 10.0:
        return f"{s:.4f}"
    if s < 100.0:
        return f"{s:.2f}"
    return f"{s:.1f}"


def gt_check_cell(gt_queries: float, gt_mismatches: float) -> str:
    """Match bench_dawg_rq.cpp print_table GT column."""
    gq = int(gt_queries) if not pd.isna(gt_queries) else 0
    gm = int(gt_mismatches) if not pd.isna(gt_mismatches) else 0
    if gq == 0:
        return "n/a"
    if gm == 0:
        return "OK"
    return f"FAIL {gm}/{gq}"


def md_escape_cell(s: str) -> str:
    """Escape pipes so Markdown tables stay valid."""
    return s.replace("|", "\\|")


def dataframe_to_rq_tables_md(df: pd.DataFrame, *, title: str) -> str:
    """Build Markdown: one section per _source_file, one ### per group_bits, table per bench print_table."""
    miss = [c for c in TABLE_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Cannot export Markdown tables: CSV missing columns {miss}")

    lines: list[str] = [f"# {title}", ""]

    for src in df["_source_file"].unique():
        sub_all = df[df["_source_file"] == src].copy()
        lines.append(f"## Source: `{src}`")
        lines.append("")

        meta = sub_all.iloc[0]
        if "dimensions" in sub_all.columns and pd.notna(meta.get("dimensions")):
            lines.append(f"- **Dimensions:** {int(meta['dimensions'])}")
        if "data_file" in sub_all.columns and pd.notna(meta.get("data_file")):
            lines.append(f"- **Dataset:** `{meta['data_file']}`")
        if "queries_per_step" in sub_all.columns and pd.notna(meta.get("queries_per_step")):
            qps = int(meta["queries_per_step"])
            mpq = int(meta["matches_per_query"]) if "matches_per_query" in sub_all.columns and pd.notna(meta.get("matches_per_query")) else None
            if mpq is not None:
                lines.append(f"- **Queries / key count:** {qps}; **matches per query:** {mpq}")
            else:
                lines.append(f"- **Queries / key count:** {qps}")
        lines.append("")

        for gb in sorted(sub_all["group_bits"].dropna().unique().astype(int)):
            sub = sub_all[sub_all["group_bits"] == gb].sort_values("n_keys")
            lines.append(f"### GROUP_BITS = {gb}")
            lines.append("")
            lines.append(
                "| N (keys) | Queries | Sort(s) | Build(s) | Size | Avg Query(ms) | GT check |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")

            for _, row in sub.iterrows():
                n_keys = int(row["n_keys"])
                qrun = int(row["queries_run"])
                sort_s = float(row["sort_s"])
                build_s = float(row["build_s"])
                sz = float(row["size_bytes"])
                aq = float(row["avg_query_ms"])
                gt_q = row["gt_queries"]
                gt_m = row["gt_mismatches"]

                cells = [
                    comma_fmt(n_keys),
                    comma_fmt(qrun),
                    time_fmt(sort_s),
                    time_fmt(build_s),
                    size_fmt(sz),
                    f"{aq:.2f}",
                    gt_check_cell(gt_q, gt_m),
                ]
                lines.append("| " + " | ".join(md_escape_cell(c) for c in cells) + " |")

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        miss = [c for c in REQUIRED_COLS if c not in df.columns]
        if miss:
            raise SystemExit(f"{p}: missing columns {miss}")
        df["_source_file"] = str(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in ("group_bits", "n_keys"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out["avg_query_ms"] = pd.to_numeric(out["avg_query_ms"], errors="coerce")
    out = out.dropna(subset=list(REQUIRED_COLS) + ["avg_query_ms"])
    return out


def max_n_row_per_group_bits(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group_bits: the row with maximum n_keys (tie: first)."""
    idx = df.groupby("group_bits", sort=True)["n_keys"].idxmax()
    return df.loc[idx].sort_values("group_bits")


def title_context(df: pd.DataFrame) -> str:
    parts = []
    if "dimensions" in df.columns and not df["dimensions"].dropna().empty:
        dims = sorted(int(x) for x in df["dimensions"].dropna().unique())
        parts.append("dim=" + (str(dims[0]) if len(dims) == 1 else ",".join(str(d) for d in dims)))
    if "queries_per_step" in df.columns and not df["queries_per_step"].dropna().empty:
        qps = sorted(int(x) for x in df["queries_per_step"].dropna().unique())
        label = str(qps[0]) if len(qps) == 1 else ",".join(str(q) for q in qps)
        parts.append(f"queries/N={label}")
    if "matches_per_query" in df.columns and not df["matches_per_query"].dropna().empty:
        mpq = sorted(int(x) for x in df["matches_per_query"].dropna().unique())
        label = str(mpq[0]) if len(mpq) == 1 else ",".join(str(m) for m in mpq)
        parts.append(f"matches/query={label}")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        required=True,
        type=Path,
        help="bench_dawg_rq CSV (repeat -i to merge multiple files)",
    )
    ap.add_argument("-o", "--output", type=Path, default=Path("plots/rq_bench.png"), help="Output PNG path")
    ap.add_argument("--logx-left", action="store_true", help="Log2 x-axis on left panel (group_bits)")
    ap.add_argument("--logx-right", action="store_true", help="Log10 x-axis on right panel (n_keys)")
    ap.add_argument("--logy", action="store_true", help="Log10 y-axis on both panels")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default="", help="Figure suptitle (default: auto from CSV)")
    ap.add_argument(
        "--export-md",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help="Write Markdown tables like bench_dawg_rq stdout (per GROUP_BITS). "
        "PATH defaults to <PNG_stem>_tables.md next to the PNG.",
    )
    args = ap.parse_args()

    df = load_concat(args.inputs)
    if df.empty:
        raise SystemExit("No rows after load.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    gb_maxn = max_n_row_per_group_bits(df)
    gbs = sorted(df["group_bits"].dropna().unique().astype(int))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Left: group_bits vs avg_query_ms at max N
    x_l = gb_maxn["group_bits"].astype(int).values
    y_l = gb_maxn["avg_query_ms"].astype(float).values
    ax_l.plot(x_l, y_l, marker="o", linewidth=1.5, markersize=7, color="C0")
    ax_l.set_xlabel("GROUP_BITS (at max N in CSV)")
    ax_l.set_ylabel("Avg query time (ms)")
    max_n = int(df["n_keys"].max())
    ax_l.set_title(f"Avg query vs GROUP_BITS\n(max N = {max_n:,})")
    if args.logx_left:
        ax_l.set_xscale("log", base=2)
    if args.logy:
        ax_l.set_yscale("log")

    # Right: n_keys vs avg_query_ms, one series per group_bits
    for gb in gbs:
        sub = df[df["group_bits"] == gb].sort_values("n_keys")
        ax_r.plot(
            sub["n_keys"].astype(int),
            sub["avg_query_ms"].astype(float),
            marker="o",
            linewidth=1.4,
            markersize=5,
            label=f"GB={gb}",
        )
    ax_r.set_xlabel("N (keys in DAWG)")
    ax_r.set_ylabel("Avg query time (ms)")
    ax_r.set_title("Avg query vs N (by GROUP_BITS)")
    ax_r.legend(title="Edge label", fontsize=8, title_fontsize=8)
    if args.logx_right:
        ax_r.set_xscale("log")
    if args.logy:
        ax_r.set_yscale("log")

    supt = args.title.strip()
    if not supt:
        ctx = title_context(df)
        supt = f"Range-query bench  {ctx}" if ctx else "Range-query bench"
    fig.suptitle(supt, fontsize=11)

    fig.savefig(args.output, dpi=args.dpi)
    print(f"Wrote {args.output.resolve()}")

    if args.export_md is not None:
        if args.export_md == "__AUTO__":
            md_path = args.output.with_name(args.output.stem + "_tables.md")
        else:
            md_path = Path(args.export_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_title = "CompactDawg Range Query Benchmark (from CSV)"
        try:
            body = dataframe_to_rq_tables_md(df, title=md_title)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        md_path.write_text(body, encoding="utf-8")
        print(f"Wrote {md_path.resolve()}")


if __name__ == "__main__":
    main()
