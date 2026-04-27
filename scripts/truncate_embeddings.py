#!/usr/bin/env python3
"""Truncate 1024D embeddings to lower dimensionalities.

By default this script operates inside the repo's local embedding layout under
data/embeddings/qwen3-embedding-0.6b/msmarco_v2. It slices the first N
dimensions from each 1024D file and writes the truncated binaries plus JSON
metadata alongside the originals.
"""

import argparse
import json
import os
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_SRC_DIR = os.path.join(
    REPO_ROOT, "data", "embeddings", "qwen3-embedding-0.6b", "msmarco_v2"
)
SRC_DIM = 1024
DATASET = "msmarco_v2"
CHUNK_SIZE = 10_000

DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
}

SPLITS = ("corpus", "queries")


def parse_target_dims(text: str) -> list[int]:
    dims = []
    for token in text.split(","):
        token = token.strip()
        if token:
            dims.append(int(token))
    if not dims:
        raise argparse.ArgumentTypeError("target dims list must not be empty")
    return dims


def src_filename(dataset: str, split: str, dtype_name: str) -> str:
    return f"{dataset}_{split}_{SRC_DIM}d_{dtype_name}.bin"


def dst_filename(dataset: str, split: str, dim: int, dtype_name: str) -> str:
    return f"{dataset}_{split}_{dim}d_{dtype_name}.bin"


def read_num_vectors(path: str, src_dim: int, np_dtype: np.dtype) -> int:
    file_size = os.path.getsize(path)
    row_bytes = src_dim * np_dtype.itemsize
    assert file_size % row_bytes == 0, (
        f"{path}: size {file_size} not divisible by row size {row_bytes}"
    )
    return file_size // row_bytes


def truncate_file(src_path: str, dst_path: str, src_dim: int, target_dim: int,
                  np_dtype: np.dtype, num_vectors: int) -> None:
    src_row = src_dim * np_dtype.itemsize
    dst_row = target_dim * np_dtype.itemsize
    written = 0

    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        while written < num_vectors:
            batch = min(CHUNK_SIZE, num_vectors - written)
            raw = fin.read(batch * src_row)
            chunk = np.frombuffer(raw, dtype=np_dtype).reshape(batch, src_dim)
            chunk[:, :target_dim].tofile(fout)
            written += batch

            if num_vectors >= 100_000 and written % (num_vectors // 10) < batch:
                pct = 100.0 * written / num_vectors
                print(f"  {pct:5.1f}%  ({written:,} / {num_vectors:,})", flush=True)

    actual = os.path.getsize(dst_path)
    expected = num_vectors * dst_row
    assert actual == expected, f"Size mismatch: {actual} != {expected}"


def write_metadata(dst_json: str, split: str, dim: int, dtype_name: str,
                   num_vectors: int, np_dtype: np.dtype, src_file: str) -> None:
    meta = {
        "embed_dim": dim,
        "dtype": dtype_name,
        "item_size_bytes": dim * np_dtype.itemsize,
        "merged_file": os.path.basename(dst_json).replace(".json", ".bin"),
        "total_size_bytes": num_vectors * dim * np_dtype.itemsize,
        "total_vectors": num_vectors,
        "truncated_from_dim": SRC_DIM,
        "source_file": src_file,
        "truncated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(dst_json, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truncate 1024D embedding binaries in place")
    parser.add_argument("--src-dir", default=DEFAULT_SRC_DIR,
                        help="Directory containing the 1024D source binaries")
    parser.add_argument("--dataset", default=DATASET,
                        help="Dataset stem used in filenames (default: msmarco_v2)")
    parser.add_argument("--target-dims", type=parse_target_dims,
                        default=parse_target_dims("16,32,64,128,256,512,768,1024"),
                        help="Comma-separated target dimensionalities")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Vectors processed per chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size

    if not os.path.isdir(args.src_dir):
        raise SystemExit(f"Source directory does not exist: {args.src_dir}")

    for dtype_name, np_dtype in DTYPES.items():
        for split in SPLITS:
            src_file = src_filename(args.dataset, split, dtype_name)
            src_path = os.path.join(args.src_dir, src_file)

            if not os.path.exists(src_path):
                print(f"SKIP (source missing): {src_path}")
                continue

            num_vectors = read_num_vectors(src_path, SRC_DIM, np.dtype(np_dtype))

            for dim in args.target_dims:
                dst_file = dst_filename(args.dataset, split, dim, dtype_name)
                dst_path = os.path.join(args.src_dir, dst_file)
                dst_json = dst_path.replace(".bin", ".json")

                if os.path.exists(dst_path):
                    print(f"EXISTS: {dst_file} -- skipping")
                    continue

                print(f"{src_file} -> {dst_file}  "
                      f"({num_vectors:,} vectors, {SRC_DIM}D -> {dim}D, {dtype_name})")

                truncate_file(src_path, dst_path, SRC_DIM, dim,
                              np.dtype(np_dtype), num_vectors)
                write_metadata(dst_json, split, dim, dtype_name,
                               num_vectors, np.dtype(np_dtype), src_file)

                size_gb = os.path.getsize(dst_path) / (1024**3)
                print(f"  DONE: {dst_file} ({size_gb:.2f} GB)\n")

    print("All done.")


if __name__ == "__main__":
    main()
