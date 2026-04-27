#!/usr/bin/env python3
"""
Generate ground truth for 1024d range queries via brute-force linear scan.

Reads:
  - repo-root 1024d-uniq-100k.bin (100k points × 1024 dims × float32)
  - tests/testdata/1024d_rq_lower.bin (N_QUERIES × 1024 × float32)
  - tests/testdata/1024d_rq_upper.bin (N_QUERIES × 1024 × float32)

Outputs:
  - tests/testdata/1024d_rq_ground_truth.csv in format:
    [id: 0, count: 42]
    <point1_as_float_csv>
    ...
    
    [id: 1, count: 17]
    ...

Ground truth stores original floats. The C++ test converts results back to float
for comparison.
"""
import numpy as np
import os
import argparse

NUM_DIMENSIONS = 1024
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TEST_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "tests", "testdata")
DEFAULT_DATA_FILE = os.path.join(REPO_ROOT, "1024d-uniq-100k.bin")

def main():
    parser = argparse.ArgumentParser(description="Generate ground truth for range queries")
    parser.add_argument("--data-file", type=str,
                        default=DEFAULT_DATA_FILE,
                        help="Path to the data file")
    parser.add_argument("--lower-file", type=str,
                        default=os.path.join(TEST_DATA_DIR, "1024d_rq_lower.bin"),
                        help="Path to lower bounds file")
    parser.add_argument("--upper-file", type=str,
                        default=os.path.join(TEST_DATA_DIR, "1024d_rq_upper.bin"),
                        help="Path to upper bounds file")
    parser.add_argument("--output", type=str,
                        default=os.path.join(TEST_DATA_DIR, "1024d_rq_ground_truth.csv"),
                        help="Path to output ground truth file")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.data_file}...")
    data = np.fromfile(args.data_file, dtype=np.float32)
    num_points = len(data) // NUM_DIMENSIONS
    data = data.reshape(num_points, NUM_DIMENSIONS)
    print(f"Loaded {num_points} points with {NUM_DIMENSIONS} dimensions")

    # Load query bounds
    print(f"Loading bounds from {args.lower_file} and {args.upper_file}...")
    lower = np.fromfile(args.lower_file, dtype=np.float32).reshape(-1, NUM_DIMENSIONS)
    upper = np.fromfile(args.upper_file, dtype=np.float32).reshape(-1, NUM_DIMENSIONS)
    num_queries = lower.shape[0]
    assert lower.shape == upper.shape, "Lower and upper bounds must have same shape"
    print(f"Loaded {num_queries} queries")

    # Perform brute-force range search
    print("Running brute-force range search...")
    with open(args.output, 'w') as f:
        total_matches = 0
        for q in range(num_queries):
            lo = lower[q]
            hi = upper[q]
            
            # Check all dimensions at once using broadcasting
            in_range = np.all((data >= lo) & (data <= hi), axis=1)
            matching_indices = np.where(in_range)[0]
            matching_points = data[matching_indices]
            
            # Sort points lexicographically (to match C++ output which sorts for comparison)
            if len(matching_points) > 0:
                # lexsort sorts by last key first, so reverse the transpose
                sort_order = np.lexsort(matching_points.T[::-1])
                matching_points = matching_points[sort_order]
            
            count = len(matching_points)
            total_matches += count
            
            # Write header
            f.write(f"[id: {q}, count: {count}]\n")
            
            # Write matching points as CSV (original floats)
            for pt in matching_points:
                # Use repr-style formatting to preserve full float precision
                f.write(",".join(f"{v:.9g}" for v in pt) + "\n")
            
            # Blank line between queries
            f.write("\n")
            
            if (q + 1) % 5 == 0 or q == num_queries - 1:
                print(f"  Query {q+1}/{num_queries}: {count} matches")

    print(f"Total matches across all queries: {total_matches}")
    print(f"Saved ground truth to {args.output} ({os.path.getsize(args.output)} bytes)")
    print("Done!")

if __name__ == "__main__":
    main()
