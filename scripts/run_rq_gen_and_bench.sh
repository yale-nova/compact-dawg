#!/usr/bin/env bash
# Run gen_rq_queries_exact10_1024d then bench_dawg_rq with aligned arguments.
#
# Shared parameters (one definition → both tools):
#   trie-step        → gen --trie-step, bench --key-step (must match)
#   queries-per-trie → gen --queries-per-trie, bench --queries-per-step (must match)
#   matches-per-query, total-points, data-file → both where applicable
#
# Usage (from repo root, after building targets):
#   ./scripts/run_rq_gen_and_bench.sh
#   ./scripts/run_rq_gen_and_bench.sh --trie-step 500 --num-tries 20 --n-keys 500,5000,10000
#   ./scripts/run_rq_gen_and_bench.sh --skip-gen
# By default bench_dawg_rq writes results/rq_bench_<stem>.csv (--output-csv). Use --no-bench-csv to skip.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Default to the conventional out-of-tree build directory used in the README.
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

GEN_EXE="$BUILD_DIR/gen_rq_queries_exact10_1024d"
BENCH_EXE="$BUILD_DIR/bench_dawg_rq"

# ---- defaults (match gen_rq_queries_exact10_1024d + bench_dawg_rq) ----
DATA_FILE="$REPO_ROOT/1024d-uniq-100k.bin"
TOTAL_POINTS=100000
TRIE_STEP=1000
NUM_TRIES=100
QUERIES_PER_TRIE=10
MATCHES_PER_QUERY=10
MAX_ATTEMPTS_PER_QUERY=50000
SEED=""
OUT_DIR="$REPO_ROOT/tests/testdata"
OUT_LOWER=""
OUT_UPPER=""
OUT_GT=""
N_KEYS="1000,10000,50000,100000"
GROUP_BITS="32,64,128,256,512,1024"
SKIP_GEN=0
SKIP_BENCH=0
VERIFY_GT=1
BENCH_CSV=""       # empty → default $REPO_ROOT/results/rq_bench_<stem>.csv
NO_BENCH_CSV=0

die() { echo "Error: $*" >&2; exit 1; }

usage() {
    cat <<'EOF' >&2
run_rq_gen_and_bench.sh — run gen_rq_queries_exact10_1024d then bench_dawg_rq with matching flags.

EOF
    cat <<'EOF' >&2

Options (shared / generator / bench):
  --data-file PATH          Float32 dataset [1024 floats per row] (default: ./1024d-uniq-100k.bin)
  --total-points N          Rows to use from dataset (default: 100000)
  --trie-step N             Prefix step size; passed to gen AND bench --key-step (default: 1000)
  --num-tries N             Generator: number of trie sizes = num-tries * trie-step (default: 100)
  --queries-per-trie N      Must equal bench queries-per-step (default: 10)
  --matches-per-query M     Exact matches per query / GT width (default: 10)
  --max-attempts-per-query N  Generator only (default: 50000)
  --seed U64                Generator only (optional)

Outputs:
  --out-dir DIR             Directory for lower/upper/GT bins (default: tests/testdata)
  --output-lower PATH       Override lower path (default: OUT_DIR/rq_lower_<stem>.bin)
  --output-upper PATH
  --output-ground-truth PATH

Bench-only:
  --n-keys CSV              e.g. 1000,10000,100000,500000,1000000
  --group-bits CSV          e.g. 32 or 16,32,64
  --bench-csv PATH          Pass to bench_dawg_rq --output-csv (default: results/rq_bench_<stem>.csv)
  --no-bench-csv            Do not pass --output-csv (no CSV from this script)

Control:
  --build-dir DIR           CMake build dir (default: build)
  --skip-gen                Only run bench (fixtures must exist)
  --skip-bench              Only run generator
  --no-ground-truth-bench   Do not pass --ground-truth to bench_dawg_rq

Environment:
  BUILD_DIR                 Same as --build-dir

Alignment rules:
  max(n-keys) <= num-tries * trie-step
  Each N in --n-keys should be a positive multiple of --trie-step (same as generator trie sizes).
EOF
}

STEM_FROM_CONFIG() {
    echo "${TRIE_STEP}_to_$((NUM_TRIES * TRIE_STEP))_q${MATCHES_PER_QUERY}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --data-file) DATA_FILE="$2"; shift 2 ;;
        --total-points) TOTAL_POINTS="$2"; shift 2 ;;
        --trie-step) TRIE_STEP="$2"; shift 2 ;;
        --num-tries) NUM_TRIES="$2"; shift 2 ;;
        --queries-per-trie) QUERIES_PER_TRIE="$2"; shift 2 ;;
        --matches-per-query) MATCHES_PER_QUERY="$2"; shift 2 ;;
        --max-attempts-per-query) MAX_ATTEMPTS_PER_QUERY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --output-lower) OUT_LOWER="$2"; shift 2 ;;
        --output-upper) OUT_UPPER="$2"; shift 2 ;;
        --output-ground-truth) OUT_GT="$2"; shift 2 ;;
        --n-keys) N_KEYS="$2"; shift 2 ;;
        --group-bits) GROUP_BITS="$2"; shift 2 ;;
        --bench-csv) BENCH_CSV="$2"; shift 2 ;;
        --no-bench-csv) NO_BENCH_CSV=1; shift ;;
        --build-dir)
            BUILD_DIR="$2"
            GEN_EXE="$BUILD_DIR/gen_rq_queries_exact10_1024d"
            BENCH_EXE="$BUILD_DIR/bench_dawg_rq"
            shift 2
            ;;
        --skip-gen) SKIP_GEN=1; shift ;;
        --skip-bench) SKIP_BENCH=1; shift ;;
        --no-ground-truth-bench) VERIFY_GT=0; shift ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

[[ "$SKIP_GEN" -eq 0 ]] && {
    [[ -x "$GEN_EXE" ]] || die "generator not found or not executable: $GEN_EXE
  Fix: cmake --build \"$BUILD_DIR\" --target gen_rq_queries_exact10_1024d bench_dawg_rq
  Or set BUILD_DIR / --build-dir to your CMake binary dir (e.g. BUILD_DIR=build for cmake -B build)."
}
[[ "$SKIP_BENCH" -eq 0 ]] && {
    [[ -x "$BENCH_EXE" ]] || die "bench_dawg_rq not found or not executable: $BENCH_EXE
  Fix: cmake --build \"$BUILD_DIR\" --target bench_dawg_rq
  Or set BUILD_DIR to your CMake build directory."
}

mkdir -p "$OUT_DIR"
STEM="$(STEM_FROM_CONFIG)"
if [[ -z "$OUT_LOWER" ]]; then OUT_LOWER="$OUT_DIR/1024d_rq_lower_${STEM}.bin"; fi
if [[ -z "$OUT_UPPER" ]]; then OUT_UPPER="$OUT_DIR/1024d_rq_upper_${STEM}.bin"; fi
if [[ -z "$OUT_GT" ]]; then OUT_GT="$OUT_DIR/1024d_rq_ground_truth_${STEM}.bin"; fi

MAX_PREFIX=$((NUM_TRIES * TRIE_STEP))
IFS=',' read -r -a NKEYS_ARR <<<"$N_KEYS"
MAX_N=0
for n in "${NKEYS_ARR[@]}"; do
    n="${n// /}"
    [[ "$n" =~ ^[0-9]+$ ]] || die "bad --n-keys entry: $n"
    (( n > MAX_N )) && MAX_N="$n"
    (( n % TRIE_STEP != 0 )) && die "n-keys entry $n is not a multiple of trie-step $TRIE_STEP (must match generator trie sizes)"
done
(( MAX_N <= MAX_PREFIX )) || die "max n-keys ($MAX_N) > num-tries * trie-step ($MAX_PREFIX); increase --num-tries or reduce --n-keys"
(( MAX_N <= TOTAL_POINTS )) || die "max n-keys ($MAX_N) > total-points ($TOTAL_POINTS)"

NUM_STEPS_BENCH=$((MAX_N / TRIE_STEP))
# Bench uses global query indices up to (max_n/trie_step)*queries-per-trie - 1; generator writes num_tries * queries-per-trie queries.
(( NUM_TRIES >= NUM_STEPS_BENCH )) || die "num-tries ($NUM_TRIES) < max n-keys / trie-step ($NUM_STEPS_BENCH); increase --num-tries"

echo "== run_rq_gen_and_bench =="
echo "  BUILD_DIR=$BUILD_DIR"
echo "  trie-step=$TRIE_STEP  num-tries=$NUM_TRIES  queries-per-trie=$QUERIES_PER_TRIE  matches=$MATCHES_PER_QUERY"
echo "  total-points=$TOTAL_POINTS  max n-keys=$MAX_N  (bench steps=$NUM_STEPS_BENCH, queries-per-step=$QUERIES_PER_TRIE)"
echo "  outputs: lower=$OUT_LOWER"

if [[ "$SKIP_GEN" -eq 0 ]]; then
    GEN_CMD=(
        "$GEN_EXE"
        --data-file "$DATA_FILE"
        --total-points "$TOTAL_POINTS"
        --trie-step "$TRIE_STEP"
        --num-tries "$NUM_TRIES"
        --queries-per-trie "$QUERIES_PER_TRIE"
        --matches-per-query "$MATCHES_PER_QUERY"
        --max-attempts-per-query "$MAX_ATTEMPTS_PER_QUERY"
        --output-lower "$OUT_LOWER"
        --output-upper "$OUT_UPPER"
        --output-ground-truth "$OUT_GT"
    )
    [[ -n "$SEED" ]] && GEN_CMD+=(--seed "$SEED")
    echo "  running: ${GEN_CMD[*]}"
    "${GEN_CMD[@]}"
else
    echo "  (--skip-gen) using existing fixtures"
    [[ -f "$OUT_LOWER" && -f "$OUT_UPPER" ]] || die "missing fixture(s): $OUT_LOWER / $OUT_UPPER"
    [[ "$VERIFY_GT" -eq 0 || -f "$OUT_GT" ]] || die "missing ground truth: $OUT_GT (or use --no-ground-truth-bench)"
fi

if [[ "$SKIP_BENCH" -eq 0 ]]; then
    BENCH_CMD=(
        "$BENCH_EXE"
        "$DATA_FILE"
        "$OUT_LOWER"
        "$OUT_UPPER"
        --key-step "$TRIE_STEP"
        --queries-per-step "$QUERIES_PER_TRIE"
        --matches-per-query "$MATCHES_PER_QUERY"
        --total-points "$TOTAL_POINTS"
        --n-keys "$N_KEYS"
        --group-bits "$GROUP_BITS"
    )
    [[ "$VERIFY_GT" -eq 1 ]] && BENCH_CMD+=(--ground-truth "$OUT_GT")
    if [[ "$NO_BENCH_CSV" -eq 0 ]]; then
        BENCH_CSV_FILE="${BENCH_CSV:-$REPO_ROOT/results/rq_bench_${STEM}.csv}"
        mkdir -p "$(dirname "$BENCH_CSV_FILE")"
        BENCH_CMD+=(--output-csv "$BENCH_CSV_FILE")
        echo "  bench CSV: $BENCH_CSV_FILE"
    fi
    echo "  running: ${BENCH_CMD[*]}"
    "${BENCH_CMD[@]}"
else
    echo "  (--skip-bench) done after generator"
fi

echo "== done =="
