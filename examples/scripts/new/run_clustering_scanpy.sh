#!/usr/bin/env bash
# Run Scanpy clustering (Leiden and/or Louvain) on a 10x count matrix.
#
# Usage:
#   bash run_clustering_scanpy.sh DATA_DIR OUTPUT_DIR [CLUSTERING_OPTIONS...]
#
# Arguments:
#   DATA_DIR    Directory containing the 10x matrix files.
#   OUTPUT_DIR  Directory where the clustering files are written.
#   CLUSTERING_OPTIONS
#               Passed on to examples/run_scanpy_clustering.py, for example
#               --n_reps 10 --resolution 1.0 --n_neighbors 15 --n_pcs 10
#
# Environment variables:
#   METHODS     Space-separated methods to run (default: "leiden louvain").
#
# Next step: align each method's output with run_clumppling.sh.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 DATA_DIR OUTPUT_DIR [CLUSTERING_OPTIONS...]" >&2
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTERING_SCRIPT="${SCRIPT_DIR}/../../run_scanpy_clustering.py"

for METHOD in ${METHODS:-leiden louvain}; do
    echo "=== Scanpy ${METHOD} ==="
    python "$CLUSTERING_SCRIPT" \
        --data_dir   "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --method     "$METHOD" \
        "$@"
done

echo "Done. Results in: ${OUTPUT_DIR}"
