#!/usr/bin/env bash
# Run fastTopics (a mixed-membership model) on a 10x count matrix for a range of K.
# Each K runs in turn. To run them in parallel, call this script once per K with
# K_MIN equal to K_MAX.
#
# Usage:
#   bash run_clustering_fasttopics.sh DATA_DIR OUTPUT_DIR K_MIN K_MAX [CLUSTERING_OPTIONS...]
#
# Arguments:
#   DATA_DIR      Directory containing the 10x matrix files.
#   OUTPUT_DIR    Directory where the .Q and .P files are written.
#   K_MIN, K_MAX  Inclusive range of K values.
#   CLUSTERING_OPTIONS
#                 Passed on to examples/run_fasttopics_clustering.R, for example
#                 --n_reps 20
#
# Next step: align the output with "run_clumppling.sh INPUT_DIR OUTPUT_DIR mmc".

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: bash $0 DATA_DIR OUTPUT_DIR K_MIN K_MAX [CLUSTERING_OPTIONS...]" >&2
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
K_MIN="$3"
K_MAX="$4"
shift 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTERING_SCRIPT="${SCRIPT_DIR}/../../run_fasttopics_clustering.R"

for K in $(seq "$K_MIN" "$K_MAX"); do
    echo "=== fastTopics K=${K} ==="
    Rscript "$CLUSTERING_SCRIPT" \
        --data_dir   "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --K          "$K" \
        "$@"
done

echo "Done. Results in: ${OUTPUT_DIR}"
