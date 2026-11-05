#!/usr/bin/env bash
# Align the clustering replicates in one directory with Clumppling.
# Run this once per clustering method.
#
# Usage:
#   bash run_clumppling.sh INPUT_DIR OUTPUT_DIR [hard|mmc]
#
# Arguments:
#   INPUT_DIR   Directory containing the clustering files.
#   OUTPUT_DIR  Clumppling output directory. Existing output is replaced.
#   CLUSTER_TYPE
#               hard: hard clusters such as Leiden or Louvain (default).
#               mmc:  mixed-membership models such as fastTopics. This tests
#                     community detection resolutions with comm_min = 1e-4.
#
# Environment variables:
#   FMT         Input format (default: generalQ).
#   EXTENSION   Extension of the input files (default: .Q).
#   IND_LABELS  File of individual labels (default: none). If labels are used,
#               Clumppling may regroup rows. See input/ind_labels_indices.txt in
#               the output.
#   VIS         Set to True to also write figures (default: False).
#
# A log is written next to the output directory, as OUTPUT_DIR.log.
# Next step: compare aligned runs with run_compmodels.sh.

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash $0 INPUT_DIR OUTPUT_DIR [hard|mmc]" >&2
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2%/}"
CLUSTER_TYPE="${3:-hard}"

case "$CLUSTER_TYPE" in
    hard) TEST_COMM=False; COMM_MIN=1e-6 ;;
    mmc)  TEST_COMM=True;  COMM_MIN=1e-4 ;;
    *)    echo "CLUSTER_TYPE must be hard or mmc, got: ${CLUSTER_TYPE}" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$OUTPUT_DIR")"

python - "$INPUT_DIR" "$OUTPUT_DIR" "$TEST_COMM" "$COMM_MIN" \
    "${FMT:-generalQ}" "${EXTENSION:-.Q}" "${IND_LABELS:-}" "${VIS:-False}" <<'PY'
import sys
from pathlib import Path

import ace_of_clust as aoc

input_dir, output_dir, test_comm, comm_min, fmt, extension, ind_labels, vis = sys.argv[1:]

aoc.run_clumppling(
    input_dir     = Path(input_dir),
    output_dir    = Path(output_dir),
    fmt           = fmt,
    extension     = extension,
    ind_labels    = ind_labels,
    vis           = vis == "True",
    use_rep       = True,
    use_best_pair = True,
    merge         = True,
    cd_res        = 1.0,
    test_comm     = test_comm == "True",
    comm_max      = 0.1,
    comm_min      = float(comm_min),
    setup_logging = True,
    log_file      = Path(f"{output_dir}.log"),
)
PY

echo "Done. Results in: ${OUTPUT_DIR}"
