#!/usr/bin/env bash
# Run the gene set enrichment analysis for one gene set on aligned clustering results.
#
# Usage:
#   bash run_gene_set_analysis.sh ALIGN_DIR CLS_DIR GENE_NAMES GS_FILE GS_MODE OUTPUT_DIR [ANALYSIS_OPTIONS...]
#
# Arguments:
#   ALIGN_DIR   Clumppling output directory from run_clumppling.sh.
#   CLS_DIR     Clustering directory holding the *.P files.
#   GENE_NAMES  File with one gene name per line, in the row order of the P matrices.
#   GS_FILE     File with one gene name per line.
#   GS_MODE     Mode for the single-mode analysis, for example K6M1.
#   OUTPUT_DIR  Directory for the figures.
#   ANALYSIS_OPTIONS
#               Passed on to examples/run_gene_set_analysis.py, for example
#               --gs_name "B cell program" --n_perm 10000 --n_modes 9

set -euo pipefail

if [ "$#" -lt 6 ]; then
    echo "Usage: bash $0 ALIGN_DIR CLS_DIR GENE_NAMES GS_FILE GS_MODE OUTPUT_DIR [ANALYSIS_OPTIONS...]" >&2
    exit 1
fi

ALIGN_DIR="$1"
CLS_DIR="$2"
GENE_NAMES="$3"
GS_FILE="$4"
GS_MODE="$5"
OUTPUT_DIR="$6"
shift 6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/../../run_gene_set_analysis.py" \
    --align_dir  "$ALIGN_DIR" \
    --cls_dir    "$CLS_DIR" \
    --gene_names "$GENE_NAMES" \
    --gs_file    "$GS_FILE" \
    --gs_mode    "$GS_MODE" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
