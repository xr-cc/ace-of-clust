#!/usr/bin/env bash
# Regenerate the PBMC3k fastTopics clustering and Clumppling alignment results.
# These results are not stored in the repository because of their size (about 720M).
#
# Usage:
#   bash regenerate_pbmc3k_mmc.sh OUTPUT_DIR
#
# Arguments:
#   OUTPUT_DIR  Directory for all results. Use an empty directory: existing
#               results in it are overwritten.
#
# Environment variables:
#   K_MIN       Smallest K (default: 2).
#   K_MAX       Largest K (default: 13).
#   N_REPS      Replicates per K (default: 20).
#
# Output layout:
#   OUTPUT_DIR/
#       raw/filtered_gene_bc_matrices/hg19/     10x count matrix
#       clustering/full_mmc/                    *.Q, *.P, fasttopics_mmc_genes.txt
#       aligned/full_mmc/                       Clumppling output
#
# Requirements: curl, tar, Rscript, and Python with ace_of_clust installed.
# The fastTopics step can take many hours, so run this on a machine or compute
# job with enough time and memory. Results can differ numerically between
# package versions, so mode names and representatives may also differ.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash $0 OUTPUT_DIR" >&2
    exit 1
fi

OUTPUT_DIR="${1%/}"
K_MIN="${K_MIN:-2}"
K_MAX="${K_MAX:-13}"
N_REPS="${N_REPS:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_URL="https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
RAW_DIR="${OUTPUT_DIR}/raw"
DATA_DIR="${RAW_DIR}/filtered_gene_bc_matrices/hg19"
CLS_DIR="${OUTPUT_DIR}/clustering/full_mmc"
ALIGN_DIR="${OUTPUT_DIR}/aligned/full_mmc"

echo "=== 1/3 Download the 10x PBMC3k count matrix ==="
if [ -d "$DATA_DIR" ]; then
    echo "Found ${DATA_DIR}, skipping download."
else
    mkdir -p "$RAW_DIR"
    curl -fL "$DATA_URL" -o "${RAW_DIR}/pbmc3k.tar.gz"
    tar -xzf "${RAW_DIR}/pbmc3k.tar.gz" -C "$RAW_DIR"
    rm "${RAW_DIR}/pbmc3k.tar.gz"
fi

echo "=== 2/3 fastTopics clustering ==="
mkdir -p "$CLS_DIR"
bash "${SCRIPT_DIR}/run_clustering_fasttopics.sh" "$DATA_DIR" "$CLS_DIR" "$K_MIN" "$K_MAX" --n_reps "$N_REPS"

echo "=== 3/3 Clumppling alignment ==="
bash "${SCRIPT_DIR}/run_clumppling.sh" "$CLS_DIR" "$ALIGN_DIR" mmc

echo "Done. Clustering: ${CLS_DIR}"
echo "      Alignment:  ${ALIGN_DIR}"
