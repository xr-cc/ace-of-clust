#!/usr/bin/env Rscript
# fastTopics mixed-membership clustering for 10X scRNA-seq data.
#
# Produces Q and P matrix files compatible with ace-of-clust / clumppling.
#
# Usage:
#   Rscript run_fasttopics_clustering.R \
#       --data_dir   /path/to/10x_mtx_dir \
#       --output_dir /path/to/output \
#       --K          5 \
#       [--n_reps    20] \
#       [--min_features 200] \
#       [--max_features 2500] \
#       [--max_mito  5]
#
# Output layout:
#   output_dir/
#       fasttopics_mmc_genes.txt     # genes retained after QC
#       fasttopics_mmc_cells.txt     # cells retained after QC
#       ft-K{K}_1.Q, ft-K{K}_1.P    # per-rep Q and P matrices
#       ...
#       perf/ft-K{K}_perf.csv        # fitting diagnostics
#
# The Q files are named ft-K{K}_{rep}.Q so clumppling can parse K from the
# filename. Point cls_dir in pbmc3k_scRNAseq_clumppling.py at output_dir.

suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
    library(fastTopics)
    library(optparse)
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

option_list <- list(
    make_option("--data_dir",     type = "character",
                help = "Path to 10X MTX directory"),
    make_option("--output_dir",   type = "character",
                help = "Output directory for Q/P matrices and performance files"),
    make_option("--K",            type = "integer",
                help = "Number of topics (clusters)"),
    make_option("--n_reps",       type = "integer",   default = 20L,
                help = "Number of repeated runs with different seeds [default: 20]"),
    make_option("--min_features", type = "integer",   default = 200L,
                help = "Minimum features per cell [default: 200]"),
    make_option("--max_features", type = "integer",   default = 2500L,
                help = "Maximum features per cell [default: 2500]"),
    make_option("--max_mito",     type = "double",    default = 5,
                help = "Maximum mitochondrial percentage per cell [default: 5]")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$data_dir) || is.null(opt$output_dir) || is.null(opt$K)) {
    stop("--data_dir, --output_dir, and --K are required.")
}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

load_and_qc <- function(data_dir, min_features, max_features, max_mito) {
    cat("Loading data from:", data_dir, "\n")
    raw  <- Read10X(data.dir = data_dir)
    raw  <- raw[rowSums(raw) != 0, ]          # drop all-zero genes
    obj  <- CreateSeuratObject(counts = raw, project = "scRNAseq",
                               min.cells = 3, min.features = min_features)
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
    obj  <- subset(obj, subset =
        nFeature_RNA > min_features &
        nFeature_RNA < max_features &
        percent.mt   < max_mito)
    cat(sprintf("After QC: %d cells, %d genes\n", ncol(obj), nrow(obj)))
    obj
}

fit_and_save <- function(X, K, n_reps, output_dir) {
    # perf/ is placed alongside (not inside) output_dir so clumppling
    # does not mistake it for a matrix input directory.
    perf_dir <- file.path(dirname(output_dir), paste0(basename(output_dir), "_perf"))
    dir.create(perf_dir, recursive = TRUE, showWarnings = FALSE)

    df_perf <- data.frame(matrix(ncol = 7, nrow = 0))
    colnames(df_perf) <- c("K", "seed", "time",
                           "loglik", "loglik.multinom", "dev", "res")

    cat(sprintf("Fitting K=%d for %d reps ...\n", K, n_reps))
    for (i in seq_len(n_reps)) {
        cat(sprintf("  rep %d / %d\n", i, n_reps))
        ptm <- proc.time()
        set.seed(i)
        fit  <- fit_topic_model(X, k = K,
                                numiter.main   = 100, numiter.refine = 100,
                                method.main    = "em", method.refine = "scd")
        prog <- fit$progress[fit$iter, ]
        elapsed <- (proc.time() - ptm)[["elapsed"]]

        df_perf[nrow(df_perf) + 1, ] <- c(K, i, elapsed,
                                           prog$loglik, prog$loglik.multinom,
                                           prog$dev, prog$res)

        Q_file <- file.path(output_dir, sprintf("ft-K%d_%d.Q", K, i))
        P_file <- file.path(output_dir, sprintf("ft-K%d_%d.P", K, i))
        write.table(fit$L, file = Q_file, row.names = FALSE, col.names = FALSE)
        write.table(fit$F, file = P_file, row.names = FALSE, col.names = FALSE)
    }

    perf_file <- file.path(perf_dir, sprintf("ft-K%d_perf.csv", K))
    write.csv(df_perf, perf_file, row.names = FALSE)
    cat(sprintf("Performance saved to %s\n", perf_file))
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

# ── 1. Load & QC ──────────────────────────────────────────────────────────────
obj <- load_and_qc(opt$data_dir, opt$min_features, opt$max_features, opt$max_mito)

# ── 2. Build count matrix (cells × genes, sparse) ─────────────────────────────
X <- t(as(obj[["RNA"]]$counts, "sparseMatrix"))

# Remove genes that are all-zero after cell QC filtering.
# Cell subsetting can leave genes with no counts in the retained cells;
# fastTopics drops these internally, which would misalign P rows with gene names.
nonzero_genes <- colSums(X) > 0
X             <- X[, nonzero_genes]
gene_names_ft <- rownames(obj[["RNA"]]$counts)[nonzero_genes]
cat(sprintf("Count matrix: %d cells × %d genes (dropped %d all-zero genes)\n",
            nrow(X), ncol(X), sum(!nonzero_genes)))

# ── 3. Save gene / cell lists (written once; same across all K) ───────────────
genes_file <- file.path(opt$output_dir, "fasttopics_mmc_genes.txt")
cells_file <- file.path(opt$output_dir, "fasttopics_mmc_cells.txt")
if (!file.exists(genes_file)) {
    write.table(gene_names_ft, genes_file,
                quote = FALSE, row.names = FALSE, col.names = FALSE)
    write.table(colnames(obj[["RNA"]]$counts), cells_file,
                quote = FALSE, row.names = FALSE, col.names = FALSE)
    cat(sprintf("Saved %d genes and %d cells\n", ncol(X), ncol(obj)))
}

# ── 4. Fit topic model and save Q / P ─────────────────────────────────────────
fit_and_save(X, opt$K, opt$n_reps, opt$output_dir)

cat("Done.\n")
