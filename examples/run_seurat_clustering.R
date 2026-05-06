#!/usr/bin/env Rscript
# Seurat preprocessing + clustering script for 10X scRNA-seq data.
#
# Produces Q-matrix files compatible with ace-of-clust / clumppling.
#
# Usage:
#   Rscript run_seurat_clustering.R \
#       --data_dir  /path/to/10x_mtx_dir \
#       --output_dir /path/to/clustering_output \
#       [--method louvain|leiden] \
#       [--n_reps 10] \
#       [--resolution 1.0] \
#       [--n_neighbors 15] \
#       [--n_pcs 10] \
#       [--min_features 200] \
#       [--max_features 2500] \
#       [--max_mito 5]
#
# Output layout:
#   output_dir/
#       seurat_hc_genes.txt
#       seurat_hc_cells.txt
#       hvg_hc/
#           seurat_{method}/
#               seurat_{method}_1_K{K}.Q
#               seurat_{method}_2_K{K}.Q
#               ...
#
# This layout matches what pbmc3k_scRNAseq_clumppling.py expects.

suppressPackageStartupMessages({
    library(Seurat)
    library(optparse)
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

option_list <- list(
    make_option("--data_dir",     type = "character",
                help = "Path to 10X MTX directory (contains matrix.mtx, barcodes.tsv, features.tsv)"),
    make_option("--output_dir",   type = "character",
                help = "Base output directory for clustering results"),
    make_option("--method",       type = "character", default = "louvain",
                help = "Clustering algorithm: louvain (default) or leiden"),
    make_option("--n_reps",       type = "integer",   default = 10L,
                help = "Number of repeated clustering runs [default: 10]"),
    make_option("--resolution",   type = "double",    default = 1.0,
                help = "Clustering resolution [default: 1.0]"),
    make_option("--n_neighbors",  type = "integer",   default = 15L,
                help = "k for kNN graph [default: 15]"),
    make_option("--n_pcs",        type = "integer",   default = 10L,
                help = "Number of PCs used for neighbor graph [default: 10]"),
    make_option("--min_features", type = "integer",   default = 200L,
                help = "Minimum features (genes) per cell [default: 200]"),
    make_option("--max_features", type = "integer",   default = 2500L,
                help = "Maximum features (genes) per cell [default: 2500]"),
    make_option("--max_mito",     type = "double",    default = 5.0,
                help = "Maximum mitochondrial percentage per cell [default: 5]")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$data_dir) || is.null(opt$output_dir)) {
    stop("--data_dir and --output_dir are required.")
}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

load_and_qc <- function(data_dir, min_features, max_features, max_mito) {
    cat("Loading data from:", data_dir, "\n")
    raw <- Read10X(data.dir = data_dir)
    obj <- CreateSeuratObject(counts = raw, project = "scRNAseq",
                               min.cells = 3, min.features = min_features)
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
    obj <- subset(obj, subset =
        nFeature_RNA > min_features &
        nFeature_RNA < max_features &
        percent.mt   < max_mito)
    cat(sprintf("After QC: %d cells, %d genes\n",
                ncol(obj), nrow(obj)))
    obj
}

preprocess <- function(obj, n_pcs) {
    obj <- NormalizeData(obj, normalization.method = "LogNormalize",
                         scale.factor = 10000)
    obj <- FindVariableFeatures(obj, selection.method = "mean.var.plot",
                                num.bin = 20,
                                mean.cutoff     = c(0.1, 8),
                                dispersion.cutoff = c(1, Inf))
    cat(sprintf("HVGs selected: %d\n", length(VariableFeatures(obj))))

    obj <- ScaleData(obj,
                     vars.to.regress = c("nCount_RNA", "percent.mt"),
                     features = rownames(obj),
                     do.scale = TRUE, do.center = TRUE, scale.max = 10)
    obj <- RunPCA(obj, features = VariableFeatures(obj), verbose = FALSE)
    obj
}

build_neighbor_graph <- function(obj, n_neighbors, n_pcs) {
    obj <- FindNeighbors(obj, k.param = n_neighbors,
                         reduction = "pca", dims = seq_len(n_pcs))
    obj
}

compute_and_save_umap <- function(obj, output_dir, n_pcs) {
    obj <- RunUMAP(obj, reduction = "pca", dims = seq_len(n_pcs), seed.use = 0)
    umap_coords <- Embeddings(obj, reduction = "umap")
    umap_path <- file.path(output_dir, "seurat_umap.txt")
    write.table(umap_coords, umap_path, row.names = FALSE, col.names = FALSE)
    cat(sprintf("UMAP saved (%d cells x 2): %s\n", nrow(umap_coords), umap_path))
    obj
}

cluster_once <- function(obj, algorithm, resolution, seed) {
    obj <- FindClusters(obj, resolution = resolution,
                        algorithm = algorithm, random.seed = seed)
    obj$seurat_clusters
}

save_q_matrices <- function(cluster_ids_reps, method_lb, hc_method_dir, n_reps) {
    eps <- 1e-10
    for (i in seq_len(n_reps)) {
        c   <- factor(cluster_ids_reps[, i],
                      levels = sort(unique(cluster_ids_reps[, i])))
        df  <- data.frame(cluster_id = c)
        Q   <- model.matrix(~ cluster_id - 1, df)
        Q   <- unname(Q)
        K   <- ncol(Q)
        Q   <- Q + eps
        Q   <- Q / rowSums(Q)
        fname <- file.path(hc_method_dir,
                           sprintf("%s_%d_K%d.Q", method_lb, i, K))
        write.table(Q, file = fname, row.names = FALSE, col.names = FALSE)
        cat(sprintf("  rep %d: K=%d  ->  %s\n", i, K, fname))
    }
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Algorithm number: Louvain = 1, Leiden = 4
algorithm <- if (opt$method == "leiden") 4L else 1L
method_lb <- paste0("seurat_", opt$method)

# Output directories
hc_output_dir <- file.path(opt$output_dir, "hvg_hc")
hc_method_dir <- file.path(hc_output_dir, method_lb)
dir.create(hc_method_dir, recursive = TRUE, showWarnings = FALSE)

# ── 1. Load & QC ──────────────────────────────────────────────────────────────
obj <- load_and_qc(opt$data_dir, opt$min_features, opt$max_features, opt$max_mito)

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
obj <- preprocess(obj, opt$n_pcs)

# ── 3. Neighbor graph ─────────────────────────────────────────────────────────
obj <- build_neighbor_graph(obj, opt$n_neighbors, opt$n_pcs)

# ── 4. UMAP ───────────────────────────────────────────────────────────────────
obj <- compute_and_save_umap(obj, opt$output_dir, opt$n_pcs)

# ── 5. Save gene / cell lists ─────────────────────────────────────────────────
selected_genes <- VariableFeatures(obj)
cell_names     <- colnames(obj[["RNA"]]$data)
write.table(selected_genes, file.path(opt$output_dir, "seurat_hc_genes.txt"),
            quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(cell_names, file.path(opt$output_dir, "seurat_hc_cells.txt"),
            quote = FALSE, row.names = FALSE, col.names = FALSE)
cat(sprintf("Saved %d HVGs and %d cell barcodes\n",
            length(selected_genes), length(cell_names)))

# ── 6. Repeated clustering ────────────────────────────────────────────────────
cat(sprintf("Running %d clustering reps (method=%s, resolution=%.2f) ...\n",
            opt$n_reps, method_lb, opt$resolution))
reps <- vector("list", opt$n_reps)
for (i in seq_len(opt$n_reps)) {
    ids <- cluster_once(obj, algorithm, opt$resolution, seed = i)
    reps[[i]] <- ids
    cat(sprintf("  rep %d: %d clusters\n", i, length(unique(ids))))
}
cluster_ids_reps <- do.call(cbind, reps)

# ── 7. Save Q matrices ────────────────────────────────────────────────────────
cat("Saving Q matrices ...\n")
save_q_matrices(cluster_ids_reps, method_lb, hc_method_dir, opt$n_reps)

cat("Done.\n")
