"""Scanpy preprocessing + clustering script for 10X scRNA-seq data.

Produces Q-matrix files compatible with ace-of-clust / clumppling.

Usage
-----
    python run_scanpy_clustering.py \\
        --data_dir  /path/to/10x_mtx_dir \\
        --output_dir /path/to/clustering_output \\
        [--method leiden|louvain] \\
        [--n_reps 10] \\
        [--resolution 1.0] \\
        [--n_neighbors 15] \\
        [--n_pcs 10] \\
        [--min_genes 200] \\
        [--max_genes 2500] \\
        [--max_mito 0.05]

Output layout
-------------
    output_dir/
        scanpy_hc_genes.txt          # HVGs selected during preprocessing
        scanpy_hc_cells.txt          # cells retained after QC
        hvg_hc/
            scanpy_{method}/
                scanpy_{method}_0_K{K}.Q
                scanpy_{method}_1_K{K}.Q
                ...

This layout matches what pbmc3k_scRNAseq_clumppling.py expects as input.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------------
# Gene-name helpers
# ---------------------------------------------------------------------------

def _restore_canonical_gene_names(var_names):
    """Undo var_names_make_unique() name corruption.

    var_names_make_unique() deduplicates by appending '-1', '-2', ... to
    subsequent occurrences of the same symbol (e.g. a duplicate 'TMBIM4'
    becomes 'TMBIM4-1').  For genes whose canonical HGNC symbol actually
    ends in '.N' (e.g. 'TMBIM4.1'), this produces an incorrect name.

    We detect the pattern: a gene named 'BASE-N' where 'BASE' (without
    the suffix) also exists in the list — meaning 'BASE-N' was created by
    deduplication — and restore it to 'BASE.N'.

    Reference: pbmc_hc_analysis.ipynb applies the equivalent manual fix
    (TMBIM4-1 → TMBIM4.1, NDUFB8-1 → NDUFB8.1) after loading scanpy genes.
    """
    names = list(var_names)
    base_names = set(names)          # includes both BASE and BASE-N
    pattern = re.compile(r'^(.+)-(\d+)$')
    fixed = []
    for name in names:
        m = pattern.match(name)
        # Only convert if the bare base name also exists (confirming dedup)
        if m and m.group(1) in base_names:
            fixed.append(f"{m.group(1)}.{m.group(2)}")
        else:
            fixed.append(name)
    return fixed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Scanpy clustering for 10X scRNA-seq data")
    p.add_argument("--data_dir",   required=True,
                   help="Path to 10X MTX directory (contains matrix.mtx, barcodes.tsv, genes.tsv)")
    p.add_argument("--output_dir", required=True,
                   help="Base output directory for clustering results")
    p.add_argument("--method",     default="leiden", choices=["leiden", "louvain"],
                   help="Clustering algorithm (default: leiden)")
    p.add_argument("--n_reps",     type=int,   default=10,
                   help="Number of repeated clustering runs with different seeds (default: 10)")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Clustering resolution (default: 1.0)")
    p.add_argument("--n_neighbors",type=int,   default=15,
                   help="Number of neighbors for the kNN graph (default: 15)")
    p.add_argument("--n_pcs",      type=int,   default=10,
                   help="Number of PCs used for neighbor graph (default: 10)")
    p.add_argument("--min_genes",  type=int,   default=200,
                   help="Minimum genes per cell (default: 200)")
    p.add_argument("--max_genes",  type=int,   default=2500,
                   help="Maximum genes per cell (default: 2500)")
    p.add_argument("--max_mito",   type=float, default=0.05,
                   help="Maximum mitochondrial fraction per cell (default: 0.05)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_and_qc(data_dir, min_genes, max_genes, max_mito):
    """Load 10X data and apply basic QC filters."""
    print("Loading data from:", data_dir)
    adata = sc.read_10x_mtx(data_dir, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    adata.var_names = _restore_canonical_gene_names(adata.var_names)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)

    mito_genes = adata.var_names.str.startswith("MT-")
    adata.obs["percent_mito"] = (
        np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    )
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1

    adata = adata[adata.obs.n_genes > min_genes, :]
    adata = adata[adata.obs.n_genes < max_genes, :]
    adata = adata[adata.obs.percent_mito < max_mito, :]

    print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def preprocess(adata):
    """Normalise, select HVGs, regress out confounders, scale, run PCA."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_bins=20,
        min_mean=0.1, max_mean=8, min_disp=1, max_disp=np.inf,
    )
    print(f"HVGs selected: {adata.var['highly_variable'].sum()}")

    sc.pp.regress_out(adata, ["n_counts", "percent_mito"])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, use_highly_variable=True, svd_solver="arpack")
    return adata


def build_neighbor_graph(adata, n_neighbors, n_pcs):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, knn=True,
                    use_rep="X_pca", n_pcs=n_pcs)
    return adata


def compute_and_save_umap(adata, output_dir):
    """Compute UMAP on the existing neighbor graph and save coordinates."""
    sc.tl.umap(adata, random_state=0)
    umap_path = os.path.join(output_dir, "scanpy_umap.txt")
    np.savetxt(umap_path, adata.obsm["X_umap"])
    print(f"UMAP saved ({adata.n_obs} cells × 2): {umap_path}")
    return adata


def cluster_once(adata, method, resolution, seed):
    if method == "louvain":
        sc.tl.louvain(adata, resolution=resolution,
                      random_state=seed, flavor="vtraag")
        return adata.obs["louvain"].astype("category")
    else:
        sc.tl.leiden(adata, resolution=resolution,
                     random_state=seed, flavor="leidenalg")
        return adata.obs["leiden"].astype("category")


def save_q_matrices(cluster_ids_reps, method_lb, hc_method_dir, n_reps):
    eps = 1e-10
    for i in range(n_reps):
        ids = cluster_ids_reps[:, i]
        Q = pd.get_dummies(ids).astype(float).values
        K = Q.shape[1]
        Q = (Q + eps) / (Q + eps).sum(axis=1, keepdims=True)
        fname = os.path.join(hc_method_dir, f"{method_lb}_{i}_K{K}.Q")
        np.savetxt(fname, Q, delimiter=" ")
        print(f"  rep {i}: K={K}  ->  {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    method_lb = f"scanpy_{args.method}"

    # Output directories
    hc_output_dir  = os.path.join(args.output_dir, "hvg_hc")
    hc_method_dir  = os.path.join(hc_output_dir, method_lb)
    os.makedirs(hc_method_dir, exist_ok=True)

    # ── 1. Load & QC ──────────────────────────────────────────────────────────
    adata = load_and_qc(args.data_dir, args.min_genes, args.max_genes, args.max_mito)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    adata = preprocess(adata)

    # ── 3. Neighbor graph ─────────────────────────────────────────────────────
    adata = build_neighbor_graph(adata, args.n_neighbors, args.n_pcs)

    # ── 4. UMAP ───────────────────────────────────────────────────────────────
    adata = compute_and_save_umap(adata, args.output_dir)

    # ── 5. Save gene / cell lists ─────────────────────────────────────────────
    selected_genes = adata.var_names[adata.var["highly_variable"]]
    np.savetxt(os.path.join(args.output_dir, "scanpy_hc_genes.txt"), selected_genes, fmt="%s")
    np.savetxt(os.path.join(args.output_dir, "scanpy_hc_cells.txt"), adata.obs_names, fmt="%s")
    print(f"Saved {len(selected_genes)} HVGs and {adata.n_obs} cell barcodes")

    # ── 6. Repeated clustering ────────────────────────────────────────────────
    print(f"Running {args.n_reps} clustering reps (method={method_lb}, "
          f"resolution={args.resolution}) ...")
    reps = []
    for i in range(args.n_reps):
        ids = cluster_once(adata, args.method, args.resolution, seed=i)
        print(f"  rep {i}: {ids.nunique()} clusters")
        reps.append(ids)

    cluster_ids_reps = pd.concat(reps, axis=1).values

    # ── 7. Save Q matrices ────────────────────────────────────────────────────
    print("Saving Q matrices ...")
    save_q_matrices(cluster_ids_reps, method_lb, hc_method_dir, args.n_reps)

    print("Done.")


if __name__ == "__main__":
    main()
