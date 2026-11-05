"""Gene set enrichment analysis of one gene set on aligned clustering results.

Tests whether a gene set is enriched across clusters, first in one focal mode
and then across all selected modes, and saves the figures.

Usage
-----
    python run_gene_set_analysis.py \\
        --align_dir  /path/to/clumppling_output \\
        --cls_dir    /path/to/clustering_output \\
        --gene_names /path/to/genes.txt \\
        --gs_file    /path/to/gene_set.txt \\
        --gs_mode    K6M1 \\
        --output_dir /path/to/gene_set_output \\
        [--gs_name "Gene set name"] \\
        [--n_perm 10000] \\
        [--rng_seed 1] \\
        [--n_modes 9] \\
        [--fig_format png] \\
        [--umap_coords umap.txt --ref_Q reference.Q --ref_labels labels.txt]

Inputs
------
    align_dir   Clumppling output directory.
    cls_dir     Clustering directory holding the *.P files.
    gene_names  One gene name per line, in the row order of the P matrices.
    gs_file     One gene name per line.

Output layout
-------------
    output_dir/
        {number}_{figure_name}.{fig_format}
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import ace_of_clust as aoc
from ace_of_clust.plot import make_mode_grid_by_K

CB_CMAP = [
    '#E69F00', '#009E73', '#CC79A7', '#56B4E9',
    '#F0E442', '#D55E00', '#0072B2', '#7F68A1',
    '#A3C644', '#FF1493', '#8B4513', '#20B2AA',
    '#9400D3', '#FF8C00',
]


class FigureSaver:
    """Save figures to output_dir with numbered file names."""

    def __init__(self, output_dir, fig_format):
        self.output_dir = Path(output_dir)
        self.fig_format = fig_format
        self.count = 0

    def __call__(self, fig, name):
        self.count += 1
        path = self.output_dir / f"{self.count:02d}_{name}.{self.fig_format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def plot_reference_umap(args, save):
    umap_coords = np.loadtxt(args.umap_coords, delimiter=" ")
    ref_labels = np.loadtxt(args.ref_labels, dtype=str, delimiter="\t")
    ref_Q = np.loadtxt(args.ref_Q, delimiter=" ")
    cell_type_labels = ref_labels[np.argmax(ref_Q, axis=1)]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for i, cell_type in enumerate(ref_labels):
        mask = cell_type_labels == cell_type
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   s=5, color=CB_CMAP[i % len(CB_CMAP)], alpha=0.8, label=cell_type)
        ax.text(np.mean(umap_coords[mask, 0]), np.mean(umap_coords[mask, 1]),
                cell_type,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
                fontsize=8, ha="left", va="center", color="black", weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Reference cell types", fontsize=10, weight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    save(fig, "reference_umap")


# ---------------------------------------------------------------------------
# Single-mode analysis
# ---------------------------------------------------------------------------

def plot_gene_analysis(P_gs, gene_set, sep_res, df_sepCls, df_sep_unordered,
                       cluster_labels, cluster_colors, gs_title, save):
    fig = plt.figure(figsize=(11, 10), dpi=200)
    gs  = GridSpec(2, 1, figure=fig, hspace=0.3)

    gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.2)
    gs_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3, width_ratios=[1, 2])

    # Top: per-gene mean LFC and per-gene extreme LFC
    for i_kind, kind in enumerate(["mean", "extreme"]):
        ax = fig.add_subplot(gs_top[i_kind])
        df_gene_lfc = aoc.compute_gene_lfc(P_gs, gene_set, sep_res["sepL"], sep_res["sepH"], kind=kind)
        aoc.plot_gene_lfc(df_gene_lfc, sep_res["sepL"], sep_res["sepH"],
                          sep_res["gs_sepLFC"], kind=kind, show_labels=False, ax=ax)
        ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(sep_res["gs_sepLFC"], color='gray', linestyle='--', linewidth=0.5)
        ax.text(sep_res["gs_sepLFC"], len(gene_set)-1, f"GS sepLFC = {sep_res['gs_sepLFC']:.4f}",
                ha="left", va="top", fontsize=8, color="gray", style="italic")
        ax.set_title(f"Per-gene {kind} LFC between the GS sepCls", fontsize=11, weight='normal', y=1.01)

    # Bottom left: top genes by sepLFC
    ax = fig.add_subplot(gs_bot[0])
    sns.barplot(x='sepLFC', y='gene', data=df_sepCls, hue='sepCls', palette='Paired',
                ax=ax, legend=False)
    existing_labels = set()
    for i_gene, row in df_sepCls.iterrows():
        low_str  = "Cls."+"+".join(str(k+1) for k in row['sepCls'][0])
        high_str = "Cls."+"+".join(str(k+1) for k in row['sepCls'][1])
        sep_cls  = f"[{high_str}]/[{low_str}]"
        if sep_cls in existing_labels:
            sep_cls = ""
        else:
            existing_labels.add(sep_cls)
        if row['sepLFC'] > 10 or max(df_sepCls['sepLFC']) < 15:
            ax.text(0.1, i_gene, sep_cls, rotation=0, ha='left', va='center', fontsize=8)
        else:
            ax.text(row['sepLFC'] + 0.1, i_gene, sep_cls, rotation=0, ha='left', va='center', fontsize=8)
    if len(df_sepCls) < 2:
        x_max = max(df_sepCls['sepLFC']) * 1.5
        ax.set_xlim(0, x_max)
        ax.set_ylim(-1.5, len(df_sepCls) + 0.5)
    ax.set_title("Top genes by sepLFC for GS sepCls", fontsize=11, weight='normal', y=1.01)

    # Bottom right: sepLFC distribution
    ax = fig.add_subplot(gs_bot[1])
    aoc.plot_sepLFC_distribution(
        df_sep_unordered, gene_set, title="", kind="auto",
        show_gs_textbox=True, gs_textbox_threshold=10, n_gs_textbox=10,
        show_non_gs_textbox=True, n_non_gs_textbox=10, ax=ax,
    )
    ax.set_title(
        f"sepLFC for (unordered) GS sepCls: "
        f"{len(set(df_sep_unordered.index) & set(gene_set))} / {len(gene_set)} GS genes included",
        fontsize=11, weight='normal', y=1.01
    )

    fig.suptitle(f"{gs_title}", fontsize=12, weight='bold', y=0.94)
    save(fig, "gene_analysis")


def run_single_mode(results, df_mode, gene_set, gene_set_indices, args, gs_name, save):
    gs_mode        = args.gs_mode
    K_mode         = results.mode_K[gs_mode]
    cluster_labels = [f"Cls.{i+1}" for i in range(K_mode)]
    cluster_colors = CB_CMAP[:K_mode]
    gs_title       = f"{gs_name}  (mode {gs_mode})"
    print(f"gs_mode: {gs_mode}  |  K={K_mode}")

    # Relative expression matrix P: G×K; restrict to gene-set rows
    P    = results.P_aligned_by_mode[gs_mode]    # (G, K)
    P_gs = P[gene_set_indices]                   # (|S|, K)
    print(f"P shape: {P.shape}  |  P_gs shape: {P_gs.shape}")

    # Sample random gene sets of size |S| -> shared null mean-P vectors
    null_mean_P = aoc.sample_null_P(P, gene_set_indices, args.n_perm, args.rng_seed)

    # 1. P enrichment: compare the gene-set mean P to the null
    p_res = aoc.test_P_enrichment(P, gene_set_indices, null_mean_P)
    print("gs_mean_P:", dict(zip(cluster_labels, p_res["gs_mean_P"].round(6))))
    print("p_emp:    ", dict(zip(cluster_labels, p_res["p_emp"].round(4))))

    # 2. Pairwise LFC: compare pairwise LFC to the null (BH-corrected q-values)
    lfc_res = aoc.test_LFC_enrichment(P, gene_set_indices, null_mean_P)

    # 3. SepLFC: largest LFC gap in clusters sorted by mean P
    sep_res = aoc.test_sepLFC_enrichment(P, gene_set_indices, null_mean_P)
    print(f"\nsepLFC           = {sep_res['gs_sepLFC']:.4f}")
    print(f"geometric sepLFC = {sep_res['gs_geomean_LFC']:.4f}")
    print(f"  low:  {[cluster_labels[k] for k in sep_res['sepL']]}")
    print(f"  high: {[cluster_labels[k] for k in sep_res['sepH']]}")
    print(f"  p (vs null best):           {sep_res['p_vs_null_sep']:.4e}")
    print(f"  p (vs null fixed clusters): {sep_res['p_vs_null_fixed']:.4e}")

    # P enrichment
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    aoc.plot_P_enrichment_pval(p_res, cluster_labels, cluster_colors, ax=axes[0])
    aoc.plot_P_enrichment_zscore(p_res, cluster_labels, cluster_colors, ax=axes[1])
    fig.suptitle(f"{gs_title} – P enrichment", fontsize=11, weight="bold")
    fig.tight_layout()
    save(fig, "P_enrichment")

    # Pairwise LFC enrichment
    fig, _ = aoc.plot_pairwise_heatmap(
        lfc_res["obs_lfc"], sig_mat=lfc_res["test"]["q_mat"],
        labels=cluster_labels, title="Observed pairwise LFC log2(y/x)", upper_only=False,
    )
    save(fig, "pairwise_LFC_observed")
    fig, _ = aoc.plot_pairwise_heatmap(
        lfc_res["test"]["z_mat"], sig_mat=lfc_res["test"]["q_mat"],
        labels=cluster_labels, title="LFC deviation from null – z-score", upper_only=False,
    )
    save(fig, "pairwise_LFC_zscore")
    fig, _ = aoc.plot_top_pairwise_df(
        lfc_res["df"], value_col="z", sig_col="q", alpha=0.05, labels=cluster_labels,
    )
    save(fig, "pairwise_LFC_top")
    fig, _ = aoc.plot_pairwise_heatmap_bidir(
        upper_mat=lfc_res["test"]["z_mat"],
        lower_mat=lfc_res["obs_lfc"],
        upper_sig=lfc_res["test"]["q_mat"],
        lower_sig=lfc_res["test"]["q_mat"],
        upper_label="z-score",
        lower_label="obs LFC",
        labels=cluster_labels,
        title=f"{gs_title} – pairwise LFC",
    )
    save(fig, "pairwise_LFC_combined")

    # SepLFC
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    aoc.plot_sepLFC_null_sep(sep_res, ax=axes[0])
    aoc.plot_sepLFC_null_fixed(sep_res, cluster_labels, ax=axes[1])
    fig.suptitle(f"{gs_title} – sepLFC enrichment", fontsize=11, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "sepLFC_enrichment")

    # Per-gene P
    fig, _, _ = aoc.plot_per_cluster_P(
        P_gs, gene_set, cluster_labels, cluster_colors,
        null_mean_P=null_mean_P,
        gs_title=gs_title,
    )
    save(fig, "per_cluster_P")
    fig, _ = aoc.plot_gene_P_stacked(
        P_gs, gene_set, cluster_labels, gs_title=gs_title, sort_by_sum=True, top_n=10
    )
    save(fig, "gene_P_stacked")
    fig, _ = aoc.plot_gene_P_bars(P_gs, gene_set, cluster_labels, cluster_colors, top_n=10)
    save(fig, "gene_P_bars")

    # Gene analysis: per-gene LFC between the separating clusters
    df_sepCls = aoc.analyze_sep_genes(df_mode, sep_res['sepH'], sep_res['sepL'],
                                      gene_set, top_n=15)
    df_sepCls['gene'] = df_sepCls.index

    sepL_t = tuple(int(x) for x in sep_res['sepL'])
    sepH_t = tuple(int(x) for x in sep_res['sepH'])

    # Ordered match: sepCls == (sepL, sepH) exactly
    df_sep_ordered = df_mode[df_mode['sepCls'] == (sepL_t, sepH_t)].copy()

    # Unordered match: any permutation of sepL and sepH groups
    sepCls_unordered = {
        (permL, permH)
        for permL in itertools.permutations(sepL_t)
        for permH in itertools.permutations(sepH_t)
    }
    df_sep_unordered = df_mode[df_mode['sepCls'].apply(lambda x: x in sepCls_unordered)].copy()

    for _df in [df_sep_ordered, df_sep_unordered]:
        _df['rank_sepLFC'] = _df['sepLFC'].rank(ascending=False, method='min').astype(int)
    df_sep_ordered   = df_sep_ordered.sort_values('sepLFC', ascending=False)
    df_sep_unordered = df_sep_unordered.sort_values('sepLFC', ascending=False)

    print(f"Genes in ordered sepCls:   {len(set(df_sep_ordered.index)  & set(gene_set))} / {len(gene_set)}")
    print(f"Genes in unordered sepCls: {len(set(df_sep_unordered.index) & set(gene_set))} / {len(gene_set)}")

    plot_gene_analysis(P_gs, gene_set, sep_res, df_sepCls, df_sep_unordered,
                       cluster_labels, cluster_colors, gs_title, save)

    print("Top non-GS genes by sepLFC in unordered sepCls:")
    print(df_sep_unordered[~df_sep_unordered.index.isin(gene_set)].head(10).index)

    df_gene_lfc = aoc.compute_gene_lfc(
        P_gs, gene_set, sep_res["sepL"], sep_res["sepH"], kind="mean"
    )
    fig, _ = aoc.plot_gene_lfc(
        df_gene_lfc,
        sep_res["sepL"], sep_res["sepH"],
        sep_res["gs_sepLFC"], kind="mean", show_labels=True,
    )
    save(fig, "gene_LFC")

    fig, _ = aoc.plot_seplfc_bipartite(
        gene_set, sep_res["sepH"], sep_res["sepL"], cluster_labels,
        df_mode=df_mode,
        top_n_per_pair=5,
        gs_title=gs_title,
        colors=cluster_colors,
        cmap='cividis',
        figsize=(8, 5),
    )
    save(fig, "sepLFC_bipartite")


# ---------------------------------------------------------------------------
# Across all modes
# ---------------------------------------------------------------------------

def run_across_modes(results, gene_set_indices, args, gs_name, save):
    res_by_mode = aoc.run_gs_enrichment(
        results, gene_set_indices, results.modes, args.n_perm, args.rng_seed
    )

    # P enrichment
    fig, ax_by_mode = make_mode_grid_by_K(results)
    aoc.plot_P_enrichment_grid(res_by_mode, ax_by_mode, results, CB_CMAP, kind="pval")
    fig.suptitle(f"{gs_name} – P enrichment across modes", fontsize=11, weight="bold", y=0.99)
    fig.tight_layout()
    save(fig, "modes_P_enrichment_grid")

    fig, _ = aoc.plot_P_enrichment_by_cluster(
        res_by_mode, results, CB_CMAP, kind="pval", ncols=4, sig_threshold=0.05
    )
    fig.suptitle(f"{gs_name} – P enrichment by cluster across modes",
                 fontsize=11, weight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "modes_P_enrichment_by_cluster")

    fig, _ = aoc.plot_P_enrichment_heatmap(
        res_by_mode, results, value="z",
        title=f"{gs_name} – P enrichment across modes",
        cmap="Blues",
    )
    save(fig, "modes_P_enrichment_heatmap")

    # Pairwise LFC
    fig, ax_by_mode = make_mode_grid_by_K(results)
    aoc.plot_LFC_enrichment_grid(res_by_mode, ax_by_mode, results, CB_CMAP)
    fig.suptitle(f"{gs_name} – pairwise LFC z-score across modes", fontsize=11, weight="bold", y=0.99)
    fig.tight_layout()
    save(fig, "modes_LFC_grid")

    fig, _ = aoc.plot_LFC_enrichment_heatmap(
        res_by_mode, results, value="z",
        title=f"{gs_name} – pairwise LFC z-score across modes",
    )
    save(fig, "modes_LFC_heatmap")

    # sepLFC
    fig, ax_by_mode = make_mode_grid_by_K(results)
    aoc.plot_sepLFC_enrichment_grid(res_by_mode, ax_by_mode, results, CB_CMAP, kind="null_sep")
    fig.suptitle(f"{gs_name} – sepLFC enrichment across modes", fontsize=11, weight="bold", y=0.99)
    fig.tight_layout()
    save(fig, "modes_sepLFC_grid")

    fig, _ = aoc.plot_sepLFC_distribution_heatmap(
        res_by_mode, n_bins=60, kind='null_sep', pval_threshold=0.05, annotate_pval=True
    )
    fig.suptitle(f"{gs_name} – null sepLFC distributions across modes (largest sepLFC)",
                 fontsize=11, weight="bold", x=0.45, y=1.01)
    fig.tight_layout()
    save(fig, "modes_sepLFC_null_best")

    fig, _ = aoc.plot_sepLFC_distribution_heatmap(
        res_by_mode, n_bins=60, kind='null_fixed', pval_threshold=0.05, annotate_pval=True
    )
    fig.suptitle(f"{gs_name} – null sepLFC distributions across modes (LFC across GS sepCls)",
                 fontsize=11, weight="bold", x=0.45, y=1.01)
    fig.tight_layout()
    save(fig, "modes_sepLFC_null_fixed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gene set enrichment analysis of one gene set")
    p.add_argument("--align_dir",  required=True,
                   help="Clumppling output directory")
    p.add_argument("--cls_dir",    required=True,
                   help="Clustering directory holding the *.P files")
    p.add_argument("--gene_names", required=True,
                   help="File with one gene name per line, in the row order of the P matrices")
    p.add_argument("--gs_file",    required=True,
                   help="File with one gene name per line")
    p.add_argument("--gs_mode",    required=True,
                   help="Mode for the single-mode analysis, e.g. K6M1")
    p.add_argument("--output_dir", required=True,
                   help="Directory for the figures")
    p.add_argument("--gs_name",    default=None,
                   help="Gene set name in figure titles (default: gs_file name)")
    p.add_argument("--n_perm",     type=int, default=10000,
                   help="Number of random gene sets in the null")
    p.add_argument("--rng_seed",   type=int, default=1,
                   help="Random seed")
    p.add_argument("--n_modes",    type=int, default=None,
                   help="Use only the first n modes (default: all modes)")
    p.add_argument("--fig_format", default="png",
                   help="Figure file format")
    p.add_argument("--umap_coords", default=None,
                   help="Optional: UMAP coordinates of the cells, for a reference plot")
    p.add_argument("--ref_Q",      default=None,
                   help="Optional: reference .Q file, for a reference plot")
    p.add_argument("--ref_labels", default=None,
                   help="Optional: reference cluster labels, for a reference plot")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save = FigureSaver(output_dir, args.fig_format)
    gs_name = args.gs_name or Path(args.gs_file).stem.replace("_", " ")

    reference = [args.umap_coords, args.ref_Q, args.ref_labels]
    if any(reference) and not all(reference):
        sys.exit("Set all of --umap_coords, --ref_Q and --ref_labels, or none of them.")
    if all(reference):
        plot_reference_umap(args, save)

    gene_names = np.loadtxt(args.gene_names, dtype=str)
    print(f"{len(gene_names)} genes")

    results = aoc.load_clumppling_results(
        align_dir = args.align_dir,
        suffix    = "rep",
        cls_dir   = args.cls_dir,
        load_P    = True,
        strict_P  = True,
    )

    # Per-gene metrics for all modes (used in the gene analysis)
    df_by_mode = aoc.compute_all_feature_metrics(results, feature_names=gene_names)

    results = aoc.subset_results(results, results.modes[:args.n_modes])
    print("Modes:", results.modes)
    if args.gs_mode not in results.modes:
        sys.exit(f"--gs_mode {args.gs_mode} is not in the selected modes: {results.modes}")

    gs_raw = np.loadtxt(args.gs_file, dtype=str, ndmin=1)
    gene_set, gene_set_indices = aoc.resolve_gene_set(gs_raw.tolist(), gene_names)
    print(f"{len(gene_set)} / {len(gs_raw)} genes matched for {gs_name}:\n{', '.join(gene_set)}")

    print("\n=== Single-mode analysis ===")
    run_single_mode(results, df_by_mode[args.gs_mode], gene_set, gene_set_indices,
                    args, gs_name, save)

    print("\n=== Across all modes ===")
    run_across_modes(results, gene_set_indices, args, gs_name, save)

    print("Done.")


if __name__ == "__main__":
    main()
