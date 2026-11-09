# ACE-OF-Clust (`ace-of-clust`)

ACE-OF-Clust (**A**lignment, **C**omparison, and **E**valuation of **O**mics **F**eatures in **Clust**ering) is a Python package built on top of [`clumppling`](https://pypi.org/project/clumppling/) that streamlines clustering-alignment workflows and supports downstream comparisons, summaries, and feature-level analyses for single-cell omics clustering results.

- **PyPI name (install):** `ace-of-clust`
- **Python import (module):** `ace_of_clust`
- **Current Version:** 0.3.1
- **Last Update:** Nov 2026

Tutorials and API reference: [documentation site](https://xr-cc.github.io/ace-of-clust).

## Preprint

Liu, X., et al. (2026). Systematic clustering alignment and feature characterization for single-cell omics using ACE-OF-Clust. [https://doi.org/10.64898/2026.03.09.710668](https://doi.org/10.64898/2026.03.09.710668)

## Installation

```bash
pip install ace-of-clust
```

Optional, to repel overlapping labels in some plots:

```bash
pip install "ace-of-clust[adjusttext]"
```

## Quickstart

```python
import ace_of_clust as aoc

cls_dir = "path/to/clustering_results"   # one .Q file per run (plus .P for mixed membership)
align_dir = "path/to/alignment_results"

aoc.run_clumppling(input_dir=cls_dir, output_dir=align_dir, fmt="generalQ", extension=".Q", vis=False)
results = aoc.load_clumppling_results(align_dir=align_dir, suffix="rep", cls_dir=cls_dir, load_P=True)

fig, axes = aoc.plot_cluster_overlay(results, coords=X_coords, val_threshold=0.5)
```

`X_coords` holds 2D coordinates of the data points (e.g. UMAP). See the [documentation site](https://xr-cc.github.io/ace-of-clust) for model comparison with *compModels* and feature analysis.

## Reproducing Examples

The examples reproduce the figures in the paper. Install the package if you just want to use the tool, and clone this repository for `examples/` if you want to look at the data examples used in the paper.

The raw 10x Genomics datasets below are free to download after registering on the 10x Genomics website.

**Toy example**

- `examples/toy_example.ipynb`: walkthrough on simulated data
- `examples/prep_toy/generate_toy.ipynb`: generates the toy data

**PBMC3k scRNA-seq**

- Raw data: [3k PBMCs from a Healthy Donor](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0) (10x Genomics)
- `examples/pbmc3k_scRNAseq_clumppling.py`: *Clumppling* alignment of hard-clustering results
- `examples/pbmc3k_scRNAseq_compModels.py`: comparison across models
- `examples/pbmc3k_scRNAseq_hc_aligned.ipynb`: analysis of aligned hard-clustering results
- `examples/pbmc3k_scRNAseq_gs.ipynb` and `examples/run_gene_set_analysis.py`: gene set analysis of mixed-membership results

**HBC spatial transcriptomics** (data not included)

- Raw data: [Human Breast Cancer (Block A Section 1)](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) (10x Genomics)
- `docs/tutorials/ex2_hbc_mmc.ipynb`: analysis of aligned mixed-membership results, including gene set analysis

**PBMC10k Multiome** (data not included)

- Raw data: [PBMC from a Healthy Donor – Granulocytes Removed Through Cell Sorting (10k)](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0) (10x Genomics)
- `examples/pbmc10k_multiomic_mmc_aligned.ipynb`: RNA vs. ATAC clustering comparison

**Clustering and alignment scripts**

- `examples/run_scanpy_clustering.py`, `examples/run_seurat_clustering.R`, `examples/run_fasttopics_clustering.R`: clustering with Scanpy, Seurat and fastTopics
- `examples/scripts/new/`: bash scripts for clustering, *Clumppling*, *compModels* and gene set analysis on any data

## Bug Reports and Feedback

If you encounter any issues using the tool, have suggestions for additional functions to include in the package, or have general feedback on our method, please contact the author at [xiran_liu1 at brown dot edu](mailto:xiran_liu1@brown.edu).
