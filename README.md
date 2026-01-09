# ACE-OF-Clust (`ace-of-clust`)
ACE-OF-Clust (**A**lignment, **C**omparison, and **E**valuation of **O**mics **F**eatures in **Clust**ering) is a Python package built on top of [`clumppling`](https://pypi.org/project/clumppling/) that streamlines clustering-alignment workflows and supports downstream comparisons, summaries, and feature-level analyses for single-cell omics clustering results.

- **PyPI name (install):** `ace-of-clust`
- **Python import (module):** `ace_of_clust`
- **Current Version:** 0.1.0
- **Release Date:** Jan 2026

## Installation

```bash
pip install ace-of-clust
````

Optional (better label adjustment in some plots):

```bash
pip install "ace-of-clust[adjusttext]"
```

## Quickstart

### Run `clumppling` / `compModels` via wrappers

```python
from pathlib import Path

import ace_of_clust as aoc

# Example: run clumppling on an existing results directory / config
cls_dir = Path("input/clustering_res")
align_dir = Path("output/clumppling_run")
aoc.run_clumppling_via_main(
    input_dir=cls_dir,
    output_dir=align_dir,
    fmt="generalQ")

# Example: prepare and run compModels (paths/args will depend on your pipeline)
models = ['model1', 'model2']
suffixes = ["rep", "rep"]
model_dirs = [Path("output/clumppling_run_model_1") Path("output/clumppling_run_model_2")]
model_comp_dir = Path("output/clumppling_models")
qfilelists, qnamelists, mode_stats_files = aoc.prepare_comp_models_inputs(
    models=models,
    model_dirs=model_dirs,
    comp_dir=model_comp_dir,
    suffixes=suffixes,
)

model_comp_output_dir = Path("output/aligned_models")
aoc.run_comp_models(
    models=models,
    comp_dir=model_comp_dir,
    output_dir=model_comp_output_dir)

```

### Load, analyze, and visualize results

```python
import pandas as pd
import ace_of_clust as aoc

# load results
comp_res = aoc.load_compmodels_results(
    res_dir=model_comp_output_dir,
    input_dir=model_comp_dir,
)

# extract mode-pair mappings 
pair_mappings = aoc.extract_all_mode_pair_mappings(
    mode_names=comp_res.full_mode_names,
    all_modes_alignment=comp_res.all_modes_alignment,
    alignment_acrossK=comp_res.alignment_across_all,
)

# visualize cluster memberships (hard clustering)
fig, ax = aoc.plot_compmodels_membership_grid(
    comp_res,
    coords, # coordinates for scatter plot
    colors=colors,  # colors used for clusters
    val_threshold=0.5, # only plot points with membership values above this threshold
    suptitle="Cluster Memberships",
)
```

## Reproducing examples

This repo keeps example scripts/notebooks separate from the installable library code.
To reproduce examples:

1. Install the package (`pip install ace-of-clust`)
2. Clone this repository (for `examples/`, etc.)
3. Run the example scripts while using the installed package.

