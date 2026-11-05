# Run Clumppling

**Dataset**: [10x PBMC3k](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0) scRNA-seq.
**Clustering**: Seurat and Scanpy, each with Louvain and Leiden (`examples/data/pbmc3k/clustering/hvg_hc/`).

We cover:
1. Aligning each method's runs with *Clumppling*
2. Comparing the methods with *compModels*

The same steps as scripts: `examples/pbmc3k_scRNAseq_clumppling.py` and `examples/pbmc3k_scRNAseq_compModels.py`. For other data, see `examples/scripts/new/run_clumppling.sh` and `run_compmodels.sh`.

## Imports

```python
from pathlib import Path
import ace_of_clust as aoc
```

## Paths

```python
base_dir = Path("path/to/ace-of-clust")
example_data_dir = base_dir / "examples" / "data" / "pbmc3k"

methods = ["seurat_louvain", "seurat_leiden", "scanpy_louvain", "scanpy_leiden"]
```

## Run *Clumppling*

Align the runs of each method separately.

```python
for method in methods:
    aoc.run_clumppling(
        input_dir=example_data_dir / "clustering" / "hvg_hc" / method,
        output_dir=example_data_dir / "aligned" / "hvg_hc" / method,
        fmt="generalQ",
        vis=False,
        use_rep=True,
        use_best_pair=True,
        merge=True,
        cd_res=1.0,
        test_comm=False,
        comm_max=0.1,
        comm_min=1e-6,
        log_file=example_data_dir / "clustering" / f"{method}_hvg_align.log",
    )
```

## Prepare *compModels* Inputs

```python
model_comp_dir = example_data_dir / "comp_models" / "hvg_hc"
model_comp_output_dir = example_data_dir / "comp_models" / "hvg_hc_output"

models = ["seurat.louvain", "seurat.leiden", "scanpy.louvain", "scanpy.leiden"]
model_dirs = [example_data_dir / "aligned" / "hvg_hc" / m for m in methods]

qfilelists, qnamelists, mode_stats_files = aoc.prepare_compmodels(
    models=models,
    model_dirs=model_dirs,
    comp_dir=model_comp_dir,
    suffixes=["rep"] * len(models),
)
```

Optionally, add the Scanpy tutorial cell types as a reference model (used in the PBMC3k example).

```python
ref_qfilelist = model_comp_dir / "ref.qfilelist"
ref_qnamelist = model_comp_dir / "ref.qnamelist"
ref_qfilelist.write_text(str(example_data_dir / "reference" / "scanpy_tutorial.Q") + "\n")
ref_qnamelist.write_text("scanpy.tutorial\n")

models.insert(0, "scanpy.tutorial")
qfilelists.insert(0, str(ref_qfilelist))
qnamelists.insert(0, str(ref_qnamelist))
mode_stats_files.insert(0, example_data_dir / "reference" / "mode_stats.txt")
```

## Run *compModels*

```python
aoc.run_comp_models(
    models=models,
    comp_dir=model_comp_dir,
    output_dir=model_comp_output_dir,
    vis=False,
    qfilelists=qfilelists,
    qnamelists=qnamelists,
    mode_stats_files=mode_stats_files,
)
```
