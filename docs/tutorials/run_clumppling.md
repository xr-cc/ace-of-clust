# Clumppling tutorial (example workflow)

This tutorial walks through a minimal, end-to-end run of **Clumppling** on an example dataset generated using [PBMC 3k](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0) data.

---

## 0) Setup

```python
from pathlib import Path
import ace_of_clust as aoc
```

---

## 1) Set example data paths

Set path to the example data (using `seurat_leiden` as an example).

```python
base_dir =  Path("..").resolve()
example_data_dir = base_dir / "examples" / "data" / "pbmc3k"
method_lb = "seurat_leiden"
cls_dir = example_data_dir / "clustering" / "hvg_hc" / method_lb
```

---

## 2) Run *Clumppling* alignment (main)

```python
log_file = example_data_dir / "clustering" / f"{method_lb}_hvg_align.log"
align_dir = example_data_dir / "aligned" / "hvg_hc" / method_lb
aoc.run_clumppling_via_main(
        input_dir=cls_dir,
        output_dir=align_dir,
        fmt="generalQ",                    # -f generalQ
        vis=False,                         # -v F
        use_rep=True,                      # --use_rep T
        use_best_pair=True,                # --use_best_pair T
        merge=True,                        # --merge T
        cd_res=1.0,                        # --cd_res 1.0
        test_comm=False,                   # --test_comm T
        comm_max=0.1,
        comm_min=1e-6,
        setup_logging=True,
        log_file=log_file,
    )
```

---

## 3) Prepare input for *clumppling.compModels* 

Move *Clumppling* output files from aligned clustering results of multiple models.

```python
model_comp_dir = example_data_dir / "comp_models" / f"hvg_hc"
model_comp_output_dir = example_data_dir / "comp_models" / f"hvg_hc_output"
models = [ 
    "seurat.louvain", "seurat.leiden", "scanpy.louvain", "scanpy.leiden",
]
suffixes = ["rep"] * len(models)

model_dirs = [
    example_data_dir / "aligned" / f"hvg_hc" / "seurat_louvain",
    example_data_dir / "aligned" / f"hvg_hc" / "seurat_leiden",
    example_data_dir / "aligned" / f"hvg_hc" / "scanpy_louvain",
    example_data_dir / "aligned" / f"hvg_hc" / "scanpy_leiden",
]

qfilelists, qnamelists, mode_stats_files = aoc.prepare_comp_models_inputs(
    models=models,
    model_dirs=model_dirs,
    comp_dir=model_comp_dir,
    suffixes=suffixes,
)
```

---

## 4) Run *Clumppling.compModels*
```python
aoc.run_comp_models(
    models=models,
    comp_dir=model_comp_dir,
    output_dir=model_comp_output_dir,
    vis=False,
    bg_colors=None,   
    include_sim_in_label=True,
    ind_labels="",    
    qfilelists=qfilelists,
    qnamelists=qnamelists,
    mode_stats_files=mode_stats_files,
)
```