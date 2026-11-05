# Example script to run clumppling on PBMC3k scRNA-seq clustering results

from pathlib import Path
import ace_of_clust as aoc

base_dir =  Path("..").resolve()
example_data_dir = base_dir / "examples" / "data" / "pbmc3k"


############## PBMC3k HC HVG alignment #######################

# --- paths & project settings -------------------
for method_lb in ["seurat_louvain", "seurat_leiden", "scanpy_louvain", "scanpy_leiden"]:

    log_file = example_data_dir / "clustering" / f"{method_lb}_hvg_align.log"
    cls_dir = example_data_dir / "clustering" / "hvg_hc" / method_lb
    align_dir = example_data_dir / "aligned" / "hvg_hc" / method_lb

    # --- call clumppling via the wrapper ----------------------------------------
    aoc.run_clumppling(
        input_dir=cls_dir,
        output_dir=align_dir,
        fmt="generalQ",
        vis=False,
        use_rep=True,
        use_best_pair=True,
        merge=True,
        cd_res=1.0,
        test_comm=False,
        comm_max=0.1,
        comm_min=1e-6,
        setup_logging=True,
        log_file=log_file,
    )
