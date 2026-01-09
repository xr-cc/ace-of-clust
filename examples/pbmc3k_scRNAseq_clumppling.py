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


# # ############## PBMC3k MMC full alignment #######################
# # Note: This example may generate many row sum not equal to 1 errors/warnings. Please ignore them for now.
# # --- paths & project settings -------------------
# log_file = example_data_dir / "clustering" / "full_align.log"
# cls_dir = example_data_dir / "clustering" / "full_mmc"
# align_dir = example_data_dir / "aligned" / "full_mmc"

# # --- call clumppling via the wrapper ----------------------------------------
# aoc.run_clumppling_via_main(
#     input_dir=cls_dir,
#     output_dir=align_dir,
#     fmt="generalQ",                    # -f generalQ
#     vis=False,                         # -v F
#     use_rep=True,                      # --use_rep T
#     use_best_pair=True,                # --use_best_pair T
#     merge=True,                        # --merge T
#     cd_res=1.0,                        # --cd_res 1.0
#     test_comm=True,                    # --test_comm T
#     comm_max=0.1,
#     comm_min=1e-4,
#     setup_logging=True,
#     log_file=log_file,
# )

