from pathlib import Path
import ace_of_clust as aoc

base_dir =  Path("..").resolve()
example_data_dir = base_dir / "examples" / "data" / "pbmc3k"

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

# 1) Prepare qfilelist / qnamelist / mode_stats files
qfilelists, qnamelists, mode_stats_files = aoc.prepare_compmodels(
    models=models,
    model_dirs=model_dirs,
    comp_dir=model_comp_dir,
    suffixes=suffixes,
)

# 2) Add the reference clustering results "scanpy.tutorial" qfilelist, qnamelist, mode_stats file
gt_qfile = example_data_dir / "dummy" / "scanpy_tutorial.Q"
gt_qfile_file = example_data_dir / "comp_models" / f"hvg_hc" / "ref.qfilelist"
# write this to a .qfilelist file
with open(gt_qfile_file, "w") as f:
    f.write(str(gt_qfile) + "\n")
gt_qname_file = example_data_dir / "comp_models" / f"hvg_hc" / "ref.qnamelist"
# write this to a .qnamelist file
with open(gt_qname_file, "w") as f:
    f.write("scanpy.tutorial\n")
qfilelists.insert(0, str(gt_qfile_file))
qnamelists.insert(0, str(gt_qname_file))
# add "scanpy.tutorial" mode_stats file
gt_mode_stats_file = example_data_dir / "dummy" / "mode_stats.txt"
mode_stats_files.insert(0, gt_mode_stats_file)
# add "scanpy.tutorial" to models list
models.insert(0, "scanpy.tutorial")

# 3) Run compModels
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