#!/usr/bin/env bash
# Compare aligned Clumppling runs from different models with compModels.
#
# Usage:
#   bash run_compmodels.sh COMP_DIR OUTPUT_DIR NAME=ALIGNED_DIR [NAME=ALIGNED_DIR ...]
#
# Arguments:
#   COMP_DIR    Directory for the generated input lists (*.qfilelist, *.qnamelist,
#               *_mode_stats.txt).
#   OUTPUT_DIR  compModels output directory.
#   NAME=ALIGNED_DIR
#               A model name and its run_clumppling.sh output directory, for
#               example scanpy.leiden=aligned/scanpy_leiden
#
# Environment variables (optional, set all three to add a reference clustering):
#   REF_NAME        Name of the reference, for example scanpy.tutorial
#   REF_Q           Path to the reference .Q file.
#   REF_MODE_STATS  Path to the reference mode_stats.txt.
#
# The generated lists in COMP_DIR contain the paths to your files, so check them
# before adding them to version control.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash $0 COMP_DIR OUTPUT_DIR NAME=ALIGNED_DIR [NAME=ALIGNED_DIR ...]" >&2
    exit 1
fi

python - "${REF_NAME:-}" "${REF_Q:-}" "${REF_MODE_STATS:-}" "$@" <<'PY'
import sys
from pathlib import Path

import ace_of_clust as aoc

ref_name, ref_q, ref_mode_stats, comp_dir, output_dir, *model_specs = sys.argv[1:]
comp_dir = Path(comp_dir)
comp_dir.mkdir(parents=True, exist_ok=True)

models, model_dirs = [], []
for spec in model_specs:
    name, sep, model_dir = spec.partition("=")
    if not sep:
        sys.exit(f"Expected NAME=ALIGNED_DIR, got: {spec}")
    models.append(name)
    model_dirs.append(Path(model_dir).resolve())

# 1) Prepare qfilelist / qnamelist / mode_stats files
qfilelists, qnamelists, mode_stats_files = aoc.prepare_compmodels(
    models=models,
    model_dirs=model_dirs,
    comp_dir=comp_dir,
    suffixes=["rep"] * len(models),
)

# 2) Optionally add the reference clustering
ref_values = [ref_name, ref_q, ref_mode_stats]
if any(ref_values) and not all(ref_values):
    sys.exit("Set all of REF_NAME, REF_Q and REF_MODE_STATS, or none of them.")
if ref_name:
    ref_qfilelist = comp_dir / "ref.qfilelist"
    ref_qnamelist = comp_dir / "ref.qnamelist"
    ref_qfilelist.write_text(str(Path(ref_q).resolve()) + "\n")
    ref_qnamelist.write_text(ref_name + "\n")
    qfilelists.insert(0, str(ref_qfilelist))
    qnamelists.insert(0, str(ref_qnamelist))
    mode_stats_files.insert(0, Path(ref_mode_stats))
    models.insert(0, ref_name)

# 3) Run compModels
aoc.run_comp_models(
    models=models,
    comp_dir=comp_dir,
    output_dir=Path(output_dir),
    vis=False,
    bg_colors=None,
    include_sim_in_label=True,
    ind_labels="",
    qfilelists=qfilelists,
    qnamelists=qnamelists,
    mode_stats_files=mode_stats_files,
)
PY

echo "Done. Results in: ${2}"
