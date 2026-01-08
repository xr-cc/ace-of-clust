"""
wrappers.py

Python wrapper(s) around the clumppling CLI.

Usage
-----
run_clumppling_via_main(
    input_dir="/path/to/input",
    output_dir="/path/to/output",
    fmt="generalQ",
    ...
)

prepare_comp_models_inputs(
    models=["model1", "model2"],
    model_dirs=["/path/to/model1_dir", "/path/to/model2_dir"],
    comp_dir="/path/to/comp_dir",
)

run_comp_models(
    models=["model1", "model2"],
    comp_dir="/path/to/comp_dir",
    output_dir="/path/to/comp_models_output",
    ...
)

"""

from __future__ import annotations

import os
from argparse import Namespace
import argparse
import re
import shutil
from pathlib import Path
from typing import Union, Optional, Sequence, Tuple, List, Dict

from clumppling.__main__ import main as clumppling_main
from clumppling.compModels.__main__ import main as compmodels_main
from clumppling.log_config import setup_logger
from clumppling.utils import disp_params

PathLike = Union[str, os.PathLike]
_KM_RE = re.compile(r"K(\d+)(?:M(\d+))?", re.IGNORECASE)


def _build_clumppling_args(
    input_dir: PathLike,
    output_dir: PathLike,
    fmt: str = "generalQ",
    *,
    vis: bool = True,
    custom_cmap: str = "",
    plot_type: str = "graph",
    include_cost: bool = True,
    include_label: bool = True,
    alt_color: bool = True,
    ind_labels: str = "",
    ordered_uniq_labels: str = "",
    regroup_ind: bool = True,
    reorder_within_group: bool = True,
    reorder_by_max_k: bool = True,
    order_cls_by_label: bool = True,
    plot_unaligned: bool = False,
    fig_format: str = "tiff",
    extension: str = "",
    skip_rows: int = 0,
    remove_missing: bool = True,
    cd_method: str = "louvain",
    cd_res: float = 1.0,
    test_comm: bool = True,
    comm_min: float = 1e-6,
    comm_max: float = 1e-2,
    merge: bool = True,
    use_rep: bool = True,
    use_best_pair: bool = True,
) -> Namespace:
    """
    Internal helper: construct an argparse.Namespace with the same fields
    that clumppling.parse_args() would provide from the CLI.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # clumppling expects strings for paths and the same attribute names as
    # defined in parse_args(): input, output, format, vis, custom_cmap, ...
    args = Namespace(
        input=str(input_dir),
        output=str(output_dir),
        format=fmt,
        vis=vis,
        custom_cmap=custom_cmap,
        plot_type=plot_type,
        include_cost=include_cost,
        include_label=include_label,
        alt_color=alt_color,
        ind_labels=ind_labels,
        ordered_uniq_labels=ordered_uniq_labels,
        regroup_ind=regroup_ind,
        reorder_within_group=reorder_within_group,
        reorder_by_max_k=reorder_by_max_k,
        order_cls_by_label=order_cls_by_label,
        plot_unaligned=plot_unaligned,
        fig_format=fig_format,
        extension=extension,
        skip_rows=skip_rows,
        remove_missing=remove_missing,
        cd_method=cd_method,
        cd_res=cd_res,
        test_comm=test_comm,
        comm_min=comm_min,
        comm_max=comm_max,
        merge=merge,
        use_rep=use_rep,
        use_best_pair=use_best_pair,
    )
    return args


def run_clumppling_via_main(
    input_dir: PathLike,
    output_dir: PathLike,
    fmt: str = "generalQ",
    *,
    vis: bool = True,
    custom_cmap: str = "",
    plot_type: str = "graph",
    include_cost: bool = True,
    include_label: bool = True,
    alt_color: bool = True,
    ind_labels: str = "",
    ordered_uniq_labels: str = "",
    regroup_ind: bool = True,
    reorder_within_group: bool = True,
    reorder_by_max_k: bool = True,
    order_cls_by_label: bool = True,
    plot_unaligned: bool = False,
    fig_format: str = "tiff",
    extension: str = "",
    skip_rows: int = 0,
    remove_missing: bool = True,
    cd_method: str = "louvain",
    cd_res: float = 1.0,
    test_comm: bool = True,
    comm_min: float = 1e-6,
    comm_max: float = 1e-2,
    merge: bool = True,
    use_rep: bool = True,
    use_best_pair: bool = True,
    setup_logging: bool = True,
    log_file: Optional[PathLike] = None,
) -> Namespace:
    """
    Programmatic wrapper around `clumppling.main(args)`.

    Parameters
    ----------
    input_dir, output_dir:
        Directories for clumppling input and output. These correspond to
        `-i/--input` and `-o/--output` in the CLI.
    fmt:
        Input format: one of {"generalQ", "admixture", "structure", "fastStructure"}.
        This corresponds to `-f/--format`.
    vis, custom_cmap, plot_type, include_cost, include_label, alt_color, ...
        All other options map directly to the CLI arguments with the same name.
        See clumppling's `parse_args()` help text for semantics.

    setup_logging:
        If True (default), configure the clumppling logger the same way the
        CLI does, writing to `<output_dir>/clumppling.log` unless `log_file`
        is provided.
    log_file:
        Optional explicit path for the log file.

    Returns
    -------
    args : argparse.Namespace
        The Namespace object passed into `clumppling.main(args)`.
        This can be useful for debugging/logging.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = _build_clumppling_args(
        input_dir=input_dir,
        output_dir=output_dir,
        fmt=fmt,
        vis=vis,
        custom_cmap=custom_cmap,
        plot_type=plot_type,
        include_cost=include_cost,
        include_label=include_label,
        alt_color=alt_color,
        ind_labels=ind_labels,
        ordered_uniq_labels=ordered_uniq_labels,
        regroup_ind=regroup_ind,
        reorder_within_group=reorder_within_group,
        reorder_by_max_k=reorder_by_max_k,
        order_cls_by_label=order_cls_by_label,
        plot_unaligned=plot_unaligned,
        fig_format=fig_format,
        extension=extension,
        skip_rows=skip_rows,
        remove_missing=remove_missing,
        cd_method=cd_method,
        cd_res=cd_res,
        test_comm=test_comm,
        comm_min=comm_min,
        comm_max=comm_max,
        merge=merge,
        use_rep=use_rep,
        use_best_pair=use_best_pair,
    )

    # Set up logging & echo parameters
    if setup_logging:
        if log_file is None:
            log_file = output_dir / "clumppling.log"
        else:
            log_file = Path(log_file)
        setup_logger(str(log_file))

    disp_params(args, title="CLUMPPLING")

    # Call the actual clumppling main
    clumppling_main(args)

    return args


def prepare_comp_models_inputs(
    models: Sequence[str],
    model_dirs: Sequence[str | Path],
    comp_dir: str | Path,
    *,
    suffixes: Sequence[str] | None = None,
    modes_aligned_subdir: str = "modes_aligned",
    mode_stats_relpath: str = "modes/mode_stats.txt",
    exist_ok: bool = True,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Prepare input files for clumppling.compModels from multiple aligned
    clumppling runs.
    """
    models = list(models)
    model_dirs = [Path(d) for d in model_dirs]
    comp_dir = Path(comp_dir)
    comp_dir.mkdir(parents=True, exist_ok=exist_ok)

    if suffixes is None:
        suffixes = ["rep"] * len(models)
    else:
        suffixes = list(suffixes)

    if not (len(models) == len(model_dirs) == len(suffixes)):
        raise ValueError(
            "models, model_dirs, and suffixes must all have the same length "
            f"(got {len(models)}, {len(model_dirs)}, {len(suffixes)})."
        )

    qfilelists: List[Path] = []
    qnamelists: List[Path] = []
    mode_stats_files: List[Path] = []

    for model, model_dir, suffix in zip(models, model_dirs, suffixes):
        q_dir = model_dir / modes_aligned_subdir
        pattern = f"*_{suffix}.Q"

        qfiles = list(q_dir.glob(pattern))

        if not qfiles:
            raise FileNotFoundError(
                f"No Q files matching '{pattern}' found in {q_dir} for model {model}"
            )

        # --- numeric sort by K then M -----------------------------------
        def _sort_key(p: Path):
            name = p.stem  # e.g. "K18M1_rep" or "scanpy.leiden_K18M1_rep"
            if suffix:
                token = f"_{suffix}"
                if name.endswith(token):
                    name = name[: -len(token)]  # remove trailing _rep

            m = _KM_RE.search(name)
            if m:
                k = int(m.group(1))
                mnum = int(m.group(2) or 0)
                return (0, k, mnum, name)  # K asc, then M asc
            return (1, name)  # fallback lexicographic

        qfiles = sorted(qfiles, key=_sort_key)

        # qfilelist: full paths to Q files
        qfilelist_path = comp_dir / f"{model}.qfilelist"
        with qfilelist_path.open("w") as f:
            for q in qfiles:
                f.write(str(q) + "\n")

        # qnamelist: basename without "_{suffix}.Q"
        qnamelist_path = comp_dir / f"{model}.qnamelist"
        with qnamelist_path.open("w") as f:
            for q in qfiles:
                stem = q.stem  # e.g. "K18M1_rep"
                name = stem
                if suffix:
                    token = f"_{suffix}"
                    if name.endswith(token):
                        name = name[: -len(token)]
                f.write(name + "\n")

        # copy mode_stats.txt → {model}_mode_stats.txt
        src_mode_stats = model_dir / mode_stats_relpath
        if not src_mode_stats.is_file():
            raise FileNotFoundError(
                f"Mode stats file not found for model {model}: {src_mode_stats}"
            )
        dst_mode_stats = comp_dir / f"{model}_mode_stats.txt"
        shutil.copy2(src_mode_stats, dst_mode_stats)

        qfilelists.append(qfilelist_path)
        qnamelists.append(qnamelist_path)
        mode_stats_files.append(dst_mode_stats)

    return qfilelists, qnamelists, mode_stats_files



def run_comp_models(
    models: Sequence[str],
    comp_dir: str | Path,
    output_dir: str | Path,
    *,
    fig_format: str = "png",
    vis: bool = True,
    custom_cmap: str = "",
    bg_colors: Sequence[str] | None = None,
    include_sim_in_label: bool = True,
    ind_labels: str = "",
    qfilelists: Sequence[str | Path] | None = None,
    qnamelists: Sequence[str | Path] | None = None,
    mode_stats_files: Sequence[str | Path] | None = None,
    setup_logging: bool = True,
    log_file: Optional[PathLike] = None,
) -> None:
    """
    Programmatic wrapper around `clumppling.compModels.main(args)`.

    Parameters
    ----------
    models : sequence of str
        Model names passed to --models.
    comp_dir : path-like
        Directory where the *.qfilelist, *.qnamelist, *_mode_stats.txt live.
        If qfilelists / qnamelists / mode_stats_files are not provided, they
        are inferred as:
            comp_dir/{model}.qfilelist
            comp_dir/{model}.qnamelist
            comp_dir/{model}_mode_stats.txt
    output_dir : path-like
        Output directory passed to --output.
    fig_format : str, default "png"
        Figure format for output files.
    vis : bool
        Passed to --vis.
    custom_cmap : str
        Passed to --custom_cmap (path to color file or "").
    bg_colors : sequence of str or None
        Passed to --bg_colors (list of colors) if given.
    include_sim_in_label : bool
        Passed to --include_sim_in_label.
    ind_labels : str
        Passed to --ind_labels (path to labels file or "").
    """
    comp_dir = Path(comp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(models)

    if qfilelists is None:
        qfilelists = [comp_dir / f"{m}.qfilelist" for m in models]
    else:
        qfilelists = [Path(p) for p in qfilelists]

    if qnamelists is None:
        qnamelists = [comp_dir / f"{m}.qnamelist" for m in models]
    else:
        qnamelists = [Path(p) for p in qnamelists]

    if mode_stats_files is None:
        mode_stats_files = [comp_dir / f"{m}_mode_stats.txt" for m in models]
    else:
        mode_stats_files = [Path(p) for p in mode_stats_files]

    if not (
        len(models)
        == len(qfilelists)
        == len(qnamelists)
        == len(mode_stats_files)
    ):
        raise ValueError(
            "models, qfilelists, qnamelists, and mode_stats_files "
            "must all have the same length."
        )

    # Build a Namespace that matches parse_args() for compModels
    args = argparse.Namespace(
        models=list(models),
        qfilelists=[str(p) for p in qfilelists],
        qnamelists=[str(p) for p in qnamelists],
        mode_stats_files=[str(p) for p in mode_stats_files],
        ind_labels=str(ind_labels) if ind_labels is not None else "",
        output=str(output_dir),
        vis=vis,
        custom_cmap=str(custom_cmap) if custom_cmap is not None else "",
        bg_colors=list(bg_colors) if bg_colors is not None else None,
        include_sim_in_label=include_sim_in_label,
        fig_format=fig_format,
    )

    # Set up logging & echo parameters
    if setup_logging:
        if log_file is None:
            log_file = output_dir / "clumppling.compModels.log"
        else:
            log_file = Path(log_file)
        # clumppling's setup_logger expects a string path
        setup_logger(str(log_file))

    # Call your patched compModels.main(args)
    compmodels_main(args)

