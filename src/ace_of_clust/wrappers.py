"""Wrappers for Clumppling and compModels."""

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
    """Build the argparse.Namespace that Clumppling's command line would produce."""
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
        random_seed=None,
    )
    return args


def run_clumppling(
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
    Run Clumppling on the clustering runs in one directory.

    Arguments other than those below are Clumppling options of the same name; see `clumppling --help`.

    Parameters
    ----------
    input_dir
        Directory with the clustering files (-i).
    output_dir
        Output directory (-o).
    fmt
        Input format (-f): "generalQ", "admixture", "structure" or "fastStructure".
    setup_logging
        Write a log, as the command line does.
    log_file
        Log file; defaults to output_dir/clumppling.log.
    vis, custom_cmap, plot_type, include_cost, include_label, alt_color, ind_labels, ordered_uniq_labels, regroup_ind, reorder_within_group, reorder_by_max_k, order_cls_by_label, plot_unaligned, fig_format, extension, skip_rows, remove_missing, cd_method, cd_res, test_comm, comm_min, comm_max, merge, use_rep, use_best_pair
        Clumppling options.

    Returns
    -------
    argparse.Namespace
        Arguments passed to Clumppling.
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

    clumppling_main(args)

    return args


def prepare_compmodels(
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
    Write the compModels input files for several aligned models.

    Parameters
    ----------
    models
        Model names, used to name the output files.
    model_dirs
        Clumppling output directory of each model.
    comp_dir
        Directory for the output files.
    suffixes
        Suffix of the aligned Q files per model, "rep" or "avg"; None uses "rep" for all.
    modes_aligned_subdir
        Subdirectory of each model directory with the aligned Q files.
    mode_stats_relpath
        Path of the mode statistics file in each model directory.
    exist_ok
        Allow comp_dir to exist already.

    Returns
    -------
    qfilelists
        One .qfilelist file per model.
    qnamelists
        One .qnamelist file per model.
    mode_stats_files
        One copied mode_stats file per model.

    Raises
    ------
    ValueError
        If models, model_dirs and suffixes differ in length.
    FileNotFoundError
        If Q files or mode_stats files are missing.
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
    Run compModels to align and compare several models.

    Parameters
    ----------
    models
        Model names (--models).
    comp_dir
        Directory with the input files from `prepare_compmodels`.
    output_dir
        Output directory (--output).
    fig_format
        Figure format.
    vis
        Write figures.
    custom_cmap
        Color file, or "" for the default colors.
    bg_colors
        Background colors, or None.
    include_sim_in_label
        Show similarity in figure labels.
    ind_labels
        File of cell labels, or "".
    qfilelists, qnamelists, mode_stats_files
        Input files per model; None uses comp_dir/{model}.qfilelist, comp_dir/{model}.qnamelist and comp_dir/{model}_mode_stats.txt.
    setup_logging
        Write a log, as the command line does.
    log_file
        Log file; defaults to output_dir/clumppling.compModels.log.
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

    compmodels_main(args)

__all__ = [
    "run_clumppling",
    "prepare_compmodels",
    "run_comp_models",
]