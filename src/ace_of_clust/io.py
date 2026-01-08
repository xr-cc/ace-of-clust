"""
io.py

Functions for loading clumppling's main and compModel outputs, as well as other related data.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Optional, Literal, Set
from collections import defaultdict
import numpy as np
import pandas as pd
import gzip
import bisect

from clumppling.utils import str_to_pattern  # inverse of pattern_to_str

PathLike = Union[str, os.PathLike]
GeneT = Tuple[int, int, str] # Gene tuple: (start, end, gene_name)

@dataclass
class ClumpplingResults:
    """
    Container for all core clumppling outputs needed for analysis/plots.
    """

    # paths
    align_dir: Path
    suffix: str

    # metadata
    mode_alignment: pd.DataFrame
    mode_stats: pd.DataFrame

    # mode-level info
    modes: List[str]                         # flat list of mode names (e.g. ["K17M1", "K17M2", ...])
    mode_K: Dict[str, int]                  # {mode_name -> K for that mode}
    K_range: List[int]                      # sorted unique K values
    K_max: int                              # max K across all modes
    mode_names_list: List[List[str]]        # grouped mode names by K (same structure as notebook)

    # aligned Q matrices
    Q_by_mode: Dict[str, np.ndarray]        # {mode_name -> aligned membership matrix}

    # alignment info
    alignment_acrossK: Dict[str, Sequence[int]]  # {"A-B" -> mapping from B->A (original indices)}
    cost_acrossK: Dict[str, float]
    all_modes_alignment: Dict[str, Sequence[int]]  # {mode_name -> reordering (aligned columns)}

    # layout helpers for plotting
    mode_coord_dict: Dict[str, Tuple[int, int]]         # {mode_name -> (row_idx, col_idx)} grid by K
    mode_sep_coord_dict: Dict[Tuple[str, int], Tuple[int, int]]  # {(mode_name, cls_idx) -> (row_idx, col_idx)}

    # optional: input meta and P matrices for feature-level analysis
    input_meta: pd.DataFrame | None = None
    Q_unaligned_by_mode: Dict[str, np.ndarray] | None = None
    P_unaligned_by_mode: Dict[str, np.ndarray] | None = None
    P_aligned_by_mode: Dict[str, np.ndarray] | None = None

    def reorder_inds(self, reorder_idx: Sequence[int]) -> ClumpplingResults:
        """
        Return a new CompModelsResults with all Q matrices reordered
        according to reorder_idx.
        """
        new_Q_by_mode = {
            mode: Q[reorder_idx, :]
            for mode, Q in self.Q_by_mode.items()
        }
        new_Q_unaligned_by_mode = None
        if self.Q_unaligned_by_mode is not None:
            new_Q_unaligned_by_mode = {
                mode: Q[reorder_idx, :]
                for mode, Q in self.Q_unaligned_by_mode.items()
            }

        return ClumpplingResults(
            align_dir=self.align_dir,
            suffix=self.suffix,
            mode_alignment=self.mode_alignment,
            mode_stats=self.mode_stats,
            modes=self.modes,
            mode_K=self.mode_K,
            K_range=self.K_range,
            K_max=self.K_max,
            mode_names_list=self.mode_names_list,
            Q_by_mode=new_Q_by_mode,
            alignment_acrossK=self.alignment_acrossK,
            cost_acrossK=self.cost_acrossK,
            all_modes_alignment=self.all_modes_alignment,
            mode_coord_dict=self.mode_coord_dict,
            mode_sep_coord_dict=self.mode_sep_coord_dict,
            input_meta=self.input_meta,
            Q_unaligned_by_mode=new_Q_unaligned_by_mode,
            P_unaligned_by_mode=self.P_unaligned_by_mode,
            P_aligned_by_mode=self.P_aligned_by_mode,
        )


@dataclass
class CompModelsResults:
    """
    Container for clumppling.compModels outputs and associated metadata.

    Attributes
    ----------
    res_dir : Path
        Directory containing compModels outputs
        (e.g. output/comp_models/pbmc10k-tutorial_hc_output).
    input_dir : Optional[Path]
        Directory containing per-model input stats for compModels
        (e.g. output/comp_models/pbmc10k-tutorial_hc).
    models : List[str]
        Model names (e.g. "rna.seurat.louvain", "rna.seurat.leiden", ...).
    modes_by_model : Dict[str, List[str]]
        For each model, a list of *short* mode names with the model prefix
        stripped, e.g. {"rna.seurat.louvain": ["K21M1", "K21M2", ...]}.
    full_mode_names_by_model : Dict[str, List[str]]
        For each model, the full mode names as used in filenames,
        e.g. {"rna.seurat.louvain": ["rna.seurat.louvain_K21M1", ...]}.
    full_mode_names : List[str]
        Flat list of all full mode names across all models.
    Q_by_mode : Dict[str, np.ndarray]
        Mapping full mode name -> aligned membership matrix Q loaded from
        res_dir / "aligned" / f"{mode}.Q".
    all_modes_alignment : Dict[str, Sequence[int]]
        Mapping full mode name -> global alignment pattern, parsed from
        res_dir / "aligned" / "all_modes_alignment.txt" if present.
        (Keys are full mode names; values are index patterns.)
    alignment_across_all : Optional[Dict[str, Sequence[int]]]
        Reserved for cross-mode alignment patterns (currently left as None
        unless you later decide to parse additional files).
    cost_across_all : Optional[Dict[str, float]]
        Reserved for cross-mode alignment costs (currently None).
    mode_stats_by_model : Dict[str, pd.DataFrame]
        For each model, its original mode_stats DataFrame loaded from
        input_dir / f"{model}_mode_stats.txt", if available.
    K_max : int
    K_max_by_model: Dict[str, int]
    """

    res_dir: Path
    input_dir: Optional[Path]

    models: List[str]
    modes_by_model: Dict[str, List[str]]
    full_mode_names_by_model: Dict[str, List[str]]
    full_mode_names: List[str]

    Q_by_mode: Dict[str, np.ndarray]

    all_modes_alignment: Dict[str, Sequence[int]]
    alignment_across_all: Optional[Dict[str, Sequence[int]]]
    cost_across_all: Optional[Dict[str, float]]

    mode_stats_by_model: Dict[str, pd.DataFrame]

    K_max: int
    K_max_by_model: Dict[str, int]


    def get_Q(self, full_mode_name: str) -> np.ndarray:
        """Return the aligned Q matrix for a full mode name."""
        return self.Q_by_mode[full_mode_name]

    def get_Q_for(self, model: str, mode_short: str) -> np.ndarray:
        """
        Return the aligned Q matrix for a (model, short_mode_name) pair,
        where short_mode_name is e.g. 'K21M1'.
        """
        full_name = f"{model}_{mode_short}"
        return self.Q_by_mode[full_name]
    
    def reorder_inds(self, reorder_idx: Sequence[int]) -> CompModelsResults:
        """
        Return a new CompModelsResults with all Q matrices reordered
        according to reorder_idx.
        """
        new_Q_by_mode = {
            mode: Q[reorder_idx, :]
            for mode, Q in self.Q_by_mode.items()
        }

        return CompModelsResults(
            res_dir=self.res_dir,
            input_dir=self.input_dir,
            models=self.models,
            modes_by_model=self.modes_by_model,
            full_mode_names_by_model=self.full_mode_names_by_model,
            full_mode_names=self.full_mode_names,
            Q_by_mode=new_Q_by_mode,
            all_modes_alignment=self.all_modes_alignment,
            alignment_across_all=self.alignment_across_all,
            cost_across_all=self.cost_across_all,
            mode_stats_by_model=self.mode_stats_by_model,
            K_max=self.K_max,
            K_max_by_model=self.K_max_by_model,
        )


def load_mode_alignment(align_dir: PathLike) -> pd.DataFrame:
    """
    Load the 'mode_alignment.txt' table produced by clumppling.

    Parameters
    ----------
    align_dir : path-like
        The main output directory passed as `-o/--output` to clumppling.

    Returns
    -------
    mode_alignment : pd.DataFrame
    """
    align_dir = Path(align_dir)
    fname = align_dir / "modes" / "mode_alignment.txt"
    return pd.read_csv(fname)


def load_mode_stats(align_dir: PathLike) -> pd.DataFrame:
    """
    Load the 'mode_stats.txt' table produced by clumppling.

    Parameters
    ----------
    align_dir : path-like
        The main output directory passed as `-o/--output` to clumppling.

    Returns
    -------
    mode_stats : pd.DataFrame
    """
    align_dir = Path(align_dir)
    fname = align_dir / "modes" / "mode_stats.txt"
    return pd.read_csv(fname)


def get_mode_names(mode_alignment: pd.DataFrame) -> List[str]:
    """
    Extract the list of unique mode names from mode_alignment.

    Parameters
    ----------
    mode_alignment : pd.DataFrame

    Returns
    -------
    modes : list of str
    """
    if "Mode" not in mode_alignment.columns:
        raise KeyError("mode_alignment must contain a 'Mode' column")

    if "Representative" in mode_alignment.columns:
        modes = (
            mode_alignment
            .drop_duplicates(["Mode", "Representative"])
            .set_index("Mode")
            .index
            .tolist()
        )
    else:
        modes = mode_alignment["Mode"].drop_duplicates().tolist()

    return modes


def load_input_meta(align_dir: PathLike) -> pd.DataFrame:
    """
    Load the 'input_meta.txt' table that links original Q/P files to modes.

    Parameters
    ----------
    align_dir : path-like
        The main clumppling output directory.

    Returns
    -------
    input_meta : pd.DataFrame
    """
    align_dir = Path(align_dir)
    fname = align_dir / "input" / "input_meta.txt"
    input_meta = pd.read_csv(fname, dtype=str, header=None)

    # extract file names from the directory paths and strip ".Q"
    for col in (0, 1):
        input_meta[col] = (
            input_meta[col]
            .str.strip()
            .str.split("/")
            .str[-1]
            .str.replace(".Q", "", regex=False)
        )

    input_meta.columns = ["Orig_File", "Mode_Name", "K"]
    return input_meta


def load_unaligned_for_modes(
    cls_dir: PathLike,
    align_dir: PathLike,
    *,
    mat_type: Literal["P", "Q"],
    modes: Sequence[str] | None = None,
    mode_stats: pd.DataFrame | None = None,
    input_meta: pd.DataFrame | None = None,
    delimiter: str = " ",
) -> Dict[str, np.ndarray]:
    """
    Load the unaligned P matrices for each mode.

    Parameters
    ----------
    cls_dir : path-like
        Directory that contains the original MMC / clustering outputs, i.e.
        where the '*.P' files live.
    align_dir : path-like
        Main clumppling output directory (used to load mode_stats and input_meta
        if they are not provided).
    mat_type : Literal["P", "Q"]
        Type of matrix to load. Currently "P" and "Q" are supported.
    modes : sequence of str, optional
        Mode names to load. If None, they are inferred from mode_alignment.
    mode_stats : pd.DataFrame, optional
        If already loaded, pass it to avoid re-reading.
    input_meta : pd.DataFrame, optional
        If already loaded, pass it to avoid re-reading.
    delimiter : str, default " "
        Delimiter for the P files.

    Returns
    -------
    mat_by_mode : dict
        {mode_name -> np.ndarray of shape (n_features, K_mode)}
    """
    cls_dir = Path(cls_dir)
    align_dir = Path(align_dir)

    if mode_stats is None:
        mode_stats = load_mode_stats(align_dir)
    if modes is None:
        mode_alignment = load_mode_alignment(align_dir)
        modes = get_mode_names(mode_alignment)
    if input_meta is None:
        input_meta = load_input_meta(align_dir)

    mat_by_mode: Dict[str, np.ndarray] = {}

    for mode_name in modes:
        # Representative run name for this mode
        repr_name = mode_stats.loc[mode_stats["Mode"] == mode_name, "Representative"].values[0]
        # Original Q/P file stem for that representative
        orig_file_name = input_meta.loc[input_meta["Mode_Name"] == repr_name, "Orig_File"].values[0]

        mat_path = cls_dir / f"{orig_file_name}.{mat_type}"
        if not mat_path.exists():
            raise FileNotFoundError(f"Unaligned {mat_type} file not found: {mat_path}")

        mat = np.loadtxt(mat_path, dtype=float, delimiter=delimiter)
        mat_by_mode[mode_name] = mat

    return mat_by_mode


def load_aligned_Qs(
    align_dir: PathLike,
    modes: Sequence[str],
    suffix: str = "rep",
    *,
    delimiter: str = " ",
) -> Dict[str, np.ndarray]:
    """
    Load aligned membership matrices for each mode from 'modes_aligned'.

    Files are expected to be named like:
        <mode_name>_<suffix>.Q
    e.g. "K17M1_rep.Q"

    Parameters
    ----------
    align_dir : path-like
        Main clumppling output directory (same as used in load_mode_alignment).
    modes : sequence of str
        Mode names to load, e.g. from get_mode_names(...).
    suffix : {"rep", "avg"}, default "rep"
        Suffix used by clumppling when writing aligned modes.
    delimiter : str, default " "
        Delimiter for the Q files.

    Returns
    -------
    Q_by_mode : dict
        {mode_name -> np.ndarray of shape (n_individuals, K_mode)}
    """
    align_dir = Path(align_dir)
    q_dir = align_dir / "modes_aligned"

    Q_by_mode: Dict[str, np.ndarray] = {}

    for mode_name in modes:
        fname = q_dir / f"{mode_name}_{suffix}.Q"
        if not fname.exists():
            raise FileNotFoundError(f"Aligned Q file not found: {fname}")
        Q = np.loadtxt(fname, dtype=float, delimiter=delimiter)
        Q_by_mode[mode_name] = Q

    return Q_by_mode


def load_alignment_across_k(
    align_file: str | PathLike,
) -> Tuple[Dict[str, Sequence[int]], Dict[str, float]]:
    """
    Load alignment_acrossK and cost_acrossK from the file written by
    clumppling.write_alignment_across_k.

    Parameters
    ----------
    align_file : str | PathLike
        Path to the alignment file.

    Returns
    -------
    alignment_acrossK : dict
        {pair_label -> alignment pattern}, where `alignment pattern` is
        a sequence of ints mapping clusters between two modes.
    cost_acrossK : dict
        {pair_label -> float cost}
    """

    alignment_acrossK: Dict[str, Sequence[int]] = {}
    cost_acrossK: Dict[str, float] = {}

    with align_file.open("r") as f:
        # skip header
        _ = f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Only split on the first two commas in case the alignment string itself contains commas
            pair_label, cost_str, alignment_str = line.split(",", 2)

            cost = float(cost_str)
            idxQ2P = str_to_pattern(alignment_str)  # inverse of pattern_to_str

            alignment_acrossK[pair_label] = idxQ2P
            cost_acrossK[pair_label] = cost

    return alignment_acrossK, cost_acrossK


def load_all_modes_alignment(
    align_dir: PathLike,
    suffix: str = "rep",
    *,
    filename: str | None = None,
) -> Dict[str, Sequence[int]]:
    """
    Load all_modes_alignment from the file written by clumppling.write_reordered_across_k.

    Parameters
    ----------
    align_dir : path-like
        Main clumppling output directory.
    suffix : {"rep", "avg"}, default "rep"
        Suffix used when writing the all-modes alignment file.
    filename : str, optional
        Override the default filename if needed. If None, uses
        f"all_modes_alignment_{suffix}.txt".

    Returns
    -------
    all_modes_alignment : dict
        {mode_label -> alignment_pattern}, where alignment_pattern is obtained
        by applying str_to_pattern to the stored pattern string.
    """
    align_dir = Path(align_dir)
    if filename is None:
        filename = f"all_modes_alignment_{suffix}.txt"

    fname = align_dir / "modes_aligned" / filename

    if not fname.exists():
        raise FileNotFoundError(f"Alignment file not found: {fname}")

    all_modes_alignment: Dict[str, Sequence[int]] = {}

    with fname.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # mode_label:pattern_string
            mode_label, pattern_str = line.split(":", 1)

            alignment_pattern = str_to_pattern(pattern_str)
            all_modes_alignment[mode_label] = alignment_pattern

    return all_modes_alignment


def make_mode_names_list(mode_names: Sequence[str]) -> List[List[str]]:
    """
    Group a flat list of mode names into a list-of-lists by K.

    Given a list like:
        ['K17M1', 'K17M2', 'K18M1', 'K19M1']
    return:
        [['K17M1', 'K17M2'], ['K18M1'], ['K19M1']]

    This mirrors the notebook helper.

    Parameters
    ----------
    mode_names : sequence of str

    Returns
    -------
    mode_names_list : list of list of str
        One sublist per K, ordered by increasing K.
    """
    groups: Dict[int, List[str]] = defaultdict(list)

    for name in mode_names:
        # Extract the K value between 'K' and 'M'
        # e.g. 'K17M2' -> 'K17' -> '17' -> 17
        k_str = name.split("M")[0][1:]
        k = int(k_str)
        groups[k].append(name)

    # Return groups ordered by increasing K
    return [groups[k] for k in sorted(groups.keys())]


def infer_K_range_from_modes(mode_names: Sequence[str]) -> List[int]:
    """
    Convenience helper to get sorted K values from mode names.

    Parameters
    ----------
    mode_names : sequence of str, e.g. ['K17M1', 'K17M2', 'K18M1']

    Returns
    -------
    K_range : list of int, e.g. [17, 18]
    """
    K_set = set()
    for name in mode_names:
        k_str = name.split("M")[0][1:]
        K_set.add(int(k_str))
    return sorted(K_set)


def load_clumppling_results(
    align_dir: PathLike,
    *,
    suffix: str = "rep",
    round_Q: bool = False,
    cls_dir: PathLike | None = None,
    load_unaligned: bool = False,
    load_P: bool = True,
    strict_P: bool = False,
) -> ClumpplingResults:
    """
    Load clumppling results from the specified directory.

    Parameters
    ----------
    align_dir : path-like
        The clumppling output directory used as `-o/--output`.
    suffix : {"rep", "avg"}, default "rep"
        Suffix used in the aligned Q filenames, e.g. "K17M1_rep.Q".
    round_Q : bool, default False
        If True, apply np.rint to each aligned Q matrix to get hard cluster
        memberships.
    cls_dir : path-like, optional
        Directory containing the original clustering outputs (*.P files).
        If provided and load_P is True, P matrices will be loaded.
    load_P : bool, default True
        Whether to attempt loading P matrices at all. Set to False if you
        know this run has no P files (e.g. hard clustering only).
    strict_P : bool, default False
        If True, missing P files raise FileNotFoundError.
        If False, missing P files will emit a warning and skip P loading.
    """
    align_dir = Path(align_dir)

    # --- load basic mode-level info ----------------------------------
    mode_alignment = load_mode_alignment(align_dir)
    mode_stats = load_mode_stats(align_dir)

    modes = get_mode_names(mode_alignment)  # e.g. ["K17M1", "K17M2", "K18M1", ...]

    Q_by_mode = load_aligned_Qs(align_dir, modes=modes, suffix=suffix)
    mode_K: Dict[str, int] = {m: Q.shape[1] for m, Q in Q_by_mode.items()}

    if round_Q:
        Q_by_mode = {m: np.rint(Q) for m, Q in Q_by_mode.items()}

    K_max = max(mode_K.values()) if mode_K else 0

    mode_names_list = make_mode_names_list(modes)
    K_range = infer_K_range_from_modes(modes)

    # --- load alignment information ----------------------------------
    align_file = align_dir / "alignment_acrossK" / f"alignment_acrossK_{suffix}.txt"
    alignment_acrossK, cost_acrossK = load_alignment_across_k(align_file=align_file)
    all_modes_alignment = load_all_modes_alignment(align_dir, suffix=suffix)

    # --- layout dicts for plotting -----------------------------------

    mode_coord_dict: Dict[str, Tuple[int, int]] = {}
    for i_row, K in enumerate(K_range):
        modes_at_K = mode_names_list[i_row]
        for i_col, mode_name in enumerate(modes_at_K):
            mode_coord_dict[mode_name] = (i_row, i_col)

    mode_sep_coord_dict: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for i_row_sep_k, mode_name in enumerate(modes):
        K = mode_K[mode_name]
        for k in range(K):
            mode_sep_coord_dict[(mode_name, k)] = (i_row_sep_k, k)

    # --- optional: load and align P matrices and unaligned Q matrices -------------------------
    input_meta = None
    Q_unaligned_by_mode = None
    P_unaligned_by_mode = None
    P_aligned_by_mode = None

    if cls_dir is not None and (load_P or load_unaligned):
        try:
            input_meta = load_input_meta(align_dir)
            if load_unaligned:
                Q_unaligned_by_mode = load_unaligned_for_modes(
                    cls_dir=cls_dir,
                    align_dir=align_dir,
                    mat_type="Q",
                    modes=modes,
                    mode_stats=mode_stats,
                    input_meta=input_meta,
                )
            if load_P:
                P_unaligned_by_mode = load_unaligned_for_modes(
                    cls_dir=cls_dir,
                    align_dir=align_dir,
                    mat_type="P",
                    modes=modes,
                    mode_stats=mode_stats,
                    input_meta=input_meta,
                )
                # align P matrices using the same column reordering as Q
                P_aligned_by_mode = {}
                for mode_name in modes:
                    P = P_unaligned_by_mode[mode_name]
                    alignment = all_modes_alignment[mode_name]
                    P_aligned_by_mode[mode_name] = P[:, alignment]

        except FileNotFoundError as e:
            if strict_P:
                # preserve old behaviour
                raise
            else:
                # soft-fail: warn and continue without P
                print(
                    f"[sc_clumppling] Warning: {e}. "
                    "No unaligned Q or P matrices will be attached to ClumpplingResults for this run."
                )
                input_meta = None
                Q_unaligned_by_mode = None
                P_unaligned_by_mode = None
                P_aligned_by_mode = None

    return ClumpplingResults(
        align_dir=align_dir,
        suffix=suffix,
        mode_alignment=mode_alignment,
        mode_stats=mode_stats,
        modes=modes,
        mode_K=mode_K,
        K_range=K_range,
        K_max=K_max,
        mode_names_list=mode_names_list,
        Q_by_mode=Q_by_mode,
        alignment_acrossK=alignment_acrossK,
        cost_acrossK=cost_acrossK,
        all_modes_alignment=all_modes_alignment,
        mode_coord_dict=mode_coord_dict,
        mode_sep_coord_dict=mode_sep_coord_dict,
        input_meta=input_meta,
        Q_unaligned_by_mode=Q_unaligned_by_mode,
        P_unaligned_by_mode=P_unaligned_by_mode,
        P_aligned_by_mode=P_aligned_by_mode,
    )


def _load_mode_lists_from_replicates(
    res_dir: Path,
) -> Tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    List[str],
]:
    """
    Discover models and per-model mode lists from *_replicates.txt files.

    Each such file is expected to live directly in `res_dir` and have a
    name like '{model}_replicates.txt', containing one full mode name
    per line, e.g.:

        rna.seurat.louvain_K21M1
        rna.seurat.louvain_K21M2
        ...

    Returns
    -------
    models : list[str]
        Model names derived from filenames (stem minus '_replicates').
    modes_by_model : dict[str, list[str]]
        For each model, list of *short* mode names (prefix stripped),
        e.g. {"rna.seurat.louvain": ["K21M1", "K21M2", ...]}.
    full_mode_names_by_model : dict[str, list[str]]
        For each model, the full mode names (as in files).
    full_mode_names : list[str]
        Flat list of all full mode names across all models.
    """
    replicates_files = sorted(res_dir.glob("*_replicates.txt"))
    if not replicates_files:
        raise FileNotFoundError(
            f"No *_replicates.txt files found in {res_dir}. "
            "Is this a compModels output directory?"
        )

    models: List[str] = []
    modes_by_model: Dict[str, List[str]] = {}
    full_mode_names_by_model: Dict[str, List[str]] = {}
    full_mode_names: List[str] = []

    for f in replicates_files:
        # e.g. "rna.seurat.louvain_replicates.txt" -> "rna.seurat.louvain"
        model = f.stem.replace("_replicates", "")
        models.append(model)

        raw = np.loadtxt(f, dtype=str)
        if raw.ndim == 0:
            full_names = [str(raw)]
        else:
            full_names = raw.tolist()

        full_mode_names_by_model[model] = full_names
        short_names = [name.replace(f"{model}_", "", 1) for name in full_names]
        modes_by_model[model] = short_names
        full_mode_names.extend(full_names)

    return models, modes_by_model, full_mode_names_by_model, full_mode_names


def load_compmodels_results(
    res_dir: str | Path,
    input_dir: str | Path | None = None,
) -> CompModelsResults:
    """
    Load outputs from `clumppling.compModels` into a CompModelsResults object.

    Parameters
    ----------
    res_dir : str or Path
        Directory containing compModels outputs, e.g.
        .../output/comp_models/pbmc10k-tutorial_hc_output
    input_dir : str or Path, optional
        Directory containing per-model input stats used for compModels,
        e.g. .../output/comp_models/pbmc10k-tutorial_hc.
        If None, mode_stats_by_model will be empty.

    Returns
    -------
    CompModelsResults
        Structured container for multi-model Q matrices, mode lists,
        global alignment patterns, and per-model mode_stats.
    """
    res_dir = Path(res_dir)
    if input_dir is not None:
        input_dir = Path(input_dir)

    # Discover models and modes from *_replicates.txt
    (
        models,
        modes_by_model,
        full_mode_names_by_model,
        full_mode_names,
    ) = _load_mode_lists_from_replicates(res_dir)

    # Load aligned Q matrices from res_dir / "aligned" / "{mode}.Q"
    aligned_dir = res_dir / "aligned"
    if not aligned_dir.is_dir():
        raise FileNotFoundError(f"Aligned directory not found: {aligned_dir}")

    Q_by_mode: Dict[str, np.ndarray] = {}
    for mode_name in full_mode_names:
        q_path = aligned_dir / f"{mode_name}.Q"
        if not q_path.exists():
            raise FileNotFoundError(
                f"Aligned Q file not found for mode '{mode_name}': {q_path}"
            )
        Q_by_mode[mode_name] = np.loadtxt(q_path)

    # Load all_modes_alignment (optional)
    all_modes_alignment_file = aligned_dir / "all_modes_alignment.txt"
    all_modes_alignment: Dict[str, Sequence[int]] = {}
    if all_modes_alignment_file.exists():
        with all_modes_alignment_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                mode_label, pat_str = line.split(":", 1)
                mode_label = mode_label.strip()
                pat_str = pat_str.strip()
                if not mode_label:
                    continue
                all_modes_alignment[mode_label] = str_to_pattern(pat_str)
    else:
        # If it's not there, just leave the dict empty.
        all_modes_alignment = {}

    # Load cross-mode alignment/cost slots.
    align_file = res_dir / "alignment_across_all.txt"
    alignment_across_all, cost_across_all = load_alignment_across_k(align_file=align_file)

    # Load per-model mode_stats from input_dir, if provided
    mode_stats_by_model: Dict[str, pd.DataFrame] = {}
    if input_dir is not None:
        for model_name in models:
            stats_file = input_dir / f"{model_name}_mode_stats.txt"
            if stats_file.exists():
                stats_df = pd.read_csv(stats_file, index_col=0)
                mode_stats_by_model[model_name] = stats_df
            else:
                # Soft warning, not fatal
                print(
                    f"[load_compmodels_results] Warning: mode_stats file not found "
                    f"for model '{model_name}': {stats_file}"
                )

    # Additionally, compute K_max for each model and K_max for all
    K_max = 0
    for full_name, Q in Q_by_mode.items():
        K_max = max(K_max, Q.shape[1])
    K_max = max(K_max, 1)
    K_max_by_model = {}
    for model in models:
        K_model = 0
        for mode_short in modes_by_model[model]:
            full_name = f"{model}_{mode_short}"
            Q = Q_by_mode[full_name]
            K_model = max(K_model, Q.shape[1])
        K_max_by_model[model] = max(K_model, 1)

    return CompModelsResults(
        res_dir=res_dir,
        input_dir=input_dir,
        models=models,
        modes_by_model=modes_by_model,
        full_mode_names_by_model=full_mode_names_by_model,
        full_mode_names=full_mode_names,
        Q_by_mode=Q_by_mode,
        all_modes_alignment=all_modes_alignment,
        alignment_across_all=alignment_across_all,
        cost_across_all=cost_across_all,
        mode_stats_by_model=mode_stats_by_model,
        K_max=K_max,
        K_max_by_model=K_max_by_model,
    )


def update_clumppling_with_alignment(
    res: ClumpplingResults, 
    alignment: Dict[str, Sequence[int]]
    ) -> ClumpplingResults:
    """
    Return a *new* ClumpplingResults with alignment updated.
    """
    # make sure all keys in alignment are in modes
    assert np.all([mode in res.modes for mode in alignment])
    for mode in res.modes:
        if mode not in alignment:
            alignment[mode] = np.arange(res.mode_K[mode]).astype(int)

    # variables that do not change
    align_dir = res.align_dir
    suffix = res.suffix
    mode_stats = res.mode_stats
    modes = res.modes
    mode_K = res.mode_K
    K_range = res.K_range
    K_max = res.K_max
    mode_names_list=res.mode_names_list
    cost_acrossK=res.cost_acrossK
    mode_coord_dict=res.mode_coord_dict,
    mode_sep_coord_dict=res.mode_sep_coord_dict,
    input_meta=res.input_meta,
    Q_unaligned_by_mode=res.Q_unaligned_by_mode
    P_unaligned_by_mode=res.P_unaligned_by_mode
    mode_alignment=res.mode_alignment,
    alignment_acrossK=res.alignment_acrossK,

    # variables that change
    Q_by_mode_new = {}
    P_aligned_by_mode_new = {} if res.P_aligned_by_mode is not None else None
    all_modes_alignment_new = {}
    for mode in modes:
        Q_old = res.Q_by_mode[mode]
        align_pattern = np.array(alignment[mode])
        Q_new = Q_old[:, align_pattern]
        Q_by_mode_new[mode] = Q_new

        if P_aligned_by_mode_new is not None:
            P_old = res.P_aligned_by_mode[mode]
            P_new = P_old[:, align_pattern]
            P_aligned_by_mode_new[mode] = P_new

        old_alignment = res.all_modes_alignment[mode]
        # compose old and new alignment patterns
        align_pattern = np.array(old_alignment)[align_pattern]
        all_modes_alignment_new[mode] = align_pattern

    new_res = ClumpplingResults(
        align_dir=align_dir,
        suffix=suffix,
        mode_alignment=mode_alignment,
        mode_stats=mode_stats,
        modes=modes,
        mode_K=mode_K,
        K_range=K_range,
        K_max=K_max,
        mode_names_list=mode_names_list,
        Q_by_mode=Q_by_mode_new,
        alignment_acrossK=alignment_acrossK,
        cost_acrossK=cost_acrossK,
        all_modes_alignment=all_modes_alignment_new,
        mode_coord_dict=mode_coord_dict,
        mode_sep_coord_dict=mode_sep_coord_dict,
        input_meta=input_meta,
        Q_unaligned_by_mode=Q_unaligned_by_mode,
        P_unaligned_by_mode=P_unaligned_by_mode,
        P_aligned_by_mode=P_aligned_by_mode_new,
    )

    return new_res


def subset_compmodels_by_K(
    comp_res: CompModelsResults,
    K_min: Optional[int] = None,
    K_max: Optional[int] = None,
    K_values: Optional[Sequence[int]] = None,
) -> CompModelsResults:
    """
    Return a new CompModelsResults object restricted to a subset of K values.

    A mode is kept if its number of clusters K = Q.shape[1] satisfies:
      - if K_values is not None: K in K_values
      - else: K_min <= K <= K_max (with open ends if K_min/K_max is None)

    Parameters
    ----------
    comp_res : CompModelsResults
        Original full comparison results.
    K_min, K_max : int, optional
        Lower / upper bounds for K. Ignored if K_values is provided.
    K_values : sequence of int, optional
        Explicit set of K values to keep.

    Returns
    -------
    CompModelsResults
        New results object with only the selected modes and updated metadata.
    """
    # --- compute K per mode ------------------------------------------
    K_by_mode: Dict[str, int] = {
        full_name: Q.shape[1]
        for full_name, Q in comp_res.Q_by_mode.items()
    }

    if K_values is not None:
        K_set = set(K_values)

        def _keep(K: int) -> bool:
            return K in K_set
    else:
        def _keep(K: int) -> bool:
            if K_min is not None and K < K_min:
                return False
            if K_max is not None and K > K_max:
                return False
            return True

    # --- decide which full modes to keep --------------------------------
    kept_full_modes: List[str] = [
        name for name in comp_res.full_mode_names
        if _keep(K_by_mode[name])
    ]

    if not kept_full_modes:
        raise ValueError(
            "subset_compmodels_by_K: no modes remain after K filtering. "
            "Check K_min/K_max or K_values."
        )

    kept_full_modes_set = set(kept_full_modes)

    # --- rebuild per-model lists -------------------------------------
    modes_by_model_sub: Dict[str, List[str]] = {}
    full_mode_names_by_model_sub: Dict[str, List[str]] = {}
    models_sub: List[str] = []

    for model in comp_res.models:
        short_modes = comp_res.modes_by_model.get(model, [])
        full_for_model = []
        short_for_model = []

        for short in short_modes:
            full_name = f"{model}_{short}"
            if full_name in kept_full_modes_set:
                full_for_model.append(full_name)
                short_for_model.append(short)

        if full_for_model:
            models_sub.append(model)
            full_mode_names_by_model_sub[model] = full_for_model
            modes_by_model_sub[model] = short_for_model

    if not models_sub:
        raise ValueError(
            "subset_compmodels_by_K: no models have any modes left after K filtering."
        )

    # keep full_mode_names in original order
    full_mode_names_sub: List[str] = [
        name for name in comp_res.full_mode_names
        if name in kept_full_modes_set
    ]

    # --- construct Q_by_mode, all_modes_alignment ------------------------------
    Q_by_mode_sub: Dict[str, np.ndarray] = {
        name: comp_res.Q_by_mode[name]
        for name in full_mode_names_sub
    }

    all_modes_alignment_sub: Dict[str, Sequence[int]] = {
        name: comp_res.all_modes_alignment[name]
        for name in full_mode_names_sub
        if name in comp_res.all_modes_alignment
    }

    # --- construct alignment_across_all / cost_across_all ----------------------
    if comp_res.alignment_across_all is not None:
        alignment_across_all_sub: Dict[str, Sequence[int]] = {}
        cost_across_all_sub: Dict[str, float] = {}

        for key, mapping in comp_res.alignment_across_all.items():
            # key is "A-B"
            A, B = key.split("-", 1)
            if A in kept_full_modes_set and B in kept_full_modes_set:
                alignment_across_all_sub[key] = mapping
                if comp_res.cost_across_all is not None and key in comp_res.cost_across_all:
                    cost_across_all_sub[key] = comp_res.cost_across_all[key]
        if not alignment_across_all_sub:
            alignment_across_all_sub = {}
            cost_across_all_sub = {}
    else:
        alignment_across_all_sub = None
        cost_across_all_sub = None

    # --- construct mode_stats_by_model -----------------------------------------
    mode_stats_by_model_sub: Dict[str, pd.DataFrame] = {}
    for model in models_sub:
        if model not in comp_res.mode_stats_by_model:
            continue
        stats_df = comp_res.mode_stats_by_model[model]
        # index is expected to be the short mode name (e.g., 'K21M1')
        short_modes_sub = modes_by_model_sub[model]
        # keep only available ones (in case of missing rows)
        short_modes_sub = [m for m in short_modes_sub if m in stats_df.index]
        if short_modes_sub:
            mode_stats_by_model_sub[model] = stats_df.loc[short_modes_sub].copy()

    # --- recompute K_max and K_max_by_model --------------------------
    K_max_sub = 0
    for full_name, Q in Q_by_mode_sub.items():
        K_max_sub = max(K_max_sub, Q.shape[1])
    K_max_sub = max(K_max_sub, 1)

    K_max_by_model_sub: Dict[str, int] = {}
    for model in models_sub:
        K_model = 0
        for short in modes_by_model_sub[model]:
            full_name = f"{model}_{short}"
            Q = Q_by_mode_sub[full_name]
            K_model = max(K_model, Q.shape[1])
        K_max_by_model_sub[model] = max(K_model, 1)

    # --- build new CompModelsResults ---------------------------------
    return CompModelsResults(
        res_dir=comp_res.res_dir,
        input_dir=comp_res.input_dir,
        models=models_sub,
        modes_by_model=modes_by_model_sub,
        full_mode_names_by_model=full_mode_names_by_model_sub,
        full_mode_names=full_mode_names_sub,
        Q_by_mode=Q_by_mode_sub,
        all_modes_alignment=all_modes_alignment_sub,
        alignment_across_all=alignment_across_all_sub,
        cost_across_all=cost_across_all_sub,
        mode_stats_by_model=mode_stats_by_model_sub,
        K_max=K_max_sub,
        K_max_by_model=K_max_by_model_sub,
    )


def _open_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def _extract_attr(attr: str, key: str) -> Optional[str]:
    """
    Fast attribute extractor for GTF attributes.
    Looks for: key "value";
    Avoids regex for speed on huge files.
    """
    token = key + ' "'
    i = attr.find(token)
    if i == -1:
        return None
    i += len(token)
    j = attr.find('"', i)
    if j == -1:
        return None
    return attr[i:j]


def load_gene_intervals(
    gtf_file: str,
    *,
    upstream: int = 5000,
    downstream: int = 0,
    feature_type: str = "gene",
    source: Optional[str] = "HAVANA",
    gene_type_allowlist: Optional[Set[str]] = None,
) -> Dict[str, List[GeneT]]:
    """
    Stream a (possibly gzipped) GTF and extract only the intervals needed.
    Returns dict: chrom -> sorted list of (start, end, gene_name).
    """
    gtf: Dict[str, List[GeneT]] = {}
    append = None  # minor micro-opt for inner loop readability

    with _open_maybe_gzip(gtf_file) as f:
        for line in f:
            if not line or line[0] == "#":
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            chrom, src, feat = parts[0], parts[1], parts[2]
            if feat != feature_type:
                continue
            if source and src != source:
                continue

            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attr = parts[8]

            if gene_type_allowlist is not None:
                gt = _extract_attr(attr, "gene_type") or _extract_attr(attr, "gene_biotype")
                if gt is None or gt not in gene_type_allowlist:
                    continue

            gene_name = (
                _extract_attr(attr, "gene_name")
                or _extract_attr(attr, "gene")
                or _extract_attr(attr, "gene_id")
                or "unknown_gene"
            )

            # Extend relative to strand (same semantics as your reference)
            if strand == "-":
                ext_start = max(1, start - downstream)
                ext_end = end + upstream
            else:
                ext_start = max(1, start - upstream)
                ext_end = end + downstream

            if chrom not in gtf:
                gtf[chrom] = []
            gtf[chrom].append((ext_start, ext_end, gene_name))

    # Sort per chromosome for efficient overlap scan
    for chrom in list(gtf.keys()):
        gtf[chrom].sort(key=lambda g: (g[0], g[1], g[2]))

    return gtf


def parse_peak_string(peak: str) -> Tuple[str, int, int]:
    """
    'chr1:10109-10357' -> ('chr1', 10109, 10357)
    """
    chrom, rest = peak.split(":", 1)
    start, end = rest.split("-", 1)
    return chrom, int(start), int(end)


def build_peak_index(
    peaks: Iterable[str],
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, List[int]]]:
    """
    Build per-chromosome sorted intervals and start positions for fast overlap queries.
    Returns:
        intervals_by_chr: {chrom: [(start, end), ...] sorted by start}
        starts_by_chr:    {chrom: [start1, start2, ...] sorted}
    """
    intervals_by_chr: Dict[str, List[Tuple[int, int]]] = {}

    for s in peaks:
        chrom, start, end = parse_peak_string(s)
        intervals_by_chr.setdefault(chrom, []).append((start, end))

    # sort per chromosome
    starts_by_chr: Dict[str, List[int]] = {}
    for chrom, intervals in intervals_by_chr.items():
        intervals.sort(key=lambda x: x[0])
        starts_by_chr[chrom] = [s for s, _ in intervals]

    return intervals_by_chr, starts_by_chr


def has_overlap(
    chrom: str,
    start: int,
    end: int,
    intervals_by_chr: Dict[str, List[Tuple[int, int]]],
    starts_by_chr: Dict[str, List[int]],
) -> bool:
    """
    Check if the interval (chrom, start, end) overlaps any peak interval.
    Uses binary search on sorted starts for efficiency.
    """
    if chrom not in intervals_by_chr:
        return False

    intervals = intervals_by_chr[chrom]
    starts = starts_by_chr[chrom]

    # find first interval whose start >= end
    idx = bisect.bisect_left(starts, end)

    # scan backwards a bit to check possible overlaps
    # (all intervals with start < end and end_i > start overlap)
    for i in range(idx - 1, -1, -1):
        s_i, e_i = intervals[i]
        if e_i <= start:  # intervals before this can't overlap either
            break
        # overlap condition for closed intervals:
        # s_i <= end and start <= e_i
        if not (e_i <= start or end <= s_i):
            return True

    return False


def filter_bed_by_peaks_in_memory(
    bed_path: str | Path,
    peaks: Iterable[str],
    *,
    ccre_id_col: int = 3,
) -> Tuple[List[List[str]], Set[str]]:
    """
    Stream a BED file and keep only lines that overlap any of the given peaks.

    Returns
    -------
    filtered_rows : list of list[str]
        Each inner list is the BED line split by '\t'.
    kept_ids : set of str
        Set of cCRE IDs (from column `ccre_id_col`) for filtered rows.
    """
    intervals_by_chr, starts_by_chr = build_peak_index(peaks)

    bed_path = Path(bed_path)
    filtered_rows: List[List[str]] = []
    kept_ids: Set[str] = set()

    with bed_path.open() as fin:
        for line in fin:
            if not line.strip():
                continue
            if line.startswith("#"):
                # skip comments; remove this if you want to keep them
                continue

            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            if has_overlap(chrom, start, end, intervals_by_chr, starts_by_chr):
                filtered_rows.append(fields)
                if len(fields) > ccre_id_col:
                    kept_ids.add(fields[ccre_id_col])

    return filtered_rows, kept_ids


def filter_gene_links_by_ccre_ids_in_memory(
    gene_links_path: str | Path,
    kept_ids: Set[str],
    *,
    ccre_id_col: int = 0,
    keep_header: bool = True,
) -> Tuple[List[str] | None, List[List[str]]]:
    """
    Stream a gene-link file and keep only rows whose cCRE ID is in kept_ids.

    Returns
    -------
    header : list[str] or None
        Header columns if keep_header=True and file has at least one line,
        otherwise None.
    filtered_rows : list of list[str]
        Data rows (split by '\t') with cCRE IDs in kept_ids.
    """
    gene_links_path = Path(gene_links_path)

    header: List[str] | None = None
    filtered_rows: List[List[str]] = []
    first = True

    with gene_links_path.open() as fin:
        for line in fin:
            if not line.strip():
                continue

            fields = line.rstrip("\n").split("\t")

            if first and keep_header:
                header = fields
                first = False
                continue

            first = False

            if len(fields) <= ccre_id_col:
                continue

            ccre_id = fields[ccre_id_col]
            if ccre_id in kept_ids:
                filtered_rows.append(fields)

    return header, filtered_rows
