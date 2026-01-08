"""
analysis.py

Functions for organizing, processing, and analyzing clumppling results.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union, Optional, Any, Iterable, Callable

import numpy as np
import pandas as pd
from clumppling.core import alignQ_wrtP
from clumppling.utils import cost_membership, get_uniq_lb_sep

from .io import load_gene_intervals
from .io import ClumpplingResults, CompModelsResults

PathLike = Union[str, Path]

def subset_clumppling_results_by_modes(
    results: ClumpplingResults,
    modes_subset: Sequence[str],
) -> ClumpplingResults:
    """
    Return a new ClumpplingResults object containing only a subset of modes.

    Parameters
    ----------
    results : ClumpplingResults
        Original full results.
    modes_subset : sequence of str
        Mode names to keep (must exist in results.Q_by_mode).

    Returns
    -------
    subset : ClumpplingResults
        New object with the same fields as the original, but restricted
        to the selected modes.
    """
    # ---- Normalize + validate modes_subset ----
    modes_subset = list(dict.fromkeys(modes_subset))  # dedupe, keep order

    missing = [m for m in modes_subset if m not in results.Q_by_mode]
    if missing:
        raise KeyError(f"Modes not found in results.Q_by_mode: {missing}")

    # ---- Subset per-mode dicts (Q, P, stats, mode_alignment, all_modes_alignment) ----
    Q_by_mode = {m: results.Q_by_mode[m] for m in modes_subset}

    Q_unaligned_by_mode = None
    if results.Q_unaligned_by_mode is not None:
        Q_unaligned_by_mode = {
            m: results.Q_unaligned_by_mode[m]
            for m in modes_subset
            if m in results.Q_unaligned_by_mode
        }

    P_unaligned_by_mode = None
    if results.P_unaligned_by_mode is not None:
        P_unaligned_by_mode = {
            m: results.P_unaligned_by_mode[m]
            for m in modes_subset
            if m in results.P_unaligned_by_mode
        }

    P_aligned_by_mode = None
    if results.P_aligned_by_mode is not None:
        P_aligned_by_mode = {
            m: results.P_aligned_by_mode[m]
            for m in modes_subset
            if m in results.P_aligned_by_mode
        }

    # mode_stats
    mode_stats = results.mode_stats
    if mode_stats is not None and not mode_stats.empty:
        if "Mode" in mode_stats.columns:
            mode_stats = mode_stats[mode_stats["Mode"].isin(modes_subset)].copy()
        else:
            mode_stats = mode_stats.loc[
                mode_stats.index.intersection(modes_subset)
            ].copy()
    else:
        mode_stats = None

    # mode_alignment
    mode_alignment = results.mode_alignment
    if mode_alignment is not None:
        # DataFrame with "Mode" column
        if hasattr(mode_alignment, "columns") and "Mode" in getattr(
            mode_alignment, "columns", []
        ):
            mode_alignment = mode_alignment[
                mode_alignment["Mode"].isin(modes_subset)
            ].copy()
        # dict keyed by mode
        elif isinstance(mode_alignment, dict):
            mode_alignment = {
                m: mode_alignment[m] for m in modes_subset if m in mode_alignment
            }

    all_modes_alignment = results.all_modes_alignment
    if all_modes_alignment is not None:
        all_modes_alignment = {
            m: all_modes_alignment[m]
            for m in modes_subset
            if m in all_modes_alignment
        }

    # ---- Recompute structural layout fields from Q_by_mode ----
    modes = modes_subset

    # K per mode and aggregates
    mode_K_map: Dict[str, int] = {m: Q_by_mode[m].shape[1] for m in modes}
    mode_K = [mode_K_map[m] for m in modes]

    K_range = sorted(set(mode_K_map.values()))
    K_max = max(mode_K_map.values())

    # mode_names_list
    mode_names_list: list[list[str]] = [
        [m for m in modes if mode_K_map[m] == K] for K in K_range
    ]

    # mode_coord_dict: where each mode sits in (row=K index, col=within-K index)
    mode_coord_dict: Dict[str, Tuple[int, int]] = {}
    for row_idx, (K, row_modes) in enumerate(zip(K_range, mode_names_list)):
        for col_idx, m in enumerate(row_modes):
            mode_coord_dict[m] = (row_idx, col_idx)

    # mode_sep_coord_dict: (mode, k) -> (row=mode index, col=cluster index)
    mode_sep_coord_dict: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for row_idx, m in enumerate(modes):
        K = mode_K_map[m]
        for k in range(K):
            mode_sep_coord_dict[(m, k)] = (row_idx, k)

    # ---- alignment_acrossK / cost_acrossK for the subset ----
    alignment_acrossK = {}
    cost_acrossK = {}
    if results.alignment_acrossK is not None:
        for pair_label, mapping in results.alignment_acrossK.items():
            mode_A, mode_B = pair_label.split("-")
            if mode_A in modes and mode_B in modes:
                alignment_acrossK[pair_label] = mapping
                if (
                    results.cost_acrossK is not None
                    and pair_label in results.cost_acrossK
                ):
                    cost_acrossK[pair_label] = results.cost_acrossK[pair_label]

    # ---- Build the new ClumpplingResults object
    subset = ClumpplingResults(
        align_dir=results.align_dir,
        suffix=results.suffix,
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
        input_meta=results.input_meta,
        Q_unaligned_by_mode=Q_unaligned_by_mode,
        P_unaligned_by_mode=P_unaligned_by_mode,
        P_aligned_by_mode=P_aligned_by_mode,
    )
    return subset


# ---------------------------------------------------------------------
# Feature-level summaries (P/Q -> sepLFC, sepCls, weighted_Psum)
# ---------------------------------------------------------------------

def compute_profile(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute clustering profile for feature-level P.

    Parameters
    ----------
    P : array-like, shape (M, K)
        Per-feature values over clusters (e.g. log-P or scores).

    Returns
    -------
    LFC_sorted : (M, K-1)
        Log2 fold-changes between consecutive sorted values per feature.
    idx_sorted : (M, K)
        Indices of clusters sorted per feature (ascending).
    """
    K = P.shape[1]
    P_sorted = np.sort(P, axis=1)
    idx_sorted = np.argsort(P, axis=1)
    # log2 ratio between consecutive sorted entries
    LFC_sorted = np.log2(P_sorted[:, 1:]) - np.log2(P_sorted[:, : K - 1])
    return LFC_sorted, idx_sorted


def get_sepLFC_from_profile(
    LFC_sorted: np.ndarray,
    idx_sorted: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """
    Compute sepLFC and sepCls from clustering profile.
    
    Parameters
    ----------
    LFC_sorted : np.ndarray
        Log2 fold-changes between consecutive sorted values per feature.
    idx_sorted : np.ndarray
        Indices of clusters sorted per feature (ascending).

    Returns
    -------
    sepLFC : np.ndarray
        Maximum log2 fold-change per feature.
    sepCls : list of tuples
        Each tuple contains two tuples representing the indices of clusters
        on each side of the maximum gap, in original cluster indices.
    """
    # index of max gap (in sorted-order coordinates)
    idx_sepLFC = np.argmax(LFC_sorted, axis=1)
    sepLFC = np.max(LFC_sorted, axis=1)

    M = LFC_sorted.shape[0]
    sepCls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for i_g in range(M):
        # in sorted space: low clusters up to and including idx_sepLFC
        idx_l = idx_sorted[i_g, 0 : idx_sepLFC[i_g] + 1]
        # high clusters after the gap
        idx_h = idx_sorted[i_g, idx_sepLFC[i_g] + 1 :]
        sepCls.append((tuple(idx_l), tuple(idx_h)))

    return sepLFC, sepCls


def get_wPsum_from_PQ(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute a weighted sum of P across clusters, using cluster weights from Q.

    Parameters
    ----------
    P : np.ndarray
        Feature-by-cluster loadings.
    Q : np.ndarray
        Cell-by-cluster memberships (aligned).

    Returns
    -------
    weighted_Psum : np.ndarray
        Weighted sum of P across clusters.
    """
    cls_wt = np.sum(Q, axis=0)
    cls_wt /= np.sum(cls_wt)
    weighted_Psum = P @ cls_wt
    return weighted_Psum


def compute_feature_metrics_for_mode(
    P: np.ndarray,
    Q: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Compute sepLFC, sepCls, and weighted_Psum for a single mode.

    Parameters
    ----------
    P : array, shape (n_features, K)
        Feature-by-cluster loadings.
    Q : array, shape (n_cells, K)
        Cell-by-cluster memberships (aligned).
    feature_names : sequence of str
        Names for each row of P (e.g. gene IDs/symbols). Must have length
        equal to P.shape[0].

    Returns
    -------
    df : DataFrame
        Index = feature_names
        Columns = ["weighted_Psum", "sepLFC", "sepCls"]
    """
    if P.shape[0] != len(feature_names):
        raise ValueError(
            f"P.shape[0] ({P.shape[0]}) != len(feature_names) ({len(feature_names)})"
        )

    LFC_sorted, idx_sorted = compute_profile(P)
    sepLFC, sepCls = get_sepLFC_from_profile(LFC_sorted, idx_sorted)
    weighted_Psum = get_wPsum_from_PQ(P, Q)

    df = pd.DataFrame(
        {
            "weighted_Psum": weighted_Psum,
            "sepLFC": sepLFC,
            "sepCls": sepCls,
        },
        index=list(feature_names),
    )

    # drop duplicate feature names, keeping first occurrence
    df = df[~df.index.duplicated(keep="first")]
    return df


def compute_feature_metrics_all_modes(
    results: ClumpplingResults,
    feature_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Compute feature-level metrics (weighted_Psum, sepLFC, sepCls) for all modes.

    Parameters
    ----------
    results : ClumpplingResults
        Must have P_aligned_by_mode populated (i.e. load_clumppling_results
        was called with cls_dir=...).
    feature_names : sequence of str
        Names for each row of P (e.g. gene IDs/symbols).

    Returns
    -------
    df_by_mode : dict
        {mode_name -> DataFrame as returned by compute_feature_metrics_for_mode}
    """
    if results.P_aligned_by_mode is None:
        raise ValueError(
            "results.P_aligned_by_mode is None. Did you call load_clumppling_results "
            "with cls_dir=... ?"
        )

    df_by_mode: Dict[str, pd.DataFrame] = {}

    for mode_name in results.modes:
        P = results.P_aligned_by_mode[mode_name]
        Q = results.Q_by_mode[mode_name]
        df_by_mode[mode_name] = compute_feature_metrics_for_mode(P, Q, feature_names)

    return df_by_mode


def select_top_features_by_weighted_Psum(
    df_by_mode: Mapping[str, pd.DataFrame],
    top_quantile: float = 0.1,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, set]:
    """
    For each mode, select the top features by weighted_Psum and related information.

    Parameters
    ----------
    df_by_mode : mapping
        {mode_name -> feature metrics DataFrame}.
    top_quantile : float, default 0.1
        We keep features with weighted_Psum above this upper quantile
        (i.e. the top (1 - top_quantile) fraction).

    Returns
    -------
    selected_by_mode : dict
        {mode_name -> DataFrame of selected features, with columns suffixed
         by f"_{mode_name}"}.
    df_selected_all : DataFrame
        Inner join of all per-mode selected DataFrames.
    overlap : set
        Set of feature names present in all per-mode selections.
    """
    selected_by_mode: Dict[str, pd.DataFrame] = {}
    df_selected_list: List[pd.DataFrame] = []
    selected_feature_sets: List[set] = []

    for mode_name, df in df_by_mode.items():
        if "weighted_Psum" not in df.columns:
            raise KeyError(f"'weighted_Psum' not found in df_by_mode[{mode_name!r}].columns")

        wPsum_thre = df["weighted_Psum"].quantile(1 - top_quantile)
        df_sel = df[df["weighted_Psum"] > wPsum_thre].copy()

        selected_feature_sets.append(set(df_sel.index))

        # suffix columns with mode name to avoid collisions when concatenating
        df_sel_suffixed = df_sel.add_suffix(f"_{mode_name}")
        selected_by_mode[mode_name] = df_sel_suffixed
        df_selected_list.append(df_sel_suffixed)

    if not df_selected_list:
        df_selected_all = pd.DataFrame()
        overlap: set = set()
    else:
        df_selected_all = pd.concat(df_selected_list, axis=1, join="inner")
        # intersection over all per-mode selections
        overlap = set.intersection(*selected_feature_sets) if selected_feature_sets else set()

    return selected_by_mode, df_selected_all, overlap


# ---------------------------------------------------------------------
# Alignment graph & pairwise cluster mappings
# ---------------------------------------------------------------------

def _build_alignment_graph(
    alignment_acrossK: Mapping[str, Sequence[int]],
) -> Dict[str, List[str]]:
    """
    Build a directed graph where each key "A-B" gives an edge B -> A,
    since mapping[k_B] = k_A maps clusters in B to clusters in A.
    """
    from collections import defaultdict

    graph: Dict[str, List[str]] = defaultdict(list)
    for pair in alignment_acrossK.keys():
        mode1, mode2 = pair.split("-")
        # mapping is from mode2 -> mode1
        graph[mode2].append(mode1)
    return graph


def _find_directed_path(
    graph: Mapping[str, Sequence[str]],
    src: str,
    dst: str,
) -> List[str] | None:
    """
    BFS to find a path from src to dst in a directed graph (edges in graph[src]).
    Returns a list [src, ..., dst], or None if no path.
    """
    from collections import deque

    q = deque([src])
    parent = {src: None}

    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in graph.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)

    if dst not in parent:
        return None

    # reconstruct path
    path: List[str] = []
    v = dst
    while v is not None:
        path.append(v)
        v = parent[v]
    path.reverse()
    return path


def extract_all_mode_pair_mappings(mode_names, all_modes_alignment, alignment_acrossK):
    """
    For each pair of modes (A, B), compute the mapping from clusters in B to clusters
    in A, in *aligned column space*, using paths through intermediate modes.

    Parameters
    ----------
    mode_names : list of str
        Modes you care about (e.g. sorted list of all_modes_alignment.keys()).
    all_modes_alignment : dict
        {mode_name -> reordering}, where `reordering` is the alignment pattern
        used for columns in that mode (same object you indexed in your plots).
    alignment_acrossK : dict
        {"A-B" -> mapping}, where for key "A-B", `mapping[k_B] = k_A` maps
        original cluster index in mode B to original index in mode A.

    Returns
    -------
    pair_mappings : dict
        {
          "A-B": [(col_idx_in_A, col_idx_in_B), ...],
          ...
        }
        All indices are in the *current aligned column order* (after alignment),
        i.e. x-axis column indices in your plots.
        For each pair, the mapping is from clusters of B → clusters of A.
    """
    # directed graph: edges B -> A for key "A-B"
    graph = _build_alignment_graph(alignment_acrossK)

    # make sure we have reordering as lists
    all_modes_alignment_lists = {
        m: list(all_modes_alignment[m]) for m in mode_names
    }

    pair_mappings = {}

    for i, mode_A in enumerate(mode_names):
        reord_A = all_modes_alignment_lists[mode_A]
        K_A = len(reord_A)

        for j, mode_B in enumerate(mode_names):
            if i == j:
                continue

            reord_B = all_modes_alignment_lists[mode_B]
            K_B = len(reord_B)
            if K_A <= K_B:
                # find a path B -> ... -> A using directed edges (B->A)
                path = _find_directed_path(graph, src=mode_B, dst=mode_A)
                if path is None:
                    print(f"Skipping {mode_B}->{mode_A}")
                    # no directed path in B->A orientation; skip
                    continue
                # start with identity: each cluster in B maps to itself
                idx_vec = np.arange(K_B, dtype=int)

                # compose mappings along the path
                for t in range(len(path) - 1):
                    cur = path[t]        # current mode (source of mapping step)
                    nxt = path[t + 1]    # next mode (closer to A)

                    key = f"{nxt}-{cur}"   # mapping from cur -> nxt: mapping[k_cur] = k_nxt
                    if key not in alignment_acrossK:
                        raise KeyError(f"Missing alignment key '{key}' for step {cur} -> {nxt}")

                    mapping = alignment_acrossK[key]
                    # map each current index via mapping (cur indices -> nxt indices)
                    idx_vec = np.array([mapping[k] for k in idx_vec], dtype=int)

                # idx_vec[k_B] is now the ORIGINAL cluster index in mode_A
                # corresponding to cluster k_B in mode_B.
                col_pairs = []
                for k_B, orig_kA in enumerate(idx_vec):
                    # aligned column index for that original cluster in A
                    col_A = reord_A.index(orig_kA)
                    # aligned column index for cluster k_B in B
                    col_B = reord_B.index(k_B)
                    col_pairs.append((col_A, col_B))

                pair_label = f"{mode_A}-{mode_B}"  # A is "target", B is "source"
                pair_mappings[pair_label] = col_pairs

    return pair_mappings


# ---------------------------------------------------------------------
# Mapping and merging of alt_Q into ref_Q space
# ---------------------------------------------------------------------

def map_alt_to_ref(
    ref_Q: np.ndarray,
    alt_Q: np.ndarray,
    pair_mapping: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map alt_Q into ref_Q space using pair_mapping.

    Parameters
    ----------
    ref_Q : np.ndarray
        Reference membership matrix (n_cells, ref_K).
    alt_Q : np.ndarray
        Alternative membership matrix (n_cells, alt_K).
    pair_mapping : Sequence[Tuple[int, int]]
        Mapping pairs (i_ref, j_alt) indicating how clusters in alt_Q map to clusters in ref_Q.

    Assumes
        ref_K <= alt_K
        pair_mapping is (i_ref, j_alt)

    Returns
    -------
    alt_Q_mapped : np.ndarray
        Mapped alternative membership matrix (n_cells, ref_K).
    diff_Q : np.ndarray
        Absolute difference between ref_Q and alt_Q_mapped.
    """
    if ref_Q.ndim != 2 or alt_Q.ndim != 2:
        raise ValueError("ref_Q and alt_Q must be 2D.")

    n_cells, alt_K = alt_Q.shape
    ref_K = int(ref_Q.shape[1])

    if ref_Q.shape[0] != n_cells:
        raise ValueError("ref_Q and alt_Q must have the same number of cells.")

    if ref_K > alt_K:
        raise ValueError(
            f"map_alt_to_ref requires ref_K <= alt_K, got {ref_K} > {alt_K}."
        )

    pairs = np.asarray(pair_mapping, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_mapping must be a sequence of (i_ref, j_alt) pairs.")

    i_ref = pairs[:, 0]
    j_alt = pairs[:, 1]

    if np.any(i_ref < 0) or np.any(i_ref >= ref_K):
        raise ValueError("pair_mapping contains i_ref out of bounds for ref_K.")
    if np.any(j_alt < 0) or np.any(j_alt >= alt_K):
        raise ValueError("pair_mapping contains j_alt out of bounds for alt_K.")

    alt_Q_mapped_T = np.zeros((ref_K, n_cells), dtype=float)
    np.add.at(alt_Q_mapped_T, i_ref, alt_Q[:, j_alt].T)
    alt_Q_mapped = alt_Q_mapped_T.T

    diff_Q = np.abs(ref_Q - alt_Q_mapped)
    return alt_Q_mapped, diff_Q


# ---------------------------------------------------------------------
# Membership differences
# ---------------------------------------------------------------------

def compute_overall_membership_difference(
    diff_Q: np.ndarray,
) -> float:
    """
    Compute overall membership difference as average absolute difference
    per cell, aggregated over all clusters.

    Parameters
    ----------
    diff_Q : array-like, shape (n_cells, K)
        Per-cell, per-cluster absolute differences.

    Returns
    -------
    overall_diff : float
        Average absolute difference per cell.
    """
    n_cells = diff_Q.shape[0]
    overall_diff = float(diff_Q.sum() / n_cells)
    return overall_diff

def compute_per_cell_membership_difference(
    diff_Q: np.ndarray,
    *,
    aggregation: str = "sum",
) -> np.ndarray:
    """
    Compute per-cell membership difference by aggregating over clusters.

    Parameters
    ----------
    diff_Q : array-like, shape (n_cells, K)
        Per-cell, per-cluster absolute differences.
    aggregation : str, default "max"
        Aggregation method over clusters:
          - "sum"  : recommended, normalized by 2 (maximum possible diff per cell)
          - "max"
          - "mean"

    Returns
    -------
    per_cell_diff : array, shape (n_cells,)
        Aggregated per-cell difference scores.
    """
    if diff_Q.ndim != 2:
        raise ValueError("diff_Q must be 2D (n_cells, K).")

    agg = aggregation.lower().strip()
    if agg == "sum":
        return diff_Q.sum(axis=1)/2
    if agg == "max":
        return diff_Q.max(axis=1)
    if agg == "mean":
        return diff_Q.mean(axis=1)
    raise ValueError("aggregation must be one of: 'max', 'sum', 'mean'.")

def get_compmodels_diff_matrices_against_ref(
    comp_res,
    pair_mappings: Dict[str, Sequence[Tuple[int, int]]],
    ref_mode: str,
    models: Optional[Sequence[str]] = None,
    *,
    strict_pair_mapping: bool = True,
):
    """
    Compute per-cell membership difference matrices for all modes vs a reference mode.

    For each model and each mode, this function:
      - Aligns that mode's Q matrix to the reference Q (ref_mode) using map_alt_to_ref
        and the provided pair_mappings.
      - Returns the aligned difference matrix diff_Q (same as used in plotting),
        without any plotting.

    Logic mirrors plot_compmodels_diff_grid_against_ref:
      - When ref_K <= K_cur:
          uses pair_mappings[f"{ref_mode}-{full_name}"]
          and calls map_alt_to_ref(ref_Q, Q, pair_mapping).
      - When ref_K > K_cur:
          uses pair_mappings[f"{full_name}-{ref_mode}"]
          and calls map_alt_to_ref(Q, ref_Q, pair_mapping).

    Parameters
    ----------
    comp_res
        Object holding compModels results, expected to have:
          - comp_res.modes_by_model : Dict[str, List[str]]
          - comp_res.get_Q(name)    : -> np.ndarray of shape (n_cells, K)
    pair_mappings : dict
        Mapping of keys like "refMode-altMode" to a list of (i_ref, i_alt) index pairs.
    ref_mode : str
        Full mode name used as the reference (e.g. "rna.seurat_louvain_K15M1").
    models : sequence of str, optional
        Subset of models to process. If None, all models in comp_res.modes_by_model
        are processed.
    strict_pair_mapping : bool, default True
        If True, missing pair mappings raise KeyError.
        If False, missing mappings yield a zero diff_Q of shape (n_cells, min(ref_K, K_cur)).

    Returns
    -------
    diff_by_model_mode : Dict[str, Dict[str, np.ndarray]]
        Nested dict:
          diff_by_model_mode[model_name][short_mode] = diff_Q

        - `diff_Q` has shape (n_cells, K_eff), where K_eff is typically the smaller
          K between ref and current (as produced by map_alt_to_ref).
        - For the reference mode itself, diff_Q is a zero matrix with the same shape
          as ref_Q.

    Notes
    -----
    - This function *does not* call compute_overall_membership_difference; it returns
      the full diff_Q matrices so you can aggregate however you like.
    """

    # ---- get model -> modes mapping ----
    modes_by_model = getattr(comp_res, "modes_by_model", None)
    if modes_by_model is None:
        raise AttributeError("comp_res must have `modes_by_model`")

    all_models = list(modes_by_model.keys())
    if models is None:
        models = all_models

    # ---- reference Q ----
    ref_Q = comp_res.get_Q(ref_mode)
    n_cells, ref_K = int(ref_Q.shape[0]), int(ref_Q.shape[1])

    def _to_full_mode(model: str, mode_entry: str) -> str:
        """Ensure mode name has model prefix."""
        return mode_entry if str(mode_entry).startswith(model + "_") else f"{model}_{mode_entry}"

    def _short_mode(model: str, full_name: str) -> str:
        """Strip model_ prefix for compact key."""
        prefix = model + "_"
        return full_name[len(prefix):] if full_name.startswith(prefix) else full_name

    diff_by_model_mode: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in models:
        mode_entries = list(modes_by_model[model_name])
        diff_by_model_mode[model_name] = {}

        for mode_entry in mode_entries:
            full_name = _to_full_mode(model_name, str(mode_entry))
            short_name = _short_mode(model_name, full_name)

            # Reference mode: diff is 0 by definition.
            if full_name == ref_mode:
                diff_Q = np.zeros_like(ref_Q, dtype=float)
                diff_by_model_mode[model_name][short_name] = diff_Q
                continue

            # Get current Q and its K
            Q = comp_res.get_Q(full_name)
            K_cur = int(Q.shape[1])

            # If ref_K <= K_cur, map alt -> ref space
            if ref_K <= K_cur:
                key = f"{ref_mode}-{full_name}"
                pair_mapping = pair_mappings.get(key)
                if pair_mapping is None:
                    if strict_pair_mapping:
                        raise KeyError(f"Missing pair mapping key: {key}")
                    # Fallback: zero diff in ref-space
                    diff_Q = np.zeros((n_cells, ref_K), dtype=float)
                else:
                    _, diff_Q = map_alt_to_ref(ref_Q, Q, pair_mapping)

            # If ref_K > K_cur, map ref -> current space
            else:
                key = f"{full_name}-{ref_mode}"
                pair_mapping = pair_mappings.get(key)
                if pair_mapping is None:
                    if strict_pair_mapping:
                        raise KeyError(f"Missing pair mapping key: {key}")
                    # Fallback: zero diff in current K space
                    diff_Q = np.zeros((n_cells, K_cur), dtype=float)
                else:
                    _, diff_Q = map_alt_to_ref(Q, ref_Q, pair_mapping)

            diff_by_model_mode[model_name][short_name] = diff_Q

    return diff_by_model_mode



def get_pairwise_diff_Q(mode_list_1, mode_list_2, pair_mappings, comp_results):
    """
    For each pair of modes (m1 in mode_list_1, m2 in mode_list_2),
    compute the diff_Q matrix between their Q matrices after 
    mapping the smaller-K mode into the larger-K mode's space.

    Parameters
    ----------
    mode_list_1 : list of str
        First list of mode names.
    mode_list_2 : list of str
        Second list of mode names.
    pair_mappings : dict
        As returned by extract_all_mode_pair_mappings.
    comp_results : ClumpplingResults
        Must have Q_by_mode populated.

    Returns
    -------
    diff_Q_dict : dict
        {(m1, m2): diff_Q array}
    """
    diff_Q_dict: Dict[Tuple[str, str], np.ndarray] = {}

    for m1 in mode_list_1:
        if m1 not in comp_results.Q_by_mode:
            raise KeyError(f"Mode {m1!r} not found in comp_results.Q_by_mode.")
        Q1 = comp_results.Q_by_mode[m1]

        for m2 in mode_list_2:
            if m2 not in comp_results.Q_by_mode:
                raise KeyError(f"Mode {m2!r} not found in comp_results.Q_by_mode.")
            Q2 = comp_results.Q_by_mode[m2]

            if m1 == m2:
                diff_Q = np.zeros_like(Q1)
            else:
                K1 = int(Q1.shape[1])
                K2 = int(Q2.shape[1])

                if K1 <= K2:
                    key = f"{m1}-{m2}"
                    pair_mapping = pair_mappings.get(key)
                    if pair_mapping is None:
                        raise KeyError(f"Missing pair mapping key: {key}")
                    _, diff_Q = map_alt_to_ref(Q1, Q2, pair_mapping)
                else:
                    key = f"{m2}-{m1}"
                    pair_mapping = pair_mappings.get(key)
                    if pair_mapping is None:
                        raise KeyError(f"Missing pair mapping key: {key}")
                    _, diff_Q = map_alt_to_ref(Q2, Q1, pair_mapping)

            diff_Q_dict[(m1, m2)] = diff_Q

    return diff_Q_dict


def get_pairwise_overall_membership_diff(
    diff_Q_dict: Mapping[Tuple[str, str], np.ndarray],
) -> Dict[Tuple[str, str], float]:
    """
    From a dict of diff_Q matrices, compute overall membership differences.

    Parameters
    ----------
    diff_Q_dict : dict
        {(m1, m2): diff_Q array}

    Returns
    -------
    overall_diff_dict : dict
        {(m1, m2): overall_diff}
    """
    overall_diff_dict: Dict[Tuple[str, str], float] = {}

    for mode_pair, diff_Q in diff_Q_dict.items():
        overall_diff = compute_overall_membership_difference(diff_Q)
        overall_diff_dict[mode_pair] = overall_diff

    return overall_diff_dict


# ---------------------------------------------------------------------
# Annotation-group difference summaries
# ---------------------------------------------------------------------

def build_annotation_group_indices(
    annotation_labels: Sequence[Any] | pd.Series,
) -> Dict[str, np.ndarray]:
    """
    Build mapping: annotation_group -> np.ndarray of integer indices.

    Parameters
    ----------
    annotation_labels
        Per-cell labels (e.g., cell types, domains, batches).
        Length must equal n_cells.

    Returns
    -------
    dict
        {group_label: indices}
    """
    s = pd.Series(annotation_labels, dtype="object")
    idx_dict = s.groupby(s).indices  # label -> Int64Index
    return {str(k): np.asarray(v, dtype=int) for k, v in idx_dict.items()}


def compute_annotation_group_sizes(
    annotation_labels: Sequence[Any] | pd.Series,
) -> pd.Series:
    """
    Compute group sizes as a sorted Series.
    """
    group_indices = build_annotation_group_indices(annotation_labels)
    sizes = {g: len(idxs) for g, idxs in group_indices.items()}
    return pd.Series(sizes).sort_index()


def compute_annotation_group_membership_difference(
    per_cell_diff: np.ndarray,
    group_indices: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute average membership_difference of cells in the annotation groups.

    Parameters
    ----------
    per_cell_diff
        1D array of length n_cells.
        Represents an aggregated per-cell difference score.
    group_indices
        dict: group -> indices

    Returns
    -------
    dict
        group -> fraction_diff
    """
    out: Dict[str, float] = {}
    for group, idx in group_indices.items():
        if len(idx) == 0:
            out[group] = 0.0
            continue
        vals = per_cell_diff[idx]
        out[group] = float(vals.mean())
    return out

def compute_mode_total_and_annotation_group_diffs(
    mode_Q: np.ndarray,
    ref_Q: np.ndarray,
    mode_name: Sequence[str],
    ref_mode: str,
    pair_mappings: Mapping[str, Sequence[Tuple[int, int]]],
    annotation_labels: Optional[Sequence[Any]] = None,
    aggregation: str = "sum",
) -> Tuple[float, np.ndarray, Optional[Dict[str, float]]]:
    """
    Compute overall, per-cell, and annotation-group membership differences

    Parameters
    ----------
    mode_Q : np.ndarray
        Membership matrix for the current mode (n_cells, alt_K).
    ref_Q : np.ndarray
        Reference membership matrix (n_cells, ref_K).
    mode_name : Sequence[str]
        Name of the current mode.
    ref_mode : str
        Name of the reference mode.
    pair_mappings : Mapping[str, Sequence[Tuple[int, int]]]
        Mapping pairs for cluster correspondences between modes.
    annotation_labels : Optional[Sequence[Any]], optional
        Per-cell annotation labels for grouping, by default None
    aggregation : str, optional
        Aggregation method for per-cell differences, by default "sum"

    Returns
    -------
    Tuple[float, np.ndarray, Optional[Dict[str, float]]]
        Overall difference, per-cell differences, and optional group differences.
    """
    n_cells = int(ref_Q.shape[0])
    ref_K = int(ref_Q.shape[1])
    alt_K = int(mode_Q.shape[1])
    if ref_K <= alt_K:
        key = f"{ref_mode}-{mode_name}"
        pair_mapping = pair_mappings.get(key)
        if pair_mapping is None:
            raise KeyError(f"Missing pair mapping key: {key}")
        else:
            Q_mapped, diff_Q = map_alt_to_ref(ref_Q, mode_Q, pair_mapping)
    else:
        # current has smaller K; map ref into current space
        key = f"{mode_name}-{ref_mode}"
        pair_mapping = pair_mappings.get(key)
        if pair_mapping is None:
            raise KeyError(f"Missing pair mapping key: {key}")
        else:
            ref_mapped, diff_Q = map_alt_to_ref(mode_Q, ref_Q, pair_mapping)
    overall_diff = compute_overall_membership_difference(diff_Q)
    per_cell_diff = compute_per_cell_membership_difference(
        diff_Q,
        aggregation=aggregation,
    )
    if annotation_labels is None:
        group_diff = None
    else:
        annotation_group_indices = build_annotation_group_indices(annotation_labels)
        group_diff = compute_annotation_group_membership_difference(
            per_cell_diff,
            annotation_group_indices,
        )
    return overall_diff, per_cell_diff, group_diff

def compute_modes_total_and_annotation_group_diffs(
    comp_results: ClumpplingResults, 
    ref_mode: str,
    pair_mappings: Mapping[str, Sequence[Tuple[int, int]]],
    annotation_labels: Optional[Sequence[Any]] = None,
    aggregation: str = "sum"
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute overall and annotation-group membership differences for all modes against a reference mode.
    
    Parameters
    ----------
    comp_results : ClumpplingResults
        Results object containing membership matrices and mode information.
    ref_mode : str
        Name of the reference mode.
    pair_mappings : Mapping[str, Sequence[Tuple[int, int]]]
        Mapping pairs for cluster correspondences between modes.
    annotation_labels : Optional[Sequence[Any]], optional
        Per-cell annotation labels for grouping, by default None
    aggregation : str, optional
        Aggregation method for per-cell differences, by default "sum"

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, Dict[str, float]]]
        Overall differences and annotation-group differences for all modes.
    """
    if ref_mode not in comp_results.Q_by_mode:
        raise KeyError(f"Reference mode {ref_mode!r} not found in comp_results.Q_by_mode.")
    ref_Q = comp_results.Q_by_mode[ref_mode]
    ref_K = int(ref_Q.shape[1])
    n_cells = int(ref_Q.shape[0])

    modes_total_diff: Dict[str, float] = {}
    modes_group_diff: Dict[str, Dict[str, float]] = {}

    for mode_name in comp_results.full_mode_names:
        Q = comp_results.Q_by_mode[mode_name]

        if mode_name == ref_mode:
            modes_total_diff[mode_name] = 0.0
            if annotation_labels is not None:
                annotation_group_indices = build_annotation_group_indices(annotation_labels)
                modes_group_diff[mode_name] = {g: 0.0 for g in annotation_group_indices.keys()}
            else:
                modes_group_diff[mode_name] = {}
            continue

        K_cur = int(Q.shape[1])

        # pick mapping direction
        if ref_K <= K_cur:
            key = f"{ref_mode}-{mode_name}"
        else:
            key = f"{mode_name}-{ref_mode}"

        pair_mapping = pair_mappings.get(key)
        if pair_mapping is None:
            raise KeyError(f"Missing pair mapping key: {key}")

        overall_diff, per_cell_diff, group_diff = compute_mode_total_and_annotation_group_diffs(
            mode_Q=Q,
            ref_Q=ref_Q,
            mode_name=mode_name,
            ref_mode=ref_mode,
            pair_mappings=pair_mappings,
            annotation_labels=annotation_labels,
            aggregation=aggregation,
        )

        modes_total_diff[mode_name] = float(overall_diff)
        modes_group_diff[mode_name] = group_diff

    return modes_total_diff, modes_group_diff
    

def build_mode_annotation_group_diff_df(
    mode_group_diff: Dict[str, Dict[str, float]],
    *,
    fill_value: float = 0.0,
    sort_index: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame of annotation-group membership differences per mode.

    Parameters
    ----------
    mode_group_diff : dict
        {mode_name: {group_label: fraction_diff}}
    fill_value : float, default 0.0
        Value to fill for missing group-mode combinations.
    sort_index : bool, default True
        If True, sort the DataFrame index.
    Returns
    -------
    pd.DataFrame
        Index: group labels
        Columns: mode names
        Values: fraction_diff
    """

    df = pd.DataFrame(mode_group_diff).fillna(fill_value).T
    if sort_index:
        df = df.sort_index()
    return df

def compute_avg_cluster_memberships(
    models: Sequence[str],
    res_models: Mapping[str, Any],
    selected_modes: Mapping[str, str],
    annotation_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    annot_col: str = "annot",
    cluster_prefix: str = "cluster_",
    sort_groups: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    For each model, compute average cluster membership per annotation group.

    This generalizes to any number of models.

    Expected res_models structure (as in your code):
        res_models[model].Q_by_mode[mode] -> array-like of shape (n_cells, K)

    Parameters
    ----------
    models
        List/sequence of model names to process.
    res_models
        Dict-like mapping model -> result object containing Q_by_mode.
    selected_modes
        Dict-like mapping model -> mode to use.
    annotation_labels
        Length n_cells labels for grouping.
    annot_col
        Name of the annotation column in output.
    cluster_prefix
        Prefix for cluster columns in output.
    sort_groups
        Passed to groupby(sort=...). Default False preserves observed order.
    verbose
        If True, print model/mode and shapes.

    Returns
    -------
    avg_cls_memberships
        Dict of model -> DataFrame with columns:
            [annot_col, f"{cluster_prefix}0", ..., f"{cluster_prefix}{K-1}"]
    """
    orig_order_uniq_labels = np.unique(annotation_labels).tolist()
    annotation_labels = pd.Series(annotation_labels, name=annot_col)

    avg_cls_memberships: Dict[str, pd.DataFrame] = {}

    for model in models:
        mode = selected_modes[model]
        res_model = res_models[model]

        Q = res_model.Q_by_mode[mode]

        # Accept numpy arrays or pandas DataFrames
        if isinstance(Q, pd.DataFrame):
            Q_arr = Q.to_numpy()
            K = Q.shape[1]
        else:
            Q_arr = np.asarray(Q)
            if Q_arr.ndim != 2:
                raise ValueError(f"Q for model '{model}', mode '{mode}' must be 2D.")
            K = Q_arr.shape[1]

        n_cells = Q_arr.shape[0]
        if len(annotation_labels) != n_cells:
            raise ValueError(
                f"annotation_labels length ({len(annotation_labels)}) "
                f"does not match Q rows ({n_cells}) for model '{model}', mode '{mode}'."
            )

        if verbose:
            print(f"Model: {model}, Mode: {mode}, Q shape: {Q_arr.shape}")

        cluster_cols = [f"{cluster_prefix}{k}" for k in range(K)]
        df_cells = pd.DataFrame(Q_arr, columns=cluster_cols)
        df_cells.insert(0, annot_col, annotation_labels.values)

        df_grouped = (
            df_cells
            .groupby(annot_col, sort=sort_groups, observed=False)
            .mean(numeric_only=True)
            .reset_index()
        )

        if not sort_groups:
            df_grouped.set_index(annot_col, inplace=True)
            df_grouped = df_grouped.loc[orig_order_uniq_labels].reset_index()
            df_grouped.reset_index(drop=True, inplace=True)

        avg_cls_memberships[model] = df_grouped

        if verbose:
            print(f"Average cluster memberships for model: {model}, mode: {mode}")

    return avg_cls_memberships


def compute_FSTruct(Q: np.ndarray, *, check_row_sums: bool = True, atol: float = 1e-8) -> Dict[str, float]:
    """
    Compute FST/FST_MAX.

    Parameters
    ----------
    Q
        2D array of shape (I, K), each row is an individual and
        each row is assumed to sum to 1 (cluster probabilities).
    check_row_sums
        If True, verify that each row sums to 1 within `atol`.
    atol
        Absolute tolerance for the row-sum check.

    Returns
    -------
    dict with keys:
        - "Fst"
        - "FstMax"
        - "ratio" = Fst / FstMax (0 if FstMax == 0)
    """
    Q = np.asarray(Q, dtype=float)
    I, K = Q.shape  # I: number of individuals, K: number of clusters

    if check_row_sums:
        row_sums = Q.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=atol):
            raise ValueError(
                f"Each row of Q must sum to 1 (within atol={atol}). "
                f"Got row sums in [{row_sums.min():.6g}, {row_sums.max():.6g}]."
            )
    
    # Column sums
    p = Q.sum(axis=0)          # shape (K,)
    sig1 = float(p.max())      # max column sum
    J = int(np.ceil(1.0 / sig1))
    sig1_floor = np.floor(sig1)
    sig1_frac = sig1 - sig1_floor

    if np.isclose(sig1, I):
        # degenerate case
        FstMax = 0.0
        Fst = 0.0
        ratio = 0.0
    else:
        # FstMax
        if sig1 <= 1.0:
            t = 1.0 - sig1 * (J - 1.0) * (2.0 - J * sig1)
            FstMax = ((I - 1.0) * t) / (I - t)
        else:
            num = (
                I * (I - 1.0)
                - sig1**2
                + sig1_floor
                - 2.0 * (I - 1.0) * sig1_frac
                + (2.0 * I - 1.0) * sig1_frac**2
            )
            den = (
                I * (I - 1.0)
                - sig1**2
                - sig1_floor
                + 2.0 * sig1
                - sig1_frac**2
            )
            FstMax = num / den

        # Fst:
        # sum(Q^2) / I - sum(colSums(Q / I)^2)
        # but colSums(Q / I) = p / I
        Q_sq_sum = float(np.sum(Q**2))
        p_sq_sum = float(np.sum(p**2))
        denom2 = 1.0 - p_sq_sum / (I**2)

        if np.isclose(denom2, 0.0):
            Fst = 0.0
        else:
            Fst = (Q_sq_sum / I - p_sq_sum / (I**2)) / denom2

        # ratio = Fst / FstMax
        if FstMax == 0.0 or np.isnan(FstMax):
            ratio = 0.0
        else:
            ratio = Fst / FstMax

    return {
        "Fst": Fst,
        "FstMax": FstMax,
        "ratio": ratio,
    }


def compute_FSTruct_over_annotation_groups(
    Q: np.ndarray,
    annotation_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    annot_col: str = "annot",
    sort_groups: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute FST/FST_MAX per annotation group.

    Parameters
    ----------
    Q
        2D array of shape (n_cells, K), each row is a cell and
        each row is assumed to sum to 1 (cluster probabilities).
    annotation_labels
        Length n_cells labels for grouping.
    annot_col
        Name of the annotation column in output.
    sort_groups
        Passed to groupby(sort=...). Default False preserves observed order.
    verbose
        If True, print shapes.

    Returns
    -------
    pd.DataFrame
        Columns: [annot_col, "Fst", "FstMax", "ratio"]
    """
    orig_order_uniq_labels = np.unique(annotation_labels).tolist()
    annotation_labels = pd.Series(annotation_labels, name=annot_col)

    if len(annotation_labels) != Q.shape[0]:
        raise ValueError(
            f"annotation_labels length ({len(annotation_labels)}) "
            f"does not match Q rows ({Q.shape[0]})."
        )

    if verbose:
        print(f"Q shape: {Q.shape}")

    df_cells = pd.DataFrame(Q)
    df_cells.insert(0, annot_col, annotation_labels.values)

    df_grouped = (
        df_cells
        .groupby(annot_col, sort=sort_groups, observed=False)
        .apply(lambda dfg: pd.Series(compute_FSTruct(dfg.iloc[:, 1:].to_numpy())))
        .reset_index()
    )

    if not sort_groups:
        df_grouped.set_index(annot_col, inplace=True)
        df_grouped = df_grouped.loc[orig_order_uniq_labels].reset_index()
        df_grouped.reset_index(drop=True, inplace=True)

    return df_grouped


def compute_FStruct_over_models_and_annotation_groups(
    comp_res: CompModelsResults,
    annotation_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    selected_modes: Sequence[str] = None,
    annot_col: str = "annot",
    sort_groups: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    For each model, compute FST/FST_MAX per annotation group.

    This generalizes to any number of models.

    Expected res_models structure (as in your code):
        res_models[model].Q_by_mode[mode] -> array-like of shape (n_cells, K)
    Parameters
    ----------
    comp_res
        Object containing results for multiple models.
    selected_modes
        Mapping from model name to selected mode name.
    annotation_labels
        Length n_cells labels for grouping.
    annot_col
        Name of the annotation column in output.
    sort_groups
        Passed to groupby(sort=...). Default False preserves observed order.
    verbose
        If True, print shapes.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys are model names, values are DataFrames with columns:
        [annot_col, "Fst", "FstMax", "ratio"]
    """
    orig_order_uniq_labels = np.unique(annotation_labels).tolist()
    annotation_labels = pd.Series(annotation_labels, name=annot_col)

    if selected_modes is None:
        selected_modes = comp_res.full_mode_names

    fst_results: Dict[str, pd.DataFrame] = {}

    for mode in selected_modes:

        Q = comp_res.Q_by_mode[mode]

        if len(annotation_labels) != Q.shape[0]:
            raise ValueError(
                f"annotation_labels length ({len(annotation_labels)}) "
                f"does not match Q rows ({Q.shape[0]}) for mode '{mode}'."
            )

        if verbose:
            print(f"Mode: {mode}, Q shape: {Q.shape}")

        df_fst = compute_FSTruct_over_annotation_groups(
            Q=Q,
            annotation_labels=annotation_labels,
            annot_col=annot_col,
            sort_groups=sort_groups,
            verbose=verbose,
        )

        fst_results[mode] = df_fst

    return fst_results


def bootstrap_FSTruct_over_annotation_groups(
    Q: np.ndarray,
    annotation_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    n_boot: int = 1000,
    annot_col: str = "annot",
    sort_groups: bool = False,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Bootstrap FST/FST_MAX per annotation group by resampling within each group.

    For each group g:
      - Resample rows of Q belonging to g with replacement (same group size).
      - Compute FSTruct on the bootstrap sample.
      - Repeat n_boot times.

    Parameters
    ----------
    Q
        2D array of shape (n_cells, K), each row is a cell and
        each row is assumed to sum to 1 (cluster probabilities).
    annotation_labels
        Length n_cells labels for grouping.
    n_boot
        Number of bootstrap replicates per group.
    annot_col
        Name of the annotation column in output.
    sort_groups
        If True, groups are processed in sorted order.
        If False, preserve first-occurrence order.
    random_state
        Optional seed for reproducibility.
    verbose
        If True, print basic diagnostics.

    Returns
    -------
    pd.DataFrame
        Columns: [annot_col, "boot", "Fst", "FstMax", "ratio"]

        One row per (group, bootstrap replicate).
    """
    # --- labels as Series ---
    annotation_labels = pd.Series(annotation_labels, name=annot_col)

    if len(annotation_labels) != Q.shape[0]:
        raise ValueError(
            f"annotation_labels length ({len(annotation_labels)}) "
            f"does not match Q rows ({Q.shape[0]})."
        )

    if verbose:
        print(f"Q shape: {Q.shape}")

    # --- group order ---
    uniq = pd.unique(annotation_labels)
    if sort_groups:
        groups = sorted(uniq)
    else:
        groups = list(uniq)

    rng = np.random.default_rng(random_state)
    records = []

    for g in groups:
        idx = np.flatnonzero(annotation_labels.values == g)
        n_g = idx.size
        if n_g < 2:
            if verbose:
                print(f"Skipping group {g!r}: size {n_g} < 2 (cannot bootstrap).")
            continue

        if verbose:
            print(f"Group {g!r}: n={n_g}, bootstraps={n_boot}")

        for b in range(n_boot):
            # sample indices within group WITH replacement
            boot_idx = rng.choice(idx, size=n_g, replace=True)
            Q_boot = Q[boot_idx, :]

            # assume compute_FSTruct returns (Fst, FstMax, ratio)
            res = compute_FSTruct(Q_boot)
            Fst, FstMax, ratio  = res["Fst"], res["FstMax"], res["ratio"]

            records.append((g, b, Fst, FstMax, ratio))

    df_boot = pd.DataFrame(records, columns=[annot_col, "boot", "Fst", "FstMax", "ratio"])
    return df_boot


def bootstrap_FStruct_over_models_and_annotation_groups(
    comp_res: CompModelsResults,
    annotation_labels: Sequence[str] | pd.Series | np.ndarray,
    *,
    n_boot: int = 1000,
    selected_modes: Sequence[str] = None,
    annot_col: str = "annot",
    sort_groups: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    For each model, compute boostrapped FST/FST_MAX per annotation group.

    This generalizes to any number of models.

    Expected res_models structure (as in your code):
        res_models[model].Q_by_mode[mode] -> array-like of shape (n_cells, K)
    Parameters
    ----------
    comp_res
        Object containing results for multiple models.
    selected_modes
        Mapping from model name to selected mode name.
    annotation_labels
        Length n_cells labels for grouping.
    annot_col
        Name of the annotation column in output.
    sort_groups
        Passed to groupby(sort=...). Default False preserves observed order.
    verbose
        If True, print shapes.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys are model names, values are DataFrames with columns:
        [annot_col, "Fst", "FstMax", "ratio"]
    """
    orig_order_uniq_labels = np.unique(annotation_labels).tolist()
    annotation_labels = pd.Series(annotation_labels, name=annot_col)

    if selected_modes is None:
        selected_modes = comp_res.full_mode_names

    fst_results: Dict[str, pd.DataFrame] = {}

    for mode in selected_modes:

        Q = comp_res.Q_by_mode[mode]

        if len(annotation_labels) != Q.shape[0]:
            raise ValueError(
                f"annotation_labels length ({len(annotation_labels)}) "
                f"does not match Q rows ({Q.shape[0]}) for mode '{mode}'."
            )

        if verbose:
            print(f"Mode: {mode}, Q shape: {Q.shape}")

        df_boot = bootstrap_FSTruct_over_annotation_groups(
            Q=Q,
            annotation_labels=annotation_labels,
            n_boot=n_boot,
            annot_col=annot_col,
            sort_groups=sort_groups,
            verbose=verbose,
        )

        fst_results[mode] = df_boot

    return fst_results


# ---------------------------------------------------------------------
# Mode summaries
# ---------------------------------------------------------------------

def build_mode_sizes_from_comp_res(
    comp_res,
    *,
    size_col: str = "Size",
    fill_missing: float = np.nan,
    sort_index: bool = True,
) -> pd.Series:
    """
    Build a Series of mode sizes indexed by FULL mode names.

    Parameters
    ----------
    comp_res
        Your comp result object.
    size_col
        Column containing mode sizes in each model's stats DataFrame.
    fill_missing
        Value used if a mode size is missing from stats.
    sort_index
        If True, sort the Series by full mode name.

    Returns
    -------
    pd.Series
        Index: full mode names (e.g., "{model}_{short_mode}")
        Values: mode sizes
    """
    modes_by_model = comp_res.modes_by_model
    stats_by_model = comp_res.mode_stats_by_model
    full_mode_names = list(comp_res.full_mode_names)

    # Build sizes using the authoritative short-name lists
    size_map: Dict[str, float] = {}

    for model, short_modes in modes_by_model.items():
        stats_df = stats_by_model.get(model)
        if stats_df is None or size_col not in stats_df.columns:
            continue

        # Series indexed by short mode names
        s_sizes = stats_df[size_col]

        for short in short_modes:
            full = f"{model}_{short}"
            val = s_sizes.get(short, fill_missing)
            size_map[full] = float(val) if val is not None else float(fill_missing)

    # Ensure we return sizes for exactly comp_res.full_mode_names
    # (fill any that didn't get populated)
    out = {full: size_map.get(full, float(fill_missing)) for full in full_mode_names}

    ser = pd.Series(out)

    return ser.sort_index() if sort_index else ser



# ---------------------------------------------------------------------
# Other helper functions 
# ---------------------------------------------------------------------

def _parse_peak(peak: str) -> Tuple[str, int, int]:
    """
    Parse peaks like "chr1:819912-823500".
    """
    chrom, rest = peak.split(":", 1)
    start_s, end_s = rest.split("-", 1)
    start = int(start_s.replace(",", ""))
    end = int(end_s.replace(",", ""))
    if end < start:
        start, end = end, start
    return chrom, start, end


def _normalize_chrom_lookup(chrom: str, gtf_chroms: Set[str]) -> Optional[str]:
    """
    Handle simple 'chr' vs no-'chr' mismatches.
    """
    if chrom in gtf_chroms:
        return chrom
    if chrom.startswith("chr"):
        alt = chrom[3:]
        if alt in gtf_chroms:
            return alt
    else:
        alt = "chr" + chrom
        if alt in gtf_chroms:
            return alt
    return None


def match_peaks_to_genes(
    peaks: Iterable[str],
    gtf_file: str,
    *,
    upstream: int = 5000,
    downstream: int = 0,
    feature_type: str = "gene",
    source: Optional[str] = "HAVANA",
    gene_type_allowlist: Optional[Set[str]] = None,
    intergenic_label: str = "intergenic",
    unassigned_label: str = "unassigned",
) -> List[str]:
    """
    Map each peak to overlapping gene(s).

    Efficiency features:
    - Streaming GTF load with early filters.
    - Minimal attribute parsing.
    - Per-chromosome sorted genes + sorted peaks.
    - Two-pointer scan: ~O(#genes + #peaks) per chromosome.

    Returns one label per input peak in the same order.
    """
    peaks_list = list(peaks)
    n = len(peaks_list)
    out: List[str] = [unassigned_label] * n

    # Load genes efficiently
    gtf = load_gene_intervals(
        gtf_file,
        upstream=upstream,
        downstream=downstream,
        feature_type=feature_type,
        source=source,
        gene_type_allowlist=gene_type_allowlist,
    )
    if not gtf:
        return [unassigned_label] * n

    gtf_chroms = set(gtf.keys())

    # Group peaks by chromosome with original index
    # Feature tuple: (start, end, idx)
    peaks_by_chrom: Dict[str, List[Tuple[int, int, int]]] = {}
    for i, p in enumerate(peaks_list):
        chrom, start, end = _parse_peak(p)
        peaks_by_chrom.setdefault(chrom, []).append((start, end, i))

    # Match per chrom
    for chrom_in, feats in peaks_by_chrom.items():
        chrom = _normalize_chrom_lookup(chrom_in, gtf_chroms)
        if chrom is None or chrom not in gtf:
            for _, _, idx in feats:
                out[idx] = unassigned_label
            continue

        genes = gtf[chrom]
        if not genes:
            for _, _, idx in feats:
                out[idx] = unassigned_label
            continue

        feats_sorted = sorted(feats, key=lambda x: (x[0], x[1]))

        g_i = 0
        g_len = len(genes)

        # Local variable bindings for tiny speed wins in tight loops
        genes_local = genes
        out_local = out

        for f_start, f_end, idx in feats_sorted:
            # Advance gene pointer while gene ends before feature starts
            while g_i < g_len and genes_local[g_i][1] < f_start:
                g_i += 1

            # Collect overlaps
            names: List[str] = []
            j = g_i
            while j < g_len and genes_local[j][0] <= f_end:
                if genes_local[j][1] >= f_start:
                    names.append(genes_local[j][2])
                j += 1

            if not names:
                out_local[idx] = intergenic_label
            else:
                uniq = sorted(set(names))
                out_local[idx] = uniq[0] if len(uniq) == 1 else ";".join(uniq)

    return out


def peaks_with_top_gene_overlap(
    df_informative_peaks_sorted,
    df_informative_genes_sorted,
    n_top=None,
    mapped_col="mapped_genes",
    sep=";",
    drop_labels=("intergenic", "unassigned", ""),
):
    """
    For informative peaks, identify those overlapping top informative genes.
    """
    # Subset
    if n_top is None:
        peaks_top = df_informative_peaks_sorted.copy()
        top_genes = set(df_informative_genes_sorted.index.astype(str))
    else:
        peaks_top = df_informative_peaks_sorted.head(n_top).copy()
        top_genes = set(df_informative_genes_sorted.head(n_top).index.astype(str))

    # Split + explode
    tmp = (
        peaks_top[[mapped_col]]
        .dropna()
        .astype({mapped_col: str})
        .assign(_gene=lambda d: d[mapped_col].str.split(sep))
        .explode("_gene")
    )
    tmp["_gene"] = tmp["_gene"].str.strip()

    # Clean + filter to top genes
    tmp = tmp[~tmp["_gene"].isin(drop_labels)]
    tmp = tmp[tmp["_gene"].isin(top_genes)]

    # Group back to peaks, keep original peak index
    overlap = (
        tmp.groupby(tmp.index)["_gene"]
        .apply(lambda s: sorted(set(s)))
        .to_frame("overlap_genes")
    )

    # Join to keep all peak metadata
    out = peaks_top.join(overlap, how="left")
    out["has_overlap"] = out["overlap_genes"].notna()

    # Optional: keep only overlapping peaks
    out_overlap_only = out[out["has_overlap"]].copy()

    return out_overlap_only, out


def make_col_unique(df, col, na_label: str = "NA"):
    """Make entries in `col` unique by appending suffixes to duplicates."""
    dup_counts = df.groupby(col, dropna=False).cumcount()

    # Turn the column into strings, with a clean label for NaNs
    base = df[col].fillna(na_label).astype(str)

    # First occurrence: just the name; later ones: name_1, name_2, ...
    unique_col = np.where(
        dup_counts == 0,
        base,
        base + "_" + dup_counts.astype(str),
    )
    return unique_col