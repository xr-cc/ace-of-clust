"""Feature metrics, mode subsets and cluster mappings between modes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union, Optional, Any, Iterable, Callable

import numpy as np
import pandas as pd
from clumppling.core import alignQ_wrtP
from clumppling.utils import cost_membership, get_uniq_lb_sep

from .io import ClumpplingResults, CompModelsResults

PathLike = Union[str, Path]

def subset_results(
    results: ClumpplingResults,
    modes_subset: Sequence[str],
) -> ClumpplingResults:
    """
    Restrict a ClumpplingResults to a subset of modes.

    Parameters
    ----------
    results
        Full results.
    modes_subset
        Mode names to keep.

    Returns
    -------
    ClumpplingResults
        New results with only the selected modes.
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
        mode_K=mode_K_map,
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
    Sort each feature's P values and compute log2 ratios between neighbors.

    Zeros in P are replaced by the smallest positive value so that log2 stays finite.

    Parameters
    ----------
    P
        P matrix (n_features, K).

    Returns
    -------
    LFC_sorted
        Log2 ratios between consecutive sorted values (n_features, K-1).
    idx_sorted
        Cluster indices sorting each row in ascending order (n_features, K).
    """
    K = P.shape[1]
    # P can contain exact zeros (e.g. from NMF); treat them as near-zero so log2 stays finite
    P = np.where(P == 0, P[P > 0].min(initial=1e-14), P)
    P_sorted = np.sort(P, axis=1)
    idx_sorted = np.argsort(P, axis=1)
    # log2 ratio between consecutive sorted entries
    LFC_sorted = np.log2(P_sorted[:, 1:]) - np.log2(P_sorted[:, : K - 1])
    return LFC_sorted, idx_sorted


def get_sepLFC(
    LFC_sorted: np.ndarray,
    idx_sorted: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """
    Compute sepLFC and sepCls from a sorted profile.

    Parameters
    ----------
    LFC_sorted
        Output of `compute_profile`.
    idx_sorted
        Output of `compute_profile`.

    Returns
    -------
    sepLFC
        Largest log2 ratio per feature.
    sepCls
        Per feature, (low clusters, high clusters) on either side of the largest gap.
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


def compute_weighted_Psum(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Sum P across clusters, weighted by cluster size.

    Parameters
    ----------
    P
        P matrix (n_features, K).
    Q
        Aligned Q matrix (n_cells, K); its column sums give the cluster sizes.

    Returns
    -------
    np.ndarray
        weighted_Psum per feature.
    """
    cls_wt = np.sum(Q, axis=0)
    cls_wt /= np.sum(cls_wt)
    weighted_Psum = P @ cls_wt
    return weighted_Psum


def compute_feature_metrics(
    P: np.ndarray,
    Q: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Compute weighted_Psum, sepLFC and sepCls for one mode.

    Parameters
    ----------
    P
        P matrix (n_features, K).
    Q
        Aligned Q matrix (n_cells, K).
    feature_names
        Names of the P rows.

    Returns
    -------
    pd.DataFrame
        Indexed by feature name, with columns weighted_Psum, sepLFC and sepCls.
    """
    if P.shape[0] != len(feature_names):
        raise ValueError(
            f"P.shape[0] ({P.shape[0]}) != len(feature_names) ({len(feature_names)})"
        )
    if np.any(P < 0):
        raise ValueError("P contains negative values; cannot compute log.")
    LFC_sorted, idx_sorted = compute_profile(P)
    sepLFC, sepCls = get_sepLFC(LFC_sorted, idx_sorted)
    weighted_Psum = compute_weighted_Psum(P, Q)

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


def compute_all_feature_metrics(
    results: ClumpplingResults,
    feature_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Compute feature metrics for all modes.

    Parameters
    ----------
    results
        Results loaded with P matrices.
    feature_names
        Names of the P rows.

    Returns
    -------
    dict
        {mode name: DataFrame from `compute_feature_metrics`}.
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
        df_by_mode[mode_name] = compute_feature_metrics(P, Q, feature_names)

    return df_by_mode


def select_top_features(
    df_by_mode: Mapping[str, pd.DataFrame],
    top_quantile: float = 0.1,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, set]:
    """
    Select the top features by weighted_Psum in each mode.

    Parameters
    ----------
    df_by_mode
        {mode name: feature metrics DataFrame}.
    top_quantile
        Fraction of features to keep in each mode.

    Returns
    -------
    selected_by_mode
        {mode name: selected features}, with columns suffixed by "_{mode name}".
    df_selected_all
        Inner join of all selections.
    overlap
        Features selected in every mode.
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


def analyze_sep_genes(
    df_mode: pd.DataFrame,
    sepH,
    sepL,
    gene_set: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find genes of a gene set whose sepCls matches a (sepH, sepL) split.

    Prints the number of matching genes (ordered and unordered match) and returns the top ones by sepLFC.

    Parameters
    ----------
    df_mode
        Feature metrics of one mode, indexed by gene name.
    sepH
        High cluster indices.
    sepL
        Low cluster indices.
    gene_set
        Gene names to rank.
    top_n
        Number of genes to return.

    Returns
    -------
    pd.DataFrame
        Top matching genes, with columns sepLFC, sepCls and weighted_Psum.
    """
    import itertools

    sepH_t = tuple(int(x) for x in sepH)
    sepL_t = tuple(int(x) for x in sepL)

    sepCls_ordered = (sepL_t, sepH_t)
    df_sep_ordered = df_mode[df_mode["sepCls"] == sepCls_ordered]

    sepCls_unordered = [
        (permL, permH)
        for permL in itertools.permutations(sepL_t)
        for permH in itertools.permutations(sepH_t)
    ]
    df_sep_unordered = df_mode[df_mode["sepCls"].apply(lambda x: x in sepCls_unordered)]
    print(f"sepCls ordered:   {sepCls_ordered}  |  n_genes (all): {len(df_sep_ordered)}")
    print(f"sepCls unordered: {len(sepCls_unordered)} permutations  |  n_genes (all): {len(df_sep_unordered)}")

    sepH_set, sepL_set = frozenset(sepH_t), frozenset(sepL_t)
    selected = [
        g for g in gene_set
        if g in df_mode.index
        and isinstance(df_mode.loc[g, "sepCls"], tuple)
        and len(df_mode.loc[g, "sepCls"]) == 2
        and {frozenset(df_mode.loc[g, "sepCls"][0]), frozenset(df_mode.loc[g, "sepCls"][1])}
            == {sepH_set, sepL_set}
    ]
    print(f"gene_set matches: {len(selected)} / {len(gene_set)}  |  returning top {min(top_n, len(selected))}")
    top_genes = sorted(selected, key=lambda g: df_mode.loc[g, "sepLFC"], reverse=True)[:top_n]
    return df_mode.loc[top_genes, ["sepLFC", "sepCls", "weighted_Psum"]]


# ---------------------------------------------------------------------
# Alignment graph & pairwise cluster mappings
# ---------------------------------------------------------------------

def _build_alignment_graph(
    alignment_acrossK: Mapping[str, Sequence[int]],
) -> Dict[str, List[str]]:
    """Build a directed graph with an edge B -> A for each key "A-B"."""
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
    """Find a path from src to dst by breadth-first search; None if there is none."""
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


def get_mode_pair_mappings(mode_names, all_modes_alignment, alignment_acrossK):
    """
    Map clusters between every pair of modes, through intermediate modes if needed.

    Parameters
    ----------
    mode_names
        Mode names.
    all_modes_alignment
        {mode name: column order after alignment}.
    alignment_acrossK
        {"A-B": mapping}, where mapping[k_B] = k_A in original cluster indices.

    Returns
    -------
    dict
        {"A-B": [(column in A, column in B), ...]}, in aligned column indices.
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
    Map an alternative Q matrix onto the clusters of a reference Q matrix.

    Parameters
    ----------
    ref_Q
        Reference Q matrix (n_cells, ref_K).
    alt_Q
        Alternative Q matrix (n_cells, alt_K), with alt_K >= ref_K.
    pair_mapping
        (reference column, alternative column) pairs.

    Returns
    -------
    alt_Q_mapped
        Alternative Q matrix summed into the reference clusters (n_cells, ref_K).
    diff_Q
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


def compute_profile_unnorm(P: np.ndarray):
    """
    Same as `compute_profile`, for any matrix of non-negative values (e.g. null mean-P vectors).

    Adds 1e-10 before log2 instead of replacing zeros.

    Parameters
    ----------
    P
        Values (n_rows, K).

    Returns
    -------
    LFC_sorted
        Log2 ratios between consecutive sorted values (n_rows, K-1).
    idx_sorted
        Column indices sorting each row in ascending order (n_rows, K).
    """
    P = np.asarray(P, dtype=float) + 1e-10
    idx_sorted = np.argsort(P, axis=1)
    P_sorted   = np.take_along_axis(P, idx_sorted, axis=1)
    LFC_sorted = np.log2(P_sorted[:, 1:]) - np.log2(P_sorted[:, :-1])
    return LFC_sorted, idx_sorted


__all__ = [
    "subset_results",
    "compute_profile",
    "get_sepLFC",
    "compute_weighted_Psum",
    "compute_feature_metrics",
    "compute_all_feature_metrics",
    "select_top_features",
    "analyze_sep_genes",
    "get_mode_pair_mappings",
    "map_alt_to_ref",
    "compute_profile_unnorm",
]
