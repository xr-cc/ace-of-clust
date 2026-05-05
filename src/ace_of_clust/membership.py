"""
membership.py

Functions for membership differences, annotation group summaries, and
FST-based population-structure statistics.

"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .io import ClumpplingResults, CompModelsResults
from .analysis import map_alt_to_ref


def compute_membership_diff(
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

def compute_per_cell_diff(
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

def get_diff_matrices(
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

    Logic mirrors plot_compmodels_diff_grid:
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
          `diff_by_model_mode[model_name][short_mode] = diff_Q`

        - `diff_Q` has shape (n_cells, K_eff), where K_eff is typically the smaller
          K between ref and current (as produced by map_alt_to_ref).
        - For the reference mode itself, diff_Q is a zero matrix with the same shape
          as ref_Q.

    Notes
    -----
    - This function *does not* call compute_membership_diff; it returns
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
            short_mode = _short_mode(model_name, full_name)

            # Reference mode: diff is 0 by definition.
            if full_name == ref_mode:
                diff_Q = np.zeros_like(ref_Q, dtype=float)
                diff_by_model_mode[model_name][short_mode] = diff_Q
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

            diff_by_model_mode[model_name][short_mode] = diff_Q

    return diff_by_model_mode



def get_pairwise_diff(mode_list_1, mode_list_2, pair_mappings, comp_results):
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
        As returned by get_mode_pair_mappings.
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


def get_pairwise_membership_diff(
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
        overall_diff = compute_membership_diff(diff_Q)
        overall_diff_dict[mode_pair] = overall_diff

    return overall_diff_dict


# ---------------------------------------------------------------------
# Annotation-group difference summaries
# ---------------------------------------------------------------------

def get_group_indices(
    annotation_labels: Sequence[Any],
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


def compute_group_sizes(
    annotation_labels: Sequence[Any],
) -> pd.Series:
    """
    Compute group sizes as a sorted Series.
    Parameters
    ----------
    annotation_labels
        Per-cell labels (e.g., cell types, domains, batches).
        Length must equal n_cells.
    Returns
    -------
    pd.Series
        index = group_label, value = size
    """
    group_indices = get_group_indices(annotation_labels)
    sizes = {g: len(idxs) for g, idxs in group_indices.items()}
    return pd.Series(sizes).sort_index()


def compute_group_diff(
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

def compute_mode_diffs(
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
    overall_diff = compute_membership_diff(diff_Q)
    per_cell_diff = compute_per_cell_diff(
        diff_Q,
        aggregation=aggregation,
    )
    if annotation_labels is None:
        group_diff = None
    else:
        annotation_group_indices = get_group_indices(annotation_labels)
        group_diff = compute_group_diff(
            per_cell_diff,
            annotation_group_indices,
        )
    return overall_diff, per_cell_diff, group_diff

def compute_all_mode_diffs(
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
                annotation_group_indices = get_group_indices(annotation_labels)
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

        overall_diff, per_cell_diff, group_diff = compute_mode_diffs(
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


def build_diff_df(
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

def compute_avg_memberships(
    models: Sequence[str],
    res_models: Mapping[str, Any],
    selected_modes: Mapping[str, str],
    annotation_labels: Sequence[str],
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


def compute_fstruct(Q: np.ndarray, *, check_row_sums: bool = True, atol: float = 1e-8) -> Dict[str, float]:
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


def compute_fstruct_by_group(
    Q: np.ndarray,
    annotation_labels: Sequence[str],
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
        .apply(lambda dfg: pd.Series(compute_fstruct(dfg.iloc[:, 1:].to_numpy())))
        .reset_index()
    )

    if not sort_groups:
        df_grouped.set_index(annot_col, inplace=True)
        df_grouped = df_grouped.loc[orig_order_uniq_labels].reset_index()
        df_grouped.reset_index(drop=True, inplace=True)

    return df_grouped


def compute_fstruct_by_model_group(
    comp_res: CompModelsResults,
    annotation_labels: Sequence[str],
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

        df_fst = compute_fstruct_by_group(
            Q=Q,
            annotation_labels=annotation_labels,
            annot_col=annot_col,
            sort_groups=sort_groups,
            verbose=verbose,
        )

        fst_results[mode] = df_fst

    return fst_results


def bootstrap_fstruct_by_group(
    Q: np.ndarray,
    annotation_labels: Sequence[str],
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

            # assume compute_fstruct returns (Fst, FstMax, ratio)
            res = compute_fstruct(Q_boot)
            Fst, FstMax, ratio  = res["Fst"], res["FstMax"], res["ratio"]

            records.append((g, b, Fst, FstMax, ratio))

    df_boot = pd.DataFrame(records, columns=[annot_col, "boot", "Fst", "FstMax", "ratio"])
    return df_boot


def bootstrap_fstruct_by_model_group(
    comp_res: CompModelsResults,
    annotation_labels: Sequence[str],
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

        df_boot = bootstrap_fstruct_by_group(
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

def get_mode_sizes(
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


__all__ = [
    "compute_membership_diff",
    "compute_per_cell_diff",
    "get_diff_matrices",
    "get_pairwise_diff",
    "get_pairwise_membership_diff",
    "get_group_indices",
    "compute_group_sizes",
    "compute_group_diff",
    "compute_mode_diffs",
    "compute_all_mode_diffs",
    "build_diff_df",
    "compute_avg_memberships",
    "compute_fstruct",
    "compute_fstruct_by_group",
    "compute_fstruct_by_model_group",
    "bootstrap_fstruct_by_group",
    "bootstrap_fstruct_by_model_group",
    "get_mode_sizes",
]
