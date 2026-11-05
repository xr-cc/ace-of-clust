"""Membership differences between modes, summaries by annotation group, and FSTruct statistics."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .io import ClumpplingResults, CompModelsResults
from .analysis import map_alt_to_ref


def compute_membership_diff(
    diff_Q: np.ndarray,
) -> float:
    """Overall membership difference: diff_Q summed over clusters, averaged over cells."""
    n_cells = diff_Q.shape[0]
    overall_diff = float(diff_Q.sum() / n_cells)
    return overall_diff

def compute_per_cell_diff(
    diff_Q: np.ndarray,
    *,
    aggregation: str = "sum",
) -> np.ndarray:
    """
    Membership difference per cell.

    Parameters
    ----------
    diff_Q
        Absolute membership differences (n_cells, K).
    aggregation
        How to combine clusters: "sum" (halved, so in [0, 1]), "max" or "mean".

    Returns
    -------
    np.ndarray
        Difference per cell (n_cells,).
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
    Compute membership differences of every mode from a reference mode.

    The mode with the smaller K is mapped onto the other with `map_alt_to_ref`.

    Parameters
    ----------
    comp_res
        compModels results.
    pair_mappings
        Output of `get_mode_pair_mappings`.
    ref_mode
        Full mode name of the reference mode.
    models
        Models to include; None uses all.
    strict_pair_mapping
        Raise an error if a pair mapping is missing; otherwise use zero differences.

    Returns
    -------
    dict
        {model: {mode name: diff_Q (n_cells, min(ref_K, K))}}; zeros for the reference mode.
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
        """Add the model prefix to a mode name."""
        return mode_entry if str(mode_entry).startswith(model + "_") else f"{model}_{mode_entry}"

    def _short_mode(model: str, full_name: str) -> str:
        """Remove the model prefix from a full mode name."""
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
    Compute diff_Q for every pair of modes from two lists.

    The mode with the smaller K is mapped onto the other.

    Parameters
    ----------
    mode_list_1
        First mode names.
    mode_list_2
        Second mode names.
    pair_mappings
        Output of `get_mode_pair_mappings`.
    comp_results
        Results with Q_by_mode.

    Returns
    -------
    dict
        {(mode 1, mode 2): diff_Q}.
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
    """Overall membership difference of each diff_Q, as {(mode 1, mode 2): difference}."""
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
    """Cell indices of each annotation group, as {group: indices}."""
    s = pd.Series(annotation_labels, dtype="object")
    idx_dict = s.groupby(s).indices  # label -> Int64Index
    return {str(k): np.asarray(v, dtype=int) for k, v in idx_dict.items()}


def compute_group_sizes(
    annotation_labels: Sequence[Any],
) -> pd.Series:
    """Number of cells per annotation group, as a sorted Series."""
    group_indices = get_group_indices(annotation_labels)
    sizes = {g: len(idxs) for g, idxs in group_indices.items()}
    return pd.Series(sizes).sort_index()


def compute_group_diff(
    per_cell_diff: np.ndarray,
    group_indices: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    """Mean per-cell difference in each annotation group, as {group: difference}."""
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
    mode_name: str,
    ref_mode: str,
    pair_mappings: Mapping[str, Sequence[Tuple[int, int]]],
    annotation_labels: Optional[Sequence[Any]] = None,
    aggregation: str = "sum",
) -> Tuple[float, np.ndarray, Optional[Dict[str, float]]]:
    """
    Compute the membership difference of one mode from a reference mode.

    Parameters
    ----------
    mode_Q
        Q matrix of the mode (n_cells, K).
    ref_Q
        Q matrix of the reference mode (n_cells, ref_K).
    mode_name
        Name of the mode.
    ref_mode
        Name of the reference mode.
    pair_mappings
        Output of `get_mode_pair_mappings`.
    annotation_labels
        Annotation label per cell; None skips the group differences.
    aggregation
        How to combine clusters per cell (see `compute_per_cell_diff`).

    Returns
    -------
    tuple
        Overall difference, difference per cell, and difference per annotation group (or None).
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
    comp_results: CompModelsResults,
    ref_mode: str,
    pair_mappings: Mapping[str, Sequence[Tuple[int, int]]],
    annotation_labels: Optional[Sequence[Any]] = None,
    aggregation: str = "sum"
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute the membership difference of every mode from a reference mode.

    Parameters
    ----------
    comp_results
        compModels results.
    ref_mode
        Full mode name of the reference mode.
    pair_mappings
        Output of `get_mode_pair_mappings`.
    annotation_labels
        Annotation label per cell; None skips the group differences.
    aggregation
        How to combine clusters per cell (see `compute_per_cell_diff`).

    Returns
    -------
    modes_total_diff
        {full mode name: overall difference}.
    modes_group_diff
        {full mode name: {group: difference}}.
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
    Table of differences per mode (rows) and annotation group (columns).

    Parameters
    ----------
    mode_group_diff
        {mode name: {group: difference}}.
    fill_value
        Value for missing entries.
    sort_index
        Sort the rows by mode name.

    Returns
    -------
    pd.DataFrame
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
    Mean cluster membership per annotation group, for one mode of each model.

    Parameters
    ----------
    models
        Model names.
    res_models
        {model: results with Q_by_mode}.
    selected_modes
        {model: mode name}.
    annotation_labels
        Annotation label per cell.
    annot_col
        Name of the annotation column in the output.
    cluster_prefix
        Prefix of the cluster columns in the output.
    sort_groups
        Sort the groups; otherwise keep their order of appearance.
    verbose
        Print the mode and shape of each model.

    Returns
    -------
    dict
        {model: DataFrame with annot_col and one column per cluster ({cluster_prefix}0, {cluster_prefix}1, ...)}.
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
    Compute FSTruct (Fst / FstMax) of a Q matrix.

    Parameters
    ----------
    Q
        Q matrix (n_cells, K) with rows summing to 1.
    check_row_sums
        Check that rows sum to 1.
    atol
        Tolerance of the row-sum check.

    Returns
    -------
    dict
        Fst, FstMax and ratio (Fst / FstMax, or 0 if FstMax is 0).
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
    Compute FSTruct per annotation group.

    Parameters
    ----------
    Q
        Q matrix (n_cells, K).
    annotation_labels
        Annotation label per cell.
    annot_col
        Name of the annotation column in the output.
    sort_groups
        Sort the groups; otherwise keep their order of appearance.
    verbose
        Print shapes.

    Returns
    -------
    pd.DataFrame
        Columns annot_col, Fst, FstMax and ratio.
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
    Compute FSTruct per annotation group for several modes.

    Parameters
    ----------
    comp_res
        compModels results.
    annotation_labels
        Annotation label per cell.
    selected_modes
        Full mode names; None uses all.
    annot_col
        Name of the annotation column in the output.
    sort_groups
        Sort the groups; otherwise keep their order of appearance.
    verbose
        Print shapes.

    Returns
    -------
    dict
        {full mode name: output of `compute_fstruct_by_group`}.
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
    Bootstrap FSTruct per annotation group, resampling cells within each group.

    Parameters
    ----------
    Q
        Q matrix (n_cells, K).
    annotation_labels
        Annotation label per cell.
    n_boot
        Number of bootstrap samples per group.
    annot_col
        Name of the annotation column in the output.
    sort_groups
        Sort the groups; otherwise keep their order of appearance.
    random_state
        Random seed.
    verbose
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns annot_col, boot, Fst, FstMax and ratio; one row per group and bootstrap sample.
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
    Bootstrap FSTruct per annotation group for several modes.

    Parameters
    ----------
    comp_res
        compModels results.
    annotation_labels
        Annotation label per cell.
    n_boot
        Number of bootstrap samples per group.
    selected_modes
        Full mode names; None uses all.
    annot_col
        Name of the annotation column in the output.
    sort_groups
        Sort the groups; otherwise keep their order of appearance.
    verbose
        Print shapes.

    Returns
    -------
    dict
        {full mode name: output of `bootstrap_fstruct_by_group`}.
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
    Mode sizes (number of runs) indexed by full mode name.

    Parameters
    ----------
    comp_res
        compModels results.
    size_col
        Column of the mode_stats tables with the sizes.
    fill_missing
        Value for modes without a size.
    sort_index
        Sort by full mode name.

    Returns
    -------
    pd.Series
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
