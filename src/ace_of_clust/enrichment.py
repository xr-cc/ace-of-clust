"""
enrichment.py

Gene set enrichment statistical tests.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import compute_profile_unnorm, get_sepLFC


def resolve_gene_set(symbols_or_indices: list, used_genes: list) -> Tuple[List[str], np.ndarray]:
    """
    Resolve a gene set to the subset present in a reference gene list.

    Accepts either gene symbols (strings) or integer indices into
    ``used_genes``, and returns the matched symbols together with their
    positions in ``used_genes``.

    Parameters
    ----------
    symbols_or_indices : list of str or list of int
        Gene symbols to look up, or integer indices into ``used_genes``.
    used_genes : list of str
        Reference gene list (e.g. row names of the P matrix).

    Returns
    -------
    gene_set : list of str
        Gene symbols that are present in ``used_genes``.
    gene_set_indices : np.ndarray of int
        Positions of ``gene_set`` in ``used_genes``.
    """
    used_genes = list(used_genes)
    if len(symbols_or_indices) > 0 and isinstance(symbols_or_indices[0], (int, np.integer)):
        gene_set = [used_genes[i] for i in symbols_or_indices]
    else:
        used_set = set(used_genes)
        gene_set = [g for g in symbols_or_indices if g in used_set]
    gene_set_indices = np.array([used_genes.index(g) for g in gene_set])
    return gene_set, gene_set_indices


def compute_pairwise_lfc(x: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs log2 fold-change matrix from a cluster loading vector.

    Parameters
    ----------
    x : np.ndarray, shape (K,)
        Per-cluster values (e.g. mean gene-set loading per cluster).

    Returns
    -------
    lfc : np.ndarray, shape (K, K)
        Antisymmetric matrix where ``lfc[j, k] = log2(x[j]) - log2(x[k])``.
    """
    x = np.asarray(x, dtype=float)
    return np.subtract.outer(np.log2(x + 1e-12), np.log2(x + 1e-12))   # (K, K)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : np.ndarray
        Flat array of raw p-values.

    Returns
    -------
    qvals : np.ndarray
        BH-adjusted q-values, same shape as ``pvals``, clipped to [0, 1].
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    q = pvals[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def test_pairwise_lfc(obs: np.ndarray, nulls: np.ndarray, alternative="two-sided",
                                center="mean", compute_maxT=True, atol=1e-8):
    """
    Entrywise permutation test for an observed K×K pairwise LFC matrix.

    For each upper-triangle pair (i, j), the observed LFC is compared to
    its permutation null distribution to produce a z-score and empirical
    p-value.  P-values are corrected with Benjamini-Hochberg FDR.
    Optionally, familywise error rate control via the maxT method is applied.

    Parameters
    ----------
    obs : np.ndarray, shape (K, K)
        Observed antisymmetric LFC matrix.
    nulls : np.ndarray, shape (B, K, K)
        Null LFC matrices from ``B`` permutations.
    alternative : {"two-sided", "greater", "less"}
        Direction of the test. Default ``"two-sided"``.
    center : {"mean", "median"}
        How to centre the null distribution. Default ``"mean"``.
    compute_maxT : bool
        If True, also compute familywise maxT-adjusted p-values.
        Default True.
    atol : float
        Unused tolerance parameter (reserved). Default 1e-8.

    Returns
    -------
    result : dict
        Keys: ``p_mat``, ``q_mat``, ``z_mat``, ``effect_mat``,
        ``tested_pairs``, ``p_vec``, ``q_vec``, ``z_vec``, ``effect_vec``.
        If ``compute_maxT`` is True, also ``p_maxT_mat``, ``p_maxT_vec``.
    """
    obs   = np.asarray(obs,   dtype=float)
    nulls = np.asarray(nulls, dtype=float)
    B, K, _ = nulls.shape
    iu = np.triu_indices(K, k=1)
    obs_vec  = obs[iu]                    # (M,)
    null_vec = nulls[:, iu[0], iu[1]]    # (B, M)

    null_center = null_vec.mean(axis=0) if center == "mean" else np.median(null_vec, axis=0)
    null_std    = null_vec.std(axis=0, ddof=0)
    effect_vec  = obs_vec - null_center
    z_vec       = effect_vec / (null_std + 1e-12)

    if alternative == "greater":
        p_vec = (1 + (null_vec >= obs_vec).sum(axis=0)) / (B + 1)
    elif alternative == "less":
        p_vec = (1 + (null_vec <= obs_vec).sum(axis=0)) / (B + 1)
    else:  # two-sided
        p_vec = (1 + (np.abs(null_vec - null_center) >= np.abs(effect_vec)).sum(axis=0)) / (B + 1)
    q_vec = _fdr_bh(p_vec)

    def _fill(mat, upper, lower=None):
        mat[iu] = upper
        mat[(iu[1], iu[0])] = upper if lower is None else lower
    p_mat = np.full((K, K), np.nan);  _fill(p_mat, p_vec)
    q_mat = np.full((K, K), np.nan);  _fill(q_mat, q_vec)
    z_mat = np.zeros((K, K));         _fill(z_mat,  z_vec, -z_vec)
    eff_mat = np.zeros((K, K));       _fill(eff_mat, effect_vec, -effect_vec)
    np.fill_diagonal(p_mat, np.nan); np.fill_diagonal(q_mat, np.nan)

    res = dict(p_mat=p_mat, q_mat=q_mat, z_mat=z_mat, effect_mat=eff_mat,
               tested_pairs=iu, p_vec=p_vec, q_vec=q_vec, z_vec=z_vec,
               effect_vec=effect_vec)

    if compute_maxT:
        if alternative == "two-sided":
            obs_stat  = np.abs(effect_vec)
            null_stat = np.abs(null_vec - null_center)
        elif alternative == "greater":
            obs_stat, null_stat = effect_vec, null_vec - null_center
        else:
            obs_stat, null_stat = -effect_vec, null_center - null_vec
        p_maxT_vec = (1 + (null_stat.max(axis=1)[:, None] >= obs_stat).sum(axis=0)) / (B + 1)
        p_maxT_mat = np.full((K, K), np.nan); _fill(p_maxT_mat, p_maxT_vec)
        np.fill_diagonal(p_maxT_mat, np.nan)
        res.update(p_maxT_vec=p_maxT_vec, p_maxT_mat=p_maxT_mat)
    return res


def sample_null_P(P: np.ndarray, gs_idx: np.ndarray, n_perm: int, rng_seed: int = 1) -> np.ndarray:
    """
    Sample a permutation null distribution of mean gene-set loadings.

    For each permutation, a random gene set of the same size as ``gs_idx``
    is drawn without replacement from all genes, and the mean loading vector
    across clusters is recorded.

    Parameters
    ----------
    P : np.ndarray, shape (G, K)
        Full gene-by-cluster loading matrix.
    gs_idx : np.ndarray of int
        Indices of the gene set within ``P``.
    n_perm : int
        Number of permutations.
    rng_seed : int, optional
        Random seed for reproducibility. Default 1.

    Returns
    -------
    null_mean_P : np.ndarray, shape (n_perm, K)
        Mean loading vector for each random gene set.
    """
    rng = np.random.default_rng(rng_seed)
    null_mean_P = np.zeros((n_perm, P.shape[1]), dtype=np.float32)
    for i in range(n_perm):
        ridx = rng.choice(P.shape[0], len(gs_idx), replace=False)
        null_mean_P[i] = P[ridx].mean(axis=0)
    return null_mean_P


def test_P_enrichment(P: np.ndarray, gs_idx: np.ndarray, null_mean_P: np.ndarray):
    """
    Test whether a gene set has elevated mean loading in each cluster.

    Compares the observed gene-set mean loading per cluster to the
    permutation null produced by ``sample_null_P``.

    Parameters
    ----------
    P : np.ndarray, shape (G, K)
        Full gene-by-cluster loading matrix.
    gs_idx : np.ndarray of int
        Indices of the gene set within ``P``.
    null_mean_P : np.ndarray, shape (n_perm, K)
        Null mean loading vectors from ``sample_null_P``.

    Returns
    -------
    result : dict with keys
        ``gs_mean_P`` : np.ndarray, shape (K,) — observed gene-set mean loading.
        ``z``         : np.ndarray, shape (K,) — z-score vs. null.
        ``p_emp``     : np.ndarray, shape (K,) — one-sided empirical p-value.
    """
    gs_mean_P = P[gs_idx].mean(axis=0)                    # (K,)
    null_mu  = null_mean_P.mean(axis=0)
    null_std = null_mean_P.std(axis=0)
    z     = (gs_mean_P - null_mu) / (null_std + 1e-12)
    p_emp = (1 + (null_mean_P >= gs_mean_P).sum(axis=0)) / (len(null_mean_P) + 1)
    return {"gs_mean_P": gs_mean_P, "z": z, "p_emp": p_emp}


def test_LFC_enrichment(P: np.ndarray, gs_idx: np.ndarray, null_mean_P: np.ndarray):
    """
    Test all pairwise log2 fold-changes of gene-set mean loadings.

    Computes the observed all-pairs LFC matrix from the gene-set mean
    loading vector and tests each upper-triangle entry against its
    permutation null via ``test_pairwise_lfc``.

    Parameters
    ----------
    P : np.ndarray, shape (G, K)
        Full gene-by-cluster loading matrix.
    gs_idx : np.ndarray of int
        Indices of the gene set within ``P``.
    null_mean_P : np.ndarray, shape (n_perm, K)
        Null mean loading vectors from ``sample_null_P``.

    Returns
    -------
    result : dict with keys
        ``obs_lfc``  : np.ndarray, shape (K, K) — observed pairwise LFC matrix.
        ``null_lfc`` : np.ndarray, shape (n_perm, K, K) — null LFC matrices.
        ``test``     : dict — output of ``test_pairwise_lfc``.
        ``df``       : pd.DataFrame — per-pair statistics sorted by q-value,
                       with columns ``i``, ``j``, ``obs_lfc``, ``z``, ``p``,
                       ``q``, and optionally ``p_maxT``.
    """
    obs_lfc  = compute_pairwise_lfc(P[gs_idx].mean(axis=0))
    null_lfc = np.array([compute_pairwise_lfc(null_mean_P[i])
                         for i in range(len(null_mean_P))], dtype=np.float32)
    test = test_pairwise_lfc(obs_lfc, null_lfc,
                                       alternative="two-sided", compute_maxT=True)
    iu = test["tested_pairs"]
    df = pd.DataFrame({
        "i": iu[0], "j": iu[1], "obs_lfc": obs_lfc[iu],
        "z": test["z_vec"], "p": test["p_vec"], "q": test["q_vec"],
    })
    if "p_maxT_vec" in test:
        df["p_maxT"] = test["p_maxT_vec"]
    return {"obs_lfc": obs_lfc, "null_lfc": null_lfc,
            "test": test, "df": df.sort_values("q").reset_index(drop=True)}


def test_sepLFC_enrichment(P: np.ndarray, gs_idx: np.ndarray, null_mean_P: np.ndarray):
    """
    Test the separating LFC of a gene set against permutation null distributions.

    The sepLFC is the largest log2 gap between consecutive sorted cluster mean
    loadings.  Two null comparisons are made: one where each random gene set
    finds its own best sepLFC (``null_sepLFC``), and one where the null sets
    are evaluated at the fixed cluster bipartition determined by the observed
    gene set (``null_lfc_at_sep``).

    Parameters
    ----------
    P : np.ndarray, shape (G, K)
        Full gene-by-cluster loading matrix.
    gs_idx : np.ndarray of int
        Indices of the gene set within ``P``.
    null_mean_P : np.ndarray, shape (n_perm, K)
        Null mean loading vectors from ``sample_null_P``.

    Returns
    -------
    result : dict with keys
        ``gs_sepLFC``      : float — **arithmetic-mean sepLFC**.
                             ``log2(min_h mean_g(P[:,h]) / max_l mean_g(P[:,l]))``.
                             Averages P across genes first, then takes the log
                             ratio.  Sensitive to absolute-scale genes; robust
                             to per-gene outliers.
        ``gs_geomean_LFC`` : float — **geometric-mean sepLFC**.
                             ``mean_g(log2(min_h P[g,h] / max_l P[g,l]))``.
                             Takes each gene's log ratio first, then averages.
                             Gives equal weight to every gene regardless of
                             absolute magnitude; captures consistent directional
                             enrichment across the gene set.
        ``sepL``           : list of int — cluster indices in the low group.
        ``sepH``           : list of int — cluster indices in the high group.
        ``idx_sorted_gs``  : list of int — cluster order sorted ascending by
                             gene-set mean loading.
        ``null_sepLFC``    : np.ndarray, shape (n_perm,) — best sepLFC of each
                             random gene set.
        ``null_lfc_at_sep``: np.ndarray, shape (n_perm,) — LFC of each random
                             gene set evaluated at the observed bipartition.
        ``p_vs_null_sep``  : float — empirical p-value vs. ``null_sepLFC``.
        ``p_vs_null_fixed``: float — empirical p-value vs. ``null_lfc_at_sep``.
    """
    P_gs = P[gs_idx]   # (n_gs, K)
    gs_mean_P = P_gs.mean(axis=0)
    LFC_s, idx_s = compute_profile_unnorm(gs_mean_P[np.newaxis, :])
    gs_sepLFC_arr, gs_sepCls = get_sepLFC(LFC_s, idx_s)
    gs_sepLFC    = float(gs_sepLFC_arr[0])
    sepL         = [int(i) for i in np.atleast_1d(gs_sepCls[0][0])]
    sepH         = [int(i) for i in np.atleast_1d(gs_sepCls[0][1])]
    idx_sorted_gs = idx_s[0].tolist()

    # Geometric-mean sepLFC: mean of per-gene log-ratios at the observed bipartition.
    # Uses conservative reduction (min over sepH, max over sepL) per gene — same
    # convention as compute_gene_lfc(kind="extreme").
    _P_high_g = P_gs[:, sepH].min(axis=1)   # (n_gs,) weakest high cluster per gene
    _P_low_g  = P_gs[:, sepL].max(axis=1)   # (n_gs,) strongest low cluster per gene
    if np.any(_P_high_g <= 0):
        raise ValueError(
            f"P[gs_idx] has zero/negative min over sepH for gene indices "
            f"{np.where(_P_high_g <= 0)[0].tolist()}; cannot compute gs_geomean_LFC."
        )
    if np.any(_P_low_g <= 0):
        raise ValueError(
            f"P[gs_idx] has zero/negative max over sepL for gene indices "
            f"{np.where(_P_low_g <= 0)[0].tolist()}; cannot compute gs_geomean_LFC."
        )
    gs_geomean_LFC = float(np.mean(np.log2(_P_high_g) - np.log2(_P_low_g)))

    LFC_n, idx_n = compute_profile_unnorm(null_mean_P)
    null_sepLFC, _ = get_sepLFC(LFC_n, idx_n)
    null_sepLFC    = np.array(null_sepLFC, dtype=float)

    import warnings
    _sepH_min = null_mean_P[:, sepH].min(axis=1)
    _sepL_max = null_mean_P[:, sepL].max(axis=1)
    _zero_mask = (_sepH_min <= 0) | (_sepL_max <= 0)
    if _zero_mask.any():
        warnings.warn(
            f"null_mean_P has zero/negative value(s) for "
            f"{int(_zero_mask.sum())} null sample(s) at indices "
            f"{np.where(_zero_mask)[0].tolist()}; log2 will be -inf.",
            RuntimeWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore"):
        null_lfc_at_sep = np.log2(_sepH_min) - np.log2(_sepL_max)
    n = len(null_sepLFC)
    p_vs_null_sep   = (1 + (null_sepLFC     >= gs_sepLFC).sum()) / (n + 1)
    p_vs_null_fixed = (1 + (null_lfc_at_sep >= gs_sepLFC).sum()) / (n + 1)

    return {
        "gs_sepLFC": gs_sepLFC, "gs_geomean_LFC": gs_geomean_LFC,
        "sepL": sepL, "sepH": sepH,
        "idx_sorted_gs": idx_sorted_gs,
        "null_sepLFC": null_sepLFC, "null_lfc_at_sep": null_lfc_at_sep,
        "p_vs_null_sep": p_vs_null_sep, "p_vs_null_fixed": p_vs_null_fixed,
    }


def compute_gene_lfc(P_gs: np.ndarray, gene_set: List[str], sepL: List[int], sepH: List[int], kind="extreme"):
    """
    Compute each gene's individual LFC between the high and low cluster groups.

    Parameters
    ----------
    P_gs : np.ndarray, shape (n_gs, K)
        Gene-set rows of the aligned P matrix.
    gene_set : list of str
        Gene names corresponding to rows of ``P_gs``.
    sepL : list of int
        Cluster indices in the low group (from ``test_sepLFC_enrichment``).
    sepH : list of int
        Cluster indices in the high group.
    kind : {"extreme", "mean"}
        How to summarize each group's loading.  ``"extreme"`` uses the
        most-conservative value (min of sepH, max of sepL); ``"mean"`` uses
        the group mean. Default ``"extreme"``.

    Returns
    -------
    df : pd.DataFrame
        Columns ``gene`` and ``LFC``, sorted descending by LFC.
    """
    if kind == "extreme":
        P_high = P_gs[:, sepH].min(axis=1)   # conservative: min of high group
        P_low  = P_gs[:, sepL].max(axis=1)   # conservative: max of low group
    elif kind == "mean":
        P_high = P_gs[:, sepH].mean(axis=1)
        P_low  = P_gs[:, sepL].mean(axis=1)
    else:
        raise ValueError(f"Invalid kind {kind!r}, expected 'extreme' or 'mean'.")
    if np.any(P_high <= 0):
        bad = np.where(P_high <= 0)[0].tolist()
        raise ValueError(
            f"P_high contains zero or negative value(s) for gene indices {bad}; "
            f"cannot compute log2 LFC."
        )
    if np.any(P_low <= 0):
        bad = np.where(P_low <= 0)[0].tolist()
        raise ValueError(
            f"P_low contains zero or negative value(s) for gene indices {bad}; "
            f"cannot compute log2 LFC."
        )
    lfc = np.log2(P_high) - np.log2(P_low)
    return (pd.DataFrame({"gene": gene_set, "LFC": lfc})
            .sort_values("LFC", ascending=False)
            .reset_index(drop=True))


def run_gs_enrichment(results, gene_set_indices, modes, n_perm, rng_seed):
    """
    Run P, LFC, and sepLFC enrichment for every requested mode.

    Parameters
    ----------
    results : ClumpplingResults
        Clumppling alignment results containing ``P_aligned_by_mode``.
    gene_set_indices : np.ndarray of int
        Gene indices for the gene set (output of ``resolve_gene_set``).
    modes : list of str
        Mode names to evaluate (subset of ``results.modes``).
    n_perm : int
        Number of permutations passed to ``sample_null_P``.
    rng_seed : int
        Random seed passed to ``sample_null_P``.

    Returns
    -------
    res_by_mode : dict
        Mapping ``mode -> {"p_res": ..., "lfc_res": ..., "sep_res": ...}``
        where each value is the output of the corresponding
        ``compute_*_enrichment`` function.
    """
    res_by_mode = {}
    for mode in modes:
        P           = results.P_aligned_by_mode[mode]
        null_mean_P = sample_null_P(P, gene_set_indices, n_perm, rng_seed)
        res_by_mode[mode] = {
            "p_res":   test_P_enrichment(P, gene_set_indices, null_mean_P),
            "lfc_res": test_LFC_enrichment(P, gene_set_indices, null_mean_P),
            "sep_res": test_sepLFC_enrichment(P, gene_set_indices, null_mean_P),
        }
    return res_by_mode


__all__ = [
    "resolve_gene_set",
    "compute_pairwise_lfc",
    "test_pairwise_lfc",
    "sample_null_P",
    "test_P_enrichment",
    "test_LFC_enrichment",
    "test_sepLFC_enrichment",
    "compute_gene_lfc",
    "run_gs_enrichment",
]
