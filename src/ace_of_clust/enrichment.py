"""Gene set enrichment tests on P matrices."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis import compute_profile_unnorm, get_sepLFC


def resolve_gene_set(symbols_or_indices: list, used_genes: list) -> Tuple[List[str], np.ndarray]:
    """
    Keep the genes of a gene set that are in a reference gene list.

    Parameters
    ----------
    symbols_or_indices
        Gene names, or indices into used_genes.
    used_genes
        Reference gene names (the P rows).

    Returns
    -------
    gene_set
        Gene names found in used_genes.
    gene_set_indices
        Their indices in used_genes.
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
    Compute log2 fold changes between all pairs of clusters.

    Parameters
    ----------
    x
        Value per cluster (K,), e.g. gene set mean P.

    Returns
    -------
    np.ndarray
        lfc[j, k] = log2(x[j]) - log2(x[k]), shape (K, K).
    """
    x = np.asarray(x, dtype=float)
    return np.subtract.outer(np.log2(x + 1e-12), np.log2(x + 1e-12))   # (K, K)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, clipped to [0, 1]."""
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
                                center="mean", compute_maxT=True):
    """
    Test each pairwise LFC against its null distribution.

    Each upper-triangle pair gets a z-score and an empirical p-value, adjusted with Benjamini-Hochberg and, optionally, maxT.

    Parameters
    ----------
    obs
        Observed LFC matrix (K, K).
    nulls
        Null LFC matrices (B, K, K).
    alternative
        "two-sided", "greater" or "less".
    center
        Center of the null: "mean" or "median".
    compute_maxT
        Also compute maxT-adjusted p-values.

    Returns
    -------
    dict
        p_mat, q_mat, z_mat, effect_mat, tested_pairs, p_vec, q_vec, z_vec and effect_vec; with compute_maxT, also p_maxT_mat and p_maxT_vec.
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
    Sample random gene sets of the same size and record their mean P.

    Parameters
    ----------
    P
        P matrix (n_genes, K).
    gs_idx
        Gene set indices in P.
    n_perm
        Number of random gene sets.
    rng_seed
        Random seed.

    Returns
    -------
    np.ndarray
        Mean P of each random gene set (n_perm, K).
    """
    rng = np.random.default_rng(rng_seed)
    null_mean_P = np.zeros((n_perm, P.shape[1]), dtype=np.float32)
    for i in range(n_perm):
        ridx = rng.choice(P.shape[0], len(gs_idx), replace=False)
        null_mean_P[i] = P[ridx].mean(axis=0)
    return null_mean_P


def test_P_enrichment(P: np.ndarray, gs_idx: np.ndarray, null_mean_P: np.ndarray):
    """
    Test whether the gene set mean P is higher than the null in each cluster.

    Parameters
    ----------
    P
        P matrix (n_genes, K).
    gs_idx
        Gene set indices in P.
    null_mean_P
        Output of `sample_null_P`.

    Returns
    -------
    dict
        gs_mean_P (gene set mean P), z (z-score) and p_emp (one-sided empirical p-value), each of shape (K,).
    """
    gs_mean_P = P[gs_idx].mean(axis=0)                    # (K,)
    null_mu  = null_mean_P.mean(axis=0)
    null_std = null_mean_P.std(axis=0)
    z     = (gs_mean_P - null_mu) / (null_std + 1e-12)
    p_emp = (1 + (null_mean_P >= gs_mean_P).sum(axis=0)) / (len(null_mean_P) + 1)
    return {"gs_mean_P": gs_mean_P, "z": z, "p_emp": p_emp}


def test_LFC_enrichment(P: np.ndarray, gs_idx: np.ndarray, null_mean_P: np.ndarray):
    """
    Test the pairwise LFCs of the gene set mean P against the null.

    Parameters
    ----------
    P
        P matrix (n_genes, K).
    gs_idx
        Gene set indices in P.
    null_mean_P
        Output of `sample_null_P`.

    Returns
    -------
    dict
        obs_lfc (K, K), null_lfc (n_perm, K, K), test (output of `test_pairwise_lfc`) and df (one row per pair, sorted by q-value).
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
    Test the gene set sepLFC against two nulls.

    sepLFC is the largest log2 gap between the sorted cluster means of P. The first null uses the best sepLFC of each random gene set; the second uses each random gene set's LFC at the observed split.

    Parameters
    ----------
    P
        P matrix (n_genes, K).
    gs_idx
        Gene set indices in P.
    null_mean_P
        Output of `sample_null_P`.

    Returns
    -------
    dict
        gs_sepLFC
            log2 of min over sepH / max over sepL of the gene set mean P.
        gs_geomean_LFC
            Mean over genes of log2(min over sepH / max over sepL of P).
        sepL, sepH
            Low and high cluster indices.
        idx_sorted_gs
            Clusters sorted by gene set mean P (ascending).
        null_sepLFC
            Best sepLFC of each random gene set (n_perm,).
        null_lfc_at_sep
            LFC of each random gene set at the observed split (n_perm,).
        p_vs_null_sep
            Empirical p-value against null_sepLFC.
        p_vs_null_fixed
            Empirical p-value against null_lfc_at_sep.
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
    Compute each gene's LFC between the high and low clusters.

    Parameters
    ----------
    P_gs
        Gene set rows of the P matrix (n_gs, K).
    gene_set
        Gene names of the P_gs rows.
    sepL
        Low cluster indices.
    sepH
        High cluster indices.
    kind
        "extreme" (min over sepH vs. max over sepL) or "mean" (group means).

    Returns
    -------
    pd.DataFrame
        Columns gene and LFC, sorted by LFC (descending).
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
    Run the P, LFC and sepLFC enrichment tests for several modes.

    Parameters
    ----------
    results
        Results loaded with P matrices.
    gene_set_indices
        Gene set indices in P (from `resolve_gene_set`).
    modes
        Mode names.
    n_perm
        Number of random gene sets.
    rng_seed
        Random seed.

    Returns
    -------
    dict
        {mode name: {"p_res", "lfc_res", "sep_res"}}, the outputs of `test_P_enrichment`, `test_LFC_enrichment` and `test_sepLFC_enrichment`.
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
