"""
plot_enrichment.py

Gene set enrichment visualizations.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns

from .io import ClumpplingResults

# ---------------------------------------------------------------------
# Gene set analysis plots
# ---------------------------------------------------------------------

def plot_pairwise_heatmap(value_mat: np.ndarray, sig_mat=None, labels=None, title=None,
                          upper_only=True, cmap="coolwarm", center_zero=True,
                          sig_level=0.05, figsize=(7, 6), dpi=150, ax=None):
    """Heatmap of a KxK matrix with optional significance overlay."""
    mat = np.asarray(value_mat, dtype=float).copy()
    K   = mat.shape[0]
    if upper_only:
        mat[np.tril_indices(K, k=0)] = np.nan
    vmax = np.nanmax(np.abs(mat)) if center_zero else np.nanmax(mat)
    vmin = -vmax if center_zero else np.nanmin(mat)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ticks = np.arange(K)
    lbs   = labels or [str(i) for i in range(K)]
    ax.set_xticks(ticks); ax.set_xticklabels(lbs, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(lbs)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)

    if sig_mat is not None:
        sig = np.asarray(sig_mat, dtype=float)
        for i in range(K):
            for j in range(i + 1 if upper_only else 0, K):
                if np.isfinite(sig[i, j]) and sig[i, j] < sig_level:
                    ax.scatter(j, i, s=18, marker="o", edgecolors="black", facecolors="none")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig, ax


def plot_pairwise_heatmap_bidir(
    upper_mat: np.ndarray,
    lower_mat: np.ndarray,
    upper_cmap="coolwarm",
    lower_cmap="PuOr",
    upper_sig: np.ndarray | None = None,
    lower_sig: np.ndarray | None = None,
    sig_level: float = 0.05,
    labels=None,
    upper_label: str = "",
    lower_label: str = "",
    title: str | None = None,
    center_zero: bool = True,
    figsize=(6, 5),
    dpi: int = 150,
    ax=None,
):
    """Heatmap with two KxK matrices split across upper and lower triangles.

    Parameters
    ----------
    upper_mat, lower_mat : (K, K) arrays
        Values for the upper / lower triangle respectively.
    upper_cmap, lower_cmap : colormap names
    upper_sig, lower_sig : (K, K) arrays, optional
        p-value (or any criterion) matrices; cells where value < sig_level
        are annotated with *.
    sig_level : float, default 0.05
    labels : list of str, optional
    upper_label, lower_label : colorbar axis labels
    center_zero : bool
        If True, color scale is symmetric around 0.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    u = np.asarray(upper_mat, dtype=float).copy()
    l = np.asarray(lower_mat, dtype=float).copy()
    K = u.shape[0]
    lbs = labels or [str(i) for i in range(K)]

    u[np.tril_indices(K, k=0)] = np.nan  # show upper triangle only
    l[np.triu_indices(K, k=0)] = np.nan  # show lower triangle only

    def _vlim(mat):
        vmax = max(float(np.nanmax(np.abs(mat))), 1e-12)
        return (-vmax, vmax) if center_zero else (float(np.nanmin(mat)), float(np.nanmax(mat)))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.set_facecolor("white")  # white diagonal cells to visually separate the two triangles

    u_vmin, u_vmax = _vlim(u)
    l_vmin, l_vmax = _vlim(l)
    im_u = ax.imshow(u, cmap=upper_cmap, vmin=u_vmin, vmax=u_vmax, interpolation="nearest")
    im_l = ax.imshow(l, cmap=lower_cmap, vmin=l_vmin, vmax=l_vmax, interpolation="nearest")

    # Ticks and grid
    ticks = np.arange(K)
    ax.set_xticks(ticks)
    ax.set_xticklabels(lbs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(lbs, fontsize=8)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Significance stars
    def _stars(sig_mat, pairs):
        sig = np.asarray(sig_mat, dtype=float)
        for i, j in pairs:
            if np.isfinite(sig[i, j]) and sig[i, j] < sig_level:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=11, color="black", fontweight="bold")

    if upper_sig is not None:
        _stars(upper_sig, [(i, j) for i in range(K) for j in range(i + 1, K)])
    if lower_sig is not None:
        _stars(lower_sig, [(i, j) for i in range(1, K) for j in range(i)])

    if title:
        ax.set_title(title, fontsize=10, pad=8)

    # Colorbars: left ← lower triangle, right ← upper triangle
    divider = make_axes_locatable(ax)
    cax_l = divider.append_axes("left",  size="5%", pad=0.6)
    cax_r = divider.append_axes("right", size="5%", pad=0.08)

    cb_l = fig.colorbar(im_l, cax=cax_l)
    cb_l.ax.yaxis.set_ticks_position("left")
    cb_l.ax.yaxis.set_label_position("left")
    if lower_label:
        cb_l.set_label(lower_label, fontsize=8)
    cb_l.ax.tick_params(labelsize=7)

    cb_r = fig.colorbar(im_u, cax=cax_r)
    if upper_label:
        cb_r.set_label(upper_label, fontsize=8)
    cb_r.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig, ax


def plot_P_enrichment_heatmap(
    res_by_mode: dict,
    results,
    value: str = "z",
    sig_level: float = 0.05,
    cmap: str = "OrRd",
    center_zero: bool = False,
    figsize: tuple | None = None,
    dpi: int = 150,
    title: str | None = None,
    ax=None,
):
    """Single heatmap of per-cluster P enrichment across all modes.

    Rows = modes, columns = clusters (C1 … CK_max).  Each cell shows the
    P enrichment z-score (``value="z"``) or empirical p-value
    (``value="p"``) for that cluster in that mode.  Cells exceeding a mode's
    K are shown as NaN.  Cells where p_emp < sig_level are annotated with *.

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``: ``{mode: {p_res, ...}}``.
    results :
        Clumppling results object with ``.modes`` and ``.mode_K`` attributes.
    value : {"z", "p"}
        Which quantity to colour: z-score or empirical p-value.
    sig_level : float
        Significance threshold for * annotation (applied to p_emp).
    cmap : str
        Defaults to ``"OrRd"`` for z-score (one-sided enrichment);
        use ``"coolwarm"`` if you expect negative z-scores.
    center_zero : bool
        Symmetric colour scale around 0.  Default False (z-scores are
        typically positive for enrichment).
    figsize : tuple, optional
    title : str, optional
    """
    modes = list(res_by_mode.keys())
    # check if results.mode_K is a dict or list and get K for this mode    
    if isinstance(results.mode_K, dict):
        mode_K = {m: results.mode_K[m] for m in modes}
    else:   
        mode_K = {m: results.mode_K[results.modes.index(m)] for m in modes}
    K_max = max(mode_K.values())
    n_modes = len(modes)

    data = np.full((n_modes, K_max), np.nan)
    p_mat = np.full((n_modes, K_max), np.nan)
    for m_idx, mode in enumerate(modes):
        K = mode_K[mode]
        p_res = res_by_mode[mode]["p_res"]
        vals = p_res["z"] if value == "z" else p_res["p_emp"]
        data[m_idx, :K] = vals
        p_mat[m_idx, :K] = p_res["p_emp"]

    vmax = float(np.nanmax(np.abs(data))) if center_zero else float(np.nanmax(data))
    vmin = -vmax if center_zero else float(np.nanmin(data))

    if figsize is None:
        figsize = (max(4, 0.55 * K_max + 1.5), 0.45 * n_modes + 1.5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")

    # Significance stars (based on p_emp regardless of value choice)
    for m_idx in range(n_modes):
        for k in range(K_max):
            if np.isfinite(p_mat[m_idx, k]) and p_mat[m_idx, k] < sig_level:
                ax.text(k, m_idx, "*", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")

    ax.set_yticks(np.arange(n_modes))
    ax.set_yticklabels(modes, fontsize=8)
    ax.set_xticks(np.arange(K_max))
    ax.set_xticklabels([f"C{k+1}" for k in range(K_max)], fontsize=8)

    ax.set_xticks(np.arange(-0.5, K_max, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_modes, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb_label = "z-score" if value == "z" else "p-value"
    cb.set_label(cb_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if title:
        ax.set_title(title, fontsize=10, pad=8)

    fig.tight_layout()
    return fig, ax


def plot_LFC_enrichment_heatmap(
    res_by_mode: dict,
    results,
    value: str = "z",
    sig_level: float = 0.05,
    cmap: str = "coolwarm",
    center_zero: bool = True,
    figsize: tuple | None = None,
    dpi: int = 150,
    title: str | None = None,
    ax=None,
):
    """Single heatmap of pairwise LFC enrichment across all modes.

    Rows = modes, columns = cluster pairs (i < j) ordered lexicographically up
    to K_max.  The same pair (e.g. C1 vs C2) occupies the same column for every
    mode, making it easy to compare enrichment of a given pair across modes.
    Cells are NaN (blank) for pairs that exceed a mode's K.
    Cells where q < sig_level are annotated with *.

    Pairs are grouped visually by their first cluster index with light vertical
    separators; the secondary x-axis labels each group "Cv*" (e.g. C1v*).

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``: ``{mode: {lfc_res, ...}}``.
    results :
        Clumppling results object with ``.modes`` and ``.mode_K`` attributes.
    value : {"z", "obs"}
        Which LFC quantity to colour: z-score or observed LFC.
    sig_level : float
        Significance threshold for * annotation (applied to q values).
    cmap : str
    center_zero : bool
        Symmetric colour scale around 0.
    figsize : tuple, optional
        Defaults to ``(max(6, 0.55 * n_pairs), 0.45 * n_modes + 1.5)``.
    title : str, optional
    """
    modes = list(res_by_mode.keys())
    # check if results.mode_K is a dict or list and get K for this mode
    if isinstance(results.mode_K, dict):
        mode_K = {m: results.mode_K[m] for m in modes}
    else:
        mode_K = {m: results.mode_K[results.modes.index(m)] for m in modes}
    K_max = max(mode_K.values())

    # All unique pairs (i<j) up to K_max, ordered lexicographically
    pair_cols: list[tuple[int, int]] = [
        (i, j) for i in range(K_max) for j in range(i + 1, K_max)
    ]
    n_modes = len(modes)
    n_pairs = len(pair_cols)

    # Fill data and significance matrices
    data = np.full((n_modes, n_pairs), np.nan)
    sig  = np.full((n_modes, n_pairs), np.nan)
    for m_idx, mode in enumerate(modes):
        K = mode_K[mode]
        lfc_res = res_by_mode[mode]["lfc_res"]
        val_mat = lfc_res["test"]["z_mat"] if value == "z" else lfc_res["obs_lfc"]
        q_mat   = lfc_res["test"]["q_mat"]
        for p_idx, (i, j) in enumerate(pair_cols):
            if i < K and j < K:
                data[m_idx, p_idx] = val_mat[i, j]
                sig[m_idx, p_idx]  = q_mat[i, j]

    # Colour scale
    vmax = float(np.nanmax(np.abs(data))) if center_zero else float(np.nanmax(data))
    vmin = -vmax if center_zero else float(np.nanmin(data))

    if figsize is None:
        figsize = (max(6, 0.55 * n_pairs), 0.45 * n_modes + 1.5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")

    # Significance stars
    for m_idx in range(n_modes):
        for p_idx in range(n_pairs):
            if np.isfinite(sig[m_idx, p_idx]) and sig[m_idx, p_idx] < sig_level:
                ax.text(p_idx, m_idx, "*", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")

    # y-axis: mode names
    ax.set_yticks(np.arange(n_modes))
    ax.set_yticklabels(modes, fontsize=8)

    # x-axis: pair labels
    pair_labels = [f"C{i+1}v{j+1}" for (i, j) in pair_cols]
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=90, ha="center", fontsize=7)

    # Minor grid
    ax.set_xticks(np.arange(-0.5, n_pairs, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_modes, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Light vertical separators when the first cluster index changes (C1v* → C2v* etc.)
    prev_i = pair_cols[0][0]
    group_starts: list[int] = [0]  # column index where each i-group starts
    for p_idx, (i, _) in enumerate(pair_cols):
        if i != prev_i:
            ax.axvline(p_idx - 0.5, color="black", lw=1.0, alpha=0.4, zorder=3)
            group_starts.append(p_idx)
            prev_i = i

    # Label each i-group as "C{i+1}v*" using a blended transform (data-x, axes-y)
    # This avoids twiny(), which breaks tight_layout when combined with make_axes_locatable.
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    group_ends = group_starts[1:] + [n_pairs]
    for start, end, (i, _) in zip(group_starts, group_ends, [pair_cols[s] for s in group_starts]):
        mid = (start + end - 1) / 2
        ax.text(mid, 1.02, f"C{i+1}v*", ha="center", va="bottom",
                fontsize=8, transform=trans, clip_on=False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb_label = "LFC z-score" if value == "z" else "observed LFC"
    cb.set_label(cb_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if title:
        ax.set_title(title, fontsize=10, pad=18)

    fig.tight_layout()
    return fig, ax


def plot_top_pairwise_df(df: pd.DataFrame, value_col="z", sig_col="q", alpha=0.05,
                         top_n=-1, labels=None, sort_by="q", figsize=(8, 6), dpi=150, ax=None):
    """Dot plot of top cluster pairs sorted by significance or effect size."""
    d = df.copy()
    d["pair"] = ([f"{labels[i]} vs {labels[j]}" for i, j in zip(d["i"], d["j"])]
                 if labels is not None else [f"{i} vs {j}" for i, j in zip(d["i"], d["j"])])
    d = d.sort_values(sort_by if sort_by != "abs_value"
                      else d.assign(_v=d[value_col].abs()).sort_values("_v", ascending=False).index)
    if top_n > 0:
        d = d.head(top_n)
    d = d.iloc[::-1].reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    y = np.arange(len(d))
    ax.scatter(d[value_col], y)
    sig_mask = (d[sig_col] < alpha).values
    ax.scatter(d.loc[sig_mask, value_col], y[sig_mask],
               s=80, facecolors="none", edgecolors="black")
    ax.set_yticks(y); ax.set_yticklabels(d["pair"])
    ax.axvline(0, lw=1)
    ax.set_xlabel(value_col); ax.set_title("Pairwise LFC comparisons")
    ax.grid(axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_P_enrichment_pval(p_res: Dict[str, float], cluster_labels: List[str], colors: List[str], title: str = "",
                           figsize: Tuple[float, float] = (4, 4), dpi: int = 150, ax=None):
    """Bar chart of empirical p-values (-log10) per cluster."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    ax.bar(cluster_labels, -np.log10(p_res["p_emp"]), color=colors, edgecolor="none")
    ax.axhline(-np.log10(0.05), color="gray", lw=0.8, ls=":")
    ax.text(0.01, -np.log10(0.05) + 0.05, "p = 0.05", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("-log10(p)")
    ax.set_title(title if title else "Empirical p-value per cluster")
    fig.tight_layout()
    return fig, ax


def plot_P_enrichment_zscore(p_res: Dict[str, float], cluster_labels: List[str], colors: List[str], title: str = "",
                             figsize: Tuple[float, float] = (4, 4), dpi: int = 150, ax=None):
    """Bar chart of z-scores vs null per cluster."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    ax.bar(cluster_labels, p_res["z"], color=colors, edgecolor="none")
    if any(p_res["z"] > 2):
        ax.axhline(2,  color="gray", lw=0.6, ls=":")
    if any(p_res["z"] < -2):
        ax.axhline(-2, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("z")
    ax.set_title(title if title else "Z-score vs null")
    fig.tight_layout()
    return fig, ax


def plot_sepLFC_null_sep(sep_res: Dict[str, float], title: str = "", figsize: Tuple[float, float] = (5, 4), dpi: int = 150, ax=None):
    """Histogram of null best-sepLFC per random set, with gene-set value marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    sns.histplot(sep_res["null_sepLFC"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.axvline(sep_res["gs_sepLFC"], color="red", ls="--",
               label=f"GS (val={sep_res['gs_sepLFC']:.3f})\n$p={sep_res['p_vs_null_sep']:.2e}$")
    ax.set_xlabel("sepLFC")
    ax.set_title(title if title else "Null: largest sepLFC per random set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_sepLFC_null_fixed(sep_res: Dict[str, float], cluster_labels: List[str], title: str = "", figsize: Tuple[float, float] = (5, 4), dpi: int = 150, ax=None):
    """Histogram of null LFC at the gene-set's fixed cluster groups, with gene-set value marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    low_str  = "Cls."+"+".join(str(k+1) for k in sep_res["sepL"])
    high_str = "Cls."+"+".join(str(k+1) for k in sep_res["sepH"])
    sns.histplot(sep_res["null_lfc_at_sep"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.axvline(sep_res["gs_sepLFC"], color="red", ls="--",
               label=f"GS (val={sep_res['gs_sepLFC']:.3f})\n$p={sep_res['p_vs_null_fixed']:.2e}$")
    ax.set_xlabel(f"LFC [{high_str}]/[{low_str}]")
    ax.set_title(title if title else "Null: LFC at gene-set's clusters")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_sepLFC_distribution(
    df: pd.DataFrame,
    gs_genes,
    title: str = "",
    kind: str = "auto",
    show_gs_textbox: bool = True,
    gs_textbox_threshold: float | None = None,
    n_gs_textbox: int = 10,
    show_non_gs_textbox: bool = False,
    n_non_gs_textbox: int = 10,
    textbox_fontsize: int = 9,
    figsize: Tuple[float, float] = (6, 5),
    dpi: int = 150,
    ax=None,
):
    """Distribution of per-gene sepLFC, with gene-set genes highlighted.

    Parameters
    ----------
    df : pd.DataFrame
        Gene-indexed DataFrame with columns ``sepLFC`` and ``rank_sepLFC``
        (output of ``compute_all_feature_metrics`` filtered to a sepCls).
        Should be pre-sorted descending by ``sepLFC``.
    gs_genes : set or list of str
        Gene names belonging to the gene set of interest.
    title : str
        Axes title.
    kind : {"auto", "bar", "hist"}
        Chart type.  ``"auto"`` (default) uses bar when ``len(df) < 30``,
        hist otherwise.
    show_gs_textbox : bool
        Show a text box listing gene-set genes with high sepLFC.  Only
        rendered in hist mode.  Default ``True``.
    gs_textbox_threshold : float or None
        Minimum sepLFC for inclusion in the GS text box.  When ``None``
        (default) the top ``n_gs_textbox`` gene-set genes by rank are shown.
    n_gs_textbox : int
        Maximum number of gene-set genes in the GS text box (used when
        ``gs_textbox_threshold`` is ``None``).  Default 10.
    show_non_gs_textbox : bool
        Show a text box listing the top non-GS genes by sepLFC.  Only
        rendered in hist mode.  Default ``False``.
    n_non_gs_textbox : int
        Number of top non-GS genes to list.  Default 10.
    textbox_fontsize : int
        Font size for gene lines inside text boxes.  The box title is rendered
        one point larger and bold.  Default 9.
    figsize : tuple of (float, float)
        Figure size in inches.  Default ``(6, 5)``.
    dpi : int
        Figure resolution.  Default 150.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes if provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    gs_genes = set(gs_genes)
    genes_in_df = gs_genes & set(df.index)
    n = len(df)

    use_bar = (kind == "bar") or (kind == "auto" and n < 30)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    cmap = plt.cm.viridis

    if use_bar:
        sns.barplot(x=df.index, y=df["sepLFC"], ax=ax, color="lightblue")
        for gene in genes_in_df:
            ax.scatter(gene, df.loc[gene, "sepLFC"],
                       color="salmon", edgecolor="black", zorder=5, clip_on=False)
        if n > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    else:
        # use shared bins so the GS histogram aligns with the background one
        _bin_edges = np.histogram_bin_edges(df["sepLFC"], bins=30)
        sns.histplot(df["sepLFC"], bins=_bin_edges, kde=True, ax=ax, color="lightblue")
        if genes_in_df:
            _xs = np.array([df.loc[g, "sepLFC"] for g in genes_in_df])
            _counts, _ = np.histogram(_xs, bins=_bin_edges)
            _width = (_bin_edges[1] - _bin_edges[0]) * 0.9
            _centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
            _gs_max = max(int(_counts.max()), 1)
            _ymax   = ax.get_ylim()[1]
            _scale  = _ymax * 0.5 / _gs_max     # mirror = half the positive height
            ax.bar(_centers, -_counts * _scale, width=_width, bottom=0,
                   color="salmon", alpha=0.8, zorder=2)
            ax.set_ylim(bottom=-_ymax * 0.5 * 1.1)

            # ── Left y-axis: relabel negative ticks with GS counts in salmon
            _pos_ticks = [t for t in ax.get_yticks() if 0 <= t <= _ymax]
            _gs_count_vals = np.arange(0, _gs_max + 1, max(1, _gs_max // 4))
            _gs_y_pos = [-int(c) * _scale for c in _gs_count_vals if c > 0]
            ax.set_yticks(list(_pos_ticks) + _gs_y_pos)
            ax.set_yticklabels(
                [f"{int(t)}" for t in _pos_ticks] +
                [f"{int(c)}" for c in _gs_count_vals if c > 0]
            )
            for tick in ax.yaxis.get_major_ticks():
                if tick.get_loc() < -1e-9:
                    tick.label1.set_color("salmon")
                    tick.tick1line.set_color("salmon")

        # ── Text box 1: GS genes ───────────────────────────────────────────
        if show_gs_textbox and genes_in_df:
            from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
            df_gs = df.loc[sorted(genes_in_df, key=lambda g: df.loc[g, "rank_sepLFC"])]
            if gs_textbox_threshold is not None:
                df_gs = df_gs[df_gs["sepLFC"] > gs_textbox_threshold]
            else:
                df_gs = df_gs.head(n_gs_textbox)
            if not df_gs.empty:
                lines = [f"{g}: {df_gs.loc[g, 'sepLFC']:.2f}" for g in df_gs.index]
                _title_area = TextArea(
                    "Top GS:",
                    textprops=dict(fontsize=textbox_fontsize + 1, fontweight="bold"),
                )
                _body_area = TextArea(
                    "\n".join(lines),
                    textprops=dict(fontsize=textbox_fontsize),
                )
                _box = AnchoredOffsetbox(
                    loc="upper right",
                    child=VPacker(children=[_title_area, _body_area], pad=0, sep=1),
                    pad=1, frameon=True,
                    bbox_to_anchor=(0.98, 0.97),
                    bbox_transform=ax.transAxes,
                )
                _box.patch.set(facecolor="lightgray", alpha=0.8, linewidth=0.8)
                ax.add_artist(_box)

        # ── Text box 2: top non-GS genes ──────────────────────────────────
        if show_non_gs_textbox:
            from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
            df_non = df[~df.index.isin(gs_genes)].head(n_non_gs_textbox)
            if not df_non.empty:
                lines = [f"{g}: {df_non.loc[g, 'sepLFC']:.2f}" for g in df_non.index]
                _title_area = TextArea(
                    "Top non-GS:",
                    textprops=dict(fontsize=textbox_fontsize + 1, fontweight="bold"),
                )
                _body_area = TextArea(
                    "\n".join(lines),
                    textprops=dict(fontsize=textbox_fontsize),
                )
                _box = AnchoredOffsetbox(
                    loc="upper center",
                    child=VPacker(children=[_title_area, _body_area], pad=0, sep=1),
                    pad=1, frameon=True,
                    bbox_to_anchor=(0.5, 0.97),
                    bbox_transform=ax.transAxes,
                )
                _box.patch.set(facecolor="lightyellow", alpha=0.8, linewidth=0.8)
                ax.add_artist(_box)

    ax.set_xlabel("sepLFC")
    ax.set_ylabel("")          # clear default label added by sns.histplot / sns.barplot
    ax.set_title(title)

    # Two y-axis labels: positive side (default color) + mirror side (salmon)
    # x=-0.13 gives enough clearance from the tick labels
    _ylim = ax.get_ylim()
    _ymid_pos = _ylim[1] / 2
    _ymid_neg = _ylim[0] / 2
    ax.text(-0.12, (_ymid_pos - _ylim[0]) / (_ylim[1] - _ylim[0]),
            "count", transform=ax.transAxes,
            ha="center", va="center", rotation=90, fontsize=9)
    if _ylim[0] < 0:
        ax.text(-0.12, (_ymid_neg - _ylim[0]) / (_ylim[1] - _ylim[0]),
                "GS gene count", transform=ax.transAxes,
                ha="center", va="center", rotation=90, fontsize=9, color="salmon")

    fig.tight_layout()
    return fig, ax


def plot_gene_P_bars(
    P_gs: np.ndarray,
    gene_set: List[str],
    cluster_labels: List[str],
    colors: List[str],
    top_n: int | None = None,
    gene_label_colors: dict | None = None,
):
    """
    Per-cluster waterfall bars showing each gene's loading within each cluster.

    Genes are ranked by loading within each cluster and drawn as horizontal
    bars colored by cluster.

    Parameters
    ----------
    P_gs : np.ndarray, shape (n_gs, K)
        Gene-set rows of the aligned P matrix.
    gene_set : list of str
        Gene names corresponding to rows of ``P_gs``.
    cluster_labels : list of str
        Labels for each cluster (columns of ``P_gs``).
    colors : list of str
        One color per cluster used to fill the bars.
    top_n : int or None
        If set and ``n_gs > top_n``, each cluster panel shows only the
        top-``top_n`` genes by per-cluster P.  Default ``None`` (show all).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
        Array of K axes, one per cluster.
    """
    n_gs = P_gs.shape[0]
    K    = P_gs.shape[1]
    n_display = min(n_gs, top_n) if top_n is not None else n_gs
    fig, axes = plt.subplots(1, K, figsize=(2.5 * K, 4), dpi=150)
    for k, ax in enumerate(axes):
        order    = np.argsort(P_gs[:, k])[::-1][:n_display]
        sorted_P = P_gs[order, k]
        cum = 0
        for ii, pv in enumerate(sorted_P):
            ax.add_patch(patches.Rectangle((cum, ii), pv, 1, lw=0, fc=colors[k]))
            cum += pv
        ax.set_xlim(0, sorted_P.sum())
        ax.set_ylim(0, len(sorted_P))
        ax.set_yticks(np.arange(len(sorted_P)) + 0.5)
        ax.set_yticklabels(np.array(gene_set)[order], fontsize=8)
        if gene_label_colors:
            for tick in ax.get_yticklabels():
                c = gene_label_colors.get(tick.get_text())
                if c is not None:
                    tick.set_bbox(dict(facecolor=c, edgecolor='none',
                                       alpha=0.4, boxstyle='round,pad=0.15'))
        ax.invert_yaxis()
        ax.set_title(cluster_labels[k], weight="bold"); ax.set_xlabel("P")
    fig.tight_layout()
    return fig, axes


def plot_per_cluster_P(
    P_gs: np.ndarray,
    gene_set: List[str],
    cluster_labels: List[str],
    colors: List[str],
    null_mean_P: Optional[np.ndarray] = None,
    gs_title: str = "",
    dpi: int = 150,
):
    """
    Super-figure with 1 + K subpanels.

    Top row (1 panel spanning all columns):
        Scatter of mean gene-set P per cluster overlaid on boxplots of the
        null distribution (from ``sample_null_P``), sorted by observed mean P
        descending.  Each cluster is coloured accordingly.  Y-axis is log scale.
        If ``null_mean_P`` is None, only the scatter is drawn.
    Bottom row (K panels):
        Per-cluster waterfall plots (gene loadings, cumulative rectangles).

    Parameters
    ----------
    P_gs : np.ndarray, shape (n_genes, K)
        Gene-set rows of the aligned P matrix.
    gene_set : list of str
        Gene names corresponding to rows of P_gs.
    cluster_labels : list of str
        Labels for each cluster (columns of P_gs).
    colors : list of str
        One colour per cluster.
    null_mean_P : np.ndarray, shape (n_perm, K), optional
        Null mean loading vectors from ``sample_null_P``.  If provided, a
        boxplot of the null distribution is drawn behind the scatter.
    gs_title : str
        Optional title prefix for the top panel.
    dpi : int
        Figure resolution. Default 150.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_top : matplotlib.axes.Axes
    axes_bottom : list of matplotlib.axes.Axes, length K
    """
    P_gs = np.asarray(P_gs, dtype=float)
    n_genes, K = P_gs.shape
    gene_set = list(gene_set)

    fig_w = max(10, 2.5 * K)
    fig_h = 8

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(
        2, K,
        height_ratios=[1, 2],
        hspace=0.45,
        wspace=0.5,
    )

    # ── Top panel: null boxplot + observed scatter ────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])

    obs_mean = P_gs.mean(axis=0)                       # shape (K,)
    cluster_order = np.argsort(obs_mean)[::-1]         # descending observed P

    for i, k_idx in enumerate(cluster_order):
        col = colors[k_idx]

        # Boxplot of null distribution (horizontal)
        if null_mean_P is not None:
            ax_top.boxplot(
                null_mean_P[:, k_idx],
                positions=[i],
                widths=0.5,
                vert=False,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=col, alpha=0.35, linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color=col, linewidth=0.8),
                capprops=dict(color=col, linewidth=0.8),
                flierprops=dict(marker=".", color=col, alpha=0.3,
                                markersize=3, linestyle="none"),
            )

        # Scatter of observed gene-set mean P
        ax_top.scatter(
            obs_mean[k_idx], i,
            color=col, s=70, zorder=5,
            edgecolors="black", linewidths=0.8,
        )

    ax_top.set_yticks(range(K))
    ax_top.set_yticklabels(
        [cluster_labels[i] for i in cluster_order],
        fontsize=10,
    )
    ax_top.invert_yaxis()
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Mean P (log scale)", fontsize=12)
    ax_top.tick_params(axis="x", labelsize=10)
    title = f"{gs_title} – P distribution (gene set vs. null)" if gs_title else "P distribution (gene set vs. null)"
    ax_top.set_title(title, fontsize=12, weight="bold")

    # ── Bottom panels: exact plot_gene_P_bars logic ───────────────────────────
    axes_bottom = []
    for k in range(K):
        ax = fig.add_subplot(gs[1, k])
        axes_bottom.append(ax)

        order    = np.argsort(P_gs[:, k])[::-1]
        sorted_P = P_gs[order, k]
        cum = 0
        for ii, pv in enumerate(sorted_P):
            ax.add_patch(patches.Rectangle((cum, ii), pv, 1, lw=0, fc=colors[k]))
            cum += pv
        ax.set_xlim(0, sorted_P.sum())
        ax.set_ylim(0, len(sorted_P))
        ax.set_yticks(np.arange(len(sorted_P)) + 0.5)
        ax.set_yticklabels(np.array(gene_set)[order], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(cluster_labels[k], weight="bold", fontsize=12)
        ax.set_xlabel("P", fontsize=12)

    return fig, ax_top, axes_bottom


def plot_gene_P_stacked(
    P_gs: np.ndarray,
    gene_set: List[str],
    cluster_labels: List[str],
    gs_title: str = "",
    log_scale: bool = True,
    sort_by_sum: bool = False,
    top_n: int | None = None,
    gene_colors=None,
    figsize: Tuple[float, float] = (6, 4),
    dpi: int = 150,
):
    """Stacked bar chart of per-gene P values across clusters.

    Parameters
    ----------
    P_gs : np.ndarray
        Shape (n_gs, K). Gene-set rows of the P matrix.
    gene_set : list of str
        Gene names corresponding to rows of P_gs.
    cluster_labels : list of str
        Labels for each cluster (columns of P_gs).
    gs_title : str
        Title prefix for the plot. Default ``""``.
    log_scale : bool
        If True, use a log y-axis. Default True.
    sort_by_sum : bool
        If True, sort clusters in descending order of total P sum. Default False.
    top_n : int or None
        If set and ``n_gs > top_n``, restrict to the top-``top_n`` genes by
        total P across clusters (before any cluster sorting).  Default ``None``
        (show all genes).
    gene_colors : list or None
        One color per gene (after any ``top_n`` subsetting).  If ``None``
        (default), colors are drawn from the ``tab20`` colormap.
    figsize : tuple of (float, float)
        Figure size in inches. Default ``(6, 4)``.
    dpi : int
        Figure resolution. Default 150.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    gene_set = list(gene_set)
    n_gs = P_gs.shape[0]
    if top_n is not None and n_gs > top_n:
        _gene_order = np.argsort(P_gs.sum(axis=1))[::-1][:top_n]
        P_gs     = P_gs[_gene_order, :]
        gene_set = [gene_set[i] for i in _gene_order]

    if sort_by_sum:
        order = np.argsort(P_gs.sum(axis=0))[::-1]
        P_gs = P_gs[:, order]
        cluster_labels = [cluster_labels[i] for i in order]

    tab_colors = (gene_colors if gene_colors is not None
                  else plt.cm.get_cmap("tab20", len(gene_set)).colors)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bottom = np.zeros(P_gs.shape[1])
    for i in range(P_gs.shape[0]):
        ax.bar(
            np.arange(P_gs.shape[1]), P_gs[i, :],
            bottom=bottom,
            label=gene_set[i],
            color=tab_colors[i % len(tab_colors)],
        )
        bottom += P_gs[i, :]
    ax.set_xticks(np.arange(P_gs.shape[1]))
    ax.set_xticklabels(cluster_labels, rotation=45, ha="right")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sum of P values")
    title = f"{gs_title} – P values stacked by gene" if gs_title else "P values stacked by gene"
    ax.set_title(title)
    ax.legend(title="Genes", bbox_to_anchor=(1.05, 1), loc="upper left")
    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig, ax


def plot_seplfc_bipartite(
    gene_set: List[str],
    sepH: List[int],
    sepL: List[int],
    cluster_labels: List[str],
    df_mode,
    top_n_per_pair: int = 5,
    gs_title: str = "",
    lw_scale: float = 10.0,
    min_lw: float = 0.5,
    seg_gap: float = 0.008,
    label_mode: str = "auto",
    label_fontsize: float = 7.0,
    arrow_fan: float = 0.06,
    cmap: str = "Spectral",
    vmin: float = None,
    vmax: float = None,
    colors=None,
    figsize: Tuple[float, float] = (8, 4),
    dpi: int = 150,
    ax=None,
):
    """Bipartite diagram where each gene's single segment sits on the one edge
    that corresponds to its own best cluster separation (``sepLFC`` in
    ``df_mode``).

    Unlike the older bipartite approach that assigns every gene to every
    H-L edge based on ``P_gs``, this function:

    1. For each gene, finds the boundary pair ``(A, B)`` where A is the
       lowest-P cluster in the gene's upper group and B is the highest-P
       cluster in the gene's lower group (adjacent clusters across the max gap).
    2. Keeps only genes whose boundary pair has A ∈ ``sepH`` and B ∈ ``sepL``.
    3. On each edge ``(A, B)``, selects the top ``top_n_per_pair`` genes by
       their ``df_mode.sepLFC`` value.
    4. Draws each selected gene as a single segment on its assigned edge.

    All arrows point from sepH toward sepL (LFC is always positive by
    construction).

    Parameters
    ----------
    gene_set : list of str
        Gene names (must be a subset of ``df_mode.index``).
    sepH : list of int
        Cluster indices in the high group (top row nodes).
    sepL : list of int
        Cluster indices in the low group (bottom row nodes).
    cluster_labels : list of str
        Labels for all K clusters.
    df_mode : pd.DataFrame
        Feature-metrics DataFrame (from ``compute_feature_metrics`` /
        ``compute_all_feature_metrics``) indexed by gene name, with columns
        ``sepLFC`` and ``sepCls``.
    top_n_per_pair : int
        Maximum number of genes to show per (sepH, sepL) edge. Default 5.
    gs_title : str
        Title prefix.
    lw_scale : float
        Maximum line width (for the edge with the largest total sepLFC).
    min_lw : float
        Minimum line width.
    seg_gap : float
        Gap at each end of every gene segment (in t-space). Default 0.008.
    label_mode : str
        Controls gene-name labels on segments. One of:

        * ``"all"``  – label every segment regardless of size.
        * ``"auto"`` – label only segments whose fraction of the edge total
          exceeds ``1 / top_n_per_pair`` (i.e. roughly the equal-share
          threshold). Default.
        * ``"none"`` – suppress all labels.
    label_fontsize : float
        Font size for gene-name labels. Default 7.0.
    arrow_fan : float
        Half-width (in data units) of the fan applied at each node so
        arrowheads from different edges spread out. Default 0.06.
    cmap : str
        Matplotlib colormap name used to color segments by ``sepLFC``.
        Default ``"Spectral"``.
    vmin, vmax : float or None
        Colormap range. ``None`` → inferred from the selected genes' sepLFC
        values.
    colors : list or None
        Per-cluster colors for node fill (indexed by cluster index).
    figsize : tuple of (float, float)
        Figure size in inches. Default ``(8, 4)``.
    dpi : int
        Figure resolution. Default 150.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes if provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    sepH_set = set(sepH)
    sepL_set = set(sepL)

    # ── Assign each gene to its boundary edge ────────────────────────────────
    # boundary_upper = sepCls[1][0] (lowest-P cluster of upper group)
    # boundary_lower = sepCls[0][-1] (highest-P cluster of lower group)
    # Both must lie in the GS sepH / sepL respectively.
    edge_genes: dict = {}   # (hi, li) -> list of (gene, sepLFC)
    for gene in gene_set:
        if gene not in df_mode.index:
            continue
        sepCls = df_mode.loc[gene, "sepCls"]
        if not isinstance(sepCls, tuple) or len(sepCls) != 2:
            continue
        lower_grp, upper_grp = sepCls
        if len(lower_grp) == 0 or len(upper_grp) == 0:
            continue
        b_upper = upper_grp[0]   # lowest-P cluster of upper group (boundary)
        b_lower = lower_grp[-1]  # highest-P cluster of lower group (boundary)
        if b_upper not in sepH_set or b_lower not in sepL_set:
            continue
        hi = list(sepH).index(b_upper)
        li = list(sepL).index(b_lower)
        edge_genes.setdefault((hi, li), []).append(
            (gene, float(df_mode.loc[gene, "sepLFC"]))
        )

    # Select top_n_per_pair genes per edge, sorted by sepLFC descending
    edge_genes = {
        edge: sorted(glist, key=lambda x: x[1], reverse=True)[:top_n_per_pair]
        for edge, glist in edge_genes.items()
    }

    # ── Colormap setup (color segments by sepLFC) ─────────────────────────────
    all_selected = []
    for (hi, li) in sorted(edge_genes):
        for gene, _ in edge_genes[(hi, li)]:
            if gene not in all_selected:
                all_selected.append(gene)

    _all_lfcs = [
        float(df_mode.loc[g, "sepLFC"]) for g in all_selected if g in df_mode.index
    ]
    _vmin = vmin if vmin is not None else (min(_all_lfcs) if _all_lfcs else 0.0)
    _vmax = vmax if vmax is not None else (max(_all_lfcs) if _all_lfcs else 1.0)
    if _vmin == _vmax:
        _vmin, _vmax = _vmin - 1, _vmax + 1
    _cmap = plt.cm.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

    # ── Layout ────────────────────────────────────────────────────────────────
    x_top = np.linspace(0, 1, len(sepH)) if len(sepH) > 1 else np.array([0.5])
    x_bot = np.linspace(0, 1, len(sepL)) if len(sepL) > 1 else np.array([0.5])
    y_top, y_bot = 1.0, 0.0
    _x_range, _y_range = 1.36, 1.50
    _ax_aspect = (figsize[0] / _x_range) / (figsize[1] / _y_range)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if not edge_genes:
        ax.set_title("No genes match any (sepH, sepL) boundary pair")
        ax.axis("off")
        return fig, ax

    # Max total sepLFC across all edges (for line width scaling)
    max_total = max(
        sum(lfc for _, lfc in glist) for glist in edge_genes.values()
    )

    # ── Fan offsets ───────────────────────────────────────────────────────────
    _n_H, _n_L = len(sepH), len(sepL)
    _fan_top = {
        (hi, li): (li / (_n_L - 1) - 0.5) * arrow_fan if _n_L > 1 else 0.0
        for hi in range(_n_H) for li in range(_n_L)
    }
    _fan_bot = {
        (hi, li): (hi / (_n_H - 1) - 0.5) * arrow_fan if _n_H > 1 else 0.0
        for hi in range(_n_H) for li in range(_n_L)
    }

    # ── Draw edges ────────────────────────────────────────────────────────────
    for (hi, li), glist in edge_genes.items():
        total = sum(lfc for _, lfc in glist)
        lw = max(min_lw, lw_scale * total / max_total)

        x1 = x_top[hi] + _fan_top[(hi, li)]
        y1 = y_top
        x2 = x_bot[li] + _fan_bot[(hi, li)]
        y2 = y_bot

        disp_dx = (x2 - x1) * _ax_aspect
        disp_dy = (y2 - y1)
        edge_angle = np.degrees(np.arctan2(disp_dy, disp_dx))
        if edge_angle < -90:
            edge_angle += 180
        elif edge_angle > 90:
            edge_angle -= 180

        _ms = max(20.0, lw * 4)
        _seg_disp_in = np.sqrt(
            ((x2 - x1) * figsize[0] / _x_range) ** 2 +
            ((y2 - y1) * figsize[1] / _y_range) ** 2
        )
        _arrow_head_t = (0.4 * _ms / 72.0) / max(_seg_disp_in, 1e-6)

        cumfrac = 0.0
        for gene, lfc in glist:
            frac = lfc / total
            t0, t1 = cumfrac, cumfrac + frac
            t0_draw = t0 + seg_gap
            t1_draw = t1 - seg_gap
            if t1_draw <= t0_draw:
                cumfrac = t1
                continue

            seg_len = t1_draw - t0_draw
            dt = min(_arrow_head_t / 2.0, seg_len / 4.0)
            color = _cmap(norm(lfc))
            _dark = tuple(np.clip(np.array(mcolors.to_rgb(color)) * 0.55, 0, 1))

            # Line body (leave room for arrowhead at sepL end)
            t1_line = t1_draw - 2 * dt
            xs = [x1 + t0_draw * (x2 - x1), x1 + t1_line * (x2 - x1)]
            ys = [y1 + t0_draw * (y2 - y1), y1 + t1_line * (y2 - y1)]
            ax.plot(xs, ys, color=color, linestyle="solid", lw=lw,
                    solid_capstyle="butt", zorder=2)

            # Arrowhead(s): always toward sepL (positive LFC)
            end_t = t1_draw - dt
            if seg_len > 0.50:
                arrow_centers = [t0_draw + (t1_draw - t0_draw) / 2, end_t]
            else:
                arrow_centers = [end_t]

            for mid_t in arrow_centers:
                tip_t, tail_t = mid_t + dt, mid_t - dt
                ax.annotate(
                    "",
                    xy=(x1 + tip_t * (x2 - x1), y1 + tip_t * (y2 - y1)),
                    xytext=(x1 + tail_t * (x2 - x1), y1 + tail_t * (y2 - y1)),
                    arrowprops=dict(
                        arrowstyle="-|>", fc=color, ec=color,
                        lw=0, mutation_scale=_ms,
                        shrinkA=0, shrinkB=0,
                    ),
                    zorder=4,
                )

            _draw_label = (
                label_mode == "all"
                or (label_mode == "auto" and frac >= 1.0 / (top_n_per_pair+1))
            )
            if _draw_label:
                _mid_t = (t0_draw + t1_draw - dt) / 2
                mx = x1 + _mid_t * (x2 - x1)
                my = y1 + _mid_t * (y2 - y1)
                ax.text(mx, my, gene,
                        ha="center", va="center",
                        fontsize=label_fontsize, rotation=edge_angle,
                        rotation_mode="anchor", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.7))
            cumfrac = t1

    # ── Nodes ─────────────────────────────────────────────────────────────────
    node_kw = dict(s=180, zorder=5, linewidths=1.5)
    for hi, h in enumerate(sepH):
        node_color = colors[h] if colors is not None else "white"
        ax.scatter(x_top[hi], y_top, color=node_color, edgecolors="black", **node_kw)
        ax.text(x_top[hi], y_top + 0.07, cluster_labels[h],
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for li, l in enumerate(sepL):
        node_color = colors[l] if colors is not None else "white"
        ax.scatter(x_bot[li], y_bot, color=node_color, edgecolors="black", **node_kw)
        ax.text(x_bot[li], y_bot - 0.07, cluster_labels[l],
                ha="center", va="top", fontsize=10, fontweight="bold")

    ax.text(-0.06, y_top, "sepH", ha="right", va="center",
            fontsize=10, style="italic", color="gray")
    ax.text(-0.06, y_bot, "sepL", ha="right", va="center",
            fontsize=10, style="italic", color="gray")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
    _sm.set_array([])
    fig.colorbar(_sm, ax=ax, label="sepLFC", shrink=0.7, pad=0.02)

    title = (f"{gs_title} – per-gene sepLFC segments" if gs_title
             else "Per-gene sepLFC segments (bipartite)")
    ax.set_title(title + f" (top {top_n_per_pair} genes per pair)", 
                 fontsize=11, y=0.95)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.25, 1.25)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def plot_gene_lfc(
    df_gene_lfc: pd.DataFrame,
    cluster_labels: List[str],
    sepL: List[int],
    sepH: List[int],
    gs_sepLFC: float,
    colors: List[str],
    figsize: Tuple[float, float] = (5, 3),
    dpi: int = 150,
    ax=None,
    kind: str = "mean",
    top_n: int | None = None,
    show_labels: bool | None = None,
):
    """
    Horizontal bar chart of per-gene LFC between high and low cluster groups.

    Bars are colored on a diverging coolwarm scale centered on zero and
    saturated at ±``gs_sepLFC``.

    Parameters
    ----------
    df_gene_lfc : pd.DataFrame
        Output of ``compute_gene_lfc`` with columns ``gene`` and ``LFC``.
    cluster_labels : list of str
        Not currently used; retained for API compatibility.
    sepL : list of int
        Cluster indices in the low group.
    sepH : list of int
        Cluster indices in the high group.
    gs_sepLFC : float
        Observed gene-set sepLFC; sets the colorbar saturation limits.
    colors : list of str
        Not currently used; retained for API compatibility.
    figsize : tuple of (float, float)
        Figure size in inches. Default ``(5, 3)``.
    dpi : int
        Figure resolution. Default 150.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes if provided.
    kind : {"extreme", "mean"}
        Determines the x-axis label: ``"extreme"`` labels min/max of group P;
        ``"mean"`` labels mean of group P. Default ``"mean"``.
    top_n : int or None
        If set and the number of genes exceeds ``top_n``, restrict to the
        top-``top_n`` genes by absolute LFC.  Default ``None`` (show all).
    show_labels : bool or None
        Whether to draw gene-name tick labels on the y-axis.  If ``None``
        (default), labels are shown when ``top_n`` is set or when
        ``n_genes <= 30``; hidden otherwise.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    df = df_gene_lfc.copy()

    # Optionally restrict to top-n genes by |LFC|
    if top_n is not None and len(df) > top_n:
        df = df.reindex(df["LFC"].abs().nlargest(top_n).index)

    # Auto-determine label visibility
    if show_labels is None:
        show_labels = (top_n is not None) or (len(df) <= 30)

    low_str  = "Cls." + "+".join(str(k+1) for k in sepL)
    high_str = "Cls." + "+".join(str(k+1) for k in sepH)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    y          = np.arange(len(df))
    norm       = mcolors.Normalize(vmin=-gs_sepLFC, vmax=gs_sepLFC, clip=True)
    cmap       = plt.cm.get_cmap("coolwarm")
    bar_colors = [cmap(norm(v)) for v in df["LFC"]]
    ax.barh(y, df["LFC"], color=bar_colors, edgecolor="none")

    if show_labels:
        ax.set_yticks(y)
        ax.set_yticklabels(df["gene"], fontsize=8)
    else:
        ax.set_yticks([])

    # detail = "min/max of group" if kind == "extreme" else "mean of group"
    ax.set_xlabel(f"LFC of [{high_str}] vs. [{low_str}]")
    ax.set_title("Per-gene LFC between gene-set's clusters")
    fig.tight_layout()
    return fig, ax


def _mode_cluster_labels(results, mode, cb_cmap):
    """Return ``(labels, colors)`` for a given mode.

    Parameters
    ----------
    results : ClumpplingResults
        Alignment results with ``mode_K`` and ``modes`` attributes.
    mode : str
        Mode name to look up.
    cb_cmap : list
        Per-cluster color list; the first K entries are returned.

    Returns
    -------
    labels : list of str
        Cluster labels ``["Cls.1", ..., "Cls.K"]``.
    colors : list
        First K entries of ``cb_cmap``.
    """
    # check if results.mode_K is a dict or list and get K for this mode
    if isinstance(results.mode_K, dict):
        K = results.mode_K[mode]
    else:
        K = results.mode_K[results.modes.index(mode)]
    return [f"Cls.{k+1}" for k in range(K)], cb_cmap[:K]


def plot_P_enrichment_grid(res_by_mode, ax_by_mode, results, cb_cmap, kind="pval"):
    """
    Fill a mode-grid figure with per-cluster P enrichment bars.

    Iterates over modes and calls either ``plot_P_enrichment_pval`` or
    ``plot_P_enrichment_zscore`` into the corresponding axes.

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``; must contain a ``"p_res"``
        key for each mode.
    ax_by_mode : dict
        Mapping ``mode -> matplotlib.axes.Axes``, e.g. from
        ``make_mode_grid_by_K``.
    results : ClumpplingResults
        Used to look up K and generate cluster labels via
        ``_mode_cluster_labels``.
    cb_cmap : list
        Per-cluster color list passed to ``_mode_cluster_labels``.
    kind : {"pval", "zscore"}
        Whether to plot empirical p-values or z-scores. Default ``"pval"``.
    """
    plot_fn = plot_P_enrichment_pval if kind == "pval" else plot_P_enrichment_zscore
    for mode, ax in ax_by_mode.items():
        labels, colors = _mode_cluster_labels(results, mode, cb_cmap)
        if isinstance(results.mode_K, dict):
            K = results.mode_K[mode]
        else:
            K = results.mode_K[results.modes.index(mode)]
        labels = [f"{k+1}" for k in range(K)]
        plot_fn(res_by_mode[mode]["p_res"], labels, colors, title=mode, ax=ax)


def plot_LFC_enrichment_grid(res_by_mode, ax_by_mode, results, cb_cmap):
    """
    Fill a mode-grid figure with pairwise LFC z-score heatmaps.

    Iterates over modes and calls ``plot_pairwise_heatmap`` with the LFC
    z-score matrix into the corresponding axes.

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``; must contain a ``"lfc_res"``
        key for each mode.
    ax_by_mode : dict
        Mapping ``mode -> matplotlib.axes.Axes``.
    results : ClumpplingResults
        Used to look up K and generate cluster labels.
    cb_cmap : list
        Per-cluster color list (passed to ``_mode_cluster_labels``).
    """
    for mode, ax in ax_by_mode.items():
        labels, _ = _mode_cluster_labels(results, mode, cb_cmap)
        lfc_res = res_by_mode[mode]["lfc_res"]
        plot_pairwise_heatmap(
            lfc_res["test"]["z_mat"], sig_mat=lfc_res["test"]["q_mat"],
            labels=labels, title=mode, ax=ax,
        )


def plot_sepLFC_enrichment_grid(res_by_mode, ax_by_mode, results, cb_cmap, kind="null_sep"):
    """
    Fill a mode-grid figure with sepLFC null-distribution plots.

    Iterates over modes and calls either ``plot_sepLFC_null_sep``
    or ``plot_sepLFC_null_fixed`` into the corresponding axes.

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``; must contain a ``"sep_res"``
        key for each mode.
    ax_by_mode : dict
        Mapping ``mode -> matplotlib.axes.Axes``.
    results : ClumpplingResults
        Used to look up K and generate cluster labels.
    cb_cmap : list
        Per-cluster color list (passed to ``_mode_cluster_labels``).
    kind : {"null_sep", "null_fixed"}
        Which null comparison to visualize.  ``"null_sep"`` compares to each
        random set's own best sepLFC; ``"null_fixed"`` compares to the null
        evaluated at the observed bipartition. Default ``"null_sep"``.
    """
    for mode, ax in ax_by_mode.items():
        labels, _ = _mode_cluster_labels(results, mode, cb_cmap)
        sep_res = res_by_mode[mode]["sep_res"]
        if kind == "null_sep":
            plot_sepLFC_null_sep(sep_res, title=mode, ax=ax)
        else:
            plot_sepLFC_null_fixed(sep_res, labels, title=mode, ax=ax)



def plot_P_enrichment_by_cluster(
    res_by_mode,
    results,
    cb_cmap,
    kind="pval",
    ncols=None,
    figsize_per_panel=(3.0, 2.8),
    dpi=150,
    sig_threshold=None,
):
    """
    One subplot per cluster; each panel shows that cluster's P enrichment
    across all modes that contain it.

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``; each value contains a ``"p_res"``
        dict with ``"p_emp"`` and ``"z"`` arrays of length K for that mode.
    results : ClumpplingResults
        Used to look up K per mode (via ``mode_K`` and ``modes``).
    cb_cmap : list
        Per-cluster colour list; cluster k gets ``cb_cmap[k]``.
    kind : {"pval", "zscore"}
        ``"pval"`` plots -log10(p_emp); ``"zscore"`` plots the z-score.
        Default ``"pval"``.
    ncols : int or None
        Columns in the subplot grid. Defaults to K_max.
    figsize_per_panel : (float, float)
        Width × height for each individual subplot.
    dpi : int
    sig_threshold : float or None
        Threshold for the reference line.  For ``"pval"``, a horizontal line
        is drawn at -log10(sig_threshold); defaults to ``0.05``.  For
        ``"zscore"``, lines are drawn at ±sig_threshold; defaults to ``2``.
        Pass ``None`` to use the kind-appropriate default.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes, shape (nrows, ncols)
    """
    modes = results.modes
    mode_K_lookup = {
        m: (results.mode_K[m] if isinstance(results.mode_K, dict)
            else results.mode_K[results.modes.index(m)])
        for m in modes
    }
    K_max = max(mode_K_lookup.values())

    ncols = ncols or K_max
    nrows = int(np.ceil(K_max / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=dpi,
        squeeze=False,
        sharex=True,
    )

    # Collect per-cluster data first so we can set shared x ticks once.
    all_mode_names = []
    cluster_data = {}
    for k in range(K_max):
        mode_names, values = [], []
        for m in modes:
            if mode_K_lookup[m] > k:
                p_res = res_by_mode[m]["p_res"]
                v = (-np.log10(p_res["p_emp"][k])
                     if kind == "pval" else p_res["z"][k])
                mode_names.append(m)
                values.append(v)
        cluster_data[k] = (mode_names, values)
        if len(mode_names) > len(all_mode_names):
            all_mode_names = mode_names

    n_modes = len(all_mode_names)

    for k in range(K_max):
        ax = axes[k // ncols][k % ncols]
        color = cb_cmap[k]
        mode_names, values = cluster_data[k]

        # Use positions aligned to the shared x axis.
        positions = [all_mode_names.index(m) for m in mode_names]
        ax.bar(positions, values, color=color, edgecolor="none", width=0.7)
        ax.set_xlim(-0.5, n_modes - 0.5)

        if kind == "pval":
            _thresh = sig_threshold if sig_threshold is not None else 0.05
            thresh_line = -np.log10(_thresh)
            if values and max(values) >= thresh_line:
                ax.axhline(thresh_line, color="gray", lw=0.8, ls=":")
            ax.set_ylabel("-log10(p)", fontsize=8)
        else:
            _thresh = sig_threshold if sig_threshold is not None else 2
            if any(v > _thresh for v in values):
                ax.axhline(_thresh, color="gray", lw=0.6, ls=":")
            if any(v < -_thresh for v in values):
                ax.axhline(-_thresh, color="gray", lw=0.6, ls=":")
            ax.set_ylabel("z-score", fontsize=8)

        ax.set_title(f"Cluster {k + 1}", fontsize=9, color=color, weight="bold")
        ax.tick_params(axis="y", labelsize=7)

        # Show x tick labels only on the bottom visible row of each column.
        col = k % ncols
        last_visible_row = (min(K_max - 1, nrows * ncols - 1) // ncols)
        # Determine the last k in this column that is visible.
        last_k_in_col = max(
            ki for ki in range(K_max) if ki % ncols == col
        )
        if k == last_k_in_col:
            ax.set_xticks(range(n_modes))
            ax.set_xticklabels(all_mode_names, rotation=45, ha="right", fontsize=7)
        else:
            ax.tick_params(axis="x", labelbottom=False)

    for k in range(K_max, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    return fig, axes


def plot_sepLFC_distribution_heatmap(
    res_by_mode,
    n_bins=60,
    cmap="Blues",
    figsize=(10, 0.45),
    dpi=150,
    kind="null_sep",
    pval_threshold=None,
    annotate_pval=False,
):
    """
    Heatmap summary of the null sepLFC distribution across all modes.

    Each row is one mode; colour encodes the density of the null distribution
    in each histogram bin.  The observed gene-set sepLFC is overlaid as a red
    dot on each row, making enrichment strength and consistency across modes
    visible at a glance.

    Parameters
    ----------
    res_by_mode : dict
        Output of ``run_gs_enrichment``; each value must contain a
        ``"sep_res"`` dict.
    n_bins : int
        Number of histogram bins shared across all modes. Default ``60``.
    cmap : str or Colormap
        Colormap for the density heatmap. Default ``"Blues"``.
    figsize : (float, float)
        ``(width, height_per_row)``; total figure height is
        ``height_per_row × n_modes``.
    dpi : int
    kind : {"null_sep", "null_fixed"}
        Which null distribution to display.
        ``"null_sep"`` (default) uses the best-sepLFC null (``null_sepLFC``).
        ``"null_fixed"`` uses the fixed-cluster-group null (``null_lfc_at_sep``).
    pval_threshold : float or None
        If set to a value ``> 0``, the empirical one-sided p-value
        (fraction of null ≥ observed) is computed for each mode.  When the
        p-value is below ``pval_threshold`` the observed point is shown as a
        star (``*``) instead of a dot.  Set to ``<= 0`` (or leave as ``None``)
        to always show a dot regardless of significance.
    annotate_pval : bool
        If ``True``, the empirical p-value is printed in scientific notation
        to the right of each observed dot.  The x-axis right limit is
        expanded automatically so the text stays within the frame.
        Requires ``pval_threshold`` to be set (or any positive value) to
        trigger p-value computation; if ``pval_threshold`` is ``None`` or
        ``<= 0`` and ``annotate_pval`` is ``True``, p-values are still
        computed but the star logic is skipped.  Default ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if kind not in ("null_sep", "null_fixed"):
        raise ValueError(f"kind must be 'null_sep' or 'null_fixed', got {kind!r}")

    null_key = "null_sepLFC" if kind == "null_sep" else "null_lfc_at_sep"
    xlabel   = "sepLFC"       if kind == "null_sep" else "LFC at gene-set's clusters"
    cbar_lbl = "null density"  if kind == "null_sep" else "null fixed density"

    use_pval = annotate_pval or (pval_threshold is not None and pval_threshold > 0)
    use_star = pval_threshold is not None and pval_threshold > 0

    modes = list(res_by_mode.keys())

    all_null = np.concatenate(
        [res_by_mode[m]["sep_res"][null_key] for m in modes]
    )
    bin_edges = np.linspace(all_null.min(), all_null.max(), n_bins + 1)

    density  = np.zeros((len(modes), n_bins))
    obs_vals = []
    p_emps   = []
    for i, m in enumerate(modes):
        sep_res = res_by_mode[m]["sep_res"]
        counts, _ = np.histogram(sep_res[null_key], bins=bin_edges, density=True)
        density[i] = counts
        obs = sep_res["gs_sepLFC"]
        obs_vals.append(obs)
        if use_pval:
            _pkey = 'p_vs_null_sep' if kind == 'null_sep' else 'p_vs_null_fixed'
            if _pkey in sep_res:
                p_emps.append(sep_res[_pkey])
            else:
                null_arr = np.asarray(sep_res[null_key])
                p_emps.append(np.mean(null_arr >= obs))

    fig, ax = plt.subplots(
        figsize=(figsize[0], figsize[1] * len(modes)), dpi=dpi
    )
    im = ax.imshow(
        density, aspect="auto", cmap=cmap,
        extent=[bin_edges[0], bin_edges[-1], len(modes) - 0.5, -0.5],
        interpolation="nearest",
    )

    for i, v in enumerate(obs_vals):
        p = p_emps[i] if use_pval else None
        significant = use_star and p < pval_threshold
        marker = "*" if significant else "o"
        msize  = 80  if significant else 40
        ax.scatter(v, i, color="red", s=msize, marker=marker, zorder=5,
                   linewidths=0, label="observed" if i == 0 else None)

    for i in range(len(modes) - 1):
        ax.axhline(i + 0.5, color="white", lw=1.5, alpha=0.9)

    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mode", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(cbar_lbl, fontsize=8)

    # Place legend at the corner with the lowest average heatmap density.
    # If annotations are being added to the right, right-side corners are
    # penalised so the legend prefers the left.
    _ncb = max(1, n_bins // 5)
    _nrw = max(1, len(modes) // 3) if len(modes) >= 3 else 1
    _corner_density = {
        "upper left":  density[:_nrw,  :_ncb].mean(),
        "upper right": density[:_nrw,  -_ncb:].mean(),
        "lower left":  density[-_nrw:, :_ncb].mean(),
        "lower right": density[-_nrw:, -_ncb:].mean(),
    }
    _legend_loc = min(_corner_density, key=_corner_density.get)
    ax.legend(fontsize=8, loc=_legend_loc)

    # Annotate p-values to the right of each dot and expand xlim to fit.
    if annotate_pval and use_pval:
        x_span   = bin_edges[-1] - bin_edges[0]
        x_offset = x_span * 0.02
        max_obs  = max(obs_vals)
        # Estimate extra space needed: ~8 characters in scientific notation.
        # We expand the right xlim after placing all annotations.
        ax.set_xlim(bin_edges[0], bin_edges[-1])  # reset before expansion
        for i, (v, p) in enumerate(zip(obs_vals, p_emps)):
            ax.text(v + x_offset, i, f"{p:.2e}", va="center", ha="left",
                    fontsize=8, color="red", zorder=6)
        # Expand right edge so annotations fit inside the axes frame.
        fig.canvas.draw()
        renderer  = fig.canvas.get_renderer()
        right_max = bin_edges[-1]
        for txt in ax.texts:
            bb = txt.get_window_extent(renderer=renderer)
            bb_data = ax.transData.inverted().transform(bb)
            right_max = max(right_max, bb_data[1, 0])
        ax.set_xlim(bin_edges[0], right_max + x_span * 0.05)

    fig.tight_layout()
    return fig, ax


__all__ = [
    "plot_pairwise_heatmap",
    "plot_pairwise_heatmap_bidir",
    "plot_P_enrichment_heatmap",
    "plot_LFC_enrichment_heatmap",
    "plot_top_pairwise_df",
    "plot_P_enrichment_pval",
    "plot_P_enrichment_zscore",
    "plot_sepLFC_null_sep",
    "plot_sepLFC_null_fixed",
    "plot_gene_P_bars",
    "plot_gene_P_stacked",
    "plot_gene_lfc",
    "plot_P_enrichment_grid",
    "plot_LFC_enrichment_grid",
    "plot_sepLFC_enrichment_grid",
    "plot_P_enrichment_by_cluster",
    "plot_sepLFC_distribution_heatmap",
]
