"""
plot.py

Functions for visualizations.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Any, Callable

import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap, to_rgb
import matplotlib.cm as cm
from matplotlib import cm, colorbar as mcolorbar
from matplotlib import gridspec
import matplotlib.lines as mlines
from matplotlib.path import Path
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, Rectangle


try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

from clumppling.utils import get_uniq_lb_sep
from clumppling.plot import plot_alignment_list, plot_membership

from .io import ClumpplingResults, CompModelsResults
from .analysis import (compute_profile, 
                       extract_all_mode_pair_mappings, 
                       map_alt_to_ref, 
                       compute_overall_membership_difference)

PathLike = Union[str, Path]
ColorSpec = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]

# ---------------------------------------------------------------------
# Membership-based visualizations
# ---------------------------------------------------------------------

def plot_mode_Q_heatmap(
    results: ClumpplingResults,
    mode_name: str,
    *,
    sort_by: str = "max",  # {"max", "none"}
    cmap: Optional[str] = None,
    colorbar: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of Q for a single mode.

    Parameters
    ----------
    results : ClumpplingResults
        Container with aligned Q matrices.
    mode_name : str
        Mode to plot (must be a key in results.Q_by_mode).
    sort_by : {"max", "none"}, default "max"
        If "max", sort individuals by their max cluster membership.
        If "none", keep original row order.
    cmap : str or Colormap, optional
        Colormap to use in imshow (e.g. "viridis", "plasma").
    colorbar : bool, default True
        Whether to add a colorbar for this subplot.
    ax : matplotlib Axes, optional
        If provided, draw into this axes; otherwise create a new Figure.

    Returns
    -------
    fig, ax
    """
    Q = results.Q_by_mode[mode_name]

    if sort_by == "max":
        order = np.argmax(Q, axis=1).argsort()
        Q_plot = Q[order]
    else:
        Q_plot = Q

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(Q_plot, aspect="auto", cmap=cmap)
    # mark border lines between clusters
    for i in range(1, Q.shape[1]):
        ax.axvline(i - 0.5, color="white", linewidth=1)
    ax.set_xticks(np.arange(Q.shape[1]))
    ax.set_xticklabels([str(i+1) for i in range(Q.shape[1])])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Individual")
    ax.set_title(f"Q heatmap: {mode_name}")
    if colorbar:
        fig.colorbar(im, ax=ax, label="Membership")

    return fig, ax


def plot_all_modes_Q_grid(
    results: ClumpplingResults,
    *,
    sort_by: str = "max",   # passed to plot_mode_Q_heatmap
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    n_ticks: int = 8,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot Q heatmaps for all modes in a grid, using results.mode_names_list
    as layout (rows by K, columns by mode within each K), with a single
    shared colorbar on the right.
    """
    n_rows = len(results.K_range)
    max_modes_per_K = max(len(row) for row in results.mode_names_list)

    if figsize is None:
        figsize = (2.5 * max_modes_per_K + 1, 2.5 * n_rows)

    fig, axes = plt.subplots(n_rows, max_modes_per_K, figsize=figsize, squeeze=False)

    # Hide all axes initially
    for i in range(n_rows):
        for j in range(max_modes_per_K):
            axes[i, j].set_visible(False)

    first_im = None

    # Fill in per-mode heatmaps
    for row_idx, modes_at_K in enumerate(results.mode_names_list):
        for col_idx, mode_name in enumerate(modes_at_K):
            ax = axes[row_idx, col_idx]
            ax.set_visible(True)

            Q = results.Q_by_mode[mode_name]
            if sort_by == "max":
                order = np.argmax(Q, axis=1).argsort()
                Q_plot = Q[order]
            else:
                Q_plot = Q

            im = ax.imshow(Q_plot, aspect="auto", cmap=cmap)
            # mark border lines between clusters
            for i in range(1, Q.shape[1]):
                ax.axvline(i - 0.5, color="white", linewidth=1)
            # set ticks so that there are no more than 8 ticks
            n_clusters = Q.shape[1]
            if n_clusters <= n_ticks:
                ax.set_xticks(np.arange(n_clusters))
                ax.set_xticklabels([str(i+1) for i in range(n_clusters)])
            else:
                # subsample ticks to at most 8
                step = max(1, n_clusters // n_ticks)
                ticks = np.arange(0, n_clusters, step)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(i+1) for i in ticks])
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Individual")
            ax.set_title(f"{mode_name}")

            if first_im is None:
                first_im = im

    # Lay out the subplots first
    fig.tight_layout()

    # Make room on the right, then add a shared colorbar
    if first_im is not None:
        # shrink the grid slightly to leave a right margin
        fig.subplots_adjust(right=0.9)

        fig.colorbar(
            first_im,
            ax=axes,
            label="Membership",
            location="right",
            fraction=0.04,
            pad=0.02,
        )

    return fig, axes


def plot_mode_cluster_bars(
    results: ClumpplingResults,
    mode_name: str,
    colors: Optional[Sequence] = None,
    *,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot bar chart of total membership per cluster for a given mode.

    Parameters
    ----------
    results : ClumpplingResults
    mode_name : str
        Mode to plot.
    ax : Axes, optional
        If given, draw into this Axes.

    Returns
    -------
    fig, ax
    """
    Q = results.Q_by_mode[mode_name]
    K = Q.shape[1]

    totals = Q.sum(axis=0)
    frac = totals / totals.sum()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if colors is not None:
        if len(colors) != K:
            raise ValueError("colors must have length K")
        ax.bar(np.arange(K), frac, color=colors)
    else:
        ax.bar(np.arange(K), frac)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([str(i+1) for i in range(K)])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction of total membership")
    ax.set_title(f"Cluster sizes: {mode_name}")

    return fig, ax


def scatter_by_cluster(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    max_colorbar_ticks: int = 8,
    **scatter_kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of 2D coordinates colored by (discrete) cluster labels.
    """
    coords = np.asarray(coords)
    cluster_labels = np.asarray(cluster_labels)

    if coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")

    # Map arbitrary labels -> integer indices 0..K-1
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx = np.array([label_to_idx[lab] for lab in cluster_labels])

    # Build a discrete colormap with K colors
    if cmap is None:
        base_cmap = plt.get_cmap(None)  # current default
    else:
        base_cmap = plt.get_cmap(cmap)

    colors = base_cmap(np.linspace(0, 1, n_clusters))
    discrete_cmap = ListedColormap(colors)

    # Make boundaries so each integer index is a solid block
    boundaries = np.arange(-0.5, n_clusters + 0.5, 1)
    norm = BoundaryNorm(boundaries, ncolors=n_clusters)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=idx,
        cmap=discrete_cmap,
        norm=norm,
        **scatter_kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, label="Cluster")

        # Full set of tick positions (0..K-1) – these are centers of blocks
        full_tick_idx = np.arange(n_clusters)

        # Subsample to at most max_colorbar_ticks ticks
        if n_clusters <= max_colorbar_ticks:
            tick_idx = full_tick_idx
        else:
            tick_idx = np.linspace(
                0, n_clusters - 1, num=max_colorbar_ticks, dtype=int
            )
            tick_idx = np.unique(tick_idx)

        # Positions: the integer indices (centers of color blocks)
        cbar.set_ticks(tick_idx)

        # Labels: the original cluster IDs (could be ints or strings)
        tick_labels = [str(unique_labels[i]+1) for i in tick_idx]
        cbar.set_ticklabels(tick_labels)

    return fig, ax


def plot_single_spatial_membership(
    Q,
    coords,
    ref_color,
    *,
    cls_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    val_threshold: float = 0.0,
    vmin: float = 0.0,
    vmax: float = 1.0,
    s: float = 1.0,
    alpha: float = 1.0,
    title: Optional[str] = None,
    keep_ticks: bool = False,
):
    """
    Plot a single colored scatter layer of 2D coordinates weighted by membership.

    Parameters
    ----------
    Q : array-like
        Either an (n_cells, K) membership matrix, or an (n_cells,) vector.
    coords : array-like
        Either:
          - (n_cells, 2) array of [x, y] coordinates, or
          - tuple (x, y) of 1D arrays.
    ref_color : color spec
        Base color for the membership colormap (e.g. cmap(k), 'tab:blue', (r,g,b)).
    cls_idx : int, optional
        If Q is (n_cells, K), which column to use as membership.
        Ignored if Q is 1D.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    val_threshold : float, default 0.0
        Only plot points with membership > val_threshold.
    vmin, vmax : float
        Range of membership values for colormap normalization.
    s : float
        Marker size.
    alpha : float
        Marker alpha.
    title : str, optional
        Title for the axis (only set if not None).
    keep_ticks : bool, optional
        If False (default), remove x/y ticks.

    Returns
    -------
    ax : matplotlib.axes.Axes
    sp : PathCollection
        The scatter object.
    """
    Q = np.asarray(Q)

    # membership vector
    if Q.ndim == 2:
        membership = Q[:, cls_idx]
    else:
        membership = Q

    # coordinates
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
    else:
        coords = np.asarray(coords)
        x, y = coords[:, 0], coords[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    # white → ref_color colormap
    cmap_custom = LinearSegmentedColormap.from_list(
        "custom_cmap", ["white", ref_color]
    )

    ax.set_facecolor("lightgray")

    mask = membership > val_threshold

    sp = ax.scatter(
        x[mask],
        y[mask],
        c=membership[mask],
        cmap=cmap_custom,
        vmin=vmin,
        vmax=vmax,
        s=s,
        alpha=alpha,
    )

    if title is not None:
        ax.set_title(title)

    if not keep_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, sp

# ---------------------------------------------------------------------
# Feature-based visualizations
# ---------------------------------------------------------------------

def plot_feature_scatter(
    df: pd.DataFrame,
    *,
    mode_name: str | None = None,
    x: str = "weighted_Psum",
    y: str = "sepLFC",
    highlight: Iterable[str] | None = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of feature metrics, e.g. weighted_Psum vs sepLFC.

    Parameters
    ----------
    df : DataFrame
        Must contain columns `x` and `y`, index = feature names.
    mode_name : str, optional
        For titling; purely cosmetic.
    x, y : str
        Column names in df to use as axes.
    highlight : iterable of str, optional
        Feature names (index values) to annotate.
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"df must contain columns {x!r} and {y!r}")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(df[x].values, df[y].values, s=5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    title = "Feature scatter"
    if mode_name is not None:
        title += f" ({mode_name})"
    ax.set_title(title)

    if highlight is not None:
        highlight = list(highlight)
        for gene in highlight:
            if gene in df.index:
                ax.annotate(
                    gene,
                    (df.loc[gene, x], df.loc[gene, y]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )

    return fig, ax


def in_outer_contour(x: float, y: float, paths) -> bool:
    """Return True if (x, y) lies inside ANY of the given matplotlib.path.Path objects."""
    pt = (x, y)
    return any(p.contains_point(pt) for p in paths)


def plot_feature_kde_with_outliers(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    outlier_mask: np.ndarray,
    *,
    mode_name: str | None = None,
    label_col: str | None = None,
    levels: int = 8,
    cmap: str = "viridis_r",
    bg_point_size: float = 10.0,
    bg_alpha: float = 0.1,
    outlier_point_size: float = 30.0,
    outlier_alpha: float = 0.85,
    x_pad_frac: float = 0.02,
    y_pad_frac: float = 0.05,
    min_x_pad: float = 0.005,
    min_y_pad: float = 1.0,
    adjust_text_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a scatter + filled KDE contour + labeled outlier points for a (x, y) feature pair.
    Parameters
    ----------
    df : DataFrame
        Must contain columns `x_col` and `y_col`.
    x_col, y_col : str
        Column names in df to use as x and y axes.
    outlier_mask : ndarray (bool)   
        Boolean mask aligned to df.index indicating which points to label as outliers.
    mode_name : str, optional
        For titling; purely cosmetic.
    label_col : str, optional   
        Column name in df to use for outlier labels; if None, use df.index.
    levels : int, default 8
        Number of KDE contour levels.
    cmap : str, default "viridis_r"
        Colormap for filled KDE contours.
    bg_point_size : float, default 10.0
        Size of background scatter points.
    bg_alpha : float, default 0.1
        Alpha for background scatter points.
    outlier_point_size : float, default 30.0
        Size of outlier scatter points.
    outlier_alpha : float, default 0.85
        Alpha for outlier scatter points.
    x_pad_frac, y_pad_frac : float, default 0.02, 0.05
        Fractional padding to add to x and y axis limits.
    min_x_pad, min_y_pad : float, default 0.005, 1.0
        Minimum padding to add to x and y axis limits.
    adjust_text_kwargs : dict, optional
        Additional keyword arguments to pass to adjust_text.
    ax : Axes, optional
        Matplotlib Axes to plot on; if None, a new figure and axes are created.
    dpi : int, default 150
        Resolution of the figure in dots per inch.

    Returns
    -------
    fig, ax
    """

    x_data = df[x_col].to_numpy()
    y_data = df[y_col].to_numpy()

    # basic scale
    x_max = float(np.nanmax(x_data))
    y_max = float(np.nanmax(y_data))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 7), dpi=dpi)
    else:
        fig = ax.figure

    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0)
    ax.set_facecolor("white")

    # --- Filled KDE contour ---
    kde = sns.kdeplot(
        data=df,
        x=x_col,
        y=y_col,
        levels=levels,
        cut=0,
        alpha=0.9,
        fill=True,
        cmap=cmap,
        linewidths=0,
        ax=ax,
    )

    # make sure contour edges are off
    for coll in kde.collections:
        coll.set_edgecolor("none")
        coll.set_linewidth(0)

    # axis limits + padding
    x_pad = max(x_pad_frac * x_max, min_x_pad)
    y_pad = max(y_pad_frac * y_max, min_y_pad)
    ax.set_xlim(0, x_max + x_pad)
    ax.set_ylim(0, y_max + y_pad)

    # faint background points
    ax.scatter(
        x_data,
        y_data,
        s=bg_point_size,
        alpha=bg_alpha,
        edgecolors="none",
        facecolors="lightgray",
        zorder=1,
    )

    # label outliers: points outside the outermost contour
    labels_artists = []
    for idx, row in df[outlier_mask].iterrows():

    # for idx, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])

        ax.scatter(
            x,
            y,
            clip_on=False,
            alpha=outlier_alpha,
            edgecolors="none",
            facecolors="C0",
            s=outlier_point_size,
            zorder=3,
        )
        label_text = (
            str(row[label_col]) if label_col is not None else str(idx)
        )
        labels_artists.append(
            ax.text(x, y, label_text, color="k", fontsize=7, zorder=4)
        )

    # move labels nicely if adjustText is available
    if adjust_text is not None and labels_artists:
        default_adj = dict(
            expand_points=(5, 5),
            force_text=(0.2, 0.2),
            max_move=(15, 15),
            arrowprops=dict(arrowstyle="-", color="crimson", lw=0.4),
            ax=ax,
        )
        if adjust_text_kwargs is not None:
            default_adj.update(adjust_text_kwargs)
        adjust_text(labels_artists, **default_adj)

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    if mode_name is not None:
        ax.set_title(
            mode_name,
            x=0.98,
            y=0.96,
            ha="right",
            fontsize=10,
            weight="bold",
        )

    fig.tight_layout()
    return fig, ax


def get_feature_kde_outliers(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_x: float | None = 0.0,
    levels: int = 8,
    cut: float = 0,
    top_n: int | None = None,
    scale: str = "zscore",  # "none", "zscore", or "robust"
    return_mask: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]:
    """
    KDE-based outlier detection, with optional ranking of top_n most extreme points.

    Outlier definition (unchanged):
      - Fit 2D KDE on (x_col, y_col) for eligible points
      - Find points outside outermost contour

    Ranking (when top_n is not None):
      - Compute distance in optionally scaled (x, y) space.
      - scale="zscore": standardize by mean & std
      - scale="robust": standardize by median & IQR
      - scale="none": use raw (x, y)

    Returns
    -------
    outliers_df
    mask (optional)
    """

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    finite = np.isfinite(x) & np.isfinite(y)
    eligible = finite & (x > min_x) if min_x is not None else finite

    if not np.any(eligible):
        mask = pd.Series(False, index=df.index)
        outliers = df.loc[mask]
        return (outliers, mask) if return_mask else outliers

    # --- KDE to get contour paths ---
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=72)
    try:
        kde = sns.kdeplot(
            data=df.loc[eligible, [x_col, y_col]],
            x=x_col,
            y=y_col,
            levels=levels,
            cut=cut,
            fill=True,
            cmap="viridis_r",
            linewidths=0,
            ax=ax,
        )

        if not kde.collections:
            mask = pd.Series(False, index=df.index)
            outliers = df.loc[mask]
            return (outliers, mask) if return_mask else outliers

        outer_paths = kde.collections[0].get_paths()

        points = np.column_stack([x, y])
        inside = np.zeros(len(df), dtype=bool)

        elig_idx = np.flatnonzero(eligible)
        elig_points = points[elig_idx]

        for p in outer_paths:
            inside_elig = p.contains_points(elig_points)
            inside[elig_idx] |= inside_elig

        full_outlier_mask = eligible & (~inside)

        # --- If not ranking, just return all KDE outliers ---
        if top_n is None or top_n < 0:
            mask = pd.Series(full_outlier_mask, index=df.index)
            outliers = df.loc[mask]
            return (outliers, mask) if return_mask else outliers

        # --- Otherwise: rank outliers based on scaled distance in (x, y) ---
        out_idx = np.flatnonzero(full_outlier_mask)
        if out_idx.size == 0:
            mask = pd.Series(False, index=df.index)
            outliers = df.loc[mask]
            return (outliers, mask) if return_mask else outliers

        # use only eligible points to compute scaling
        x_elig = x[eligible]
        y_elig = y[eligible]

        if scale == "zscore":
            # standard mean/std z-score
            mx, sx = np.mean(x_elig), np.std(x_elig)
            my, sy = np.mean(y_elig), np.std(y_elig)
            # avoid division by zero
            sx = sx if sx > 0 else 1.0
            sy = sy if sy > 0 else 1.0

            x_scaled = (x[out_idx] - mx) / sx
            y_scaled = (y[out_idx] - my) / sy

        elif scale == "robust":
            # median / IQR scaling
            mx = np.median(x_elig)
            my = np.median(y_elig)
            qx25, qx75 = np.percentile(x_elig, [25, 75])
            qy25, qy75 = np.percentile(y_elig, [25, 75])
            sx = qx75 - qx25
            sy = qy75 - qy25
            sx = sx if sx > 0 else 1.0
            sy = sy if sy > 0 else 1.0

            x_scaled = (x[out_idx] - mx) / sx
            y_scaled = (y[out_idx] - my) / sy

        else:  # "none"
            x_scaled = x[out_idx]
            y_scaled = y[out_idx]

        # distance in scaled space (no sqrt needed for ranking)
        scores = x_scaled * x_scaled + y_scaled * y_scaled

        order = np.argsort(-scores)  # descending
        keep_n = min(top_n, out_idx.size)
        keep_idx = out_idx[order[:keep_n]]

        top_mask_np = np.zeros(len(df), dtype=bool)
        top_mask_np[keep_idx] = True
        mask = pd.Series(top_mask_np, index=df.index)

        outliers = df.loc[mask]
        return (outliers, mask) if return_mask else outliers

    finally:
        plt.close(fig)


def plot_top_features_bar(
    df: pd.DataFrame,
    *,
    mode_name: str | None = None,
    metric: str = "weighted_Psum",
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of top-N features by a given metric (e.g. weighted_Psum).

    Parameters
    ----------
    df : DataFrame
        Index = feature names, must contain column `metric`.
    mode_name : str, optional
        For titling.
    metric : str, default "weighted_Psum"
    top_n : int, default 20
        Number of top features to show.
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if metric not in df.columns:
        raise KeyError(f"df must contain column {metric!r}")

    df_sorted = df.sort_values(metric, ascending=False).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * top_n)))
    else:
        fig = ax.figure

    ax.barh(np.arange(len(df_sorted)), df_sorted[metric].values)
    ax.set_yticks(np.arange(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel(metric)

    title = f"Top {top_n} features by {metric}"
    if mode_name is not None:
        title += f" ({mode_name})"
    ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_P_sorted(
    P_sorted: np.ndarray,
    LFC_sorted: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot sorted log2(P) along cluster index, coloring each gene's curve by the argmax of its LFC profile
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=150)
    else:
        fig = ax.figure

    M, K = P_sorted.shape

    for i_g in range(M):
        color_idx = int(np.argmax(LFC_sorted[i_g, :]))
        ax.plot(
            np.log2(P_sorted[i_g, :]),
            lw=0.2,
            alpha=0.1,
            color=f"C{color_idx}",
        )

    ax.set_xticks(np.arange(K))
    if title:
        ax.set_title(title)

    return fig, ax


def _get_mode_P(results: ClumpplingResults, mode_name: str) -> np.ndarray:
    """Helper to fetch a P matrix for a mode from ClumpplingResults."""
    if results.P_aligned_by_mode is not None and mode_name in results.P_aligned_by_mode:
        return results.P_aligned_by_mode[mode_name]
    if results.P_unaligned_by_mode is not None and mode_name in results.P_unaligned_by_mode:
        return results.P_unaligned_by_mode[mode_name]
    raise KeyError(f"No P matrix found for mode '{mode_name}'")


def plot_mode_P_sorted(
    results: ClumpplingResults,
    mode_name: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    For a single mode, compute the clustering profile and plot sorted log P.
    """
    P = _get_mode_P(results, mode_name)
    P_sorted = np.sort(P, axis=1)
    LFC_sorted, _ = compute_profile(P)

    fig, ax = plot_P_sorted(
        P_sorted,
        LFC_sorted,
        ax=ax,
        title=mode_name if title is None else title,
    )

    ax.set_xlabel("Index of sorted clusters")
    K = P.shape[1]
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(np.arange(1, K + 1), fontsize=8)

    return fig, ax


def plot_mode_sepLFC_distribution(
    results: ClumpplingResults,
    mode_name: str,
    *,
    lfc_threshold: float = 10.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    For a single mode, plot distribution of sepLFC by 'how many clusters are
    separated' (index of sorted cluster before the sepLFC gap).
    """

    P = _get_mode_P(results, mode_name)
    M, K = P.shape

    LFC_sorted, idx_sorted = compute_profile(P)

    # sepLFC_idx = index of the maximal gap in LFC_sorted
    sepLFC_idx = np.argmax(LFC_sorted, axis=1)
    sepLFC = LFC_sorted[np.arange(M), sepLFC_idx]

    df = pd.DataFrame({"LFC": sepLFC, "sepLFC_idx": sepLFC_idx})
    df["sepLFC_idx"] = df["sepLFC_idx"].astype(int)
    df["largeLFC"] = df["LFC"] > lfc_threshold

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=150)
    else:
        fig = ax.figure

    sns.boxenplot(
        data=df,
        x="sepLFC_idx",
        y="LFC",
        hue="largeLFC",
        legend=False,
        gap=0.2,
        ax=ax,
    )

    ax.set_title(mode_name if title is None else title)
    ax.set_xlabel("Index of the sorted cluster\nbefore $sepLFC$ gap")

    ax.set_xticks(np.arange(K - 1))
    ax.set_xticklabels(np.arange(1, K), fontsize=8)

    return fig, ax


# ---------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------

def make_mode_grid(
    modes: list[str],
    *,
    n_cols: int = 4,
    panel_size: tuple[float, float] = (4.0, 2.5),
    dpi: int = 150,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    Create a figure + gridspec layout for a list of modes, returning
    a dict {mode_name: ax}.

        - Rows/cols computed from len(modes) and n_cols.
        - panel_size gives (width, height) in inches per cell.

    Example usage:

        fig, ax_by_mode = make_mode_grid(modes, n_cols=4)
        for mode in modes:
            plot_mode_P_sorted(results, mode, ax=ax_by_mode[mode])
        fig.tight_layout()
    """
    n_modes = len(modes)
    if n_modes == 0:
        raise ValueError("modes list is empty")

    n_cols = max(1, n_cols)
    n_rows = math.ceil(n_modes / n_cols)

    fig_width = panel_size[0] * n_cols
    fig_height = panel_size[1] * n_rows

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(n_rows, n_cols)

    ax_by_mode: dict[str, plt.Axes] = {}
    for i_m, mode_name in enumerate(modes):
        row = i_m // n_cols
        col = i_m % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax_by_mode[mode_name] = ax

    return fig, ax_by_mode


def make_mode_grid_by_K(
    results: ClumpplingResults,
    *,
    modes: Sequence[str] | None = None,
    panel_size: tuple[float, float] = (3.0, 2.5),
    dpi: int = 150,
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Create a figure whose axes layout matches `plot_all_modes_Q_grid`:

        - Rows correspond to distinct K values (sorted).
        - Within each row, columns correspond to modes with that K,
          in the order of `modes` (or results.modes if None).
        - Returns a mapping {mode_name: ax} for the cells actually used.

    Parameters
    ----------
    results : ClumpplingResults
        Must have Q_by_mode populated.
    modes : sequence of str, optional
        If provided, only these modes are laid out (in this order).
        Otherwise use results.modes.
    panel_size : (width, height) in inches per panel.
    dpi : int, default 150

    Returns
    -------
    fig : Figure
    axes_by_mode : dict
        Mapping mode_name -> Axes in the grid.
    """
    if modes is None:
        modes = results.modes
    modes = list(modes)

    # K per mode from Q_by_mode
    K_by_mode: Dict[str, int] = {}
    for m in modes:
        if m not in results.Q_by_mode:
            raise KeyError(f"Q_by_mode missing for mode '{m}'")
        K_by_mode[m] = results.Q_by_mode[m].shape[1]

    K_values = sorted(set(K_by_mode.values()))
    rows_modes: list[list[str]] = [
        [m for m in modes if K_by_mode[m] == K] for K in K_values
    ]

    n_rows = len(K_values)
    n_cols = max(len(row) for row in rows_modes) if rows_modes else 0
    if n_cols == 0:
        raise ValueError("No modes found to place on grid.")

    fig_width = panel_size[0] * n_cols
    fig_height = panel_size[1] * n_rows

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(n_rows, n_cols)

    axes_by_mode: Dict[str, plt.Axes] = {}

    for row_idx, row_modes in enumerate(rows_modes):
        for col_idx, mode_name in enumerate(row_modes):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes_by_mode[mode_name] = ax

    return fig, axes_by_mode


def plot_single_cluster_in_grid(
    results: ClumpplingResults,
    coords: np.ndarray,
    mode_name: str,
    cluster_index: int,
    *,
    cmap: Optional[str] = None,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    base_size: float = 5.0,
    size_scale: float = 20.0,
    figsize: Optional[Tuple[float, float]] = None,
    colorbar: bool = True,
    **scatter_kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot membership for a single (mode, cluster) in the full grid layout
    where rows = modes and columns = clusters (0..K_max-1), using
    `results.mode_sep_coord_dict` to place that cluster in the correct cell.

    All other cells are left empty / invisible.

    Parameters
    ----------
    results : ClumpplingResults
    coords : array, shape (n_samples, 2)
        2D coordinates (UMAP, t-SNE, etc.).
    mode_name : str
        Mode name, must be present in results.mode_sep_coord_dict keys.
    cluster_index : int
        Cluster index (column in Q) for that mode.
    cmap : str or Colormap, optional
        Colormap for membership intensity.
    xlabel, ylabel : str
        Axis labels for the occupied cell.
    base_size : float, default 5.0
        Base point size.
    size_scale : float, default 20.0
        Additional scale times membership value.
    figsize : tuple, optional
        Figure size for the full grid.
    colorbar : bool, default True
        Whether to draw a colorbar for the occupied cell.
    **scatter_kwargs :
        Extra kwargs passed to `ax.scatter` for that cell.

    Returns
    -------
    fig, axes : Figure and 2D axes array for the full grid.
    """
    Q = results.Q_by_mode[mode_name]

    if coords.shape[0] != Q.shape[0]:
        raise ValueError(
            f"coords.shape[0] ({coords.shape[0]}) != Q.shape[0] ({Q.shape[0]}) "
            f"for mode {mode_name}"
        )

    if (mode_name, cluster_index) not in results.mode_sep_coord_dict:
        raise KeyError(
            f"(mode_name={mode_name!r}, cluster_index={cluster_index}) "
            "not found in results.mode_sep_coord_dict."
        )

    membership = Q[:, cluster_index]

    n_rows = len(results.modes)
    n_cols = results.K_max

    if figsize is None:
        figsize = (0.8 * n_cols, 0.8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Hide all axes first
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].set_visible(False)

    row_idx, col_idx = results.mode_sep_coord_dict[(mode_name, cluster_index)]
    ax = axes[row_idx, col_idx]
    ax.set_visible(True)

    user_s = scatter_kwargs.get("s", None)
    if user_s is not None:
        s = user_s
    else:
        s = base_size + size_scale * membership

    kwargs = dict(scatter_kwargs)
    kwargs["s"] = s

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=membership,
        cmap=cmap,
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{mode_name}, cluster {cluster_index}")

    if colorbar:
        fig.colorbar(sc, ax=ax, label="Membership")

    fig.tight_layout()
    return fig, axes


def separate_scatter_for_cluster_mode(
    results: ClumpplingResults,
    coords: np.ndarray,
    *,
    cluster_colors: Optional[Sequence] = None,
    val_threshold: float = 0.0,
    s: float = 1.0,
    alpha: float = 1.0,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    suptitle: Optional[str] = None,
    suptitle_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, Dict[Tuple[str, int], plt.Axes]]:
    """
    Plot membership on 2D coords for each (mode, cluster) in a grid:

        rows  = modes (in results.modes order)
        cols  = cluster index 0..K_max-1

    using results.mode_sep_coord_dict to place each (mode, cluster).

    Each cell contains ONE cluster's membership (white→cluster_color).
    """
    modes = results.modes
    K_max = results.K_max

    n_rows = len(modes)
    n_cols = K_max

    # default cluster colors if not provided
    if cluster_colors is None:
        cmap = plt.get_cmap("tab20")
        if K_max == 1:
            cluster_colors = [cmap(0.0)]
        else:
            cluster_colors = [cmap(i / (K_max - 1)) for i in range(K_max)]
    else:
        if len(cluster_colors) < K_max:
            raise ValueError("cluster_colors must have length >= K_max")

    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(n_rows, n_cols)

    axes_handles: Dict[Tuple[str, int], plt.Axes] = {}

    for mode_name in modes:
        Q = results.Q_by_mode[mode_name]
        K = Q.shape[1]
        for i_k in range(K):
            ax_row, ax_col = results.mode_sep_coord_dict[(mode_name, i_k)]
            ax = fig.add_subplot(gs[ax_row, ax_col])
            axes_handles[(mode_name, i_k)] = ax

            plot_single_spatial_membership(
                Q,
                coords,
                ref_color=cluster_colors[i_k],
                cls_idx=i_k,
                ax=ax,
                val_threshold=val_threshold,
                vmin=vmin,
                vmax=vmax,
                s=s,
                alpha=alpha,
                keep_ticks=False,
            )

            # bottom row: cluster labels
            if ax_row == n_rows - 1:
                ax.set_xlabel(
                    f"Cluster {i_k + 1}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    weight="bold",
                )
            else:
                ax.set_xlabel("")

            # first column: mode labels
            if ax_col == 0:
                ax.set_ylabel(
                    f"{mode_name}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
            else:
                ax.set_ylabel("")
    
    # optional global title
    if suptitle is not None:
        default_kwargs = dict(
            fontsize=12,
            weight="bold",
            x=0.01,
            y=0.99,
            ha="left",
            va="bottom",
        )
        if suptitle_kwargs is not None:
            default_kwargs.update(suptitle_kwargs)
        fig.suptitle(suptitle, **default_kwargs)

    fig.tight_layout()
    return fig, axes_handles


def overlay_scatter_for_mode(
    results: ClumpplingResults,
    coords: np.ndarray,
    *,
    cluster_colors: Optional[Sequence] = None,
    val_threshold: float = 0.5,
    s: float = 0.05,
    alpha: float = 0.6,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    suptitle: Optional[str] = None,
    suptitle_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Overlay membership for all clusters within each mode, on a mode-grid:

        rows = K values (in results.K_range order)
        cols = modes within each K (using results.mode_coord_dict)

    Each axis shows all clusters for that mode, with different base colors.
    """
    modes = results.modes
    n_rows = len(results.K_range)
    n_cols = max(len(row) for row in results.mode_names_list)

    K_max = results.K_max

    # default cluster colors if not provided
    if cluster_colors is None:
        cmap = plt.get_cmap("tab20")
        if K_max == 1:
            cluster_colors = [cmap(0.0)]
        else:
            cluster_colors = [cmap(i / (K_max - 1)) for i in range(K_max)]
    else:
        if len(cluster_colors) < K_max:
            raise ValueError("cluster_colors must have length >= K_max")

    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(n_rows, n_cols)

    axes_handles: Dict[str, plt.Axes] = {}

    # optional mode stats (for sizes in titles)
    stats_by_mode = None
    if results.mode_stats is not None and not results.mode_stats.empty:
        if "Mode" in results.mode_stats.columns:
            stats_by_mode = results.mode_stats.set_index("Mode")
        else:
            stats_by_mode = results.mode_stats

    for mode_name in modes:
        Q = results.Q_by_mode[mode_name]
        i_row, i_col = results.mode_coord_dict[mode_name]
        ax = fig.add_subplot(gs[i_row, i_col])

        # title with size if available
        if stats_by_mode is not None and mode_name in stats_by_mode.index and "Size" in stats_by_mode.columns:
            size = stats_by_mode.loc[mode_name, "Size"]
            title = f"{mode_name} (size={size})"
        else:
            title = mode_name

        # overlay all clusters for this mode
        for i_k in range(Q.shape[1]):
            plot_single_spatial_membership(
                Q,
                coords,
                ref_color=cluster_colors[i_k],
                cls_idx=i_k,
                ax=ax,
                val_threshold=val_threshold,
                vmin=vmin,
                vmax=vmax,
                s=s,
                alpha=alpha,
                keep_ticks=False,
            )

        ax.set_title(title, fontsize=10, loc="left")
        axes_handles[mode_name] = ax

        # leftmost column: label K
        if i_col == 0:
            ax.set_ylabel(
                f"K={Q.shape[1]}",
                rotation=0,
                ha="right",
                fontsize=10,
                weight="bold",
            )
        else:
            ax.set_ylabel("")

    # optional global title
    if suptitle is not None:
        default_kwargs = dict(
            fontsize=12,
            weight="bold",
            x=0.01,
            y=0.99,
            ha="left",
            va="bottom",
        )
        if suptitle_kwargs is not None:
            default_kwargs.update(suptitle_kwargs)
        fig.suptitle(suptitle, **default_kwargs)

    fig.tight_layout()
    return fig, axes_handles


# ---------------------------------------------------------------------
# Feature level summaries 
# ---------------------------------------------------------------------

def plot_top_sepLFC_labels(
    df_selected: pd.DataFrame,
    modes: Sequence[str],
    *,
    sepLFC_threshold: float = 0.0,
    cmap: str = "Reds",
    vmin: float = 1e-5,
    vmax: float | None = None,
    y_max: float = 40.0,
    hi_sepLFC_threshold: float = 32.0,
    n_top_hi: int = 15,
    n_top_lo: int = 8,
    figsize_scale: float = 0.95,
    dpi: int = 150,
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    For each mode in `modes`, plot:

      - a vertical axis of sepLFC values,
      - a rug plot of all genes with sepLFC > sepLFC_threshold,
      - labeled horizontal lines for the top sepLFC genes, colored by weighted_Psum,
      - all panels share a single horizontal colorbar (weighted_Psum) on top.

    Parameters
    ----------
    df_selected : DataFrame
        Wide table with columns like:
          - weighted_Psum_{mode_name}
          - sepLFC_{mode_name}
        and index = gene IDs.
    modes : sequence of str
        Mode names used to derive the column suffixes.
    sepLFC_threshold : float, default 0.0
        Only genes with sepLFC > threshold are included per mode.
    cmap : str, default "Reds"
        Colormap used to encode weighted_Psum.
    vmin, vmax : float, optional
        For LogNorm. If vmax is None, it's computed from df_selected across all
        modes and sepLFC > sepLFC_threshold.
    y_max : float, default 40.0
        ymax used for y-axis; also used in label positioning logic.
    hi_sepLFC_threshold : float, default 32.0
        If the top sepLFC in a mode exceeds this, up to `n_top_hi` labels per
        mode are shown; otherwise, up to `n_top_lo`.
    n_top_hi, n_top_lo : int
        See above.
    figsize_scale : float, default 0.95
        Scale factor for figure width: width = figsize_scale * len(modes).
    dpi : int, default 150

    Returns
    -------
    fig : Figure
    axes_by_mode : dict
        Mapping mode_name -> Axes for that panel.
    """

    modes = list(modes)

    # --- Compute global vmax if not provided ---
    if vmax is None:
        max_vals = []
        for mode_name in modes:
            wPsum_col = f"weighted_Psum_{mode_name}"
            sepLFC_col = f"sepLFC_{mode_name}"
            if wPsum_col not in df_selected.columns or sepLFC_col not in df_selected.columns:
                continue
            mask = df_selected[sepLFC_col] > sepLFC_threshold
            if mask.any():
                max_vals.append(df_selected.loc[mask, wPsum_col].max())
        if max_vals:
            vmax = float(np.max(max_vals))
        else:
            vmax = 1.0

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap_wPsum = cm.get_cmap(cmap)

    n_cols = len(modes)
    n_rows = 2

    fig = plt.figure(figsize=(figsize_scale * n_cols, 5), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0)

    # top row: colorbar; bottom row: per-mode panels
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1, 24])

    axes_by_mode: Dict[str, plt.Axes] = {}

    for i_m, mode_name in enumerate(modes):
        ax = fig.add_subplot(gs[1, i_m])
        ax.set_facecolor("white")
        axes_by_mode[mode_name] = ax

        wPsum_col = f"weighted_Psum_{mode_name}"
        sepLFC_col = f"sepLFC_{mode_name}"

        if wPsum_col not in df_selected.columns or sepLFC_col not in df_selected.columns:
            # missing columns -> empty panel
            ax.set_xlim(0, 2)
            ax.set_ylim(0, y_max)
            ax.set_title(mode_name, fontsize=10)
            if i_m > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("sepLFC", fontsize=10)
            continue

        # Filter + sort
        df_mode = (
            df_selected[df_selected[sepLFC_col] > sepLFC_threshold]
            .sort_values(by=sepLFC_col, ascending=False)
        )

        # If nothing passes threshold, keep an empty panel with proper axes
        if df_mode.empty:
            ax.set_xlim(0, 2)
            ax.set_ylim(0, y_max)
            ax.set_title(mode_name, fontsize=10)
            if i_m > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("sepLFC", fontsize=10)
            continue

        # Basic axis styling
        ax.set_xlim(0, 2)
        ax.set_xticks([])
        ax.set_ylim(0, y_max)

        # rugplot of sepLFC
        sns.rugplot(
            data=df_mode,
            y=sepLFC_col,
            color="gray",
            height=0.12,
            lw=1,
            alpha=0.1,
            ax=ax,
        )

        # top sepLFC and how many top genes to label
        top_sfval = float(df_mode.iloc[0][sepLFC_col])
        n_top_raw = n_top_hi if top_sfval > hi_sepLFC_threshold else n_top_lo
        n_top = min(n_top_raw, len(df_mode))

        top_df = df_mode.iloc[:n_top]

        prev_h = 0.98
        last_color = "black"

        for _, r in top_df.iterrows():
            g = r.name  # index = gene name
            y_val = float(r[sepLFC_col])
            v = float(r[wPsum_col])

            # Color based on weighted_Psum
            c = cmap_wPsum(norm(v))
            last_color = c

            # horizontal line in data coords, short segment near left
            ax.axhline(y_val, xmin=0, xmax=0.2, lw=0.5, ls="-", c=c)

            # text y-position in axes coordinates (some spacing vs previous label)
            h = min(y_val / y_max, prev_h - 0.027)
            ax.text(
                0.24,
                h,
                str(g),
                ha="left",
                va="center",
                color=c,
                fontsize=7.5,
                transform=ax.transAxes,
            )
            prev_h = h

        # indicate truncated list if needed
        if (n_top_raw <= n_top) and (top_sfval > hi_sepLFC_threshold) and (len(df_mode) > n_top):
            ax.text(
                0.24,
                prev_h - 0.027,
                "... ...",
                ha="left",
                va="center",
                color=last_color,
                fontsize=7.5,
                transform=ax.transAxes,
            )

        ax.set_ylabel("")
        ax.set_title(mode_name, fontsize=10)
        if i_m > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("sepLFC", fontsize=10)

    # --- Shared colorbar on top ---
    cax = fig.add_subplot(gs[0, :])
    cb = mcolorbar.ColorbarBase(
        cax,
        cmap=cmap_wPsum,
        norm=norm,
        orientation="horizontal",
    )
    cb.set_label("weighted_Psum", fontsize=10, labelpad=-45, loc="right")

    fig.tight_layout()
    return fig, axes_by_mode


def plot_mode_metrics_sepCls(
    df_mode,
    mode_name,
    x_col="weighted_Psum",
    y_col="sepLFC",
    sep_col="sepCls",
    annot_mask=None,
    xmax=None,
    ymax=None,
    custom_color_dict=None,
):
    """
    Scatter plot of feature metrics for a given mode, colored by separating class pattern.

    Parameters
    ----------
    df_mode : pd.DataFrame
        DataFrame containing feature metrics for the mode. Must include 'sepCls', x_col, and y_col.
    mode_name : str
        Name of the mode (for title).
    x_col : str, optional
        Column name for x-axis metric (default is 'weighted_Psum').
    y_col : str, optional
        Column name for y-axis metric (default is 'sepLFC').
    annot_mask : pd.Series or None, optional
        Boolean mask for annotating points (default is None).
    xmax : float or None, optional
        Maximum x-axis limit (default is None, which auto-scales).
    ymax : float or None, optional
        Maximum y-axis limit (default is None, which auto-scales).
    custom_color_dict : dict or None, optional
        Custom color dictionary for 'sepType' categories (default is None).

    Returns
    -------
    None
    """
    assert sep_col in df_mode.columns
    assert x_col in df_mode.columns
    assert y_col in df_mode.columns

    df = df_mode.sort_values(by=[x_col],ascending=False)
    df['fewer_is_high'] = df[sep_col].apply(lambda x: len(x[0])>=len(x[1]))
    df['sepCls_fewer'] = df[sep_col].apply(lambda x: np.sort(x[0])+1 if len(x[0])<len(x[1]) else np.sort(x[1])+1)
    df['sepType'] = df['sepCls_fewer'].apply(lambda x: 'Cls.{}'.format(x[0]) if len(x)==1 else 'Multi.Cls')
    print("Separating Pattern counts:")
    print(df['sepType'].value_counts())

    if not custom_color_dict:
        unique_sepTypes = df['sepType'].unique()
        custom_color_dict = dict()
        base_colors = sns.color_palette("Set2", n_colors=len(unique_sepTypes))
        for i, sepType in enumerate(sorted(unique_sepTypes)):
            custom_color_dict[sepType] = base_colors[i]
    if not xmax:
        xmax = np.ceil(df[x_col].max()/0.005)*0.005
    if not ymax:
        ymax = np.ceil(df[y_col].max()/5)*5
    g = sns.jointplot(data=df, x=x_col, y=y_col, 
                    hue='sepType', palette=custom_color_dict, 
                    hue_order = [k for k in custom_color_dict.keys() if k in df['sepType'].values],
                    s=30, alpha=0.7, lw=0.2, edgecolor='k', 
                    xlim=(0, xmax), ylim=(0, ymax))
    ax = g.ax_joint
    ax.legend(title="Separating Pattern", ncol=1, handletextpad=0.2, labelspacing=0.2, borderpad=0.2)

    sepLFC_med = df[y_col].median()
    ax.axhline(sepLFC_med, lw=0.5, ls='--', c='blue')
    ax.text(xmax,sepLFC_med,'median',va='top',ha='right', fontsize=8, color='blue')

    if adjust_text is not None:
        if annot_mask:
            labels = []
            for r in df[annot_mask].iterrows():
                x = r[x_col]
                y = r[y_col]
                lb = r.name + '*' if r['fewer_is_high'] else r.name
                labels.append(ax.text(x, y, lb, color='k', fontsize=8))
            adjust_text(labels, expand_points=(2, 2),
                        arrowprops=dict(arrowstyle="-", color='red', lw=0.5), ax=ax)
    g.fig.suptitle(mode_name)

    return g, ax


def plot_focal_gene_pvs_across_modes(
    df_pvs_modes: dict[str, pd.DataFrame],
    modes: list[str],
    focal_gene: str,
    custom_color_dict: dict[str, str],
    *,
    x_col="weighted_Psum",
    y_col="sepLFC",
    sep_col="sepCls",
    xlim=None,
    ylim=None,
    figsize=(3.5, 4),
    dpi: int = 150,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor: tuple[float, float] = (0.0, 0.9),
    style_label: list[str] = None,
    ax: plt.Axes | None = None,
):
    """
    For a focal gene, collect (weighted_Psum, sepLFC, sepCls) across modes
    and make the scatter-with-labels plot in one shot.
    """
    res = []
    for mode_name in modes:
        res.append(df_pvs_modes[mode_name].loc[focal_gene])
    df = pd.concat(res, axis=1).T
    df.index = modes  # mode names as index

    assert sep_col in df.columns
    assert x_col in df.columns
    assert y_col in df.columns

    df["fewer_is_high"] = df[sep_col].apply(lambda x: len(x[0]) >= len(x[1]))
    df["sepCls_fewer"] = df[sep_col].apply(
        lambda x: np.sort(x[0]) + 1 if len(x[0]) < len(x[1]) else np.sort(x[1]) + 1
    )
    df["sepType"] = df["sepCls_fewer"].apply(
        lambda x: f"Cls.{x[0]}" if len(x) == 1 else "Multi.Cls"
    )

    all_sepCls = sorted(
        list(set().union(*[set(arr) for arr in df["sepCls_fewer"]]))
    )
    # subset palette to relevant clusters + Multi.Cls
    subset_palette = {}
    if "Multi.Cls" in custom_color_dict:
        subset_palette["Multi.Cls"] = custom_color_dict["Multi.Cls"]
    for c in all_sepCls:
        key = f"Cls.{c}"
        if key in custom_color_dict:
            subset_palette[key] = custom_color_dict[key]

    if style_label is None:
        style_label = ["L (solid)", "H (dotted)"]
    df["Side"] = df["fewer_is_high"].apply(
        lambda x: style_label[1] if x else style_label[0]
    )
    df["Separated Cls."] = df["sepType"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="Separated Cls.",
        palette=subset_palette,
        hue_order=list(subset_palette.keys()),
        style="Side",
        style_order=style_label,
        legend="full",
        s=50,
        alpha=0.7,
        lw=0.2,
        edgecolor="k",
        ax=ax,
    )

    # legend
    ax.legend(
        title="Separated Cls.",
        alignment="center",
        handletextpad=0.2,
        labelspacing=0.15,
        borderpad=0.5,
        fontsize=9,
        title_fontsize=12,
        ncol=2,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        columnspacing=0.1,
    )

    # x/y limits 
    if xlim is not None:
        if len(xlim) > 1:
            ax.set_xlim(xlim[0], xlim[1])
        else:
            ax.set_xlim(0, xlim[0])
    if ylim is not None:
        if len(ylim) > 1:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            ax.set_ylim(0, ylim[0])

    ax.set_ylabel("sepLFC", fontsize=10)
    ax.set_xlabel("weighted_Psum", fontsize=10)

    # labels
    labels = []
    for i in range(len(df)):
        x = df.iloc[i][x_col]
        y = df.iloc[i][y_col]
        lb = df.index[i]
        if len(df.iloc[i]["sepCls_fewer"]) > 1:
            lb += "({})".format(
                ",".join(str(s) for s in df.iloc[i]["sepCls_fewer"])
            )
        labels.append(ax.text(x, y, lb, color="k", fontsize=10))

    adjust_text(
        labels,
        expand_points=(2, 2),
        arrowprops=dict(arrowstyle="-", color="red", lw=0.5),
        ax=ax,
    )

    return fig, ax, df


def plot_feature_sepLFC_across_modes(
    res_model,
    df_pvs_modes: Mapping[str, "pd.DataFrame"],
    selected_feature: str,
    feature_names: Sequence[str],
    colors: Sequence,
    *,
    label_rank: bool = True,
    dpi: int = 150,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar plot of sepLFC for a focal gene across all modes.

    Parameters
    ----------
    res_model
        ClumpplingResults-like object, with attributes:
        - modes: list of mode names
        - mode_K: dict[mode_name -> K]
        - P_aligned_by_mode: dict[mode_name -> P matrix] (not used, but available)
    df_pvs_modes
        Dict mapping mode_name -> DataFrame with columns ['sepLFC', 'sepCls'].
        Row order must align with `feature_names`.
    focal_gene
        Gene name to plot.
    feature_names
        Sequence of all feature/gene names; focal_gene must be in this list.
    plot_colors
        Sequence of colors indexed by cluster index (0-based).
    label_rank
        If True, annotate each bar with the rank of the focal gene by sepLFC.
    dpi
        Figure DPI.
    ax
        Optional existing Axes to plot into.

    Returns
    -------
    fig, ax
        Matplotlib Figure and Axes.
    """
    # index of focal gene
    try:
        i_g = feature_names.index(selected_feature)
    except ValueError:
        raise ValueError(f"selected_feature {selected_feature!r} not found in feature_names")

    modes = list(res_model.modes)
    n_modes = len(modes)

    # create fig/ax if needed
    if ax is None:
        fig, ax = plt.subplots(
            1, 1,
            figsize=(4, n_modes * 0.8),
            dpi=dpi,
        )
    else:
        fig = ax.figure

    max_val = 0.0

    for i_m, mode_name in enumerate(modes):
        # pull per-mode metrics
        df_pvs_mode = df_pvs_modes[mode_name]

        # sepLFC: 1D array over features
        sepLFC = df_pvs_mode["sepLFC"].to_numpy()
        # sepCls: array of (idx_l, idx_h) tuples
        sepCls_arr = df_pvs_mode["sepCls"].to_numpy()

        idx_l, idx_h = sepCls_arr[i_g]  # (low-group idxs, high-group idxs)

        # rank of focal gene (1-based, descending sepLFC)
        order_desc = np.argsort(-sepLFC)
        r = int(np.where(order_desc == i_g)[0][0]) + 1

        # choose which side to label (more clusters)
        idx_labeled = idx_l if len(idx_h) > len(idx_l) else idx_h
        clss_l = ",".join(str(s + 1) for s in idx_l)
        clss_h = ",".join(str(s + 1) for s in idx_h)
        label = "Cls.{}".format(
            clss_l + "(L)" if len(clss_h) > len(clss_l) else clss_h + "(H)"
        )

        val = float(sepLFC[i_g])
        bar_h = 0.4 / max(len(idx_labeled), 1)

        # colored sub-bars for each cluster in the labeled group
        for i_lb, lb in enumerate(idx_labeled):
            c = colors[lb]
            ax.barh(
                i_m - i_lb * bar_h,
                val,
                color=c,
                align="edge",
                height=-bar_h,
                lw=0,
                edgecolor="none",
                zorder=0,
            )

        # outline bar spanning full 0.4 height
        ax.barh(
            i_m,
            val,
            facecolor="none",
            align="edge",
            height=-0.4,
            lw=0.5,
            edgecolor="k",
            zorder=999,
        )

        # text label for pattern (e.g., "Cls.1,3(H)")
        ax.text(
            0.1,
            i_m + 0.05,
            label,
            va="top",
            ha="left",
            fontsize=9,
            color="k",
        )

        # rank label
        if label_rank:
            ax.text(
                val,
                i_m,
                str(r),
                color="gray",
                va="bottom",
                ha="left",
                fontsize=8,
            )

        max_val = max(max_val, val)

    # y-axis: modes
    ax.set_yticks(np.arange(n_modes))
    ax.set_yticklabels(modes)

    ax.set_xlabel("sepLFC")
    # round up x-limit nicely to nearest 10 above (max_val + 5)
    if max_val > 0:
        xmax = float(np.round((max_val + 5) / 10.0, 0) * 10.0)
    else:
        xmax = 1.0
    ax.set_xlim(0, xmax)

    ax.invert_yaxis()
    ax.set_title(f"Feature: {selected_feature}")

    fig.tight_layout()
    return fig, ax


def plot_compmodels_membership_grid(
    comp_res: CompModelsResults,
    coords,
    models: Optional[Sequence[str]] = None,
    models_plot_order: Optional[Sequence[str]] = None,
    val_threshold: float = 0.5,
    s: float = 0.05,
    colors: Optional[Sequence] = None,
    figsize_scale: Tuple[float, float] = (2.5, 2.0),
    suptitle: Optional[str] = None,
    y_suptitle: float = 0.92,
):
    """
    Plot membership on 2D coords (e.g. UMAP) for all modes in each model.

    Layout: columns = models, rows = modes within each model.

    Parameters
    ----------
    comp_res : CompModelsResults
        Loaded comparison results (from io.load_compmodels_results).
    coords : array-like
        (n_cells, 2) or (x, y) tuple; same individuals as in Q_by_mode.
    models : list of str, optional
        Subset of models to include; defaults to all in comp_res.models.
    models_plot_order : list of str, optional
        Order of columns; if None, uses `models`.
    val_threshold : float
        Membership threshold below which points are omitted for each cluster.
    s : float
        Marker size passed to plot_single_spatial_membership.
    colors : Sequence
        Sequence of colors used for clusters; default is tab20.
    figsize_scale : (float, float)
        Scale factors for figure size: (width_per_col, height_per_row).
    suptitle : str, optional
        Overall figure title.
    y_suptitle : float
        y position of suptitle.
    """
    # unpack coordinates
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
    else:
        arr = np.asarray(coords)
        x, y = arr[:, 0], arr[:, 1]

    if models is None:
        models = comp_res.models
    else:
        models = list(models)

    if models_plot_order is None:
        models_plot_order = list(models)
    else:
        # keep only those that actually exist
        models_plot_order = [m for m in models_plot_order if m in models]

    # how many rows/cols?
    n_col = len(models_plot_order)
    n_row = max(len(comp_res.modes_by_model[m]) for m in models_plot_order)

    width_per_col, height_per_row = figsize_scale
    fig = plt.figure(
        figsize=(n_col * width_per_col, n_row * height_per_row),
        dpi=300
    )
    gs = fig.add_gridspec(n_row, n_col, wspace=0.5, hspace=0.1)

    # choose a colormap for clusters
    if colors is None:
        colors = cm.get_cmap("tab20").colors

    axes_by_model_mode = {}

    # Determine global max K to normalize colors
    K_max = 0
    for full_name, Q in comp_res.Q_by_mode.items():
        K_max = max(K_max, Q.shape[1])
    K_max = max(K_max, 1)

    for j_model, model_name in enumerate(models_plot_order):
        short_modes = comp_res.modes_by_model[model_name]

        for i_mode, short_mode in enumerate(short_modes):
            row = i_mode
            col = j_model
            ax = fig.add_subplot(gs[row, col])

            full_mode_name = f"{model_name}_{short_mode}"
            if full_mode_name not in comp_res.Q_by_mode:
                ax.set_axis_off()
                continue

            Q = comp_res.Q_by_mode[full_mode_name]
            K = Q.shape[1]

            # overlay all clusters for this (model, mode)
            for i_k,k in enumerate(range(K)):
                plot_single_spatial_membership(
                    Q,
                    (x, y),
                    ref_color=colors[i_k],
                    cls_idx=k,
                    ax=ax,
                    keep_ticks=False,
                    val_threshold=val_threshold,
                    s=s,
                    alpha=0.6,
                    vmin=0.0,
                    vmax=1.0,
                )

            # title uses size from stats if available
            size_text = ""
            stats_df = comp_res.mode_stats_by_model.get(model_name)
            if stats_df is not None and short_mode in stats_df.index:
                size_val = stats_df.loc[short_mode]["Size"]
                size_text = f"(size={size_val})"
            ax.set_ylabel(f"{short_mode}\n{size_text}", fontsize=9, rotation=0, ha='right', va='center')

            if row == 0:
                ax.set_title(
                    f"{model_name}",
                    loc='left',
                    weight="bold",
                )
            else:
                ax.set_title("")

            ax.set_xticks([])
            ax.set_yticks([])
            axes_by_model_mode[(model_name, short_mode)] = ax

    if suptitle is None:
        suptitle = f"Membership patterns across models (membership > {val_threshold})"
    fig.suptitle(
        suptitle,
        y=y_suptitle,
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()

    return fig, axes_by_model_mode


def plot_compmodels_membership_selected(
    comp_res,
    coords,
    model_mode_list: Sequence[Tuple[str, str]],
    *,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    val_threshold: float = 0.5,
    s: float = 0.05,
    colors: Optional[Sequence] = None,
    figsize_scale: Tuple[float, float] = (2.5, 2.0),
    suptitle: Optional[str] = None,
    y_suptitle: float = 0.92,
):
    """
    Plot membership on 2D coords (e.g. UMAP) for a selected set of modes.

    Layout: one panel per (model, mode) in model_mode_list.
    Grid size can be specified by n_rows / n_cols; otherwise defaults to a single row.

    Parameters
    ----------
    comp_res : CompModelsResults
        Loaded comparison results (from io.load_compmodels_results).
        Must have attributes:
          - Q_by_mode : dict[full_mode_name -> ndarray (n_cells, K)]
          - mode_stats_by_model : dict[model_name -> DataFrame]
            with index 'Mode' (short_mode) and column 'Size'
    coords : array-like
        (n_cells, 2) or (x, y) tuple; same individuals as in Q_by_mode.
    model_mode_list : sequence of (model_name, short_mode)
        List of specific modes to plot, e.g.
            [("rna.seurat.louvain", "K20M1"),
             ("rna.seurat.louvain", "K20M2"),
             ("rna.scanpy.leiden", "K18M1")]
    n_rows : int, optional
        Number of rows in the grid. If None and n_cols is None, uses 1 row.
    n_cols : int, optional
        Number of columns in the grid. If None and n_rows is None,
        uses len(model_mode_list) columns (single row).
    val_threshold : float
        Membership threshold below which points are omitted for each cluster.
    s : float
        Marker size passed to plot_single_spatial_membership.
    colors : Sequence, optional
        Sequence of colors used for clusters; default is tab20.
    figsize_scale : (float, float)
        Scale factors for figure size: (width_per_col, height_per_row).
    suptitle : str, optional
        Overall figure title.
    y_suptitle : float
        y position of suptitle.

    Returns
    -------
    fig, axes_by_model_mode
        fig : matplotlib.figure.Figure
        axes_by_model_mode : dict[(model_name, short_mode) -> Axes]
    """
    # ---- unpack coordinates ----
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
        x, y = np.asarray(x), np.asarray(y)
    else:
        arr = np.asarray(coords)
        x, y = arr[:, 0], arr[:, 1]

    model_mode_list = list(model_mode_list)
    n_panels = len(model_mode_list)
    if n_panels == 0:
        raise ValueError("model_mode_list is empty; nothing to plot.")

    # ---- determine grid shape ----
    if n_rows is None and n_cols is None:
        n_rows = 1
        n_cols = n_panels
    elif n_rows is None and n_cols is not None:
        n_rows = int(np.ceil(n_panels / n_cols))
    elif n_rows is not None and n_cols is None:
        n_cols = int(np.ceil(n_panels / n_rows))

    # safety
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive.")

    width_per_col, height_per_row = figsize_scale
    fig = plt.figure(
        figsize=(n_cols * width_per_col, n_rows * height_per_row),
        dpi=300,
    )
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.5, hspace=0.2)

    # ---- choose cluster colors ----
    if colors is None:
        colors = cm.get_cmap("tab20").colors
    colors = list(colors)

    axes_by_model_mode: Dict[Tuple[str, str], plt.Axes] = {}

    # convenience: get stats one time
    stats_by_model = comp_res.mode_stats_by_model

    for idx, (model_name, short_mode) in enumerate(model_mode_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        full_mode_name = f"{model_name}_{short_mode}"
        if full_mode_name not in comp_res.Q_by_mode:
            ax.set_axis_off()
            continue

        Q = comp_res.Q_by_mode[full_mode_name]
        K = Q.shape[1]

        # overlay all clusters for this (model, mode)
        for k in range(K):
            color_k = colors[k % len(colors)]
            plot_single_spatial_membership(
                Q,
                (x, y),
                ref_color=color_k,
                cls_idx=k,
                ax=ax,
                keep_ticks=False,
                val_threshold=val_threshold,
                s=s,
                alpha=0.6,
                vmin=0.0,
                vmax=1.0,
            )

        # Title + size annotation
        size_text = ""
        stats_df = stats_by_model.get(model_name)
        if stats_df is not None:
            if short_mode in stats_df.index and "Size" in stats_df.columns:
                size_val = stats_df.loc[short_mode]["Size"]
                size_text = f"(size={size_val})"

        ylabel = f"{short_mode}\n{size_text}"
        ax.set_ylabel(
            ylabel,
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
        )

        ax.set_title(model_name, fontsize=10, weight="bold", loc="left")
        # ax.set_title("", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        axes_by_model_mode[(model_name, short_mode)] = ax

    # turn off any unused panels if grid is larger than n_panels
    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_axis_off()

    if suptitle is None:
        suptitle = f"Selected membership patterns (membership > {val_threshold})"
    fig.suptitle(
        suptitle,
        y=y_suptitle,
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()

    return fig, axes_by_model_mode


def plot_compmodels_diff_grid_against_ref(
    comp_res,
    pair_mappings: Dict[str, Sequence[Tuple[int, int]]],
    coords,
    ref_mode: str,
    models_plot_order: Optional[Sequence[str]] = None,
    val_threshold: float = 0.5,
    diff_threshold: float = 0.5,
    *,
    colors: Optional[Sequence] = None,
    s: float = 0.05,
    alpha: float = 0.6,
    figsize_scale: Tuple[float, float] = (2.5, 2.0),
    suptitle: Optional[str] = None,
    y_suptitle: float = 0.92,
    strict_pair_mapping: bool = True
):
    """
    Plot difference in membership on 2D coords for all modes across models.

    - Use map_alt_to_ref to compute aligned differences.
    - For non-ref panels, plot a single overlaid diff scatter (per-cell).
    - Compute Δ = fraction(per_cell_diff > diff_threshold)

    Parameters
    ----------
    comp_res : CompModelsResults
        Loaded comparison results (from io.load_compmodels_results).
    pair_mappings : dict
        Dict mapping "ref_mode-alt_mode" -> list of (ref_k, alt_k) tuples.
    coords : array-like     
        (n_cells, 2) or (x, y) tuple; same individuals as in Q_by_mode.
    ref_mode : str
        Full mode name (e.g. "model_shortmode") to use as reference.
    models_plot_order : list of str, optional
        Order of models (columns); if None, uses all models in comp_res.
    val_threshold : float
        Membership threshold below which points are omitted for each cluster.
    diff_threshold : float
        Threshold for difference in membership to consider significant.
    colors : Sequence, optional
        Sequence of colors used for clusters; default is tab20.
    s : float
        Marker size passed to plot_single_spatial_membership.   
    alpha : float
        Alpha value for scatter points. 
    figsize_scale : (float, float)
        Scale factors for figure size: (width_per_col, height_per_row).
    suptitle : str, optional
        Overall figure title.
    y_suptitle : float
        y position of suptitle.
    strict_pair_mapping : bool
        If True, raise an error if a required pair mapping is missing.
    """
    # coords -> (x, y)
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
        xy = (np.asarray(x), np.asarray(y))
    else:
        arr = np.asarray(coords)
        xy = (arr[:, 0], arr[:, 1])

    if colors is None:
        colors = cm.get_cmap("tab20").colors

    # get model->modes mapping 
    modes_by_model = getattr(comp_res, "modes_by_model", None)
    if modes_by_model is None:
        raise AttributeError("comp_res must have `modes_by_model`")

    all_models = list(modes_by_model.keys())
    if models_plot_order is None:
        models_plot_order = all_models

    # reference Q
    ref_Q = comp_res.get_Q(ref_mode)
    ref_K = int(ref_Q.shape[1])
    n_cells = int(ref_Q.shape[0])

    # determine grid size
    n_col = len(models_plot_order)
    n_row = max([len(modes_by_model[m]) for m in models_plot_order]) if n_col else 0

    width_per_col, height_per_row = figsize_scale
    fig = plt.figure(figsize=(n_col * width_per_col, n_row * height_per_row), dpi=300)
    gs = fig.add_gridspec(n_row, n_col, wspace=0.5, hspace=0.1)

    axes_by_model_mode = {}

    # prebuild mode size lookups (per model)
    size_lookup: Dict[str, Dict[str, Optional[float]]] = {}
    for model in models_plot_order:
        stats_df = comp_res.mode_stats_by_model.get(model)
        d: Dict[str, Optional[float]] = {}
        if stats_df is not None:
            if "Mode" in stats_df.columns:
                try:
                    sub = stats_df.set_index("Mode")
                    if "Size" in sub.columns:
                        d.update(sub["Size"].to_dict())
                except Exception:
                    pass
            else:
                try:
                    if "Size" in stats_df.columns:
                        d.update(stats_df["Size"].to_dict())
                except Exception:
                    pass
        size_lookup[model] = d

    def _to_full_mode(model: str, mode_entry: str) -> str:
        return mode_entry if str(mode_entry).startswith(model + "_") else f"{model}_{mode_entry}"

    def _short_mode(model: str, full_name: str) -> str:
        prefix = model + "_"
        return full_name[len(prefix):] if full_name.startswith(prefix) else full_name

    # main plotting
    for i_model, model_name in enumerate(models_plot_order):
        mode_entries = list(modes_by_model[model_name])

        for i_mode, mode_entry in enumerate(mode_entries):
            if i_model == 0 and i_mode == 0:
                ax = fig.add_subplot(gs[i_mode, i_model])
                ref_ax = ax
            else:
                ax = fig.add_subplot(gs[i_mode, i_model], sharex=ref_ax, sharey=ref_ax)

            full_name = _to_full_mode(model_name, str(mode_entry))
            short_mode = _short_mode(model_name, full_name)

            Q = comp_res.get_Q(full_name)
            K_cur = int(Q.shape[1])

            mode_size = size_lookup.get(model_name, {}).get(short_mode, None)
            size_text = f"(size:{mode_size})" if mode_size is not None else "(size:NA)"

            if full_name == ref_mode:
                # --- plot reference membership as-is (overlay all clusters) ---
                for i_k in range(K_cur):
                    plot_single_spatial_membership(
                        Q,
                        xy,
                        colors[i_k],
                        cls_idx=i_k,
                        ax=ax,
                        keep_ticks=False,
                        val_threshold=val_threshold,
                        s=s,
                        alpha=alpha,
                        vmin=0.0,
                        vmax=1.0,
                    )

                ax.set_ylabel(
                    f"{short_mode}\n{size_text}\n[ref]",
                    fontsize=10,
                    rotation=0,
                    ha="right",
                    va="center",
                    weight="bold",
                    color="red",
                )

            else:
                # compute aligned diff using map_alt_to_ref
                # per_cell_diff = None

                if ref_K <= K_cur:
                    key = f"{ref_mode}-{full_name}"
                    pair_mapping = pair_mappings.get(key)
                    if pair_mapping is None:
                        if strict_pair_mapping:
                            raise KeyError(f"Missing pair mapping key: {key}")
                        overall_diff = np.zeros(n_cells, dtype=float)
                    else:
                        Q_mapped, diff_Q = map_alt_to_ref(ref_Q, Q, pair_mapping)
                        # compute difference
                        overall_diff = compute_overall_membership_difference(diff_Q)

                else:
                    # current has smaller K; map ref into current space
                    key = f"{full_name}-{ref_mode}"
                    pair_mapping = pair_mappings.get(key)
                    if pair_mapping is None:
                        if strict_pair_mapping:
                            raise KeyError(f"Missing pair mapping key: {key}")
                        overall_diff = np.zeros(n_cells, dtype=float)
                    else:
                        ref_mapped, diff_Q = map_alt_to_ref(Q, ref_Q, pair_mapping)
                        # compute difference
                        overall_diff = compute_overall_membership_difference(diff_Q)
                        Q_mapped = Q

                # thresholded single-layer diff overlay
                for i_k in range(diff_Q.shape[1]):

                    diff_Q_col = diff_Q[:, i_k]
                    if K_cur <= ref_K:
                        plot_mask = (Q[:, i_k] > val_threshold) & (diff_Q_col > diff_threshold)
                        plot_single_spatial_membership(
                            Q[:, i_k][plot_mask],
                            (xy[0][plot_mask], xy[1][plot_mask]),
                            cls_idx=0,
                            ax=ax,
                            title="",
                            ref_color=colors[i_k],
                            s=s,
                        )
                    else: # ref_K < K_cur
                        # get the original Q values for plotting
                        col_indices = []
                        for i_k_ref, i_kk in pair_mapping:  
                            if i_k_ref == i_k:
                                col_indices.append(i_kk)
                        if not col_indices:
                            continue
                        for i_kk in col_indices:
                            plot_mask = (Q[:, i_kk] > val_threshold) & (diff_Q_col > diff_threshold)
                            plot_single_spatial_membership(
                                Q[:, i_kk][plot_mask],
                                (xy[0][plot_mask], xy[1][plot_mask]),
                                cls_idx=0,
                                ax=ax,
                                title="",
                                ref_color=colors[i_kk],
                                s=s,
                            )

                    ax.set_ylabel(
                        f"{short_mode}\n{size_text}\nΔ={overall_diff:.2f}",
                        fontsize=10,
                        rotation=0,
                        ha="right",
                        va="center",
                    )

            if i_mode == 0:
                ax.set_title(model_name, fontsize=12, weight="bold", loc="left")

            ax.set_xticks([])
            ax.set_yticks([])

            axes_by_model_mode[(model_name, short_mode)] = ax

    if suptitle is None:
        suptitle = f"Difference in results across models (Δ>{diff_threshold})"

    fig.suptitle(suptitle, y=y_suptitle, fontsize=14, weight="bold")
    fig.tight_layout()

    return fig, axes_by_model_mode


def plot_compmodels_diff_selected_against_ref(
    comp_res,
    pair_mappings: Dict[str, Sequence[Tuple[int, int]]],
    coords,
    ref_mode: str,
    model_mode_list: Sequence[Tuple[str, str]],
    *,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    val_threshold: float = 0.5,
    diff_threshold: float = 0.5,
    colors: Optional[Sequence] = None,
    s: float = 0.05,
    alpha: float = 0.6,
    figsize_scale: Tuple[float, float] = (2.5, 2.0),
    suptitle: Optional[str] = None,
    y_suptitle: float = 0.92,
    strict_pair_mapping: bool = True,
):
    """
    Plot difference in membership on 2D coords for a selected set of modes.
    Layout: one panel per (model, mode) in model_mode_list.
    Grid size can be specified by n_rows / n_cols; otherwise defaults to a single row.
    Parameters follow same pattern as 'plot_compmodels_diff_grid_against_ref'.
    """
    # coords -> (x, y)
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
        xy = (np.asarray(x), np.asarray(y))
    else:
        arr = np.asarray(coords)
        xy = (arr[:, 0], arr[:, 1])

    if colors is None:
        colors = cm.get_cmap("tab20").colors
    colors = list(colors)

    modes_by_model = getattr(comp_res, "modes_by_model", None)
    if modes_by_model is None:
        raise AttributeError("comp_res must have `modes_by_model`")

    model_mode_list = list(model_mode_list)
    n_panels = len(model_mode_list)
    if n_panels == 0:
        raise ValueError("model_mode_list is empty; nothing to plot.")

    # reference Q
    ref_Q = comp_res.get_Q(ref_mode)
    ref_K = int(ref_Q.shape[1])
    n_cells = int(ref_Q.shape[0])

    # grid shape
    if n_rows is None and n_cols is None:
        n_rows = 1
        n_cols = n_panels
    elif n_rows is None and n_cols is not None:
        n_rows = int(np.ceil(n_panels / n_cols))
    elif n_rows is not None and n_cols is None:
        n_cols = int(np.ceil(n_panels / n_rows))
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive.")

    width_per_col, height_per_row = figsize_scale
    fig = plt.figure(
        figsize=(n_cols * width_per_col, n_rows * height_per_row),
        dpi=300,
    )
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.5, hspace=0.2)

    axes_by_model_mode: Dict[Tuple[str, str], plt.Axes] = {}

    # prebuild mode size lookups (per model)
    size_lookup: Dict[str, Dict[str, Optional[float]]] = {}
    for model in set(m for m, _ in model_mode_list):
        stats_df = comp_res.mode_stats_by_model.get(model)
        d: Dict[str, Optional[float]] = {}
        if stats_df is not None:
            if "Mode" in stats_df.columns:
                try:
                    sub = stats_df.set_index("Mode")
                    if "Size" in sub.columns:
                        d.update(sub["Size"].to_dict())
                except Exception:
                    pass
            else:
                try:
                    if "Size" in stats_df.columns:
                        d.update(stats_df["Size"].to_dict())
                except Exception:
                    pass
        size_lookup[model] = d

    def _to_full_mode(model: str, mode_entry: str) -> str:
        return mode_entry if str(mode_entry).startswith(model + "_") else f"{model}_{mode_entry}"

    def _short_mode(model: str, full_name: str) -> str:
        prefix = model + "_"
        return full_name[len(prefix):] if full_name.startswith(prefix) else full_name

    ref_ax = None

    for idx, (model_name, short_mode_in) in enumerate(model_mode_list):
        row = idx // n_cols
        col = idx % n_cols

        if ref_ax is None:
            ax = fig.add_subplot(gs[row, col])
            ref_ax = ax
        else:
            ax = fig.add_subplot(gs[row, col], sharex=ref_ax, sharey=ref_ax)

        full_name = _to_full_mode(model_name, str(short_mode_in))
        short_mode = _short_mode(model_name, full_name)

        Q = comp_res.get_Q(full_name)
        K_cur = int(Q.shape[1])

        mode_size = size_lookup.get(model_name, {}).get(short_mode, None)
        size_text = f"(size:{mode_size})" if mode_size is not None else "(size:NA)"

        # reference panel
        if full_name == ref_mode:
            for i_k in range(K_cur):
                plot_single_spatial_membership(
                    Q,
                    xy,
                    colors[i_k % len(colors)],
                    cls_idx=i_k,
                    ax=ax,
                    keep_ticks=False,
                    val_threshold=val_threshold,
                    s=s,
                    alpha=alpha,
                    vmin=0.0,
                    vmax=1.0,
                )

            ax.set_ylabel(
                f"{short_mode}\n{size_text}\n[ref]",
                fontsize=9,
                rotation=0,
                ha="right",
                va="center",
                weight="bold",
                color="red",
            )

        # non-ref: compute diff_Q once, reuse pair_mapping
        else:
            if ref_K <= K_cur:
                # ref has <= topics; map ref -> current space
                key = f"{ref_mode}-{full_name}"
                pair_mapping = pair_mappings.get(key)
                if pair_mapping is None:
                    if strict_pair_mapping:
                        raise KeyError(f"Missing pair mapping key: {key}")
                    overall_diff = np.zeros(n_cells, dtype=float)
                    diff_Q = np.zeros((n_cells, ref_K), dtype=float)
                else:
                    Q_mapped, diff_Q = map_alt_to_ref(ref_Q, Q, pair_mapping)
                    overall_diff = compute_overall_membership_difference(diff_Q)
            else:
                # ref has more topics; map current -> ref space
                key = f"{full_name}-{ref_mode}"
                pair_mapping = pair_mappings.get(key)
                if pair_mapping is None:
                    if strict_pair_mapping:
                        raise KeyError(f"Missing pair mapping key: {key}")
                    overall_diff = np.zeros(n_cells, dtype=float)
                    diff_Q = np.zeros((n_cells, K_cur), dtype=float)
                else:
                    ref_mapped, diff_Q = map_alt_to_ref(Q, ref_Q, pair_mapping)
                    overall_diff = compute_overall_membership_difference(diff_Q)

            # thresholded per-topic overlays, using the SAME pair_mapping
            for i_k in range(diff_Q.shape[1]):
                diff_Q_col = diff_Q[:, i_k]

                if K_cur <= ref_K:
                    # case: K_cur <= ref_K  (includes ref_K > K_cur and equality)
                    plot_mask = (Q[:, i_k] > val_threshold) & (diff_Q_col > diff_threshold)
                    if np.any(plot_mask):
                        plot_single_spatial_membership(
                            Q[:, i_k][plot_mask],
                            (xy[0][plot_mask], xy[1][plot_mask]),
                            cls_idx=0,
                            ax=ax,
                            title="",
                            ref_color=colors[i_k % len(colors)],
                            s=s,
                        )
                else:
                    # case: ref_K < K_cur – need to know which Q columns map to ref i_k
                    # IMPORTANT: reuse pair_mapping computed above (ref_mode-full_name)
                    col_indices = [
                        i_kk for i_k_ref, i_kk in pair_mapping if i_k_ref == i_k
                    ]
                    if not col_indices:
                        continue
                    for i_kk in col_indices:
                        plot_mask = (Q[:, i_kk] > val_threshold) & (diff_Q_col > diff_threshold)
                        if np.any(plot_mask):
                            plot_single_spatial_membership(
                                Q[:, i_kk][plot_mask],
                                (xy[0][plot_mask], xy[1][plot_mask]),
                                cls_idx=0,
                                ax=ax,
                                title="",
                                ref_color=colors[i_kk % len(colors)],
                                s=s,
                            )

            ax.set_ylabel(
                f"{short_mode}\n{size_text}\nΔ={overall_diff:.2f}",
                fontsize=9,
                rotation=0,
                ha="right",
                va="center",
            )
        ax.set_title(model_name, fontsize=10, weight="bold", loc="left")

        ax.set_xticks([])
        ax.set_yticks([])

        axes_by_model_mode[(model_name, short_mode)] = ax

    # turn off any unused cells if grid > n_panels
    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_axis_off()

    if suptitle is None:
        suptitle = f"Selected differences vs ref (Δ>{diff_threshold})"

    fig.suptitle(suptitle, y=y_suptitle, fontsize=14, weight="bold")
    fig.tight_layout()

    return fig, axes_by_model_mode


def plot_multimodel_major_and_weighted_diff(
    comp_res,
    mat_diffs,
    coords,
    models_plot_order=None,
    *,
    colors=None,                 
    figsize_scale=(2.5, 2.5),
    diff_cmap="RdPu",
    diff_vmin=0.0,
    diff_vmax=1.0,
    point_size=2.0,
    alpha=1.0,
    suptitle=None,
):
    """
    For each model, plot:
      - Top row: major mode clustering (largest Size) on 2D coords,
                 colored by discrete cluster labels using `colors`,
                 analogous to `plot_compmodels_membership_grid`.
      - Bottom row: per-cell weighted average difference vs reference,
                    aggregated across modes and weighted by mode size.

    Parameters
    ----------
    comp_res
        Object containing compModels results. Must have:
          - modes_by_model : Dict[str, List[str]]   (short mode names, e.g. "K20M1")
          - mode_stats_by_model : Dict[str, DataFrame] with columns ['Mode', 'Size']
          - get_Q(full_mode_name) -> np.ndarray (n_cells x K)
    mat_diffs : dict
        Nested dict of diff matrices, typically from
        get_compmodels_diff_matrices_against_ref:
          mat_diffs[model_name][short_mode] = diff_Q
        where diff_Q has shape (n_cells, K_eff).
    coords : array-like or (x, y)
        2D coordinates per cell. Either:
          - array of shape (n_cells, 2), or
          - tuple/list (x, y) of 1D arrays.
    models_plot_order : sequence of str, optional
        Order of models (columns). Defaults to list(mat_diffs.keys()).
    colors : sequence, optional
        Sequence of discrete colors used for clusters in the TOP row,
        same semantics as in plot_compmodels_membership_grid.
        If None, defaults to tab20 colors.
    figsize_scale : (float, float), default (2.5, 2.5)
        (width_per_model, height_per_row) used to derive overall figure size.
    diff_cmap : str, default "RdPu"
        Colormap for the weighted difference panel (bottom row).
    diff_vmin, diff_vmax : float
        vmin/vmax for the difference colormap.
    point_size : float
        Scatter point size.
    alpha : float
        Scatter alpha.
    suptitle : str or None
        Optional figure-level title.

    Returns
    -------
    fig, axes
        Matplotlib Figure and Axes array of shape (2, n_models).
    """

    # coords -> (x, y) 
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = np.asarray(coords[0]), np.asarray(coords[1])
    else:
        arr = np.asarray(coords)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("coords must be (n_cells, 2) or (x, y).")
        x, y = arr[:, 0], arr[:, 1]

    # models
    if models_plot_order is None:
        models_plot_order = list(mat_diffs.keys())

    n_models = len(models_plot_order)
    if n_models == 0:
        raise ValueError("No models to plot.")

    # figure / axes 
    width_per_model, height_per_row = figsize_scale
    fig_width = width_per_model * n_models
    fig_height = height_per_row * 2  # 2 rows
    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(fig_width, fig_height),
        dpi=300,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # colors for TOP row clusters (like plot_compmodels_membership_grid) 
    if colors is None:
        colors = cm.get_cmap("tab20").colors
    colors = np.array(list(colors))
    n_colors = len(colors)

    # Helper to build full mode name if needed
    def _to_full_mode(model: str, mode_entry: str) -> str:
        return mode_entry if str(mode_entry).startswith(model + "_") else f"{model}_{mode_entry}"

    # Pre-build size lookups
    size_lookup = {}
    for model in models_plot_order:
        stats_df = comp_res.mode_stats_by_model.get(model)
        stats_df = stats_df.reset_index()
        if stats_df is None:
            raise KeyError(f"Missing mode_stats_by_model entry for model: {model}")
        if "Mode" not in stats_df.columns or "Size" not in stats_df.columns:
            raise ValueError(
                f"mode_stats_by_model[{model}] must have columns 'Mode' and 'Size'."
            )
        size_lookup[model] = stats_df.set_index("Mode")["Size"]

    # We'll keep a handle to the last diff scatter for the colorbar
    last_scatter = None

    for j, model_name in enumerate(models_plot_order):
        stats_series = size_lookup[model_name]

        # Top row: major mode clustering, colored with `colors`
        ax_top = axes[0, j]

        # Short mode name with largest Size
        major_short_mode = stats_series.idxmax()
        major_size = int(stats_series.loc[major_short_mode])

        major_full_mode = _to_full_mode(model_name, major_short_mode)
        Q_major = comp_res.get_Q(major_full_mode)  # (n_cells x K_major)
        n_cells, K_major = Q_major.shape

        # Hard assignment
        lbs_major = np.asarray(Q_major).argmax(axis=1)  # 0..K_major-1
        n_cls_major = int(lbs_major.max()) + 1 if lbs_major.size > 0 else 0

        # Map each label to a color from the provided palette, cycling if needed
        cell_colors = colors[lbs_major % n_colors]

        ax_top.scatter(
            x,
            y,
            c=cell_colors,
            s=point_size,
            alpha=alpha,
        )

        tit = model_name.replace("_", " ").title()
        title_top = f"{tit}\n$Major\\;Mode\\;(K={n_cls_major},\\, size={major_size})$"
        ax_top.set_title(title_top, fontsize=11, weight="bold", loc="left")
        ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_top.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        # Bottom row: weighted average difference vs ref
        ax_bot = axes[1, j]

        diff_weighted_sum = np.zeros(n_cells, dtype=float)
        total_weight = 0.0

        # Loop over all modes for this model
        modes_for_model = list(mat_diffs[model_name].keys())
        for short_mode in modes_for_model:
            if short_mode not in stats_series.index:
                # If diff exists but no size info, skip.
                continue

            size = float(stats_series.loc[short_mode])
            diff_Q = mat_diffs[model_name][short_mode]  # (n_cells x K_eff)

            # Per-cell scalar difference; here L1/2 gives [0,1] if diff_Q
            # is a difference of probability vectors.
            per_cell_diff = np.abs(diff_Q).sum(axis=1) / 2.0

            diff_weighted_sum += size * per_cell_diff
            total_weight += size

        if total_weight <= 0:
            weighted_avg_diff = np.zeros(n_cells, dtype=float)
        else:
            weighted_avg_diff = diff_weighted_sum / total_weight

        avg_diff = float(weighted_avg_diff.mean())

        ax_bot.set_facecolor("lightgray")
        sc = ax_bot.scatter(
            x,
            y,
            c=weighted_avg_diff,
            vmin=diff_vmin,
            vmax=diff_vmax,
            s=point_size,
            alpha=alpha,
            cmap=diff_cmap,
        )
        last_scatter = sc

        ax_bot.set_title(f"Average $\\Delta={avg_diff:.3f}$", fontsize=11, loc="left")
        ax_bot.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_bot.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    # ---- colorbar for the bottom row ----
    if last_scatter is not None:
        cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.33])
        fig.colorbar(
            last_scatter,
            cax=cbar_ax,
            label="Weighted average difference\nvs reference (per cell)",
        )

    if suptitle is not None:
        fig.suptitle(suptitle, y=0.99, fontsize=14, weight="bold")

    fig.subplots_adjust(wspace=0.1)

    return fig, axes


def _parse_short_mode_km(short_mode: str) -> Tuple[int, int]:
    """
    Parse 'K14M2' -> (14, 2). Fallback returns (-1, -1) if not matched.
    """
    m = re.match(r"K(\d+)M(\d+)", short_mode)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def _split_full_mode(full_mode: str) -> Tuple[str, str]:
    """
    Full mode names in compModels are 'model_shortMode'.
    Model names should not contain '_' by design.
    """
    if "_" not in full_mode:
        return full_mode, ""
    model, short = full_mode.split("_", 1)
    return model, short


def _build_K_grouped_full_names(
    full_mode_names: Sequence[str],
    models_plot_order: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Reconstruct compModels-style order:
    group by K, sort K asc, then sort within K by model order, then by M.
    """
    # group full names by K
    by_K: Dict[int, List[str]] = defaultdict(list)
    unknown: List[str] = []

    for name in full_mode_names:
        model, short = _split_full_mode(name)
        K, M = _parse_short_mode_km(short)
        if K < 0:
            unknown.append(name)
        else:
            by_K[K].append(name)

    Ks_sorted = sorted(by_K.keys())

    # model order index for stable sorting within each K
    model_rank = {}
    if models_plot_order is not None:
        model_rank = {m: i for i, m in enumerate(models_plot_order)}

    def _within_K_sort_key(full_name: str):
        model, short = _split_full_mode(full_name)
        K, M = _parse_short_mode_km(short)
        # models not in order list go to the end but keep alphabetical stable
        mr = model_rank.get(model, 10**6)
        return (mr, model, M, short)

    ordered = []
    for K in Ks_sorted:
        names = sorted(by_K[K], key=_within_K_sort_key)
        ordered.extend(names)

    # tack on unknowns at the end (rare)
    ordered.extend(sorted(unknown))

    return ordered


def plot_compmodels_alignment_list(
    comp_res,
    cmap=None,
    marker_size: int = 250,
    figsize: Tuple[float, float] = (6, 6),
):
    """
    CompModels alignment pattern list using clumppling.plot_alignment_list,
    but with correct K-grouped ordering to avoid KeyError.

    Requires comp_res to have:
      - full_mode_names
      - alignment_across_all
      - all_modes_alignment
      - get_Q(full_mode_name)
    """
    if cmap is None:
        cmap = cm.get_cmap("tab20")

    # Build correct order (K-grouped, like separate_Qs_by_K)
    Q_names_reordered = _build_K_grouped_full_names(
        comp_res.full_mode_names,
    )

    # Compute mode_K in this same order
    Q_aligned_list = [comp_res.get_Q(name) for name in Q_names_reordered]
    mode_K = [Q.shape[1] for Q in Q_aligned_list]

    # Plot
    fig = plot_alignment_list(
        mode_K,
        Q_names_reordered,
        cmap,
        comp_res.alignment_across_all,
        comp_res.all_modes_alignment,
        marker_size=marker_size,
    )

    fig.set_size_inches(*figsize)
    if fig.axes:
        fig.axes[0].set_ylabel("Replicates")

    return fig, Q_names_reordered, mode_K


def _split_full_mode(full_name: str) -> Tuple[str, str]:
    """
    Split a full mode name 'model_KxMy' into (model, 'KxMy').
    """
    parts = full_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot split full mode name '{full_name}' into model/mode")
    return parts[0], parts[1]


def plot_compmodels_alignment_by_model(
    comp_res,
    cmap=None,
    *,
    models: Optional[Sequence[str]] = None,
    models_plot_order: Optional[Sequence[str]] = None,
    row_by_K: bool = False,
    wspace_padding: float = 1.3,
    marker_size: float = 200.0,
    alt_ls: bool = False,
    ls_alt: Sequence[str] = ("-", "--"),
    lw: float = 1.0,
    connect_identity: bool = False,
    adjacent_only: bool = True,
    label_modes: bool = True,
    figsize_scale: Tuple[float, float] = (0.3, 2),
    dpi: int = 150,
    pair_mappings: Optional[Dict[str, Sequence[Tuple[int, int]]]] = None,
):
    """
    Plot alignment between multiple models in a single graph.

    Modes can be arranged in rows either by:
      - mode index within each model (row_by_K=False), or
      - grouped by K across models (row_by_K=True), so that modes with the
        same K value line up on the same row "band" across models.

    When row_by_K=True:
      - For each K, determine the maximum number of modes with that K across
        all selected models.
      - Allocate that many rows for that K.
      - If a model has fewer modes for that K, the corresponding slots are left
        empty (no markers drawn).

    Parameters
    ----------
    comp_res : CompModelsResults
        Must provide:
          - models
          - modes_by_model: dict[model -> list[str]]  (short mode names)
          - full_mode_names: list[str]                (e.g. "rna.seurat.K21M1")
          - all_modes_alignment: dict[full_mode_name -> list[int]]
          - alignment_across_all: dict["A-B" -> mapping list[int]]
          - K_max: int
          - get_Q(full_mode_name) or Q_by_mode[full_mode_name]
    cmap
        Either:
          - a matplotlib colormap (e.g. cm.get_cmap("tab20"))
          - a sequence of RGB tuples
          - None (defaults to tab20).
    models : sequence of str, optional
        Subset of models to include. Defaults to all comp_res.models.
    models_plot_order : sequence of str, optional
        Order of columns. Defaults to `models`.
    row_by_K : bool
        If True, modes are grouped by K across models; only modes with the same
        K appear in the same row band. If False, rows are mode index per model.
    wspace_padding : float
        Horizontal spacing factor between model columns, scaled by K_max.
    marker_size : float
        Size of the cluster markers.
    alt_ls : bool
        If True, use `ls_alt` to style edges for better visibility.
    ls_alt : sequence of str
        Line styles; ls_alt[0] used for non-identity edges,
        ls_alt[1] used for identity edges (if connect_identity=True).
    lw : float
        Line width for edges.
    connect_identity : bool
        If True, also draw thin light-grey (or ls_alt[1]) lines for identity
        mappings (same aligned column index). If False, only draw non-identity.
    adjacent_only : bool
        If True, draw edges only between modes in *adjacent model columns*
        (to reduce clutter). If False, draw edges between any model pair.
    label_modes : bool
        If True, write mode labels near each block; column headers = models.
    figsize_scale : (float, float)
        Scale factors for figure size: (width_per_K, height_per_row).
        Width = n_models * K_max * width_per_K
        Height = n_rows * height_per_row
    dpi : int
        Figure dpi.
    pair_mappings : dict, optional
        Optional within-model pair mappings ("A-B" -> list[(col_idx_A, col_idx_B)])
        to draw extra edges between successive modes of the same model.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes.
    """
    if cmap is None:
        cmap = cm.get_cmap("tab20").colors

    # --- select models ---
    if models is None:
        models = list(comp_res.models)
    else:
        models = list(models)

    if models_plot_order is None:
        models_plot_order = list(models)

    modes_by_model = comp_res.modes_by_model  # dict[model -> list[short_modes]]
    n_col = len(models_plot_order)
    K_max = comp_res.K_max

    # Helper: get K for a full mode name
    def _get_K(full_name: str) -> int:
        if hasattr(comp_res, "get_Q"):
            Q = comp_res.get_Q(full_name)
        else:
            Q = comp_res.Q_by_mode[full_name]
        return Q.shape[1]

    # --- Decide row layout and mode positions ---
    mode_pos: Dict[str, Tuple[int, int]] = {}  # full_mode_name -> (row, col)
    row_meta: Sequence[Tuple[int, int]] = []   # for row_by_K: list of (K, slot_index)

    if row_by_K:
        # Group modes by K for each model
        from collections import defaultdict

        modes_by_model_and_K: Dict[str, Dict[int, list]] = {
            m: defaultdict(list) for m in models_plot_order
        }
        Ks_set = set()

        for model in models_plot_order:
            short_modes = modes_by_model.get(model, [])
            for short_mode in short_modes:
                full_name = f"{model}_{short_mode}"
                K = _get_K(full_name)
                modes_by_model_and_K[model][K].append(short_mode)
                Ks_set.add(K)

        Ks_sorted = sorted(Ks_set)

        row_meta_list = []
        row_idx = 0

        # For each K, allocate rows = max #modes(K) across models
        for K in Ks_sorted:
            max_slots = max(len(modes_by_model_and_K[model].get(K, []))
                            for model in models_plot_order)
            for slot in range(max_slots):
                # For each model, if it has a mode in this K-slot, assign a position
                for col, model in enumerate(models_plot_order):
                    modes_at_K = modes_by_model_and_K[model].get(K, [])
                    if slot < len(modes_at_K):
                        short_mode = modes_at_K[slot]
                        full_name = f"{model}_{short_mode}"
                        mode_pos[full_name] = (row_idx, col)
                row_meta_list.append((K, slot))
                row_idx += 1

        n_row = row_idx
        row_meta = row_meta_list

    else:
        # Original behavior: rows are just mode index within model
        n_row = max(len(modes_by_model[m]) for m in models_plot_order)
        for col, model in enumerate(models_plot_order):
            short_modes = modes_by_model.get(model, [])
            for row, short_mode in enumerate(short_modes):
                full_name = f"{model}_{short_mode}"
                mode_pos[full_name] = (row, col)

    # grid spacing in x-direction 
    x_step = int(K_max * wspace_padding)
    start_positions = np.zeros((n_row, n_col), dtype=int)
    for col, model in enumerate(models_plot_order):
        for row in range(n_row):
            start_positions[row, col] = col * x_step

    # figure creation 
    width = max(1.0, n_col * K_max * figsize_scale[0])
    height = max(1.0, n_row * figsize_scale[1])
    fig, ax = plt.subplots(
        1, 1,
        figsize=(width, height),
        dpi=dpi,
    )

    # plot clusters (markers) for each mode
    all_modes_alignment = comp_res.all_modes_alignment

    for full_name, (row, col) in mode_pos.items():
        reordering = all_modes_alignment[full_name]
        K = len(reordering)

        # x positions in global axis
        xs = np.array(reordering) + start_positions[row, col]
        ys = np.full(K, row, dtype=float)

        # colors per cluster index
        cols = [cmap[r] for r in reordering]

        ax.scatter(
            xs,
            ys,
            s=marker_size,
            linewidths=0.5,
            edgecolors="k",
            c=cols,
            zorder=6,
        )

        # highlight the mode with a subtle rectangle
        rect = Rectangle(
            (start_positions[row, col] - 0.5, row - 0.2),
            K,
            0.4,
            linewidth=0.5,
            edgecolor="lightgrey",
            facecolor="lightgrey",
            alpha=0.2,
            joinstyle="round",
            capstyle="round",
            zorder=1,
        )
        ax.add_patch(rect)

        if label_modes:
            # Label short mode near the block
            model, short = _split_full_mode(full_name)
            ax.text(
                start_positions[row, col] - 0.5,
                row,
                short.replace("_", " "),
                c="gray",
                fontsize=8,
                ha="right",
                va="center",
            )

    # draw alignment edges between modes (using alignment_across_all) 
    alignment_acrossK = comp_res.alignment_across_all

    for key, mapping in alignment_acrossK.items():
        mode_A, mode_B = key.split("-", 1)  # mapping: B -> A
        if mode_A not in mode_pos or mode_B not in mode_pos:
            continue

        model_A, _ = _split_full_mode(mode_A)
        model_B, _ = _split_full_mode(mode_B)

        # only plot between different models
        if model_A == model_B:
            continue

        row_A, col_A = mode_pos[mode_A]
        row_B, col_B = mode_pos[mode_B]

        # optionally only draw between adjacent model columns
        if adjacent_only and abs(col_A - col_B) != 1:
            continue

        reord_A = list(all_modes_alignment[mode_A])
        reord_B = list(all_modes_alignment[mode_B])
        K_B = len(reord_B)

        for kp1 in range(K_B):
            orig_kB = kp1
            orig_kA = mapping[kp1]

            # aligned x index in each mode
            x_idx_A = reord_A.index(orig_kA)
            x_idx_B = reord_B.index(orig_kB)

            xA = x_idx_A + start_positions[row_A, col_A]
            yA = row_A
            xB = x_idx_B + start_positions[row_B, col_B]
            yB = row_B

            same_column_index = (x_idx_A == x_idx_B)

            if same_column_index and not connect_identity:
                # skip identity if requested
                continue

            if same_column_index:
                # faint identity line
                color = "lightgrey"
                ls = ls_alt[1]
                lw_use = lw / 2.0
                z = 1
            else:
                # highlight non-1–1 (or shifted) alignments
                color = "black"
                if alt_ls:
                    ls = ls_alt[(row_A + row_B + col_A + col_B + kp1) % len(ls_alt)]
                else:
                    ls = ls_alt[0]
                lw_use = lw
                z = 2

            ax.plot(
                [xB, xA],
                [yB, yA],
                c=color,
                ls=ls,
                lw=lw_use,
                zorder=z,
            )

    # draw within-model edges between modes (using pair_mappings) 
    if pair_mappings is not None:
        for model in models_plot_order:
            short_modes = modes_by_model.get(model, [])
            n_modes = len(short_modes)

            for i in range(n_modes - 1):
                mode_1 = f"{model}_{short_modes[i]}"
                mode_2 = f"{model}_{short_modes[i + 1]}"

                if mode_1 not in mode_pos or mode_2 not in mode_pos:
                    continue

                row_1, col_1 = mode_pos[mode_1]
                row_2, col_2 = mode_pos[mode_2]

                pair_key = f"{mode_1}-{mode_2}"
                if pair_key not in pair_mappings:
                    continue
                mapping_pairs = pair_mappings[pair_key]

                for c1, c2 in mapping_pairs:
                    x_idx_1 = c1
                    x_idx_2 = c2

                    x1 = x_idx_1 + start_positions[row_1, col_1]
                    y1 = row_1
                    x2 = x_idx_2 + start_positions[row_2, col_2]
                    y2 = row_2

                    same_column_index = (x_idx_1 == x_idx_2)

                    if same_column_index and not connect_identity:
                        continue

                    if same_column_index:
                        color = "lightgrey"
                        ls = ls_alt[1]
                        lw_use = lw / 2.0
                        z = 1
                    else:
                        color = "black"
                        if alt_ls:
                            ls = ls_alt[(row_1 + row_2 + col_1 + col_2 + c1) % len(ls_alt)]
                        else:
                            ls = ls_alt[0]
                        lw_use = lw
                        z = 2

                    ax.plot(
                        [x2, x1],
                        [y2, y1],
                        c=color,
                        ls=ls,
                        lw=lw_use,
                        zorder=z,
                    )

    # axes formatting 
    tick_positions = [
        col * x_step + (K_max - 1) / 2.0 for col in range(n_col)
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(models_plot_order, fontsize=12, weight="bold")
    ax.set_xlabel("Models", fontsize=12)

    ax.set_yticks(range(n_row))
    if row_by_K and row_meta:
        # Label rows by K (and slot index if multiple rows share the same K)
        K_counts = {}
        for K, _ in row_meta:
            K_counts[K] = K_counts.get(K, 0) + 1

        y_labels = []
        K_seen_slot = {K: 0 for K in K_counts}
        for K, slot in row_meta:
            K_seen_slot[K] += 1
            if K_counts[K] > 1:
                y_labels.append(f"") #K={K} (#{K_seen_slot[K]})
            else:
                y_labels.append(f"K={K}")
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.set_ylabel("Modes", fontsize=10)
    else:
        ax.set_yticklabels([])

    if label_modes:
        ax.set_xlim(-2.5, (n_col - 1) * x_step + K_max + 0.5)
    else:
        ax.set_xlim(-1, (n_col - 1) * x_step + K_max)

    ax.set_ylim(-0.5, n_row - 0.5)
    ax.invert_yaxis()

    # aspect: height scaling via figsize_scale[1]
    ax.set_aspect(1.0 * figsize_scale[1], adjustable="box")

    if row_by_K:
        title = "Alignment of clusters across models (rows grouped by K)"
    else:
        title = "Alignment of clusters across models (columns = models, rows = modes)"
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    return fig, ax


def plot_discrete_colorbar(
    colors: Sequence[ColorSpec],
    K_max: Optional[int] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    facecolor: str = "white",
):
    """
    Plot a simple discrete colorbar-like strip for cluster colors.

    Parameters
    ----------
    colors
        A sequence of color specs. Can be:
        - list of RGB tuples (0-1 range)
        - hex strings
        - named matplotlib colors
    K_max
        Number of clusters. If None, inferred as len(colors).
    labels
        X tick labels. If None, defaults to ["Cls.1", ..., "Cls.K"].
    ax
        Existing axes to draw on. If None, a new figure/axes is created.
    figsize
        Figure size (only used if ax is None). Default scales with K.
    dpi
        Figure dpi (only used if ax is None).
    facecolor
        Figure/axes facecolor.

    Returns
    -------
    fig, ax, im
        im is the AxesImage returned by imshow.
    """
    if K_max is None:
        K_max = len(colors)

    if K_max != len(colors):
        raise ValueError(
            f"Number of colors in colormap ({len(colors)}) "
            f"must match K_max ({K_max})."
        )

    if labels is None:
        labels = [f"Cls.{k}" for k in range(1, K_max + 1)]
    else:
        if len(labels) != K_max:
            raise ValueError("labels length must match K_max.")

    created_fig = False
    if ax is None:
        created_fig = True
        if figsize is None:
            figsize = (max(1.5, K_max * 1.0), 1.0)
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            subplot_kw=dict(yticks=[]),
        )
    else:
        fig = ax.figure

    ax.set_facecolor(facecolor)

    rgba = np.array([mcolors.to_rgba(c) for c in colors], dtype=float)
    img = rgba.reshape(1, K_max, 4)

    im = ax.imshow(img, aspect="auto", extent=(0, K_max, 0, 1))
    ax.get_yaxis().set_visible(False)

    ax.set_xticks(
        ticks=np.arange(0.5, K_max + 0.5, 1),
        labels=labels,
    )

    ax.tick_params(axis="x", length=0)
    if created_fig:
        fig.tight_layout()

    return fig, ax, im


def add_hlines_when_model_switches(
    ax: plt.Axes,
    mode_index: Sequence[str],
    *,
    model_from_mode: Optional[Callable[[str], str]] = None,
    line_kwargs: Optional[dict] = None,
) -> List[int]:
    """
    Add horizontal separator lines between heatmap rows whenever the
    'model' prefix switches.

    Parameters
    ----------
    ax
        The heatmap axis.
    mode_index
        The row index used in the heatmap (e.g., df.index).
    model_from_mode
        Function to extract model name from a mode string.
        Default assumes "model_shortmode" and takes the part before first "_".
    line_kwargs
        Passed to ax.axhline (e.g. dict(color="black", linewidth=1.2, alpha=0.8)).

    Returns
    -------
    boundaries
        List of row indices where a line was drawn (y positions).
    """
    if model_from_mode is None:
        model_from_mode = lambda s: str(s).split("_", 1)[0]

    if line_kwargs is None:
        line_kwargs = dict(color="black", linewidth=1.0, alpha=0.8)

    modes = list(map(str, mode_index))
    models = [model_from_mode(m) for m in modes]

    boundaries = []
    for i in range(1, len(models)):
        if models[i] != models[i - 1]:
            # boundary between row i-1 and i is at y = i
            ax.axhline(i, **line_kwargs)
            boundaries.append(i)

    return boundaries


def plot_mode_annotation_group_diff(
    df_mode_group_diff: pd.DataFrame,
    *,
    mode_sizes: Optional[pd.Series] = None,
    annotation_group_sizes: Optional[pd.Series] = None,
    ref_mode: Optional[str] = None,
    show_top: bool = True,
    show_left: bool = True,
    annot = None,
    cmap: str = "Reds",
    cbar_label: str = "Fraction of different cells",
    top_ylabel: str = "#cells in the group",
    left_xlabel: str = "Mode size",
    x_label: str = "Annotation groups",
    y_label: str = "Modes",
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
    height_ratios: Tuple[float, float] = (1, 6),
    width_ratios: Tuple[float, float] = (1.5, 6),
    wspace: float = 0.01,
    hspace: float = 0.01,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_fraction: float = 0.6,
    xtick_rotation: float = 45,
    xtick_fontsize: int = 8,
    ytick_fontsize: int = 8,
    label_fontsize: int = 7,
    border_width: float = 0.5,
    border_color: str = "lightgray",
    zero_label_eps: float = 1e-3,
    top_round_to: int = 500,
    show_mode_size_labels: bool = True,
    add_model_separators: bool = True,
    model_sep_kwargs: Optional[dict] = None,
):
    """
    Plot a heatmap of mode-by-annotation-group differences.
    Optionally add marginal bar plots:
      - Top: annotation group sizes
      - Left: mode sizes

    Parameters
    ----------
    df_mode_group_diff
        DataFrame with index=modes and columns=annotation groups.
    mode_sizes
        Series of mode sizes indexed by FULL mode names.
        Required if show_left=True.
    annotation_group_sizes
        Series of group sizes indexed by group names.
        Required if show_top=True.
    ref_mode
        If provided, highlights this row label in red/bold.
    show_top, show_left
        Toggle marginal bars.

    Returns
    -------
    fig, axes
        axes is a dict with keys: "heatmap", "top", "left"
    """
    if df_mode_group_diff is None or df_mode_group_diff.empty:
        raise ValueError("df_mode_group_diff is empty.")

    if show_top and annotation_group_sizes is None:
        raise ValueError("annotation_group_sizes is required when show_top=True.")
    if show_left and mode_sizes is None:
        raise ValueError("mode_sizes is required when show_left=True.")

    # ---- Align marginal series to heatmap order (if used) ----
    if show_left:
        mode_sizes_aligned = (
            mode_sizes.reindex(df_mode_group_diff.index).astype(float)
        )
        mode_sizes_aligned = mode_sizes_aligned.fillna(0.0)

    if show_top:
        group_sizes_aligned = (
            annotation_group_sizes.reindex(df_mode_group_diff.columns).astype(float)
        )
        group_sizes_aligned = group_sizes_aligned.fillna(0.0)

    # ---- Build figure/axes layout ----
    # Cases:
    #   both  -> 2x2
    #   top   -> 2x1
    #   left  -> 1x2
    #   none  -> 1x1

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)

    ax_top = None
    ax_left = None
    ax_hm = None

    if show_top and show_left:
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            height_ratios=list(height_ratios),
            width_ratios=list(width_ratios),
            wspace=wspace, hspace=hspace
        )
        ax_top = fig.add_subplot(gs[0, 1])
        ax_left = fig.add_subplot(gs[1, 0])
        ax_hm = fig.add_subplot(gs[1, 1])

    elif show_top and not show_left:
        gs = fig.add_gridspec(
            nrows=2, ncols=1,
            height_ratios=list(height_ratios),
            wspace=wspace, hspace=hspace
        )
        ax_top = fig.add_subplot(gs[0, 0])
        ax_hm = fig.add_subplot(gs[1, 0])

    elif show_left and not show_top:
        gs = fig.add_gridspec(
            nrows=1, ncols=2,
            width_ratios=list(width_ratios),
            wspace=wspace, hspace=hspace
        )
        ax_left = fig.add_subplot(gs[0, 0])
        ax_hm = fig.add_subplot(gs[0, 1])

    else:
        ax_hm = fig.add_subplot(1, 1, 1)


    # Top bar: annotation group sizes
    if show_top and ax_top is not None:
        ax_top.bar(group_sizes_aligned.index, group_sizes_aligned.values)
        ax_top.set_ylabel(top_ylabel, rotation=90, ha="center", va="bottom")
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["top"].set_visible(False)

        ax_top.set_xticks(np.arange(len(group_sizes_aligned)))
        ax_top.set_xticklabels(
            group_sizes_aligned.index,
            rotation=xtick_rotation,
            fontsize=xtick_fontsize,
            ha="right",
        )
        ax_top.set_xlim(-0.5, len(group_sizes_aligned) - 0.5)

        max_size = float(group_sizes_aligned.max()) if len(group_sizes_aligned) else 0.0
        if top_round_to > 0 and max_size > 0:
            max_size_rounded = int(np.ceil(max_size / float(top_round_to)) * top_round_to)
        else:
            max_size_rounded = int(max_size)

        ylim_top = max_size_rounded if max_size_rounded > 0 else 1
        ax_top.set_ylim(0, ylim_top)
        ax_top.set_yticks(np.linspace(0, ylim_top, 4))


    # Left bar: mode sizes (horizontal)
    if show_left and ax_left is not None:
        
        ax_left.barh(mode_sizes_aligned.index, mode_sizes_aligned.values)

        if show_mode_size_labels:
            for i, v in enumerate(mode_sizes_aligned.values):
                ax_left.text(v + 0.1, i, str(int(v)), va="center", ha="right", fontsize=8)

        ax_left.invert_yaxis()
        ax_left.invert_xaxis()
        ax_left.set_xlabel(left_xlabel)
        ax_left.set_yticklabels([])
        ax_left.spines["left"].set_visible(False)
        ax_left.spines["top"].set_visible(False)
        ax_left.set_yticks(np.arange(len(mode_sizes_aligned)))
        ax_left.set_ylim(-0.5, len(mode_sizes_aligned) - 0.5)

    # Heatmap
    vals = df_mode_group_diff.to_numpy(dtype=float)
    if annot is not None:
        labels = annot
    else:
        labels = np.full(vals.shape, "", dtype=object)
        mask = np.abs(vals) >= float(zero_label_eps)
        if np.any(mask):
            labels[mask] = np.char.mod("%.3f", vals[mask])

    sns.heatmap(
        df_mode_group_diff,
        ax=ax_hm,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        cbar_kws={"label": cbar_label, "ticks": np.linspace(0, 1, 6), "shrink": cbar_fraction},
        annot=labels, fmt="",
        annot_kws={"size": label_fontsize},
        linewidths=border_width,
        linecolor=border_color,
    )

    ax_hm.set_xlabel(x_label)
    ax_hm.set_ylabel(y_label)

    # Fix tick-setting (matplotlib-safe)
    ax_hm.set_xticks(np.arange(len(df_mode_group_diff.columns)) + 0.5)
    ax_hm.set_xticklabels(
        df_mode_group_diff.columns,
        rotation=xtick_rotation,
        ha="right",
        fontsize=xtick_fontsize,
    )
    ax_hm.set_yticks(np.arange(len(df_mode_group_diff.index)) + 0.5)
    ax_hm.set_yticklabels(
        df_mode_group_diff.index,
        fontsize=ytick_fontsize,
    )

    # Optional model-switch separators
    if add_model_separators:
        if model_sep_kwargs is None:
            model_sep_kwargs = dict(color="black", linewidth=1.2, alpha=0.9)

        add_hlines_when_model_switches(
            ax_hm,
            df_mode_group_diff.index,
            line_kwargs=model_sep_kwargs,
        )

    # Highlight ref row label
    if ref_mode is not None:
        for lab in ax_hm.get_yticklabels():
            if lab.get_text() == ref_mode:
                lab.set_color("red")
                lab.set_fontweight("bold")

    axes = {"heatmap": ax_hm, "top": ax_top, "left": ax_left}

    return fig, axes


def _add_mapping_connection_lines(
    fig,
    axes_handles,
    pair_mapping,
    *,
    src_row,
    dst_row,
    color="k",
    alpha=0.25,
    lw=0.8,
    zorder=1,
):
    """
    Draw lines from the *bottom* of axes in src_row to the *top* of axes in dst_row
    using pair_mapping = [(i_small, j_large), ...].
    """
    if not pair_mapping:
        return

    def _row_anchor_points(row, where: str):
        # where: "bottom" or "top"
        pts = {}
        cols = [c for (r, c) in axes_handles.keys() if r == row]
        for c in cols:
            bbox = axes_handles[(row, c)].get_position()
            x = 0.5 * (bbox.x0 + bbox.x1)
            y = bbox.y0 if where == "bottom" else bbox.y1
            pts[c] = (x, y)
        return pts

    src_pts = _row_anchor_points(src_row, "bottom")
    dst_pts = _row_anchor_points(dst_row, "top")

    for i_small, j_large in pair_mapping:
        p1 = src_pts.get(i_small)
        p2 = dst_pts.get(j_large)
        if p1 is None or p2 is None:
            continue

        fig.add_artist(
            mlines.Line2D(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                transform=fig.transFigure,
                color=color,
                alpha=alpha,
                lw=lw,
                zorder=zorder,
            )
        )


def plot_ref_alt_mapping_grid(
    *,
    ref_Q: np.ndarray,
    alt_Q: np.ndarray,
    pair_mappings: Dict[str, Sequence[Tuple[int, int]]],
    ref_mode: str,
    alt_mode: str,
    coords,
    colors: Sequence,
    show: Sequence[str] = ("alt",),
    dpi: int = 150,
    s: float = 0.5,
    figsize_scale: Tuple[float, float] = (2.0, 2.0),
    strict_pair_mapping: bool = True,
    # --- new, optional connection-line controls ---
    connect_lines: bool = True,
    connect_color: str = "k",
    connect_alpha: float = 0.25,
    connect_lw: float = 0.8,
):
    """
    Plot reference/alt Qs, mapped alt, and per-column abs differences.

    Row order (when included):
        0) reference (the *smaller-K* space used for mapping)
        1) mapped alt (larger-K mapped into smaller-K space)
        2) original alt (the larger-K Q)
        3) diff (abs(reference - mapped_alt))

    The `show` argument controls which of:
        {"alt", "mapped_alt", "diff"}
    are added in addition to the reference row.

    Default:
        show=("alt",)   -> reference + original alt
    """
    # ---- coords -> (x, y) ----
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
        xy = (np.asarray(x), np.asarray(y))
    else:
        arr = np.asarray(coords)
        xy = (arr[:, 0], arr[:, 1])

    show_set: Set[str] = set(show)
    valid = {"alt", "mapped_alt", "diff"}
    bad = show_set - valid
    if bad:
        raise ValueError(f"Invalid show entries: {sorted(bad)}; valid={sorted(valid)}")

    ref_K = int(ref_Q.shape[1])
    alt_K = int(alt_Q.shape[1])

    # ---- decide mapping direction based on smaller K ----
    if ref_K <= alt_K:
        small_Q, large_Q = ref_Q, alt_Q
        small_mode, large_mode = ref_mode, alt_mode
        small_is_ref = True
    else:
        small_Q, large_Q = alt_Q, ref_Q
        small_mode, large_mode = alt_mode, ref_mode
        small_is_ref = False

    key = f"{small_mode}-{large_mode}"
    pair_mapping = pair_mappings.get(key)

    # ---- compute mapping if needed ----
    mapped_Q = None
    diff_Q = None
    overall_diff = 0.0

    if pair_mapping is None:
        if strict_pair_mapping and (show_set & {"mapped_alt", "diff"}):
            raise KeyError(f"Missing pair mapping key: {key}")
    else:
        mapped_Q, diff_Q = map_alt_to_ref(small_Q, large_Q, pair_mapping)
        overall_diff = compute_overall_membership_difference(diff_Q) if diff_Q is not None else 0.0

    # ---- row plan ----
    row_specs: List[Tuple[str, str, np.ndarray, int]] = []
    row_specs.append(("ref", f"{small_mode}\n[ref]", small_Q, int(small_Q.shape[1])))

    if "mapped_alt" in show_set and mapped_Q is not None:
        row_specs.append(("mapped", f"{large_mode}\n[aligned]", mapped_Q, int(mapped_Q.shape[1])))

    if "alt" in show_set:
        row_specs.append(("alt", f"{large_mode}\n[orig]", large_Q, int(large_Q.shape[1])))

    if "diff" in show_set and diff_Q is not None:
        row_specs.append(("diff", f"Abs Diff\n{overall_diff:.4f}", diff_Q, int(diff_Q.shape[1])))

    n_rows = len(row_specs)
    n_cols = max(ref_K, alt_K)

    w_scale, h_scale = figsize_scale
    fig = plt.figure(figsize=(n_cols * w_scale, n_rows * h_scale), dpi=dpi)
    gs = fig.add_gridspec(n_rows, n_cols)

    axes_handles: dict[tuple[int, int], plt.Axes] = {}
    kind_to_row: dict[str, int] = {}

    n_cell = int(small_Q.shape[0])

    for r, (kind, ylab, mat, K) in enumerate(row_specs):
        kind_to_row[kind] = r

        # plot existing clusters
        for k in range(K):
            ax = fig.add_subplot(gs[r, k])

            if kind == "diff":
                cluster_diff = float(mat[:, k].sum() / n_cell) if n_cell > 0 else 0.0
                title = f"Diff {cluster_diff:.4f}"
            else:
                title = f"Cluster {k+1}"

            plot_single_spatial_membership(
                mat,
                xy,
                cls_idx=k,
                ax=ax,
                title=title,
                ref_color=colors[k],
                s=s,
            )
            ax.set_title(title, zorder=10)

            axes_handles[(r, k)] = ax

        # y-label on leftmost axis
        if (r, 0) in axes_handles:
            axes_handles[(r, 0)].set_ylabel(
                ylab, fontsize=14, rotation=0, ha="right", va="center"
            )

        # hide unused columns for this row
        for k in range(K, n_cols):
            ax = fig.add_subplot(gs[r, k])
            ax.axis("off")

    # Optional semantic reminder
    if not small_is_ref:
        fig.suptitle(
            "Note: alt has smaller K; alignment shown in alt space",
            fontsize=10,
            y=0.995,
        )

    # Finalize layout BEFORE reading axis positions for connections
    fig.tight_layout()

    # ---- add connection lines between rows (as requested) ----
    # Conditions:
    #   - Only if orig row exists
    #   - Use aligned->orig if aligned shown, else ref->orig
    if (
        connect_lines
        and pair_mapping is not None
        and "alt" in kind_to_row
    ):
        dst_row = kind_to_row["alt"]
        src_row = kind_to_row["mapped"] if "mapped" in kind_to_row else kind_to_row["ref"]

        _add_mapping_connection_lines(
            fig,
            axes_handles,
            pair_mapping,
            src_row=src_row,
            dst_row=dst_row,
            color=connect_color,
            alpha=connect_alpha,
            lw=connect_lw,
        )

    return fig, axes_handles

def plot_pair_mapping_alignment(
    *,
    pair_mapping: Sequence[Tuple[int, int]],
    ref_K: int,
    alt_K: int,
    ref_mode: str,
    alt_mode: str,
    plot_colors: Union[Sequence, Mapping[int, str]],
    figsize: Tuple[float, float] = (5, 2),
    dpi: int = 150,
    node_size: float = 150,
    node_edgecolor: str = "black",
    node_linewidth: float = 0.5,
    line_color: str = "k",
    line_alpha: float = 0.5,
    line_lw: float = 1.0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a simple two-row pair-mapping alignment diagram.

    Parameters
    ----------
    pair_mapping
        Sequence of (c_ref, c_alt) index pairs. Assumes ref row at y=1, alt row at y=0.
    ref_K, alt_K
        Number of clusters in ref/alt spaces for x-limit.
    ref_mode, alt_mode
        Labels for y-axis and title.
    plot_colors
        Colors indexed by cluster id. Can be a list/tuple or dict.
    ax
        If provided, draws into existing axis.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    def _get_color(i: int):
        if isinstance(plot_colors, Mapping):
            return plot_colors.get(i, "grey")
        # sequence
        if 0 <= i < len(plot_colors):
            return plot_colors[i]
        return "grey"

    # Draw connections + nodes
    for c1, c2 in pair_mapping:
        ax.plot([c1, c2], [1, 0],
                color=line_color, alpha=line_alpha, lw=line_lw, zorder=2)

        ax.scatter([c1], [1],
                   color=_get_color(c1),
                   s=node_size,
                   edgecolors=node_edgecolor,
                   linewidths=node_linewidth,
                   zorder=9)

        ax.scatter([c2], [0],
                   color=_get_color(c2),
                   s=node_size,
                   edgecolors=node_edgecolor,
                   linewidths=node_linewidth,
                   zorder=9)

    ax.set_xlim(-0.5, max(int(ref_K), int(alt_K)) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([0, 1], labels=[alt_mode, ref_mode])

    if title is None:
        title = f"Pair mapping alignment: {ref_mode} to {alt_mode}"
    ax.set_title(title)

    return fig, ax


def strip_leading_zero(x, decimals=2):
    """
    Format a float to a string with given decimals, stripping leading zero.
    """
    if pd.isna(x):
        return ""
    s = f"{x:.{decimals}f}"
    # 0.99 -> .99
    if s.startswith("0"):
        s = s[1:]
    # -0.99 -> -.99
    elif s.startswith("-0"):
        s = "-" + s[2:]
    return s


def plot_cross_model_membership_diff_heatmap(
    cross_model_overall_membership_diff: Union[
        Mapping[Tuple[str, str], float], pd.Series
    ],
    comp_res: Any,
    models: Sequence[str],
    *,
    figsize: Tuple[float, float] = (9, 8),
    dpi: int = 150,
    cmap: str = "Reds",
    decimals: int = 2,
    vmin: float = 0.0,
    vmax: float = 1.0,
    linewidths: float = 0.5,
    linecolor: str = "white",
    ax: Optional[plt.Axes] = None,
    cbar: bool = True,
    annot_size: int = 8,
    tight_layout: bool = True,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot a cross-model overall membership difference heatmap.

    Parameters
    ----------
    cross_model_overall_membership_diff
        Dict-like or pd.Series with keys as (mode_name_model0, mode_name_model1)
        and values in [0, 1].
    comp_res
        An object that contains:
        comp_res.full_mode_names_by_model[model_name] -> ordered list of full mode names.
        This ordering is used to reindex rows/cols (no lexical sorting issues like K10 vs K3).
    models
        Sequence of two model names in the same order used in the diff keys.
    figsize, dpi, cmap, annot, vmin, vmax, linewidths, linecolor
        Seaborn/Matplotlib styling options.
    ax
        If provided, plot into this axis. Otherwise create a new figure/axis.
    cbar
        Whether to show colorbar.
    tight_layout
        Whether to call plt.tight_layout().
    show
        If True, calls plt.show().

    Returns
    -------
    fig, ax, mat
        The figure, axis, and the reindexed matrix used for the heatmap.
    """
    if len(models) != 2:
        raise ValueError(f"`models` must have length 2, got {len(models)}.")

    # Build Series
    if isinstance(cross_model_overall_membership_diff, pd.Series):
        df_cross_model_diff = cross_model_overall_membership_diff.copy()
    else:
        df_cross_model_diff = pd.Series(cross_model_overall_membership_diff)

    # Set MultiIndex with clear names
    df_cross_model_diff.index = pd.MultiIndex.from_tuples(
        df_cross_model_diff.index, names=[models[0], models[1]]
    )

    # Unstack into matrix (rows=models[0], cols=models[1])
    mat = df_cross_model_diff.unstack()

    # Reindex using the explicit order from comp_res
    try:
        row_order = comp_res.full_mode_names_by_model[models[0]]
        col_order = comp_res.full_mode_names_by_model[models[1]]
    except Exception as e:
        raise AttributeError(
            "comp_res must provide `full_mode_names_by_model[model_name]` "
            "for both models."
        ) from e

    mat = mat.reindex(index=row_order, columns=col_order)

    # Create fig/ax if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    annot_mat = mat.applymap(lambda x: strip_leading_zero(x, decimals=decimals))

    sns.heatmap(
        mat,
        annot=annot_mat if decimals >= 0 else False,
        fmt="",
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
        linecolor=linecolor,
        cbar=cbar,
        annot_kws={"fontsize": annot_size},
    )

    # Optional labels/title (safe defaults)
    ax.set_xlabel(models[1])
    ax.set_ylabel(models[0])

    if tight_layout:
        plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, mat



def plot_multimodel_avg_membership_barh(
    avg_cls_memberships: Dict[str, pd.DataFrame],
    *,
    annot_col: str = "annot",
    model_order: Optional[Sequence[str]] = None,
    cluster_order: Optional[Sequence[str]] = None,
    cluster_mode: Literal["intersection", "union", "auto"] = "auto",
    colors: Optional[Sequence[str]] = None,
    figsize_per_cluster: Tuple[float, float] = (2.2, 4.0),
    dpi: int = 150,
) -> Tuple[plt.Figure, list[plt.Axes]]:
    """
    Generalized horizontal grouped bar charts comparing cluster memberships
    across multiple models.

    Parameters
    ----------
    avg_cls_memberships
        Dict of {model_name: df}.
        Each df: rows = annotation groups, columns = clusters,
        plus an `annot_col` column.
        Example: {"rna": df_rna, "atac": df_atac, "multiome": df_mo}
    annot_col
        Column name for annotation groups.
    model_order
        Order of modalities in the legend/hue. If None, uses dict insertion order.
    cluster_order
        Optional explicit order for clusters (subset will be used).
    cluster_mode
        How to determine clusters across models:
        - "intersection": use only clusters present in all models
        - "union": use all clusters across models
        - "auto": use intersection if non-empty else union
    colors
        Optional colors used for cluster title text (per cluster index).
    figsize_per_cluster
        (width_per_cluster, height) for 1 x K layout.
    dpi
        Figure DPI.

    Returns
    -------
    fig, axes
    """
    if len(avg_cls_memberships) < 2:
        raise ValueError("avg_cls_memberships must contain at least two models.")

    # model order
    if model_order is None:
        model_order = list(avg_cls_memberships.keys())
    else:
        model_order = [m for m in model_order if m in avg_cls_memberships]
        if not model_order:
            raise ValueError("model_order does not match keys in avg_cls_memberships.")

    # --- tidy + index by annot ---
    idx_dfs: Dict[str, pd.DataFrame] = {}
    for model in model_order:
        df = avg_cls_memberships[model]
        if annot_col not in df.columns:
            raise ValueError(f"Model '{model}' is missing '{annot_col}' column.")
        idx_dfs[model] = (
            df.assign(**{annot_col: lambda d: d[annot_col].astype(str)})
              .set_index(annot_col)
        )

    # determine clusters across models
    model_cols = [set(df.columns) for df in idx_dfs.values()]
    inter = set.intersection(*model_cols)
    uni = set.union(*model_cols)

    if cluster_mode == "intersection":
        clusters = sorted(inter)
    elif cluster_mode == "union":
        clusters = sorted(uni)
    else:  # auto
        clusters = sorted(inter) if inter else sorted(uni)

    if cluster_order is not None:
        clusters = [c for c in cluster_order if c in clusters]

    if not clusters:
        raise ValueError("No cluster columns found after applying cluster_mode/cluster_order.")

    # annot order: stable union of all modalities
    annot_index = None
    for df in idx_dfs.values():
        annot_index = df.index if annot_index is None else annot_index.union(df.index)
    annot_order = annot_index.tolist() if annot_index is not None else []

    # plot: 1 x K, horizontal grouped bars
    K = len(clusters)
    fig_w = figsize_per_cluster[0] * K
    fig_h = figsize_per_cluster[1]
    fig, axes = plt.subplots(1, K, figsize=(fig_w, fig_h), dpi=dpi, sharey=True)
    if K == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # Pre-build zero series template to avoid repeated tiny allocations
    for i, (ax, clu) in enumerate(zip(axes, clusters)):
        rows = []
        for model in model_order:
            dfm = idx_dfs[model]
            vals = dfm.get(clu, pd.Series(0, index=dfm.index)).reindex(annot_order).fillna(0)
            rows.append(
                pd.DataFrame(
                    {annot_col: annot_order, "modality": model, "membership": vals.values}
                )
            )

        plot_df = pd.concat(rows, ignore_index=True)

        sns.barplot(
            data=plot_df,
            y=annot_col,
            x="membership",
            hue="modality",
            order=annot_order,
            hue_order=list(model_order),
            orient="h",
            ax=ax,
        )

        title_color = colors[i] if colors is not None and i < len(colors) else None
        ax.set_title(
            str(clu),
            color=title_color,
            loc="left",
            fontsize=12,
            weight="bold",
        )

        ax.set_xlabel("membership")
        ax.set_ylabel("")

        if i == 0:
            ax.legend(frameon=False, title="")
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    return fig, axes


def plot_feature_count(
    feature_counts,
    coords: np.ndarray,
    *,
    feature_name: str | None = "",
    log_transformed: bool = True,
    vmax: float | None = 6,
    vmin: float | None = None,
    size: float = 5,
    cmap: str = "RdYlBu_r",
    cbar_loc: str = "bottom",
    cbar_label: str | None = None,
    ax=None,
):
    """
    Plot a single gene's expression over 2D coordinates.

    Parameters
    ----------
    feature_counts : array-like or sparse
        Per-cell values for one feature. Shape (n_cells,) or (n_cells, 1).
        If sparse, will be densified.
    coords : np.ndarray
        2D coordinates of shape (n_cells, 2).
    feature_name : str or None
        Title annotation.
    log_transformed : bool
        If False, apply log1p to feature_counts.
        If True, assume feature_counts already on log scale.
    vmax : float or None
        Color max. If None or 0, inferred from data.
    vmin : float or None
        Color min. If None, inferred by matplotlib.
    size : float
        Point size.
    cmap : str
        Colormap name.
    cbar_loc : {"bottom", "top", "left", "right"}
        Colorbar location.
    cbar_label : str or None
        Overrides default colorbar label.
    ax : matplotlib.axes.Axes or None
        Existing axis to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # coords validation 
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (n_cells, 2).")

    if feature_counts.shape[0] != coords.shape[0]:
        raise ValueError(
            f"feature_counts length ({feature_counts.shape[0]}) does not match "
            f"coords n_cells ({coords.shape[0]})."
        )

    # log handling
    feature_vals = feature_counts if log_transformed else np.log1p(feature_counts)

    # axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5), dpi=150)

    ax.set_facecolor("lightgray")

    # vmax inference
    if vmax is None or vmax == 0:
        vmax = float(np.nanmax(feature_vals)) if feature_vals.size else 0.0

    # scatter
    sp_sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=feature_vals,
        cmap=cmap,
        s=size,
        alpha=1,
        vmin=vmin,
        vmax=vmax,
    )

    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # colorbar
    if cbar_loc not in {"bottom", "top", "left", "right"}:
        raise ValueError("cbar_loc must be one of: bottom, top, left, right.")

    horizontal = cbar_loc in {"bottom", "top"}
    cbar = plt.colorbar(
        sp_sc,
        ax=ax,
        orientation="horizontal" if horizontal else "vertical",
        location=cbar_loc,
        shrink=0.6,
        pad=0.02 if horizontal else 0.03,
    )

    if cbar_label is None:
        cbar_label = "log1p(count)" if not log_transformed else "expression"
    cbar.set_label(cbar_label, labelpad=0.01)

    ax.set_title(feature_name)

    return ax.get_figure()


def plot_membership_clsind_reordered(P,cmap,lbs,ax,title="",annot=""):
    """
    Plot membership with reordered cluster indices.
    P : np.ndarray, shape (n_samples, n_clusters)
        Membership matrix.
    cmap : list of colors
        Colors for each cluster.
    lbs : array-like, shape (n_samples,)
        Labels for each sample (used to group samples).
    ax : matplotlib.axes.Axes
        Axis to plot on.
    title : str
        Y-axis label.
    annot : str
        Annotation text (shown at top-right).
    """

    N = P.shape[0]
    K = P.shape[1]
    mbsp_sum = P.sum(axis=0)
    mbsp_sortidx = np.argsort(-mbsp_sum) # from largest to smallest
    P = P[:,mbsp_sortidx]

    for lb in np.unique(lbs):
        lb_indices = np.where(lbs==lb)[0]
        lb_P = P[lb_indices,:]
        largest_cls_mbsp = lb_P[:,0]
        ind_sortidx = np.argsort(-largest_cls_mbsp)
        lb_P_sorted = lb_P[ind_sortidx,:]
        lb_P_aug = np.hstack((np.zeros((lb_P_sorted.shape[0],1)),lb_P_sorted))
        for k in range(K):
            ax.bar(lb_indices, lb_P_aug[:,(k+1)], bottom=np.sum(lb_P_aug[:,0:(k+1)],axis=1), 
                    width=1.0, edgecolor='w', linewidth=0, facecolor=cmap[mbsp_sortidx[k]])

    ax.set_xticks([])
    ax.set_xlim([0,N])
    ax.set_ylim([0,1])
    ax.set_xticks([])
    if title:
        if len(title)>8:
            ax.set_ylabel("\n".join(title.split()), rotation=90, fontsize=14, labelpad=30, va="center" )
        else:
            ax.set_ylabel("\n".join(title.split()), rotation=0, fontsize=16, labelpad=30, va="center" )
    else:
        ax.set_ylabel("")
    if annot:
        ax.set_title(annot, fontsize=14, loc="right", pad=5)
    return


def plot_structure_modes_two_level(
    results: ClumpplingResults,
    *,
    modes: Optional[Sequence[str]] = None,
    cmap=None,
    grp_labels: Sequence[str] = (),
    supgrp_labels: Optional[Sequence[str]] = None,
    mode_labels: Optional[Sequence[str]] = None,
    reorder_clsind: bool = True,
    grp_seps_ymin: float = -0.2,
    supgrp_seps_ymin: float = -0.6,
    lb_suffix_sep: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Two-level group version of plot_structure_modes.

    - Pulls Q matrices from `results.Q_by_mode`.
    - Computes grp_info inside using get_uniq_lb_sep.
    - Works for any two-level labels (grp + optional supgrp).
    - Optionally reorders samples by (supgrp, grp).

    Parameters
    ----------
    results
        ClumpplingResults.
    modes
        Which modes to plot. If None, uses `results.modes`.
    cmap
        Colormap list passed to plot_membership.
    grp_labels
        Lower-level labels per sample (length n_cells).
    supgrp_labels
        Higher-level labels per sample (length n_cells), optional.
    grp_seps_ymin, supgrp_seps_ymin
        How far separator lines extend below axis.
    """
    if modes is None:
        modes = results.modes
    modes = list(modes)

    if mode_labels is None or len(mode_labels) != len(modes):
        mode_labels = modes

    grp = np.asarray(grp_labels, dtype=str)
    sup = np.asarray(supgrp_labels, dtype=str) if supgrp_labels is not None else None

    if grp.size == 0:
        raise ValueError("grp_labels must be provided.")
    if sup is not None and sup.size != grp.size:
        raise ValueError("supgrp_labels must have the same length as grp_labels.")

    # compute grp_info
    uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx = get_uniq_lb_sep(grp)
    grp_info = {
        "grp_seps": uniq_lbs_sep_idx,
        "grp_idx": uniq_lbs_indices,
        "grp_lbs": list(uniq_lbs),
        "grp_lbs_suffix": [str(lb).split(lb_suffix_sep)[-1] for lb in uniq_lbs] if lb_suffix_sep else list(uniq_lbs),
        "grp_seps_ymin": grp_seps_ymin,
    }

    if sup is not None:
        uniq_sup, uniq_sup_indices, uniq_sup_sep_idx = get_uniq_lb_sep(sup)
        grp_info.update(
            {
                "supgrp_seps": uniq_sup_sep_idx,
                "supgrp_idx": uniq_sup_indices,
                "supgrp_lbs": list(uniq_sup),
                "supgrp_seps_ymin": supgrp_seps_ymin,
            }
        )

    if figsize is None:
        figsize = (9, max(1.0, 1.0 * len(modes)))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(len(modes), 1)

    # plot each mode
    for i_m, mode in enumerate(modes):
        ax = fig.add_subplot(gs[i_m, 0])

        Q = results.Q_by_mode[mode]

        # assumed signature:
        if reorder_clsind:
            plot_membership_clsind_reordered(Q, cmap, grp_labels, ax, "", annot="")
        else:
            plot_membership(Q, cmap, ax=ax, ylab="", title="", fontsize=14)
        # plot_membership(Q, cmap, ax=ax, ylab="", title="")

        ax.set_ylabel(str(mode_labels[i_m]), fontsize=12)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)

        # lower-level separators + labels
        for v in grp_info["grp_seps"]:
            ax.axvline(
                v,
                ymin=grp_info["grp_seps_ymin"],
                ymax=1,
                color="darkgray",
                ls="--",
                lw=0.8,
                clip_on=False,
            )
        ax.set_xticks(grp_info["grp_idx"])
        ax.set_xticklabels(grp_info["grp_lbs_suffix"], fontsize=9)
        ax.tick_params(axis="x", length=0)

        # super-level separators + secondary axis only on last panel
        if "supgrp_seps" in grp_info:
            for v in grp_info["supgrp_seps"]:
                ax.axvline(
                    v,
                    ymin=grp_info["supgrp_seps_ymin"],
                    ymax=1,
                    color="k",
                    ls="--",
                    lw=0.8,
                    clip_on=False,
                )

            if i_m == len(modes) - 1:
                sec = ax.secondary_xaxis("bottom")
                sec.set_xticks(grp_info["supgrp_idx"])
                sec.set_xticklabels(grp_info["supgrp_lbs"], fontsize=12)
                sec.tick_params(axis="x", pad=15, length=0)

    fig.tight_layout()
    return fig


def plot_structure_modes_one_level(
    results,
    *,
    modes: Optional[Sequence[str]] = None,
    cmap=None,
    grp_labels: Sequence[str] = (),
    mode_labels: Optional[Sequence[str]] = None,
    reorder_clsind: bool = True,
    grp_seps_ymin: float = -0.2,
    lb_suffix_sep: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    x_rot: float = 0,
    x_ha="center",
) -> plt.Figure:
    """
    One-level group version of plot_structure_modes.

    - Pulls Q matrices from `results.Q_by_mode`.
    - Computes grp_info inside using get_uniq_lb_sep.
    - Works for any single-level labels (e.g., sample group / batch / cell type).
    - Optionally reorders samples by grp_labels via plot_membership_clsind_reordered.

    Parameters
    ----------
    results
        ClumpplingResults-like object with attributes:
          - Q_by_mode : dict[mode_name -> (n_cells, K) array]
          - modes     : sequence of mode names (if modes is None)
    modes
        Which modes to plot. If None, uses `results.modes`.
    cmap
        Colormap list passed to plot_membership / plot_membership_clsind_reordered.
    grp_labels
        Group labels per sample (length n_cells).
    mode_labels
        Labels for each mode row (defaults to `modes` if None or wrong length).
    reorder_clsind
        If True, use plot_membership_clsind_reordered(Q, cmap, grp_labels, ...);
        otherwise use plot_membership(Q, cmap, ...).
    grp_seps_ymin
        How far separator lines extend below axis (in axis fraction).
    lb_suffix_sep
        Optional separator; if provided, only the suffix (after lb_suffix_sep)
        is used in x tick labels.
    figsize
        Figure size (width, height). If None, chosen based on number of modes.
    dpi
        Figure DPI.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if modes is None:
        modes = results.modes
    modes = list(modes)

    if mode_labels is None or len(mode_labels) != len(modes):
        mode_labels = modes

    grp = np.asarray(grp_labels, dtype=str)
    if grp.size == 0:
        raise ValueError("grp_labels must be provided and non-empty.")

    # compute grp_info
    uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx = get_uniq_lb_sep(grp)
    grp_info = {
        "grp_seps": uniq_lbs_sep_idx,
        "grp_idx": uniq_lbs_indices,
        "grp_lbs": list(uniq_lbs),
        "grp_lbs_suffix": (
            [str(lb).split(lb_suffix_sep)[-1] for lb in uniq_lbs]
            if lb_suffix_sep
            else list(uniq_lbs)
        ),
        "grp_seps_ymin": grp_seps_ymin,
    }

    # figure
    if figsize is None:
        figsize = (9, max(1.0, 1.0 * len(modes)))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(len(modes), 1)

    # plot each mode
    for i_m, mode in enumerate(modes):
        ax = fig.add_subplot(gs[i_m, 0])

        Q = results.Q_by_mode[mode]

        # plot membership
        if reorder_clsind:
            plot_membership_clsind_reordered(Q, cmap, grp_labels, ax, "", annot="")
        else:
            plot_membership(Q, cmap, ax=ax, ylab="", title="", fontsize=14)

        ax.set_ylabel(str(mode_labels[i_m]), fontsize=12)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)

        # group separators + labels
        for v in grp_info["grp_seps"]:
            ax.axvline(
                v,
                ymin=grp_info["grp_seps_ymin"],
                ymax=1,
                color="darkgray",
                ls="--",
                lw=0.8,
                clip_on=False,
            )
        ax.set_xticks(grp_info["grp_idx"])
        ax.tick_params(axis="x", length=0)
        ax.set_xticklabels([])
    ax.set_xticklabels(grp_info["grp_lbs_suffix"], fontsize=9, rotation=x_rot, ha=x_ha)

    fig.tight_layout()
    return fig


def plot_single_membership_reordered(ax, membership, lbs, color):
    """
    Plot single membership vector with reordered cluster indices.
    membership : np.ndarray, shape (n_samples,)
        Membership vector.
    lbs : array-like, shape (n_samples,)
        Labels for each sample (used to group samples).
    ax : matplotlib.axes.Axes
        Axis to plot on.
    color : str
        Color for the bars.
    """

    N = len(membership)
    for lb in np.unique(lbs):
        lb_indices = np.where(lbs==lb)[0]
        lb_mbsp = membership[lb_indices]
        ind_sortidx = np.argsort(-lb_mbsp)
        lb_mbsp_sorted = lb_mbsp[ind_sortidx]
        ax.bar(lb_indices, lb_mbsp_sorted, 
                width=1.0, edgecolor='w', linewidth=0, facecolor=color)
    ax.set_xlim([0,N])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim([0,1])
    ax.set_yticks([])    
    ax.set_yticklabels([])

    return


def plot_spatial_and_structure_membership_grid(
    results: ClumpplingResults,
    coords: np.ndarray,
    grps: Sequence[str],
    *,
    modes: Optional[Sequence[str]] = None,
    cmap: Optional[Any] = None,
    mode_labels: Optional[Sequence[str]] = None,
    grp_seps: Optional[Sequence[float]] = None,
    reorder_cls: bool = True,
    s: float = 1.0,
    alpha: float = 1.0,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Optimized spatial + structure membership grid.

    Parameters
    ----------
    results : ClumpplingResults
    coords : (n_cells, 2)
    grps : per-cell group labels for ordering the 1D trace
    modes : optional subset of modes (defaults to results.modes)
    cmap :
        - None -> auto tab20 colors
        - list/tuple of colors length >= K_max
        - matplotlib colormap callable
    grp_seps :
        optional separators for group boundaries on the structure plot.
        If None, computed from sorted grps.
    reorder_cls :
        if True, place clusters by aligned index.
    """
    if modes is None:
        modes = results.modes
    modes = list(modes)

    if mode_labels is None or len(mode_labels) != len(modes):
        mode_labels = modes

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (n_cells, 2).")

    grps_arr = np.asarray(grps, dtype=str)
    n_cells = grps_arr.size

    # choose Q source
    Q_source = results.Q_by_mode #results.Q_unaligned_by_mode or 

    # compute K_max over selected modes
    K_max = max(results.mode_K[i] for i,m in enumerate(modes))

    if grp_seps is None:
        _, _, uniq_sep = get_uniq_lb_sep(grps_arr)
        grp_seps = uniq_sep

    # resolve cluster colors
    cluster_colors: List[Any]
    if cmap is None:
        base = plt.get_cmap("tab20")
        if K_max == 1:
            cluster_colors = [base(0.0)]
        else:
            cluster_colors = [base(i / (K_max - 1)) for i in range(K_max)]
    elif isinstance(cmap, (list, tuple, np.ndarray)) or isinstance(cmap, (list, str)):
        cluster_colors = list(cmap)
        if len(cluster_colors) < K_max:
            raise ValueError("cmap as a list must have length >= K_max")
    else:
        # assume callable colormap
        denom = max(K_max - 1, 1)
        cluster_colors = [cmap(i / denom) for i in range(K_max)]

    n_rows = len(modes)
    n_cols = K_max

    if figsize is None:
        figsize = (n_cols * 2.1, n_rows * 2.4)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        2 * n_rows + 1,
        n_cols,
        height_ratios=[1, 6] * n_rows + [1],
        width_ratios=[9] * n_cols,
    )

    # cache custom colormaps
    cmap_cache: Dict[Any, LinearSegmentedColormap] = {}
    # one scatter handle per MODE row for colorbar
    sp_cb: Dict[int, Any] = {}

    for i_m, mode_name in enumerate(modes):
        Q = Q_source[mode_name]
        if Q.shape[0] != n_cells:
            raise ValueError(
                f"grps length ({n_cells}) does not match Q rows ({Q.shape[0]}) for {mode_name}"
            )

        K = Q.shape[1]
        for i_cls in range(K):
            membership = Q[:, i_cls]
            color = cluster_colors[i_cls]

            # structure plot on top
            ax_bar = fig.add_subplot(gs[i_m * 2, i_cls])
            plot_single_membership_reordered(ax_bar, membership, grps, color)
            for v in grp_seps:
                ax_bar.axvline(v, ymin=0, ymax=1, color='darkgray', ls='--', lw=0.3, clip_on=False)
            
            # spatial scatter (bottom row for this mode)
            ax_sp = fig.add_subplot(gs[i_m * 2 + 1, i_cls])

            if color not in cmap_cache:
                cmap_cache[color] = LinearSegmentedColormap.from_list(
                    "custom_cmap", ["white", color, "black"]
                )
            cmap_custom = cmap_cache[color]

            ax_sp.set_facecolor("lightgray")
            sp = ax_sp.scatter(
                coords[:, 0],
                coords[:, 1],
                c=membership,
                cmap=cmap_custom,
                vmin=vmin,
                vmax=vmax,
                s=s,
                alpha=alpha,
            )
            ax_sp.set_xticks([])
            ax_sp.set_yticks([])

            # row label once (first cluster column)
            if i_cls == 0:
                # get mode size
                mode_size = results.mode_stats.set_index('Mode').loc[mode_name].get("Size", None)
                sim = results.mode_stats.set_index('Mode').loc[mode_name].get("Performance", None)
                ax_sp.set_ylabel(f"{mode_labels[i_m]} ({mode_size})\nsim: {sim:.2f}", fontsize=14, ha="center", va="bottom")

            sp_cb[i_cls] = sp

    # per-mode colorbars in last row
    for i_cls, mappable in sp_cb.items():
        ax_cbar = fig.add_subplot(gs[2 * n_rows, i_cls])
        cbar = fig.colorbar(
            mappable,
            ticks=[0, 0.5, 1],
            cax=ax_cbar,
            orientation="horizontal",
            shrink=0.8,
            pad=0.05,
        )
        cbar.ax.set_xlabel(f"Cls.{i_cls + 1} Membership", fontsize=11)

    fig.tight_layout()
    return fig


def plot_separated_clusters_for_focal_gene(
    results,
    coords,
    df_pvs_modes: dict,
    focal_gene: str,
    *,
    modes=None,
    colors=None,
    plot_both_sides: bool = False,
    val_threshold: float = 0.0,
    w_scale: float = 1.2,
    h_scale: float = 1.4,
    dpi: int = 150,
    suptitle: str | None = None,
):
    """
    Plot spatial membership for separated clusters for a single focal gene
    across multiple modes.

    Parameters
    ----------
    results
        Object with `Q_by_mode[mode] -> Q (n_cells, K)`.
    coords : array-like
        (n_cells, 2) or (x, y) tuple for spatial / UMAP coordinates.
    df_pvs_modes : dict[str, pandas.DataFrame]
        Mapping: mode_name -> DataFrame with index including `focal_gene`
        and a column 'sepCls' that stores (group0, group1) lists
        of 0-based cluster indices.
    focal_gene : str
        Gene name / index key used in df_pvs_modes[mode].loc[focal_gene].
    modes : sequence of str, optional
        Subset / order of modes to plot. Defaults to all keys in df_pvs_modes
        that contain `focal_gene`.
    colors
        Either a sequence of colors indexable by cluster index, or a colormap.
        If None, defaults to tab20.
    plot_both_sides : bool
        If False: plot only the “fewer” side clusters across modes in a
        big [modes × all_sepCls] grid.
        If True: for each mode, left = sepCls[0], right = sepCls[1],
        separated by a vertical dashed line.
    val_threshold : float
        Membership threshold passed to plot_single_spatial_membership.
    w_scale, h_scale : float
        Width/height scaling factors for figure size.
    dpi : int
        Figure DPI.
    suptitle : str or None
        Optional figure-level title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict[(mode_name, col_idx) -> Axes]
    """

    # unpack coordinates
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        x, y = coords
    else:
        arr = np.asarray(coords)
        x, y = arr[:, 0], arr[:, 1]

    # choose modes
    if modes is None:
        # only modes where focal_gene exists in that df
        modes = [
            m for m, df in df_pvs_modes.items()
            if focal_gene in df.index
        ]
    else:
        modes = [
            m for m in modes
            if m in df_pvs_modes and focal_gene in df_pvs_modes[m].index
        ]

    if len(modes) == 0:
        raise ValueError(f"No modes found with focal_gene '{focal_gene}'.")

    # build a small DataFrame per focal_gene with sepCls etc.
    data_rows = []
    for mode in modes:
        sepCls = df_pvs_modes[mode].loc[focal_gene, "sepCls"]
        data_rows.append((mode, sepCls))

    sep_df = pd.DataFrame(data_rows, columns=["mode", "sepCls"]).set_index("mode")
    sep_df["fewer_is_high"] = sep_df["sepCls"].apply(
        lambda x: len(x[0]) >= len(x[1])
    )

    # sepCls_fewer: 1-based cluster labels for the side with fewer clusters
    def _compute_sepCls_fewer(sep_pair):
        g0, g1 = sep_pair
        g0 = np.asarray(g0, dtype=int)
        g1 = np.asarray(g1, dtype=int)
        if len(g0) < len(g1):
            return np.sort(g0) + 1
        else:
            return np.sort(g1) + 1

    sep_df["sepCls_fewer"] = sep_df["sepCls"].apply(_compute_sepCls_fewer)

    if colors is None:
        base_colors = cm.get_cmap("tab20").colors
    else:
        base_colors = colors

    def _get_color(idx: int):
        if hasattr(base_colors, "__call__"):
            return base_colors(idx)
        else:
            return base_colors[idx % len(base_colors)]

    # ================================================================
    # CASE 1: only the "fewer" side across modes (grid over all_sepCls)
    # ================================================================
    if not plot_both_sides:
        # union of all fewer-side separated clusters (1-based labels)
        all_sepCls = set.union(
            *([set(arr) for arr in sep_df["sepCls_fewer"]])
        )
        all_sepCls = sorted(list(all_sepCls))

        n_rows = len(modes)
        n_cols = len(all_sepCls)

        fig = plt.figure(
            figsize=(w_scale * n_cols, h_scale * n_rows),
            dpi=dpi,
        )
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

        axes = {}
        first_row_axes = {}

        for i_m, mode_name in enumerate(modes):
            Q = results.Q_by_mode[mode_name]
            K = Q.shape[1]
            row_info = sep_df.loc[mode_name]
            sep_fewer = row_info["sepCls_fewer"]  # 1-based labels
            fewer_is_high = row_info["fewer_is_high"]

            for i_col, cls_label in enumerate(all_sepCls):
                if cls_label > K:
                    continue

                # share y within the same row
                if i_col == 0:
                    ax = fig.add_subplot(gs[i_m, i_col])
                    first_row_axes[i_m] = ax
                else:
                    ax = fig.add_subplot(gs[i_m, i_col], sharey=first_row_axes[i_m])

                cls_idx0 = cls_label - 1
                color = _get_color(cls_idx0)

                plot_single_spatial_membership(
                    Q,
                    (x, y),
                    ref_color=color,
                    cls_idx=cls_idx0,
                    ax=ax,
                    keep_ticks=False,
                    val_threshold=val_threshold,
                    s=0.15,
                    alpha=1.0,
                    vmin=0.0,
                    vmax=1.0,
                )

                ax.set_facecolor("lightgray")
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                )
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )

                if i_col == 0:
                    ax.set_ylabel(mode_name, fontsize=11)

                # highlight clusters that are actually in the fewer set
                if cls_label in sep_fewer:
                    ls = "dotted" if fewer_is_high else "solid"
                    for spine in ax.spines.values():
                        spine.set_edgecolor("k")
                        spine.set_linewidth(1.5)
                        spine.set_linestyle(ls)
                    ax.set_title(
                        f"Cls.{cls_label}",
                        fontsize=10,
                        loc="left",
                        y=0.95,
                        color="k",
                    )
                else:
                    for spine in ax.spines.values():
                        spine.set_linewidth(0)
                    ax.set_title(
                        f"Cls.{cls_label}",
                        fontsize=10,
                        loc="left",
                        y=0.95,
                        color="lightgray",
                    )
                    rect = Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        facecolor="white",
                        alpha=0.75,
                        zorder=2,
                    )
                    ax.add_patch(rect)

                axes[(mode_name, i_col)] = ax

        if suptitle is None:
            suptitle = f"Separated clusters (fewer side) – {focal_gene}"
        fig.suptitle(suptitle, y=0.98, fontsize=12)
        fig.tight_layout()
        return fig, axes

    # ================================================================
    # CASE 2: both sides per mode (left = sepCls[0], right = sepCls[1])
    # ================================================================
    left_right_counts = []
    for mode_name in modes:
        g0, g1 = sep_df.loc[mode_name, "sepCls"]
        left_right_counts.append(len(g0) + len(g1))

    n_rows = len(modes)
    n_cols = max(left_right_counts) if left_right_counts else 0

    fig = plt.figure(
        figsize=(w_scale * n_cols, h_scale * n_rows),
        dpi=dpi,
    )
    gs = fig.add_gridspec(n_rows, n_cols)

    axes = {}
    axes_handles = {}

    for i_m, mode_name in enumerate(modes):
        Q = results.Q_by_mode[mode_name]
        K = Q.shape[1]
        g0, g1 = sep_df.loc[mode_name, "sepCls"]  # 0-based indices
        g0 = list(g0)
        g1 = list(g1)

        for i_sep, cls_idx in enumerate(g0):
            if cls_idx >= K:
                continue
            col = i_sep
            if col >= n_cols:
                continue  # safety, though it shouldn't happen

            ax = fig.add_subplot(gs[i_m, col])
            color = _get_color(cls_idx)
            plot_single_spatial_membership(
                Q,
                (x, y),
                ref_color=color,
                cls_idx=cls_idx,
                ax=ax,
                alpha=0.7,
                s=0.8,
                val_threshold=val_threshold,
            )
            ax.set_title(f"Cluster {cls_idx + 1}", fontsize=9)
            if i_sep == 0:
                ax.set_ylabel(mode_name, va="center", fontsize=11)

            axes[(mode_name, col)] = ax
            axes_handles[(i_m, col)] = ax

        offset = len(g0)
        for i_sep, cls_idx in enumerate(g1):
            if cls_idx >= K:
                continue
            col = offset + i_sep
            if col >= n_cols:
                continue  # safety

            ax = fig.add_subplot(gs[i_m, col])
            color = _get_color(cls_idx)
            plot_single_spatial_membership(
                Q,
                (x, y),
                ref_color=color,
                cls_idx=cls_idx,
                ax=ax,
                alpha=0.7,
                s=0.8,
                val_threshold=val_threshold,
            )
            ax.set_title(f"Cluster {cls_idx + 1}", fontsize=9)

            axes[(mode_name, col)] = ax
            axes_handles[(i_m, col)] = ax

        if len(g0) > 0 and len(g1) > 0:
            left_last_col = len(g0) - 1
            right_first_col = len(g0)

            if (
                (i_m, left_last_col) in axes_handles
                and (i_m, right_first_col) in axes_handles
            ):
                ax_left = axes_handles[(i_m, left_last_col)]
                ax_right = axes_handles[(i_m, right_first_col)]

                pos_left = ax_left.get_position()
                pos_right = ax_right.get_position()

                x_mid = pos_left.x1 + (pos_right.x0 - pos_left.x1) / 2.0

                y0 = min(pos_left.y0, pos_right.y0)
                y1 = max(pos_left.y1, pos_right.y1)

                # optional small padding so the line doesn’t touch the panel edges
                pad = 0.01
                y0 = max(0.0, y0 + pad)
                y1 = min(1.0, y1 - pad)
                line = mlines.Line2D(
                    [x_mid, x_mid],
                    [y0, y1],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=1,
                    linestyle="--",
                )
                fig.add_artist(line)

    if suptitle is None:
        suptitle = f"Separated clusters (both sides) – {focal_gene}"
    fig.suptitle(suptitle, y=0.96, fontsize=12)
    if not plot_both_sides:
        fig.tight_layout()
    return fig, axes