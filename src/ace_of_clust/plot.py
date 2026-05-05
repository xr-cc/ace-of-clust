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
import matplotlib.patches as patches
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
                       get_mode_pair_mappings,
                       map_alt_to_ref)
from .membership import compute_membership_diff

PathLike = Union[str, Path]
ColorSpec = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]

# ---------------------------------------------------------------------
# Membership-based visualizations
# ---------------------------------------------------------------------

def plot_Q_heatmap(
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


def plot_Q_grid(
    results: ClumpplingResults,
    *,
    sort_by: str = "max",   # passed to plot_Q_heatmap
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


def plot_cluster_bars(
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


def plot_cluster_scatter(
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


def plot_spatial_membership(
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
        for feature in highlight:
            if feature in df.index:
                ax.annotate(
                    feature,
                    (df.loc[feature, x], df.loc[feature, y]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
                ax.scatter(df.loc[feature, x], df.loc[feature, y], s=10, color='red')

    return fig, ax


def in_outer_contour(x: float, y: float, paths) -> bool:
    """Return True if (x, y) lies inside ANY of the given matplotlib.path.Path objects."""
    pt = (x, y)
    return any(p.contains_point(pt) for p in paths)


def plot_feature_kde(
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


def get_kde_outliers(
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

    Outlier definition:
      - Fit 2D KDE on (x_col, y_col) for eligible points
      - Find points outside outermost contour

    Ranking (when top_n is not None):
      - Compute distance in optionally scaled (x, y) space.
      - scale="zscore": standardize by mean & std
      - scale="robust": standardize by median & IQR
      - scale="none": use raw (x, y)

    Parameters
    ----------
    df : DataFrame
        Must contain columns `x_col` and `y_col`.
    x_col, y_col : str
        Column names in df to use as x and y axes.
    min_x : float or None, default 0.0
        Minimum x value for eligibility; points with x <= min_x are ignored.
        If None, all finite points are eligible.
    levels : int, default 8
        Number of KDE contour levels.
    cut : float, default 0
        KDE cut parameter (see seaborn.kdeplot).
    top_n : int or None, default None
        If not None, return only the top_n most extreme outliers.
    scale : {"none", "zscore", "robust"}, default "zscore"
        Scaling method for distance computation when ranking outliers.
    return_mask : bool, default False
        If True, also return a boolean mask aligned to df.index indicating outlier status.

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


def plot_feature_bar(
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


def plot_P_profile(
    P_sorted: np.ndarray,
    LFC_sorted: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "",
    lw: float = 0.2,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot sorted log2(P) along cluster index, coloring each gene's curve by the argmax of its LFC profile.
    Parameters
    ----------
    P_sorted : ndarray
        (M, K) array of sorted P values per gene.
    LFC_sorted : ndarray    
        (M, K) array of log fold change values per gene.
    ax : Axes, optional
        If given, draw into this Axes.
    title : str, optional
        Title for the plot.
    lw : float, default 0.2
        Line width for each gene's curve.
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
            lw=lw,
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


def plot_mode_P_profile(
    results: ClumpplingResults,
    mode_name: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    lw: float = 0.2,
) -> tuple[plt.Figure, plt.Axes]:
    """
    For a single mode, compute the clustering profile and plot sorted log P.
    """
    P = _get_mode_P(results, mode_name)
    P_sorted = np.sort(P, axis=1)
    LFC_sorted, _ = compute_profile(P)

    fig, ax = plot_P_profile(
        P_sorted,
        LFC_sorted,
        ax=ax,
        title=mode_name if title is None else title,
        lw=lw,
    )

    ax.set_xlabel("Index of sorted clusters")
    K = P.shape[1]
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(np.arange(1, K + 1), fontsize=8)

    return fig, ax


def plot_sepLFC_dist(
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
            plot_mode_P_profile(results, mode, ax=ax_by_mode[mode])
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
    Create a figure whose axes layout matches `plot_Q_grid`:

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


def plot_cluster_in_grid(
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


def plot_cluster_panels(
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

            plot_spatial_membership(
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


def plot_cluster_overlay(
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
            plot_spatial_membership(
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

def plot_sepLFC_labels(
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


def plot_feature_metrics(
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


def plot_feature_across_modes(
    df_pvs_modes: dict[str, pd.DataFrame],
    modes: list[str],
    selected_feature: str,
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
        res.append(df_pvs_modes[mode_name].loc[selected_feature])
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

    if adjust_text is not None:
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
    selected_feature
        Feature name to plot.
    feature_names
        Sequence of all feature names; selected_feature must be in this list.
    colors
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

__all__ = [
    "plot_Q_heatmap",
    "plot_Q_grid",
    "plot_cluster_bars",
    "plot_cluster_scatter",
    "plot_spatial_membership",
    "plot_feature_scatter",
    "plot_feature_kde",
    "get_kde_outliers",
    "plot_feature_bar",
    "plot_P_profile",
    "plot_mode_P_profile",
    "plot_sepLFC_dist",
    "make_mode_grid",
    "make_mode_grid_by_K",
    "plot_cluster_in_grid",
    "plot_cluster_panels",
    "plot_cluster_overlay",
    "plot_sepLFC_labels",
    "plot_feature_metrics",
    "plot_feature_across_modes",
    "plot_feature_sepLFC_across_modes",
]
