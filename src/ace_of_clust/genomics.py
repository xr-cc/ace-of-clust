"""Mapping between ATAC peaks and genes."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .io import load_gene_intervals


def _parse_peak(peak: str) -> Tuple[str, int, int]:
    """Parse a peak string such as "chr1:819912-823500"."""
    chrom, rest = peak.split(":", 1)
    start_s, end_s = rest.split("-", 1)
    start = int(start_s.replace(",", ""))
    end = int(end_s.replace(",", ""))
    if end < start:
        start, end = end, start
    return chrom, start, end


def _normalize_chrom_lookup(chrom: str, gtf_chroms: Set[str]) -> Optional[str]:
    """Match a chromosome name to the GTF, with or without the "chr" prefix."""
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
    Map each peak to the genes it overlaps.

    Parameters
    ----------
    peaks
        Peak strings such as "chr1:819912-823500".
    gtf_file
        GTF file.
    upstream
        Bases added upstream of each gene.
    downstream
        Bases added downstream of each gene.
    feature_type
        GTF feature type to use (e.g. "gene").
    source
        GTF source to keep (e.g. "HAVANA"); None keeps all.
    gene_type_allowlist
        gene_type values to keep; None keeps all.
    intergenic_label
        Label for peaks that overlap no gene.
    unassigned_label
        Label for peaks on chromosomes not in the GTF.

    Returns
    -------
    list of str
        Per peak, the gene name, ";"-joined gene names, or a label.
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
    Find top peaks whose mapped genes are among the top genes.

    Parameters
    ----------
    df_informative_peaks_sorted
        Peaks, sorted by importance.
    df_informative_genes_sorted
        Genes, indexed by name and sorted by importance.
    n_top
        Number of top peaks and genes to use; None uses all.
    mapped_col
        Column of the peaks DataFrame with the mapped gene names.
    sep
        Separator between gene names in mapped_col.
    drop_labels
        Gene names to ignore (e.g. "intergenic").

    Returns
    -------
    out_overlap_only
        Top peaks with at least one top gene.
    out
        All top peaks, with columns overlap_genes and has_overlap.
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
    """
    Make the values of a column unique by adding suffixes (_1, _2, ...) to duplicates.

    Parameters
    ----------
    df
        Input DataFrame.
    col
        Column name.
    na_label
        Label for missing values.

    Returns
    -------
    np.ndarray
        Unique values, in the order of df[col].
    """
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


__all__ = [
    "match_peaks_to_genes",
    "peaks_with_top_gene_overlap",
    "make_col_unique",
]
