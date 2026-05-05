"""
genomics.py

Genomic peak-gene mapping utilities.

"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .io import load_gene_intervals


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
    Parameters
    ----------
    peaks
        Iterable of peak strings like "chr1:819912-823500".
    gtf_file
        Path to GTF file.
    upstream
        Number of bases upstream of gene TSS to include.
    downstream
        Number of bases downstream of gene end to include.
    feature_type
        GTF feature type to use (e.g., "gene", "transcript").
    source
        GTF source to filter on (e.g., "HAVANA"), or None for no filtering.
    gene_type_allowlist
        Set of gene_type values to include, or None for no filtering.
    intergenic_label
        Label to assign to peaks with no overlapping genes.
    unassigned_label
        Label to assign to peaks on chromosomes not in the GTF.
    Returns
    -------
    List[str]
        List of gene name(s) per peak, or intergenic/unassigned labels.
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
    Parameters
    ----------
    df_informative_peaks_sorted
        DataFrame of informative peaks, sorted by informativeness.
    df_informative_genes_sorted
        DataFrame of informative genes, sorted by informativeness.
    n_top
        Number of top features to consider. If None, use all.
    mapped_col
        Column in peaks DataFrame containing mapped gene names.
    sep
        Separator for multiple gene names in mapped_col.
    drop_labels
        Labels to exclude from consideration.
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
    Make entries in `col` unique by appending suffixes to duplicates.
    Parameters
    ----------
    df
        Input DataFrame.
    col
        Column name to make unique.
    na_label
        Label to use for NaN entries.
    Returns
    -------
    np.ndarray
        Array of unique strings corresponding to df[col].
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
