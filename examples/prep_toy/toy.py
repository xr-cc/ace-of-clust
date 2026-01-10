import numpy as np
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def toy_Q_dirichlet_groups(
    group_sizes,
    pis,
    concentration=30.0,
    seed=0,
    return_groups=False,
    shuffle_cols=False,
    col_shuffle_seed=None,
    shuffle_mode="global",   # "global" or "per_group"
    return_permutation=False,
):
    """
    Generate an (n_total x K) membership matrix Q for L groups, where each group l has
    its own target fractions pi[l]. Optionally shuffle the order of cluster columns
    (label-switching simulation).

    Parameters
    ----------
    group_sizes : Sequence[int]
        Sizes for each group, e.g. [100, 100, 50].
    pis : array-like, shape (L, K)
        Group-specific fractions; each row will be normalized to sum 1.
    concentration : float
        Dirichlet concentration. Larger => rows closer to pi.
    seed : int
        RNG seed for sampling Q values.
    return_groups : bool
        If True, return group labels array (0..L-1).
    shuffle_cols : bool
        If True, shuffle column order of pis (and generated Q).
    col_shuffle_seed : int | None
        RNG seed for column shuffling. If None and shuffle_cols=True, uses `seed`.
    shuffle_mode : {"global","per_group"}
        "global": one permutation applied to all groups.
        "per_group": different permutation per group.
    return_permutation : bool
        If True, also return the permutation(s) used.

    Returns
    -------
    Q : (sum(group_sizes), K)
    groups : (sum(group_sizes),) if return_groups=True
    perm : (K,) if global, or (L, K) if per_group, if return_permutation=True
    """
    rng = np.random.default_rng(seed)

    group_sizes = list(map(int, group_sizes))
    L = len(group_sizes)

    pis = np.asarray(pis, dtype=float)
    if pis.ndim != 2 or pis.shape[0] != L:
        raise ValueError(f"`pis` must be shape (L, K) with L={L}; got {pis.shape}")
    pis = pis / pis.sum(axis=1, keepdims=True)
    K = pis.shape[1]

    # Column shuffling RNG
    if shuffle_cols:
        shuf_rng = np.random.default_rng(seed if col_shuffle_seed is None else col_shuffle_seed)

        if shuffle_mode == "global":
            perm = shuf_rng.permutation(K)
            pis_used = pis[:, perm]
        elif shuffle_mode == "per_group":
            perms = np.vstack([shuf_rng.permutation(K) for _ in range(L)])
            pis_used = np.vstack([pis[g, perms[g]] for g in range(L)])
            perm = perms
        else:
            raise ValueError('shuffle_mode must be "global" or "per_group"')
    else:
        pis_used = pis
        perm = np.arange(K)

    Q_chunks = []
    group_labels = []

    for g, n_g in enumerate(group_sizes):
        alpha = concentration * pis_used[g]
        Qg = rng.dirichlet(alpha, size=n_g)
        Q_chunks.append(Qg)
        if return_groups:
            group_labels.append(np.full(n_g, g, dtype=int))

    Q = np.vstack(Q_chunks)

    outputs = [Q]
    if return_groups:
        outputs.append(np.concatenate(group_labels))
    if return_permutation:
        outputs.append(perm)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def row_normalize(A, eps=1e-12):
    A = np.asarray(A, dtype=float)
    s = A.sum(axis=1, keepdims=True)
    return A / np.maximum(s, eps)


def generate_true_P_dirichlet(K, F, concentration_P=0.5, seed=0):
    """
    P_true: (K,F), each row sums to 1. Smaller concentration => sparser peaks.
    """
    rng = np.random.default_rng(seed)
    alpha = np.full(F, concentration_P, dtype=float)
    return rng.dirichlet(alpha, size=K)


def generate_true_Q_groups(group_sizes, pis, concentration_Q=20.0, seed=0):
    """
    Uses toy_Q_dirichlet_groups to generate Q_true and group labels.
    """
    Q_true, groups = toy_Q_dirichlet_groups(
        group_sizes,
        pis,
        concentration=concentration_Q,
        seed=seed,
        return_groups=True,
        shuffle_cols=False,
    )
    return Q_true, groups


def generate_counts_X(Q_true, P_true, n_counts=200, seed=0):
    """
    X_i ~ Multinomial(n_counts, p_i) where p_i = Q_true[i] @ P_true
    Returns int counts matrix X: (n,F).
    """
    rng = np.random.default_rng(seed)
    probs = Q_true @ P_true  # (n,F)
    X = np.vstack([rng.multinomial(n_counts, probs[i]) for i in range(probs.shape[0])])
    return X.astype(int)


def estimate_QP_nmf(X, K, seed=0, max_iter=2000):
    """
    Returns:
      Q_hat: (n,K) row-normalized
      P_hat: (K,F) row-normalized
    """
    
    model = NMF(
        n_components=K,
        init="nndsvda",
        random_state=seed,
        max_iter=max_iter,
    )
    W = model.fit_transform(X)   # (n,K)
    H = model.components_        # (K,F)
    return row_normalize(W), row_normalize(H)


def estimate_QP_lda(X, K, seed=0, max_iter=50, learning_method="batch"):
    """
    Returns:
      Q_hat: (n,K) row-normalized
      P_hat: (K,F) row-normalized
    """
    lda = LatentDirichletAllocation(
        n_components=K,
        random_state=seed,
        max_iter=max_iter,
        learning_method=learning_method,
    )
    Q_hat = lda.fit_transform(X)   # (n,K)
    P_hat = lda.components_        # (K,F)
    return row_normalize(Q_hat), row_normalize(P_hat)


def estimate_QP_runs(
    X,
    K,
    *,
    n_runs=5,
    base_seed_fit=0,
    method="nmf",        # "nmf" or "lda"
    shuffle_cols=False,  # simulate label switching in ESTIMATED outputs
    perm_seed=123,
):
    """
    Returns:
      Q_runs: list of (name, Q_hat)
      P_runs: list of (name, P_hat)
      perms:  list of (name, perm) if shuffle_cols else []
    """
    Q_runs, P_runs, perms = [], [], []
    perm_rng = np.random.default_rng(perm_seed)

    for r in range(n_runs):
        seed_fit = base_seed_fit + r

        if method == "nmf":
            Q_hat, P_hat = estimate_QP_nmf(X, K, seed=seed_fit)
        elif method == "lda":
            Q_hat, P_hat = estimate_QP_lda(X, K, seed=seed_fit)
        else:
            raise ValueError('method must be "nmf" or "lda"')

        if shuffle_cols:
            perm = perm_rng.permutation(K)
            Q_hat = Q_hat[:, perm]
            P_hat = P_hat[perm, :]
            perms.append((f"K{K}_R{r}.perm.txt", perm.astype(int)))

        Q_runs.append((f"K{K}_R{r}.Q", Q_hat))
        P_runs.append((f"K{K}_R{r}.P", P_hat))

    return Q_runs, P_runs, perms


def estimate_Q_runs_kmeans(
    X,
    K,
    *,
    n_runs=5,
    base_seed_fit=0,
    shuffle_cols=True,
    perm_seed=123,
    n_init=10,
    max_iter=300,
):
    """
    Run KMeans multiple times and return one-hot Q matrices.

    Returns
    -------
    Q_runs : list[(name, Q)]
        name like "kmeans_K{K}_R{r}.Q"
    perms : list[(name, perm)]
        column permutations applied (if shuffle_cols=True)
    """
    from sklearn.cluster import KMeans

    Q_runs = []
    perms = []
    perm_rng = np.random.default_rng(perm_seed)

    n = X.shape[0]

    for r in range(n_runs):
        seed_fit = base_seed_fit + r

        km = KMeans(
            n_clusters=K,
            random_state=seed_fit,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = km.fit_predict(X).astype(int)

        # one-hot Q
        Q = np.zeros((n, K), dtype=float)
        Q[np.arange(n), labels] = 1.0

        if shuffle_cols:
            perm = perm_rng.permutation(K)
            Q = Q[:, perm]
            perms.append((f"kmeans_K{K}_R{r}.perm.txt", perm.astype(int)))

        Q_runs.append((f"kmeans_K{K}_R{r}.Q", Q))

    return Q_runs, perms


def kmeans_hard_cluster(X, K, seed=0, n_init=10, max_iter=300):
    """
    Perform hard clustering using K-means.

    Returns
    -------
    labels : (n,) int
        Cluster assignments in {0, ..., K-1}.
    centers : (K, F) float
        Cluster centers.
    """
    model = KMeans(
        n_clusters=K,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = model.fit_predict(X)
    return labels.astype(int), model.cluster_centers_


def onehot_Q_from_labels(labels, K=None, dtype=float):
    """
    One-hot encode hard cluster labels as a Q matrix.

    Parameters
    ----------
    labels : (n,) array-like of int
        Cluster assignments. Typically in {0, ..., K-1}.
    K : int or None
        Number of clusters. If None, inferred as max(labels)+1.
    dtype : type
        Output dtype (float is typical for Q).

    Returns
    -------
    Q : (n, K) array
        One-hot membership matrix. Each row sums to 1.
    """
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1:
        raise ValueError(f"`labels` must be 1D; got shape {labels.shape}")

    if K is None:
        if labels.size == 0:
            raise ValueError("Cannot infer K from empty labels.")
        K = int(labels.max()) + 1

    if labels.min() < 0 or labels.max() >= K:
        raise ValueError(f"Labels out of range [0, {K-1}]. Got min={labels.min()}, max={labels.max()}.")

    Q = np.zeros((labels.size, K), dtype=dtype)
    Q[np.arange(labels.size), labels] = 1
    return Q


def generate_toy_truth_and_X(
    group_sizes,
    pis,
    *,
    F=200,
    concentration_Q=20.0,
    concentration_P=0.5,
    n_counts=200,
    seed_data=0,
):
    """
    Returns:
      Q_true, P_true, X, groups
    """
    K = np.asarray(pis).shape[1]
    P_true = generate_true_P_dirichlet(K, F, concentration_P=concentration_P, seed=seed_data + 10)
    Q_true, groups = generate_true_Q_groups(group_sizes, pis, concentration_Q=concentration_Q, seed=seed_data + 20)
    X = generate_counts_X(Q_true, P_true, n_counts=n_counts, seed=seed_data + 30)
    return Q_true, P_true, X, groups


def save_toy_QP_outputs(
    out_dir,
    *,
    Q_true=None,          # (n,K)
    P_true=None,          # (K,F)
    X=None,               # (n,F) counts or nonnegative
    groups=None,          # (n,)
    Q_runs=None,          # list of (name, Q_hat)
    P_runs=None,          # list of (name, P_hat)
    perms=None,           # list of (name, perm)
    overwrite=False,
    fmt_float="%.6g",
):
    """
    Save toy Q/P data and per-run estimates.

    Parameters
    ----------
    out_dir : str | Path
        Output directory.
    Q_true, P_true, X, groups : arrays or None
        Shared "truth" and observed data to save.
    Q_runs, P_runs, perms : list[(str, array)] or None
        Items to save, where each tuple is (filename, array).
        Example Q filename: "K3_R0.Q"
        Example P filename: "K3_R0.P"
        Example perm filename: "K3_R0.perm.txt"
    overwrite : bool
        If False, skip writing files that already exist.
    fmt_float : str
        Float format for saving Q/P.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_array(path: Path, arr: np.ndarray):
        if path.exists() and (not overwrite):
            return
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.integer):
            np.savetxt(path, arr, fmt="%d")
        else:
            np.savetxt(path, arr, fmt=fmt_float)

    # Save shared objects (use simple conventional names if you didn’t pass names)
    if Q_true is not None:
        _save_array(out_dir / "Q_true.Q", Q_true)
    if P_true is not None:
        _save_array(out_dir / "P_true.P", P_true.T)  # save P as (F,K) for consistency with AOC
    if X is not None:
        _save_array(out_dir / "X.txt", X)
    if groups is not None:
        _save_array(out_dir / "groups.txt", np.asarray(groups, dtype=int))

    # Save per-run outputs
    if Q_runs:
        for name, Q in Q_runs:
            _save_array(out_dir / name, Q)
    if P_runs:
        for name, P in P_runs:
            _save_array(out_dir / name, P.T)  # save P as (F,K) for consistency with AOC
    if perms:
        for name, perm in perms:
            _save_array(out_dir / name, np.asarray(perm, dtype=int))


def plot_toy_Q(Q, title="Membership Matrix Q", figsize=(8, 3), dpi=150, ax=None) -> plt.Axes:
    """
    Plot Q as stacked bar chart (structure plot).
    """
    n, K = Q.shape
    ind = np.arange(n)
    bottom = np.zeros(n)

    colors = plt.cm.tab20.colors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for k in range(K):
        ax.bar(
            ind,
            Q[:, k],
            bottom=bottom,
            color=colors[k % len(colors)],
            label=f"Cluster {k+1}",
            width=1.0,
        )
        bottom += Q[:, k]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Membership Proportion")
    ax.set_title(title)

    return ax
