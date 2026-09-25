"""Shared utilities for the antibody active learning pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_embeddings(emb_dir: Path, pca_dim: int = 128, pca_seed: int = 42):
    """Load X/y/top2p from emb_dir, apply PCA and StandardScaler on y.

    Returns
    -------
    X_pca : np.ndarray  shape (N, pca_dim), float32
    y     : np.ndarray  shape (N,), raw affinities
    y_norm: np.ndarray  shape (N,), z-scored affinities, float32
    t2p   : np.ndarray  shape (N,), bool mask of top-2% compounds
    scaler: StandardScaler fitted on y (for inverse transform if needed)
    pca   : PCA object (for feature attribution alignment)
    """
    X   = np.load(emb_dir / "X.npy")
    y   = np.load(emb_dir / "y.npy").astype(np.float64)
    t2p = np.load(emb_dir / "top2p.npy").astype(bool)

    pca    = PCA(n_components=min(pca_dim, X.shape[0]-1, X.shape[1]),
                 random_state=pca_seed)
    X_pca  = pca.fit_transform(X).astype(np.float32)
    scaler = StandardScaler()
    y_norm = scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    return X_pca, y, y_norm, t2p, scaler, pca


def top_k_recall(selected: list[int], top_mask: np.ndarray) -> float:
    """Fraction of top-k compounds that have been selected."""
    n_top = int(top_mask.sum())
    if n_top == 0:
        return 0.0
    found = int(top_mask[selected].sum())
    return found / n_top


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation; returns 0.0 when undefined."""
    from scipy.stats import spearmanr
    if len(y_true) < 3:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0
