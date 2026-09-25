"""
Core active-learning loop.

stratified_seed  — initial seed selection spanning full affinity range
run_al_cycle     — single AL cycle: train GP, evaluate, acquire
run_experiment   — full experiment: all cycles, saves CSV
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .gp_model import GPRegressionModel, build_kernel, train_gp
from .acquisition import select_random, select_ucb, select_ei, select_pi

PRED_BATCH  = 5_000
DEFAULT_SEED = 42


# ── Stratified seeding (label-blind) ───────────────────────────────────────────
def stratified_seed(X_pca: np.ndarray, n: int, rng: np.random.Generator) -> List[int]:
    """Sample n variants spanning the embedding-feature space (one per KMeans
    cluster on X_pca). Ensures early GP exposure to diverse variants without
    using any affinity labels.

    This replaces an earlier version that binned the pool by true affinity
    quantiles (y), which used labels a real active-learning campaign would not
    have available at seed time — see git history for the original.

    Args:
        X_pca: PCA-reduced embeddings for the pool eligible for seeding
        n:     seed set size
        rng:   NumPy Generator (seeded externally for reproducibility)
    """
    n = min(n, len(X_pca))
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.integers(1 << 31)))
    labels = km.fit_predict(X_pca)
    selected = []
    for c in range(n):
        members = np.where(labels == c)[0]
        if len(members):
            selected.append(int(rng.choice(members)))
    if len(selected) < n:
        rest = np.array(sorted(set(range(len(X_pca))) - set(selected)))
        selected.extend(rng.choice(rest, n - len(selected), replace=False).tolist())
    return selected[:n]


# ── Single cycle ──────────────────────────────────────────────────────────────
def run_al_cycle(
    X_pca: np.ndarray,
    y: np.ndarray,
    y_norm: np.ndarray,
    selected: List[int],
    test_idx: np.ndarray,
    protocol: str,
    kernel_name: str,
    batch_size: int,
    beta: float,
    t2p: np.ndarray,
    t5p: np.ndarray,
    gp_epochs: int,
    gp_lr: float,
    gp_lr_decay: float,
    rng: np.random.Generator,
) -> tuple[Dict, List[int]]:
    """Train GP on labelled set, evaluate metrics, acquire next batch.

    Args:
        test_idx: fixed evaluation split reserved before AL started — never in
            `selected` and never eligible for acquisition. Spearman is reported
            on this fixed set, not on the shrinking remainder of the pool: as
            acquisition removes high-affinity variants from the pool, the
            remainder's variance shrinks and its rank correlation mechanically
            collapses regardless of model quality (range restriction).

    Returns:
        row:      dict of metrics for this cycle
        new_sel:  indices of newly acquired variants (empty at final cycle)
    """
    tx = torch.tensor(X_pca[selected]).float()
    ty = torch.tensor(y_norm[selected]).float()

    lik   = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(tx, ty, lik, build_kernel(kernel_name))
    train_gp(tx, ty, lik, model, epochs=gp_epochs, lr=gp_lr, lr_decay=gp_lr_decay)

    # Predict full pool
    mu_parts = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X_pca), PRED_BATCH):
            bx = torch.tensor(X_pca[i:i + PRED_BATCH]).float()
            mu_parts.append(lik(model(bx)).mean.numpy())
    mu = np.concatenate(mu_parts)

    rho_full, _ = spearmanr(y, mu)
    rho_test, _ = spearmanr(y[test_idx], mu[test_idx])
    t2_found  = int(t2p[selected].sum())
    t5_found  = int(t5p[selected].sum())
    row = {
        "n_labelled":         len(selected),
        "spearman_full_pool": round(float(rho_full) if not np.isnan(rho_full) else 0.0, 4),
        "spearman_fixed_test": round(float(rho_test) if not np.isnan(rho_test) else 0.0, 4),
        "recall_2p":  round(t2_found / max(1, int(t2p.sum())), 4),
        "recall_5p":  round(t5_found / max(1, int(t5p.sum())), 4),
        "top2_found": t2_found,
        "top5_found": t5_found,
        "best_found": round(float(y[selected].max()), 4),
    }

    # Acquire next batch — test_idx is passed as already-"selected" so it is
    # never eligible for acquisition either.
    best_yn = float(ty.max())
    n_total = len(X_pca)
    excluded = selected + list(test_idx)
    if protocol == "random":
        new_sel = select_random(n_total, excluded, batch_size, rng)
    elif protocol == "ucb":
        new_sel = select_ucb(X_pca, model, lik, excluded, batch_size, beta)
    elif protocol == "ei":
        new_sel = select_ei(X_pca, model, lik, excluded, batch_size, best_yn)
    elif protocol == "pi":
        new_sel = select_pi(X_pca, model, lik, excluded, batch_size, best_yn)
    else:
        raise ValueError(f"Unknown protocol '{protocol}'. Choose: random, ucb, ei, pi")

    return row, new_sel


# ── Full experiment ───────────────────────────────────────────────────────────
def run_experiment(
    emb_dir: Path,
    out_csv: Path,
    protocol: str,
    kernel: str,
    seed_size: int,
    batch_size: int,
    n_cycles: int,
    beta: float = 1.0,
    pca_dim: int = 128,
    gp_epochs: int = 100,
    gp_lr: float = 0.1,
    gp_lr_decay: float = 0.95,
    random_seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Run a full active-learning experiment and save results to CSV.

    Args:
        emb_dir:     directory containing X.npy, y.npy, top2p.npy, top5p.npy
        out_csv:     output CSV path (skipped if already exists)
        protocol:    acquisition function — 'random', 'ucb', 'ei', 'pi'
        kernel:      GP kernel — 'tanimoto', 'matern', 'rbf', 'rq'
        seed_size:   initial labelled set size
        batch_size:  variants acquired per AL cycle
        n_cycles:    number of AL cycles after seeding
        beta:        UCB exploration weight (only used when protocol='ucb')
        pca_dim:     PCA components (128 throughout paper)
        gp_epochs:   Adam steps for GP hyperparameter optimisation (100)
        gp_lr:       initial learning rate (0.1)
        gp_lr_decay: ExponentialLR gamma (0.95)
        random_seed: global random seed for reproducibility (42)
        verbose:     print progress

    Returns:
        DataFrame of per-cycle results, or None if out_csv already existed.
    """
    if out_csv.exists():
        return None

    X   = np.load(emb_dir / "X.npy")
    y   = np.load(emb_dir / "y.npy")
    t2p = np.load(emb_dir / "top2p.npy").astype(bool)
    t5p = np.load(emb_dir / "top5p.npy").astype(bool)

    pca_dim_eff = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca         = PCA(n_components=pca_dim_eff, random_state=random_seed)
    scaler      = StandardScaler()
    X_pca       = pca.fit_transform(X).astype(np.float32)
    y_norm      = scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    rng = np.random.default_rng(random_seed)

    # Fixed test split, reserved before AL starts: never eligible for seeding or
    # acquisition. See run_al_cycle docstring for why this differs from
    # evaluating on whatever remains in the pool after acquisition.
    n_test    = max(1, int(round(0.15 * len(X_pca))))
    test_idx  = rng.choice(len(X_pca), size=n_test, replace=False)
    test_mask = np.zeros(len(X_pca), dtype=bool); test_mask[test_idx] = True
    eligible  = np.where(~test_mask)[0]

    selected = [int(eligible[i]) for i in stratified_seed(X_pca[eligible], seed_size, rng)]
    rows: List[Dict] = []

    if verbose:
        print(f"  [{kernel}/{protocol}] N={len(y):,} d={X.shape[1]}->{pca_dim_eff} "
              f"seed={seed_size} batch={batch_size} cycles={n_cycles} β={beta}")

    for cycle in range(n_cycles + 1):
        t0  = time.perf_counter()
        row, new_sel = run_al_cycle(
            X_pca, y, y_norm, selected, test_idx, protocol, kernel,
            batch_size, beta, t2p, t5p,
            gp_epochs, gp_lr, gp_lr_decay, rng,
        )
        row["cycle"]    = cycle
        row["strategy"] = "seed" if cycle == 0 else protocol
        row["elapsed_s"] = round(time.perf_counter() - t0, 2)
        rows.append(row)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        if cycle == n_cycles:
            break
        selected = selected + new_sel

    return pd.DataFrame(rows)
