"""
Genuine fixed absolute-budget experiment (100 seed + 10x50 acquisition = 600
labels for every dataset regardless of pool size), Tanimoto kernel, UCB and
Random, corrected label-blind/held-out protocol throughout.

This exists because a naive "nearest completed cycle to N labels" comparison
mixes fully-completed runs, partial runs, and untouched seed sets across
datasets of very different sizes -- see the paper's rebuttal discussion.
Results -> results_final/fixed_budget/
"""
from __future__ import annotations
import sys, time
from itertools import product
from pathlib import Path
import numpy as np, torch, gpytorch, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from antibody_al.core import (DATASETS, DEVICE, PCA_DIM, build_kernel, GPModel, train_gp,
                               stratified_seed_blind, acquire, _predict, TEST_FRAC)

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "results_final" / "fixed_budget"
SEEDS   = [42, 123, 456]
PROTOCOLS = ["ucb", "random"]
SEED_SIZE  = 100
BATCH_SIZE = 50
CYCLES     = 10  # 100 + 10*50 = 600 total labels, same for every dataset


def run_fixed(dataset, protocol, seed):
    out_csv = OUT_DIR / dataset / f"{protocol}_seed{seed}.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    X = np.load(EMB_DIR / dataset / "X.npy"); y = np.load(EMB_DIR / dataset / "y.npy")
    t2p = np.load(EMB_DIR / dataset / "top2p.npy").astype(bool)

    pca = PCA(n_components=min(PCA_DIM, X.shape[0]-1, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X).astype(np.float32)
    y_norm = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(TEST_FRAC * len(X_pca))))
    test_idx = rng.choice(len(X_pca), size=n_test, replace=False)
    test_mask = np.zeros(len(X_pca), dtype=bool); test_mask[test_idx] = True
    eligible = np.where(~test_mask)[0]

    seed_n = min(SEED_SIZE, len(eligible))
    selected = [int(eligible[i]) for i in stratified_seed_blind(X_pca[eligible], seed_n, rng)]

    rows = []
    for cycle in range(CYCLES + 1):
        tx = torch.tensor(X_pca[selected]).float().to(DEVICE)
        ty = torch.tensor(y_norm[selected]).float().to(DEVICE)
        lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        with gpytorch.settings.cholesky_jitter(1e-3):
            kern = build_kernel("tanimoto").to(DEVICE)
            model = GPModel(tx, ty, lik, kern).to(DEVICE)
            train_gp(tx, ty, lik, model)

        mu_all, _ = _predict(X_pca, model, lik, np.arange(len(X_pca)))
        rho_test, _ = spearmanr(y[test_idx], mu_all[test_idx])
        t2f = int(t2p[selected].sum())
        rows.append({
            "cycle": cycle, "n_labelled": len(selected),
            "spearman_fixed_test": round(float(rho_test) if not np.isnan(rho_test) else 0.0, 4),
            "recall_2p": round(t2f / max(1, int(t2p.sum())), 4),
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        if cycle == CYCLES: break
        best_yn = float(ty.max().cpu())
        new = acquire(protocol, X_pca, model, lik, selected + list(test_idx),
                      min(BATCH_SIZE, len(eligible) - len(selected)), rng, best_yn)
        if not new: break
        selected = selected + new
        del model, kern, lik, tx, ty
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    combos = list(product(DATASETS, PROTOCOLS, SEEDS))
    t0 = time.perf_counter()
    for i, (ds, proto, seed) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {ds} | {proto} | seed={seed}")
        run_fixed(ds, proto, seed)
    print(f"Done in {(time.perf_counter()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
