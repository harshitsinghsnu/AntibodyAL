"""
BindingGYM active-learning benchmark Section 4.2.

Matches the ALLM-Ab (Furui & Ohue, JCIM 2025) evaluation protocol exactly:
  - 100 held-out test samples (excluded from AL pool)
  - 100 seed labels + 12 acquisition cycles × 50 labels = 700 total labels
  - Spearman ρ evaluated on the 100-sample held-out test set
  - 3 independent random seeds, results averaged

Datasets: 5A12_Ang2, 4D5_HER2, 5A12_VEGF (BindingGYM subsets used by ALLM-Ab)
Embeddings must be pre-computed with:
    python scripts/compute_embeddings_bindinggym.py

Results → results_bindinggym/{dataset}/{protocol}_{kernel}_seed{seed}.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from itertools import product

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "final_pipeline"))
from core import (
    DEVICE, PCA_DIM, GP_EPOCHS, GP_LR, GP_DECAY, UCB_BETA, EI_XI, PI_XI,
    TanimotoKernel, build_kernel, GPModel, train_gp, _predict,
)
from torch.distributions import Normal

EMB_DIR = ROOT / "data" / "embeddings_bindinggym"
OUT_DIR = ROOT / "results_bindinggym"

DATASETS  = ["5a12_ang2", "4d5_her2", "5a12_vegf"]
PROTOCOLS = ["ucb", "ei", "pi", "random"]
KERNELS   = ["tanimoto"]          # paper uses Tanimoto+UCB for comparison
SEEDS     = [0, 1, 2]            # ALLM-Ab uses 3 seeds

SEED_SIZE  = 100
BATCH_SIZE = 50
CYCLES     = 12
TEST_SIZE  = 100
PRED_BATCH = 5_000

_norm = Normal(torch.tensor(0.0), torch.tensor(1.0))


def acquire_bindinggym(protocol, X_pca, model, lik, selected, batch, rng, best_yn=0.0):
    pool = np.array(sorted(set(range(len(X_pca))) - set(selected)))
    if len(pool) == 0:
        return []
    if protocol == "random":
        return rng.choice(pool, min(batch, len(pool)), replace=False).tolist()
    mu, sd = _predict(X_pca, model, lik, pool)
    if protocol == "ucb":
        scores = mu + UCB_BETA * sd
    elif protocol == "ei":
        mu_t = torch.tensor(mu); sd_t = torch.tensor(sd)
        z = (mu_t - best_yn - EI_XI) / (sd_t + 1e-9)
        scores = ((mu_t - best_yn - EI_XI) * _norm.cdf(z)
                  + sd_t * torch.exp(_norm.log_prob(z))).numpy()
    elif protocol == "pi":
        z = (torch.tensor(mu) - best_yn - PI_XI) / (torch.tensor(sd) + 1e-9)
        scores = _norm.cdf(z).numpy()
    else:
        raise ValueError(protocol)
    top = np.argsort(scores)[::-1][:batch]
    return pool[top].tolist()


def run_bindinggym(dataset, protocol, kernel, seed):
    out_csv = OUT_DIR / dataset / f"{protocol}_{kernel}_seed{seed}.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    emb_dir = EMB_DIR / dataset
    X = np.load(emb_dir / "X.npy")
    y = np.load(emb_dir / "y.npy")
    N = len(X)

    # Fixed held-out test set (same split as ALLM-Ab)
    rng_split = np.random.default_rng(seed)
    test_idx  = rng_split.choice(N, TEST_SIZE, replace=False)
    test_mask = np.zeros(N, dtype=bool); test_mask[test_idx] = True
    pool_idx  = np.where(~test_mask)[0]

    X_pool = X[pool_idx]; y_pool = y[pool_idx]
    X_test = X[test_idx]; y_test = y[test_idx]

    # PCA on pool
    pca    = PCA(n_components=min(PCA_DIM, X_pool.shape[0]-1, X_pool.shape[1]),
                 random_state=42)
    scaler = StandardScaler()
    X_pool_pca = scaler.fit_transform(pca.fit_transform(X_pool)).astype(np.float32)
    X_test_pca = scaler.transform(pca.transform(X_test)).astype(np.float32)
    y_scaler   = StandardScaler()
    y_pool_n   = y_scaler.fit_transform(y_pool.reshape(-1,1)).ravel().astype(np.float32)

    # Top-2% in pool (AL-eligible only, per ALLM-Ab protocol)
    t2_thresh = np.percentile(y_pool, 98)
    top2_pool = y_pool >= t2_thresh

    # Seed selection from pool
    rng = np.random.default_rng(seed + 1000)
    pool_selected_local = rng.choice(len(X_pool), SEED_SIZE, replace=False).tolist()

    rows = []
    for cycle in range(CYCLES + 1):
        tx = torch.tensor(X_pool_pca[pool_selected_local]).float().to(DEVICE)
        ty = torch.tensor(y_pool_n[pool_selected_local]).float().to(DEVICE)
        lik  = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        kern = build_kernel(kernel).to(DEVICE)
        with gpytorch.settings.cholesky_jitter(1e-3):
            model = GPModel(tx, ty, lik, kern).to(DEVICE)
            train_gp(tx, ty, lik, model)

        # Evaluate on held-out test set
        mu_test, _ = _predict(X_test_pca, model, lik, np.arange(TEST_SIZE))
        rho_test, _ = spearmanr(y_test, mu_test)

        # Full-pool Spearman (for reference)
        mu_pool, _ = _predict(X_pool_pca, model, lik, np.arange(len(X_pool)))
        rho_pool, _ = spearmanr(y_pool, mu_pool)

        # Recall@2% of top-2% pool variants acquired so far
        t2_found = int(top2_pool[pool_selected_local].sum())
        t2_total = int(top2_pool.sum())

        # Top-10 mean affinity of acquired variants
        top10_mean = float(np.sort(y_pool[pool_selected_local])[::-1][:10].mean())

        rows.append({
            "cycle":          cycle,
            "n_labelled":     len(pool_selected_local),
            "spearman_test":  round(float(rho_test) if not np.isnan(rho_test) else 0.0, 4),
            "spearman":       round(float(rho_pool) if not np.isnan(rho_pool) else 0.0, 4),
            "recall_2p":      round(t2_found / max(1, t2_total), 4),
            "top_mean_at_10": round(top10_mean, 4),
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        if cycle == CYCLES:
            break
        best_yn = float(ty.max().cpu())
        new = acquire_bindinggym(protocol, X_pool_pca, model, lik,
                                 pool_selected_local, BATCH_SIZE, rng, best_yn)
        if not new:
            break
        pool_selected_local = pool_selected_local + new

        del model, kern, lik, tx, ty
        torch.cuda.empty_cache()

    last = rows[-1]
    print(f"    {dataset} | {protocol} | {kernel} | seed={seed} "
          f"→ test_rho={last['spearman_test']:+.4f}  recall@2%={last['recall_2p']:.4f}")
    return pd.DataFrame(rows)


def main():
    combos  = list(product(DATASETS, PROTOCOLS, KERNELS, SEEDS))
    pending = [(d,p,k,s) for d,p,k,s in combos
               if not (OUT_DIR/d/f"{p}_{k}_seed{s}.csv").exists()]
    print(f"BindingGYM benchmark — Total: {len(combos)}  Pending: {len(pending)}")

    t0 = time.perf_counter()
    for i, (ds, proto, kern, seed) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {ds} | {proto} | {kern} | seed={seed}")
        run_bindinggym(ds, proto, kern, seed)
    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min")
    print(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
