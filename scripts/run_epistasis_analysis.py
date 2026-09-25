"""
Landscape epistasis quantification Section 4.4.

Computes two complementary landscape-ruggedness proxies for each dataset:

  lin_R2 : 5-fold cross-validated R² of ridge regression on 128-dim ESM-2
            embeddings.  Low values indicate strong epistasis among mutated
            positions (Poelwijk et al., 2019).

  r_nn   : Spearman correlation between a variant's affinity and the mean
            affinity of its 10 nearest neighbours in embedding space.
            High values indicate a smooth, locally predictable landscape.

Paper results (Table in Section 4.4):
  Dataset      lin_R²   r_nn   Recall@2%
  3gbn_h1      0.903   0.976    0.895
  3gbn_h9      0.868   0.975    0.955
  4fqi_h3      0.586   0.582    1.000
  aayl49       0.255   0.384    0.410
  aayl51       0.159   0.321    0.376
  aayl49_ml    0.146   0.330    0.304
  aayl50       0.259   0.326    0.626
  aayl52       0.323   0.418    0.817

  Influenza mean lin_R² = 0.785,  r_nn = 0.844
  AAYL mean    lin_R² = 0.228,  r_nn = 0.356
  Spearman(lin_R², Recall@2%) = 0.905   (strongest predictor)

Usage:
    python scripts/run_epistasis_analysis.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
OUT_CSV = ROOT / "results_final" / "compiled" / "epistasis_summary.csv"

DATASETS = [
    "3gbn_h1", "3gbn_h9", "4fqi_h3",
    "aayl49", "aayl51", "aayl49_ml", "aayl50", "aayl52",
]

# Tanimoto+UCB Recall@2% from paper (mean across 3 seeds, final cycle)
PAPER_RECALL = {
    "3gbn_h1":   0.895,
    "3gbn_h9":   0.955,
    "4fqi_h3":   1.000,
    "aayl49":    0.410,
    "aayl51":    0.376,
    "aayl49_ml": 0.304,
    "aayl50":    0.626,
    "aayl52":    0.817,
}

SUBSAMPLE = 2000  # subsample for large datasets to keep runtime manageable
N_NEIGHBOURS = 10
KF_SPLITS = 5
RIDGE_ALPHA = 1.0
RANDOM_STATE = 42


def compute_lin_r2(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold cross-validated R² of Ridge regression on raw embeddings."""
    kf = KFold(n_splits=KF_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in kf.split(X):
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(scores))


def compute_r_nn(X: np.ndarray, y: np.ndarray, k: int = N_NEIGHBOURS) -> float:
    """Spearman(y_i, mean affinity of k nearest neighbours)."""
    nn  = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self
    knn_mean = y[idx].mean(axis=1)
    return float(spearmanr(y, knn_mean).correlation)


def analyse_dataset(dataset: str) -> dict:
    emb_dir = EMB_DIR / dataset
    X = np.load(emb_dir / "X.npy")
    y = np.load(emb_dir / "y.npy")

    # Subsample for large datasets
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X) > SUBSAMPLE:
        idx = rng.choice(len(X), SUBSAMPLE, replace=False)
        X, y = X[idx], y[idx]

    lin_r2 = compute_lin_r2(X, y)
    r_nn   = compute_r_nn(X, y)
    recall = PAPER_RECALL[dataset]

    print(f"  {dataset:<14}  lin_R²={lin_r2:.3f}  r_nn={r_nn:.3f}  "
          f"recall@2%={recall:.3f}")
    return {
        "dataset":   dataset,
        "lin_r2":    round(lin_r2, 3),
        "r_nn":      round(r_nn, 3),
        "recall_2p": recall,
    }


def main():
    print("=== Epistasis quantification ===\n")
    rows = [analyse_dataset(ds) for ds in DATASETS]
    df = pd.DataFrame(rows)

    # Group summaries
    influenza = df[df.dataset.isin(["3gbn_h1", "3gbn_h9", "4fqi_h3"])]
    aayl      = df[~df.dataset.isin(["3gbn_h1", "3gbn_h9", "4fqi_h3"])]
    print(f"\nInfluenza:  lin_R²={influenza.lin_r2.mean():.3f}±{influenza.lin_r2.std():.3f}"
          f"  r_nn={influenza.r_nn.mean():.3f}±{influenza.r_nn.std():.3f}")
    print(f"AAYL comb.: lin_R²={aayl.lin_r2.mean():.3f}±{aayl.lin_r2.std():.3f}"
          f"  r_nn={aayl.r_nn.mean():.3f}±{aayl.r_nn.std():.3f}")

    rho_lin, _ = spearmanr(df.lin_r2, df.recall_2p)
    rho_rnn, _ = spearmanr(df.r_nn,   df.recall_2p)
    print(f"\nSpearman(lin_R², Recall@2%) = {rho_lin:.3f}  "
          f"(paper: 0.905)")
    print(f"Spearman(r_nn,   Recall@2%) = {rho_rnn:.3f}  "
          f"(paper: 0.786)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
