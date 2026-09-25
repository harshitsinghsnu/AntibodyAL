"""
AA representation ablation: Bag-of-AA(n=5), BLOSUM62-Eigen, One-Hot
Ã— Tanimoto+UCB Ã— 8 datasets Ã— 3 seeds.
Results â†’ results_final/aa_rep/{rep}/{dataset}/
"""
from __future__ import annotations
import sys, time
from itertools import product
from pathlib import Path

from antibody_al.core import DATASETS, SEEDS, run_experiment

ROOT     = Path(__file__).resolve().parent.parent
EMB_ROOT = ROOT / "data"
OUT_DIR  = ROOT / "results_final" / "aa_rep"

AA_REPS = {
    "bag_of_aa5":    EMB_ROOT / "embeddings_bag_of_aa5",
    "blosum_eigen":  EMB_ROOT / "embeddings_blosum_eigen",
    "one_hot":       EMB_ROOT / "embeddings_one_hot",
}

def main():
    combos  = list(product(AA_REPS.items(), DATASETS, SEEDS))
    pending = [((n,p),d,s) for (n,p),d,s in combos
               if not (OUT_DIR/n/d/f"ucb_tanimoto_seed{s}.csv").exists()]
    print(f"Total: {len(combos)}  Pending: {len(pending)}")

    t0 = time.perf_counter()
    for i, ((rep_name, emb_root), ds, seed) in enumerate(pending, 1):
        out = OUT_DIR / rep_name / ds / f"ucb_tanimoto_seed{seed}.csv"
        print(f"[{i}/{len(pending)}] {rep_name} | {ds} | seed={seed}")
        run_experiment(emb_root/ds, out, ds, "ucb", "tanimoto", random_seed=seed)
    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()

