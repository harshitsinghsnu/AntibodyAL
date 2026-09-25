"""
PLM ablation: 6 PLMs Ã— Tanimoto+UCB Ã— 8 datasets Ã— 3 seeds.
Results â†’ results_final/plm/{plm}/{dataset}/
"""
from __future__ import annotations
import sys, time
from itertools import product
from pathlib import Path

from antibody_al.core import DATASETS, SEEDS, run_experiment

ROOT     = Path(__file__).resolve().parent.parent
EMB_ROOT = ROOT / "data"
OUT_DIR  = ROOT / "results_final" / "plm"

PLMS = {
    "esm2":             EMB_ROOT / "embeddings",
    "protbert":         EMB_ROOT / "embeddings_protbert",
    "antiberty_concat": EMB_ROOT / "embeddings_antiberty",
    "antiberty_paired": EMB_ROOT / "embeddings_antiberty_paired",
    "ablang2":          EMB_ROOT / "embeddings_ablang2",
    "progen2":          EMB_ROOT / "embeddings_progen2",
}

def main():
    combos  = list(product(PLMS.items(), DATASETS, SEEDS))
    pending = [((n,p),d,s) for (n,p),d,s in combos
               if not (OUT_DIR/n/d/f"ucb_tanimoto_seed{s}.csv").exists()]
    print(f"Total: {len(combos)}  Pending: {len(pending)}")

    t0 = time.perf_counter()
    for i, ((plm_name, emb_root), ds, seed) in enumerate(pending, 1):
        out = OUT_DIR / plm_name / ds / f"ucb_tanimoto_seed{seed}.csv"
        print(f"[{i}/{len(pending)}] {plm_name} | {ds} | seed={seed}")
        run_experiment(emb_root/ds, out, ds, "ucb", "tanimoto", random_seed=seed)
    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()

