"""
UCB beta-schedule ablation. Section 3.5 of the paper describes six schedules;
this runs the five not covered by the main grid's b1.0 default (Tanimoto
kernel, corrected label-blind/held-out protocol, 3 seeds).
Results -> results_final/ucb_beta_ablation/
"""
from __future__ import annotations
import sys, time
from itertools import product
from pathlib import Path

from antibody_al.core import DATASETS, run_experiment

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "results_final" / "ucb_beta_ablation"
SEEDS   = [42, 123, 456]
VARIANTS = ["b0.1", "b0.5", "b2.0", "grad", "alt"]  # b1.0 already in the main grid

def main():
    combos = list(product(DATASETS, VARIANTS, SEEDS))
    pending = [(d, v, s) for d, v, s in combos
               if not (OUT_DIR / d / f"ucb_{v}_seed{s}.csv").exists()]
    print(f"Total: {len(combos)}  Pending: {len(pending)}")
    t0 = time.perf_counter()
    for i, (ds, variant, seed) in enumerate(pending, 1):
        out = OUT_DIR / ds / f"ucb_{variant}_seed{seed}.csv"
        print(f"[{i}/{len(pending)}] {ds} | ucb-{variant} | seed={seed}")
        run_experiment(EMB_DIR / ds, out, ds, "ucb", "tanimoto", random_seed=seed, beta_variant=variant)
    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
