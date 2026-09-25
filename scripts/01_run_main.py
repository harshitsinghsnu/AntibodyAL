"""
Main benchmark: 4 kernels Ã— 4 protocols Ã— 8 datasets Ã— 3 seeds = 384 runs.
10% unified budget (v1-eq). Results â†’ results_final/main/
"""
from __future__ import annotations
import sys, time
from itertools import product
from pathlib import Path

from antibody_al.core import DATASETS, KERNELS, PROTOCOLS, SEEDS, run_experiment

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "results_final" / "main"

def main():
    # 4fqi_h3 (65k variants): Tanimoto kernel only â€” other kernels take 20+ min/run
    # All other datasets: full 4-kernel sweep
    combos = []
    for d in DATASETS:
        ks = ["tanimoto"] if d == "4fqi_h3" else KERNELS
        for k in ks:
            for p in PROTOCOLS:
                for s in SEEDS:
                    combos.append((d, k, p, s))

    pending = [(d,k,p,s) for d,k,p,s in combos
               if not (OUT_DIR/d/f"{p}_{k}_seed{s}.csv").exists()]
    print(f"Total: {len(combos)}  Pending: {len(pending)}")

    t0 = time.perf_counter()
    for i,(ds,kern,proto,seed) in enumerate(pending, 1):
        out = OUT_DIR / ds / f"{proto}_{kern}_seed{seed}.csv"
        print(f"[{i}/{len(pending)}] {ds} | {kern} | {proto} | seed={seed}")
        run_experiment(EMB_DIR/ds, out, ds, proto, kern, random_seed=seed)
    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()

