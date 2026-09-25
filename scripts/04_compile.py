"""
Compile all results into summary DataFrames for figure generation.
Outputs â†’ results_final/compiled/
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from antibody_al.core import DATASETS, KERNELS, PROTOCOLS, SEEDS, N_TOTAL, CONFIGS

BASE  = Path(__file__).resolve().parent.parent / "results_final"
OUT   = BASE / "compiled"
OUT.mkdir(parents=True, exist_ok=True)

def last_row(csv):
    df = pd.read_csv(csv)
    return df.iloc[-1].to_dict()

def all_rows(csv):
    return pd.read_csv(csv)

# â”€â”€ 1. Main benchmark summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compile_main():
    rows = []
    for ds in DATASETS:
        for kern in KERNELS:
            for proto in PROTOCOLS:
                vals_r2, vals_r5, vals_rho = [], [], []
                curves = []
                for seed in SEEDS:
                    f = BASE/"main"/ds/f"{proto}_{kern}_seed{seed}.csv"
                    if not f.exists(): continue
                    lr = last_row(f)
                    vals_r2.append(float(lr["recall_2p"]))
                    vals_r5.append(float(lr["recall_5p"]))
                    vals_rho.append(float(lr["spearman_fixed_test"]))
                    curves.append(all_rows(f))
                if not vals_r2: continue
                rows.append({
                    "dataset":  ds, "kernel": kern, "protocol": proto,
                    "n_seeds":  len(vals_r2),
                    "recall_2p_mean": round(np.mean(vals_r2),4),
                    "recall_2p_std":  round(np.std(vals_r2,ddof=1) if len(vals_r2)>1 else 0,4),
                    "recall_5p_mean": round(np.mean(vals_r5),4),
                    "recall_5p_std":  round(np.std(vals_r5,ddof=1) if len(vals_r5)>1 else 0,4),
                    "spearman_mean":  round(np.mean(vals_rho),4),
                    "spearman_std":   round(np.std(vals_rho,ddof=1) if len(vals_rho)>1 else 0,4),
                    "budget_pct":     round(float(last_row(BASE/"main"/ds/
                        f"{proto}_{kern}_seed{SEEDS[0]}.csv").get("pct_labelled",10)),2),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"main_summary.csv", index=False)
    print(f"Main: {len(df)} rows")
    return df

# â”€â”€ 2. Budget curves (for panel A) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compile_curves():
    all_curves = []
    for ds in DATASETS:
        for seed in SEEDS:
            f = BASE/"main"/ds/f"ucb_tanimoto_seed{seed}.csv"
            if not f.exists(): continue
            df = all_rows(f); df["dataset"] = ds; df["seed"] = seed
            all_curves.append(df)
    if all_curves:
        out = pd.concat(all_curves, ignore_index=True)
        out.to_csv(OUT/"budget_curves.csv", index=False)
        print(f"Budget curves: {len(out)} rows")

# â”€â”€ 3. PLM summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compile_plm():
    PLMS = ["esm2","protbert","antiberty_concat","antiberty_paired","ablang2","progen2"]
    rows = []
    for plm in PLMS:
        for ds in DATASETS:
            vals = []
            for seed in SEEDS:
                f = BASE/"plm"/plm/ds/f"ucb_tanimoto_seed{seed}.csv"
                if f.exists(): vals.append(float(last_row(f)["recall_2p"]))
            if vals:
                rows.append({"plm":plm,"dataset":ds,
                             "recall_2p_mean":round(np.mean(vals),4),
                             "recall_2p_std": round(np.std(vals,ddof=1) if len(vals)>1 else 0,4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"plm_summary.csv", index=False)
    print(f"PLM: {len(df)} rows")
    return df

# â”€â”€ 4. AA rep summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compile_aa_rep():
    REPS = ["bag_of_aa5","blosum_eigen","one_hot"]
    rows = []
    for rep in REPS:
        for ds in DATASETS:
            vals = []
            for seed in SEEDS:
                f = BASE/"aa_rep"/rep/ds/f"ucb_tanimoto_seed{seed}.csv"
                if f.exists(): vals.append(float(last_row(f)["recall_2p"]))
            if vals:
                rows.append({"rep":rep,"dataset":ds,
                             "recall_2p_mean":round(np.mean(vals),4),
                             "recall_2p_std": round(np.std(vals,ddof=1) if len(vals)>1 else 0,4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"aa_rep_summary.csv", index=False)
    print(f"AA rep: {len(df)} rows")
    return df

if __name__ == "__main__":
    compile_main()
    compile_curves()
    compile_plm()
    compile_aa_rep()
    print("Compilation done â†’", OUT)

