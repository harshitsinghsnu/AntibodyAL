"""
Preprocess the 3 BindingGYM DMS datasets used by ALLM-Ab (Furui & Ohue, JCIM 2025)
into the same vh_sequence/vl_sequence/affinity/top_2p/top_5p parquet format used
by the rest of this project, for a same-data comparison.

Source: https://zenodo.org/records/12514160 (BindingGYM, MIT license)
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(r"D:\explainable_AL\data\bindinggym\input\Binding_substitutions_DMS")
OUT_DIR = Path(r"D:\explainable_AL\data\processed_bindinggym")

# (filename, heavy_chain_key, light_chain_key)
DATASETS = {
    "4d5_her2": ("4D5_HER2_fitness_1N8Z.csv", "B", "A"),
    "5a12_ang2": ("5A12_Ang2_fitness_4ZFG.csv", "H", "L"),
    "5a12_vegf": ("5A12_VEGF_fitness_4ZFF.csv", "H", "L"),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (fname, h_key, l_key) in DATASETS.items():
        df = pd.read_csv(RAW_DIR / fname)
        mutated = df["mutated_sequence"].apply(ast.literal_eval)
        vh = mutated.apply(lambda d: d[h_key])
        vl = mutated.apply(lambda d: d[l_key])
        affinity = df["DMS_score"].astype(np.float32)

        top2_thresh = affinity.quantile(0.98)
        top5_thresh = affinity.quantile(0.95)
        top_2p = (affinity >= top2_thresh).astype(np.int8)
        top_5p = (affinity >= top5_thresh).astype(np.int8)

        out = pd.DataFrame({
            "vh_sequence": vh.values,
            "vl_sequence": vl.values,
            "affinity": affinity.values,
            "top_2p": top_2p.values,
            "top_5p": top_5p.values,
            "dataset_name": name,
        })
        out_path = OUT_DIR / f"{name}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"{name}: N={len(out)}  top2={int(top_2p.sum())}  top5={int(top_5p.sum())}  "
              f"affinity=[{affinity.min():.3f}, {affinity.max():.3f}]  -> {out_path}")


if __name__ == "__main__":
    main()
