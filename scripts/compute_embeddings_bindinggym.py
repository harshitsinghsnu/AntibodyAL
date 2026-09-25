"""
Compute S0 (VH+G+VL concatenation) ESM-2 embeddings for the 3 BindingGYM
datasets used by ALLM-Ab, for a same-data comparison against that paper.

Output layout matches data/embeddings/{dataset}/ (X.npy, y.npy, top2p.npy, top5p.npy).
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from antibody_al.featuriser import ESM2AbFeaturiser

PROCESSED = Path(r"D:\explainable_AL\data\processed_bindinggym")
EMB_DIR   = Path(r"D:\explainable_AL\data\embeddings_bindinggym")
CACHE     = r"D:\explainable_AL\cache\ab_embeddings_bindinggym.h5"

DATASETS = ["4d5_her2", "5a12_ang2", "5a12_vegf"]

feat = ESM2AbFeaturiser(model_size="650m", device="auto",
                        batch_size=32, cache_path=CACHE)

for ds in DATASETS:
    p = PROCESSED / f"{ds}.parquet"
    if not p.exists():
        print(f"SKIP {ds}"); continue
    df = pd.read_parquet(p)
    print(f"\n{ds}: {len(df):,} sequences")

    sequence = df["vh_sequence"].str[:512] + "G" + df["vl_sequence"].str[:512]
    X = feat.transform(sequence.tolist())
    y = df["affinity"].values.astype(np.float32)

    out = EMB_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "top2p.npy", df["top_2p"].values.astype(np.int8))
    np.save(out / "top5p.npy", df["top_5p"].values.astype(np.int8))

    with open(out / "meta.json", "w") as f:
        json.dump({
            "dataset": ds, "n": len(df), "dim": int(X.shape[1]),
            "affinity_min": float(y.min()), "affinity_max": float(y.max()),
            "affinity_mean": float(y.mean()),
            "top2p_count": int(df["top_2p"].sum()),
            "top5p_count": int(df["top_5p"].sum()),
            "source": "BindingGYM (Zenodo 12514160), used by ALLM-Ab (Furui & Ohue, JCIM 2025)",
        }, f, indent=2)
    print(f"  OK X={X.shape} -> {out}/")
