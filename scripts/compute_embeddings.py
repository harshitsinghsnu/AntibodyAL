"""
Compute and cache ESM-2 embeddings for all processed datasets.

Output layout:
  data/embeddings/{dataset}/
      X.npy       float32 (N, 1280)   — ESM-2 embeddings
      y.npy       float32 (N,)        — affinity values (−log Kd)
      top2p.npy   int8    (N,)        — 1 if top 2% binder
      top5p.npy   int8    (N,)        — 1 if top 5% binder
      meta.json                        — dataset stats
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from antibody_al.featuriser import ESM2AbFeaturiser

PROCESSED = Path(r"D:\explainable_AL\data\processed")
EMB_DIR   = Path(r"D:\explainable_AL\data\embeddings")
CACHE     = r"D:\explainable_AL\cache\ab_embeddings.h5"

DATASETS  = ["3gbn_h9","aayl49","aayl49_ml","aayl51","aayl50","aayl52",
             "4fqi_h1","4fqi_h3"]

feat = ESM2AbFeaturiser(model_size="650m", device="auto",
                        batch_size=32, cache_path=CACHE)

for ds in DATASETS:
    p = PROCESSED / f"{ds}.parquet"
    if not p.exists():
        print(f"SKIP {ds}"); continue
    df = pd.read_parquet(p)
    print(f"\n{ds}: {len(df):,} sequences")

    X = feat.transform(df["sequence"].tolist())
    y = df["affinity"].values.astype(np.float32)

    out = EMB_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"X.npy",     X)
    np.save(out/"y.npy",     y)
    np.save(out/"top2p.npy", df["top_2p"].values.astype(np.int8))
    np.save(out/"top5p.npy", df["top_5p"].values.astype(np.int8))

    with open(out/"meta.json","w") as f:
        json.dump({"dataset":ds,"n":len(df),"dim":int(X.shape[1]),
                   "affinity_min":float(y.min()),"affinity_max":float(y.max()),
                   "affinity_mean":float(y.mean()),"top2p_count":int(df["top_2p"].sum()),
                   "top5p_count":int(df["top_5p"].sum())}, f, indent=2)
    print(f"  ✓ X={X.shape} → {out}/")
