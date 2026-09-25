"""
Preprocess all AbBiBench parquet files.
  1. Standardise column names to: vh_sequence, vl_sequence, affinity
  2. Drop rows missing VH, VL, or affinity
  3. Add top_2p flag  (top 2% affinity within each dataset)
  4. Add top_5p flag  (top 5% affinity within each dataset)
  5. Concatenate: sequence = VH[:512] + "G" + VL[:512]
  6. Save to data/processed/{dataset}.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path(r"D:\explainable_AL\data\raw")
OUT_DIR  = Path(r"D:\explainable_AL\data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── confirmed names ──────────────────────────────────────────────────────────
VH_COL       = "vh_sequence"   # actual column name from parquet
VL_COL       = "vl_sequence"   # actual column name from parquet
AFFINITY_COL = "neg_log_kd"    # actual column name from parquet
# ─────────────────────────────────────────────────────────────────────────────

LC_DATASETS = {"aayl50", "aayl52"}   # VH is constant; VL varies

for path in sorted(RAW_DIR.glob("*.parquet")):
    name = path.stem
    df   = pd.read_parquet(path)

    # Rename to canonical names
    df = df.rename(columns={
        VH_COL:       "vh_sequence",
        VL_COL:       "vl_sequence",
        AFFINITY_COL: "affinity",
    })

    # Drop missing
    before = len(df)
    df = df.dropna(subset=["vh_sequence", "vl_sequence", "affinity"])
    df = df[(df["vh_sequence"].str.strip() != "") &
            (df["vl_sequence"].str.strip() != "")]
    print(f"{name}: {before} → {len(df)} rows after cleaning")

    # Top-2% and top-5% flags (within dataset, not globally)
    t2 = df["affinity"].quantile(0.98)
    t5 = df["affinity"].quantile(0.95)
    df["top_2p"] = (df["affinity"] >= t2).astype(int)
    df["top_5p"] = (df["affinity"] >= t5).astype(int)
    print(f"  top_2p >= {t2:.3f}  count={df['top_2p'].sum()}")
    print(f"  top_5p >= {t5:.3f}  count={df['top_5p'].sum()}")

    # Concatenated sequence: VH + glycine separator + VL
    # Truncate each chain to 512 so combined <= 1025 (within ESM-2 limit)
    df["sequence"] = (
        df["vh_sequence"].str[:512] + "G" + df["vl_sequence"].str[:512]
    )

    df["dataset"]       = name
    df["is_lc_dataset"] = name in LC_DATASETS

    out = OUT_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    print(f"  → {out}\n")
