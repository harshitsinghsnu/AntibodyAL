"""
Computes three classical baseline featurizations of the concatenated
VH+"G"+VL antibody sequence (matching the S0-Concat representation used
as the main model throughout the paper) for ablation comparison against
ESM-2 650M embeddings.

(1) One-hot: each of the 20 standard amino acids gets a unique 20-dim
    one-hot vector; the per-sequence representation is the mean of all
    per-position one-hots (= amino acid composition vector, 20-dim).
    Fixed-size regardless of sequence length, naturally interpretable.
    "Each letter in the alphabet of amino acids gets assigned a unique
    one-hot vector and the encoding is their concatenation" (per-position)
    — we mean-pool to collapse to fixed size, consistent with how all other
    representations here are mean-pooled to a fixed-size embedding before PCA.

(2) Bag of AA (n=5 k-mers): count-vector of all 5-gram amino acid substrings
    in the concatenated sequence. Analogous to the bag-of-words embedding of
    Jurafsky & Martin (2000), set n=5. The vocabulary is hashed to a fixed
    dimension (1024 via hashing trick to keep it tractable; PCA reduces further).

(3) BLOSUM eigendecomposition: encode each amino acid as the row of U|D|^{1/2}
    from the eigendecomposition of BLOSUM62 (BLOSUM62 = U D U^T), motivated by
    Oglic & Gärtner 2018/2019 on flip-spectrum transformations of indefinite
    kernels. Mean-pool per-position BLOSUM-eigen vectors → 20-dim per sequence.

Outputs: data/embeddings_{one_hot,bag_of_aa5,blosum_eigen}/{dataset}/X.npy
(y.npy, top2p.npy, top5p.npy copied from data/embeddings/{dataset}/)
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA = Path(r"D:\explainable_AL\data")
ESM2_EMB = DATA / "embeddings"
DATASETS = ["3gbn_h1", "3gbn_h9", "4fqi_h1", "4fqi_h3",
            "aayl49", "aayl49_ml", "aayl50", "aayl51", "aayl52"]

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {aa: i for i, aa in enumerate(AA20)}

# ── BLOSUM62 20×20 matrix (standard, indexed by AA20 order) ─────────────────
_BLOSUM62_ROWS = {
    "A": [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    "R": [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    "N": [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    "D": [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    "C": [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    "Q": [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    "E": [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    "G": [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    "H": [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    "I": [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    "L": [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    "K": [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    "M": [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    "F": [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    "P": [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    "S": [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    "T": [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    "W": [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    "Y": [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    "V": [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
}
_BLOSUM62 = np.array([[_BLOSUM62_ROWS[a][j] for j in range(20)] for a in AA20],
                      dtype=np.float64)  # [20, 20]


def _blosum_eigen_encoder() -> np.ndarray:
    vals, vecs = np.linalg.eigh(_BLOSUM62)
    enc = vecs * np.abs(vals) ** 0.5          # [20, 20], rows = AA embedding
    return enc.astype(np.float32)


BLOSUM_ENC = _blosum_eigen_encoder()          # [20, 20] rows indexed by AA20


def one_hot_encode(seq: str) -> np.ndarray:
    vecs = []
    for aa in seq:
        v = np.zeros(20, dtype=np.float32)
        if aa in AA_IDX:
            v[AA_IDX[aa]] = 1.0
        vecs.append(v)
    return np.mean(vecs, axis=0) if vecs else np.zeros(20, dtype=np.float32)


def bag_of_aa5(seq: str, n_bins: int = 1024) -> np.ndarray:
    vec = np.zeros(n_bins, dtype=np.float32)
    for i in range(len(seq) - 4):
        kmer = seq[i:i + 5]
        bin_idx = hash(kmer) % n_bins
        vec[bin_idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def blosum_encode(seq: str) -> np.ndarray:
    vecs = []
    for aa in seq:
        if aa in AA_IDX:
            vecs.append(BLOSUM_ENC[AA_IDX[aa]])
    return np.mean(vecs, axis=0) if vecs else np.zeros(20, dtype=np.float32)


FEATURIZERS = {
    "one_hot":      (one_hot_encode, 20),
    "bag_of_aa5":   (bag_of_aa5, 1024),
    "blosum_eigen": (blosum_encode, 20),
}


def main():
    for fname, (fn, dim) in FEATURIZERS.items():
        out_root = DATA / f"embeddings_{fname}"
        for ds in DATASETS:
            out_dir = out_root / ds
            if (out_dir / "X.npy").exists():
                print(f"SKIP {fname}/{ds}")
                continue
            esm2_dir = ESM2_EMB / ds
            df = pd.read_parquet(DATA / "processed" / f"{ds}.parquet")
            y = np.load(esm2_dir / "y.npy")
            t2p = np.load(esm2_dir / "top2p.npy")
            t5p = np.load(esm2_dir / "top5p.npy")

            # Concatenate VH + "G" + VL — same as S0-Concat
            seqs = (df["vh_sequence"] + "G" + df["vl_sequence"]).tolist()
            print(f"  {fname}/{ds}: encoding {len(seqs):,} sequences (dim={dim})...")
            X = np.stack([fn(s) for s in seqs]).astype(np.float32)
            assert X.shape == (len(seqs), dim), X.shape

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "X.npy", X)
            np.save(out_dir / "y.npy", y)
            np.save(out_dir / "top2p.npy", t2p)
            np.save(out_dir / "top5p.npy", t5p)
            print(f"    OK X={X.shape} -> {out_dir}/")


if __name__ == "__main__":
    main()
