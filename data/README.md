# Data

## AbBiBench Datasets

The eight antibody datasets used in this paper come from **AbBiBench** (Antibody Binding Benchmark).

| Dataset     | Tier   | N variants | Description                           |
|-------------|--------|-----------|---------------------------------------|
| 3gbn_h1     | Small  | 1,887     | 3GBN antigen, heavy-chain library H1  |
| 3gbn_h9     | Small  | 1,842     | 3GBN antigen, heavy-chain library H9  |
| aayl49      | Small  | 4,312     | AAYL49 humanised Ab, VH mutants       |
| aayl51      | Small  | 4,320     | AAYL51 humanised Ab, VH mutants       |
| aayl49_ml   | Medium | 8,953     | AAYL49 ML-expanded library            |
| aayl50      | Medium | 11,473    | AAYL50 humanised Ab, VH+VL mutants    |
| aayl52      | Medium | 13,324    | AAYL52 humanised Ab, VH+VL mutants    |
| 4fqi_h3     | Large  | 65,535    | 4FQI antigen, heavy-chain CDR-H3 lib  |

Download and place each CSV in `data/raw/{dataset}.csv`.  
Columns required: `vh` (VH sequence), `vl` (VL sequence), `affinity` (binding affinity, e.g. ΔΔG or -log Kd).

## BindingGYM Datasets (Section 5.2 comparison)

Download from the BindingGYM repository:
- `5A12_Ang2` (Angiopoietin-2 antibody)
- `4D5_HER2`  (HER2/ErbB2 antibody)

Place CSVs in `data/raw/{dataset}.csv` and run `python scripts/01_compute_embeddings.py --emb-dir data/embeddings_bindinggym`.

## Computing Embeddings

```bash
# ESM-2 650M (main paper) — writes data/embeddings/{dataset}/{X,y,top2p,top5p}.npy
conda activate al
python scripts/01_compute_embeddings.py

# PLM ablation embeddings
# ProtBert
python scripts/compute_embeddings_protbert.py   # data/embeddings_protbert/
# AntiBERTy (VH+VL concat input)
python scripts/compute_embeddings_antiberty.py  # data/embeddings_antiberty/
# AntiBERTy (paired — separate VH/VL then concat embeddings)
python scripts/compute_embeddings_antiberty_paired.py
# AbLang2
python scripts/compute_embeddings_ablang2.py    # data/embeddings_ablang2/
# ProGen2 (see below)
python scripts/compute_embeddings_progen2.py    # data/embeddings_progen2/
```

### ProGen2 installation

ProGen2 is not on PyPI. Install from source:
```bash
git clone https://github.com/salesforce/progen
pip install -e progen/progen2
```

## PDB Structures

Crystal structures for paratope recall analysis. Download from RCSB PDB and place in `data/structures/`:

| File                  | PDB ID | Description              |
|-----------------------|--------|--------------------------|
| `3GBN.pdb`            | 3GBN   | Anti-VEGF Fab complex    |
| `4FQI.pdb`            | 4FQI   | Anti-lysozyme Fab complex|
| `AAYL49_bca.pdb`      | 8A6D   | AAYL49 Fab-antigen       |
| `AAYL50_bca.pdb`      | 8A6E   | AAYL50 Fab-antigen       |
| `AAYL51_bca.pdb`      | 8A6F   | AAYL51 Fab-antigen       |
| `AAYL52_bca.pdb`      | 8A6G   | AAYL52 Fab-antigen       |
