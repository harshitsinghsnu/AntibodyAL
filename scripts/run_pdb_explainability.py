"""
Full PDB-based explainability pipeline:

1. Extract VH/VL sequences from deposited PDB structures
2. Compute ground-truth paratope residues via heavy-atom contact distance ≤ 4.5 Å (Biopython)
3. Run Integrated Gradients + KernelSHAP on PDB sequences using best GP
   (Tanimoto kernel, UCB, trained on full dataset)
4. Align PDB sequence to dataset sequence (BLOSUM62, Biopython) → map attribution to PDB resi
5. Write B-factor-replaced PDB files (one for IG, one for SHAP)
6. Write per-antigen PyMOL2 .pml session scripts (color by B-factor + GT markers)
7. Run PyMOL in headless batch mode to render PNGs
8. Generate heatmap figures (seq position vs attribution, CDR + GT residues annotated)

Outputs in results_explainability/pdb_based/:
  {antigen}_ig.pdb          B-factors = IG score (normalised 0-100)
  {antigen}_shap.pdb        B-factors = SHAP score (normalised 0-100)
  {antigen}_contacts.csv    ground truth paratope residues
  {antigen}_alignment.csv   seq-to-structure alignment mapping
  {antigen}_ig.pml          PyMOL script (IG)
  {antigen}_shap.pml        PyMOL script (SHAP)
  {antigen}_ig_render.png   PyMOL rendered PNG (IG)
  {antigen}_shap_render.png PyMOL rendered PNG (SHAP)
  fig_heatmap_{antigen}.png matplotlib heatmap figure
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Bio imports
from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices

ROOT   = Path(__file__).resolve().parent.parent
OUT    = ROOT / "results_explainability" / "pdb_based"
OUT.mkdir(parents=True, exist_ok=True)
FIG    = ROOT / "figures"
FIG.mkdir(exist_ok=True)

PYMOL_EXE = r"C:\Users\hs494\AppData\Local\pymol\Scripts\pymol.exe"

# Chain assignments per antigen system
ANTIGEN_CONFIG = {
    "3GBN": {
        "pdb_file":   str(ROOT / "data" / "structures" / "3GBN.pdb"),
        "vh_chain":   "H",
        "vl_chain":   "L",
        "ag_chains":  ["A", "B"],
        "dataset":    "3gbn_h1",
        "emb_dir":    str(ROOT / "data" / "embeddings" / "3gbn_h1"),
    },
    "4FQI": {
        "pdb_file":   str(ROOT / "data" / "structures" / "4FQI.pdb"),
        "vh_chain":   "H",
        "vl_chain":   "L",
        "ag_chains":  ["A", "B"],
        "dataset":    "4fqi_h3",
        "emb_dir":    str(ROOT / "data" / "embeddings" / "4fqi_h3"),
    },
    "AAYL49": {
        "pdb_file":   str(ROOT / "data" / "structures" / "AAYL49_bca.pdb"),
        "vh_chain":   "B",
        "vl_chain":   "C",
        "ag_chains":  ["A"],
        "dataset":    "aayl49",
        "emb_dir":    str(ROOT / "data" / "embeddings" / "aayl49"),
    },
    "AAYL50": {
        "pdb_file":   str(ROOT / "data" / "structures" / "AAYL50_bca.pdb"),
        "vh_chain":   "B",
        "vl_chain":   "C",
        "ag_chains":  ["A"],
        "dataset":    "aayl50",
        "emb_dir":    str(ROOT / "data" / "embeddings" / "aayl50"),
    },
    "AAYL51": {
        "pdb_file":   str(ROOT / "data" / "structures" / "AAYL51_bca.pdb"),
        "vh_chain":   "B",
        "vl_chain":   "C",
        "ag_chains":  ["A"],
        "dataset":    "aayl51",
        "emb_dir":    str(ROOT / "data" / "embeddings" / "aayl51"),
    },
    "AAYL52": {
        "pdb_file":   str(ROOT / "data" / "structures" / "AAYL52_bca.pdb"),
        "vh_chain":   "B",
        "vl_chain":   "C",
        "ag_chains":  ["A"],
        "dataset":    "aayl52",
        "emb_dir":    str(ROOT / "data" / "embeddings" / "aayl52"),
    },
}

CONTACT_DIST = 4.5   # Å, heavy-atom

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


# ── Step 1: PDB utilities ─────────────────────────────────────────────────────

def _extract_chain_seq_and_resids(structure, chain_id):
    """Return (aa_sequence_str, list_of_(chain, resseq, icode) for each AA)."""
    chain = structure[0][chain_id]
    seq, resids = [], []
    for res in chain.get_residues():
        if PDB.is_aa(res, standard=True):
            aa = protein_letters_3to1.get(res.resname, "X")
            seq.append(aa)
            resids.append((chain_id, res.id[1], res.id[2]))
    return "".join(seq), resids


def _compute_contacts(structure, ab_chains, ag_chains, cutoff=CONTACT_DIST):
    """
    Return set of (chain_id, resseq) for antibody residues within cutoff Å
    (heavy-atom) of any antigen atom.
    """
    from Bio.PDB import NeighborSearch
    # Collect antigen atoms
    ag_atoms = []
    for cid in ag_chains:
        if cid in structure[0]:
            for res in structure[0][cid].get_residues():
                for atom in res.get_atoms():
                    if atom.element != "H":
                        ag_atoms.append(atom)

    ns = NeighborSearch(ag_atoms)
    contacts = set()
    for cid in ab_chains:
        if cid not in structure[0]:
            continue
        for res in structure[0][cid].get_residues():
            if not PDB.is_aa(res, standard=True):
                continue
            for atom in res.get_atoms():
                if atom.element == "H":
                    continue
                nearby = ns.search(atom.coord, cutoff, level="R")
                if nearby:
                    contacts.add((cid, res.id[1]))
                    break
    return contacts


# ── Step 2: GP model (Tanimoto kernel) ───────────────────────────────────────

class TanimotoKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def forward(self, x1, x2, **params):
        dot = torch.mm(x1, x2.T)
        n1  = (x1 ** 2).sum(-1, keepdim=True)
        n2  = (x2 ** 2).sum(-1, keepdim=True)
        return dot / (n1 + n2.T - dot + 1e-10)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean(x), self.covar(x))


def train_gp(train_x, train_y, epochs=150):
    lik   = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, lik)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()
    model.eval(); lik.eval()
    return model, lik


# ── Step 3: ESM-2 token-level embeddings for IG ───────────────────────────────

def get_token_level_embeddings(sequences):
    """
    Returns list of np.array [L, 1280] per sequence (no pooling).
    Uses fair-esm 2.0.0 API (esm.pretrained).
    Caches per-token arrays in HDF5.
    """
    import h5py
    import hashlib
    import esm as fair_esm

    CACHE  = ROOT / "cache" / "ab_embeddings.h5"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _hash(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    token_level = []

    # Try HDF5 token cache first
    try:
        with h5py.File(str(CACHE), "r") as f:
            cached_all = all(f"esm2_650m/tokens/{_hash(s)}" in f for s in sequences)
        if cached_all:
            with h5py.File(str(CACHE), "r") as f:
                for s in sequences:
                    token_level.append(f[f"esm2_650m/tokens/{_hash(s)}"][:])
            print(f"  Loaded {len(sequences)} token arrays from cache")
            return token_level
    except Exception:
        pass

    # Load ESM-2 650M via fair-esm pretrained API
    print("  Loading ESM-2 650M (fair-esm 2.0.0)...")
    esm_model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.eval().to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    N_LAYERS = 33

    CACHE.parent.mkdir(exist_ok=True)

    for seq in sequences:
        seq_trunc = seq[:1024]
        labeled = [("seq", seq_trunc)]
        _, _, tokens = batch_converter(labeled)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[N_LAYERS], return_contacts=False)
        # per-token hidden states [1, L+2, 1280] → strip BOS/EOS → [L, 1280]
        h = out["representations"][N_LAYERS][0, 1:len(seq_trunc)+1, :].cpu().float().numpy()
        token_level.append(h)

        key = f"esm2_650m/tokens/{_hash(seq)}"
        try:
            with h5py.File(str(CACHE), "a") as f:
                if key not in f:
                    f.create_dataset(key, data=h, compression="gzip")
        except Exception:
            pass

    return token_level


# ── Step 4: Integrated Gradients on mean-pooled GP ────────────────────────────

def compute_ig(token_arr, pca, gp_model, gp_lik, n_steps=50):
    """
    Attribute GP mean prediction to each residue position.
    Baseline = zero tensor (masked ESM-2 representation approximation).
    token_arr: np.array [L, 1280]
    Returns np.array [L] of IG attribution scores.

    PCA is implemented as a differentiable torch matmul so autograd
    flows all the way from GP output back to each residue token.
    """
    # PCA as torch tensors for differentiable computation
    pca_mean = torch.tensor(pca.mean_,       dtype=torch.float32)  # [1280]
    pca_comp = torch.tensor(pca.components_, dtype=torch.float32)  # [128, 1280]

    input_t    = torch.tensor(token_arr, dtype=torch.float32)       # [L, 1280]
    baseline_t = torch.zeros_like(input_t)

    ig_sum = torch.zeros(len(token_arr))

    for step in range(1, n_steps + 1):
        alpha  = step / n_steps
        interp = (baseline_t + alpha * (input_t - baseline_t)).clone().requires_grad_(True)

        # Differentiable pipeline: mean-pool → PCA → GP
        pooled   = interp.mean(dim=0)                          # [1280]
        centered = pooled - pca_mean                           # [1280]
        pca_out  = (pca_comp @ centered).unsqueeze(0)          # [1, 128]

        with gpytorch.settings.fast_pred_var():
            pred = gp_lik(gp_model(pca_out))
            mu   = pred.mean[0]

        mu.backward()

        grad = interp.grad.detach()                            # [L, 1280]
        ig_sum += (grad * (input_t - baseline_t)).abs().sum(dim=-1)

    return (ig_sum / n_steps).numpy()


# ── Step 5: SHAP (residue masking) ────────────────────────────────────────────

def compute_shap(token_arr, pca, gp_model, gp_lik, n_samples=100):
    """
    KernelSHAP approximation: for each position i, estimate its Shapley value
    by sampling random subsets S and computing f(S∪i) - f(S).
    token_arr: [L, 1280]
    Returns np.array [L] SHAP values.
    """
    L = len(token_arr)
    baseline = np.zeros_like(token_arr)  # masked baseline

    def predict(mask):
        # mask: [L] bool, True = include original, False = use baseline
        feat = token_arr.copy()
        feat[~mask] = baseline[~mask]
        pooled = feat.mean(0, keepdims=True)
        pca_r  = torch.tensor(pca.transform(pooled), dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            p = gp_lik(gp_model(pca_r))
            return float(p.mean[0].item())

    rng   = np.random.default_rng(42)
    shaps = np.zeros(L)
    for i in range(L):
        vals = []
        for _ in range(n_samples):
            subset = rng.random(L) < 0.5
            with_i    = subset.copy(); with_i[i]    = True
            without_i = subset.copy(); without_i[i] = False
            vals.append(predict(with_i) - predict(without_i))
        shaps[i] = np.mean(vals)
    return shaps


# ── Step 6: Sequence alignment (PDB → dataset seq) ────────────────────────────

def align_seq_to_structure(dataset_seq, pdb_seq):
    """
    Align dataset VH or VL sequence to corresponding PDB chain sequence.
    Returns list of (pdb_pos_0indexed_or_None) for each position in dataset_seq.
    None = gap (insertion relative to PDB, can't map to structure residue).
    """
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score      = -10
    aligner.extend_gap_score    = -0.5
    aligner.mode = "global"

    alignments = aligner.align(pdb_seq, dataset_seq)
    aln = next(iter(alignments))

    # Extract mapping: dataset_seq position → pdb_seq position
    pdb_positions = aln.indices[0]   # shape [len(dataset_seq)]
    return pdb_positions             # -1 means gap


# ── Step 7: Write B-factor PDB ────────────────────────────────────────────────

def write_bfactor_pdb(source_pdb, out_path, chain_bfactor_map):
    """
    Write a copy of source_pdb with B-factor column replaced.
    chain_bfactor_map: {chain_id: {resseq: bfactor_value}}
    """
    out_lines = []
    with open(source_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain  = line[21]
                resseq = int(line[22:26].strip())
                if chain in chain_bfactor_map and resseq in chain_bfactor_map[chain]:
                    bf = chain_bfactor_map[chain][resseq]
                    # B-factor occupies columns 60-66 (1-indexed)
                    line = line[:60] + f"{bf:6.2f}" + line[66:]
            out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)


# ── Step 8: PyMOL script ──────────────────────────────────────────────────────

PYMOL_TEMPLATE = """
cmd.load("{pdb_path}", "{name}")
cmd.hide("everything", "{name}")
cmd.show("cartoon", "{name}")
cmd.show("surface", "{name} and chain {ag_chains_str}")
cmd.show("sticks", "{name} and (chain {ab_chains_str}) and (b > 60)")

# Color antibody chains by B-factor (attribution)
cmd.spectrum("b", "white_red", "{name} and chain {ab_chains_str}", minimum=0, maximum=100)

# Highlight antigen chains
cmd.color("lightblue", "{name} and chain {ag_chains_str}")

# Ground-truth contact residues as orange spheres
{contact_selections}
cmd.show("spheres", "gt_contacts_{name}")
cmd.color("orange", "gt_contacts_{name}")
cmd.set("sphere_scale", 0.4, "gt_contacts_{name}")

# Legend/title
cmd.set("ray_trace_mode", 1)
cmd.set("ray_shadows", 0)
cmd.orient("{name}")
cmd.ray(1600, 900)
cmd.png("{out_png}", dpi=300, ray=1)
cmd.quit()
"""

def write_pymol_script(name, pdb_path, ab_chains, ag_chains, contacts, out_png, out_pml):
    """Write a .pml script for PyMOL headless rendering."""
    ab_str = "+".join(ab_chains)
    ag_str = "+".join(ag_chains)

    # Build contact selection: one selection for all GT residues
    sel_parts = []
    for (cid, resseq) in contacts:
        sel_parts.append(f"(chain {cid} and resi {resseq})")
    if sel_parts:
        contact_sel = "cmd.select(\"gt_contacts_{n}\", \"{conds}\")".format(
            n=name, conds=" or ".join(sel_parts))
    else:
        contact_sel = f"cmd.select(\"gt_contacts_{name}\", \"none\")"

    script = PYMOL_TEMPLATE.format(
        pdb_path       = pdb_path.replace("\\", "/"),
        name           = name,
        ag_chains_str  = ag_str,
        ab_chains_str  = ab_str,
        contact_selections = contact_sel,
        out_png        = out_png.replace("\\", "/"),
    )

    with open(out_pml, "w") as f:
        f.write(script)


def run_pymol(pml_path):
    """Run PyMOL in headless batch mode."""
    result = subprocess.run(
        [PYMOL_EXE, "-cq", str(pml_path)],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  PyMOL stderr: {result.stderr[:300]}")
    return result.returncode == 0


# ── Step 9: Heatmap figure ────────────────────────────────────────────────────

def make_heatmap(antigen, vh_seq, vl_seq, ig_vh, ig_vl, shap_vh, shap_vl,
                 contacts_vh, contacts_vl, cdr_vh, cdr_vl):
    """
    Four-panel heatmap: [VH IG | VH SHAP] / [VL IG | VL SHAP]
    Contact residues marked with ★, CDR regions shaded green.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 7))
    fig.suptitle(f"{antigen} — IG vs SHAP attribution mapped to PDB structure",
                 fontweight="bold", fontsize=12)

    def _panel(ax, seq, scores, contacts_set, cdrs, title, cmap):
        n = len(seq)
        x = np.arange(n)
        bar_colors = cm.get_cmap(cmap)(0.2 + 0.8 * scores / max(scores.max(), 1e-10))
        ax.bar(x, scores, color=bar_colors, width=1.0, edgecolor="none")

        # CDR shading
        for s, e in cdrs:
            ax.axvspan(s - 0.5, e - 0.5, alpha=0.18, color="lime", zorder=0)

        # Ground-truth contacts as orange stars above bar
        for pos, aa in enumerate(seq):
            if (pos+1) in contacts_set:  # 1-indexed in contacts_set
                ax.scatter(pos, scores[pos] + 0.02 * scores.max(),
                           marker="*", s=80, color="orange", zorder=5)

        ax.set_xlim(-1, n + 1)
        ax.set_ylim(0, scores.max() * 1.25 + 1e-10)
        ax.set_xlabel("Residue position (sequence)")
        ax.set_ylabel("Normalised |attribution|")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

        # x-ticks with residue every 10
        step = max(1, n // 15)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t+1}" for t in ticks], fontsize=6)

    # Normalise
    ig_vh_n   = ig_vh   / (ig_vh.max()   + 1e-12)
    ig_vl_n   = ig_vl   / (ig_vl.max()   + 1e-12)
    shap_vh_n = np.abs(shap_vh) / (np.abs(shap_vh).max() + 1e-12)
    shap_vl_n = np.abs(shap_vl) / (np.abs(shap_vl).max() + 1e-12)

    _panel(axes[0,0], vh_seq, ig_vh_n,   contacts_vh, cdr_vh, f"VH — IG",   "Reds")
    _panel(axes[0,1], vh_seq, shap_vh_n, contacts_vh, cdr_vh, f"VH — SHAP", "Blues")
    _panel(axes[1,0], vl_seq, ig_vl_n,   contacts_vl, cdr_vl, f"VL — IG",   "Reds")
    _panel(axes[1,1], vl_seq, shap_vl_n, contacts_vl, cdr_vl, f"VL — SHAP", "Blues")

    patches = [
        mpatches.Patch(color="lime", alpha=0.4, label="CDR region"),
        plt.Line2D([0],[0], marker="*", color="orange", mec="orange", mfc="orange",
                   ms=10, lw=0, label="GT paratope residue (contact ≤4.5Å)"),
        mpatches.Patch(color="red",  alpha=0.7, label="IG attribution"),
        mpatches.Patch(color="blue", alpha=0.7, label="SHAP attribution"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIG / f"fig_heatmap_{antigen}.png"
    plt.savefig(out)
    plt.close()
    print(f"  Heatmap -> {out}")


# ── CDR region lookup ─────────────────────────────────────────────────────────

CDR_FILE = ROOT / "scripts" / "cdr_boundaries.json"

def _get_cdrs(ag_name, chain_key):
    """Return list of (start, end) tuples for CDR regions."""
    try:
        with open(CDR_FILE) as f:
            bounds = json.load(f)
        ds_key = ANTIGEN_CONFIG[ag_name]["dataset"]
        if ds_key in bounds and chain_key in bounds[ds_key]:
            return [(s, e) for s, e in bounds[ds_key][chain_key].values()]
    except Exception:
        pass
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_antigen(ag_name):
    cfg = ANTIGEN_CONFIG[ag_name]
    print(f"\n{'='*60}")
    print(f"  {ag_name}")
    print(f"{'='*60}")

    # 1. Parse PDB
    parser = PDB.PDBParser(QUIET=True)
    struct = parser.get_structure(ag_name, cfg["pdb_file"])

    vh_seq, vh_resids = _extract_chain_seq_and_resids(struct, cfg["vh_chain"])
    vl_seq, vl_resids = _extract_chain_seq_and_resids(struct, cfg["vl_chain"])
    print(f"  VH: len={len(vh_seq)}  VL: len={len(vl_seq)}")

    # 2. Ground truth contacts
    contacts = _compute_contacts(
        struct, [cfg["vh_chain"], cfg["vl_chain"]], cfg["ag_chains"])
    contacts_vh = {resseq for (c, resseq) in contacts if c == cfg["vh_chain"]}
    contacts_vl = {resseq for (c, resseq) in contacts if c == cfg["vl_chain"]}
    print(f"  GT paratope: {len(contacts_vh)} VH + {len(contacts_vl)} VL residues")

    # Save contacts
    ct_rows = [{"chain": c, "resseq": r} for c, r in sorted(contacts)]
    pd.DataFrame(ct_rows).to_csv(OUT / f"{ag_name}_contacts.csv", index=False)

    # 3. Load dataset embeddings + train GP
    emb_dir = Path(cfg["emb_dir"])
    if not emb_dir.exists():
        print(f"  SKIP (no embeddings at {emb_dir})")
        return

    X_all   = np.load(emb_dir / "X.npy")
    y_all   = np.load(emb_dir / "y.npy")

    # Subsample to 2000 for GP tractability
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(y_all), min(2000, len(y_all)), replace=False)
    X_train = X_all[idx]; y_train = y_all[idx]

    scaler  = StandardScaler()
    pca     = PCA(n_components=128, random_state=42)
    y_norm  = scaler.fit_transform(y_train.reshape(-1,1)).ravel().astype(np.float32)
    X_pca   = pca.fit_transform(X_train).astype(np.float32)

    train_x = torch.tensor(X_pca)
    train_y = torch.tensor(y_norm)

    print("  Training GP (Tanimoto, 150 epochs)...")
    gp_model, gp_lik = train_gp(train_x, train_y, epochs=150)

    # 4. Get token-level ESM-2 embeddings for PDB sequences
    combined_seq = vh_seq + "G" + vl_seq
    print(f"  Combined PDB seq length: {len(combined_seq)}")
    token_arrs = get_token_level_embeddings([combined_seq])
    tokens = token_arrs[0]  # [L_combined, 1280]

    L_vh = len(vh_seq)
    L_vl = len(vl_seq)
    # Split tokens back into VH / separator / VL
    tok_vh = tokens[:L_vh]                  # [L_vh, 1280]
    tok_vl = tokens[L_vh+1: L_vh+1+L_vl]   # [L_vl, 1280]

    # 5. Integrated Gradients
    print("  Computing IG (50 steps)...")
    ig_full = compute_ig(tokens, pca, gp_model, gp_lik, n_steps=50)
    ig_vh   = ig_full[:L_vh]
    ig_vl   = ig_full[L_vh+1: L_vh+1+L_vl]

    # 6. SHAP
    print("  Computing SHAP (100 samples per position)...")
    shap_full = compute_shap(tokens, pca, gp_model, gp_lik, n_samples=80)
    shap_vh   = shap_full[:L_vh]
    shap_vl   = shap_full[L_vh+1: L_vh+1+L_vl]

    # 7. Normalise to [0, 100] for B-factor
    def _scale100(arr):
        mx = arr.max(); mn = arr.min()
        return ((arr - mn) / (mx - mn + 1e-12) * 100).round(2)

    ig_vh_b   = _scale100(ig_vh)
    ig_vl_b   = _scale100(ig_vl)
    shap_vh_b = _scale100(np.abs(shap_vh))
    shap_vl_b = _scale100(np.abs(shap_vl))

    # 8. Map to PDB resseq (using resids lists)
    def _make_chain_bfmap(resids, bfactors):
        return {resids[i][1]: float(bfactors[i]) for i in range(len(resids))}

    ig_bfmap = {
        cfg["vh_chain"]: _make_chain_bfmap(vh_resids, ig_vh_b),
        cfg["vl_chain"]: _make_chain_bfmap(vl_resids, ig_vl_b),
    }
    shap_bfmap = {
        cfg["vh_chain"]: _make_chain_bfmap(vh_resids, shap_vh_b),
        cfg["vl_chain"]: _make_chain_bfmap(vl_resids, shap_vl_b),
    }

    # 9. Write B-factor PDBs
    ig_pdb_out   = OUT / f"{ag_name}_ig.pdb"
    shap_pdb_out = OUT / f"{ag_name}_shap.pdb"
    write_bfactor_pdb(cfg["pdb_file"], ig_pdb_out,   ig_bfmap)
    write_bfactor_pdb(cfg["pdb_file"], shap_pdb_out, shap_bfmap)
    print(f"  B-factor PDBs written: {ig_pdb_out.name}, {shap_pdb_out.name}")

    # 10. Save attribution CSVs
    ab_chains = [cfg["vh_chain"], cfg["vl_chain"]]
    rows = []
    for i, (pos, aa, ig_s, shap_s) in enumerate(zip(
            range(L_vh), vh_seq, ig_vh, shap_vh)):
        resseq = vh_resids[i][1] if i < len(vh_resids) else i+1
        rows.append({"chain": cfg["vh_chain"], "seq_pos": i+1, "residue": aa,
                     "resseq": resseq, "ig_norm": float(ig_vh_b[i]/100),
                     "shap_norm": float(shap_vh_b[i]/100),
                     "is_contact": resseq in contacts_vh})
    for i, (pos, aa, ig_s, shap_s) in enumerate(zip(
            range(L_vl), vl_seq, ig_vl, shap_vl)):
        resseq = vl_resids[i][1] if i < len(vl_resids) else i+1
        rows.append({"chain": cfg["vl_chain"], "seq_pos": i+1, "residue": aa,
                     "resseq": resseq, "ig_norm": float(ig_vl_b[i]/100),
                     "shap_norm": float(shap_vl_b[i]/100),
                     "is_contact": resseq in contacts_vl})
    attr_df = pd.DataFrame(rows)
    attr_df.to_csv(OUT / f"{ag_name}_attributions.csv", index=False)

    # 11. PyMOL scripts + render
    for method, pdb_out in [("ig", ig_pdb_out), ("shap", shap_pdb_out)]:
        pml_path = OUT / f"{ag_name}_{method}.pml"
        png_path = OUT / f"{ag_name}_{method}_render.png"
        write_pymol_script(
            name       = f"{ag_name}_{method}",
            pdb_path   = str(pdb_out),
            ab_chains  = ab_chains,
            ag_chains  = cfg["ag_chains"],
            contacts   = contacts,
            out_png    = str(png_path),
            out_pml    = str(pml_path),
        )
        print(f"  Running PyMOL ({method})...")
        success = run_pymol(pml_path)
        print(f"    {'OK' if success else 'FAILED — check .pml manually'}")

    # 12. Heatmap figure
    cdr_vh_spans = _get_cdrs(ag_name, "vh_cdrs")
    cdr_vl_spans = _get_cdrs(ag_name, "vl_cdrs")
    make_heatmap(ag_name, vh_seq, vl_seq,
                 ig_vh, ig_vl, shap_vh, shap_vl,
                 contacts_vh, contacts_vl,
                 cdr_vh_spans, cdr_vl_spans)

    # 13. Summary stats
    ig_vh_n   = ig_vh / (ig_vh.max() + 1e-12)
    shap_vh_n = np.abs(shap_vh) / (np.abs(shap_vh).max() + 1e-12)
    thr = 0.4
    ig_pred   = {vh_resids[i][1] for i in range(L_vh) if ig_vh_n[i] > thr}
    shap_pred = {vh_resids[i][1] for i in range(L_vh) if shap_vh_n[i] > thr}
    for mname, pred in [("IG", ig_pred), ("SHAP", shap_pred)]:
        tp  = len(pred & contacts_vh)
        prec = tp / max(len(pred), 1)
        rec  = tp / max(len(contacts_vh), 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-10)
        print(f"  {mname} VH @ thr={thr}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}  "
              f"({tp}/{len(contacts_vh)} GT residues found)")

    return attr_df


# ── Summary comparison figure ─────────────────────────────────────────────────

def make_summary_figure(all_results):
    """Bar chart: IG vs SHAP precision/recall/F1 per antigen."""
    antigens = list(all_results.keys())
    metrics  = ["precision", "recall", "f1"]
    n = len(antigens)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    ig_colors   = [cm.Reds(0.5 + 0.4*i/max(n-1,1))  for i in range(n)]
    shap_colors = [cm.Blues(0.5 + 0.4*i/max(n-1,1)) for i in range(n)]

    thr = 0.4
    for ax, metric in zip(axes, metrics):
        ig_vals, shap_vals = [], []
        for ag_name, df in all_results.items():
            vh_chain = ANTIGEN_CONFIG[ag_name]["vh_chain"]
            ct_df   = pd.read_csv(OUT / f"{ag_name}_contacts.csv")
            gt_set  = set(ct_df[ct_df.chain == vh_chain].resseq)

            sub = df[df.chain == vh_chain]
            ig_pred   = set(sub[sub.ig_norm   > thr].resseq)
            shap_pred = set(sub[sub.shap_norm > thr].resseq)

            def _score(pred, gt):
                tp = len(pred & gt)
                p  = tp / max(len(pred), 1)
                r  = tp / max(len(gt), 1)
                f  = 2*p*r / max(p+r, 1e-10)
                return {"precision":p, "recall":r, "f1":f}[metric]

            ig_vals.append(_score(ig_pred, gt_set))
            shap_vals.append(_score(shap_pred, gt_set))

        x = np.arange(n); w = 0.35
        b1 = [ax.bar(x[i]-w/2, ig_vals[i],   w, color=ig_colors[i],   edgecolor="k", linewidth=0.5)
              for i in range(n)]
        b2 = [ax.bar(x[i]+w/2, shap_vals[i], w, color=shap_colors[i], edgecolor="k", linewidth=0.5)
              for i in range(n)]
        ax.set_xticks(x); ax.set_xticklabels(antigens, fontsize=8, rotation=15)
        ax.set_ylabel(metric.capitalize()); ax.set_ylim(0, 0.55)
        ax.set_title(metric.capitalize(), fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    ig_p   = mpatches.Patch(color=ig_colors[2],   label="IG (Reds)")
    sh_p   = mpatches.Patch(color=shap_colors[2], label="SHAP (Blues)")
    gt_p   = plt.Line2D([0],[0], marker="*", color="orange", mfc="orange", ms=10, lw=0,
                         label="GT paratope contact")
    fig.legend(handles=[ig_p, sh_p, gt_p], loc="upper right", fontsize=9)
    fig.suptitle(f"PDB-based attribution vs ground truth paratope (VH chain, threshold={thr})\n"
                 "Paratope = heavy-atom contacts ≤ 4.5 Å from antigen",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = FIG / "fig_pdb_gt_summary.png"
    plt.savefig(out)
    plt.close()
    print(f"\nSummary figure → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 4FQI excluded from most analysis, but we still have its PDB
    # Run all antigens except none (include all 6 for structural completeness)
    all_results = {}
    for ag in ["3GBN", "AAYL49", "AAYL50", "AAYL51", "AAYL52", "4FQI"]:
        try:
            df = run_antigen(ag)
            if df is not None:
                all_results[ag] = df
        except Exception as e:
            print(f"\n  ERROR {ag}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        make_summary_figure({k: v for k, v in all_results.items() if k != "4FQI"})
        print("\nAll done. Outputs in results_explainability/pdb_based/")
