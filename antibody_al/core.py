"""Shared GP + AL core — GPU-accelerated (NVIDIA RTX A4000)."""
from __future__ import annotations
from pathlib import Path
from typing import List
import gpytorch, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.distributions import Normal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[core] Using device: {DEVICE}" +
      (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type=="cuda" else ""))

# ── v1(eq) 10% budget configs ─────────────────────────────────────────────────
CONFIGS = {
    "3gbn_h1":   {"seed": 19,  "batch": 19,  "cycles": 9},
    "3gbn_h9":   {"seed": 18,  "batch": 18,  "cycles": 9},
    "aayl49":    {"seed": 43,  "batch": 43,  "cycles": 9},
    "aayl51":    {"seed": 43,  "batch": 43,  "cycles": 9},
    "aayl49_ml": {"seed": 90,  "batch": 90,  "cycles": 9},
    "aayl50":    {"seed": 115, "batch": 115, "cycles": 9},
    "aayl52":    {"seed": 133, "batch": 133, "cycles": 9},
    "4fqi_h3":   {"seed": 655, "batch": 655, "cycles": 9},
}
DATASETS  = list(CONFIGS.keys())
N_TOTAL   = {"3gbn_h1":1887,"3gbn_h9":1842,"aayl49":4312,"aayl51":4320,
             "aayl49_ml":8953,"aayl50":11473,"aayl52":13324,"4fqi_h3":65535}
SEEDS     = [42, 123, 456, 789, 1011]
KERNELS   = ["tanimoto","matern","rbf","rq"]
PROTOCOLS = ["ucb","ei","pi","random"]

PCA_DIM    = 128       # fixed across ALL experiments — do not reduce
GP_EPOCHS  = 100      # full training — do not reduce, results quality depends on convergence
GP_LR      = 0.1
GP_DECAY   = 0.95
PRED_BATCH = 10_000
UCB_BETA   = 1.0
EI_XI      = 0.01
PI_XI      = 0.01

_norm = Normal(torch.tensor(0.0), torch.tensor(1.0))


# ── Kernels ───────────────────────────────────────────────────────────────────
class TanimotoKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def forward(self, x1, x2, **kw):
        d  = x1 @ x2.transpose(-2, -1)
        n1 = x1.pow(2).sum(-1, keepdim=True)
        n2 = x2.pow(2).sum(-1, keepdim=True)
        return d / (n1 + n2.transpose(-2, -1) - d + 1e-8)

def build_kernel(name: str):
    if name == "tanimoto": return gpytorch.kernels.ScaleKernel(TanimotoKernel())
    if name == "matern":   return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
    if name == "rbf":      return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    if name == "rq":       return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
    raise ValueError(name)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik, kern):
        super().__init__(tx, ty, lik)
        self.mean  = gpytorch.means.ConstantMean()
        self.covar = kern
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.covar(x))

def train_gp(tx, ty, lik, model):
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=GP_LR)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, GP_DECAY)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(GP_EPOCHS):
        opt.zero_grad()
        (-mll(model(tx), ty)).backward()
        opt.step(); sch.step()
    model.eval(); lik.eval()


# ── Acquisition ───────────────────────────────────────────────────────────────
def _predict(X_pca, model, lik, indices):
    """Run GP prediction on indices, returns numpy mu and sd (on CPU)."""
    mu_l, sd_l = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(indices), PRED_BATCH):
            bx   = torch.tensor(X_pca[indices[i:i+PRED_BATCH]]).float().to(DEVICE)
            dist = lik(model(bx))
            mu_l.append(dist.mean.cpu().numpy())
            sd_l.append(dist.stddev.cpu().numpy())
    return np.concatenate(mu_l), np.concatenate(sd_l)

def get_beta(variant, cycle, n_cycles):
    """Beta for a given UCB schedule variant at a given cycle (0-indexed)."""
    if variant == "b0.1": return 0.1
    if variant == "b0.5": return 0.5
    if variant == "b1.0": return 1.0
    if variant == "b2.0": return 2.0
    if variant == "grad":
        t = cycle / max(n_cycles - 1, 1)
        return 2.0 - (2.0 - 0.1) * t
    if variant == "alt":
        return 2.0 if cycle % 2 == 0 else 0.1
    raise ValueError(f"Unknown beta variant '{variant}'")


def acquire(protocol, X_pca, model, lik, selected, batch, rng, best_yn=0.0, beta=UCB_BETA):
    pool = np.array(sorted(set(range(len(X_pca))) - set(selected)))
    if len(pool) == 0: return []
    if protocol == "random":
        return rng.choice(pool, min(batch, len(pool)), replace=False).tolist()
    mu, sd = _predict(X_pca, model, lik, pool)
    if protocol == "ucb":
        scores = mu + beta * sd
    elif protocol == "ei":
        mu_t = torch.tensor(mu); sd_t = torch.tensor(sd)
        z      = (mu_t - best_yn - EI_XI) / (sd_t + 1e-9)
        scores = ((mu_t - best_yn - EI_XI)*_norm.cdf(z)
                  + sd_t * torch.exp(_norm.log_prob(z))).numpy()
    elif protocol == "pi":
        z      = (torch.tensor(mu) - best_yn - PI_XI) / (torch.tensor(sd) + 1e-9)
        scores = _norm.cdf(z).numpy()
    else:
        raise ValueError(protocol)
    top = np.argsort(scores)[::-1][:batch]
    return pool[top].tolist()


# ── Label-blind stratified seed ────────────────────────────────────────────────
def stratified_seed_blind(X_pca, n, rng):
    """Sample n variants spanning the embedding-feature space (KMeans clusters on
    X_pca), one point per cluster. Label-blind — unlike the previous version,
    which binned the pool on true affinity quantiles (see git history)."""
    n = min(n, len(X_pca))
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.integers(1 << 31)))
    labels = km.fit_predict(X_pca)
    picks = []
    for c in range(n):
        members = np.where(labels == c)[0]
        if len(members):
            picks.append(int(rng.choice(members)))
    pool = [i for i in range(len(X_pca)) if i not in set(picks)]
    while len(picks) < n and pool:
        idx = int(rng.choice(pool)); picks.append(idx); pool.remove(idx)
    return picks[:n]


# ── Single experiment ─────────────────────────────────────────────────────────
TEST_FRAC = 0.15  # fixed test split, reserved before AL starts, never eligible for seeding/acquisition

def run_experiment(emb_dir, out_csv, dataset, protocol, kernel,
                   random_seed=42, verbose=True, beta_variant="b1.0"):
    cfg     = CONFIGS[dataset]
    out_csv = Path(out_csv)
    if out_csv.exists():
        if verbose: print(f"    skip (exists)")
        return pd.read_csv(out_csv)

    X   = np.load(Path(emb_dir) / "X.npy")
    y   = np.load(Path(emb_dir) / "y.npy")
    t2p = np.load(Path(emb_dir) / "top2p.npy").astype(bool)
    t5p = np.load(Path(emb_dir) / "top5p.npy").astype(bool)

    pca    = PCA(n_components=min(PCA_DIM, X.shape[0]-1, X.shape[1]), random_state=42)
    X_pca  = pca.fit_transform(X).astype(np.float32)
    y_norm = StandardScaler().fit_transform(y.reshape(-1,1)).ravel().astype(np.float32)

    rng = np.random.default_rng(random_seed)

    # Fixed test split: drawn once, before seeding, never eligible for seeding or
    # acquisition. Evaluating on "whatever remains after AL" instead is also biased
    # (range restriction: AL's own acquisition depletes the high-affinity tail from
    # the remaining pool, mechanically deflating its rank correlation).
    n_test    = max(1, int(round(TEST_FRAC * len(X_pca))))
    test_idx  = rng.choice(len(X_pca), size=n_test, replace=False)
    test_mask = np.zeros(len(X_pca), dtype=bool); test_mask[test_idx] = True
    eligible  = np.where(~test_mask)[0]

    selected = [int(eligible[i]) for i in stratified_seed_blind(X_pca[eligible], cfg["seed"], rng)]
    rows: List[dict] = []

    for cycle in range(cfg["cycles"] + 1):
        # Move training data to GPU
        tx  = torch.tensor(X_pca[selected]).float().to(DEVICE)
        ty  = torch.tensor(y_norm[selected]).float().to(DEVICE)
        lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        with gpytorch.settings.cholesky_jitter(1e-3):
            kern  = build_kernel(kernel).to(DEVICE)
            model = GPModel(tx, ty, lik, kern).to(DEVICE)
            train_gp(tx, ty, lik, model)

        mu_all, _   = _predict(X_pca, model, lik, np.arange(len(X_pca)))
        rho_full, _ = spearmanr(y, mu_all)
        rho_test, _ = spearmanr(y[test_idx], mu_all[test_idx])

        t2f = int(t2p[selected].sum()); t5f = int(t5p[selected].sum())
        rows.append({
            "cycle":        cycle,
            "n_labelled":   len(selected),
            "pct_labelled": round(len(selected)/N_TOTAL[dataset]*100, 2),
            "spearman_full_pool":  round(float(rho_full) if not np.isnan(rho_full) else 0.0, 4),
            "spearman_fixed_test": round(float(rho_test) if not np.isnan(rho_test) else 0.0, 4),
            "recall_2p":    round(t2f / max(1, int(t2p.sum())), 4),
            "recall_5p":    round(t5f / max(1, int(t5p.sum())), 4),
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        if cycle == cfg["cycles"]: break
        best_yn = float(ty.max().cpu())
        beta = get_beta(beta_variant, cycle, cfg["cycles"] + 1) if protocol == "ucb" else UCB_BETA
        new = acquire(protocol, X_pca, model, lik, selected + list(test_idx), cfg["batch"], rng, best_yn, beta=beta)
        if not new: break
        selected = selected + new

        # Free GPU memory between cycles
        del model, kern, lik, tx, ty
        torch.cuda.empty_cache()

    last = rows[-1]
    if verbose:
        print(f"    recall@2%={last['recall_2p']:.4f}  recall@5%={last['recall_5p']:.4f}"
              f"  rho_test={last['spearman_fixed_test']:+.4f}  n={last['n_labelled']} ({last['pct_labelled']}%)")
    return pd.DataFrame(rows)
