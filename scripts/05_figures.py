"""
Paper figures â€” new 10%-budget results.
Style: identical to generate_combined_panels.py
Error bars: std across 4 GP kernels (panels A,C) or std across 8 targets (panel B)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as _cm
import matplotlib.patches as mpatches

ROOT      = Path(__file__).resolve().parent.parent
COMP      = ROOT / "results_final" / "compiled"
MAIN_DIR  = ROOT / "results_final" / "main"
EXPL      = ROOT / "results_explainability" / "pdb_based"
OUT_PAPER = ROOT / "results_final" / "paper_figures"
OUT_NEW   = ROOT / "results_final" / "figures"
for d in (OUT_PAPER, OUT_NEW): d.mkdir(parents=True, exist_ok=True)

# â”€â”€ rcParams â€” identical to generate_combined_panels.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})

# â”€â”€ Constants â€” identical to generate_combined_panels.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASETS      = ["3gbn_h1","3gbn_h9","aayl49","aayl51","aayl49_ml","aayl50","aayl52","4fqi_h3"]
DS_LABELS     = {"3gbn_h1":"3GBN H1","3gbn_h9":"3GBN H9","aayl49":"AAYL49",
                 "aayl51":"AAYL51","aayl49_ml":"AAYL49 ML","aayl50":"AAYL50",
                 "aayl52":"AAYL52","4fqi_h3":"4FQI H3"}
PROTOCOLS     = ["random","ucb","ei","pi"]
PROTO_LABELS  = {"random":"Random","ucb":"UCB","ei":"EI","pi":"PI"}
KERNELS       = ["tanimoto","rq","matern","rbf"]
KERNEL_LABELS = {"tanimoto":"Tanimoto","rq":"RQ","matern":"Matern","rbf":"RBF"}
SEEDS         = [42, 123, 456]

TAB10 = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))

def seq_colors(n, cmap="Blues"):
    return [_cm.get_cmap(cmap)(0.30 + 0.60*i/max(n-1,1)) for i in range(n)]

PROTO_COLORS  = dict(zip(PROTOCOLS, seq_colors(4, "Greens")))
KERNEL_COLORS = dict(zip(KERNELS,   seq_colors(4, "Purples")))

# PLM + AA representation palette â€” single colour sequence for combined panel
ALL_REP_KEYS   = ["esm2","antiberty_paired","protbert","ablang2",
                  "antiberty_concat","progen2",
                  "bag_of_aa5","blosum_eigen","one_hot"]
ALL_REP_NAMES  = ["ESM-2\n(ours)","AntiBERTy\n(paired)","ProtBert",
                  "AbLang2","AntiBERTy\n(concat)","ProGen2",
                  "Bag-of-AA\n(n=5)","BLOSUM62\nEigen","One-Hot"]
PLM_SLICE      = slice(0,6)
AA_SLICE       = slice(6,9)
PLM_COLORS     = seq_colors(6, "GnBu")
AA_COLORS      = seq_colors(3, "PuBuGn")
ALL_REP_COLORS = PLM_COLORS + AA_COLORS


def _save(fig, name):
    for d in (OUT_PAPER, OUT_NEW):
        fig.savefig(d/name,                         dpi=300, bbox_inches="tight")
        fig.savefig(d/name.replace(".png",".pdf"),   bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


# â”€â”€ Data helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_main():  return pd.read_csv(COMP/"main_summary.csv")
def load_plm():   return pd.read_csv(COMP/"plm_summary.csv")
def load_aa():    return pd.read_csv(COMP/"aa_rep_summary.csv")

def _pad(arr, n):
    if len(arr) >= n: return arr[:n]
    return np.concatenate([arr, np.full(n-len(arr), arr[-1])])

def load_ucb_curves_by_kernel():
    """
    Returns dict: {dataset: {kernel: array of shape (n_cycles,)}}
    Values are mean Recall@2% across 3 seeds per cycle.
    """
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for kern in KERNELS:
            seed_curves = []
            for s in SEEDS:
                f = MAIN_DIR/ds/f"ucb_{kern}_seed{s}.csv"
                if f.exists():
                    seed_curves.append(pd.read_csv(f)["recall_2p"].values)
            if seed_curves:
                max_len = max(len(c) for c in seed_curves)
                mat = np.array([_pad(c, max_len) for c in seed_curves])
                data[ds][kern] = mat.mean(axis=0)   # mean over seeds
    return data

def mean_std_across_kernels(ucb_curves, ds):
    """Given curves per kernel for one dataset, return (mean, std, x) across kernels."""
    arrs = list(ucb_curves[ds].values())
    if not arrs: return None, None, None
    max_len = max(len(a) for a in arrs)
    mat = np.array([_pad(a, max_len) for a in arrs])
    return mat.mean(0), mat.std(0), np.arange(max_len)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PANEL A â€” overview  (violin Â· bars Â· UCB curves Â· scatter)
# Caption: (a) Distribution over 16 kernelâ€“acq combos.  (b) Per-target meanÂ±std
#          across 4 kernels.  (c) UCB curves meanÂ±std across 4 kernels.
#          (d) Recall@2% vs Spearman Ï.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def panel_A():
    print("Panel A: overview...")
    df        = load_main()
    ucb_curves = load_ucb_curves_by_kernel()

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.34,
                            top=0.93, bottom=0.08, left=0.08, right=0.97)

    # (a) Violin â€” distribution across 16 combos per target
    ax_a = fig.add_subplot(gs[0,0])
    df_active = df[df.protocol != "random"]
    data_per_ds = [df_active[df_active.dataset==ds]["recall_2p_mean"].values
                   for ds in DATASETS]
    parts = ax_a.violinplot(data_per_ds, positions=range(len(DATASETS)),
                            showmeans=True, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], TAB10):
        pc.set_facecolor(col); pc.set_alpha(0.72)
    parts["cmeans"].set_color("black"); parts["cmedians"].set_color("red")
    ax_a.set_xticks(range(len(DATASETS)))
    ax_a.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11,
                          rotation=45, ha="right")
    ax_a.set_ylabel("Final Recall@2%")
    for i, ds in enumerate(DATASETS):
        v = df_active[df_active.dataset==ds]["recall_2p_mean"]
        if len(v):
            ax_a.text(i, min(v.mean()+v.std()+0.04, 1.12), f"{v.mean():.2f}",
                      ha="center", fontsize=10, fontweight="bold")
    ax_a.set_ylim(-0.05, 1.18)
    ax_a.text(-0.04, 1.06, "(a)", transform=ax_a.transAxes,
              fontsize=19, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # (b) Per-target grouped bars â€” meanÂ±std across 4 kernels per protocol
    ax_b = fig.add_subplot(gs[0,1])
    x = np.arange(len(DATASETS)); w = 0.18
    for ki, proto in enumerate(PROTOCOLS):
        means, errs = [], []
        for ds in DATASETS:
            sub = df[(df.dataset==ds)&(df.protocol==proto)]["recall_2p_mean"]
            means.append(sub.mean() if len(sub) else np.nan)
            errs.append(sub.std()   if len(sub)>1 else 0)
        ax_b.bar(x + (ki-1.5)*w, means, w, label=PROTO_LABELS[proto],
                 color=PROTO_COLORS[proto], edgecolor="k", linewidth=0.5,
                 yerr=errs, capsize=3,
                 error_kw={"elinewidth":1.1,"ecolor":"#333","capthick":1.1})
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11,
                          rotation=35, ha="right")
    ax_b.set_ylabel("Recall@2% (mean Â± std, 4 kernels)")
    ax_b.legend(ncol=4, fontsize=11, framealpha=0.85,
                loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax_b.set_ylim(0, 1.28)
    ax_b.text(-0.04, 1.06, "(b)", transform=ax_b.transAxes,
              fontsize=19, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # (c) UCB learning curves â€” meanÂ±std across 4 kernels (averaged over 3 seeds)
    ax_c = fig.add_subplot(gs[1,0])
    for i, ds in enumerate(DATASETS):
        m, s, x_vals = mean_std_across_kernels(ucb_curves, ds)
        if m is None: continue
        ax_c.plot(x_vals, m, color=TAB10[i], linewidth=1.6, label=DS_LABELS[ds])
        ax_c.fill_between(x_vals, np.clip(m-s,0,1), np.clip(m+s,0,1),
                           alpha=0.15, color=TAB10[i])
    ax_c.set_xlabel("AL Cycle")
    ax_c.set_ylabel("Recall@2% (UCB, mean Â± std, 4 kernels)")
    ax_c.legend(fontsize=10, ncol=4, framealpha=0.85,
                loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.text(-0.04, 1.06, "(c)", transform=ax_c.transAxes,
              fontsize=19, fontweight="bold", va="bottom", ha="right", clip_on=False)

    # (d) Recall@2% vs Spearman scatter â€” all non-random combos
    ax_d = fig.add_subplot(gs[1,1])
    df_sc = df[df.protocol != "random"]
    for i, ds in enumerate(DATASETS):
        sub = df_sc[df_sc.dataset==ds]
        ax_d.errorbar(sub["spearman_mean"], sub["recall_2p_mean"],
                      xerr=sub["spearman_std"], yerr=sub["recall_2p_std"],
                      fmt="o", ms=4, alpha=0.60, color=TAB10[i],
                      label=DS_LABELS[ds], elinewidth=0.6, capsize=0)
    ax_d.set_xlabel("Spearman Ï (full pool ranking)")
    ax_d.set_ylabel("Final Recall@2%")
    plt.setp(ax_d.get_xticklabels(), rotation=30, ha="right")
    ax_d.legend(fontsize=10, ncol=4, framealpha=0.85,
                loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax_d.text(-0.04, 1.06, "(d)", transform=ax_d.transAxes,
              fontsize=19, fontweight="bold", va="bottom", ha="right", clip_on=False)

    _save(fig, "panel_A_overview.png")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PANEL C â€” per-target ablation
# (a) ProtocolÃ—kernel grouped bars  (b) AL curves UCB/EI/Random Â± std across kernels
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def panel_C():
    print("Panel C: per-target ablation...")
    df        = load_main()
    ucb_curves = load_ucb_curves_by_kernel()

    # Also load EI and Random curves across kernels
    def load_proto_curves(proto):
        data = {}
        for ds in DATASETS:
            data[ds] = {}
            for kern in KERNELS:
                seed_curves = []
                for s in SEEDS:
                    f = MAIN_DIR/ds/f"{proto}_{kern}_seed{s}.csv"
                    if f.exists():
                        seed_curves.append(pd.read_csv(f)["recall_2p"].values)
                if seed_curves:
                    max_len = max(len(c) for c in seed_curves)
                    mat = np.array([_pad(c, max_len) for c in seed_curves])
                    data[ds][kern] = mat.mean(axis=0)
        return data

    ei_curves     = load_proto_curves("ei")
    random_curves = load_proto_curves("random")

    plt.rcParams.update({
        "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 17,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    })

    fig = plt.figure(figsize=(18, 13))
    gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.50,
                                 top=0.88, bottom=0.05, left=0.06, right=0.99)

    # (a) Protocol Ã— kernel grouped bars â€” mean over 3 seeds, error bar = std over 3 seeds
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[0],
                                             hspace=0.42, wspace=0.30)
    kern_handles = [mpatches.Patch(color=KERNEL_COLORS[k], label=KERNEL_LABELS[k])
                    for k in KERNELS]
    for idx, ds in enumerate(DATASETS):
        ax = fig.add_subplot(gs_a[idx//4, idx%4])
        x = np.arange(len(PROTOCOLS)); w = 0.18
        for ki, kern in enumerate(KERNELS):
            means, errs = [], []
            for proto in PROTOCOLS:
                sub = df[(df.dataset==ds)&(df.protocol==proto)&(df.kernel==kern)]
                means.append(sub["recall_2p_mean"].mean() if len(sub) else 0)
                errs.append(sub["recall_2p_std"].mean()   if len(sub) else 0)
            ax.bar(x + (ki-(len(KERNELS)-1)/2)*w, means, w,
                   color=KERNEL_COLORS[kern], edgecolor="k", linewidth=0.4,
                   yerr=errs, capsize=2,
                   error_kw={"elinewidth":0.8,"ecolor":"#333","capthick":0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([PROTO_LABELS[p] for p in PROTOCOLS], fontsize=16)
        ax.set_title(DS_LABELS[ds], fontweight="bold", fontsize=18, pad=3)
        ax.set_ylim(0, 1.18)
        if idx % 4 == 0: ax.set_ylabel("Recall@2%", fontsize=16)
        if idx == 0:
            ax.text(-0.20, 1.12, "(a)", transform=ax.transAxes,
                    fontsize=21, fontweight="bold", va="bottom")
    fig.legend(handles=kern_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.96), fontsize=16, framealpha=0.85, edgecolor="#aaa")

    # (b) AL curves â€” UCB, EI, Random; meanÂ±std across 4 kernels
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[1],
                                             hspace=0.68, wspace=0.30)
    protos_lc  = ["ucb","ei","random"]
    curves_map = {"ucb": ucb_curves, "ei": ei_curves, "random": random_curves}
    for idx, ds in enumerate(DATASETS):
        ax = fig.add_subplot(gs_b[idx//4, idx%4])
        for proto in protos_lc:
            m, s, x_vals = mean_std_across_kernels(curves_map[proto], ds)
            if m is None: continue
            ax.plot(x_vals, m, color=PROTO_COLORS[proto], linewidth=1.5,
                    label=PROTO_LABELS[proto])
            ax.fill_between(x_vals, np.clip(m-s,0,1), np.clip(m+s,0,1),
                            alpha=0.15, color=PROTO_COLORS[proto])
        ax.set_title(DS_LABELS[ds], fontweight="bold", fontsize=18, pad=3)
        ax.set_xlabel("AL Cycle", fontsize=16)
        ax.set_ylim(0, 1.05)
        if idx % 4 == 0: ax.set_ylabel("Recall@2%", fontsize=16)
        if idx == 0:
            ax.text(-0.20, 1.12, "(b)", transform=ax.transAxes,
                    fontsize=21, fontweight="bold", va="bottom")

    proto_handles = [plt.Line2D([0],[0], color=PROTO_COLORS[p], linewidth=2.5,
                                label=PROTO_LABELS[p]) for p in protos_lc]
    fig.legend(handles=proto_handles, loc="center", ncol=3,
               bbox_to_anchor=(0.5, 0.455), fontsize=16, framealpha=0.85, edgecolor="#aaa")

    _save(fig, "panel_C_perdataset.png")
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PANEL B â€” combined representation ablation (PLM + AA on same axes)
# Error bars = std across 8 targets; dots = individual targets
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def panel_B():
    print("Panel B: combined representation ablation...")
    df  = load_main()
    plm = load_plm()
    aa  = load_aa()
    rng = np.random.default_rng(42)

    # Build per-representation arrays (one value per dataset = mean over 3 seeds)
    esm_base = df[(df.protocol=="ucb")&(df.kernel=="tanimoto")]

    rep_map = {
        "esm2":            ("plm", esm_base),
        "antiberty_paired":("plm", plm[plm.plm=="antiberty_paired"]),
        "protbert":        ("plm", plm[plm.plm=="protbert"]),
        "ablang2":         ("plm", plm[plm.plm=="ablang2"]),
        "antiberty_concat":("plm", plm[plm.plm=="antiberty_concat"]),
        "progen2":         ("plm", plm[plm.plm=="progen2"]),
        "bag_of_aa5":      ("aa",  aa[aa.rep=="bag_of_aa5"]),
        "blosum_eigen":    ("aa",  aa[aa.rep=="blosum_eigen"]),
        "one_hot":         ("aa",  aa[aa.rep=="one_hot"]),
    }

    means, stds, dots = [], [], []
    for key in ALL_REP_KEYS:
        _, sub = rep_map[key]
        vals = []
        for ds in DATASETS:
            row = sub[sub.dataset==ds]["recall_2p_mean"] if "dataset" in sub.columns \
                  else pd.Series(dtype=float)
            if len(row): vals.append(float(row.iloc[0]))
        arr = np.array(vals)
        means.append(arr.mean() if len(arr) else np.nan)
        stds.append(arr.std()   if len(arr) > 1 else 0)
        dots.append(arr)

    fig, ax = plt.subplots(figsize=(16, 5.5))

    x = np.arange(len(ALL_REP_NAMES))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=ALL_REP_COLORS, edgecolor="k", linewidth=0.8,
                  error_kw={"elinewidth":1.6,"ecolor":"#333","capthick":1.6})

    # Dots = individual targets
    for j, vals in enumerate(dots):
        if len(vals):
            jitter = rng.uniform(-0.22, 0.22, len(vals))
            ax.scatter(j + jitter, vals, color="k", s=24, zorder=5,
                       alpha=0.55, linewidths=0)

    # Value labels
    for i, (m, s_) in enumerate(zip(means, stds)):
        if not np.isnan(m):
            ax.text(i, m + s_ + 0.030, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Divider between PLM and classical
    n_plm = 6
    ax.axvline(n_plm - 0.5, color="#888", linewidth=1.2, linestyle="--", zorder=0)
    ax.text(n_plm/2 - 0.5, 1.07, "Protein Language Models",
            ha="center", fontsize=13, color="#333", style="italic")
    ax.text(n_plm + 1.0, 1.07, "Classical sequence encodings",
            ha="center", fontsize=13, color="#333", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_REP_NAMES, fontsize=12)
    ax.set_ylabel("Recall@2%\n(UCB + Tanimoto, mean Â± std across 8 targets)", fontsize=13)
    ax.set_ylim(0, 1.18)
    ax.text(0.99, 0.97,
            "Error bars = cross-target std\nDots = individual targets (8 datasets, 3 seeds)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))

    plt.tight_layout()
    _save(fig, "panel_B_representation.png")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PANEL D â€” compact: kernelÃ—protocol heatmap + paratope recall
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def panel_D():
    print("Panel D: heatmap + paratope recall...")
    df = load_main()

    protos  = ["ucb","ei","pi","random"]
    kernels = ["tanimoto","rq","matern","rbf"]

    # Mean over datasets, std over datasets (cross-dataset spread)
    pivot_m = df.groupby(["protocol","kernel"])["recall_2p_mean"].mean().unstack()
    pivot_s = df.groupby(["protocol","kernel"])["recall_2p_mean"].std().unstack()
    pivot_m = pivot_m.reindex(index=protos, columns=kernels)
    pivot_s = pivot_s.reindex(index=protos, columns=kernels)

    # Paratope recall
    SYSTEMS = {"3GBN": EXPL/"3GBN_attributions.csv",
               "AAYL49":EXPL/"AAYL49_attributions.csv",
               "AAYL50":EXPL/"AAYL50_attributions.csv",
               "AAYL51":EXPL/"AAYL51_attributions.csv",
               "AAYL52":EXPL/"AAYL52_attributions.csv",
               "4FQI":  EXPL/"4FQI_attributions.csv"}
    CONTACT = {k: EXPL/f"{k}_contacts.csv" for k in SYSTEMS}
    THRS    = [0.30, 0.40, 0.50]
    ags, ig_recall, sh_recall, ig_err, sh_err = [], [], [], [], []
    for ag, attr_f in SYSTEMS.items():
        try:
            at = pd.read_csv(attr_f); ct = pd.read_csv(CONTACT[ag])
            gt = set(zip(ct.chain, ct.resseq))
            def rec(col, thr, _at=at, _gt=gt):
                pred = set(zip(_at[_at[col]>thr].chain, _at[_at[col]>thr].resseq))
                return len(pred&_gt)/max(len(_gt),1)
            ags.append(ag)
            ig_recall.append(rec("ig_norm",   0.40))
            sh_recall.append(rec("shap_norm", 0.40))
            ig_err.append(np.std([rec("ig_norm",   t) for t in THRS], ddof=1))
            sh_err.append(np.std([rec("shap_norm", t) for t in THRS], ddof=1))
        except Exception as e:
            print(f"  SKIP {ag}: {e}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"width_ratios":[1,1.1]})

    # (a) Heatmap â€” mean Â± std across datasets
    im = ax_a.imshow(pivot_m.values, cmap="RdPu", aspect="auto", vmin=0.05, vmax=0.80)
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels([KERNEL_LABELS[k] for k in kernels], fontsize=14)
    ax_a.set_yticks(range(4))
    ax_a.set_yticklabels(["UCB","EI","PI","Random"], fontsize=14)
    for i in range(4):
        for j in range(4):
            m = pivot_m.values[i,j]; s_ = pivot_s.values[i,j]
            if not np.isnan(m):
                c = "white" if m > 0.52 else "black"
                label = f"{m:.2f}\nÂ±{s_:.2f}" if not np.isnan(s_) else f"{m:.2f}"
                ax_a.text(j, i, label, ha="center", va="center",
                          color=c, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_a, label="Mean Recall@2%", fraction=0.04, pad=0.04)
    ax_a.text(-0.14, 1.04, "(a)", transform=ax_a.transAxes,
              fontsize=19, fontweight="bold", va="bottom")

    # (b) Paratope recall bars with threshold-sensitivity error bars
    if ags:
        n = len(ags); x = np.arange(n); w = 0.32
        ig_col   = _cm.YlGnBu(0.65); shap_col = _cm.PuBuGn(0.65)
        ax_b.bar(x-w/2, ig_recall, w, color=ig_col,   edgecolor="k", lw=0.7,
                 label="Integrated Gradients")
        ax_b.bar(x+w/2, sh_recall, w, color=shap_col, edgecolor="k", lw=0.7,
                 label="KernelSHAP")
        ax_b.errorbar(x-w/2, ig_recall, yerr=ig_err, fmt="none",
                      ecolor="k", elinewidth=1.2, capsize=4, capthick=1.2)
        ax_b.errorbar(x+w/2, sh_recall, yerr=sh_err, fmt="none",
                      ecolor="k", elinewidth=1.2, capsize=4, capthick=1.2)
        ax_b.set_xticks(x); ax_b.set_xticklabels(ags, fontsize=14)
        ax_b.set_ylim(0, 1.08)
        ax_b.axhline(0.5, color="#aaa", linewidth=0.8, linestyle="--", zorder=0)
        ax_b.text(n-0.5, 0.52, "50%", fontsize=13, color="#888")
        ax_b.legend(fontsize=13, framealpha=0.8, loc="upper left")
    else:
        ax_b.text(0.5, 0.5, "Explainability results\nnot yet computed",
                  transform=ax_b.transAxes, ha="center", va="center",
                  fontsize=14, color="#888")
    ax_b.set_ylabel("Paratope Recall (fraction of contact residues attributed)")
    ax_b.text(-0.12, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=19, fontweight="bold", va="bottom")

    plt.tight_layout()
    _save(fig, "panel_D_compact.png")


if __name__ == "__main__":
    print(f"Generating paper figures â†’ {OUT_PAPER}\n")
    panel_A()
    panel_B()
    panel_C()
    panel_D()
    print(f"\nAll panels saved to:\n  {OUT_PAPER}\n  {OUT_NEW}")

