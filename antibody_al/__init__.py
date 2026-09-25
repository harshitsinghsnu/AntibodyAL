"""
Antibody Active Learning — core package.
ESM-2 + Tanimoto-kernel GP active learning for antibody affinity prediction.
"""
from .gp_model import TanimotoKernel, GPRegressionModel, build_kernel, train_gp
from .acquisition import select_random, select_ucb, select_ei, select_pi, get_beta
from .active_learning import stratified_seed, run_al_cycle, run_experiment

__all__ = [
    "TanimotoKernel", "GPRegressionModel", "build_kernel", "train_gp",
    "select_random", "select_ucb", "select_ei", "select_pi", "get_beta",
    "stratified_seed", "run_al_cycle", "run_experiment",
]
