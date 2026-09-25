"""
GP model components: TanimotoKernel, GPRegressionModel, kernel builder, trainer.

Exact GP with four covariance functions:
  tanimoto  — continuous Tanimoto similarity (Gessner et al. 2024)
  matern    — Matérn-5/2
  rbf       — Radial Basis Function
  rq        — Rational Quadratic

All kernels are wrapped in ScaleKernel (learnable output scale).
Training: Adam, ExponentialLR decay, exact marginal log-likelihood.
"""
from __future__ import annotations

import gpytorch
import torch


# ── Tanimoto kernel ────────────────────────────────────────────────────────────
class TanimotoKernel(gpytorch.kernels.Kernel):
    """Continuous Tanimoto similarity kernel.

    k(x, x') = (x·x') / (||x||² + ||x'||² - x·x')

    Scale-invariant, non-stationary, and well matched to PLM embedding spaces.
    Reference: Gessner et al. NeurIPS Workshop on Bayesian Decision-making
    and Uncertainty, 2024.
    """
    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
        x1_norm    = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm    = x2.pow(2).sum(dim=-1, keepdim=True)
        x1_dot_x2  = torch.matmul(x1, x2.transpose(-1, -2))
        denominator = x1_norm + x2_norm.transpose(-1, -2) - x1_dot_x2
        return x1_dot_x2 / denominator.clamp(min=1e-9)


# ── GP regression model ────────────────────────────────────────────────────────
class GPRegressionModel(gpytorch.models.ExactGP):
    """Exact GP regression with constant mean and configurable kernel."""
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(
            mean_x, covar_x.add_jitter(1e-6)
        )


# ── Kernel factory ─────────────────────────────────────────────────────────────
def build_kernel(name: str) -> gpytorch.kernels.Kernel:
    """Return a ScaleKernel-wrapped kernel by name.

    Args:
        name: one of 'tanimoto', 'matern', 'rbf', 'rq'
    """
    if name == "tanimoto":
        return gpytorch.kernels.ScaleKernel(TanimotoKernel())
    base = {
        "matern": gpytorch.kernels.MaternKernel(nu=2.5),
        "rbf":    gpytorch.kernels.RBFKernel(),
        "rq":     gpytorch.kernels.RQKernel(),
    }.get(name)
    if base is None:
        raise ValueError(f"Unknown kernel '{name}'. Choose: tanimoto, matern, rbf, rq")
    return gpytorch.kernels.ScaleKernel(base)


# ── GP trainer ─────────────────────────────────────────────────────────────────
def train_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    model: GPRegressionModel,
    epochs: int = 100,
    lr: float = 0.1,
    lr_decay: float = 0.95,
) -> None:
    """Train GP by maximising exact marginal log-likelihood.

    Args:
        train_x:    [N, D] float tensor
        train_y:    [N]    float tensor
        likelihood: GaussianLikelihood instance
        model:      GPRegressionModel instance
        epochs:     Adam steps (default 100)
        lr:         initial learning rate (default 0.1)
        lr_decay:   ExponentialLR gamma per step (default 0.95)
    """
    model.train(); likelihood.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay)
    mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimiser.step()
        scheduler.step()
    model.eval(); likelihood.eval()
