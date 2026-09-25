"""
Acquisition functions for pool-based active learning.

Four strategies:
  random  — uniform sampling (baseline)
  ucb     — Upper Confidence Bound: μ + β·σ
  ei      — Expected Improvement: (μ−f*−ξ)Φ(z) + σϕ(z)
  pi      — Probability of Improvement: Φ((μ−f*−ξ)/σ)

UCB β-scheduling (six variants used in ablation):
  fixed:   β ∈ {0.1, 0.5, 1.0, 2.0}
  grad:    β decreases linearly from 2.0 → 0.1
  alt:     β alternates between 2.0 and 0.1

All active strategies return the top-batch_size pool indices by score.
"""
from __future__ import annotations

import numpy as np
import torch
import gpytorch
from torch.distributions import Normal
from typing import List, Sequence


PRED_BATCH = 5_000  # prediction batch size to avoid OOM on large pools
_norm = Normal(torch.tensor(0.0), torch.tensor(1.0))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pool_indices(n_total: int, selected: Sequence[int]) -> np.ndarray:
    sel = set(selected)
    return np.array([i for i in range(n_total) if i not in sel], dtype=int)


def _predict_pool(
    X: np.ndarray,
    model,
    likelihood,
    pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, stddev) for pool indices only."""
    mu_parts, sd_parts = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(pool), PRED_BATCH):
            bx   = torch.tensor(X[pool[i:i + PRED_BATCH]]).float()
            dist = likelihood(model(bx))
            mu_parts.append(dist.mean.numpy())
            sd_parts.append(dist.stddev.numpy())
    return np.concatenate(mu_parts), np.concatenate(sd_parts)


def _top_k(scores: np.ndarray, pool: np.ndarray, batch: int) -> List[int]:
    top = np.argsort(scores)[::-1][:batch]
    return pool[top].tolist()


# ── Acquisition functions ─────────────────────────────────────────────────────
def select_random(
    n_total: int,
    selected: Sequence[int],
    batch_size: int,
    rng: np.random.Generator,
) -> List[int]:
    """Uniform random selection from unlabelled pool (baseline)."""
    pool = _pool_indices(n_total, selected)
    k    = min(batch_size, len(pool))
    return rng.choice(pool, k, replace=False).tolist()


def select_ucb(
    X: np.ndarray,
    model,
    likelihood,
    selected: Sequence[int],
    batch_size: int,
    beta: float = 1.0,
) -> List[int]:
    """Upper Confidence Bound: score = μ(x) + β·σ(x).

    Args:
        beta: exploration weight. Higher = more exploration.
              β=1.0 is the standard default used throughout the paper.
    """
    pool    = _pool_indices(len(X), selected)
    mu, sd  = _predict_pool(X, model, likelihood, pool)
    return _top_k(mu + beta * sd, pool, batch_size)


def select_ei(
    X: np.ndarray,
    model,
    likelihood,
    selected: Sequence[int],
    batch_size: int,
    best_y: float,
    xi: float = 0.01,
) -> List[int]:
    """Expected Improvement: (μ−f*−ξ)Φ(z) + σϕ(z), z = (μ−f*−ξ)/σ.

    Args:
        best_y: current best observed affinity value f*
        xi:     jitter to prevent over-exploitation (default 0.01)
    """
    pool   = _pool_indices(len(X), selected)
    mu, sd = _predict_pool(X, model, likelihood, pool)
    mu_t   = torch.tensor(mu); sd_t = torch.tensor(sd)
    z      = (mu_t - best_y - xi) / (sd_t + 1e-9)
    scores = ((mu_t - best_y - xi) * _norm.cdf(z)
              + sd_t * torch.exp(_norm.log_prob(z))).numpy()
    return _top_k(scores, pool, batch_size)


def select_pi(
    X: np.ndarray,
    model,
    likelihood,
    selected: Sequence[int],
    batch_size: int,
    best_y: float,
    xi: float = 0.01,
) -> List[int]:
    """Probability of Improvement: Φ((μ−f*−ξ)/σ).

    Most exploitation-heavy of the three active strategies — only considers
    probability of improvement, not magnitude.

    Args:
        best_y: current best observed affinity value f*
        xi:     jitter to prevent over-exploitation (default 0.01)
    """
    pool   = _pool_indices(len(X), selected)
    mu, sd = _predict_pool(X, model, likelihood, pool)
    z      = (torch.tensor(mu) - best_y - xi) / (torch.tensor(sd) + 1e-9)
    return _top_k(_norm.cdf(z).numpy(), pool, batch_size)


# ── UCB β-scheduling ──────────────────────────────────────────────────────────
def get_beta(variant: str, cycle: int, n_cycles: int) -> float:
    """Return β for a given UCB variant and cycle.

    Variants:
      'b0.1'  fixed β=0.1  (exploit-heavy)
      'b0.5'  fixed β=0.5
      'b1.0'  fixed β=1.0  (standard default)
      'b2.0'  fixed β=2.0  (explore-heavy)
      'grad'  β decreases linearly 2.0→0.1 over n_cycles
      'alt'   β alternates 2.0/0.1 each cycle

    Args:
        variant:  one of the six strings above
        cycle:    current AL cycle (0-indexed)
        n_cycles: total number of AL cycles (used for 'grad')
    """
    if variant == "b0.1":  return 0.1
    if variant == "b0.5":  return 0.5
    if variant == "b1.0":  return 1.0
    if variant == "b2.0":  return 2.0
    if variant == "grad":
        t = cycle / max(n_cycles - 1, 1)
        return 2.0 - (2.0 - 0.1) * t
    if variant == "alt":
        return 2.0 if cycle % 2 == 0 else 0.1
    raise ValueError(f"Unknown β variant '{variant}'. "
                     f"Choose: b0.1, b0.5, b1.0, b2.0, grad, alt")
