"""Standard CIR Lambda(t) generator -- the H = 1/2 case (Workstream C).

The Jaisson-Rosenbaum scaling limit of a nearly-unstable Hawkes process with an
*integrable* (light-tailed) kernel is the Cox-Ingersoll-Ross (CIR) process

    dX_t = a (b - X_t) dt + sigma sqrt(X_t) dW_t ,   X_t >= 0.

This module simulates it with the full-truncation Euler scheme (Lord et al.),
which keeps the process non-negative.  X_t plays the role of the (rescaled,
mean) cosmological-constant intensity Lambda(t) in the everpresent-Lambda model.
"""

from __future__ import annotations

import numpy as np


def simulate_cir(a: float, b: float, sigma: float, x0: float,
                 t_max: float, dt: float, rng: np.random.Generator,
                 n_paths: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `n_paths` CIR trajectories.  Returns (t, X) with X of shape
    (n_paths, n_steps+1)."""
    n = int(np.round(t_max / dt))
    t = np.arange(n + 1) * dt
    X = np.empty((n_paths, n + 1))
    X[:, 0] = x0
    sqrt_dt = np.sqrt(dt)
    x = np.full(n_paths, float(x0))
    for k in range(n):
        xp = np.maximum(x, 0.0)  # full truncation
        dW = rng.standard_normal(n_paths) * sqrt_dt
        x = x + a * (b - xp) * dt + sigma * np.sqrt(xp) * dW
        X[:, k + 1] = x
    return t, X


def cir_stationary_moments(a: float, b: float, sigma: float) -> dict:
    """Stationary mean and variance of the CIR process."""
    return {"mean": b, "var": sigma**2 * b / (2 * a)}


def feller_condition(a: float, b: float, sigma: float) -> bool:
    """2 a b >= sigma^2 guarantees X stays strictly positive."""
    return 2 * a * b >= sigma**2
