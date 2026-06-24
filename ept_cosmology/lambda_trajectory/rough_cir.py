"""Rough (fractional) CIR Lambda(t) generator for H < 1/2 (Workstream C).

When the nearly-unstable Hawkes kernel has a power-law tail (regularly varying of
index 1+alpha, alpha in (0,1)), the Jaisson-Rosenbaum / El Euch-Rosenbaum scaling
limit is a *rough* CIR process governed by a Volterra equation

    X_t = X_0 + int_0^t K(t-s) a (b - X_s) ds + int_0^t K(t-s) sigma sqrt(X_s) dW_s

with the fractional kernel

    K(t) = t^{H - 1/2} / Gamma(H + 1/2),    H = alpha - 1/2 in (-1/2, 1/2).

For H = 1/2 the kernel is constant (K = 1) and the standard CIR is recovered.
For H < 1/2 the kernel is singular at 0 and the trajectories are rough (Holder
regularity H).

We integrate the Volterra equation with the explicit scheme of El Euch &
Rosenbaum (analytic-weight discretisation of the convolution), which is robust
and avoids the singular kernel blowing up.  This mirrors the rough-Heston
variance-process simulators in the literature.

Connection to Workstream A: the lattice model there gives effective kernel tail
(d-1)/2, i.e. 1+alpha = (d-1)/2 -> H = alpha - 1/2 = (d-4)/2.  For d=4 this is the
boundary H = 0 (very rough); the smooth H = 1/2 CIR requires an integrable kernel
(the original spacetime conjecture's d/2 with d >= 5).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


def fractional_kernel(t: np.ndarray, H: float) -> np.ndarray:
    """K(t) = t^{H-1/2} / Gamma(H+1/2)  (for t > 0)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = t[pos] ** (H - 0.5) / gamma(H + 0.5)
    return out


def _kernel_weights(H: float, dt: float, n: int) -> np.ndarray:
    """Integrated kernel weights b_j = int_{j dt}^{(j+1) dt} K(s) ds, j=0..n-1,
    using the analytic antiderivative of t^{H-1/2}:
        int t^{H-1/2} dt = t^{H+1/2}/(H+1/2).
    These weights tame the t=0 singularity of K."""
    j = np.arange(n + 1) * dt
    F = j ** (H + 0.5) / (gamma(H + 0.5) * (H + 0.5))  # antiderivative at grid pts
    return np.diff(F)  # length n


def simulate_rough_cir(a: float, b: float, sigma: float, x0: float, H: float,
                       t_max: float, dt: float, rng: np.random.Generator,
                       n_paths: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `n_paths` rough-CIR trajectories via the Volterra-Euler scheme.

    Returns (t, X), X of shape (n_paths, n_steps+1).  For H = 0.5 this reduces to
    (a discretisation of) the standard CIR.

    Cost: this naive scheme convolves the full history at every step, so it is
    O(n_paths * n_steps^2).  Keep n_steps <~ a few thousand.  For production-scale
    runs replace the convolution with a sum-of-exponentials (Markovian) kernel
    approximation a la El Euch-Rosenbaum to reach O(n_paths * n_steps * N_exp);
    that optimisation is deferred with the rest of the (gated) Workstream-C MCMC.
    """
    if not (0.0 < H <= 0.5):
        raise ValueError("H must be in (0, 0.5] for rough/standard CIR")
    n = int(np.round(t_max / dt))
    t = np.arange(n + 1) * dt
    w = _kernel_weights(H, dt, n)              # convolution weights, length n
    sqrt_dt = np.sqrt(dt)

    X = np.empty((n_paths, n + 1))
    X[:, 0] = x0
    # store the drift and diffusion increments to convolve with the kernel
    drift_incr = np.zeros((n_paths, n))        # a (b - X_s) at each step
    diff_incr = np.zeros((n_paths, n))         # sigma sqrt(X_s) dW_s / dt-ish

    x = np.full(n_paths, float(x0))
    for k in range(n):
        xp = np.maximum(x, 0.0)
        drift_incr[:, k] = a * (b - xp)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        diff_incr[:, k] = sigma * np.sqrt(xp) * dW / dt  # store as a rate

        # X_{k+1} = X_0 + sum_{j=0}^{k} w[k-j] * (drift_incr[j]*dt + diff_incr[j]*dt)
        wk = w[:k + 1][::-1]                    # w[k], w[k-1], ..., w[0]
        conv_drift = drift_incr[:, :k + 1] @ wk
        conv_diff = diff_incr[:, :k + 1] @ wk
        x = x0 + (conv_drift + conv_diff) * dt
        X[:, k + 1] = x
    return t, X


def estimate_roughness(X: np.ndarray, dt: float) -> float:
    """Estimate the Holder/Hurst regularity of a path from the scaling of
    p-variation-like increments:  E[|X_{t+s}-X_t|^2] ~ s^{2H}.  Returns the
    fitted H (averaged over paths)."""
    X = np.atleast_2d(X)
    lags = np.unique(np.round(np.logspace(0, np.log10(X.shape[1] // 4), 12)).astype(int))
    lags = lags[lags >= 1]
    Hs = []
    for path in X:
        m2 = []
        for L in lags:
            d = path[L:] - path[:-L]
            m2.append(np.mean(d**2))
        m2 = np.array(m2)
        good = m2 > 0
        slope = np.polyfit(np.log(lags[good] * dt), np.log(m2[good]), 1)[0]
        Hs.append(slope / 2)
    return float(np.mean(Hs))
