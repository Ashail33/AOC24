"""General n-vertex multivariate Hawkes simulation via Ogata thinning
(Workstream B v2).

Same exact exponential-kernel recursion as the v1 dynamics but for an arbitrary
branching matrix Phi (n x n, Phi[i,j] = mean offspring at i per event at j) and
background vector mu (length n).  Used by the stacked plaquette.

Exponential kernels give an exact O(n) per-event recursion: a memory variable
g_i decays as exp(-beta dt) between events and is kicked by Phi[i,j]*beta when
vertex j fires.  Intensity is monotone decreasing between events, so the Ogata
upper bound is the post-event total intensity (rejection only at accepted
events).
"""

from __future__ import annotations

import numpy as np


def simulate(Phi: np.ndarray, mu: np.ndarray, beta: float, t_max: float,
             rng: np.random.Generator, max_events: int = 5_000_000):
    """Return (times, marks) for the n-vertex Hawkes process on [0, t_max]."""
    Phi = np.asarray(Phi, float)
    mu = np.asarray(mu, float)
    n = Phi.shape[0]
    kick = Phi * beta                 # kick[:, j] added to g when vertex j fires
    g = np.zeros(n)
    mu_total = mu.sum()

    times, marks = [], []
    t = 0.0
    sum_g = 0.0
    while True:
        lam_bar = mu_total + sum_g    # upper bound = intensity right after last event
        if lam_bar <= 1e-12:
            break
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t >= t_max:
            break
        g *= np.exp(-beta * w)        # decay memory to time t
        sum_g = g.sum()
        lam = mu + g
        lam_tot = mu_total + sum_g
        if rng.random() * lam_bar <= lam_tot:                 # accept
            j = int(np.searchsorted(np.cumsum(lam), rng.random() * lam_tot,
                                    side="right"))
            j = min(j, n - 1)
            times.append(t)
            marks.append(j)
            g += kick[:, j]
            sum_g = g.sum()
            if len(times) > max_events:
                raise RuntimeError("exceeded max_events; reduce t_max or eta")
    return np.asarray(times), np.asarray(marks, dtype=np.int64)


def stationary_check(Phi, mu, beta, t_max, rng):
    """Empirical per-vertex intensity vs analytic (I-Phi)^{-1} mu (for tests)."""
    times, marks = simulate(Phi, mu, beta, t_max, rng)
    n = Phi.shape[0]
    emp = np.array([np.sum(marks == v) for v in range(n)]) / t_max
    pred = np.linalg.solve(np.eye(n) - Phi, np.asarray(mu, float))
    return emp, pred
