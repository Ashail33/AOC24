"""Vertex-coupled multivariate Hawkes dynamics on the 4-cycle plaquette
(Workstream B).

Standard multivariate Hawkes with exponential kernels:
    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} Phi[i,j] * beta * exp(-beta (t - t_k^j)),
with Phi the branching matrix from `plaquette.branching_matrix` (so the integral
of the i<-j kernel is Phi[i,j], i.e. mean offspring at i per event at j).

Simulated by Ogata thinning.  Exponential kernels give an exact O(1) per-event
recursion for the four memory variables g_i (decay between events, kicked on each
event), so the thinning upper bound is just the post-event total intensity
(intensity is monotone decreasing between events).
"""

from __future__ import annotations

import numpy as np

from plaquette import branching_matrix


def simulate(eta: float, rho: float, mu, beta: float, t_max: float,
             rng: np.random.Generator, max_events: int = 5_000_000):
    """Simulate the 4-vertex Hawkes process on [0, t_max].

    Returns (times, marks) with `marks` in {0,1,2,3} = {A,B,C,D}, sorted by time.
    """
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (4,):
        raise ValueError("mu must have shape (4,)")
    Phi = branching_matrix(eta, rho)        # Phi[i,j]: j excites i
    # kick to g_i when vertex j fires = Phi[i,j]*beta
    kick = Phi * beta                       # kick[:, j] applied when j fires

    g = np.zeros(4)                         # excitation memory per vertex
    mu_total = mu.sum()

    times = []
    marks = []
    t = 0.0
    sum_g = 0.0
    while True:
        lam_bar = mu_total + sum_g          # upper bound (post-event intensity)
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t >= t_max:
            break
        decay = np.exp(-beta * w)
        g *= decay
        sum_g = g.sum()
        lam = mu + g                        # per-vertex intensity at t
        lam_tot = mu_total + sum_g
        if rng.random() * lam_bar <= lam_tot:
            # choose which vertex fired, proportional to lam_i
            j = int(np.searchsorted(np.cumsum(lam), rng.random() * lam_tot, side="right"))
            j = min(j, 3)
            times.append(t)
            marks.append(j)
            g += kick[:, j]
            sum_g = g.sum()
            if len(times) > max_events:
                raise RuntimeError("exceeded max_events; reduce t_max or eta")
    return np.asarray(times), np.asarray(marks, dtype=np.int64)


def empirical_intensity(times, t_max) -> float:
    """Total event rate (events per unit time, summed over vertices)."""
    return len(times) / t_max
