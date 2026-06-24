"""The 4-cycle plaquette (Z2 x Z2 causal graph) from the eta-unification analysis
(Workstream B).

Vertices A, B, C, D (indices 0,1,2,3).
Edges:
  * spacelike (bidirectional):  A<->B,  C<->D
  * timelike  (one-directional): A->D,  B->C   (Lorentzian orientation)

So the directed cross-excitation adjacency (source -> target) is
  A -> B (spacelike), A -> D (timelike)
  B -> A (spacelike), B -> C (timelike)
  C -> D (spacelike)
  D -> C (spacelike)

Each vertex additionally self-excites.

Branching / criticality control
--------------------------------
We build a structure matrix M with M[i,i] = 1 (self) and M[i,j] = rho for every
directed edge j -> i, where rho = eta_cross / eta_self is the cross/self ratio.
The branching matrix is then
        Phi = (eta / spectral_radius(M)) * M ,
so that  spectral_radius(Phi) = eta  exactly.  Thus the swept parameter `eta` is
the overall branching ratio / criticality of the multivariate Hawkes process,
directly comparable to the Onaga-Shinomoto bursting threshold
eta* = 1 - 1/sqrt(2) ~= 0.293 and to the critical point eta = 1.
"""

from __future__ import annotations

import numpy as np

A, B, C, D = 0, 1, 2, 3
LABELS = ["A", "B", "C", "D"]

# directed cross edges as (source, target)
CROSS_EDGES = [(A, B), (B, A),   # spacelike A<->B
               (C, D), (D, C),   # spacelike C<->D
               (A, D),           # timelike A->D
               (B, C)]           # timelike B->C


def structure_matrix(rho: float) -> np.ndarray:
    """M[i,j] = 1 if i==j (self) else rho if there is a directed edge j->i."""
    M = np.eye(4)
    for (src, tgt) in CROSS_EDGES:
        M[tgt, src] = rho
    return M


def branching_matrix(eta: float, rho: float) -> np.ndarray:
    """Phi[i,j] = norm of the kernel through which vertex j excites vertex i,
    normalised so that spectral_radius(Phi) == eta."""
    M = structure_matrix(rho)
    sr = np.max(np.abs(np.linalg.eigvals(M)))
    return (eta / sr) * M


def stationary_intensity(eta: float, rho: float, mu: np.ndarray) -> np.ndarray:
    """Stationary mean intensity vector m = (I - Phi)^{-1} mu for the
    multivariate Hawkes process (valid for eta < 1)."""
    Phi = branching_matrix(eta, rho)
    return np.linalg.solve(np.eye(4) - Phi, np.asarray(mu, dtype=float))


def window_from_intensity(eta: float, rho: float, mu: np.ndarray,
                          beta: float, factor: float = 4.0) -> float:
    """Sliding-window size W = factor / (beta * mean_intensity).  This is the
    `W = 4/(beta mu_eff)` prescription with mu_eff = mean stationary intensity,
    so the window scales with the inverse event rate as eta varies."""
    m = stationary_intensity(eta, rho, mu)
    mu_eff = float(np.mean(m))
    return factor / (beta * mu_eff)
