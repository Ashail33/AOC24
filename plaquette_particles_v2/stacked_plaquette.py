"""Two-time-step (T-layer) 4-cycle stack -- the causal-structure-preserving
successor to the v1 plaquette (Workstream B v2).

Motivation
----------
The v1 plaquette has NO directed Hamiltonian cycle (the Lorentzian orientation
A->D, B->C is one-directional and there is no D->A edge), so the "cycle"
A->B->C->D cannot close and cannot self-sustain -- the measured S(eta) peak was a
rate x lifetime artifact (cycles-per-chain ~ 1).  Adding a D->A closing edge
would create a closed timelike curve.

Instead we STACK T copies of the plaquette along the time direction:
  vertices (t, v) for t in [0, T), v in {A,B,C,D},  flat index 4*t + v.
  * within-layer edges = the v1 plaquette (A<->B, C<->D spacelike;
    A->D, B->C timelike), which supports the *path* A->B->C->D inside a layer;
  * inter-layer edges (t, v) -> (t+1, v), strictly forward in the layer index.

The forward inter-layer arrows are the time direction (no CTC).  A "cycle" is the
horizontal traversal A_t->B_t->C_t->D_t at fixed layer t; genuine particle-like
persistence is that the SAME cycle recurs at layer t+1, driven causally by the
vertical arrows.  This is the observable v1 could not test.

Parameters: eta (spectral radius / criticality), rho = eta_cross/eta_self within
a layer, rho_time = inter-layer / self coupling ratio, T = number of layers.
"""

from __future__ import annotations

import numpy as np

A, B, C, D = 0, 1, 2, 3
LABELS = ["A", "B", "C", "D"]

# within-layer directed cross edges (source, target), same as v1
LAYER_EDGES = [(A, B), (B, A), (C, D), (D, C), (A, D), (B, C)]


def structure_matrix(T: int, rho: float, rho_time: float) -> np.ndarray:
    """(4T, 4T) structure matrix M[i,j]: self=1, within-layer cross edge=rho,
    forward inter-layer edge (t,v)->(t+1,v)=rho_time."""
    n = 4 * T
    M = np.eye(n)
    for t in range(T):
        base = 4 * t
        for (src, tgt) in LAYER_EDGES:
            M[base + tgt, base + src] = rho
        if t + 1 < T:
            for v in range(4):
                M[4 * (t + 1) + v, base + v] = rho_time  # (t,v) -> (t+1,v)
    return M


def branching_matrix(eta: float, T: int, rho: float, rho_time: float) -> np.ndarray:
    """Phi normalised so spectral_radius(Phi) == eta."""
    M = structure_matrix(T, rho, rho_time)
    sr = np.max(np.abs(np.linalg.eigvals(M)))
    return (eta / sr) * M


def background_vector(T: int, mu_seed: float = 0.2, mu_bg: float = 0.02) -> np.ndarray:
    """Background rates: seed mainly at A_0 (start of the cycle at the first
    layer); small uniform background elsewhere so the stack is excitable."""
    mu = np.full(4 * T, mu_bg)
    mu[A] = mu_seed
    return mu


def stationary_intensity(eta, T, rho, rho_time, mu) -> np.ndarray:
    Phi = branching_matrix(eta, T, rho, rho_time)
    return np.linalg.solve(np.eye(4 * T) - Phi, np.asarray(mu, float))


def layer_of(index: int) -> int:
    return index // 4


def node_of(index: int) -> int:
    return index % 4
