"""Unit tests for the plaquette model (Workstream B)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plaquette import (branching_matrix, structure_matrix, stationary_intensity,
                       CROSS_EDGES, window_from_intensity)
from hawkes_dynamics import simulate
from pattern_detector import detect_chains, detect_twisted, PATTERNS
from stability_metrics import metrics_from_chains


def test_branching_matrix_spectral_radius_equals_eta():
    for eta in [0.2, 0.5, 0.9]:
        for rho in [0.3, 1.0, 3.0]:
            Phi = branching_matrix(eta, rho)
            sr = np.max(np.abs(np.linalg.eigvals(Phi)))
            assert abs(sr - eta) < 1e-7


def test_structure_matrix_encodes_edges():
    M = structure_matrix(rho=0.7)
    for (src, tgt) in CROSS_EDGES:
        assert M[tgt, src] == 0.7
    assert np.all(np.diag(M) == 1.0)


def test_stationary_intensity_positive_and_consistent():
    mu = np.array([0.2, 0.05, 0.05, 0.05])
    m = stationary_intensity(0.5, 1.0, mu)
    Phi = branching_matrix(0.5, 1.0)
    # m must satisfy m = mu + Phi m
    assert np.allclose(m, mu + Phi @ m)
    assert np.all(m > 0)


def test_simulate_mean_intensity_matches_stationary():
    mu = np.array([0.3, 0.1, 0.1, 0.1])
    eta, rho, beta = 0.5, 1.0, 1.0
    rng = np.random.default_rng(0)
    T = 20000.0
    times, marks = simulate(eta, rho, mu, beta, t_max=T, rng=rng)
    m_pred = stationary_intensity(eta, rho, mu)
    for v in range(4):
        emp = np.sum(marks == v) / T
        assert abs(emp - m_pred[v]) < 0.05 * m_pred[v] + 0.02


def test_detector_counts_clean_cycles():
    times = np.arange(12, dtype=float)
    marks = np.array([0, 1, 2, 3] * 3)
    ch = detect_chains(times, marks, PATTERNS["cycle"], window=2.0)
    assert len(ch) == 1 and ch[0].n_cycles == 3


def test_detector_breaks_on_gap():
    times = np.array([0, 1, 2, 3, 50, 51, 52, 53], dtype=float)
    marks = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    ch = detect_chains(times, marks, PATTERNS["cycle"], window=2.0)
    assert len(ch) == 2 and all(c.n_cycles == 1 for c in ch)


def test_twisted_detection():
    times = np.arange(6, dtype=float)
    marks = np.array([0, 1, 2, 3, 0, 3])
    assert detect_twisted(times, marks, window=2.0) == 1


def test_metrics_S_is_product():
    times = np.arange(12, dtype=float)
    marks = np.array([0, 1, 2, 3] * 3)
    ch = detect_chains(times, marks, PATTERNS["cycle"], window=2.0)
    m = metrics_from_chains(ch, t_max=12.0)
    assert abs(m["S"] - m["tau_cycle"] * m["r_cycle"]) < 1e-12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
