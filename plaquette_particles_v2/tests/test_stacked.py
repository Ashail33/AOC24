"""Tests for the stacked-plaquette v2 model."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stacked_plaquette import (branching_matrix, structure_matrix,
                               background_vector, stationary_intensity)
from hawkes_nd import simulate, stationary_check
from persistence import (detect_layer_cycle_times, cross_layer_propagation,
                         persistence_length)


def test_spectral_radius_equals_eta():
    # M is nonnegative but imprimitive (the cycle structure gives complex
    # eigenvalues of equal modulus to the real Perron root), so eigvals resolves
    # the spectral radius only to ~1e-2 here; the construction Phi=(eta/sr)*M makes
    # the true spectral radius exactly eta and the lift(eta) science is insensitive
    # to a sub-percent offset.
    for eta in [0.2, 0.5, 0.8]:
        Phi = branching_matrix(eta, T=8, rho=1.0, rho_time=1.5)
        sr = np.max(np.abs(np.linalg.eigvals(Phi)))
        assert abs(sr - eta) < 1e-2


def test_structure_has_forward_interlayer_only():
    M = structure_matrix(T=3, rho=0.5, rho_time=2.0)
    # (t,v) -> (t+1,v): M[4*(t+1)+v, 4*t+v] == rho_time, forward only
    for t in range(2):
        for v in range(4):
            assert M[4 * (t + 1) + v, 4 * t + v] == 2.0
            assert M[4 * t + v, 4 * (t + 1) + v] == 0.0   # no backward (no CTC)


def test_nd_hawkes_mean_intensity_converges():
    T = 6
    Phi = branching_matrix(0.6, T, rho=1.0, rho_time=1.0)
    mu = background_vector(T)
    rng = np.random.default_rng(0)
    emp, pred = stationary_check(Phi, mu, 1.0, 20000.0, rng)
    assert np.mean(np.abs(emp - pred) / pred) < 0.05


def test_layer_cycle_detection():
    # one clean A->B->C->D at layer 2 (global marks 8,9,10,11)
    times = np.array([0.0, 1.0, 2.0, 3.0])
    marks = np.array([8, 9, 10, 11])
    comp = detect_layer_cycle_times(times, marks, layer=2, window=2.0)
    assert len(comp) == 1 and comp[0] == 3.0


def test_propagation_detected_on_synthetic_stack():
    # cycle at layer 0 then layer 1 shortly after -> should propagate
    times = np.array([0., 1., 2., 3.,   4., 5., 6., 7.])
    marks = np.array([0, 1, 2, 3,       4, 5, 6, 7])   # layer0 ABCD then layer1 ABCD
    r = cross_layer_propagation(times, marks, T=2, window=2.0, prop_window=5.0)
    assert r["considered"] == 1 and r["p_prop"] == 1.0


def test_persistence_length_counts_run():
    # cycle propagates through 3 layers
    times = np.arange(12, dtype=float)
    marks = np.array([0,1,2,3, 4,5,6,7, 8,9,10,11])
    L = persistence_length(times, marks, T=3, window=2.0, prop_window=5.0)
    assert L >= 2.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
