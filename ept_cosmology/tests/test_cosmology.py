"""Tests for the EPT cosmology forward model (Workstream C)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lambda_trajectory"))
sys.path.insert(0, os.path.join(ROOT, "cosmology"))

from standard_cir import simulate_cir, cir_stationary_moments
from rough_cir import simulate_rough_cir, estimate_roughness
from signed_lambda import simulate_signed_lambda
from friedmann import Background
from observables import all_observables, cmb_acoustic_scale
import forward_model as fm


def test_cir_stationary_variance():
    rng = np.random.default_rng(0)
    a, b, sigma = 1.0, 1.0, 0.5
    _, X = simulate_cir(a, b, sigma, 1.0, 300.0, 0.01, rng, n_paths=300)
    tail = X[:, X.shape[1] // 2:]
    mo = cir_stationary_moments(a, b, sigma)
    assert abs(tail.mean() - mo["mean"]) < 0.05
    assert abs(tail.var() - mo["var"]) < 0.2 * mo["var"]


def test_rough_cir_recovers_input_hurst():
    rng = np.random.default_rng(1)
    for H in [0.1, 0.3, 0.5]:
        _, X = simulate_rough_cir(1.0, 1.0, 0.3, 1.0, H, 20.0, 0.02, rng, n_paths=40)
        H_est = estimate_roughness(X, 0.02)
        assert abs(H_est - H) < 0.1


def test_signed_lambda_zero_mean():
    rng = np.random.default_rng(2)
    _, L = simulate_signed_lambda(1.0, 1.0, 0.5, 1.0, 200.0, 0.01, rng, n_paths=300)
    tail = L[:, L.shape[1] // 2:]
    se = tail.std() / np.sqrt(300)
    assert abs(tail.mean()) < 4 * se


def test_background_lcdm_matches_analytic_hubble():
    bg = Background(Omega_m=0.31, H0=67.7)
    # at z=0, E=1
    assert abs(bg.E(0.0) - 1.0) < 1e-12
    # monotonic increasing H(z)
    z = np.linspace(0, 3, 20)
    assert np.all(np.diff(bg.H(z)) > 0)


def test_cmb_acoustic_scale_reasonable():
    bg = Background(Omega_m=0.31, H0=67.7)
    cmb = cmb_acoustic_scale(bg)
    # l_A ~ 300 for Planck-like cosmology
    assert 250 < cmb["l_A"] < 350
    assert 1.5 < cmb["R"] < 1.9


def test_lcdm_limit_of_forward_model():
    res = fm.validate_lcdm_limit()
    assert res["mu_ok"] and res["w_ok"]


def test_forward_model_bands_have_width():
    res = fm.forward(eta_cosmo=0.9, H=0.3, sigma_amplitude=0.1, n_traj=40)
    # with sigma>0 the w(z) band should have nonzero width somewhere
    width = res["wz"]["hi"] - res["wz"]["lo"]
    assert np.max(width) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
