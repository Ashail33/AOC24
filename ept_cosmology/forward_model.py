"""Forward model: (eta, H, sigma) -> cosmological observables (Workstream C).

Pipeline
--------
1. Generate a (rough-)CIR / signed-Lambda trajectory whose roughness is set by H,
   mean-reversion by eta_cosmo (near-critical eta -> slow reversion -> large, long
   excursions), and fluctuation amplitude by sigma_amplitude.
2. Map the trajectory to a fractional dark-energy density perturbation delta(z)
   with RMS = sigma_amplitude, so f_DE(z) = 1 + delta(z) (Lambda-dominated, w ~ -1
   with stochastic excursions -- the everpresent-Lambda phenomenology).
3. Build the Friedmann background and evaluate observables.
4. Repeat over n_traj realisations -> mean +/- confidence band.

Validation
----------
`validate_lcdm_limit` sends sigma_amplitude -> 0 and checks the observables
collapse onto LambdaCDM (w -> -1, distances match the f_DE == 1 background) --
the eta = 0 / Sorkin Poisson everpresent-Lambda limit.

The MCMC fit against DESI is intentionally NOT implemented here (deferred per the
Workstream-A outcome); this module is the forward map that such a fit would call.
"""

from __future__ import annotations

import numpy as np

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lambda_trajectory"))
sys.path.insert(0, os.path.join(HERE, "cosmology"))

from standard_cir import simulate_cir
from rough_cir import simulate_rough_cir
from friedmann import Background
from observables import all_observables


def _trajectory(eta_cosmo, H, t_max, dt, rng, n_paths):
    """A (rough-)CIR trajectory; mean-reversion a = (1-eta_cosmo) so near-critical
    eta gives slow reversion and long excursions."""
    a = max(1e-3, 1.0 - eta_cosmo)
    b, sigma, x0 = 1.0, 0.6, 1.0
    if H >= 0.5:
        return simulate_cir(a, b, sigma, x0, t_max, dt, rng, n_paths)
    return simulate_rough_cir(a, b, sigma, x0, H, t_max, dt, rng, n_paths)


def _f_de_realisation(traj, z_grid, sigma_amplitude):
    """Map one Lambda path to f_DE(z) = 1 + delta(z), with delta the
    mean-subtracted path rescaled to RMS = sigma_amplitude and stretched across
    the redshift range (late time = low z)."""
    lam = traj - traj.mean()
    rms = lam.std()
    if rms > 0:
        lam = lam / rms * sigma_amplitude
    # map cosmic-time index -> redshift: end of trajectory = today (z=0)
    n = len(lam)
    idx = np.linspace(n - 1, 0, len(z_grid)).astype(int)  # z increases -> back in time
    return 1.0 + lam[idx]


def forward(eta_cosmo=0.9, H=0.3, sigma_amplitude=0.05, n_traj=200,
            Omega_m=0.31, H0=67.7, t_max=60.0, dt=0.05,
            z_max=1100.0, seed=0) -> dict:
    """Run the forward model and return observables with confidence bands."""
    rng = np.random.default_rng(seed)
    z_grid = np.linspace(0, z_max, 3000)
    t, paths = _trajectory(eta_cosmo, H, t_max, dt, rng, n_traj)

    z_sn = np.linspace(0.01, 2.0, 40)
    mu_stack, w_stack, H_stack = [], [], []
    DV_stack = []
    z_bao = np.array([0.38, 0.51, 0.70, 0.85, 1.49, 2.33])
    for i in range(n_traj):
        f_de = _f_de_realisation(paths[i], z_grid, sigma_amplitude)
        bg = Background(Omega_m=Omega_m, H0=H0, z_grid=z_grid, f_de=f_de)
        obs = all_observables(bg, z_sn=z_sn, z_bao=z_bao)
        mu_stack.append(obs["mu"]); w_stack.append(obs["wz"])
        H_stack.append(obs["Hz"]); DV_stack.append(obs["bao"]["DV_over_rd"])

    def band(stack):
        s = np.array(stack)
        return {"mean": s.mean(0), "lo": np.percentile(s, 16, 0),
                "hi": np.percentile(s, 84, 0)}

    return {"params": {"eta_cosmo": eta_cosmo, "H": H,
                       "sigma_amplitude": sigma_amplitude, "n_traj": n_traj},
            "z_sn": z_sn, "z_bao": z_bao,
            "mu": band(mu_stack), "wz": band(w_stack), "Hz": band(H_stack),
            "DV_over_rd": band(DV_stack)}


def validate_lcdm_limit(Omega_m=0.31, H0=67.7) -> dict:
    """sigma_amplitude -> 0 must reproduce the LambdaCDM background (w == -1,
    distances identical to f_DE == 1)."""
    z_grid = np.linspace(0, 1100, 3000)
    bg_lcdm = Background(Omega_m=Omega_m, H0=H0, z_grid=z_grid, f_de=None)
    z = np.linspace(0.01, 2.0, 40)
    mu_lcdm = bg_lcdm.distance_modulus(z)

    res = forward(eta_cosmo=0.0, H=0.5, sigma_amplitude=1e-6, n_traj=20,
                  Omega_m=Omega_m, H0=H0, z_max=1100.0)
    mu_model = res["mu"]["mean"]
    w_model = res["wz"]["mean"]
    return {"max_mu_diff": float(np.max(np.abs(mu_model - mu_lcdm))),
            "max_w_plus_one": float(np.max(np.abs(w_model + 1.0))),
            "mu_ok": bool(np.max(np.abs(mu_model - mu_lcdm)) < 1e-3),
            "w_ok": bool(np.max(np.abs(w_model + 1.0)) < 1e-3)}


def roughness_signature(eta_cosmo=0.9, sigma_amplitude=0.05, n_traj=150,
                        seed=1) -> dict:
    """Quantify how the roughness of dLambda/dz (a proxy: std of the discrete
    second difference of w(z)) varies with H -- the 'roughness signature' that
    distinguishes rough from smooth CIR."""
    out = {}
    for H in [0.1, 0.3, 0.5]:
        res = forward(eta_cosmo=eta_cosmo, H=H, sigma_amplitude=sigma_amplitude,
                      n_traj=n_traj, seed=seed)
        w = res["wz"]["mean"]
        out[H] = float(np.std(np.diff(w, 2)))
    return out


if __name__ == "__main__":
    import json
    print("LCDM limit:", json.dumps(validate_lcdm_limit(), indent=2))
    print("roughness signature (std of d^2 w):", roughness_signature())
