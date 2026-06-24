"""Convergence / sanity checks for the Lambda(t) generators (Workstream C)."""

from __future__ import annotations

import numpy as np

from standard_cir import simulate_cir, cir_stationary_moments, feller_condition
from rough_cir import simulate_rough_cir, estimate_roughness
from signed_lambda import simulate_signed_lambda


def check_cir_stationary(a=1.0, b=1.0, sigma=0.5, x0=1.0,
                         t_max=400.0, dt=0.01, n_paths=400, seed=0) -> dict:
    """The CIR stationary mean/variance should match the analytic formulas after
    burn-in."""
    rng = np.random.default_rng(seed)
    t, X = simulate_cir(a, b, sigma, x0, t_max, dt, rng, n_paths)
    burn = X.shape[1] // 2
    tail = X[:, burn:]
    emp_mean = float(tail.mean())
    emp_var = float(tail.var())
    mo: dict = cir_stationary_moments(a, b, sigma)
    return {"feller_ok": feller_condition(a, b, sigma),
            "emp_mean": emp_mean, "pred_mean": mo["mean"],
            "emp_var": emp_var, "pred_var": mo["var"],
            "mean_ok": abs(emp_mean - mo["mean"]) < 0.05 * mo["mean"],
            "var_ok": abs(emp_var - mo["var"]) < 0.20 * mo["var"]}


def check_rough_recovers_cir(a=1.0, b=1.0, sigma=0.4, x0=1.0,
                             t_max=40.0, dt=0.04, n_paths=100, seed=1) -> dict:
    """At H = 0.5 the rough-CIR Volterra scheme should reproduce the CIR
    stationary mean."""
    rng = np.random.default_rng(seed)
    t, X = simulate_rough_cir(a, b, sigma, x0, 0.5, t_max, dt, rng, n_paths)
    burn = X.shape[1] // 2
    emp_mean = float(X[:, burn:].mean())
    return {"emp_mean": emp_mean, "pred_mean": b,
            "mean_ok": abs(emp_mean - b) < 0.08 * b}


def check_roughness_scaling(a=1.0, b=1.0, sigma=0.3, x0=1.0,
                            t_max=20.0, dt=0.02, seed=2) -> dict:
    """The estimated path roughness should track the input Hurst parameter
    (qualitatively: smaller H -> rougher)."""
    rng = np.random.default_rng(seed)
    out = {}
    for H in [0.1, 0.3, 0.5]:
        t, X = simulate_rough_cir(a, b, sigma, x0, H, t_max, dt, rng, n_paths=40)
        out[H] = estimate_roughness(X, dt)
    monotone = out[0.1] <= out[0.3] <= out[0.5] + 0.05
    return {"H_estimates": out, "monotone_in_H": bool(monotone)}


def check_signed_zero_mean(a=1.0, b=1.0, sigma=0.5, x0=1.0,
                           t_max=200.0, dt=0.01, n_paths=400, seed=3) -> dict:
    """The signed process Lambda_+ - Lambda_- must have mean ~ 0."""
    rng = np.random.default_rng(seed)
    t, L = simulate_signed_lambda(a, b, sigma, x0, t_max, dt, rng,
                                  H=0.5, n_paths=n_paths)
    burn = L.shape[1] // 2
    emp_mean = float(L[:, burn:].mean())
    emp_rms = float(np.sqrt((L[:, burn:] ** 2).mean()))
    se = emp_rms / np.sqrt(n_paths)
    return {"emp_mean": emp_mean, "emp_rms": emp_rms,
            "zero_mean_ok": abs(emp_mean) < 3 * se}


def run_all() -> dict:
    return {"cir_stationary": check_cir_stationary(),
            "rough_recovers_cir": check_rough_recovers_cir(),
            "roughness_scaling": check_roughness_scaling(),
            "signed_zero_mean": check_signed_zero_mean()}


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2, default=str))
