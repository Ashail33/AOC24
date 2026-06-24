"""Signed Lambda = Lambda_+ - Lambda_- for the everpresent-Lambda context
(Workstream C).

Sorkin's everpresent (fluctuating) cosmological constant has <Lambda> = 0 with
fluctuations Lambda ~ +/- 1/sqrt(V) at spacetime volume V.  We realise the signed
process as the difference of two independent non-negative (rough-)CIR processes
with identical parameters, so that by symmetry <Lambda_+ - Lambda_-> = 0 while the
fluctuation amplitude is controlled by the CIR variance.

This is the natural signed analogue of the Jaisson-Rosenbaum positive intensity
limit and is what feeds the cosmological forward model in the everpresent-Lambda
scenario (where Lambda must fluctuate around zero, not around a positive mean).
"""

from __future__ import annotations

import numpy as np

from standard_cir import simulate_cir
from rough_cir import simulate_rough_cir


def simulate_signed_lambda(a: float, b: float, sigma: float, x0: float,
                           t_max: float, dt: float, rng: np.random.Generator,
                           H: float = 0.5, n_paths: int = 1
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Lambda(t) = Lambda_+(t) - Lambda_-(t), two i.i.d. (rough-)CIR processes.

    Returns (t, Lambda), Lambda of shape (n_paths, n_steps+1) with <Lambda> ~ 0.
    """
    if H == 0.5:
        t, Xp = simulate_cir(a, b, sigma, x0, t_max, dt, rng, n_paths)
        _, Xm = simulate_cir(a, b, sigma, x0, t_max, dt, rng, n_paths)
    else:
        t, Xp = simulate_rough_cir(a, b, sigma, x0, H, t_max, dt, rng, n_paths)
        _, Xm = simulate_rough_cir(a, b, sigma, x0, H, t_max, dt, rng, n_paths)
    return t, Xp - Xm


def calibrate_amplitude(target_rms: float, a: float, sigma: float) -> float:
    """Choose the CIR mean-reversion level b so the signed process has the target
    RMS amplitude.  Var(Lambda_+ - Lambda_-) = 2 * sigma^2 b /(2a) = sigma^2 b/a,
    so b = a * target_rms^2 / sigma^2."""
    return a * target_rms**2 / sigma**2
