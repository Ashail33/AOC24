"""Optional CAMB interface (Workstream C).

CAMB is a heavy, CPU-bound dependency and is NOT required for the self-contained
late-time observables in `observables.py`.  This module provides a thin,
gracefully-degrading wrapper: if CAMB is importable it is used to compute the
sound horizon r_drag and (optionally) the CMB power spectra for a LambdaCDM
reference; otherwise `is_available()` returns False and callers fall back to the
fixed-r_d observables.

The full Boltzmann + MCMC confrontation with DESI is deliberately deferred (see
README): per the Workstream-A outcome it should only be invested in once Paper 2's
foundations are settled.
"""

from __future__ import annotations


def is_available() -> bool:
    try:
        import camb  # noqa: F401
        return True
    except Exception:
        return False


def sound_horizon_rdrag(Omega_m=0.31, Omega_b=0.049, H0=67.7, ns=0.965,
                        As=2.1e-9) -> float:
    """Return r_drag [Mpc] from CAMB if available, else a fitting-formula value.

    The fallback uses the Aubourg et al. (2015) approximation, accurate to ~0.5%
    near the Planck cosmology."""
    if is_available():
        import camb
        pars = camb.set_params(H0=H0, ombh2=Omega_b * (H0 / 100) ** 2,
                               omch2=(Omega_m - Omega_b) * (H0 / 100) ** 2,
                               ns=ns, As=As)
        results = camb.get_background(pars)
        return float(results.get_derived_params()["rdrag"])
    # Aubourg+2015 fitting formula
    obh2 = Omega_b * (H0 / 100) ** 2
    och2 = (Omega_m - Omega_b) * (H0 / 100) ** 2
    return 55.154 * pow(2.71828, -72.3 * (obh2 + 0.0006) ** 2) / \
        (obh2 ** 0.12807 * (obh2 + och2) ** 0.25351)


def cmb_reference_cls(Omega_m=0.31, Omega_b=0.049, H0=67.7, lmax=2500):
    """Return (l, TT) for a LambdaCDM reference if CAMB is available, else None."""
    if not is_available():
        return None
    import camb
    pars = camb.set_params(H0=H0, ombh2=Omega_b * (H0 / 100) ** 2,
                           omch2=(Omega_m - Omega_b) * (H0 / 100) ** 2,
                           lmax=lmax)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    tt = powers["total"][:, 0]
    import numpy as np
    return np.arange(tt.shape[0]), tt
