"""Cosmological observables from a Friedmann background (Workstream C).

Self-contained (no CAMB/CLASS required) implementations of the headline late-time
observables used in a DESI-style confrontation:

  * H(z)                         -- expansion rate
  * w(z)                         -- DE equation of state
  * distance modulus mu(z)       -- SN Ia
  * BAO observables D_V/r_d, D_M/r_d, D_H/r_d
  * CMB acoustic scale l_A and shift parameter R (compressed CMB)

These use a fixed sound-horizon r_d (a calibration input) rather than an
early-universe Boltzmann solve; the optional `camb_interface` can replace r_d and
the full C_l with a CAMB computation when that package is available.
"""

from __future__ import annotations

import numpy as np

from friedmann import Background, C_KM_S


def hubble(bg: Background, z) -> np.ndarray:
    return bg.H(z)


def equation_of_state(bg: Background, z) -> np.ndarray:
    return np.interp(z, bg.z, bg.w_de())


def distance_modulus(bg: Background, z) -> np.ndarray:
    return bg.distance_modulus(z)


def bao_observables(bg: Background, z, r_d: float = 147.0) -> dict:
    """Standard BAO ratios.  D_H = c/H, D_M = comoving distance, D_V the
    isotropic average; all divided by the sound horizon r_d [Mpc]."""
    z = np.asarray(z, dtype=float)
    D_M = bg.comoving_distance(z)
    D_H = C_KM_S / bg.H(z)
    D_V = (z * D_M ** 2 * D_H) ** (1.0 / 3.0)
    return {"z": z, "DV_over_rd": D_V / r_d, "DM_over_rd": D_M / r_d,
            "DH_over_rd": D_H / r_d}


def cmb_acoustic_scale(bg: Background, z_star: float = 1089.0,
                       r_s_star: float = 144.4) -> dict:
    """Compressed-CMB observables: the acoustic scale l_A = pi D_M(z*)/r_s(z*)
    and the shift parameter R = sqrt(Omega_m) H0 D_M(z*)/c."""
    D_M_star = float(bg.comoving_distance(z_star))
    l_A = np.pi * D_M_star / r_s_star
    R = np.sqrt(bg.Omega_m) * bg.H0 * D_M_star / C_KM_S
    return {"z_star": z_star, "l_A": l_A, "R": R, "D_M_star": D_M_star}


def all_observables(bg: Background, z_sn=None, z_bao=None, r_d=147.0) -> dict:
    """Convenience bundle of the late-time observables on default grids."""
    if z_sn is None:
        z_sn = np.linspace(0.01, 2.0, 50)
    if z_bao is None:
        z_bao = np.array([0.38, 0.51, 0.70, 0.85, 1.49, 2.33])
    return {
        "z_sn": z_sn,
        "mu": distance_modulus(bg, z_sn),
        "Hz": hubble(bg, z_sn),
        "wz": equation_of_state(bg, z_sn),
        "bao": bao_observables(bg, z_bao, r_d=r_d),
        "cmb": cmb_acoustic_scale(bg),
    }
