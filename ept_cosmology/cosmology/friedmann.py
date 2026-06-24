"""Friedmann background integration with a (stochastic) dark-energy density
(Workstream C).

We integrate the flat-FRW expansion history given a dark-energy density that may
depend on time/redshift through a Lambda(t) trajectory.  The dimensionless
density of dark energy is f_DE(z) = rho_DE(z)/rho_DE(0); for a cosmological
constant f_DE == 1 and w == -1.  A stochastic everpresent-Lambda enters as a
fluctuating f_DE(z).

    H(z)^2 / H0^2 = Omega_m (1+z)^3 + Omega_r (1+z)^4 + Omega_de * f_DE(z)

with Omega_m + Omega_r + Omega_de = 1 (flat).  The effective equation of state is
recovered from the density history via

    w(z) = -1 + (1/3) (1+z) d ln f_DE / dz .
"""

from __future__ import annotations

import numpy as np

C_KM_S = 299792.458  # speed of light [km/s]


class Background:
    """Flat-FRW background with a tabulated dark-energy density history.

    Parameters
    ----------
    Omega_m, Omega_r : float
        Matter and radiation density parameters today.  Omega_de = 1 - Omega_m -
        Omega_r (flat).
    H0 : float
        Hubble constant [km/s/Mpc].
    z_grid : array
        Redshift grid (increasing).
    f_de : array or None
        rho_DE(z)/rho_DE(0) on z_grid.  None => cosmological constant (f_de == 1).
    """

    def __init__(self, Omega_m=0.31, Omega_r=9.0e-5, H0=67.7,
                 z_grid=None, f_de=None):
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_de = 1.0 - Omega_m - Omega_r
        self.H0 = H0
        if z_grid is None:
            z_grid = np.linspace(0, 1100, 4000)
        self.z = np.asarray(z_grid, dtype=float)
        if f_de is None:
            f_de = np.ones_like(self.z)
        self.f_de = np.asarray(f_de, dtype=float)

    # ---- expansion history ----
    def E(self, z) -> np.ndarray:
        """E(z) = H(z)/H0."""
        f = np.interp(z, self.z, self.f_de)
        return np.sqrt(self.Omega_m * (1 + z) ** 3 +
                       self.Omega_r * (1 + z) ** 4 +
                       self.Omega_de * f)

    def H(self, z) -> np.ndarray:
        return self.H0 * self.E(z)

    def w_de(self) -> np.ndarray:
        """Effective DE equation of state w(z) on the internal z grid."""
        lnf = np.log(np.maximum(self.f_de, 1e-300))
        dlnf_dz = np.gradient(lnf, self.z)
        return -1.0 + (1.0 / 3.0) * (1 + self.z) * dlnf_dz

    # ---- distances ----
    def comoving_distance(self, z) -> np.ndarray:
        """Line-of-sight comoving distance D_C(z) [Mpc] via cumulative integration
        of c/H over the internal grid, interpolated to z."""
        integrand = C_KM_S / self.H(self.z)
        Dc_grid = np.concatenate([[0.0], np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(self.z))])
        return np.interp(z, self.z, Dc_grid)

    def luminosity_distance(self, z) -> np.ndarray:
        return (1 + np.asarray(z)) * self.comoving_distance(z)

    def angular_diameter_distance(self, z) -> np.ndarray:
        return self.comoving_distance(z) / (1 + np.asarray(z))

    def distance_modulus(self, z) -> np.ndarray:
        dL = self.luminosity_distance(z)  # Mpc
        return 5.0 * np.log10(np.maximum(dL, 1e-12) * 1e6 / 10.0)


def f_de_from_lambda(t_traj: np.ndarray, lam_traj: np.ndarray,
                     z_grid: np.ndarray, lambda0: float,
                     t_today: float, t_of_z=None) -> np.ndarray:
    """Map a Lambda(t) trajectory to a dark-energy density history f_DE(z).

    A simple, explicit mapping (sufficient for the forward-model scaffold): treat
    Lambda(t) as the DE density up to the constant lambda0 = <Lambda>, with cosmic
    time related to redshift by the supplied `t_of_z` callable (default: a
    monotone linear stand-in so the pipeline is runnable without a self-consistent
    time-redshift solve).  Returns f_DE(z) = Lambda(t(z)) / lambda0.
    """
    if t_of_z is None:
        # crude monotone stand-in: t decreases with z over [0, t_today]
        zmax = z_grid.max()
        t_of_z = lambda z: t_today * (1 - np.asarray(z) / (zmax + 1e-9))
    tz = np.clip(t_of_z(z_grid), t_traj.min(), t_traj.max())
    lam_z = np.interp(tz, t_traj, lam_traj)
    return lam_z / lambda0
