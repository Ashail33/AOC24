"""Jaisson-Rosenbaum scaling limit of the nearly-unstable lattice Hawkes process
(Workstream A, task 3).

Jaisson & Rosenbaum (2015, 2016): a sequence of nearly-unstable Hawkes processes
(eta_n -> 1) with kernel of integral eta_n, suitably rescaled in time and
amplitude, converges to a CIR-type process.  The roughness of the limit is set
by the *tail of the kernel*:

  * light-tailed / integrable second moment (exponential kernel)  -> CIR,  H = 1/2
  * kernel tail  phi(tau) ~ tau^{-(1+alpha)}, alpha in (0,1)      -> rough CIR,
    fractional kernel of Hurst  H = alpha - 1/2.

For the lattice model the *effective* (spatially-aggregated) memory is the
origin-return propagator measured in `propagator.py`, with tail exponent
(d-1)/2.  Writing 1 + alpha = (d-1)/2  =>  alpha = (d-3)/2, the predicted limit
Hurst is

      H_kernel_consistent(d) = alpha - 1/2 = (d-4)/2 .

The handoff instead assumed the kernel tail d/2, giving H = (d-3)/2.  Both are
reported; the half-unit gap is the same one found in the propagator exponent.

This module:
  * simulates the stationary aggregate intensity at near-critical eta,
  * estimates the Hurst exponent of the integrated-intensity path (DFA and
    aggregated-variance estimators),
  * exposes the prediction maps for comparison.

Direct Hurst estimation from finite near-critical simulations is intrinsically
noisy; treat these as corroboration of the (clean) propagator exponent, not as
an independent high-precision measurement.
"""

from __future__ import annotations

import numpy as np

from simulator import HawkesParams, ogata_thinning


def predicted_hurst(d: int, which: str = "kernel_consistent") -> float:
    """Predicted limit Hurst exponent.
    which='kernel_consistent' -> (d-4)/2 (from the measured (d-1)/2 kernel tail);
    which='handoff'           -> (d-3)/2 (from the conjectured d/2 kernel tail)."""
    if which == "kernel_consistent":
        return (d - 4) / 2
    if which == "handoff":
        return (d - 3) / 2
    raise ValueError("which must be 'kernel_consistent' or 'handoff'")


def aggregate_intensity_path(lattice, params: HawkesParams, t_max: float,
                             rng: np.random.Generator, dt: float) -> np.ndarray:
    """Run a stationary (mu>0) simulation and return the binned global event
    counts N_k = #events in [k dt, (k+1) dt) summed over all sites -- a discrete
    sample of the integrated aggregate intensity increments."""
    ev = ogata_thinning(lattice, params, t_max=t_max, rng=rng,
                        include_background=True)
    n_bins = int(np.floor(t_max / dt))
    edges = np.arange(n_bins + 1) * dt
    counts, _ = np.histogram(ev.times, bins=edges)
    return counts.astype(float)


def hurst_dfa(x: np.ndarray, min_box: int = 8, n_scales: int = 16) -> dict:
    """Detrended Fluctuation Analysis on the cumulative profile of x.
    Returns the DFA exponent alpha_dfa (= H for a stationary fGn-like signal)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    y = np.cumsum(x)
    n = len(y)
    max_box = n // 4
    scales = np.unique(np.round(np.logspace(np.log10(min_box),
                                            np.log10(max_box), n_scales)).astype(int))
    scales = scales[scales >= min_box]
    F = []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        resid = 0.0
        cnt = 0
        t = np.arange(s)
        for seg in range(n_seg):
            ys = y[seg * s:(seg + 1) * s]
            coef = np.polyfit(t, ys, 1)
            fit = np.polyval(coef, t)
            resid += np.sum((ys - fit) ** 2)
            cnt += s
        F.append(np.sqrt(resid / cnt))
    F = np.array(F)
    scales = scales[:len(F)]
    good = F > 0
    slope = np.polyfit(np.log(scales[good]), np.log(F[good]), 1)[0]
    return {"H_dfa": float(slope), "scales": scales, "F": F}


def hurst_aggregated_variance(x: np.ndarray, n_scales: int = 16) -> dict:
    """Aggregated-variance estimator: Var of block means ~ m^{2H-2}.  Returns
    H_aggvar."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    ms = np.unique(np.round(np.logspace(0.3, np.log10(n // 8), n_scales)).astype(int))
    ms = ms[ms >= 2]
    vfunc = []
    for m in ms:
        n_blocks = n // m
        block_means = x[:n_blocks * m].reshape(n_blocks, m).mean(axis=1)
        vfunc.append(np.var(block_means))
    vfunc = np.array(vfunc)
    good = vfunc > 0
    slope = np.polyfit(np.log(ms[good]), np.log(vfunc[good]), 1)[0]
    H = 1 + slope / 2  # slope = 2H - 2
    return {"H_aggvar": float(H), "ms": ms, "vfunc": vfunc}


def measure_hurst_sequence(lattice, etas, beta, mu, t_max, dt, rng,
                           n_repeats: int = 4) -> list[dict]:
    """For each eta in the eta->1 sequence, simulate the aggregate intensity path
    and estimate Hurst by DFA and aggregated variance, averaged over repeats."""
    out = []
    for eta in etas:
        params = HawkesParams(eta=eta, beta=beta, mu=mu)
        Hd, Ha = [], []
        for _ in range(n_repeats):
            path = aggregate_intensity_path(lattice, params, t_max, rng, dt)
            Hd.append(hurst_dfa(path)["H_dfa"])
            Ha.append(hurst_aggregated_variance(path)["H_aggvar"])
        out.append({
            "eta": eta,
            "H_dfa": float(np.mean(Hd)), "H_dfa_se": float(np.std(Hd) / np.sqrt(len(Hd))),
            "H_aggvar": float(np.mean(Ha)), "H_aggvar_se": float(np.std(Ha) / np.sqrt(len(Ha))),
        })
    return out
