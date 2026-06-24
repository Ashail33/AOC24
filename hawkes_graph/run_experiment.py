"""Main experiment driver for Workstream A.

Runs the full sweep over d in {2,3,4} and eta in {0.5,0.9,0.95,0.99}, measures the
effective propagator psi(tau,D), fits the large-tau tail exponent (weighted
regression with block bootstrap, plus a Hill estimate on pooled event times),
computes the exact critical propagator for reference, and runs the
Jaisson-Rosenbaum Hurst sweep.  Results are written to results/ as JSON + npz so
analysis.ipynb can render them without re-simulating.

Usage:
    python hawkes_graph/run_experiment.py [--quick]

--quick uses fewer realisations for a fast smoke run; the default targets the
statistical power the handoff asks for (>=1e4-1e5 realisations) but is tuned to
finish in well under an hour on a handful of cores.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)

from lattice import CausalLattice
from simulator import HawkesParams, branching_cascade
from propagator import (exact_critical_propagator, fit_tail,
                        powerlaw_mle_truncated, measure_propagator_mc)
from scaling_limits import measure_hurst_sequence, predicted_hurst

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Box half-width per d: big enough that an eta<=0.99 cascade does not reach the
# boundary within the measurement window (checked via boundary_hits).
HALF_WIDTH = {2: 250, 3: 70, 4: 28}
# Measurement window per d (a few times the largest sub-cutoff scale we fit).
T_MAX = {2: 250.0, 3: 250.0, 4: 250.0}


def run_propagator_sweep(d_values, eta_values, n_real, n_batches, seed=0):
    rows = []
    curves = {}
    for d in d_values:
        lat = CausalLattice(d=d, half_width=HALF_WIDTH[d])
        for eta in eta_values:
            t0 = time.time()
            rng = np.random.default_rng(seed + 1000 * d + int(eta * 100))
            params = HawkesParams(eta=eta, beta=1.0, mu=0.0)
            res = measure_propagator_mc(lat, params, t_max=T_MAX[d],
                                        n_realizations=n_real, rng=rng,
                                        track_distances=(0, 1, 2),
                                        n_time_bins=48, n_batches=n_batches)
            # leakage check: re-simulate a small batch to count boundary hits
            bh = 0
            rng2 = np.random.default_rng(seed + 7)
            for _ in range(200):
                bh += branching_cascade(lat, params, t_max=T_MAX[d], rng=rng2).boundary_hits
            try:
                fit = fit_tail(res, D=0, cutoff_fraction=0.3, t_fit_min=3.0,
                               rng=np.random.default_rng(99))
            except ValueError as e:
                fit = {"p_hat": float("nan"), "ci95": (float("nan"), float("nan")),
                       "fit_window": (float("nan"), float("nan")),
                       "pred_spatial_diffusion": (d - 1) / 2, "pred_spacetime": d / 2,
                       "error": str(e)}
            row = {"d": d, "eta": eta, "n_real": res.n_realizations,
                   "boundary_hits_per200": bh, **{k: fit[k] for k in
                   ("p_hat", "ci95", "fit_window", "pred_spatial_diffusion",
                    "pred_spacetime") if k in fit}}
            rows.append(row)
            # store the psi(tau,0) curve
            curves[f"d{d}_eta{eta}_centers"] = res.centers
            curves[f"d{d}_eta{eta}_psi0"] = res.psi(0)
            print(f"[prop] d={d} eta={eta}: p_hat={row.get('p_hat'):.3f} "
                  f"CI={tuple(round(x,3) for x in row['ci95'])} "
                  f"(pred (d-1)/2={row['pred_spatial_diffusion']}, d/2={row['pred_spacetime']}) "
                  f"bh/200={bh}  [{time.time()-t0:.0f}s]")
    return rows, curves


def run_mle(d_values, eta=0.99, n_real=20000, seed=0):
    """Independent estimate of the propagator exponent via a truncated power-law
    MLE on the pooled origin event times inside the sub-cutoff window."""
    out = []
    for d in d_values:
        lat = CausalLattice(d=d, half_width=HALF_WIDTH[d])
        rng = np.random.default_rng(seed + d)
        params = HawkesParams(eta=eta, beta=1.0, mu=0.0)
        oi = lat.origin_index
        a, b = 3.0, 0.3 / (1 - eta)  # same window as the regression fit
        pooled = []
        for _ in range(n_real):
            ev = branching_cascade(lat, params, t_max=b, rng=rng)
            m = ev.sites == oi
            pooled.extend(ev.times[m].tolist())
        pooled = np.array(pooled)
        try:
            h = powerlaw_mle_truncated(pooled, a=a, b=b)
            out.append({"d": d, "eta": eta, "p_hat_mle": h["p_hat"],
                        "p_se": h["p_se"], "n_samples": h["n"],
                        "window": (a, b),
                        "pred_spatial_diffusion": (d - 1) / 2, "pred_spacetime": d / 2})
            print(f"[mle] d={d}: p_hat={h['p_hat']:.3f}+-{h['p_se']:.3f} "
                  f"(n={h['n']}, window=[{a:.1f},{b:.1f}])  "
                  f"(d-1)/2={(d-1)/2}, d/2={d/2}")
        except ValueError as e:
            out.append({"d": d, "error": str(e)})
            print(f"[mle] d={d}: {e}")
    return out


def run_exact(d_values):
    taus = np.logspace(0.3, 3.0, 60)
    curves = {"taus": taus}
    summary = []
    for d in d_values:
        psi = exact_critical_propagator(d, taus)
        curves[f"d{d}"] = psi
        m = taus > 30
        slope = np.polyfit(np.log(taus[m]), np.log(psi[m]), 1)[0]
        summary.append({"d": d, "exact_exponent": float(-slope),
                        "pred_spatial_diffusion": (d - 1) / 2,
                        "pred_spacetime": d / 2})
        print(f"[exact] d={d}: exponent={-slope:.4f} "
              f"((d-1)/2={(d-1)/2}, d/2={d/2})")
    return summary, curves


# Small boxes keep the (secondary, noisy) Hurst sweep fast: the lattice Ogata
# engine is O(n_sites) per event, so the aggregate-intensity sims must stay small.
HURST_HALF_WIDTH = {2: 30, 3: 6, 4: 3}
HURST_TMAX = {2: 1500.0, 3: 1500.0, 4: 1000.0}


def run_scaling(d_values, etas, seed=0):
    out = []
    for d in d_values:
        lat = CausalLattice(d=d, half_width=HURST_HALF_WIDTH[d])
        rng = np.random.default_rng(seed + 50 + d)
        seq = measure_hurst_sequence(lat, etas=etas, beta=1.0, mu=0.02,
                                     t_max=HURST_TMAX[d], dt=0.5, rng=rng, n_repeats=3)
        for s in seq:
            s["d"] = d
            s["H_pred_kernel_consistent"] = predicted_hurst(d, "kernel_consistent")
            s["H_pred_handoff"] = predicted_hurst(d, "handoff")
        out.extend(seq)
        print(f"[hurst] d={d}: " + ", ".join(
            f"eta={s['eta']} H_dfa={s['H_dfa']:.2f}" for s in seq) +
            f"  (pred kernel-consistent={predicted_hurst(d,'kernel_consistent')})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    d_values = [2, 3, 4]
    eta_values = [0.5, 0.9, 0.95, 0.99]
    if args.quick:
        n_real, n_batches, hill_real = 4000, 40, 4000
        scaling_t = [2, 3, 4]
    else:
        n_real, n_batches, hill_real = 40000, 100, 30000
        scaling_t = [2, 3, 4]

    t0 = time.time()
    print("=== exact critical propagator ===")
    exact_summary, exact_curves = run_exact(d_values)
    print("=== propagator MC sweep ===")
    prop_rows, prop_curves = run_propagator_sweep(d_values, eta_values, n_real, n_batches)
    print("=== truncated power-law MLE (eta=0.99) ===")
    mle_rows = run_mle(d_values, eta=0.99, n_real=hill_real)
    print("=== Jaisson-Rosenbaum Hurst sweep ===")
    hurst_rows = run_scaling(scaling_t, etas=[0.9, 0.95, 0.99])

    summary = {"exact": exact_summary, "propagator_fits": prop_rows,
               "mle": mle_rows, "hurst": hurst_rows,
               "elapsed_sec": time.time() - t0, "quick": args.quick}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else o)
    np.savez_compressed(os.path.join(RESULTS, "propagator_curves.npz"), **prop_curves)
    np.savez_compressed(os.path.join(RESULTS, "exact_curves.npz"), **exact_curves)
    print(f"\nDONE in {time.time()-t0:.0f}s.  Results -> {RESULTS}")


if __name__ == "__main__":
    main()
