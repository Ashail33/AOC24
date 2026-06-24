"""eta-sweep for the stacked plaquette: does cycle persistence become real?
(Workstream B v2)

Headline observable: the cross-layer propagation LIFT

    lift(eta) = p_prop(eta) - baseline(eta)

where p_prop = P(cycle at layer t+1 | cycle at layer t, same orientation, causal
window) and baseline is the Poisson chance level set by the per-layer cycle rate.
lift > 0 (with error bars excluding 0) means cycles genuinely propagate up the
stack -- particle-like persistence the v1 toy could not exhibit.  We look for a
peak of lift(eta) in (eta*, 1).

Usage: python plaquette_particles_v2/eta_sweep_v2.py [--quick]
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

from stacked_plaquette import (branching_matrix, background_vector,
                               stationary_intensity)
from hawkes_nd import simulate
from persistence import cross_layer_propagation, persistence_length

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
ETA_STAR = 1 - 1 / np.sqrt(2)


def run_one(eta, T, rho, rho_time, beta, n_events, n_real, rng):
    Phi = branching_matrix(eta, T, rho, rho_time)
    mu = background_vector(T)
    m = stationary_intensity(eta, T, rho, rho_time, mu)
    rate = float(np.sum(m))
    t_max = n_events / rate
    W = 4.0 / float(np.mean(m))         # per-vertex inverse-intensity window

    lifts, pprops, bases, plens = [], [], [], []
    for _ in range(n_real):
        times, marks = simulate(Phi, mu, beta, t_max, rng)
        r = cross_layer_propagation(times, marks, T, window=W, prop_window=W)
        pprops.append(r["p_prop"]); bases.append(r["baseline"]); lifts.append(r["lift"])
        plens.append(persistence_length(times, marks, T, window=W, prop_window=W))
    lifts = np.array(lifts)
    return {"eta": eta, "T": T, "rho": rho, "rho_time": rho_time, "window": W,
            "p_prop": float(np.mean(pprops)),
            "baseline": float(np.mean(bases)),
            "lift": float(np.mean(lifts)),
            "lift_se": float(np.std(lifts) / np.sqrt(max(len(lifts), 1))),
            "persistence_length": float(np.mean(plens)),
            "n_real": n_real}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--rho-time", type=float, default=1.5)
    args = ap.parse_args()

    etas = [0.1, 0.2, 0.293, 0.4, 0.5, 0.6, 0.7, 0.9]
    beta = 1.0
    if args.quick:
        n_events, n_real = 30000, 6
    else:
        n_events, n_real = 100000, 30

    t0 = time.time()
    rows = []
    for eta in etas:
        rng = np.random.default_rng(int(eta * 1000) + 31)
        r = run_one(eta, args.T, args.rho, args.rho_time, beta, n_events, n_real, rng)
        rows.append(r)
        sig = "***" if r["lift"] > 2 * r["lift_se"] else ""
        print(f"eta={eta:5.3f}: lift={r['lift']:+.4f}±{r['lift_se']:.4f} {sig:3} "
              f"(p_prop={r['p_prop']:.3f} base={r['baseline']:.3f}) "
              f"L_persist={r['persistence_length']:.2f}  [{time.time()-t0:.0f}s]")

    lift = np.array([r["lift"] for r in rows])
    lse = np.array([r["lift_se"] for r in rows])
    etas_a = np.array(etas)
    inwin = etas_a > ETA_STAR
    idx = int(np.argmax(np.where(inwin, lift, -np.inf)))
    Lw = lift[inwin]
    interior = bool(len(Lw) >= 3 and np.argmax(Lw) not in (0, len(Lw) - 1))
    significant = bool(lift[idx] > 2 * lse[idx])
    peak = {"eta_peak": float(etas_a[idx]), "lift_peak": float(lift[idx]),
            "lift_peak_se": float(lse[idx]), "interior_peak": interior,
            "significant_lift": significant}

    summary = {"eta_star": float(ETA_STAR), "etas": etas, "T": args.T,
               "rho": args.rho, "rho_time": args.rho_time, "rows": rows,
               "peak": peak, "quick": args.quick, "elapsed_sec": time.time() - t0}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPeak: {peak}")
    print(f"DONE in {time.time()-t0:.0f}s -> {RESULTS}/summary.json")


if __name__ == "__main__":
    main()
