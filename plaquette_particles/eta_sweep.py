"""Main eta-sweep experiment for Workstream B.

For each eta in the sweep, simulate the plaquette Hawkes process over many
realisations, detect cycle / anti-cycle / twisted patterns, compute the
stability observables (r_cycle, tau_cycle, S = tau_cycle * r_cycle), and look for
a peak in S(eta) between eta* = 1 - 1/sqrt(2) ~= 0.293 and eta = 1.

Writes results/summary.json (+ a small npz) for analysis.ipynb.

Usage:
    python plaquette_particles/eta_sweep.py [--quick]
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

from plaquette import window_from_intensity
from hawkes_dynamics import simulate, empirical_intensity
from pattern_detector import detect_chains, detect_twisted, PATTERNS
from stability_metrics import metrics_from_chains, aggregate_realizations

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

ETA_STAR = 1 - 1 / np.sqrt(2)  # ~= 0.2929  (Onaga-Shinomoto bursting threshold)


def run_one_eta(eta, rho, mu, beta, n_events_target, n_real, window_factor, rng):
    """Run n_real realisations at fixed eta; each realisation runs long enough to
    collect ~n_events_target events.  Returns aggregated metrics per pattern."""
    mu = np.asarray(mu, float)
    W = window_from_intensity(eta, rho, mu, beta, factor=window_factor)
    # choose t_max so that expected events ~ n_events_target
    from plaquette import stationary_intensity
    rate = float(np.sum(stationary_intensity(eta, rho, mu)))  # total events/time
    t_max = n_events_target / rate

    per_real = {name: [] for name in list(PATTERNS) + ["twisted"]}
    twisted_counts = []
    intensities = []
    for _ in range(n_real):
        times, marks = simulate(eta, rho, mu, beta, t_max=t_max, rng=rng)
        intensities.append(empirical_intensity(times, t_max))
        for name, patt in PATTERNS.items():
            chains = detect_chains(times, marks, patt, window=W, pattern_name=name)
            per_real[name].append(metrics_from_chains(chains, t_max))
        nt = detect_twisted(times, marks, window=W)
        twisted_counts.append(nt / t_max)  # twisted rate

    out = {"eta": eta, "rho": rho, "window": W, "t_max": t_max,
           "mean_intensity": float(np.mean(intensities)),
           "twisted_rate": float(np.mean(twisted_counts)),
           "twisted_rate_se": float(np.std(twisted_counts) / np.sqrt(n_real))}
    for name in PATTERNS:
        out[name] = aggregate_realizations(per_real[name])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                    help="eta_cross/eta_self ratio(s) to sweep")
    ap.add_argument("--n-events", type=int, default=None,
                    help="target events per realisation (default: 1e5 full / 2e4 quick)")
    ap.add_argument("--n-real", type=int, default=None,
                    help="realisations per eta (default: 50 full / 8 quick)")
    args = ap.parse_args()

    etas = [0.1, 0.2, 0.25, 0.293, 0.3, 0.4, 0.5, 0.7, 0.9]
    mu = [0.2, 0.05, 0.05, 0.05]   # seed activity mainly at A (start of the cycle)
    beta = 1.0
    if args.quick:
        n_events_target, n_real = 20000, 8
    else:
        n_events_target, n_real = 100000, 50
    if args.n_events is not None:
        n_events_target = args.n_events
    if args.n_real is not None:
        n_real = args.n_real

    t0 = time.time()
    by_rho = {}
    for rho in args.rho:
        rows = []
        for eta in etas:
            rng = np.random.default_rng(int(eta * 1000) + int(rho * 100) + 17)
            r = run_one_eta(eta, rho, mu, beta, n_events_target, n_real,
                            window_factor=4.0, rng=rng)
            rows.append(r)
            c = r["cycle"]
            print(f"rho={rho:.1f} eta={eta:5.3f}: S={c['S']:.4f}±{c['S_se']:.4f}  "
                  f"r_cycle={c['r_cycle']:.4f}  tau_cycle={c['tau_cycle']:.3f}  "
                  f"cyc/chain={c['mean_cycles_per_chain']:.2f}  "
                  f"[{time.time()-t0:.0f}s]")

        S = np.array([r["cycle"]["S"] for r in rows])
        Sse = np.array([r["cycle"]["S_se"] for r in rows])
        etas_a = np.array(etas)
        in_window = etas_a > ETA_STAR
        idx = int(np.argmax(np.where(in_window, S, -np.inf)))
        # is the in-window maximum a genuine interior peak (rises then falls)?
        Sw = S[in_window]
        interior_peak = bool(len(Sw) >= 3 and np.argmax(Sw) not in (0, len(Sw) - 1))
        peak = {"eta_peak": float(etas_a[idx]), "S_peak": float(S[idx]),
                "S_peak_se": float(Sse[idx]), "interior_peak": interior_peak}
        by_rho[str(rho)] = {"rows": rows, "S_curve": S.tolist(),
                            "S_se": Sse.tolist(), "peak": peak}
        print(f"  -> rho={rho}: peak {peak}\n")

    summary = {"eta_star": float(ETA_STAR), "etas": etas, "rho_values": args.rho,
               "by_rho": by_rho, "quick": args.quick,
               "elapsed_sec": time.time() - t0}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE in {time.time()-t0:.0f}s -> {RESULTS}/summary.json")


if __name__ == "__main__":
    main()
