"""Stability observables for plaquette cycle patterns (Workstream B).

From the detected chains we form, per (eta, pattern):
  * r_cycle   : cycle creation rate   = (total completed cycles) / T
  * tau_cycle : cycle lifetime        = mean chain duration over chains that
                completed >= 1 cycle (how long a cycle, once formed, persists with
                continued in-order events)
  * S(eta)    : stability ratio        = tau_cycle * r_cycle   (dimensionless
                'particleness': long-lived AND frequently created -> large S)

S is the quantity whose eta-dependence the refined particle conjecture predicts
to peak between eta* ~= 0.293 and eta = 1.
"""

from __future__ import annotations

import numpy as np

from pattern_detector import Chain


def metrics_from_chains(chains: list[Chain], t_max: float) -> dict:
    total_cycles = int(sum(c.n_cycles for c in chains))
    r_cycle = total_cycles / t_max
    durations = np.array([c.duration for c in chains if c.n_cycles >= 1])
    # chains with exactly one cycle have duration spanning that single traversal;
    # multi-cycle chains span the persistent run.
    tau_cycle = float(np.mean(durations)) if len(durations) else 0.0
    S = tau_cycle * r_cycle
    return {
        "n_chains": len(chains),
        "total_cycles": total_cycles,
        "r_cycle": r_cycle,
        "tau_cycle": tau_cycle,
        "S": S,
        "mean_cycles_per_chain": (total_cycles / len(chains)) if chains else 0.0,
    }


def aggregate_realizations(per_real: list[dict]) -> dict:
    """Mean +/- standard error of each observable across realisations."""
    keys = ["r_cycle", "tau_cycle", "S", "total_cycles", "mean_cycles_per_chain"]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_real], dtype=float)
        out[k] = float(np.mean(vals))
        out[k + "_se"] = float(np.std(vals) / np.sqrt(max(len(vals), 1)))
    return out
