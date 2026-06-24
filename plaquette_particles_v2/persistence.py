"""Genuine cross-layer cycle persistence on the stacked plaquette (Workstream B v2).

The v1 metric S = tau_cycle * r_cycle was shown to be a rate x lifetime artifact
(cycles never recurred -- cycles-per-chain ~ 1).  Here we measure the observable
that actually distinguishes a particle from a coincidence:

  cross-layer propagation probability
      p_prop = P(a cycle is detected at layer t+1 in the same orientation,
                 within a causal window, GIVEN a cycle was detected at layer t).

A genuine persistent excitation propagates up the stack (A_t->B_t->C_t->D_t then
A_{t+1}->B_{t+1}->C_{t+1}->D_{t+1} ...), so p_prop sits well above the chance
level set by the per-layer cycle rate.  An artifact does not propagate: p_prop
collapses to the coincidence baseline.

We also report the persistence length L_persist = mean number of consecutive
layers a cycle survives once started.
"""

from __future__ import annotations

import numpy as np

CYCLE = [0, 1, 2, 3]        # A B C D


def detect_layer_cycle_times(times, marks, layer, window):
    """Return the completion times of A->B->C->D traversals restricted to the four
    nodes of a given layer.  Marks are global flat indices 4*layer + v."""
    base = 4 * layer
    # restrict to this layer's events, relabel to 0..3
    sel = (marks >= base) & (marks < base + 4)
    lt = times[sel]
    lv = marks[sel] - base
    completions = []
    pos = 0
    last_t = None
    for t, v in zip(lt, lv):
        if pos > 0 and last_t is not None and t - last_t > window:
            pos = 1 if v == CYCLE[0] else 0
            last_t = t if pos == 1 else None
            continue
        if v == CYCLE[pos]:
            pos += 1
            last_t = t
            if pos == 4:
                completions.append(t)
                pos = 0
                last_t = t  # allow immediate restart within window
        else:
            pos = 1 if v == CYCLE[0] else 0
            last_t = t if pos == 1 else None
    return np.array(completions)


def cross_layer_propagation(times, marks, T, window, prop_window):
    """Measure p_prop and the per-layer cycle rate baseline.

    For every cycle completing at layer t (time tc), it 'propagates' if a cycle
    completes at layer t+1 within (tc, tc + prop_window].  Causal: layer t+1 is
    strictly in the future of layer t via the forward inter-layer arrows.
    """
    comp = {t: detect_layer_cycle_times(times, marks, t, window) for t in range(T)}
    n_cycles = sum(len(c) for c in comp.values())
    propagated = 0
    considered = 0
    for t in range(T - 1):
        nxt = comp[t + 1]
        for tc in comp[t]:
            considered += 1
            # any layer-(t+1) cycle in (tc, tc+prop_window]?
            if np.any((nxt > tc) & (nxt <= tc + prop_window)):
                propagated += 1
    p_prop = propagated / considered if considered else 0.0

    # chance baseline: probability a random window of length prop_window at a
    # random layer contains a cycle, given the per-layer cycle rate.
    total_time = times[-1] - times[0] if len(times) > 1 else 1.0
    per_layer_rate = np.mean([len(comp[t]) / total_time for t in range(T)])
    baseline = 1.0 - np.exp(-per_layer_rate * prop_window)  # Poisson chance

    return {"p_prop": p_prop, "baseline": baseline,
            "lift": p_prop - baseline,
            "n_cycles": n_cycles, "considered": considered,
            "per_layer_rate": per_layer_rate}


def persistence_length(times, marks, T, window, prop_window, max_chain=None):
    """Mean number of consecutive layers a cycle survives: starting from a cycle
    at layer t, follow propagation t->t+1->... and count the run length."""
    comp = {t: detect_layer_cycle_times(times, marks, t, window) for t in range(T)}
    if max_chain is None:
        max_chain = T
    runs = []
    for t0 in range(T - 1):
        for tc in comp[t0]:
            length = 1
            cur_t = t0
            cur_time = tc
            while cur_t + 1 < T:
                nxt = comp[cur_t + 1]
                hit = nxt[(nxt > cur_time) & (nxt <= cur_time + prop_window)]
                if len(hit) == 0:
                    break
                length += 1
                cur_time = hit[0]
                cur_t += 1
                if length >= max_chain:
                    break
            runs.append(length)
    return float(np.mean(runs)) if runs else 0.0
