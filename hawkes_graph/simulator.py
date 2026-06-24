"""Multivariate Hawkes simulation on the causal lattice (Workstream A).

Two independent engines are provided so they can be cross-validated against each
other and against the analytic mean intensity:

1. `ogata_thinning` -- the general continuous-time Ogata (1981) thinning
   algorithm.  Works for any background rate mu and is the reference engine
   requested in the handoff.  Exponential kernels let us track the per-site
   excitation cheaply via the standard "memory variable" recursion, so the total
   intensity is maintained in O(active sites) per event.

2. `branching_cascade` -- the *exact* Poisson cluster representation, valid when
   there is no background (a single seed event) and the process is sub-critical
   (eta < 1).  Each event independently spawns Poisson(eta) offspring; each
   offspring's delay is Exponential(beta) and its location is the parent site or
   a uniformly chosen nearest neighbour.  This is exact (not an approximation)
   and is the workhorse for the propagator measurement because it is fast and
   has no thinning overhead.

Kernel convention
-----------------
An event at site y at time s contributes to the intensity at site x at time t>s
    phi_{yx}(t-s) = eta * w_{yx} * beta * exp(-beta (t-s)),
with w_{yx} = 1/coordination for x in {y} U neighbours(y) and 0 otherwise.
Then  integral_0^inf sum_x phi_{yx}(u) du = eta  ==> branching ratio = eta, and a
single-site (or homogeneous) process has stationary mean intensity mu/(1-eta).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HawkesParams:
    eta: float            # branching ratio (sum of kernel norms), criticality knob
    beta: float = 1.0     # exponential decay rate of the elementary kernel
    mu: float = 0.0       # background (immigrant) rate per site; 0 for cascade runs

    def __post_init__(self):
        if not (0.0 <= self.eta < 1.0):
            raise ValueError("eta must be in [0, 1) for a sub-critical process")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.mu < 0:
            raise ValueError("mu must be >= 0")


@dataclass
class Events:
    """Container for a simulated event list."""
    times: np.ndarray              # (N,) event times, sorted
    sites: np.ndarray              # (N,) flat lattice indices
    distances: np.ndarray          # (N,) graph distance from origin
    boundary_hits: int = 0         # offspring that would have left the box (dropped)
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.times)


# ----------------------------------------------------------------------------
# Exact branching cascade (single seed, sub-critical, mu = 0)
# ----------------------------------------------------------------------------
def branching_cascade(lattice, params: HawkesParams, t_max: float,
                      rng: np.random.Generator, seed_site: int | None = None,
                      max_events: int = 5_000_000) -> Events:
    """Simulate the exact Poisson-cluster cascade from a single seed event at
    time 0.  Returns every event with time < t_max.

    This is exact for the Hawkes mean *and* full distribution (it *is* the Hawkes
    process restricted to a single immigrant), so it is the preferred propagator
    engine.  Generation-by-generation breadth-first expansion.
    """
    targets_full = lattice.build_full_neighbour_table()
    if seed_site is None:
        seed_site = lattice.origin_index

    times_out = [0.0]
    sites_out = [seed_site]
    boundary_hits = 0

    # current generation
    cur_t = np.array([0.0])
    cur_s = np.array([seed_site], dtype=np.int64)

    total = 1
    while len(cur_t) > 0:
        # number of offspring per parent ~ Poisson(eta)
        n_off = rng.poisson(params.eta, size=len(cur_t))
        n_total = int(n_off.sum())
        if n_total == 0:
            break
        parent_idx = np.repeat(np.arange(len(cur_t)), n_off)
        # offspring delays ~ Exp(beta); offspring times = parent time + delay
        delays = rng.exponential(1.0 / params.beta, size=n_total)
        off_t = cur_t[parent_idx] + delays
        # offspring location: pick uniformly over the FULL coordination (self +
        # all 2*space_dim neighbours); offspring whose target is out of the box
        # (-1) leak out and are dropped, so a boundary parent correctly keeps
        # branching ratio eta * (in-box fraction) rather than renormalising.
        parent_sites = cur_s[parent_idx]
        pick = (rng.random(n_total) * lattice.coordination).astype(np.int64)
        off_s = targets_full[parent_sites, pick]

        in_window = off_t < t_max
        leaked = off_s < 0
        boundary_hits += int(np.sum(leaked & in_window))  # true out-of-box leakage
        keep = in_window & ~leaked

        off_t = off_t[keep]
        off_s = off_s[keep]

        times_out.extend(off_t.tolist())
        sites_out.extend(off_s.tolist())
        total += len(off_t)
        if total > max_events:
            raise RuntimeError(f"cascade exceeded max_events={max_events}; "
                               f"eta={params.eta} too close to 1 for t_max={t_max}")
        cur_t = off_t
        cur_s = off_s

    times = np.asarray(times_out)
    sites = np.asarray(sites_out, dtype=np.int64)
    order = np.argsort(times)
    times = times[order]
    sites = sites[order]
    distances = _distances_for(lattice, sites)
    return Events(times=times, sites=sites, distances=distances,
                  boundary_hits=boundary_hits,
                  meta={"engine": "branching", "eta": params.eta})


# ----------------------------------------------------------------------------
# Ogata thinning (general; reference engine)
# ----------------------------------------------------------------------------
def ogata_thinning(lattice, params: HawkesParams, t_max: float,
                   rng: np.random.Generator, seed_site: int | None = None,
                   include_background: bool = True,
                   max_events: int = 5_000_000) -> Events:
    """General continuous-time Ogata thinning simulation on the lattice.

    Exponential kernels admit an exact O(active-sites) recursion: keep a memory
    variable g_x = sum over past events influencing x of eta*w*beta*exp(...),
    which decays as exp(-beta dt) between events and gets a kick on each event.
    The total intensity is Lambda(t) = sum_x (mu + g_x(t)) and is monotone
    decreasing between events, so the Ogata upper bound is simply the value right
    after the last event -- thinning is exact and rejection-free except at the
    instant of each accepted event.
    """
    targets, counts = lattice.build_neighbour_table()
    n_sites = lattice.n_sites
    if seed_site is None:
        seed_site = lattice.origin_index

    mu = params.mu if include_background else 0.0
    beta = params.beta
    # kick added to g_x for each event at a parent site, per target = eta*beta*w
    kick = params.eta * beta * lattice.weight

    g = np.zeros(n_sites)             # excitation memory per site
    mu_total = mu * n_sites

    times_out = []
    sites_out = []
    boundary_hits = 0

    t = 0.0
    # seed: inject one event at t=0 at the seed site (its descendants follow)
    if not include_background or mu == 0.0:
        times_out.append(0.0)
        sites_out.append(seed_site)
        _apply_kick(g, seed_site, targets, counts, kick, lattice.coordination,
                    boundary_counter := [0])
        boundary_hits += boundary_counter[0]

    sum_g = g.sum()
    while True:
        lam_bar = mu_total + sum_g  # upper bound = intensity right after last event
        if lam_bar <= 1e-12:
            # cascade exhausted (mu=0 and excitation memory negligible)
            break
        # next candidate time
        w = rng.exponential(1.0 / lam_bar)
        t = t + w
        if t >= t_max:
            break
        # decay memory to time t
        decay = np.exp(-beta * w)
        g *= decay
        sum_g *= decay
        lam_t = mu_total + sum_g
        # accept with prob lam_t / lam_bar
        if rng.random() * lam_bar <= lam_t:
            # choose which site fired, proportional to (mu + g_x)
            site = _sample_site(g, mu, sum_g, mu_total, rng, n_sites)
            times_out.append(t)
            sites_out.append(site)
            bc = [0]
            _apply_kick(g, site, targets, counts, kick, lattice.coordination, bc)
            boundary_hits += bc[0]
            sum_g = g.sum()
            if len(times_out) > max_events:
                raise RuntimeError("ogata exceeded max_events")

    times = np.asarray(times_out)
    sites = np.asarray(sites_out, dtype=np.int64)
    order = np.argsort(times)
    times = times[order]
    sites = sites[order]
    distances = _distances_for(lattice, sites)
    return Events(times=times, sites=sites, distances=distances,
                  boundary_hits=boundary_hits,
                  meta={"engine": "ogata", "eta": params.eta, "mu": mu})


def _apply_kick(g, site, targets, counts, kick, coordination, boundary_counter):
    c = counts[site]
    tg = targets[site, :c]
    g[tg] += kick
    if c < coordination:
        boundary_counter[0] += 1


def _sample_site(g, mu, sum_g, mu_total, rng, n_sites):
    """Sample the firing site with probability proportional to (mu + g_x)."""
    total = mu_total + sum_g
    u = rng.random() * total
    if mu > 0.0:
        # background contributes mu to every site; decide background vs excitation
        if u < mu_total:
            return int(u / mu)  # uniform background site
        u -= mu_total
    # excitation: sample proportional to g
    csum = np.cumsum(g)
    return int(np.searchsorted(csum, u, side="right"))


def _distances_for(lattice, sites: np.ndarray) -> np.ndarray:
    table = lattice.all_distances()
    return table[sites]


def analytic_mean_intensity(params: HawkesParams) -> float:
    """Stationary mean intensity per site for a homogeneous process: mu/(1-eta)."""
    return params.mu / (1.0 - params.eta)
