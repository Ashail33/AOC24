"""Unit tests for the Hawkes-on-graph simulator and propagator tools.

Run with:  python -m pytest hawkes_graph/tests/ -q
(or:        python hawkes_graph/tests/test_simulator.py  for a plain run)

The key correctness anchors (as flagged in the handoff):
  * branching-cascade size == 1/(1-eta)            [generating-function identity]
  * Ogata stationary bulk intensity == mu/(1-eta)  [analytic Hawkes mean]
  * Ogata and branching engines agree on cascade size  [cross-validation]
  * exact critical propagator exponent == -(d-1)/2     [random-walk return]
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lattice import CausalLattice  # noqa: E402
from simulator import (HawkesParams, analytic_mean_intensity,  # noqa: E402
                       branching_cascade, ogata_thinning)
from propagator import (exact_critical_propagator, hill_estimator,  # noqa: E402
                        measure_propagator_mc, fit_tail)


# ---------------------------------------------------------------- lattice ----
def test_lattice_index_roundtrip():
    lat = CausalLattice(d=4, half_width=5)
    for idx in [0, 1, 17, lat.origin_index, lat.n_sites - 1]:
        coord = lat.index_to_coord(idx)
        assert lat.coord_to_index(coord) == idx


def test_lattice_origin_and_distances():
    lat = CausalLattice(d=3, half_width=4)
    assert lat.graph_distance_from_origin(lat.origin_index) == 0
    # exactly 2*space_dim sites at distance 1
    dist = lat.all_distances()
    assert np.sum(dist == 1) == 2 * lat.space_dim


def test_neighbour_table_self_first_and_symmetric():
    lat = CausalLattice(d=3, half_width=3)
    targets, counts = lat.build_neighbour_table()
    # column 0 is always the self index
    assert np.all(targets[:, 0] == np.arange(lat.n_sites))
    # interior sites have full coordination
    interior = lat.origin_index
    assert counts[interior] == lat.coordination


# -------------------------------------------------------------- cascade ----
@pytest.mark.parametrize("eta", [0.3, 0.6, 0.9])
def test_branching_cascade_size_matches_1_over_1_minus_eta(eta):
    lat = CausalLattice(d=3, half_width=120)
    rng = np.random.default_rng(int(eta * 1000))
    p = HawkesParams(eta=eta, beta=1.0, mu=0.0)
    sizes = [len(branching_cascade(lat, p, t_max=1e12, rng=rng)) for _ in range(4000)]
    mean = np.mean(sizes)
    se = np.std(sizes) / np.sqrt(len(sizes))
    assert abs(mean - 1 / (1 - eta)) < 5 * se + 0.05


def test_branching_cascade_leaks_at_boundary():
    """In a small box, offspring landing out-of-box must LEAK (be dropped), so the
    mean cascade size is strictly below 1/(1-eta) and boundary_hits > 0.  Guards
    against renormalising offspring back inside the box (which would keep boundary
    parents at branching ratio eta)."""
    eta = 0.9
    p = HawkesParams(eta=eta, beta=1.0, mu=0.0)
    small = CausalLattice(d=3, half_width=2)
    rng = np.random.default_rng(1)
    sizes, bh = [], 0
    for _ in range(3000):
        ev = branching_cascade(small, p, t_max=1e12, rng=rng)
        sizes.append(len(ev)); bh += ev.boundary_hits
    assert bh > 0                       # leakage actually occurs
    assert np.mean(sizes) < 1 / (1 - eta) - 0.5   # size suppressed by leakage


def test_engines_agree_on_cascade_size():
    lat = CausalLattice(d=2, half_width=200)
    eta = 0.7
    p = HawkesParams(eta=eta, beta=1.0, mu=0.0)
    rng = np.random.default_rng(1)
    s_branch = [len(branching_cascade(lat, p, t_max=1e9, rng=rng)) for _ in range(4000)]
    rng = np.random.default_rng(2)
    s_ogata = [len(ogata_thinning(lat, p, t_max=1e6, rng=rng,
                                  include_background=False)) for _ in range(4000)]
    mb, mo = np.mean(s_branch), np.mean(s_ogata)
    se = np.sqrt(np.var(s_branch) / 4000 + np.var(s_ogata) / 4000)
    assert abs(mb - mo) < 4 * se


def test_ogata_stationary_bulk_intensity():
    lat = CausalLattice(d=2, half_width=40)
    p = HawkesParams(eta=0.6, beta=1.0, mu=0.5)
    rng = np.random.default_rng(7)
    T = 4000.0
    ev = ogata_thinning(lat, p, t_max=T, rng=rng, include_background=True)
    dist = lat.all_distances()
    bulk = dist <= 10
    bulk_sites = set(np.where(bulk)[0].tolist())
    n_bulk_events = sum(1 for s in ev.sites if int(s) in bulk_sites)
    emp = n_bulk_events / (T * bulk.sum())
    assert abs(emp - analytic_mean_intensity(p)) < 0.03


# ------------------------------------------------------------ propagator ----
@pytest.mark.parametrize("d,expected", [(2, -0.5), (3, -1.0), (4, -1.5)])
def test_exact_critical_propagator_exponent(d, expected):
    taus = np.logspace(0.5, 2.7, 25)
    psi = exact_critical_propagator(d, taus)
    m = taus > 30
    slope = np.polyfit(np.log(taus[m]), np.log(psi[m]), 1)[0]
    assert abs(slope - expected) < 0.05  # == -(d-1)/2, not -d/2


def test_mc_propagator_recovers_spatial_exponent():
    # d=3 -> expect p ~ (d-1)/2 = 1.0 inside the sub-cutoff window
    lat = CausalLattice(d=3, half_width=80)
    p = HawkesParams(eta=0.99, beta=1.0, mu=0.0)
    rng = np.random.default_rng(3)
    res = measure_propagator_mc(lat, p, t_max=200.0, n_realizations=8000,
                                rng=rng, track_distances=(0,), n_batches=40)
    fit = fit_tail(res, D=0, cutoff_fraction=0.3, t_fit_min=3.0,
                   rng=np.random.default_rng(0))
    # should sit near the spatial-diffusion prediction (1.0), not spacetime (1.5)
    assert abs(fit["p_hat"] - fit["pred_spatial_diffusion"]) < 0.3
    assert abs(fit["p_hat"] - fit["pred_spatial_diffusion"]) < \
        abs(fit["p_hat"] - fit["pred_spacetime"])


def test_hill_estimator_on_known_pareto():
    rng = np.random.default_rng(0)
    alpha_true = 1.5
    x = (1 - rng.random(200000)) ** (-1 / alpha_true)  # Pareto(alpha) on [1,inf)
    out = hill_estimator(x, tail_fraction=0.05)
    assert abs(out["alpha_hat"] - alpha_true) < 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
