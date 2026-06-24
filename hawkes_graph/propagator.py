"""Measure the effective propagator psi(tau, D) and fit its large-tau tail.

The *effective propagator* is the expected intensity at a vertex at graph
distance D from a single seed event at the origin, as a function of elapsed
time tau:
    psi(tau, D) = E[ lambda_x(tau) | one seed at origin, mu = 0 ],   |x| = D.
Because E[dN_x] = E[lambda_x] dt, we estimate psi by histogramming cascade
events by (time, distance) and normalising by (#realisations * bin_width *
shell_size(D)).

Two complementary tools:

* `exact_critical_propagator` -- the *deterministic* critical (eta -> 1)
  propagator at the origin, m(tau,0) = sum_n p_n(0->0) Erlang_n(tau), computed
  from the lazy-random-walk return probabilities via FFT.  No Monte-Carlo noise,
  no exponential cutoff -- this gives the clean asymptotic exponent.

* `measure_propagator_mc` -- the Monte-Carlo estimate from simulated cascades.
  At finite eta the power law is cut off at tau_c ~ 1/(beta(1-eta)); fit inside
  tau << tau_c (see `fit_tail`).

Predictions under test
----------------------
  spacetime path-counting :  psi ~ tau^{-d/2}        (handoff conjecture)
  spatial diffusion       :  psi ~ tau^{-(d-1)/2}     (random-walk return)
The exact computation selects the spatial-diffusion exponent (d-1)/2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from simulator import HawkesParams, branching_cascade


# ----------------------------------------------------------------------------
# Exact deterministic critical propagator at the origin
# ----------------------------------------------------------------------------
def _return_probabilities(space_dim: int, nmax: int, grid: int | None = None) -> np.ndarray:
    """p_n(0->0) for the lazy nearest-neighbour walk on Z^{space_dim}: one step
    stays put or moves to one of 2*space_dim neighbours, each with prob
    1/(2*space_dim+1).  Characteristic function phi(k) = (1 + 2 sum_i cos k_i) /
    (2*space_dim+1); p_n(0->0) = mean over the Brillouin zone of phi(k)^n."""
    if grid is None:
        grid = {1: 512, 2: 256, 3: 96}.get(space_dim, 48)
    k = np.fft.fftfreq(grid) * 2 * np.pi
    mesh = np.meshgrid(*([k] * space_dim), indexing="ij")
    phi = (1 + 2 * sum(np.cos(km) for km in mesh)) / (2 * space_dim + 1)
    p = np.empty(nmax + 1)
    phin = np.ones_like(phi)
    for n in range(nmax + 1):
        p[n] = phin.mean().real
        phin = phin * phi
    return p


def exact_critical_propagator(d: int, taus: np.ndarray, beta: float = 1.0,
                              nmax: int | None = None) -> np.ndarray:
    """Deterministic critical (eta=1) origin propagator m(tau, 0) for space-time
    dimension d (spatial dimension d-1)."""
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    space_dim = d - 1
    if nmax is None:
        nmax = int(10 * beta * taus.max() + 50)
    p = _return_probabilities(space_dim, nmax)
    n = np.arange(1, nmax + 1)
    out = np.empty_like(taus)
    for i, tau in enumerate(taus):
        # Erlang_n(tau) = beta (beta tau)^{n-1} e^{-beta tau} / (n-1)!
        log_erl = np.log(beta) + (n - 1) * np.log(beta * tau) - beta * tau - gammaln(n)
        out[i] = np.sum(p[1:] * np.exp(log_erl))
    return out


# ----------------------------------------------------------------------------
# Monte-Carlo propagator measurement
# ----------------------------------------------------------------------------
class PropagatorResult:
    """Holds the measured psi(tau, D) and the per-batch event counts needed for
    bootstrap error bars."""

    def __init__(self, centers, edges, track_distances, batch_counts, shell_sizes,
                 n_realizations, n_batches, eta, beta, d):
        self.centers = centers                 # (n_bins,) geometric bin centres
        self.edges = edges                     # (n_bins+1,)
        self.track_distances = track_distances # list of D values measured
        # batch_counts[D] : (n_batches, n_bins) raw event counts
        self.batch_counts = batch_counts
        self.shell_sizes = shell_sizes         # {D: number of vertices at distance D}
        self.n_realizations = n_realizations
        self.n_batches = n_batches
        self.eta = eta
        self.beta = beta
        self.d = d

    @property
    def widths(self):
        return np.diff(self.edges)

    def psi(self, D: int) -> np.ndarray:
        """Mean per-vertex propagator at distance D."""
        total = self.batch_counts[D].sum(axis=0)
        norm = self.n_realizations * self.widths * max(self.shell_sizes[D], 1)
        return total / norm

    def cutoff_tau(self) -> float:
        """Exponential cutoff scale 1/(beta (1-eta))."""
        return 1.0 / (self.beta * (1.0 - self.eta))


def measure_propagator_mc(lattice, params: HawkesParams, t_max: float,
                          n_realizations: int, rng: np.random.Generator,
                          track_distances=(0, 1, 2), n_time_bins: int = 48,
                          t_min: float = 0.5, n_batches: int = 100) -> PropagatorResult:
    """Estimate psi(tau, D) by Monte-Carlo over cascades (exact branching engine).

    Events are histogrammed into log-spaced time bins.  Realisations are grouped
    into `n_batches` equal batches whose per-batch histograms are retained so a
    block bootstrap over batches gives error bars on the fitted exponent.
    """
    edges = np.logspace(np.log10(t_min), np.log10(t_max), n_time_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dist_table = lattice.all_distances()
    track_distances = list(track_distances)
    shell_sizes = {D: int(np.sum(dist_table == D)) for D in track_distances}

    batch_counts = {D: np.zeros((n_batches, n_time_bins)) for D in track_distances}
    per_batch = max(1, n_realizations // n_batches)
    n_realizations = per_batch * n_batches  # round to a multiple

    for b in range(n_batches):
        local = {D: np.zeros(n_time_bins) for D in track_distances}
        for _ in range(per_batch):
            ev = branching_cascade(lattice, params, t_max=t_max, rng=rng)
            for D in track_distances:
                m = ev.distances == D
                if np.any(m):
                    c, _ = np.histogram(ev.times[m], bins=edges)
                    local[D] += c
        for D in track_distances:
            batch_counts[D][b] = local[D]

    return PropagatorResult(centers, edges, track_distances, batch_counts,
                            shell_sizes, n_realizations, n_batches,
                            params.eta, params.beta, lattice.d)


# ----------------------------------------------------------------------------
# Tail fitting
# ----------------------------------------------------------------------------
def fit_tail(result: PropagatorResult, D: int = 0, cutoff_fraction: float = 0.3,
             t_fit_min: float = 3.0, min_count: int = 20, n_boot: int = 500,
             rng: np.random.Generator | None = None) -> dict:
    """Fit psi(tau, D) ~ tau^{-p} by weighted log-log regression inside an
    intermediate window  t_fit_min <= tau <= cutoff_fraction / (beta (1-eta)),
    chosen to stay well below the exponential cutoff.  Block-bootstrap over
    batches gives a confidence interval on p.

    Returns a dict with the point estimate, bootstrap CI, the fit window and the
    candidate predictions (d-1)/2 and d/2 for comparison.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    centers = result.centers
    widths = result.widths
    tau_hi = cutoff_fraction * result.cutoff_tau()
    total = result.batch_counts[D].sum(axis=0)
    mask = (centers >= t_fit_min) & (centers <= tau_hi) & (total >= min_count)
    if mask.sum() < 3:
        raise ValueError(f"too few usable bins for D={D} (have {mask.sum()}); "
                         f"increase n_realizations or eta, or widen window")

    norm = result.n_realizations * widths * max(result.shell_sizes[D], 1)

    def slope_for(counts_per_batch):
        psi = counts_per_batch.sum(axis=0) / norm
        x = np.log(centers[mask])
        y = np.log(psi[mask])
        # weight by counts (Poisson): var(log psi) ~ 1/count
        w = total[mask]
        A = np.vstack([x, np.ones_like(x)]).T
        W = np.diag(w)
        coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
        return coef[0]

    p_hat = -slope_for(result.batch_counts[D])

    # block bootstrap over batches
    nb = result.n_batches
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nb, size=nb)
        boot[i] = -slope_for(result.batch_counts[D][idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "d": result.d,
        "D": D,
        "eta": result.eta,
        "p_hat": p_hat,
        "ci95": (lo, hi),
        "boot_std": float(np.std(boot)),
        "fit_window": (float(t_fit_min), float(tau_hi)),
        "n_bins_used": int(mask.sum()),
        "pred_spatial_diffusion": (result.d - 1) / 2,
        "pred_spacetime": result.d / 2,
    }


def powerlaw_mle_truncated(samples: np.ndarray, a: float, b: float) -> dict:
    """Maximum-likelihood estimate of the exponent p of a *truncated* power-law
    density  f(tau) = tau^{-p} / Z  on the interval [a, b], with
    Z(p) = (b^{1-p} - a^{1-p})/(1-p)  (and Z(1)=ln(b/a)).

    This is the correct ML estimator for our problem: the pooled cascade event
    times at a fixed distance follow a density proportional to the propagator
    psi(tau) ~ tau^{-p} over the sub-cutoff window [a, b].  Unlike the Hill
    estimator it does NOT assume a Pareto tail extending to infinity (our process
    has an exponential cutoff just above b), and unlike log-log least squares it
    uses the likelihood directly.

    Returns p_hat, an approximate standard error from the observed Fisher
    information, and the sample size used.
    """
    from scipy.optimize import minimize_scalar

    x = np.asarray(samples, dtype=float)
    x = x[(x >= a) & (x <= b)]
    n = len(x)
    if n < 30:
        raise ValueError(f"need >= 30 samples inside [{a},{b}] (have {n})")
    s = np.sum(np.log(x))

    def neg_ll(p):
        if abs(p - 1.0) < 1e-9:
            logZ = np.log(np.log(b / a))
        else:
            logZ = np.log(abs(b ** (1 - p) - a ** (1 - p)) / abs(1 - p))
        return p * s + n * logZ

    res = minimize_scalar(neg_ll, bounds=(0.01, 4.0), method="bounded")
    p_hat = float(res.x)
    # numerical second derivative of the per-sample log-likelihood -> SE
    h = 1e-3
    d2 = (neg_ll(p_hat + h) - 2 * neg_ll(p_hat) + neg_ll(p_hat - h)) / h ** 2
    se = float(1.0 / np.sqrt(d2)) if d2 > 0 else float("nan")
    return {"p_hat": p_hat, "p_se": se, "n": n, "window": (a, b)}


def hill_estimator(samples: np.ndarray, k: int | None = None,
                   tail_fraction: float = 0.1) -> dict:
    """Hill estimator of the power-law tail index of a *sample* (e.g. the pooled
    event times at a given distance, whose density ~ tau^{-p} implies a CCDF
    tail exponent alpha = p - 1).  Returns alpha_hat and the implied propagator
    exponent p = alpha + 1.

    k is the number of upper-order statistics used (default: tail_fraction*N)."""
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[x > 0]
    n = len(x)
    if n < 20:
        raise ValueError("need >= 20 positive samples for a stable Hill estimate")
    if k is None:
        k = max(10, int(tail_fraction * n))
    k = min(k, n - 1)
    top = x[-(k + 1):]
    xk = top[0]
    alpha = 1.0 / np.mean(np.log(top[1:] / xk))
    # standard error ~ alpha / sqrt(k)
    return {"alpha_hat": alpha, "alpha_se": alpha / np.sqrt(k),
            "p_hat": alpha + 1.0, "k": k, "n": n}
