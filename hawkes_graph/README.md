# Workstream A — Hawkes-on-graph tail-index verification

Numerically test whether a Hawkes process on the causal lattice ℤ^{1,d−1}
produces an effective (origin-return) propagator with a **power-law tail**, and
measure that tail's exponent against two competing predictions:

| prediction | exponent of ψ(τ) | origin |
|---|---|---|
| spacetime path-counting (handoff conjecture) | τ^{−d/2} | counts time as a lattice/diffusive direction |
| spatial random-walk return | τ^{−(d−1)/2} | continuous-time: time axis is *ballistic* |

**Headline result (see `results/summary.json`):** the measured exponent is
**(d−1)/2**, not d/2. The exact critical-propagator computation pins it to three
digits (−0.498, −0.997, −1.499 for d = 2, 3, 4). The naive spacetime
path-counting over-counts by exactly ½ because, in a *continuous-time* Hawkes
process, the number of cascade generations reaching time τ is sharply
concentrated at n ≈ βτ (ballistic), so the time direction contributes no
diffusive ½-power — only the d−1 spatial dimensions do.

This is a clean *falsification with a mechanism*: the conjecture holds only if
the time axis is itself discretized into a random-walk dimension. That is an
actionable distinction for Paper 2's framing.

## Model

Vertices are sites of the spatial lattice ℤ^{d−1} (the spatial slice of
ℤ^{1,d−1}). An event at site *y* excites *y* and its 2(d−1) nearest neighbours —
the immediate causal (one-light-cone-step) neighbours — through the elementary
exponential kernel

    φ_{yx}(τ) = η · w_{yx} · β e^{−βτ},   w_{yx} = 1/(2(d−1)+1),

so the branching ratio (mean offspring per event) is exactly η and the
stationary per-site mean intensity is μ/(1−η).

## Files

| file | what it does |
|---|---|
| `lattice.py` | causal lattice geometry: indexing, neighbours, L1 distances (vectorised, cached) |
| `simulator.py` | two engines — **Ogata thinning** (general) and the **exact Poisson-cluster branching cascade** (single seed); `analytic_mean_intensity` |
| `propagator.py` | MC measurement of ψ(τ,D); **exact critical propagator** via RW return probs (FFT) ⊗ Erlang weights; tail fits (weighted log-log + block bootstrap, and a Hill estimator) |
| `scaling_limits.py` | Jaisson–Rosenbaum rescaling + Hurst estimators (DFA, aggregated variance) |
| `run_experiment.py` | full d×η sweep → `results/` |
| `analysis.ipynb` | loads `results/`, plots propagators and exponent-vs-prediction |
| `tests/test_simulator.py` | correctness anchors (cascade size, mean intensity, engine cross-check, exact exponent, Hill) |

## How to run

```bash
pip install numpy scipy matplotlib pytest
python -m pytest hawkes_graph/tests/ -q          # ~30 s, 13 tests
python hawkes_graph/run_experiment.py --quick    # fast smoke run (~minutes)
python hawkes_graph/run_experiment.py            # full run (>=1e4-1e5 realisations)
jupyter nbconvert --to notebook --execute hawkes_graph/analysis.ipynb  # render figures
```

## Correctness anchors (validated in tests + during development)

- **Branching cascade size = 1/(1−η)** — exact, matched to <1% (η = 0.3/0.6/0.9).
- **Ogata bulk intensity = μ/(1−η)** — 1.2478 vs 1.2500 in the lattice bulk
  (boundary sites sit below this because they leak excitation out of the box, as
  expected; `boundary_hits` reports leakage).
- **Ogata and branching engines agree** on the cascade-size distribution.
- **Exact critical exponent = −(d−1)/2** to 3 digits.
- **Hill estimator** recovers a known Pareto index to <0.05.

## Notes on statistical care (per the handoff)

- The finite-η power law is cut off at τ_c ≈ 1/(β(1−η)); fits use only τ ≪ τ_c
  (`cutoff_fraction=0.3`). This is why naive fits in `[5,120]` at η = 0.99 read
  ≈ −0.76/−1.34/−1.82 (cutoff-biased steep) while the clean exponent is −(d−1)/2.
- Tail exponents come from **weighted regression with a block bootstrap CI** and,
  independently, a **Hill estimator** on pooled event times (density ∝ τ^{−p} ⇒
  CCDF index α = p−1). We do *not* trust a single visual log-log slope.
- The exact deterministic propagator is the authoritative arbiter; the MC run
  confirms it under finite-η/finite-box conditions.

## Scaling-limit (Jaisson–Rosenbaum) caveat

The same ½-shift propagates: J-R maps a kernel tail (1+α) to Hurst H = α−½, so
the kernel-consistent prediction is **H = (d−4)/2**, versus the handoff's
**(d−3)/2**. Direct Hurst estimation from finite near-critical simulations is
noisy and is reported as corroboration of the (clean) propagator exponent rather
than an independent high-precision measurement.
