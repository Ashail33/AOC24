# Workstream C — EPT cosmological observable pipeline

Forward model converting a Λ(t) trajectory into cosmological observables
(H(z), w(z), distance modulus, BAO ratios, compressed-CMB scale) for the
everpresent-Λ / rough-CIR scenario. This is the **input to the eventual DESI
fit** — built and validated here; the **MCMC is deliberately deferred**.

> **Gating note.** The handoff says to invest in C only once Workstream A
> confirms Paper 2. Workstream A instead *falsified* the τ^{−d/2} conjecture
> (the effective tail is τ^{−(d−1)/2}), which shifts the J-R limit roughness by
> ½. So this workstream is delivered as a **validated scaffold**: the trajectory
> generators and forward map are correct and tested, but the production CAMB +
> MCMC investment should wait until Paper 2's foundations are reformulated.

## What works and is validated

- **Standard CIR** (`lambda_trajectory/standard_cir.py`) — full-truncation Euler;
  stationary mean/variance match analytics.
- **Rough/fractional CIR** (`rough_cir.py`) — Volterra-Euler integration with the
  singular kernel K(t)=t^{H−1/2}/Γ(H+½); the recovered path roughness tracks the
  input Hurst H (estimates 0.14/0.30/0.49 for H=0.1/0.3/0.5).
- **Signed Λ = Λ₊−Λ₋** (`signed_lambda.py`) — ⟨Λ⟩≈0 for the everpresent context.
- **Friedmann + observables** (`cosmology/`) — self-contained H(z), w(z), μ(z),
  BAO D_V/D_M/D_H over r_d, CMB acoustic scale l_A and shift R (no CAMB needed).
- **Forward model** (`forward_model.py`) — (η, H, σ) → observables with 16–84%
  bands over trajectory ensembles.
- **ΛCDM limit** — σ→0 reproduces ΛCDM (w→−1, distances) to ~1e−8. This is the
  η=0 Sorkin Poisson everpresent-Λ check.
- **Roughness signature** — std of the second difference of w(z) is larger for
  rough (small H) than smooth (H=½) CIR: a concrete rough-vs-smooth discriminant.

## Files

```
ept_cosmology/
├── lambda_trajectory/{standard_cir,rough_cir,signed_lambda,validation}.py
├── cosmology/{friedmann,observables,camb_interface}.py
├── forward_model.py        # (eta,H,sigma) -> observables + bands; LCDM-limit check
├── plots.ipynb             # trajectories, w(z) ensembles, observables
└── tests/test_cosmology.py
```

## Run

```bash
python ept_cosmology/lambda_trajectory/validation.py   # generator sanity checks
python ept_cosmology/forward_model.py                  # LCDM limit + roughness signature
python -m pytest ept_cosmology/tests/ -q
```

## Explicitly deferred (do NOT run yet)

- The **MCMC fit against DESI** — gated on Paper 2 finalisation.
- Full **CAMB/CLASS** Boltzmann solve — `camb_interface.py` uses CAMB if importable,
  else falls back to the Aubourg+2015 r_drag fitting formula.
- A **fast rough-CIR integrator** — the current Volterra scheme is O(n²) per path;
  a sum-of-exponentials (El Euch–Rosenbaum) kernel approximation is the production
  upgrade.
```
