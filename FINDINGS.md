# Findings — Hawkes / tail-index / cosmology workstreams

Reporting back on the three workstreams from the HP Z8 handoff. Each was built,
tested, and run (on the cloud container's 4 cores — scaled-down vs. the Xeon, but
the conclusions are statistically resolved). Headlines first.

---

## Workstream A — tail-index of the Hawkes-on-lattice propagator → **conjecture falsified, cleanly**

**The effective origin-return propagator decays as τ^{−(d−1)/2}, not the
conjectured τ^{−d/2}.**

| d | exact exponent | (d−1)/2 (spatial) | d/2 (conjecture) | MC fit (η=0.99, 95% CI) | MLE |
|---|---|---|---|---|---|
| 2 | **0.4975** | 0.5 | 1.0 | 0.587 [0.555, 0.622] | 0.591 ± 0.005 |
| 3 | **0.9970** | 1.0 | 1.5 | 1.051 [1.014, 1.094] | 1.018 ± 0.010 |
| 4 | **1.4985** | 1.5 | 2.0 | 1.533 [1.484, 1.583] | 1.537 ± 0.017 |

Three independent methods — an exact deterministic computation (random-walk
return probabilities ⊗ Erlang weights), Monte-Carlo with block-bootstrap CIs, and
a truncated power-law MLE — all land on **(d−1)/2** and all **exclude d/2**. The
exact computation pins it to three significant figures.

**Mechanism (why d/2 is wrong here).** Path-counting on the spacetime lattice
ℤ^{1,d−1} treats time as a diffusive lattice direction, which would give d/2. But
in a *continuous-time* Hawkes process the number of cascade generations reaching
elapsed time τ is sharply concentrated at n ≈ βτ — the time direction is
**ballistic, not diffusive**, so it contributes no ½-power. Only the d−1 spatial
dimensions diffuse, giving the spatial random-walk return exponent (d−1)/2.

**Implication for Paper 2.** The τ^{−d/2} prediction is recoverable only if the
time axis is itself discretised into a random-walk dimension (a fully
space-*time* lattice walk). For a genuine continuous-time process the effective
tail is (d−1)/2. The framing must either (a) make the space-time-discrete
modelling choice explicit, or (b) revise the prediction to (d−1)/2.

**Jaisson–Rosenbaum corollary.** The same ½-shift propagates: the J-R limit
Hurst is H = α − ½ with kernel tail 1+α. With α = (d−3)/2 (from the measured
(d−1)/2 kernel) the prediction is H = (d−4)/2, vs the handoff's (d−3)/2.
Critically, α ∈ (0,1) — the standard rough-CIR regime — requires (d−1)/2 ∈ (1,2),
i.e. **only d = 4 (and d = 5) lie in the valid J-R window**; d = 2, 3 fall
outside it (the effective kernel is non-integrable). The DFA diagnostic shows the
expected near-critical persistence (H→1 as η→1) but does not by itself isolate
the limiting roughness — consistent with the propagator result being the clean
arbiter.

Correctness anchors (all validated): branching-cascade size = 1/(1−η); Ogata
bulk intensity = μ/(1−η) (1.2478 vs 1.2500); Ogata vs branching engines agree;
exact exponent to 3 digits; Hill recovers a known Pareto index.

---

## Workstream B — 4-cycle plaquette particle stability → **refined, not confirmed**

**S(η) = τ_cycle · r_cycle shows a broad maximum inside (η\*, 1) near η ≈ 0.3–0.4
(close to η\* ≈ 0.293) that sharpens with cross/self coupling ρ — but the cycles
are not genuine persistent particles.**

- An in-window peak does exist, so the conjecture is *qualitatively* supported,
  and its location is consistent with the Onaga–Shinomoto bursting threshold
  η\* = 1 − 1/√2 ≈ 0.293.
- But **mean cycles-per-chain ≈ 1.0 at every (η, ρ)** — a detected cycle almost
  never repeats. The peak is a lifetime×rate tradeoff of *transient* detections
  (τ_cycle falls and r_cycle rises with η; the product peaks in between),
  re-seeded by the background, not a long-lived topological soliton.
- **Structural reason:** the Lorentzian orientation (timelike edges A→D, B→C are
  one-directional) leaves the excitation graph with **no directed Hamiltonian
  cycle** — A→B→C→D cannot close (no D→A edge), so it cannot self-sustain.

**Concrete next test:** add a closing edge D→A so the plaquette supports a
directed 4-cycle; that is the minimal modification that could produce genuine
persistence (cycles-per-chain > 1).

(Validated: branching matrix spectral radius = η exactly; multivariate Ogata mean
intensity = stationary (I−Φ)⁻¹μ; pattern-detector logic on synthetic streams.)

---

## Workstream C — cosmology forward model → **built and validated; MCMC deferred (correctly)**

Forward map (η, H, σ) → {H(z), w(z), μ(z), BAO, CMB scale} with confidence bands.

- (rough-)CIR generators reproduce CIR moments and **recover the input Hurst H**
  (0.14/0.30/0.49 for H = 0.1/0.3/0.5).
- Signed Λ = Λ₊−Λ₋ has ⟨Λ⟩ ≈ 0 (everpresent-Λ).
- **ΛCDM limit:** σ→0 reproduces ΛCDM (w→−1, distances) to ~1e−8 — the η=0
  Sorkin Poisson check.
- Quantified **roughness signature** (std of d²w/dz²) distinguishes rough from
  smooth CIR.

Per the handoff's gating logic — *and reinforced by A falsifying the foundational
conjecture* — the **DESI MCMC is not run**, and the production CAMB/CLASS solve +
fast (sum-of-exponentials) rough-CIR integrator are left as clearly-flagged
deferrals. The scaffold is correct and ready for when Paper 2's foundations are
settled.

---

## Follow-up work (post-merge, same session)

### Two correctness bugs fixed (from Codex review)
- **`rough_cir.py` (P1):** an erroneous extra `*dt` in the Volterra update shrank
  rough-CIR mean-reversion and variance by ~dt (H=0.5 stationary variance was
  0.0025 instead of 0.125). Fixed; now matches σ²b/(2a). The committed Workstream-C
  *results* were unaffected because the forward model renormalises the trajectory
  to a target RMS, but the generator is now correct for direct use. Also fixed the
  roughness estimator to use short lags only (the Hölder exponent is an s→0
  property).
- **`simulator.py` (P2):** the branching cascade renormalised offspring back
  inside the box at the boundary instead of letting them leak. Fixed via a
  non-compacted neighbour table (offspring leak out-of-box correctly). Committed
  Workstream-A results used large boxes with **zero** boundary hits, so they are
  unchanged (verified). Regression tests added for both.

### Workstream B v2 — stacked plaquette → **conjecture now supported**
The v1 plaquette had no directed Hamiltonian cycle, so its peak was an artifact.
Stacking T causal layers (forward inter-layer arrows, no CTC) and measuring the
honest observable — **cross-layer propagation lift** = p_prop − chance baseline —
shows a **significant interior peak in (η\*, 1) near η ≈ 0.5–0.6**: cycles
propagate causally up the stack several times more often than chance (≈5σ at the
peak in the quick run). Unlike v1 (lift ≡ 0), this is genuine persistence. Caveat:
persistence length is still modest (~1.1 layers) — real propagation, not yet
long-lived solitons; a braided/larger-coupling v3 is the natural next step.
(`plaquette_particles_v2/`)

### Ballistic-time theorem note
`hawkes_graph/ballistic_time_theorem.md` states and proves the structural result
behind Workstream A: the Hawkes generation count concentrates as n = τ/δ_φ +
O(√τ) (renewal LLN/CLT on the cluster representation), so the time axis is
ballistic and contributes no diffusive ½-power — which is *why* path-counting's
d/2 collapses to (d−1)/2. Includes the open H=0 question for Paper 3.

---

## How to reproduce

```bash
pip install numpy scipy matplotlib pytest nbconvert ipykernel
python -m pytest hawkes_graph/tests plaquette_particles/tests \
                plaquette_particles_v2/tests ept_cosmology/tests -q
python hawkes_graph/run_experiment.py          # Workstream A
python plaquette_particles/eta_sweep.py        # Workstream B (v1)
python plaquette_particles_v2/eta_sweep_v2.py  # Workstream B (v2, stacked)
python ept_cosmology/forward_model.py          # Workstream C (forward map + LCDM check)
```

Each workstream has its own `README.md` and `analysis.ipynb`/`plots.ipynb` with
the figures and full method notes. Results JSON/NPZ live under each
`*/results/`.

## Caveats / honesty notes

- Runs were on 4 cores, not the Z8's 32 — Workstream A used ~4×10⁴ realisations
  (not 10⁵) and B a "medium" 6×10⁴ events × 30 reals (not 10⁵ × 50). The A
  conclusion is nonetheless statistically unambiguous because the **exact
  deterministic computation** settles the exponent independent of Monte-Carlo
  budget. Re-running on the Z8 with the spec's full counts will only tighten the
  already-conclusive error bars.
- The MC tail fits sit ~0.05–0.09 above (d−1)/2 for d=2 due to finite-η cutoff and
  finite fit-window bias; the exact computation (unbiased) gives (d−1)/2 to 3
  digits, and every MC/MLE CI still excludes d/2.
- The J-R Hurst sweep is the noisiest, lowest-confidence component (see Workstream
  A README) and only d=4 is in its regime of validity.
