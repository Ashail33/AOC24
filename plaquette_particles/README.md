# Workstream B — 4-cycle plaquette particle stability

Test the refined particle conjecture from the η-unification analysis: do stable
topological cycle patterns emerge on the small Z₂×Z₂ causal plaquette, with a
**stability peak between η\* ≈ 0.293 and η = 1**?

## The plaquette

Four vertices A, B, C, D with
- **spacelike** edges (bidirectional): A↔B, C↔D
- **timelike** edges (one-directional, Lorentzian): A→D, B→C

A multivariate Hawkes process runs on the four vertices. The branching matrix is
normalised so the swept parameter **η is the spectral radius** (overall branching
ratio / criticality), and **ρ = η_cross/η_self** is the cross/self coupling ratio.

## Observables

For the repeating traversal patterns (`cycle` A→B→C→D, `anti_cycle` A→D→C→B,
`twisted` A→B→C→D→A→D′):

- **r_cycle** — creation rate = completed cycles / T
- **τ_cycle** — lifetime = mean duration of a detection chain (how long the
  ordered pattern persists with continued in-order events)
- **S(η) = τ_cycle · r_cycle** — dimensionless "particleness"

Sliding window W = 4/(β·⟨intensity⟩) per the handoff prescription (scales with
the inverse event rate as η varies).

## Result (see `results/summary.json`)

- **S(η) has a broad maximum inside (η\*, 1)**, near η ≈ 0.3–0.4 (close to η\*),
  that **sharpens as ρ increases**. So an in-window peak does exist — the
  conjecture is *qualitatively supported*.
- **But cycles never persist:** mean cycles-per-chain ≈ 1 at all (η, ρ). The
  Lorentzian orientation leaves the excitation graph with **no directed
  Hamiltonian cycle** (no D→A edge), so A→B→C→D cannot self-sustain. The S(η)
  peak is a lifetime×rate tradeoff of *transient* detections re-seeded by the
  background, not a genuine long-lived particle.

**Verdict: refined, not confirmed.** A peak near η\* is real; persistent particles
are not. A concrete fix to test genuine persistence: add a closing edge D→A so the
plaquette supports a directed 4-cycle.

## Files

| file | role |
|---|---|
| `plaquette.py` | graph, branching matrix (spectral-radius–normalised), stationary intensity, window |
| `hawkes_dynamics.py` | 4-vertex multivariate Hawkes via Ogata thinning |
| `pattern_detector.py` | finite-state cycle / anti-cycle / twisted detection over the event stream |
| `stability_metrics.py` | r_cycle, τ_cycle, S, aggregation over realisations |
| `eta_sweep.py` | main experiment (sweeps η and ρ) → `results/summary.json` |
| `analysis.ipynb` | S(η) curves, decomposition, peak finder |
| `tests/test_plaquette.py` | branching-matrix spectral radius, intensity, detector logic |

## Run

```bash
python plaquette_particles/eta_sweep.py --quick        # fast (~30 s)
python plaquette_particles/eta_sweep.py                 # full (1e5 events × 50 reals × 3 ρ)
python plaquette_particles/eta_sweep.py --rho 0.5 2 4   # custom ρ sweep
python -m pytest plaquette_particles/tests/ -q
```
