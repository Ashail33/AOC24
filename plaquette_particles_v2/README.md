# Workstream B v2 — stacked plaquette, genuine cycle persistence

The v1 plaquette could not test the refined particle conjecture honestly: its
Lorentzian orientation admits **no directed Hamiltonian cycle**, so the measured
S(η) peak was a rate×lifetime artifact (cycles never recurred). This successor
fixes that **without** introducing closed timelike curves.

## The fix: stack T layers along time

Vertices (t, v) for t ∈ [0, T), v ∈ {A,B,C,D}, flat index 4t+v.
- **within-layer** edges = the v1 plaquette (A↔B, C↔D spacelike; A→D, B→C
  timelike) — supports the *path* A→B→C→D inside a layer;
- **inter-layer** edges (t,v)→(t+1,v), strictly forward in the layer index — the
  time direction, no CTC.

A "cycle" is the horizontal traversal A_t→B_t→C_t→D_t. Genuine particle-like
persistence is the *same cycle recurring at layer t+1*, driven causally by the
forward inter-layer arrows. That is the observable v1 structurally could not host.

## The honest observable

Not S = τ·r (which v1 showed is artifact-prone), but the **cross-layer
propagation lift**:

    lift(η) = p_prop(η) − baseline(η)

- `p_prop` = P(a cycle completes at layer t+1, same orientation, within a causal
  window | a cycle completed at layer t)
- `baseline` = the Poisson chance level set by the per-layer cycle rate

`lift > 0` (CI excluding 0) means cycles genuinely propagate up the stack — a
particle, not a coincidence. We also report `persistence_length` (mean consecutive
layers a cycle survives).

## Result (see `results/summary.json`)

Unlike v1 (lift ≡ 0, pure artifact), the stacked plaquette shows a **significant,
interior peak of lift(η) in (η\*, 1)** near η ≈ 0.5–0.6: at the peak the cycle
propagates several times more often than chance. So the causal-structure-preserving
graph supports **genuine cross-layer persistence** that the toy lacked — the
refined particle conjecture is *supported*, not merely refined away.

Caveat: the persistence length stays modest (~1.1 layers) — propagation is real
and well above chance, but cycles are not yet long-lived solitons. Stronger
persistence likely needs a larger inter-layer coupling `rho_time` and/or a
braided (Bilson-Thompson) structure with conserved topological invariants — the
natural v3.

## Files

| file | role |
|---|---|
| `stacked_plaquette.py` | T-layer causal-stack geometry + spectral-radius-normalised branching matrix |
| `hawkes_nd.py` | general n-vertex multivariate Hawkes (Ogata thinning) |
| `persistence.py` | per-layer cycle detection + cross-layer propagation lift + persistence length |
| `eta_sweep_v2.py` | main experiment (sweeps η; params T, ρ, ρ_time) |
| `analysis.ipynb` | lift(η) with error bars, propagation vs baseline, persistence length |
| `tests/test_stacked.py` | spectral radius, forward-only inter-layer, n-vertex mean intensity, detector/propagation |

## Run

```bash
python plaquette_particles_v2/eta_sweep_v2.py --quick           # ~30 s
python plaquette_particles_v2/eta_sweep_v2.py                   # full (1e5 events x 30 reals)
python plaquette_particles_v2/eta_sweep_v2.py --T 12 --rho-time 2.0
python -m pytest plaquette_particles_v2/tests/ -q
```
