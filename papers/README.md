# Event-Potential Theory — paper series (LaTeX)

Three self-contained LaTeX papers developing Event-Potential Theory (EPT) — a
framework treating events as primitive and the spacetime metric as emergent from
event density on a causal graph, with Hawkes processes as the dynamical engine.

The papers are grounded in, and report, the actual results of the repository's
three computational workstreams (`hawkes_graph`, `plaquette_particles`,
`ept_cosmology`; see top-level `FINDINGS.md`). Notably, they reflect the *clean
falsification* of the original `τ^{−d/2}` propagator-tail conjecture: the
measured/exact exponent is `(d−1)/2`, with the consequent Hurst prediction
`H = (d−4)/2`.

| file | title | grounded in |
|---|---|---|
| `paper1_foundations.tex` | Foundations: discrete general covariance + ballistic-time kernel determination | Workstream A; GMC/`d=4` analysis |
| `paper2_cosmology.tex` | Cosmology: multifractal everpresent-Λ and its forward map to observables | Workstream C (`ept_cosmology`) |
| `paper3_simulation.tex` | Simulation companion: numerical determination of the propagator tail | Workstream A (`hawkes_graph`) |

## Honesty notes (carried into the papers)

- **Paper 3** reports the falsification-with-mechanism explicitly: three
  independent methods (exact deterministic, Monte-Carlo + block bootstrap, MLE)
  agree on `(d−1)/2` and exclude `d/2`. Exact exponents: `0.4975 / 0.9970 /
  1.4985` for `d = 2/3/4`. MC/MLE fits sit slightly steep due to finite-η cutoff
  bias; the exact computation is the unbiased arbiter.
- **Paper 2** is explicit that the DESI MCMC is **deliberately deferred** (gated
  on the Paper-1 foundational reformulation), while the forward map and the
  ΛCDM-limit check (σ→0 to ~1e−8) are built and validated.
- Only `d = 4` (marginally `d = 5`) lies in the valid Jaisson–Rosenbaum
  rough-CIR window; this restriction is stated in all three papers.

## Building

Each `.tex` is standalone and compiles with a standard TeX distribution
(`amsmath`, `amsthm`, `amssymb`, `mathtools`, `hyperref`, `geometry`, `booktabs`,
`cite`):

```bash
cd papers
pdflatex paper1_foundations.tex
pdflatex paper2_cosmology.tex
pdflatex paper3_simulation.tex
```

(Run `pdflatex` twice per file to resolve cross-references.) There are no
external figure dependencies; figures from the workstream notebooks would be
added via `\includegraphics` when finalized. Author/affiliation placeholders are
left as `[Author name]` / `[Affiliation]`.
