"""Generate analysis.ipynb from a list of (markdown|code) cells.  Kept as a
plain script so the notebook is reproducible and reviewable in git."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

cells = []

def md(src): cells.append(("markdown", src))
def code(src): cells.append(("code", src))

md("""# Workstream A — effective propagator tail index

This notebook loads the results produced by `run_experiment.py` and compares the
measured tail exponent of the effective (origin-return) propagator ψ(τ) against
the two competing predictions:

- **spacetime path-counting (handoff conjecture):** ψ(τ) ~ τ^{−d/2}
- **spatial random-walk return:** ψ(τ) ~ τ^{−(d−1)/2}

**Result:** the data select **(d−1)/2**. In a continuous-time Hawkes process the
time direction is ballistic (generation count n ≈ βτ is sharply concentrated),
so it contributes no diffusive ½-power — only the d−1 spatial dimensions do. The
spacetime conjecture over-counts by exactly ½.
""")

code("""import json, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath('__file__')) if '__file__' in globals() else os.getcwd()
RES = os.path.join(HERE, 'results') if os.path.basename(HERE) == 'hawkes_graph' else 'results'
if not os.path.exists(os.path.join(RES, 'summary.json')):
    RES = os.path.join('hawkes_graph', 'results')
summary = json.load(open(os.path.join(RES, 'summary.json')))
prop = np.load(os.path.join(RES, 'propagator_curves.npz'))
exact = np.load(os.path.join(RES, 'exact_curves.npz'))
print('quick run:', summary.get('quick'), '| elapsed', round(summary['elapsed_sec']), 's')
""")

md("## 1. Exact critical propagator (deterministic arbiter)\n\n"
   "Computed from lazy-random-walk return probabilities ⊗ Erlang weights — no "
   "Monte-Carlo noise, no exponential cutoff.")

code("""taus = exact['taus']
plt.figure(figsize=(6,4.5))
for d in [2,3,4]:
    psi = exact[f'd{d}']
    plt.loglog(taus, psi/psi[0], label=f'd={d}')
    # reference slope -(d-1)/2
    ref = taus**(-(d-1)/2);
    plt.loglog(taus, ref/ref[0]*(psi[0]/psi[0]), '--', color='grey', alpha=0.4)
plt.xlabel('τ'); plt.ylabel('ψ(τ,0) (normalised)')
plt.title('Exact critical propagator vs τ^{-(d-1)/2} (dashed)')
plt.legend(); plt.tight_layout(); plt.show()

print('exact fitted exponents:')
for r in summary['exact']:
    print(f"  d={r['d']}: {r['exact_exponent']:.4f}   "
          f"(d-1)/2={r['pred_spatial_diffusion']}, d/2={r['pred_spacetime']}")
""")

md("## 2. Monte-Carlo propagator and tail fit\n\n"
   "Weighted log-log regression with a block-bootstrap CI, fit inside the "
   "sub-cutoff window τ ≪ 1/(β(1−η)). Low-η rows are NaN by construction: their "
   "cutoff is too early to host any power-law range — the tail only emerges near "
   "criticality.")

code("""# tabulate the fits
print(f"{'d':>2} {'eta':>5} {'p_hat':>7} {'CI95':>18} {'(d-1)/2':>8} {'d/2':>5}")
for r in summary['propagator_fits']:
    p = r.get('p_hat', float('nan'))
    ci = r.get('ci95', (float('nan'),)*2)
    ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci[0]==ci[0] else "   -   "
    ps = f"{p:.3f}" if p==p else "  nan"
    print(f"{r['d']:>2} {r['eta']:>5} {ps:>7} {ci_s:>18} "
          f"{r['pred_spatial_diffusion']:>8} {r['pred_spacetime']:>5}")
""")

code("""# psi(tau,0) curves at the most-critical eta with cutoff marked
plt.figure(figsize=(6,4.5))
for d in [2,3,4]:
    c = prop[f'd{d}_eta0.99_centers']; y = prop[f'd{d}_eta0.99_psi0']
    m = y>0
    plt.loglog(c[m], y[m]/y[m][0], 'o-', ms=3, label=f'd={d} (η=0.99)')
    ref = c**(-(d-1)/2)
    plt.loglog(c[m], ref[m]/ref[m][0], '--', color='grey', alpha=0.5)
plt.axvline(0.3/0.01, color='r', ls=':', alpha=0.4, label='fit cutoff 0.3/(1-η)')
plt.xlabel('τ'); plt.ylabel('ψ(τ,0) (normalised)')
plt.title('MC propagator vs τ^{-(d-1)/2} (dashed)')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("## 3. Exponent vs prediction — the verdict")

code("""fig, ax = plt.subplots(figsize=(6,4.5))
ds = [2,3,4]
# MC at eta=0.99
mc = {r['d']: r for r in summary['propagator_fits'] if r['eta']==0.99}
mle = {r['d']: r for r in summary['mle'] if 'p_hat_mle' in r}
exa = {r['d']: r for r in summary['exact']}
ax.plot(ds, [(d-1)/2 for d in ds], 'g-', label='(d-1)/2 spatial diffusion')
ax.plot(ds, [d/2 for d in ds], 'r-', label='d/2 spacetime conjecture')
ax.plot(ds, [exa[d]['exact_exponent'] for d in ds], 'k*', ms=14, label='exact critical')
for d in ds:
    r = mc.get(d)
    if r and r.get('p_hat')==r.get('p_hat'):
        ci = r['ci95']
        ax.errorbar(d-0.04, r['p_hat'], yerr=[[r['p_hat']-ci[0]],[ci[1]-r['p_hat']]],
                    fmt='bo', capsize=3, label='MC fit (η=0.99)' if d==2 else None)
    if d in mle:
        ax.errorbar(d+0.04, mle[d]['p_hat_mle'], yerr=mle[d]['p_se'],
                    fmt='ms', capsize=3, label='truncated-PL MLE' if d==2 else None)
ax.set_xticks(ds); ax.set_xlabel('spacetime dimension d')
ax.set_ylabel('propagator tail exponent p')
ax.set_title('Measured exponent selects (d-1)/2, excludes d/2')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("""## 4. Jaisson–Rosenbaum scaling limit (corroboration, lower confidence)

The same ½-shift carries into the J-R limit: a kernel tail (1+α) maps to Hurst
H = α−½. With the measured kernel exponent (d−1)/2 this gives the
**kernel-consistent** prediction H = (d−4)/2, versus the handoff's (d−3)/2.
Direct Hurst estimation from finite near-critical sims is noisy; treat as
corroboration of the (clean) propagator result, not an independent precise
measurement.""")

code("""for r in summary['hurst']:
    print(f"d={r['d']} eta={r['eta']}: H_dfa={r['H_dfa']:.3f}±{r['H_dfa_se']:.3f}  "
          f"H_aggvar={r['H_aggvar']:.3f}  | pred kernel-consistent={r['H_pred_kernel_consistent']}, "
          f"handoff={r['H_pred_handoff']}")
""")

md("""## Conclusion

For the **continuous-time** Hawkes process on the spatial causal lattice, the
effective origin-return propagator decays as **τ^{−(d−1)/2}**, matching the
spatial random-walk return exponent to three significant figures (exact
computation) and confirmed by independent Monte-Carlo regression and MLE with
CIs that **exclude the τ^{−d/2} conjecture**.

**Implication for Paper 2:** the path-counting heuristic that yields τ^{−d/2}
implicitly treats the time axis as a diffusive lattice direction. In a genuine
continuous-time process the time axis is ballistic, so the correct effective tail
is (d−1)/2. The conjecture is recoverable only under a fully space-*time*-
discretized random-walk model; that modelling choice should be made explicit (or
the prediction revised to (d−1)/2).
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {},
         **({"source": s.splitlines(keepends=True)} if t == "markdown"
            else {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
        for t, s in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(HERE, "analysis.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
