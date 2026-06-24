"""Generate analysis.ipynb for Workstream B from cell definitions."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# Workstream B — 4-cycle plaquette particle stability

Test the refined particle conjecture: stable topological cycle patterns emerge on
the Z2×Z2 plaquette with a stability peak between η* = 1 − 1/√2 ≈ 0.293 and η = 1.

Stability ratio  **S(η) = τ_cycle · r_cycle**  (lifetime × creation rate).
""")

code("""import json, os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.getcwd()
RES = 'plaquette_particles/results' if os.path.exists('plaquette_particles/results/summary.json') else 'results'
S = json.load(open(os.path.join(RES,'summary.json')))
etas = np.array(S['etas']); eta_star = S['eta_star']
print('eta* =', round(eta_star,4), '| rho values:', S['rho_values'], '| quick:', S['quick'])
""")

md("## S(η) for each cross/self ratio ρ, with η* marked")

code("""plt.figure(figsize=(6.5,4.5))
for rho, d in S['by_rho'].items():
    Sc = np.array(d['S_curve']); se = np.array(d['S_se'])
    plt.errorbar(etas, Sc, yerr=se, marker='o', ms=4, capsize=2, label=f'ρ={rho}')
plt.axvline(eta_star, color='k', ls='--', alpha=0.6, label='η*≈0.293')
plt.xlabel('η (branching ratio)'); plt.ylabel('S(η) = τ_cycle · r_cycle')
plt.title('Cycle stability ratio vs η'); plt.legend(); plt.tight_layout(); plt.show()
""")

md("## Decomposition: creation rate and lifetime (ρ from the middle of the sweep)")

code("""rho_mid = list(S['by_rho'])[len(S['by_rho'])//2]
rows = S['by_rho'][rho_mid]['rows']
r_cycle = [r['cycle']['r_cycle'] for r in rows]
tau = [r['cycle']['tau_cycle'] for r in rows]
cpc = [r['cycle']['mean_cycles_per_chain'] for r in rows]
fig, ax = plt.subplots(1,3, figsize=(13,3.8))
ax[0].plot(etas, r_cycle, 'o-'); ax[0].set_title('r_cycle (creation rate)'); ax[0].set_xlabel('η')
ax[1].plot(etas, tau, 'o-'); ax[1].set_title('τ_cycle (lifetime)'); ax[1].set_xlabel('η')
ax[2].plot(etas, cpc, 'o-'); ax[2].axhline(1, color='r', ls=':')
ax[2].set_title('mean cycles per chain'); ax[2].set_xlabel('η')
for a in ax: a.axvline(eta_star, color='k', ls='--', alpha=0.5)
plt.suptitle(f'ρ = {rho_mid}'); plt.tight_layout(); plt.show()
print('cycles-per-chain stays ~1 => detected cycles essentially never persist '
      'beyond a single traversal (no genuine long-lived particle).')
""")

md("## Cycle vs anti-cycle vs twisted (control)")

code("""rows = S['by_rho'][rho_mid]['rows']
cyc = [r['cycle']['S'] for r in rows]
anti = [r['anti_cycle']['S'] for r in rows]
twist = [r['twisted_rate'] for r in rows]
plt.figure(figsize=(6.5,4.2))
plt.plot(etas, cyc, 'o-', label='cycle  A→B→C→D')
plt.plot(etas, anti, 's-', label='anti-cycle A→D→C→B')
plt.plot(etas, twist, '^-', label='twisted rate')
plt.axvline(eta_star, color='k', ls='--', alpha=0.5, label='η*')
plt.xlabel('η'); plt.ylabel('S / rate'); plt.legend(); plt.title(f'ρ={rho_mid}')
plt.tight_layout(); plt.show()
""")

md("## Peak summary")

code("""for rho, d in S['by_rho'].items():
    p = d['peak']
    print(f"ρ={rho}: in-window max at η={p['eta_peak']} "
          f"(S={p['S_peak']:.4f}±{p['S_peak_se']:.4f}), "
          f"interior peak (rises then falls): {p['interior_peak']}")
""")

md("""## Conclusion

S(η) shows a **broad maximum inside (η*, 1)** — located near η ≈ 0.3–0.4, close to
η* ≈ 0.293 — that **sharpens as the cross/self coupling ρ increases**. So the
conjecture's prediction of an in-window stability peak is *qualitatively
supported*.

**But** the peak is a τ_cycle × r_cycle *tradeoff* (lifetime falls and creation
rate rises with η; their product peaks in between), and crucially **cycles never
persist beyond a single traversal** (mean cycles-per-chain ≈ 1 at all η, ρ). The
Lorentzian orientation (timelike edges A→D, B→C one-directional) removes any
directed Hamiltonian cycle from the excitation graph, so A→B→C→D cannot
self-sustain — there is no genuine long-lived topological 'particle', only
transient cycle detections re-seeded by the background.

**Verdict:** *refined, not confirmed.* A stability peak exists near η*, but it
does not correspond to persistent particles. If the conjecture requires genuine
persistence, the plaquette needs a closing edge (D→A) to support a directed
4-cycle; that is a concrete, testable modification.
""")

nb = {"cells": [
        {"cell_type": t, "metadata": {},
         **({"source": s.splitlines(keepends=True)} if t=="markdown"
            else {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
        for t,s in cells],
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python","version":"3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE,"analysis.ipynb"),"w"), indent=1)
print("wrote analysis.ipynb")
