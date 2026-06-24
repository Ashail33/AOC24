# Ballistic time concentration in Hawkes cascades — and why path-counting gives the wrong tail

This note states and proves the structural fact behind the Workstream-A result:
the effective lattice-Hawkes propagator decays as **τ^{−(d−1)/2}**, not the
path-counting prediction **τ^{−d/2}**. The reason is that the temporal direction
of a *continuous-time* Hawkes cascade is **ballistic** (generation count grows
linearly and concentrates), so it contributes no diffusive ½-power — only the
d−1 spatial lattice dimensions do.

## Setup: the Poisson-cluster representation

By Hawkes–Oakes (1974), a Hawkes process is a Poisson cluster process. Fix one
immigrant (the seed) at the origin at time 0. Every later event in its cluster
has a unique ancestral chain
$$ \text{seed} = g_0 \to g_1 \to \cdots \to g_n = \text{event}, $$
where each $g_k$ is a direct offspring of $g_{k-1}$. Write the elementary kernel
as $\varphi(\tau) = \eta\,\rho(\tau)$ with $\rho$ a probability density (the
normalised inter-generation delay law), branching ratio $\eta=\int\varphi$, and
finite delay moments
$$ \delta_\varphi = \int_0^\infty \tau\,\rho(\tau)\,d\tau, \qquad
   \sigma_\varphi^2 = \int_0^\infty (\tau-\delta_\varphi)^2\,\rho(\tau)\,d\tau < \infty. $$
The inter-generation delays $T_k = \mathrm{time}(g_k)-\mathrm{time}(g_{k-1})$ are
i.i.d. with law $\rho$, so an event's elapsed time is the random walk
$$ \tau = \sum_{k=1}^{n} T_k . $$

## Theorem (ballistic generation count)

> **Theorem.** Conditional on a cascade event at elapsed time $\tau$, its
> generation index $n$ satisfies
> $$ \frac{n}{\tau} \xrightarrow[\tau\to\infty]{\ \mathbb{P}\ } \frac{1}{\delta_\varphi},
>    \qquad
>    n = \frac{\tau}{\delta_\varphi} + O_{\mathbb P}(\sqrt{\tau}). $$
> The relative fluctuation is $O(\tau^{-1/2})\to 0$: the generation count is
> *ballistic* (linear in $\tau$) with vanishing relative spread.

**Proof.** Given the event time $\tau$, the generation index is exactly the
renewal count of the delay walk,
$$ n = \max\Big\{ m : \sum_{k=1}^m T_k \le \tau \Big\}. $$
The elementary renewal theorem gives $n/\tau \to 1/\delta_\varphi$ a.s. (hence in
probability), and the renewal CLT gives
$$ n \;\approx\; \frac{\tau}{\delta_\varphi}
   \;+\; \mathcal N\!\Big(0,\ \frac{\sigma_\varphi^2\,\tau}{\delta_\varphi^3}\Big), $$
so $\mathrm{sd}(n)=O(\sqrt\tau)$ and $\mathrm{sd}(n)/\mathbb E[n]=O(\tau^{-1/2})$.
$\qquad\blacksquare$

### Corollary (exponential kernel — the simulated case)

For $\varphi(\tau)=\alpha e^{-\beta\tau}$, $\rho(\tau)=\beta e^{-\beta\tau}$,
$\delta_\varphi=1/\beta$, $\sigma_\varphi^2=1/\beta^2$. The $n$-th generation time
is $\mathrm{Erlang}(n,\beta)$, and conditional on an event at $\tau$ the
generation index is distributed as the count of a rate-$\beta$ Poisson process
on $[0,\tau]$:
$$ n \sim \mathrm{Poisson}(\beta\tau), \qquad
   \mathbb E[n]=\beta\tau,\ \ \mathrm{sd}(n)=\sqrt{\beta\tau}. $$
This $\mathbb E_{K\sim\mathrm{Poisson}(\beta\tau)}[\,\cdot\,]$ weighting is exactly
what the exact propagator computation in `propagator.py` sums.

## Consequence: the correct tail exponent

The expected return propagator factorises over generations as
$$ \psi(\tau,0)\;=\;\sum_{n\ge 1} p_n(0\!\to\!0)\, w_n(\tau), $$
where $p_n(0\!\to\!0)$ is the $n$-step return probability of the **spatial**
lazy random walk on $\mathbb Z^{d-1}$ and $w_n(\tau)$ is the temporal weight that
generation $n$ contributes at time $\tau$ (the Erlang/Poisson weight above).

- **Spatial factor.** By the local CLT, $p_n(0\!\to\!0)\sim C\,n^{-(d-1)/2}$ — the
  diffusive return probability in the $d-1$ spatial dimensions.
- **Temporal factor.** By the Theorem, $w_n(\tau)$ is sharply peaked at
  $n_\star=\beta\tau$ with width $\sqrt{\beta\tau}$. The temporal direction is a
  *deterministic clock*, not a diffusive coordinate: it pins $n$ to $n_\star\propto\tau$
  rather than summing over a $\sqrt\tau$-spread $d$-th random-walk axis.

Evaluating the spatial factor at the pinned $n_\star\propto\tau$,
$$ \boxed{\ \psi(\tau,0)\ \sim\ p_{n_\star}(0\!\to\!0)\ \sim\ \tau^{-(d-1)/2}.\ } $$

The naive path-counting heuristic instead treats all $d$ lattice directions of
$\mathbb Z^{1,d-1}$ symmetrically — i.e. it lets the *time* axis contribute its own
diffusive ½-power — producing the spurious $\tau^{-d/2}$. The Theorem is precisely
the statement that this over-counts by one ½-power: a continuous-time Hawkes clock
advances ballistically, so the $d$-th (temporal) factor is $\tau^{0}$, not
$\tau^{-1/2}$.

## Numerical confirmation

The exact deterministic computation (random-walk return probabilities ⊗ Erlang
weights, `exact_critical_propagator`) yields exponents
$$ 0.4975,\quad 0.9970,\quad 1.4985 \quad\text{for } d=2,3,4, $$
matching $(d-1)/2 = 0.5,\,1.0,\,1.5$ to three significant figures, with the
Monte-Carlo and truncated-MLE estimates' confidence intervals all excluding
$d/2$. See `README.md` and `results/summary.json`.

## Status of the result

The renewal LLN/CLT for the generation count is classical (renewal theory +
Hawkes–Oakes cluster representation). What is, to our knowledge, not stated
explicitly in the literature is its use as the **obstruction to lattice
path-counting** for continuous-time Hawkes effective kernels — i.e. the clean
identification that "time is ballistic" is *why* the spacetime exponent collapses
from $d/2$ to $(d-1)/2$. That framing is the contribution relevant to Paper 2 §2.2.

## Open question (flagged, not resolved)

Under the corrected kernel exponent $(d-1)/2$, the Jaisson–Rosenbaum limit Hurst
is $H=(d-4)/2$, so **$d=4$ sits exactly at $H=0$** — the lower boundary
$\alpha_{\text{kernel}}=1/2$ of the standard rough-CIR regime ($\alpha\in(1/2,1)
\Leftrightarrow d\in(4,5)$). Whether the $H=0$ limit exists as a *log-modulated*
rough process (cf. log-modulated rough volatility; Gaussian multiplicative chaos)
or degenerates is the question gating the Paper-3 cosmological prediction.
