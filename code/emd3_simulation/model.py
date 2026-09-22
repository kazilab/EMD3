"""EMD3 mechanistic simulation: KCC8 upstream -> EMD3 (KCC4/KCC9) -> KCC10.

A two-type population model of exposure-induced phenotypic plasticity, with
arsenic in non-tumorigenic mammary epithelium as the exemplar (Yang 2026,
Food Chem Toxicol 211:116003 -- E2F2/FZD10 -> Wnt/beta-catenin).

**What this is:** an apparatus for deciding when a change in the stem-like
fraction can be attributed to INDUCTION rather than to SELECTION, and for
designing the experiment that decides it.
**What it is not:** a fitted model of any published dataset. The published
exposure studies report endpoint marker fractions, not the longitudinal
absolute counts this question needs -- which is itself the finding.

The central problem, stated exactly
-----------------------------------
Write S for the stem-like (CSC) compartment and N for the non-stem (NSCC) one:

    dS/dt = a*S + beta*N  - gamma*S
    dN/dt = b*N - beta*N  + gamma*S

with ``a = r_S - d_S`` and ``b = r_N - d_N`` the NET growth rates, ``beta`` the
dedifferentiation rate (NSCC -> CSC; this is the EMD3 event) and ``gamma`` the
differentiation rate. For the observed fraction x = S/(S+N):

    dx/dt = (a - gamma)*x + beta*(1 - x) - x*[b + (a - b)*x]

Every term that raises x appears twice over: through ``beta`` (a cell changed
state) and through ``a - b`` (one compartment outgrew the other). An experiment
that reports only x(t) is trying to read two causes from one number.

Two limiting structures
-----------------------
Setting beta = gamma = 0 leaves pure selection, and the equation collapses to
the logistic

    dx/dt = (a - b) * x * (1 - x)                 ->  x approaches 1

Setting a = b leaves pure induction, and it collapses to a linear relaxation

    dx/dt = beta - (beta + gamma) * x             ->  x approaches beta/(beta+gamma)

These have DIFFERENT asymptotes, so they are distinguishable in principle. The
question this build answers is for how long, and at what measurement precision,
you must watch before the difference is visible -- because over a short window
both are a smooth rise through the same few percentage points, and a short
window is what exposure experiments actually run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    """Rates per day. Baseline values sit in the range reported for mammary
    epithelial lines: a small stem-like fraction maintained by slow bidirectional
    exchange, with the two compartments close to fitness parity."""

    # --- net growth ------------------------------------------------------
    r_S: float = 0.55        # stem-like proliferation
    d_S: float = 0.05        # stem-like death
    r_N: float = 0.60        # non-stem proliferation
    d_N: float = 0.05        # non-stem death

    # --- state transitions ------------------------------------------------
    beta: float = 0.010      # dedifferentiation NSCC -> CSC   (the EMD3 event)
    gamma: float = 0.120     # differentiation  CSC -> NSCC

    @property
    def a(self) -> float:
        return self.r_S - self.d_S

    @property
    def b(self) -> float:
        return self.r_N - self.d_N

    @property
    def fitness_gap(self) -> float:
        """a - b. Positive means the stem-like compartment outgrows the other,
        which raises x with no cell ever changing state."""
        return self.a - self.b

    @property
    def x_star(self) -> float:
        """Transition-only fixed point, valid when a == b."""
        return self.beta / (self.beta + self.gamma)


@dataclass(frozen=True)
class Exposure:
    """How an agent is allowed to act. Keeping the two channels separate is the
    whole point: an agent may induce, may select, may do both, and may do
    neither while still looking like all three in a marker readout."""
    d_beta: float = 0.0       # ADDITIVE change in dedifferentiation (induction)
    d_fit: float = 0.0        # ADDITIVE change in (a - b), via r_S (selection)
    kill_N: float = 0.0       # ADDITIVE death rate on NSCC only (cytotoxicity)
    kill_S: float = 0.0       # ADDITIVE death rate on CSC only
    d_gamma: float = 0.0      # ADDITIVE change in differentiation

    def apply(self, p: Params) -> Params:
        return replace(
            p,
            beta=max(p.beta + self.d_beta, 0.0),
            gamma=max(p.gamma + self.d_gamma, 0.0),
            r_S=p.r_S + self.d_fit,
            d_S=p.d_S + self.kill_S,
            d_N=p.d_N + self.kill_N,
        )


def rhs(t, y, p: Params):
    S, N = y
    return [p.a * S + p.beta * N - p.gamma * S,
            p.b * N - p.beta * N + p.gamma * S]


def simulate(p: Params, t: np.ndarray, x0: float | None = None,
             n0: float = 1.0e5) -> dict:
    """Integrate from an initial stem-like fraction.

    ``x0 = None`` starts the population at the transition-only fixed point,
    which is the right control state for an unperturbed culture.
    """
    if x0 is None:
        x0 = p.x_star
    y0 = [x0 * n0, (1.0 - x0) * n0]
    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), y0, t_eval=t, args=(p,),
                    method="LSODA", rtol=1e-10, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    S, N = sol.y
    tot = S + N
    return {"t": t, "S": S, "N": N, "total": tot, "x": S / tot,
            "log_total": np.log(np.maximum(tot, 1e-300))}


def simulate_washout(p: Params, e, t_on: np.ndarray, t_off: np.ndarray,
                     x0: float | None = None, n0: float = 1.0e5) -> dict:
    """Expose, then withdraw the agent and keep watching.

    The result is a negative design finding, so it is worth stating why it must
    be: after withdrawal every arm returns to the SAME baseline parameters, and
    the matched arms are at the SAME fraction when the agent is removed. The
    post-washout fraction trajectory is therefore identical across mechanisms
    by construction, and a washout arm read as a marker fraction cannot
    discriminate. Only the count accumulated during exposure separates them.

    **This assumes the exposure effect is fully and instantly reversible**, so
    that (S, N) is the entire state carried across withdrawal. That assumption
    is what makes the arms coincide, and it is the interesting case to doubt: a
    heritable plastic change -- the thing EMD3 is ultimately about -- would
    leave beta elevated after withdrawal and the arms would separate. The
    negative result is therefore a statement about reversible exposure effects
    in a two-compartment model, not a general one about washout designs.
    """
    on = simulate(e.apply(p), t_on, x0=x0, n0=n0)
    off = simulate(p, t_off, x0=float(on["x"][-1]), n0=float(on["total"][-1]))
    return {"on": on, "off": off,
            "x_at_washout": float(on["x"][-1]),
            "log_total_at_washout": float(on["log_total"][-1]),
            "x_after": off["x"], "t_after": t_off}


def selection_only(p: Params, gap: float) -> Params:
    """Pure-selection structure: no state conversion, fitness gap free."""
    return replace(p, beta=0.0, gamma=0.0, r_S=p.r_N + gap, d_S=p.d_N)


def induction_only(p: Params, beta: float, gamma: float | None = None) -> Params:
    """Pure-induction structure: fitness parity, transitions free."""
    g = p.gamma if gamma is None else gamma
    return replace(p, beta=beta, gamma=g, r_S=p.r_N, d_S=p.d_N)


def observed_fraction(p: Params, t: np.ndarray, x0: float | None = None,
                      n_cells: int = 20000,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """x(t) as a flow-cytometry readout would report it.

    Binomial counting noise on ``n_cells`` events. This is the OPTIMISTIC noise
    model -- it ignores gating drift, antibody lot variation and well-to-well
    handling, all of which are larger than counting error in practice. If two
    models are indistinguishable under binomial noise alone, no real experiment
    will separate them.

    Nothing in the battery uses this: inference and design both run on
    ``inference.SIGMA_BIO``, replicate variability at 0.03 absolute. It is kept
    because it is the quantitative reason for that choice -- 20 000 events give
    a standard error near 0.003, an order of magnitude below the replicate term
    -- and ``test_binomial_counting_noise_is_not_the_limiting_term`` exercises
    it so the justification is checked rather than merely asserted.
    """
    x = simulate(p, t, x0=x0)["x"]
    if rng is None:
        return x
    return rng.binomial(n_cells, np.clip(x, 0.0, 1.0)) / n_cells


def control_fixed_point(p: Params) -> float:
    """Stationary stem-like fraction of the FULL system.

    ``x_star = beta/(beta+gamma)`` is only the fixed point when the two
    compartments have equal fitness. Real stem-like cells cycle more slowly than
    their differentiated progeny, so a - b is generally negative and the true
    stationary fraction sits below x_star. Starting a control culture at x_star
    makes it drift, and that drift is then easy to mistake for an effect.
    """
    from scipy.optimize import brentq

    def f(x):
        return ((p.a - p.gamma) * x + p.beta * (1.0 - x)
                - x * (p.b + (p.a - p.b) * x))

    lo, hi = 1e-9, 1.0 - 1e-9
    if f(lo) * f(hi) > 0:
        return float(np.clip(p.x_star, 0.0, 1.0))
    return float(brentq(f, lo, hi, xtol=1e-14))


def match_at(p: Params, channel: str, target_x: float, day: float,
             lo: float = 0.0, hi: float = 3.0) -> Exposure:
    """Find the exposure magnitude on one channel that hits ``target_x`` at
    ``day``.

    This is what makes the comparison fair. Two agents that produce the SAME
    marker fraction at the SAME endpoint -- one purely by induction, one purely
    by selection -- are exactly the pair a published endpoint measurement cannot
    tell apart, and they are what the rest of the module is about.
    """
    from scipy.optimize import brentq

    x0 = control_fixed_point(p)
    t = np.linspace(0.0, day, 400)

    def g(m):
        e = Exposure(**{channel: float(m)})
        return simulate(e.apply(p), t, x0=x0)["x"][-1] - target_x

    if g(lo) * g(hi) > 0:
        raise ValueError(f"target x={target_x} unreachable on channel {channel} "
                         f"in [{lo}, {hi}]")
    return Exposure(**{channel: float(brentq(g, lo, hi, xtol=1e-12))})
