"""Two-state population model of cell-state conversion and selection.

S and N denote stem-like and non-stem cells. Their net growth rates are
a = r_S - d_S and b = r_N - d_N. Conversion from N to S occurs at rate beta,
and conversion from S to N occurs at rate gamma. The stem-like fraction
x = S/(S+N) satisfies

    dx/dt = beta*(1-x) - gamma*x + (a-b)*x*(1-x).

Both conversion and differential net growth can change the marker fraction.
With equal net growth rates its equilibrium is beta/(beta+gamma). With
both conversion rates zero, the fraction follows logistic selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    """Assigned proliferation, death and transition rates in inverse days."""

    r_S: float = 0.55        # stem-like proliferation
    d_S: float = 0.05        # stem-like death
    r_N: float = 0.60        # non-stem proliferation
    d_N: float = 0.05        # non-stem death

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
        """Difference in net growth rates between stem-like and non-stem cells."""
        return self.a - self.b

    @property
    def x_star(self) -> float:
        """Transition-only fixed point, valid when a == b."""
        return self.beta / (self.beta + self.gamma)


@dataclass(frozen=True)
class Exposure:
    """Additive exposure effects on proliferation, death and the two transition rates."""
    d_beta: float = 0.0       # Additive change in dedifferentiation (induction)
    d_fit: float = 0.0        # Additive change in (a - b), via r_S (selection)
    kill_N: float = 0.0       # Additive death rate on NSCC only (cytotoxicity)
    kill_S: float = 0.0       # Additive death rate on CSC only
    d_gamma: float = 0.0      # Additive change in differentiation

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
    """Integrate from stem-like fraction x0 and total count n0.

    The default x0 is the transition equilibrium for equal net growth rates.
    Use control_fixed_point(p) for the stationary fraction at unequal rates.
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
    """Return trajectories during exposure and after immediate restoration of baseline rates.

    Both cell counts carry over at withdrawal. Arms matched on fraction then
    share a subsequent fraction trajectory, while their total counts can differ.
    """
    on = simulate(e.apply(p), t_on, x0=x0, n0=n0)
    off = simulate(p, t_off, x0=float(on["x"][-1]), n0=float(on["total"][-1]))
    return {"on": on, "off": off,
            "x_at_washout": float(on["x"][-1]),
            "log_total_at_washout": float(on["log_total"][-1]),
            "x_after": off["x"], "t_after": t_off}


def selection_only(p: Params, gap: float) -> Params:
    """Set conversion rates to zero and assign the net growth difference."""
    return replace(p, beta=0.0, gamma=0.0, r_S=p.r_N + gap, d_S=p.d_N)


def induction_only(p: Params, beta: float, gamma: float | None = None) -> Params:
    """Set net growth rates equal and assign transition rates."""
    g = p.gamma if gamma is None else gamma
    return replace(p, beta=beta, gamma=g, r_S=p.r_N, d_S=p.d_N)


def observed_fraction(p: Params, t: np.ndarray, x0: float | None = None,
                      n_cells: int = 20000,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the model fraction or sample a binomial fraction from n_cells events."""
    x = simulate(p, t, x0=x0)["x"]
    if rng is None:
        return x
    return rng.binomial(n_cells, np.clip(x, 0.0, 1.0)) / n_cells


def control_fixed_point(p: Params) -> float:
    """Solve for the stationary fraction including differential net growth."""
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
    """Find the exposure change yielding the target fraction on the specified day."""
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
