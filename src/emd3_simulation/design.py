"""Compare follow-up duration and residual marker fractions in a two-state model.

The induction and selection scenarios have the same marker fraction at the
matching time. Subsequent fraction differences are evaluated under assigned
biological replicate variability. Sorting power uses the difference between
the two scenarios' high-minus-low residual-fraction contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .model import (Exposure, Params, control_fixed_point, match_at, simulate)

SIGMA_BIO = 0.030          # SD of the measured fraction between replicates
N_REP = 3                  # replicates per arm per timepoint
Z = 1.96


@dataclass(frozen=True)
class MatchedPair:
    """Induction and selection exposures matched to the same endpoint marker fraction."""
    p: Params
    induction: Exposure
    selection: Exposure
    target_x: float
    match_day: float
    x0: float


def matched_pair(p: Params, target_x: float = 0.20,
                 match_day: float = 14.0) -> MatchedPair:
    return MatchedPair(p,
                       match_at(p, "d_beta", target_x, match_day),
                       match_at(p, "d_fit", target_x, match_day),
                       target_x, match_day, control_fixed_point(p))


def separation_over_time(mp: MatchedPair, t: np.ndarray) -> dict:
    xi = simulate(mp.induction.apply(mp.p), t, x0=mp.x0)["x"]
    xs = simulate(mp.selection.apply(mp.p), t, x0=mp.x0)["x"]
    return {"t": t, "induction": xi, "selection": xs, "diff": np.abs(xi - xs)}


def detection_day(mp: MatchedPair, sigma: float = SIGMA_BIO, n_rep: int = N_REP,
                  t_max: float = 200.0) -> float | None:
    """First sampled day after matching where separation exceeds 1.96 standard errors.

    Return None if the threshold is not crossed by t_max. Each time point
    uses two independent means with n_rep replicates per arm.
    """
    t = np.linspace(0.0, t_max, 2001)
    sep = separation_over_time(mp, t)
    se = sigma * np.sqrt(2.0 / n_rep)
    hit = np.flatnonzero((sep["diff"] > Z * se) & (t > mp.match_day))
    return float(t[hit[0]]) if len(hit) else None


def detection_surface(mp: MatchedPair, sigmas: np.ndarray,
                      reps: np.ndarray, t_max: float = 200.0) -> np.ndarray:
    """Days of follow-up required, over measurement precision and replication."""
    out = np.full((len(sigmas), len(reps)), np.nan)
    for i, s in enumerate(sigmas):
        for j, n in enumerate(reps):
            d = detection_day(mp, sigma=float(s), n_rep=int(n), t_max=t_max)
            out[i, j] = np.nan if d is None else d
    return out


def purity_titration(mp: MatchedPair, purities: np.ndarray,
                     t: np.ndarray) -> dict:
    """Simulate each matched mechanism from the specified initial marker fractions."""
    ind = mp.induction.apply(mp.p)
    sel = mp.selection.apply(mp.p)
    out = {"purities": purities, "t": t, "induction": [], "selection": []}
    for eps in purities:
        out["induction"].append(simulate(ind, t, x0=float(eps))["x"])
        out["selection"].append(simulate(sel, t, x0=float(eps))["x"])
    out["induction"] = np.asarray(out["induction"])
    out["selection"] = np.asarray(out["selection"])
    return out


def purity_dependence(mp: MatchedPair, eps_lo: float = 0.001,
                      eps_hi: float = 0.020, day: float = 10.0) -> dict:
    """Compare the effect of the initial marker fraction on each simulated endpoint.

    Each mechanism has a high-to-low ratio and a high-minus-low difference.
    The four-mean contrast compares these differences across mechanisms.
    """
    t = np.linspace(0.0, day, 400)
    res = {}
    for name, e in (("induction", mp.induction), ("selection", mp.selection)):
        q = e.apply(mp.p)
        lo = simulate(q, t, x0=eps_lo)["x"][-1]
        hi = simulate(q, t, x0=eps_hi)["x"][-1]
        res[name] = {"x_lo": float(lo), "x_hi": float(hi),
                     "ratio": float(hi / max(lo, 1e-12)),
                     "diff": float(hi - lo)}
    res["eps_lo"], res["eps_hi"], res["day"] = eps_lo, eps_hi, day
    res["purity_fold"] = eps_hi / eps_lo
    # The contrast combines four independent means with equal replicate variance.
    se = SIGMA_BIO * np.sqrt(4.0 / N_REP)
    res["contrast"] = float(abs(res["selection"]["diff"]
                                - res["induction"]["diff"]))
    res["noise_floor"] = float(Z * se)
    res["se_contrast"] = float(se)
    res["separable"] = bool(res["contrast"] > res["noise_floor"])
    # Power includes both tails. The directional exceedance probability is reported separately.
    z_hi = (res["noise_floor"] - res["contrast"]) / se
    z_lo = (-res["noise_floor"] - res["contrast"]) / se
    res["power_upper_tail"] = float(norm.sf(z_hi))
    res["power"] = float(norm.sf(z_hi) + norm.cdf(z_lo))
    res["n_rep"] = int(N_REP)
    return res


def power_vs_replication(mp: MatchedPair, folds=(20, 50, 100, 200),
                         reps=(3, 4, 6, 8), eps_lo: float = 0.001,
                         day: float = 7.0) -> dict:
    """Compute two-sided Gaussian contrast power across residual fractions and replicate counts."""
    out = {"folds": list(folds), "reps": list(reps), "eps_lo": eps_lo,
           "day": day, "sigma": SIGMA_BIO, "grid": {}}
    for fo in folds:
        r = purity_dependence(mp, eps_lo=eps_lo, eps_hi=eps_lo * fo, day=day)
        row = {}
        for n in reps:
            se = SIGMA_BIO * np.sqrt(4.0 / n)
            crit = Z * se
            row[int(n)] = float(norm.sf((crit - r["contrast"]) / se)
                                + norm.cdf((-crit - r["contrast"]) / se))
        out["grid"][int(fo)] = {"contrast": r["contrast"],
                                "eps_hi": r["eps_hi"], "power_by_n_rep": row}
    return out


def purity_fold_for_power(mp: MatchedPair, target_power: float = 0.80,
                          eps_lo: float = 0.001, day: float = 7.0,
                          hi_max: float = 0.60) -> dict:
    """Find the residual-fraction separation for the target upper-tail power.

    The returned power includes both tails of the Gaussian test.
    """
    from scipy.optimize import brentq
    se = SIGMA_BIO * np.sqrt(4.0 / N_REP)
    crit = Z * se

    def g(hi):
        r = purity_dependence(mp, eps_lo=eps_lo, eps_hi=float(hi), day=day)
        return float(norm.sf((crit - r["contrast"]) / se)) - target_power

    lo_b = eps_lo * 1.001
    if g(hi_max) < 0:
        return {"target_power": target_power, "feasible": False,
                "eps_hi": None, "fold": None, "day": day}
    hi = float(brentq(g, lo_b, hi_max, xtol=1e-6))
    r = purity_dependence(mp, eps_lo=eps_lo, eps_hi=hi, day=day)
    return {"target_power": target_power, "feasible": True,
            "eps_lo": eps_lo, "eps_hi": hi, "fold": hi / eps_lo,
            "contrast": r["contrast"], "power": r["power"], "day": day}
