"""What experiment separates induction from selection, and at what cost.

Two designs are evaluated against each other:

1. **Watch longer.** Keep an unsorted culture going past the usual endpoint and
   wait for the two mechanisms' asymptotes to separate. Cheap to describe,
   expensive to run, and the separation accrues slowly.

2. **Titrate the sorting purity.** Sort out the marker-positive cells to two
   different residual purities and follow re-emergence in each. Selection
   re-grows FROM the residue, so its trajectory scales with what was left
   behind. Induction manufactures new stem-like cells at a rate that does not
   care what the residue was. The discriminating quantity is therefore the
   RATIO of the two re-emergence trajectories, and it needs no long follow-up.

The noise model matters more than the mechanism here, so it is stated
explicitly. Marker-fraction measurements are not limited by counting error --
20 000 flow events give a standard error near 0.003 on a fraction of 0.2, which
would make almost anything significant. They are limited by biological
replicate variability: gating drift, antibody lot, passage number, plating
density. ``sigma_bio`` below is that term, and 0.03 absolute is a realistic
rather than a generous value.
(``model.observed_fraction`` implements the counting-error model, and
``test_binomial_counting_noise_is_not_the_limiting_term`` is what justifies
using this one instead.)

Each design's verdict is a contrast against a noise floor, and the floor has to
match the arithmetic of the statistic being tested -- a two-arm contrast and a
difference of differences do not carry the same standard error. See
``purity_dependence``.
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
    """An induction arm and a selection arm producing the same marker fraction
    at the same endpoint -- the pair an endpoint study cannot tell apart."""
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
    """First day on which the two matched arms are distinguishable.

    Two-sample comparison at each timepoint, ``n_rep`` replicates per arm.
    Returns None if they never separate within ``t_max``.
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


# --- design 2: purity titration -------------------------------------------

def purity_titration(mp: MatchedPair, purities: np.ndarray,
                     t: np.ndarray) -> dict:
    """Sort out the marker-positive cells to several residual purities, then
    follow re-emergence under each mechanism.

    Under selection the stem-like compartment can only re-grow from cells that
    survived the sort, so x(t) inherits the residue. Under induction it is
    manufactured from the non-stem compartment at a rate set by beta, which the
    sort did not touch.
    """
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
    """The discriminating statistic: how much does re-emergence depend on the
    residue left by the sort?

    Reported as a RATIO of the two sorted arms at a fixed readout day. A ratio
    near 1 means the trajectory did not care about the residue (induction); a
    ratio well above 1 means it did (selection).
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
    # A design succeeds when the two mechanisms' residue-dependences are
    # themselves separable. The statistic is a DIFFERENCE OF DIFFERENCES: two
    # sorted arms per mechanism, two mechanisms, so four independently measured
    # means. Its standard error is sigma*sqrt(4/n_rep), not the sigma*sqrt(2/
    # n_rep) of a single two-arm contrast -- an earlier release used the latter
    # and understated the noise floor by a factor of sqrt(2) (0.048 against
    # 0.068). The verdicts are unchanged, because the 20x design fails by a
    # wide margin under either threshold and the 100x design clears both, but
    # the quoted floor was the wrong number and the 100x margin is thinner than
    # it looked.
    se = SIGMA_BIO * np.sqrt(4.0 / N_REP)
    res["contrast"] = float(abs(res["selection"]["diff"]
                                - res["induction"]["diff"]))
    res["noise_floor"] = float(Z * se)
    res["se_contrast"] = float(se)
    res["separable"] = bool(res["contrast"] > res["noise_floor"])
    # "separable" states only that the TRUE contrast exceeds the critical
    # value, which at the boundary corresponds to 50% power. Power is the
    # quantity a design decision needs, so both are reported.
    #
    # ``power`` is the conventional two-sided 5% test: reject when the
    # estimated contrast falls in either tail. ``power_upper_tail`` is the
    # directional exceedance probability alone. The two differ materially only
    # where power is low -- 7.4% against 6.6% at a 20-fold separation, and
    # indistinguishable at 100-fold.
    z_hi = (res["noise_floor"] - res["contrast"]) / se
    z_lo = (-res["noise_floor"] - res["contrast"]) / se
    res["power_upper_tail"] = float(norm.sf(z_hi))
    res["power"] = float(norm.sf(z_hi) + norm.cdf(z_lo))
    res["n_rep"] = int(N_REP)
    return res


def power_vs_replication(mp: MatchedPair, folds=(20, 50, 100, 200),
                         reps=(3, 4, 6, 8), eps_lo: float = 0.001,
                         day: float = 7.0) -> dict:
    """Sorting-contrast power over purity separation AND replication.

    Power here is governed by two quantities the experimenter controls: how far
    apart the two sorted residual fractions are, and how many replicates each
    arm carries. Reporting one separation at one replicate count states a
    property of that combination, not of the design. At three replicates a
    100-fold separation reaches 0.56; holding the separation fixed and moving
    to six replicates reaches 0.84.

    The contrast itself is noise-free and depends only on the separation, so
    replication enters solely through the standard error sigma*sqrt(4/n_rep).
    """
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
    """Smallest residual-purity separation reaching ``target_power``.

    The build's design output should be a number an experimenter can act on. A
    verdict of "100x works" is not that when 100x carries 56% power; this
    returns the separation that actually does, and the purity the dirty arm
    would have to sit at to achieve it.
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
