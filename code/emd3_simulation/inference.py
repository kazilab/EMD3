"""Fitting the competing structures, and what happens when the truth is neither.

Three structures are fitted to the same observed marker-fraction time course:

    M_sel   beta fixed at 0, fitness gap free          (selection only)
    M_ind   fitness parity enforced, beta free         (induction only)
    M_both  both free
    M_cyto  differential DEATH free, no conversion and no growth advantage

M_cyto is not decoration. An agent that kills the non-stem compartment
preferentially raises the marker fraction without any cell changing state and
without either compartment dividing faster. If that structure is absent from the
comparison, the cytotoxic effect is absorbed by whichever structure is present,
and the answer is whatever the model set allowed.

The estimate of interest is ``d_beta`` -- the exposure-driven change in the
dedifferentiation rate -- and the manuscript's test is whether its interval
excludes zero. The purpose of this module is to show how easily that test
returns a confident, wrong answer.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .model import Exposure, Params, control_fixed_point, simulate

SIGMA_BIO = 0.030          # SD of ONE biological replicate
N_REP = 3                  # replicates averaged per arm per timepoint


def se_mean(sigma: float, n_rep: int = N_REP) -> float:
    """Standard error of a replicate MEAN -- the quantity actually observed.

    ``observe`` returns an average of ``n_rep`` replicates, so its noise is
    sigma/sqrt(n_rep), not sigma. An earlier release generated data at that SE
    but divided the fit residuals by the per-replicate sigma, overstating the
    noise by sqrt(3) and flattening the log-likelihood surface. Every profile
    interval was correspondingly too wide: correcting it moves the gap-free
    detection of a real induction agent at the short design from 30% to 82%.
    The false-positive rates under the induction-only structure are unaffected
    (100% either way), because there the misspecification, not the noise, is
    what excludes zero.

    Both halves now go through this function so they cannot drift apart again.
    """
    return float(sigma) / np.sqrt(float(n_rep))



def observe(p_exposed: Params, t: np.ndarray, x0: float,
            sigma: float = SIGMA_BIO, n_rep: int = N_REP,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Replicate-averaged marker fractions at the sampled days."""
    x = simulate(p_exposed, t, x0=x0)["x"]
    if rng is None:
        return x
    return x + rng.normal(0.0, se_mean(sigma, n_rep), size=x.shape)


def _predict(p: Params, structure: str, theta: np.ndarray,
             t: np.ndarray, x0: float) -> np.ndarray:
    if structure == "sel":
        e = Exposure(d_fit=float(theta[0]))
    elif structure == "ind":
        # fitness parity is ENFORCED, not fitted: this structure asserts that
        # the two compartments grow alike and all change is conversion
        e = Exposure(d_beta=float(theta[0]))
        p = replace(p, r_S=p.r_N, d_S=p.d_N)
    elif structure == "both":
        e = Exposure(d_beta=float(theta[0]), d_fit=float(theta[1]))
    elif structure == "cyto":
        # differential killing only: no conversion, no proliferative advantage
        e = Exposure(kill_N=float(theta[0]))
    else:
        raise ValueError(structure)
    return simulate(e.apply(p), t, x0=x0)["x"]


INIT = {"sel": [0.1], "ind": [0.02], "both": [0.02, 0.1], "cyto": [0.05]}
STRUCTURES = ("sel", "ind", "both", "cyto")


def bounds(p: Params, structure: str) -> tuple[list, list]:
    """Parameter bounds, floored so no fitted rate can go negative.

    ``d_fit`` is an ADDITIVE shift on r_S, so a lower bound of -r_S is the point
    at which stem-like proliferation reaches zero; anything below that is a
    negative division rate. The bound was a flat -1.0, which never bound at
    these baselines but would have let an optimiser wander into unphysical
    territory under other rates -- and the ensemble perturbs the rates.
    """
    lo_fit = -float(p.r_S)
    return {"sel": ([lo_fit], [3.0]),
            "ind": ([0.0], [2.0]),
            "both": ([0.0, lo_fit], [2.0, 3.0]),
            "cyto": ([0.0], [3.0])}[structure]


def _aicc(aic: float, k: int, n: int) -> float:
    """Small-sample corrected AIC.

    These designs have 5-7 timepoints against 1-2 free parameters, which is far
    inside the regime (n/k < 40) where plain AIC under-penalises the larger
    model: at n = 5, k = 2 the correction is +6 on top of AIC's penalty of 4.
    Using AIC here would let the two-parameter structure win on noise.
    """
    return aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)


def fit(p: Params, structure: str, t: np.ndarray, y: np.ndarray,
        x0: float, sigma: float = SIGMA_BIO, n_rep: int = N_REP) -> dict:
    sd = se_mean(sigma, n_rep)          # y is a replicate MEAN

    def resid(theta):
        try:
            return (_predict(p, structure, theta, t, x0) - y) / sd
        except Exception:
            return np.full_like(y, 1e3)

    lo, hi = bounds(p, structure)
    sol = least_squares(resid, INIT[structure], bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12)
    k = len(sol.x)
    rss = float(np.sum(sol.fun ** 2))
    n = len(y)
    # Gaussian log-likelihood with the observation SE known
    ll = -0.5 * rss - n * np.log(sd * np.sqrt(2 * np.pi))
    aic = 2 * k - 2 * ll
    return {"structure": structure, "theta": sol.x, "rss": rss,
            "loglik": ll, "k": k, "n": n, "aic": aic, "aicc": _aicc(aic, k, n),
            "d_beta": float(sol.x[0]) if structure in ("ind", "both") else 0.0,
            "d_fit": (float(sol.x[0]) if structure == "sel"
                      else (float(sol.x[1]) if structure == "both" else 0.0))}


def compare(p: Params, t: np.ndarray, y: np.ndarray, x0: float,
            sigma: float = SIGMA_BIO, n_rep: int = N_REP) -> dict:
    fits = {s: fit(p, s, t, y, x0, sigma, n_rep) for s in STRUCTURES}
    best = min(fits.values(), key=lambda f: f["aicc"])
    return {"fits": fits, "best": best["structure"],
            "d_aicc": {s: f["aicc"] - best["aicc"] for s, f in fits.items()}}


def beta_interval(p: Params, t: np.ndarray, y: np.ndarray, x0: float,
                  sigma: float = SIGMA_BIO, structure: str = "ind",
                  grid: np.ndarray | None = None, n_rep: int = N_REP) -> dict:
    """Profile-likelihood interval on d_beta, under ``ind`` or ``both``.

    A 2-unit drop in log-likelihood from the maximum is the usual 95% profile
    interval for one parameter. The question the manuscript poses is whether
    this interval excludes zero -- so the interval is what has to be reported,
    not the point estimate.

    **Which structure is profiled is the whole result.** Under ``ind`` the fit
    enforces fitness parity, so every route to a rising fraction has to be
    spent on d_beta and the interval excludes zero for agents that carry no
    plasticity at all. Under ``both`` the fitness gap is free, and the same
    fractions put the interval back across zero for those agents. The false
    positive belongs to the induction-only STRUCTURE, not to fraction data as
    such -- see V5 and V16. Profiling ``both`` costs power rather than validity,
    which is the honest argument for absolute counts.

    The default grid is structure-dependent: ``both`` leaves d_beta only weakly
    identified from fractions alone, and its upper limit runs past the 0.12 that
    suffices for ``ind``. On too narrow a grid the interval is right-censored
    and its width is meaningless, so ``right_censored`` is returned and the
    battery asserts on it.
    """
    if grid is None:
        grid = (np.linspace(0.0, 0.12, 61) if structure == "ind"
                else np.linspace(0.0, 0.60, 151))
    best = fit(p, structure, t, y, x0, sigma, n_rep)
    sd = se_mean(sigma, n_rep)
    lo_fit, hi_fit = bounds(p, "sel")

    lls = []
    # The nested d_fit optimum moves smoothly along the d_beta grid, so each
    # inner fit is warm-started from the previous one. Cold-starting every point
    # from a fixed guess made the `both` profile ~60x more expensive than the
    # `ind` one and put it out of reach of the per-draw battery.
    warm = [0.1]
    for b in grid:
        if structure == "ind":
            theta = np.array([b])
            ll = -0.5 * float(np.sum(((_predict(p, "ind", theta, t, x0) - y)
                                      / sd) ** 2))
        else:
            def resid(v):
                return (_predict(p, "both", np.array([b, v[0]]), t, x0) - y) / sd
            sol = least_squares(resid, warm, bounds=(lo_fit, hi_fit),
                                xtol=1e-10, ftol=1e-10)
            warm = [float(np.clip(sol.x[0], lo_fit[0] + 1e-9, hi_fit[0] - 1e-9))]
            ll = -0.5 * float(np.sum(sol.fun ** 2))
        lls.append(ll)
    lls = np.asarray(lls)
    ok = lls >= lls.max() - 2.0
    lo, hi = float(grid[ok].min()), float(grid[ok].max())
    return {"grid": grid, "loglik": lls, "admissible": grid[ok],
            "lo": lo, "hi": hi, "width": hi - lo,
            "point": best["d_beta"],
            "excludes_zero": bool(lo > grid[0] + 1e-12),
            # Right-censoring would make the width meaningless. Currently never
            # triggers (checked across every truth and design), but a future
            # change to the baseline rates could push a fit off the grid.
            "right_censored": bool(hi >= grid[-1] - 1e-12)}


# --- count-augmented inference --------------------------------------------
# From marker fractions alone, "the agent gave stem-like cells a growth
# advantage" and "the agent killed non-stem cells" are the SAME hypothesis:
#
#     a - b = (r_S - d_S) - (r_N - d_N)
#
# and raising d_N is algebraically identical to raising r_S. No amount of
# fraction data at any number of timepoints can separate them, because they are
# not two models -- they are one model written twice.
#
# Total cell counts break the degeneracy exactly, and in the obvious direction:
# a growth advantage makes the population larger, differential killing makes it
# smaller. This is the manuscript's point that "absolute cell counts ... are
# substantially more informative than relative frequencies alone", made
# quantitative.

SIGMA_LOGN = 0.15          # CV on ONE total-cell-count replicate; generous
                           # for a haemocytometer


def _contrasts(n: int) -> np.ndarray:
    """(n-1) x n Helmert contrasts: orthonormal rows, each orthogonal to 1.

    Row k is (1,...,1, -k, 0,...,0)/sqrt(k(k+1)) with k leading ones. Because
    every row sums to zero, ``H @ v`` is invariant to adding a constant to v --
    which is exactly "the seeding density is unknown" -- and because the rows
    are orthonormal, ``H @ noise`` has covariance se^2 I when noise does. That
    is what makes the count residuals genuinely independent and correctly
    scaled, rather than merely centred.
    """
    H = np.zeros((n - 1, n))
    for k in range(1, n):
        H[k - 1, :k] = 1.0
        H[k - 1, k] = -float(k)
        H[k - 1] /= np.sqrt(k * (k + 1.0))
    return H


def observe_with_counts(p_exposed: Params, t: np.ndarray, x0: float,
                        sigma: float = SIGMA_BIO, sigma_n: float = SIGMA_LOGN,
                        n_rep: int = N_REP,
                        rng: np.random.Generator | None = None) -> dict:
    sim = simulate(p_exposed, t, x0=x0)
    x, logn = sim["x"], sim["log_total"]
    if rng is not None:
        x = x + rng.normal(0.0, se_mean(sigma, n_rep), size=x.shape)
        logn = logn + rng.normal(0.0, se_mean(sigma_n, n_rep), size=logn.shape)
    return {"x": x, "log_total": logn}


def _predict_both(p: Params, structure: str, theta: np.ndarray,
                  t: np.ndarray, x0: float) -> tuple[np.ndarray, np.ndarray]:
    if structure == "sel":
        e, q = Exposure(d_fit=float(theta[0])), p
    elif structure == "ind":
        e, q = Exposure(d_beta=float(theta[0])), replace(p, r_S=p.r_N, d_S=p.d_N)
    elif structure == "both":
        e, q = Exposure(d_beta=float(theta[0]), d_fit=float(theta[1])), p
    elif structure == "cyto":
        e, q = Exposure(kill_N=float(theta[0])), p
    else:
        raise ValueError(structure)
    sim = simulate(e.apply(q), t, x0=x0)
    return sim["x"], sim["log_total"]


def fit_with_counts(p: Params, structure: str, t: np.ndarray, data: dict,
                    x0: float, sigma: float = SIGMA_BIO,
                    sigma_n: float = SIGMA_LOGN, n_rep: int = N_REP) -> dict:
    """Fit fractions and log counts jointly, with the seeding density unknown.

    The seeding density is a nuisance: the fit must not be able to buy agreement
    by rescaling it. An earlier release enforced that by differencing every log
    count against the FIRST timepoint -- which is a noisy observation, not a
    known constant. That injects -eps_0 into every differenced residual, so the
    residuals carry a correlation of exactly 0.5 with one another and a variance
    of 2*se^2, and the likelihood treated them as independent with variance
    sigma_n^2. Three errors at once: wrong correlation, wrong variance, and the
    per-replicate sigma where the SE of a mean belonged.

    The fix is to project the offset out onto an ORTHONORMAL contrast basis.
    Mean-centring alone is not enough: the centred vector has covariance
    se^2 (I - J/n), so its components stay correlated and each carries variance
    se^2 (1 - 1/n). Taking n-1 of them and calling them independent leaves the
    likelihood mis-scaled by (n-1)/n -- small, but the same kind of error as the
    one being fixed. ``_contrasts(n)`` returns an (n-1) x n matrix H with
    orthonormal rows spanning the complement of the all-ones vector, so H kills
    any constant offset exactly and H(residual) has covariance se^2 I. The count
    block then contributes exactly ``len(t) - 1`` independent observations, and
    that is what AICc is told.
    """
    sd_x = se_mean(sigma, n_rep)
    sd_n = se_mean(sigma_n, n_rep)
    y_logn = data["log_total"]
    H = _contrasts(len(t))
    n_cnt = H.shape[0]                      # = len(t) - 1

    def resid(theta):
        try:
            xp, lp = _predict_both(p, structure, theta, t, x0)
        except Exception:
            return np.full(len(t) + n_cnt, 1e3)
        # H annihilates any additive offset, so the unknown seeding density
        # drops out exactly and the retained directions are uncorrelated
        return np.concatenate([(xp - data["x"]) / sd_x,
                               (H @ (lp - y_logn)) / sd_n])

    lo, hi = bounds(p, structure)
    sol = least_squares(resid, INIT[structure], bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12)
    k = len(sol.x)
    rss = float(np.sum(sol.fun ** 2))
    n = len(t) + n_cnt
    ll = -0.5 * rss
    aic = 2 * k - 2 * ll
    return {"structure": structure, "theta": sol.x, "rss": rss, "loglik": ll,
            "k": k, "n": n, "aic": aic, "aicc": _aicc(aic, k, n),
            "d_beta": float(sol.x[0]) if structure in ("ind", "both") else 0.0}


def compare_with_counts(p: Params, t: np.ndarray, data: dict, x0: float,
                        sigma: float = SIGMA_BIO,
                        sigma_n: float = SIGMA_LOGN, n_rep: int = N_REP) -> dict:
    fits = {s: fit_with_counts(p, s, t, data, x0, sigma, sigma_n, n_rep)
            for s in STRUCTURES}
    best = min(fits.values(), key=lambda f: f["aicc"])
    return {"fits": fits, "best": best["structure"],
            "d_aicc": {s: f["aicc"] - best["aicc"] for s, f in fits.items()}}
