"""Fit four models to synthetic marker fractions or to fractions and log counts.

The exposure can increase the stem-like net growth rate (sel), increase
dedifferentiation under equal fitness (ind), affect both processes (both),
or increase non-stem death (cyto). The measurements identify different
combinations of these rates.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .model import Exposure, Params, control_fixed_point, simulate

SIGMA_BIO = 0.030          # SD across biological replicates
N_REP = 3                  # replicates averaged per arm per timepoint


def se_mean(sigma: float, n_rep: int = N_REP) -> float:
    """Standard error of the mean of n_rep independent replicates."""
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
        # This structure fixes the two compartments at equal net growth.
        e = Exposure(d_beta=float(theta[0]))
        p = replace(p, r_S=p.r_N, d_S=p.d_N)
    elif structure == "both":
        e = Exposure(d_beta=float(theta[0]), d_fit=float(theta[1]))
    elif structure == "cyto":
        e = Exposure(kill_N=float(theta[0]))
    else:
        raise ValueError(structure)
    return simulate(e.apply(p), t, x0=x0)["x"]


INIT = {"sel": [0.1], "ind": [0.02], "both": [0.02, 0.1], "cyto": [0.05]}
STRUCTURES = ("sel", "ind", "both", "cyto")


def bounds(p: Params, structure: str) -> tuple[list, list]:
    """Constrain the additive fitness shift so stem-like proliferation stays nonnegative."""
    lo_fit = -float(p.r_S)
    return {"sel": ([lo_fit], [3.0]),
            "ind": ([0.0], [2.0]),
            "both": ([0.0, lo_fit], [2.0, 3.0]),
            "cyto": ([0.0], [3.0])}[structure]


def _aicc(aic: float, k: int, n: int) -> float:
    """Return the small-sample corrected AIC or infinity if n <= k + 1."""
    return aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)


def fit(p: Params, structure: str, t: np.ndarray, y: np.ndarray,
        x0: float, sigma: float = SIGMA_BIO, n_rep: int = N_REP) -> dict:
    sd = se_mean(sigma, n_rep)

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
    """Profile the exposure-induced change in the dedifferentiation rate (d_beta).

    The interval includes grid points within two log-likelihood units of the
    maximum. The induction model assumes equal fitness. The combined model
    refits the fitness difference at each grid point. If the interval reaches
    the upper grid limit, its reported width is a lower bound.
    """
    if grid is None:
        grid = (np.linspace(0.0, 0.12, 61) if structure == "ind"
                else np.linspace(0.0, 0.60, 151))
    best = fit(p, structure, t, y, x0, sigma, n_rep)
    sd = se_mean(sigma, n_rep)
    lo_fit, hi_fit = bounds(p, "sel")

    lls = []
    # Initialise each fitness fit at the optimum from the preceding grid point.
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
            # A profile ending at the grid boundary gives a lower bound on width.
            "right_censored": bool(hi >= grid[-1] - 1e-12)}


SIGMA_LOGN = 0.15          # SD of one replicate on the log-count scale


def _contrasts(n: int) -> np.ndarray:
    """Return n-1 orthonormal Helmert contrasts that remove a constant log-count offset."""
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
    """Fit fractions and log counts with unknown seeding density.

    Helmert contrasts remove the unknown log-count offset and preserve the
    variance of independent residuals. The joint likelihood thus uses n
    fraction measurements and n-1 independent count contrasts.
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
