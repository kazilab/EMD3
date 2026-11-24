"""Sensitivity of model comparisons to the six assigned baseline rates.

Each draw uses independent log-normal rate multipliers. Exposures are
matched to the same endpoint fraction relative to that draw's control,
subject to the fraction cap. Output percentiles describe variation under
these assigned perturbations.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import inference as inf
from .conditions import DESIGNS, MATCH_DAY, TARGET_X
from .design import N_REP as DESIGN_N_REP, SIGMA_BIO as DESIGN_SIGMA, Z as DESIGN_Z
from .model import (Params, control_fixed_point, match_at, simulate,
                    simulate_washout)

# Pointwise threshold for two independent replicate means.
DETECT_THRESHOLD = DESIGN_Z * DESIGN_SIGMA * np.sqrt(2.0 / DESIGN_N_REP)

# Assigned standard deviations for log-normal baseline rate multipliers.
FREE_SIGMA = {
    "r_S": 0.25, "d_S": 0.30, "r_N": 0.25, "d_N": 0.30,
    "beta": 0.50, "gamma": 0.40,
}

DEFAULT_N = 60
DEFAULT_SEED = 20260825
INNER_REPS = 8
GRID = np.linspace(0.0, 0.20, 101)

# Match each draw relative to its own control, up to the fraction cap.
TARGET_FOLD = TARGET_X / control_fixed_point(Params())
MAX_TARGET = 0.90


def sample_params(rng: np.random.Generator, base: Params | None = None) -> Params:
    base = base or Params()
    return replace(base, **{k: float(getattr(base, k) * np.exp(rng.normal(0.0, s)))
                            for k, s in FREE_SIGMA.items()})


def _fp(p: Params, x0: float, e, t: np.ndarray, rng) -> dict:
    """Summarise induction-model profile intervals and count intervals truncated at the grid boundary."""
    excl, widths, points, cens = 0, [], [], 0
    for _ in range(INNER_REPS):
        y = inf.observe(e.apply(p), t, x0, rng=rng)
        r = inf.beta_interval(p, t, y, x0, grid=GRID)
        excl += r["excludes_zero"]
        cens += r["right_censored"]
        widths.append(r["width"]); points.append(r["point"])
    return {"ci_excl": excl / INNER_REPS,
            "width": float(np.median(widths)),
            "point": float(np.median(points)),
            "n_censored": cens}


def run_ensemble(n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                 base: Params | None = None) -> dict:
    """Evaluate model checks and follow-up separation across matched parameter draws."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    base = base or Params()

    hits = {k: 0 for k in ("V5", "V6", "V8", "V12")}
    detect, widths_s, widths_e, n_ok, n_unmatched = [], [], [], 0, 0
    n_v6 = n_v6_censored = 0
    t_on = np.linspace(0.0, MATCH_DAY, 201)
    t_off = np.linspace(0.0, 60.0, 301)

    for _ in range(n):
        p = sample_params(rng, base)
        try:
            x0 = control_fixed_point(p)
            target = min(TARGET_FOLD * x0, MAX_TARGET)
            truths = {c: match_at(p, c, target, MATCH_DAY, hi=20.0)
                      for c in ("d_beta", "d_fit", "kill_N")}
        except Exception:
            n_unmatched += 1
            continue

        try:
            sel = _fp(p, x0, truths["d_fit"], DESIGNS["short"], rng)
            sel_e = _fp(p, x0, truths["d_fit"], DESIGNS["extended"], rng)
            cyt = _fp(p, x0, truths["kill_N"], DESIGNS["short"], rng)
            cyt_e = _fp(p, x0, truths["kill_N"], DESIGNS["extended"], rng)

            # Classify synthetic datasets using both fractions and cell counts.
            picks_ind = picks_cyto = picks_sel = 0
            for _ in range(INNER_REPS):
                d = inf.observe_with_counts(truths["kill_N"].apply(p),
                                            DESIGNS["extended"], x0, rng=rng)
                b = inf.compare_with_counts(p, DESIGNS["extended"], d, x0)["best"]
                picks_ind += b == "ind"
                picks_cyto += b == "cyto"
                ds = inf.observe_with_counts(truths["d_fit"].apply(p),
                                             DESIGNS["extended"], x0, rng=rng)
                picks_sel += inf.compare_with_counts(
                    p, DESIGNS["extended"], ds, x0)["best"] == "sel"

            xs = np.vstack([simulate_washout(p, e, t_on, t_off, x0=x0)["x_after"]
                            for e in truths.values()])
            gap = float(np.max(xs.max(axis=0) - xs.min(axis=0)))

            # Find the first post-match separation on the follow-up grid.
            tl = np.linspace(0.0, 240.0, 961)
            xi = simulate(truths["d_beta"].apply(p), tl, x0=x0)["x"]
            xsl = simulate(truths["d_fit"].apply(p), tl, x0=x0)["x"]
            sep = np.abs(xi - xsl) > DETECT_THRESHOLD
            past = tl[sep & (tl > MATCH_DAY)]
        except Exception:
            continue

        n_ok += 1
        hits["V5"] += sel["ci_excl"] > 0.8
        # Exclude draws with any right-censored interval from the width comparison.
        if cyt["n_censored"] == 0 and cyt_e["n_censored"] == 0:
            n_v6 += 1
            hits["V6"] += (cyt_e["width"] < cyt["width"]
                           and cyt_e["point"] > cyt["point"])
            widths_s.append(cyt["width"]); widths_e.append(cyt_e["width"])
        else:
            n_v6_censored += 1
        # With eight datasets per draw, the nominal 5% threshold permits zero induction-only calls.
        hits["V8"] += (picks_ind == 0
                       and (picks_cyto / INNER_REPS) > 0.8
                       and (picks_sel / INNER_REPS) > 0.8)
        hits["V12"] += gap < 1e-6
        detect.append(float(past[0] - MATCH_DAY) if len(past) else np.nan)

    if not n_ok:
        raise RuntimeError("no ensemble draw completed")

    d = np.asarray(detect, dtype=float)
    d = d[np.isfinite(d)]
    band = lambda a: dict(zip(("p5", "p50", "p95"),
                              (float(v) for v in np.percentile(a, [5, 50, 95]))))
    # V6 uses the number of uncensored draws as its denominator.
    denom = {k: (n_v6 if k == "V6" else n_ok) for k in hits}
    return {
        "n_requested": n, "n_ok": n_ok, "n_unmatched": n_unmatched,
        "target_fold_over_control": float(TARGET_FOLD),
        "target_fraction_cap": float(MAX_TARGET),
        "detect_search_horizon_days": 240.0,
        "audited_claims": ["V5", "V6", "V8", "V12", "follow-up separability"],
        "not_audited_claims": ["combined-structure profiles", "sorting power",
                               "Boolean layer"],
        "seed": seed,
        "audit": {k: (v / denom[k] if denom[k] else float("nan"))
                  for k, v in hits.items()},
        "audit_denominator": denom,
        "n_v6_draws_dropped_for_censoring": n_v6_censored,
        "detect_threshold": float(DETECT_THRESHOLD),
        "detect_days_past_endpoint": band(d) if len(d) else None,
        "frac_draws_separable_within_240d": float(len(d) / n_ok),
        "cyto_ci_width": ({"short": band(np.asarray(widths_s)),
                           "extended": band(np.asarray(widths_e))}
                          if widths_s else None),
        "not_audited": {
            "V7": "The fraction depends on net growth through a - b.",
            "V3/V15": "The Boolean rules and initial TERT state determine this readout.",
        },
        "interpretation": (
            "Percentiles describe sensitivity to assigned baseline rate variation. "
            "Exposure effects are matched to a common endpoint in each draw."),
    }
