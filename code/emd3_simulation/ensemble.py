"""Uncertainty propagation for the EMD3 build.

The baseline rates are plausible values for a mammary epithelial line, not
measurements, so a point estimate of "how many days of follow-up you need"
states as fact something nothing here constrains. This module perturbs the six
baseline rates and re-scores the claims that can move.

What is and is not at risk
--------------------------
Two of this build's results are ALGEBRAIC and cannot move under any
perturbation, so they are not sampled and not audited:

- the selection / differential-death degeneracy (V7): x(t) depends on the two
  net growth rates only through a - b, so shifting r_S and shifting d_N are the
  same perturbation at every parameter value;
- the KCC9 wiring (V3, V15): TERT has no activator on the exposure path, which
  is a property of the network, not of the rates.

What CAN move is everything quantitative: false-positive rates, interval
widths, the detection day, and the purity separation a sort must achieve.

The agents are RE-MATCHED inside every draw. That is the whole point -- the
claim is about three agents that produce the same marker fraction at the same
endpoint, and the magnitude that achieves that depends on the baseline rates.
Holding the magnitudes fixed while perturbing the rates would silently unmatch
them, which is exactly the bug this build was found to have.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import inference as inf
from .conditions import DESIGNS, MATCH_DAY, TARGET_X
from .design import N_REP as DESIGN_N_REP, SIGMA_BIO as DESIGN_SIGMA, Z as DESIGN_Z
from .model import (Params, control_fixed_point, match_at, simulate,
                    simulate_washout)

# The two-sample detection bar used by design.detection_day. Defined once, here
# and there, so the ensemble cannot qualify a headline on a different threshold.
DETECT_THRESHOLD = DESIGN_Z * DESIGN_SIGMA * np.sqrt(2.0 / DESIGN_N_REP)

# Log-scale sigma on the baseline rates. Proliferation and death rates for a
# mammary epithelial line are known to within a factor of order 1.3; the two
# transition rates are far less constrained and get a wider band.
FREE_SIGMA = {
    "r_S": 0.25, "d_S": 0.30, "r_N": 0.25, "d_N": 0.30,
    "beta": 0.50, "gamma": 0.40,
}

DEFAULT_N = 60
DEFAULT_SEED = 20260825
INNER_REPS = 8
# Same 0.002 resolution as the main run's grid. A coarser grid quantises the
# interval width and makes V6's width comparison tie rather than resolve.
GRID = np.linspace(0.0, 0.20, 101)

# The nominal build matches all three agents to an absolute fraction (0.20 at
# day 14). Across draws that target can fall BELOW the control fixed point --
# the perturbed rates move the unexposed stem-like fraction anywhere from a few
# per cent to well over half -- and "raise x to 0.20" is then not an exposure at
# all. So each draw is matched to the same FOLD increase over its OWN control
# instead, which is the biologically meaningful invariant and keeps the matched-
# agent construction well posed in every draw.
TARGET_FOLD = TARGET_X / control_fixed_point(Params())
MAX_TARGET = 0.90


def sample_params(rng: np.random.Generator, base: Params | None = None) -> Params:
    base = base or Params()
    return replace(base, **{k: float(getattr(base, k) * np.exp(rng.normal(0.0, s)))
                            for k, s in FREE_SIGMA.items()})


def _fp(p: Params, x0: float, e, t: np.ndarray, rng) -> dict:
    """Profile-interval summary for one draw.

    ``n_censored`` is not decoration. A right-censored interval has hit the edge
    of the grid, so its "width" is a lower bound rather than a width, and
    comparing two such numbers says nothing. An earlier release fed them
    straight into the V6 width comparison: 240 of 960 cytotoxicity intervals in
    the nominal ensemble were truncated. Draws with ANY censored interval are
    now excluded from that audit and counted separately.
    """
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
    """Re-match the agents and re-score the movable claims in every draw."""
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

            # V8: with counts, is a cytotoxic agent identified as CYTOTOXIC and
            # a selective one as SELECTIVE? The nominal V8 asserts all three
            # parts; an earlier ensemble audited only the first, so it reported
            # support for an identification claim it never tested.
            picks_ind = picks_cyto = picks_sel = 0
            for _ in range(INNER_REPS):
                d = inf.observe_with_counts(truths["kill_N"].apply(p),
                                            DESIGNS["extended"], x0, rng=rng)
                b = inf.compare_with_counts(p, DESIGNS["extended"], d, x0)["best"]
                # counts "ind" only, matching the nominal V8 field of the same
                # name. An earlier version counted ind + both here while the
                # nominal check counted ind alone, so the two were described as
                # the same criterion while measuring different things.
                picks_ind += b == "ind"
                picks_cyto += b == "cyto"
                ds = inf.observe_with_counts(truths["d_fit"].apply(p),
                                             DESIGNS["extended"], x0, rng=rng)
                picks_sel += inf.compare_with_counts(
                    p, DESIGNS["extended"], ds, x0)["best"] == "sel"

            # V12: washout carries no information in the fraction
            xs = np.vstack([simulate_washout(p, e, t_on, t_off, x0=x0)["x_after"]
                            for e in truths.values()])
            gap = float(np.max(xs.max(axis=0) - xs.min(axis=0)))

            # V10: how long past the endpoint before the pair separates.
            # The threshold is imported from design.py rather than restated:
            # this line previously used 2.0*sigma/sqrt(3) = 0.0346 while the
            # nominal analysis used 1.96*sigma*sqrt(2/3) = 0.0480, so the
            # ensemble scored separability on a laxer bar than the headline it
            # was qualifying.
            tl = np.linspace(0.0, 240.0, 961)
            xi = simulate(truths["d_beta"].apply(p), tl, x0=x0)["x"]
            xsl = simulate(truths["d_fit"].apply(p), tl, x0=x0)["x"]
            sep = np.abs(xi - xsl) > DETECT_THRESHOLD
            past = tl[sep & (tl > MATCH_DAY)]
        except Exception:
            continue

        n_ok += 1
        hits["V5"] += sel["ci_excl"] > 0.8
        # V6 compares interval WIDTHS, which is meaningless when either
        # interval is truncated at the grid edge. Such draws are not scored
        # either way; n_v6 records how many remained.
        if cyt["n_censored"] == 0 and cyt_e["n_censored"] == 0:
            n_v6 += 1
            hits["V6"] += (cyt_e["width"] < cyt["width"]
                           and cyt_e["point"] > cyt["point"])
            widths_s.append(cyt["width"]); widths_e.append(cyt_e["width"])
        else:
            n_v6_censored += 1
        # V8 now asserts all three parts of the nominal check
        # The nominal V8 thresholds are <0.05, >0.8, >0.8 over 120 datasets.
        # With INNER_REPS = 8 the achievable resolution is 1/8 = 0.125, so the
        # induction threshold is stated as "none of the eight" rather than a
        # 0.05 that 8 draws cannot express. The other two match the nominal
        # values exactly.
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
    # V6 is scored over the draws where a width comparison is meaningful, not
    # over all of them, so its denominator differs from the others and is
    # reported alongside rather than hidden.
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
            "V7": "algebraic: x depends on the net rates only through a - b",
            "V3/V15": "network wiring, not a rate",
        },
        "interpretation": (
            "Heuristic sensitivity bands over baseline rates that were chosen, "
            "not measured. NOT posteriors, confidence intervals or credible "
            "intervals. The agents are re-matched inside every draw."),
    }
