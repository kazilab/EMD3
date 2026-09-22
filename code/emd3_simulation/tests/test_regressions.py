"""Regressions for the EMD3 build.

The one that matters most is `test_truths_are_endpoint_matched`. A previous
release hard-coded the three matched agents and the cytotoxic magnitude had gone
stale, silently under-dosing the negative control -- the check the whole build
exists to make. Nothing failed; the battery went on passing. These tests make
that class of error loud.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from emd3_simulation import boolean as bl
from emd3_simulation import design as dz
from emd3_simulation import inference as inf
from emd3_simulation.conditions import (CHECKS, DESIGNS, MATCH_DAY, TARGET_X,
                                        TRUTHS)
from emd3_simulation.ensemble import FREE_SIGMA, run_ensemble, sample_params
from emd3_simulation.model import (Exposure, Params, control_fixed_point,
                                   match_at, simulate, simulate_washout)
from emd3_simulation.model import observed_fraction as model_observed

P = Params()
X0 = control_fixed_point(P)


# --- the bug that motivated this suite ------------------------------------

def test_truths_are_endpoint_matched() -> None:
    """All three agents must reach the SAME fraction at the SAME day. The
    cytotoxic arm previously reached 0.11 against a target of 0.20."""
    t = np.linspace(0.0, MATCH_DAY, 401)
    for name, e in TRUTHS.items():
        x = float(simulate(e.apply(P), t, x0=X0)["x"][-1])
        assert x == pytest.approx(TARGET_X, abs=1e-4), f"{name} reaches {x:.4f}"


def test_selection_and_cytotoxicity_magnitudes_coincide() -> None:
    """d_fit acts on r_S and kill_N on d_N; the fraction sees only a - b, so the
    matched magnitudes must be identical. If they ever differ, one of the two
    channels has stopped being a pure shift of the fitness gap."""
    assert TRUTHS["selection"].d_fit == pytest.approx(
        TRUTHS["cytotoxicity"].kill_N, rel=1e-6)


def test_matched_agents_are_computed_not_stored() -> None:
    """Perturbing the baseline must move the matched magnitudes. A hard-coded
    table would not."""
    q = replace(P, gamma=P.gamma * 1.5)
    assert match_at(q, "d_beta", TARGET_X, MATCH_DAY).d_beta != pytest.approx(
        TRUTHS["induction"].d_beta, rel=1e-3)


# --- the algebraic results, which must hold at every parameter value -------

@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_selection_and_differential_death_are_one_model(scale: float) -> None:
    q = replace(P, r_S=P.r_S * scale, gamma=P.gamma * scale)
    t = np.linspace(0.0, 60.0, 301)
    d = 0.15
    xs = simulate(Exposure(d_fit=d).apply(q), t, x0=control_fixed_point(q))["x"]
    xc = simulate(Exposure(kill_N=d).apply(q), t, x0=control_fixed_point(q))["x"]
    assert np.max(np.abs(xs - xc)) < 1e-8


def test_selection_and_cytotoxicity_give_the_same_inference() -> None:
    """V7 as it reaches the reader.

    Table 1 of the manuscript insert prints the selection and cytotoxicity rows
    with identical interval widths and centres. That is not a coincidence to be
    rounded into existence -- the two agents are one model for the fraction, so
    the whole inference must coincide. Agreement is limited only by the
    tolerance of the root-find that matches the two magnitudes, so it is
    asserted at 1e-4 relative rather than exactly.
    """
    rng = np.random.default_rng(17)
    for t in DESIGNS.values():
        for _ in range(3):
            seed = int(rng.integers(1 << 30))
            ys = inf.observe(TRUTHS["selection"].apply(P), t, X0,
                             rng=np.random.default_rng(seed))
            yc = inf.observe(TRUTHS["cytotoxicity"].apply(P), t, X0,
                             rng=np.random.default_rng(seed))
            assert np.allclose(ys, yc, rtol=0, atol=1e-9)
            rs = inf.beta_interval(P, t, ys, X0)
            rc = inf.beta_interval(P, t, yc, X0)
            assert rs["lo"] == rc["lo"] and rs["hi"] == rc["hi"]
            assert rs["excludes_zero"] == rc["excludes_zero"]
            assert rs["point"] == pytest.approx(rc["point"], rel=1e-4)


def test_counts_do_separate_them() -> None:
    """The degeneracy is in the fraction only -- growth makes the population
    bigger, killing makes it smaller."""
    t = np.linspace(0.0, 60.0, 301)
    d = 0.15
    ns = simulate(Exposure(d_fit=d).apply(P), t, x0=X0)["log_total"]
    nc = simulate(Exposure(kill_N=d).apply(P), t, x0=X0)["log_total"]
    assert abs(ns[-1] - nc[-1]) > 1.0


def test_washout_carries_no_information_in_the_fraction() -> None:
    t_on = np.linspace(0.0, MATCH_DAY, 201)
    t_off = np.linspace(0.0, 60.0, 301)
    xs = np.vstack([simulate_washout(P, e, t_on, t_off, x0=X0)["x_after"]
                    for e in TRUTHS.values()])
    assert np.max(xs.max(axis=0) - xs.min(axis=0)) < 1e-6


# --- KCC9 wiring, which is a property of the network -----------------------

def test_exposure_can_never_reach_the_immortal_readout() -> None:
    for name, forced in bl.PERTURBATIONS.items():
        m = bl.mean_first_passage(forced, n_runs=150)
        assert m["reach_immortal"] == 0.0, name


def test_the_immortal_guard_can_actually_fail() -> None:
    """The check above is worth nothing unless it CAN report a non-zero value.

    An earlier release incremented the immortal tally once per step and broke
    out of the loop the moment STEM was reached; IMMORTAL requires STEM, so the
    tally could never fire and `reach_immortal` read exactly 0.0 whatever the
    wiring did. The KCC9 claim -- the one this build cares most about -- was
    guarded by a statistic with no power.

    So: rewire TERT onto the exposure path, which genuinely makes the
    immortalisation readout reachable, and assert the statistic notices. If
    this test ever fails, the guard has gone blind again.
    """
    intact = bl._next

    def leaky(s):
        t = list(intact(s))
        t[bl.IX["TERT"]] = s[bl.IX["TERT"]] | s[bl.IX["WNT"]]
        return tuple(t)

    try:
        bl._next = leaky
        leaked = bl.mean_first_passage(bl.UNSUPPORTED, n_runs=100)
    finally:
        bl._next = intact

    assert leaked["reach_immortal"] > 0.5
    assert np.isfinite(leaked["mfpt_immortal"])
    # and the intact wiring still reports zero, for the right reason
    assert bl.mean_first_passage(bl.UNSUPPORTED,
                                 n_runs=100)["reach_immortal"] == 0.0


def test_fast_and_canonical_rules_agree() -> None:
    """`_next` is the hot path and `rules` is the readable API. They must be
    the same function, over every state and every clamp the battery uses."""
    for forced in ({}, *bl.PERTURBATIONS.values()):
        pairs = bl._pairs(forced)
        for code in range(2 ** bl.N):
            s = tuple((code >> i) & 1 for i in range(bl.N))
            a = tuple(int(v) for v in bl.rules(np.array(s, dtype=np.int8), forced))
            assert a == bl._next_forced(s, pairs), (forced, code)


def test_basin_fractions_carry_a_monte_carlo_error() -> None:
    """Under asynchronous update a basin is a probability, not a set, so the
    number needs an error bar. A single trajectory per start gives it an MC
    error near 0.012 -- the third significant figure of every basin quoted."""
    a = bl.attractors(bl.PERTURBATIONS["arsenic"])
    assert a["n_traj"] > 1
    assert 0.0 < a["stem_basin_se"] < 0.01
    # the direction of the arsenic effect must survive its own MC error
    c = bl.attractors(bl.PERTURBATIONS["control"])
    assert (a["stem_basin"] - c["stem_basin"]
            > 3 * (a["stem_basin_se"] + c["stem_basin_se"]))


def test_immortal_basin_moves_but_reachability_does_not() -> None:
    """The misreading V15 guards: basin size responds to exposure, reachability
    does not, and only the second carries the KCC9 claim."""
    c = bl.attractors(bl.PERTURBATIONS["control"])
    a = bl.attractors(bl.PERTURBATIONS["arsenic"])
    assert a["immortal_basin"] > c["immortal_basin"]
    assert bl.mean_first_passage(bl.PERTURBATIONS["arsenic"],
                                 n_runs=150)["reach_immortal"] == 0.0


def test_pathway_knockdown_abolishes_rather_than_restores() -> None:
    ctrl = bl.attractors(bl.PERTURBATIONS["control"])["stem_basin"]
    for k in ("arsenic_FZD10_kd", "arsenic_WNT_inhib"):
        assert bl.attractors(bl.PERTURBATIONS[k])["stem_basin"] < 1e-9
    assert ctrl > 0.1          # the knockdowns go BELOW control, not back to it


def test_supported_perturbation_does_not_open_a_route() -> None:
    for k in ("control", "arsenic"):
        assert not np.isfinite(bl.mean_first_passage(
            bl.PERTURBATIONS[k], n_runs=150)["mfpt"])
    assert np.isfinite(bl.mean_first_passage(
        bl.PERTURBATIONS["unsupported"], n_runs=150)["mfpt"])


# --- inference --------------------------------------------------------------

def test_model_comparison_uses_small_sample_corrected_aic() -> None:
    """5-7 timepoints against 1-2 parameters is deep in the regime where plain
    AIC under-penalises the larger structure."""
    t = DESIGNS["short"]
    y = inf.observe(TRUTHS["induction"].apply(P), t, X0)
    f1 = inf.fit(P, "ind", t, y, X0)
    f2 = inf.fit(P, "both", t, y, X0)
    assert f1["aicc"] > f1["aic"] and f2["aicc"] > f2["aic"]
    assert (f2["aicc"] - f2["aic"]) > (f1["aicc"] - f1["aic"])
    assert "d_aicc" in inf.compare(P, t, y, X0)


def test_beta_interval_is_not_right_censored_at_the_nominal_parameters() -> None:
    rng = np.random.default_rng(0)
    for t in DESIGNS.values():
        for e in TRUTHS.values():
            for _ in range(4):
                y = inf.observe(e.apply(P), t, X0, rng=rng)
                assert not inf.beta_interval(P, t, y, X0)["right_censored"]


def test_a_pure_selection_agent_still_yields_a_confident_dbeta() -> None:
    """V5: the false positive the build exists to expose."""
    rng = np.random.default_rng(3)
    hits = sum(inf.beta_interval(
        P, DESIGNS["short"],
        inf.observe(TRUTHS["selection"].apply(P), DESIGNS["short"], X0, rng=rng),
        X0)["excludes_zero"] for _ in range(12))
    assert hits >= 10


def test_the_combined_structure_removes_the_false_positive() -> None:
    """V16, and the scope limit on V5.

    The same fractions that make the induction-only structure call selection
    and cytotoxicity plastic in every replicate put the interval back across
    zero once the fitness gap is free. The false positive belongs to the
    assumption of fitness parity, not to fraction data -- so the build must
    claim the narrower thing.
    """
    for truth in ("selection", "cytotoxicity"):
        rng = np.random.default_rng(11)
        for _ in range(6):
            y = inf.observe(TRUTHS[truth].apply(P), DESIGNS["extended"], X0,
                            rng=rng)
            r = inf.beta_interval(P, DESIGNS["extended"], y, X0,
                                  structure="both")
            assert not r["excludes_zero"], truth
            assert not r["right_censored"], truth


def test_the_combined_structure_costs_power_not_validity() -> None:
    """The price of freeing the fitness gap is detections of a REAL induction
    agent, and that price is what argues for absolute counts."""
    def rate(design):
        rng = np.random.default_rng(5)
        return sum(inf.beta_interval(
            P, DESIGNS[design],
            inf.observe(TRUTHS["induction"].apply(P), DESIGNS[design], X0,
                        rng=rng),
            X0, structure="both")["excludes_zero"] for _ in range(12)) / 12
    assert rate("extended") > rate("short")


def test_beta_interval_grid_is_wide_enough_for_the_combined_structure() -> None:
    """`both` leaves d_beta weakly identified and its interval runs past the
    0.12 that suffices for `ind`. On too narrow a grid the interval is
    right-censored and its width is meaningless."""
    rng = np.random.default_rng(2)
    for t in DESIGNS.values():
        for e in TRUTHS.values():
            y = inf.observe(e.apply(P), t, X0, rng=rng)
            assert not inf.beta_interval(P, t, y, X0,
                                         structure="both")["right_censored"]


def test_fitted_rates_cannot_go_negative() -> None:
    """d_fit is an additive shift on r_S, so its floor is -r_S: below that the
    structure is asserting a negative division rate."""
    lo, _ = inf.bounds(P, "sel")
    assert lo[0] == pytest.approx(-P.r_S)
    lo_both, _ = inf.bounds(P, "both")
    assert lo_both[1] == pytest.approx(-P.r_S)


def test_binomial_counting_noise_is_not_the_limiting_term() -> None:
    """Why the battery uses replicate variability and not counting error.

    20 000 flow events give a standard error near 0.003 on a fraction of 0.2 --
    an order of magnitude below SIGMA_BIO. Using it would make almost anything
    significant, so the optimistic model is kept as the justification for the
    realistic one rather than as the noise the results rest on.
    """
    t = np.linspace(0.0, 14.0, 15)
    rng = np.random.default_rng(0)
    draws = np.vstack([
        model_observed(TRUTHS["induction"].apply(P), t, X0, rng=rng)
        for _ in range(200)])
    counting_sd = float(np.mean(np.std(draws, axis=0)))
    assert counting_sd < inf.SIGMA_BIO / 5


# --- ensemble ---------------------------------------------------------------

def test_sample_params_perturbs_only_the_baseline_rates() -> None:
    q = sample_params(np.random.default_rng(0))
    for f in ("r_S", "d_S", "r_N", "d_N", "beta", "gamma"):
        assert f in FREE_SIGMA
    assert q.r_S != P.r_S and q.gamma != P.gamma


@pytest.mark.parametrize("n", [0, -1])
def test_run_ensemble_rejects_nonpositive_draw_counts(n: int) -> None:
    with pytest.raises(ValueError):
        run_ensemble(n=n)


def test_ensemble_rematches_agents_within_each_draw() -> None:
    """Holding the magnitudes fixed while perturbing the rates would unmatch the
    agents -- the exact bug this build was found to have."""
    r = run_ensemble(n=4, seed=5)
    assert r["n_ok"] >= 1
    assert r["audit"]["V12"] == 1.0        # algebraic, must hold in every draw
    assert r["target_fold_over_control"] > 1.0


# --- battery integrity ------------------------------------------------------

def test_every_check_has_a_distinct_id_and_a_basis() -> None:
    ids = [c.cid for c in CHECKS]
    assert len(ids) == len(set(ids))
    for c in CHECKS:
        assert c.source.strip() and c.statement.strip()


# --- likelihood calibration (Pass 6) ---------------------------------------

def test_fit_uses_the_se_of_the_replicate_mean_not_the_replicate_sd() -> None:
    """observe() returns an average of n_rep replicates, so the likelihood must
    use sigma/sqrt(n_rep). An earlier release generated at that SE and fitted
    with the per-replicate sigma, overstating the noise by sqrt(3) and making
    every profile interval too wide."""
    assert inf.se_mean(inf.SIGMA_BIO, 3) == pytest.approx(inf.SIGMA_BIO / np.sqrt(3))
    rng = np.random.default_rng(0)
    draws = np.vstack([inf.observe(TRUTHS["induction"].apply(P), DESIGNS["short"],
                                   X0, rng=rng) for _ in range(4000)])
    noise_sd = float(np.mean(np.std(draws, axis=0)))
    assert noise_sd == pytest.approx(inf.se_mean(inf.SIGMA_BIO), rel=0.06)


def test_joint_count_likelihood_is_calibrated() -> None:
    """At the true structure the scaled residual sum of squares must average its
    degrees of freedom. Differencing against a noisy first timepoint gave
    residuals correlated at 0.5 with variance 2*se^2 that the fit treated as
    independent; mean-centring alone still leaves them correlated. Orthonormal
    contrasts fix both."""
    t = DESIGNS["extended"]
    for truth, struct in (("cytotoxicity", "cyto"), ("selection", "sel")):
        rss = [inf.fit_with_counts(
            P, struct, t,
            inf.observe_with_counts(TRUTHS[truth].apply(P), t, X0,
                                    rng=np.random.default_rng(s)),
            X0)["rss"] for s in range(300)]
        dof = len(t) + (len(t) - 1) - 1
        assert np.mean(rss) == pytest.approx(dof, rel=0.12), truth


def test_count_contrasts_are_orthonormal_and_kill_an_offset() -> None:
    for n in (5, 7, 11):
        H = inf._contrasts(n)
        assert H.shape == (n - 1, n)
        assert np.allclose(H @ H.T, np.eye(n - 1))
        assert np.allclose(H @ np.ones(n), 0.0)


# --- design power (Pass 6) --------------------------------------------------

def test_titration_verdict_is_reported_with_its_power() -> None:
    """'Separable' only says the true contrast clears the critical value, which
    at the boundary is 50% power. The 100x design clears it at ~56% -- a coin
    flip that an earlier release reported as a working design."""
    mp = dz.matched_pair(P, TARGET_X, MATCH_DAY)
    r100 = dz.purity_dependence(mp, eps_hi=0.100, day=7.0)
    assert r100["separable"] and r100["power"] < 0.7
    need = dz.purity_fold_for_power(mp, 0.80, day=7.0)
    assert need["feasible"] and need["fold"] > 1.2 * r100["purity_fold"]


# --- the KCC9 result is conditional on the start state (Pass 6) -------------

def test_kcc9_separation_depends_on_the_telomerase_start() -> None:
    """MCF-10A is a spontaneously immortalised line, so a TERT-off start is not
    a validated representation of the exemplar. With TERT on, IMMORTAL <- stem
    AND tert collapses to IMMORTAL <- stem."""
    off = bl.mean_first_passage(bl.UNSUPPORTED, n_runs=150, tert_start=0)
    on = bl.mean_first_passage(bl.UNSUPPORTED, n_runs=150, tert_start=1)
    assert off["reach_immortal"] == 0.0
    assert on["reach_immortal"] > 0.9
    # the SUPPORTED perturbation is safe under either start
    for ts in (0, 1):
        assert bl.mean_first_passage(bl.SUPPORTED, n_runs=150,
                                     tert_start=ts)["reach_immortal"] == 0.0


# --- ensemble integrity (Pass 6) -------------------------------------------

def test_ensemble_uses_the_same_detection_threshold_as_the_nominal_analysis() -> None:
    from emd3_simulation.ensemble import DETECT_THRESHOLD
    assert DETECT_THRESHOLD == pytest.approx(
        dz.Z * dz.SIGMA_BIO * np.sqrt(2.0 / dz.N_REP))


def test_ensemble_excludes_truncated_intervals_from_the_width_audit() -> None:
    """A right-censored interval's width is a lower bound, not a width."""
    r = run_ensemble(n=6, seed=11)
    assert r["audit_denominator"]["V6"] <= r["n_ok"]
    assert (r["audit_denominator"]["V6"] + r["n_v6_draws_dropped_for_censoring"]
            == r["n_ok"])
