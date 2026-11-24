"""Regression tests for the two-state population model and Boolean network."""

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


def test_truths_are_endpoint_matched() -> None:
    t = np.linspace(0.0, MATCH_DAY, 401)
    for name, e in TRUTHS.items():
        x = float(simulate(e.apply(P), t, x0=X0)["x"][-1])
        assert x == pytest.approx(TARGET_X, abs=1e-4), f"{name} reaches {x:.4f}"


def test_selection_and_cytotoxicity_magnitudes_coincide() -> None:
    """Both perturbations produce the same shift in the net growth difference."""
    assert TRUTHS["selection"].d_fit == pytest.approx(
        TRUTHS["cytotoxicity"].kill_N, rel=1e-6)


def test_matching_accounts_for_baseline_rates() -> None:
    q = replace(P, gamma=P.gamma * 1.5)
    assert match_at(q, "d_beta", TARGET_X, MATCH_DAY).d_beta != pytest.approx(
        TRUTHS["induction"].d_beta, rel=1e-3)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_selection_and_differential_death_are_one_model(scale: float) -> None:
    q = replace(P, r_S=P.r_S * scale, gamma=P.gamma * scale)
    t = np.linspace(0.0, 60.0, 301)
    d = 0.15
    xs = simulate(Exposure(d_fit=d).apply(q), t, x0=control_fixed_point(q))["x"]
    xc = simulate(Exposure(kill_N=d).apply(q), t, x0=control_fixed_point(q))["x"]
    assert np.max(np.abs(xs - xc)) < 1e-8


def test_selection_and_cytotoxicity_give_the_same_inference() -> None:
    """The matched models give the same fraction likelihood under the same noise draw."""
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


def test_total_counts_distinguish_selection_and_differential_death() -> None:
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


def test_exposure_can_never_reach_the_immortal_readout() -> None:
    for name, forced in bl.PERTURBATIONS.items():
        m = bl.mean_first_passage(forced, n_runs=150)
        assert m["reach_immortal"] == 0.0, name


def test_first_passage_detects_immortalisation_in_a_positive_control() -> None:
    """The readout responds to an added regulatory edge in this model control."""
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
    assert bl.mean_first_passage(bl.UNSUPPORTED,
                                 n_runs=100)["reach_immortal"] == 0.0


def test_fast_and_canonical_rules_agree() -> None:
    for forced in ({}, *bl.PERTURBATIONS.values()):
        pairs = bl._pairs(forced)
        for code in range(2 ** bl.N):
            s = tuple((code >> i) & 1 for i in range(bl.N))
            a = tuple(int(v) for v in bl.rules(np.array(s, dtype=np.int8), forced))
            assert a == bl._next_forced(s, pairs), (forced, code)


def test_basin_fractions_carry_a_monte_carlo_error() -> None:
    a = bl.attractors(bl.PERTURBATIONS["arsenic"])
    assert a["n_traj"] > 1
    assert 0.0 < a["stem_basin_se"] < 0.01
    c = bl.attractors(bl.PERTURBATIONS["control"])
    assert (a["stem_basin"] - c["stem_basin"]
            > 3 * (a["stem_basin_se"] + c["stem_basin_se"]))


def test_immortal_basin_moves_but_reachability_does_not() -> None:
    """Basin size and reachability from a specified start are distinct quantities."""
    c = bl.attractors(bl.PERTURBATIONS["control"])
    a = bl.attractors(bl.PERTURBATIONS["arsenic"])
    assert a["immortal_basin"] > c["immortal_basin"]
    assert bl.mean_first_passage(bl.PERTURBATIONS["arsenic"],
                                 n_runs=150)["reach_immortal"] == 0.0


def test_pathway_knockdown_reduces_the_stem_basin_to_zero() -> None:
    ctrl = bl.attractors(bl.PERTURBATIONS["control"])["stem_basin"]
    for k in ("arsenic_FZD10_kd", "arsenic_WNT_inhib"):
        assert bl.attractors(bl.PERTURBATIONS[k])["stem_basin"] < 1e-9
    assert ctrl > 0.1


def test_supported_perturbation_does_not_open_a_route() -> None:
    for k in ("control", "arsenic"):
        assert not np.isfinite(bl.mean_first_passage(
            bl.PERTURBATIONS[k], n_runs=150)["mfpt"])
    assert np.isfinite(bl.mean_first_passage(
        bl.PERTURBATIONS["unsupported"], n_runs=150)["mfpt"])


def test_model_comparison_uses_small_sample_corrected_aic() -> None:
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


def test_induction_only_fit_attributes_selection_to_conversion() -> None:
    rng = np.random.default_rng(3)
    hits = sum(inf.beta_interval(
        P, DESIGNS["short"],
        inf.observe(TRUTHS["selection"].apply(P), DESIGNS["short"], X0, rng=rng),
        X0)["excludes_zero"] for _ in range(12))
    assert hits >= 10


def test_joint_fit_includes_zero_conversion_change_under_selection_and_death() -> None:
    for truth in ("selection", "cytotoxicity"):
        rng = np.random.default_rng(11)
        for _ in range(6):
            y = inf.observe(TRUTHS[truth].apply(P), DESIGNS["extended"], X0,
                            rng=rng)
            r = inf.beta_interval(P, DESIGNS["extended"], y, X0,
                                  structure="both")
            assert not r["excludes_zero"], truth
            assert not r["right_censored"], truth


def test_extended_sampling_improves_detection_of_conversion_in_joint_fit() -> None:
    def rate(design):
        rng = np.random.default_rng(5)
        return sum(inf.beta_interval(
            P, DESIGNS[design],
            inf.observe(TRUTHS["induction"].apply(P), DESIGNS[design], X0,
                        rng=rng),
            X0, structure="both")["excludes_zero"] for _ in range(12)) / 12
    assert rate("extended") > rate("short")


def test_beta_interval_grid_is_wide_enough_for_the_combined_structure() -> None:
    rng = np.random.default_rng(2)
    for t in DESIGNS.values():
        for e in TRUTHS.values():
            y = inf.observe(e.apply(P), t, X0, rng=rng)
            assert not inf.beta_interval(P, t, y, X0,
                                         structure="both")["right_censored"]


def test_fitted_rates_cannot_go_negative() -> None:
    lo, _ = inf.bounds(P, "sel")
    assert lo[0] == pytest.approx(-P.r_S)
    lo_both, _ = inf.bounds(P, "both")
    assert lo_both[1] == pytest.approx(-P.r_S)


def test_binomial_counting_noise_is_not_the_limiting_term() -> None:
    t = np.linspace(0.0, 14.0, 15)
    rng = np.random.default_rng(0)
    draws = np.vstack([
        model_observed(TRUTHS["induction"].apply(P), t, X0, rng=rng)
        for _ in range(200)])
    counting_sd = float(np.mean(np.std(draws, axis=0)))
    assert counting_sd < inf.SIGMA_BIO / 5


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
    r = run_ensemble(n=4, seed=5)
    assert r["n_ok"] >= 1
    assert r["audit"]["V12"] == 1.0
    assert r["target_fold_over_control"] > 1.0


def test_every_check_has_a_distinct_id_and_a_basis() -> None:
    ids = [c.cid for c in CHECKS]
    assert len(ids) == len(set(ids))
    for c in CHECKS:
        assert c.source.strip() and c.statement.strip()


def test_fit_uses_the_se_of_the_replicate_mean_not_the_replicate_sd() -> None:
    assert inf.se_mean(inf.SIGMA_BIO, 3) == pytest.approx(inf.SIGMA_BIO / np.sqrt(3))
    rng = np.random.default_rng(0)
    draws = np.vstack([inf.observe(TRUTHS["induction"].apply(P), DESIGNS["short"],
                                   X0, rng=rng) for _ in range(4000)])
    noise_sd = float(np.mean(np.std(draws, axis=0)))
    assert noise_sd == pytest.approx(inf.se_mean(inf.SIGMA_BIO), rel=0.06)


def test_joint_count_likelihood_is_calibrated() -> None:
    """Under the generating model, the expected scaled residual sum of squares equals its degrees of freedom."""
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


def test_titration_verdict_is_reported_with_its_power() -> None:
    mp = dz.matched_pair(P, TARGET_X, MATCH_DAY)
    r100 = dz.purity_dependence(mp, eps_hi=0.100, day=7.0)
    assert r100["separable"] and r100["power"] < 0.7
    need = dz.purity_fold_for_power(mp, 0.80, day=7.0)
    assert need["feasible"] and need["fold"] > 1.2 * r100["purity_fold"]


def test_kcc9_separation_depends_on_the_telomerase_start() -> None:
    off = bl.mean_first_passage(bl.UNSUPPORTED, n_runs=150, tert_start=0)
    on = bl.mean_first_passage(bl.UNSUPPORTED, n_runs=150, tert_start=1)
    assert off["reach_immortal"] == 0.0
    assert on["reach_immortal"] > 0.9
    for ts in (0, 1):
        assert bl.mean_first_passage(bl.SUPPORTED, n_runs=150,
                                     tert_start=ts)["reach_immortal"] == 0.0


def test_ensemble_uses_the_same_detection_threshold_as_the_nominal_analysis() -> None:
    from emd3_simulation.ensemble import DETECT_THRESHOLD
    assert DETECT_THRESHOLD == pytest.approx(
        dz.Z * dz.SIGMA_BIO * np.sqrt(2.0 / dz.N_REP))


def test_ensemble_excludes_truncated_intervals_from_the_width_audit() -> None:
    """An interval that reaches the grid boundary supplies a lower bound on its width."""
    r = run_ensemble(n=6, seed=11)
    assert r["audit_denominator"]["V6"] <= r["n_ok"]
    assert (r["audit_denominator"]["V6"] + r["n_v6_draws_dropped_for_censoring"]
            == r["n_ok"])
