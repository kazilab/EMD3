"""Conditions and the validation battery.

EMD3 differs from the other three builds in what it is for. EMD1, EMD2 and EMD4
each simulate a mechanism and ask whether a flattened alternative can reproduce
it. Here the mechanism is not in doubt -- cells demonstrably change state -- and
the question is whether any *measurement* can attribute an observed change in
the stem-like fraction to induction rather than to selection or to differential
death. So most of these checks are about inference, and several of them PASS BY
DETECTING A FAILURE. That is the deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Exposure, Params, match_at

TARGET_X = 0.20
MATCH_DAY = 14.0

# Three agents producing the SAME marker fraction at the SAME endpoint by three
# different mechanisms. Only the first is an EMD3 event.
#
# All three keep the FULL baseline conversion machinery (beta = 0.010,
# gamma = 0.120). They differ in d_beta, not in beta. Calling the selection and
# cytotoxicity agents "agents with no state conversion" -- as earlier prose did
# -- is wrong: cells convert in every arm, because these are epithelial cells
# and conversion is a baseline property, not something an exposure switches on.
# The question this build poses is whether d_beta > 0 can be established, not
# whether beta > 0.
#
# These are COMPUTED, not written down. An earlier release hard-coded them, and
# the cytotoxic magnitude had gone stale at kill_N = 0.1150, which reaches
# x = 0.11 rather than the intended 0.20. That silently under-dosed the negative
# control -- the check the whole build exists to make -- and made its
# false-positive rate look like 26% when the matched agent gives 100%.
# Deriving them from the model removes the class of error entirely.
#
# Note that the selection and cytotoxicity magnitudes come out IDENTICAL. That
# is not a coincidence and not a bug: d_fit acts on r_S and kill_N acts on d_N,
# and the observed fraction depends on them only through a - b, so the same
# shift on either channel is the same perturbation. It is V7 made visible in the
# constants themselves.
def _matched_truths() -> dict:
    p = Params()
    return {
        "induction": match_at(p, "d_beta", TARGET_X, MATCH_DAY),
        "selection": match_at(p, "d_fit", TARGET_X, MATCH_DAY),
        "cytotoxicity": match_at(p, "kill_N", TARGET_X, MATCH_DAY),
    }


TRUTHS = _matched_truths()

# Washout uses the same three endpoint-matched agents, withdrawn at MATCH_DAY.
# On marker fractions it does NOT separate them (V12): after withdrawal every
# arm returns to the same baseline parameters from the same fraction, so the
# trajectories coincide. Only the count accumulated during exposure separates
# them. Implemented as a negative design result, not as a discriminating arm.
WASHOUT_DAY = MATCH_DAY

DESIGNS = {
    "short": np.array([0.0, 3.0, 7.0, 10.0, 14.0]),
    "extended": np.array([0.0, 3.0, 7.0, 10.0, 14.0, 21.0, 28.0]),
}


@dataclass(frozen=True)
class Check:
    cid: str
    edge: str
    statement: str
    test: object
    source: str


CHECKS = [
    Check("V1", "KCC8 -> EMD3 (upstream)",
          "The experimentally supported perturbation -- E2F2 on, driving FZD10 "
          "and Wnt -- enlarges the stem-like basin of attraction",
          lambda c: c["bool"]["arsenic"]["stem_basin"]
                    > 1.2 * c["bool"]["control"]["stem_basin"],
          "Yang 2026 (Food Chem Toxicol 211:116003): arsenic induces stemness "
          "in MCF-10A via the E2F2/FZD10 axis. Network structure follows "
          "Hernandez-Magana 2024 (npj Syst Biol Appl 10:99)."),

    Check("V2", "accessibility is not the same as a larger basin",
          "The supported perturbation enlarges the basin but does not open a "
          "route from a tumour-suppressor-intact start; only adding an "
          "exploratory perturbation (p53 loss), outside the arsenic-supported "
          "clamp set, does",
          lambda c: (c["bool"]["arsenic"]["reach_frac"] < 0.05
                     and c["bool"]["unsupported"]["reach_frac"] > 0.5),
          "The manuscript is explicit that the ability of an arbitrary in-silico "
          "perturbation to stabilise a stem-like attractor does not establish "
          "that the exposure produces that perturbation in cells. Here the "
          "distinction is measurable: basin size moves, accessibility does not."),

    Check("V3", "EMD3 -/-> KCC9 (OPPOSING, STRUCTURAL; TERT-negative start)",
          "From a telomerase-negative start, driving the stem-like state to "
          "100% reachability leaves the immortalisation readout at exactly "
          "zero -- stemness is not limitless replicative potential. V17 "
          "records that this is conditional on that start",
          lambda c: (c["bool"]["unsupported"]["reach_frac"] > 0.5
                     and c["bool"]["unsupported"]["reach_immortal"] < 1e-9),
          "The refined KCC framework includes stem-cell genes among KCC9-relevant "
          "endpoints, but a stem-like state is not immortalisation. Encoded as "
          "wiring -- TERT has no activator on the exposure path -- so the model "
          "cannot score stemness as KCC9 even by accident. The reachability "
          "statistic this check reads was previously incapable of reporting "
          "anything but zero, whatever the wiring did; boolean.py records the "
          "defect and tests/test_regressions.py now rewires TERT onto the "
          "exposure path and asserts the statistic fires."),

    Check("V4", "the endpoint trap",
          "An induction agent and a selection agent tuned to the same marker "
          "fraction at the same endpoint are indistinguishable AT that endpoint",
          lambda c: abs(c["pair"]["x_ind_at_match"]
                        - c["pair"]["x_sel_at_match"]) < 1e-3,
          "This is the pair every endpoint-only exposure study cannot resolve, "
          "and it is constructed rather than assumed."),

    Check("V5", "an INDUCTION-ONLY d_beta interval is not evidence of plasticity",
          "Fitting an induction-only structure -- one that ENFORCES fitness "
          "parity -- to data generated by PURE SELECTION returns a d_beta "
          "interval excluding zero in essentially every replicate",
          lambda c: c["fp"]["short"]["selection"]["ci_excl"] > 0.8,
          "The manuscript asks whether the posterior for beta is distinguishable "
          "from zero. Under a structure that assumes the two compartments grow "
          "alike, it is -- reliably, and wrongly. A misspecified structure "
          "returns a confident answer, and confidence is not correctness. The "
          "check names the structure deliberately: V16 shows the same fractions "
          "give an interval covering zero once the fitness gap is free, so the "
          "failure belongs to the assumption, not to fraction data as such."),

    Check("V6", "more data makes the wrong answer MORE confident (RESTATED)",
          "Extending follow-up NARROWS the d_beta interval while pushing its "
          "centre FURTHER from zero -- under a cytotoxic agent that carries no "
          "plasticity at all",
          lambda c: (c["fp"]["extended"]["cytotoxicity"]["width"]
                     < c["fp"]["short"]["cytotoxicity"]["width"]
                     and c["fp"]["extended"]["cytotoxicity"]["point"]
                     > c["fp"]["short"]["cytotoxicity"]["point"]),
          "Under misspecification, additional data tightens an interval around "
          "the wrong value. Precision and validity are separate properties, and "
          "only one of them improves with n. RESTATED: this check previously "
          "compared false-positive RATES across designs. Once the cytotoxic "
          "control is properly endpoint-matched that rate is 100% in both "
          "designs, so there is no headroom for a strict inequality -- the "
          "earlier version passed only because the control was under-dosed. "
          "Interval width and centre still move, and they are what the claim "
          "was always about. See fit/FINDINGS.md."),

    Check("V7", "selection and differential death are ONE model",
          "For the observed fraction, raising stem-cell growth and raising "
          "non-stem death are algebraically the same perturbation",
          lambda c: c["degeneracy"]["max_abs_diff"] < 1e-6,
          "a - b = (r_S - d_S) - (r_N - d_N). No number of fraction timepoints "
          "separates them, because they are not two hypotheses."),

    Check("V8", "absolute counts break the degeneracy",
          "Adding total cell counts drops the induction false-positive rate "
          "under a cytotoxic agent to zero, and identifies selection and "
          "cytotoxicity correctly",
          lambda c: (c["counts"]["cytotoxicity"]["picks_ind"] < 0.05
                     and c["counts"]["cytotoxicity"]["picks_cyto"] > 0.8
                     and c["counts"]["selection"]["picks_sel"] > 0.8),
          "The manuscript's point that absolute counts are substantially more "
          "informative than relative frequencies, made quantitative: a growth "
          "advantage makes the population larger, differential killing makes it "
          "smaller, and the fraction reports neither. Stated narrowly on "
          "purpose -- counts separate THESE TWO EXPOSURE CHANNELS given a known "
          "baseline. They do not separate proliferation from death in general: "
          "a total count constrains the NET rate r - d, so equal increases in "
          "both are invisible to it. Separating r from d needs a division or "
          "death readout, not a count."),

    Check("V9", "report a model comparison, not an interval",
          "With counts, model comparison identifies a genuine induction agent "
          "as carrying a non-zero beta in essentially every replicate",
          lambda c: c["counts"]["induction"]["picks_beta"] > 0.9,
          "The defensible EMD3 analysis is a comparison across selection-only, "
          "induction-only, combined and differential-death structures -- not a "
          "credible interval on a transition rate."),

    Check("V10", "detection window (DESIGN OUTPUT)",
          "Without sorting, the matched pair becomes separable within a few "
          "weeks past the endpoint at realistic measurement precision",
          lambda c: (c["design"]["detect_day"] is not None
                     and c["design"]["detect_day"] - MATCH_DAY < 30.0),
          "Computed at sigma = 0.03 absolute on the marker fraction with three "
          "replicates -- replicate variability, not counting error, which is "
          "the term that actually limits flow-cytometric fractions."),

    Check("V11", "the sorting contrast trades purity separation against replication",
          "At three replicates per arm a 20-fold separation of residual "
          "purities reaches ~7% power and a 100-fold separation ~56%. Power "
          "rises with either input: holding 100-fold and moving to six "
          "replicates per arm reaches ~84%, while holding three replicates "
          "requires ~140-fold. Both levers are reported, because one "
          "combination is not a property of the design",
          lambda c: (c["design"]["titration_20x_power"] < 0.2
                     and 0.4 < c["design"]["titration_100x_power"] < 0.7
                     and c["design"]["power_100x_n6"] > 0.8
                     and c["design"]["fold_for_80pct_power"]
                         > 1.2 * c["design"]["titration_100x_fold"]),
          "A verdict that the true contrast exceeds its critical value is 50% "
          "power at the boundary, so it cannot by itself recommend or reject "
          "an experiment. Power is reported instead, over both quantities an "
          "experimenter controls. The calculation is a prospective benchmark "
          "under a known-noise Gaussian contrast of four measured means; "
          "applying it to a single agent of unknown mechanism requires the "
          "observational units and the model-discrimination rule to be "
          "specified for that setting."),

    Check("V12", "washout adds NOTHING on fractions (NEGATIVE DESIGN RESULT)",
          "Withdrawing the agent and following the marker fraction cannot "
          "separate the mechanisms: all arms follow identical post-washout "
          "trajectories. Only the count accumulated during exposure separates "
          "them",
          lambda c: (c["washout"]["max_frac_gap"] < 1e-6
                     and c["washout"]["logn_gap_sel_cyt"] > 1.0),
          "The EMD2-4 plan lists a washout/reversal arm among the held-out "
          "conditions. It is implemented here and it does not work, for a "
          "reason that is structural rather than numerical: after withdrawal "
          "every arm returns to the same baseline parameters, and the matched "
          "arms are at the same fraction when the agent is removed, so the "
          "subsequent trajectories coincide by construction. Reported so the "
          "arm is not run in the expectation that it discriminates."),

    Check("V13", "the arsenic basin gain runs through the attributed pathway",
          "Knocking down FZD10, or inhibiting WNT, abolishes the "
          "arsenic-induced enlargement of the stem-like basin -- but "
          "OVERSHOOTS baseline to zero, so blocking the pathway is not the "
          "same as reversing the exposure",
          lambda c: (c["bool"]["arsenic_FZD10_kd"]["stem_basin"] < 1e-9
                     and c["bool"]["arsenic_WNT_inhib"]["stem_basin"] < 1e-9
                     and c["bool"]["control"]["stem_basin"] > 0.1),
          "The plan asks whether reversal of the perturbed node restores "
          "attractor structure. It does not restore it, it removes it: the "
          "knockdowns take the stem basin to zero, below the ~20% the control "
          "network carries anyway. Epistasis on this network is not a simple "
          "reversal, and the distinction matters for interpreting a rescue "
          "experiment."),

    Check("V14", "accessibility does NOT move with dose (PLAN EXPECTATION REFUTED)",
          "Mean first-passage time into the stem-like attractor is infinite "
          "under BOTH control and the supported arsenic perturbation -- the "
          "state is never reached from a tumour-suppressor-intact start, so "
          "MFPT cannot decrease with dose",
          lambda c: (not np.isfinite(c["bool"]["control"]["mfpt"])
                     and not np.isfinite(c["bool"]["arsenic"]["mfpt"])
                     and np.isfinite(c["bool"]["unsupported"]["mfpt"])),
          "The plan's draft battery expects MFPT into the stem-like attractor "
          "to fall with dose. In this network it cannot, because the supported "
          "perturbation enlarges the basin without opening a route into it -- "
          "which is V2 restated as a rate. A build that reported a "
          "dose-dependent MFPT here would have had to reach for the "
          "unsupported perturbation to get one."),

    Check("V15", "the immortalisation BASIN is a misreading trap",
          "Exposure enlarges the immortal basin (10% -> 14%) while immortal "
          "REACHABILITY stays exactly zero. The basin statistic counts states "
          "that already carry TERT; the exposure never turns TERT on",
          lambda c: (c["bool"]["arsenic"]["immortal_basin"]
                     > 1.1 * c["bool"]["control"]["immortal_basin"]
                     and c["bool"]["arsenic"]["reach_immortal"] < 1e-9
                     and c["bool"]["control"]["reach_immortal"] < 1e-9),
          "Guards the KCC9 discipline against its most likely misreading. A "
          "reader quoting basin size alone would conclude that arsenic promotes "
          "immortalisation. It does not: every attractor in that basin is "
          "entered only from a start state where telomere maintenance is "
          "ALREADY on, and no perturbation on the exposure path can switch it "
          "on. Reachability, not basin size, is the statistic that carries the "
          "KCC9 claim."),

    Check("V16", "the false positive belongs to the STRUCTURE, not the data",
          "Profiling d_beta under the COMBINED structure -- fitness gap free -- "
          "puts the interval back across zero for both d_beta = 0 agents, on "
          "exactly the same marker fractions that the induction-only structure "
          "calls plastic in every replicate. The cost is power, not validity: "
          "a genuine induction agent is detected in far fewer replicates at the "
          "short design than at the extended one",
          lambda c: (c["fp_both"]["short"]["selection"]["ci_excl"] < 0.05
                     and c["fp_both"]["short"]["cytotoxicity"]["ci_excl"] < 0.05
                     and c["fp_both"]["extended"]["selection"]["ci_excl"] < 0.05
                     and c["fp_both"]["extended"]["cytotoxicity"]["ci_excl"] < 0.05
                     and c["fp_both"]["extended"]["induction"]["ci_excl"]
                         > c["fp_both"]["short"]["induction"]["ci_excl"]),
          "Added after review observed that the build's own M_both structure "
          "already removes the false positive V5 reports, using fractions "
          "alone. The earlier framing -- that a d_beta interval is not evidence "
          "of plasticity -- was therefore too broad: an interval from a "
          "structure that leaves the fitness gap free is well behaved. What "
          "fractions alone cannot do is tell selection from differential death "
          "(V7), and what the combined structure costs is power. Both are "
          "better arguments for absolute counts than a false-positive rate "
          "that is an artefact of assuming fitness parity, and the battery "
          "should state the narrower claim it can defend."),

    Check("V17", "the KCC9 separation depends on the START STATE, not wiring alone",
          "From a telomerase-NEGATIVE start, driving the stem-like state to "
          "100% reachability leaves immortalisation at exactly zero. From a "
          "telomerase-POSITIVE start -- which is what the exemplar line "
          "actually is -- the same perturbation reaches immortalisation in "
          "every run. The separation is conditional, and the condition is a "
          "modelling choice",
          lambda c: (c["bool"]["unsupported"]["reach_immortal"] < 1e-9
                     and c["bool_tert_on"]["unsupported"]["reach_immortal"] > 0.9
                     and c["bool_tert_on"]["arsenic"]["reach_immortal"] < 1e-9),
          "Added after review noted that MCF-10A -- the exemplar in Yang 2026 -- "
          "is a spontaneously IMMORTALISED line, so a TERT-off start is not a "
          "validated representation of it. With TERT already on, "
          "IMMORTAL <- stem AND tert collapses to IMMORTAL <- stem and stemness "
          "scores as immortalisation one for one. The KCC9 discipline survives "
          "for the SUPPORTED perturbation under either start, because arsenic "
          "never reaches the stem-like state at all; what does not survive is "
          "the claim that the model cannot score stemness as KCC9 'even by "
          "accident'. It can, under the start state the exemplar implies. This "
          "check asserts both halves so the conditional is on the record."),
]
