"""Endpoint-matched exposure scenarios, observation schedules and computational checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Exposure, Params, match_at

TARGET_X = 0.20
MATCH_DAY = 14.0

# Match the endpoint under three different exposure effects. Baseline conversion continues in every arm.
def _matched_truths() -> dict:
    p = Params()
    return {
        "induction": match_at(p, "d_beta", TARGET_X, MATCH_DAY),
        "selection": match_at(p, "d_fit", TARGET_X, MATCH_DAY),
        "cytotoxicity": match_at(p, "kill_N", TARGET_X, MATCH_DAY),
    }


TRUTHS = _matched_truths()

# At washout, matched arms share the same fraction but may differ in total count.
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
    Check("V1", "KCC8 to EMD3",
          "The E2F2 perturbation increases the stem-like basin relative to control.",
          lambda c: c["bool"]["arsenic"]["stem_basin"]
                    > 1.2 * c["bool"]["control"]["stem_basin"],
          "Yang et al. (2026) link arsenic exposure to E2F2/FZD10 signalling in MCF-10A cells. "
          "The Boolean topology draws on Hernández-Magaña et al. (2024)."),

    Check("V2", "Basin size and reachability",
          "From a tumour-suppressor-intact start, the stem-like state remains rarely reached "
          "under the exposure-supported perturbation. Reachability increases under the "
          "exploratory comparator.",
          lambda c: (c["bool"]["arsenic"]["reach_frac"] < 0.05
                     and c["bool"]["unsupported"]["reach_frac"] > 0.5),
          "Basin fractions average over all starts. Reachability here is conditional on the "
          "specified non-stem starting state."),

    Check("V3", "EMD3 and KCC9 from a TERT-off start",
          "In the exploratory condition, the stem-like readout is reachable while the "
          "immortalisation readout remains off.",
          lambda c: (c["bool"]["unsupported"]["reach_frac"] > 0.5
                     and c["bool"]["unsupported"]["reach_immortal"] < 1e-9),
          "The immortalisation rule requires STEM and TERT. The exposure leaves the initial "
          "TERT state unchanged."),

    Check("V4", "Endpoint matching",
          "Induction and selection yield the same marker fraction at the matching time.",
          lambda c: abs(c["pair"]["x_ind_at_match"]
                        - c["pair"]["x_sel_at_match"]) < 1e-3,
          "The two exposure magnitudes are solved separately for the same target fraction and "
          "time."),

    Check("V5", "Selection under an induction-only fit",
          "With synthetic selection data, the induction-only interval excludes zero in more "
          "than 80% of short-schedule datasets.",
          lambda c: c["fp"]["short"]["selection"]["ci_excl"] > 0.8,
          "The fitted structure enforces equal fitness, so it can attribute selection-driven "
          "changes in fraction to increased conversion."),

    Check("V6", "Interval width and estimate under model misspecification",
          "For synthetic differential death, longer follow-up narrows the induction-only "
          "interval and raises the estimated change in conversion.",
          lambda c: (c["fp"]["extended"]["cytotoxicity"]["width"]
                     < c["fp"]["short"]["cytotoxicity"]["width"]
                     and c["fp"]["extended"]["cytotoxicity"]["point"]
                     > c["fp"]["short"]["cytotoxicity"]["point"]),
          "Median interval width and median point estimate are compared across the two "
          "schedules. The simulated exposure-induced change in conversion is zero."),

    Check("V7", "Growth and death effects on the fraction",
          "Equal increases in stem-like proliferation or non-stem death produce the same "
          "marker-fraction trajectory.",
          lambda c: c["degeneracy"]["max_abs_diff"] < 1e-6,
          "The fraction depends on differential net growth through (r_S - d_S) - (r_N - d_N)."),

    Check("V8", "Model selection with cell counts",
          "Joint fits of fractions and counts distinguish the selection and differential-death "
          "scenarios.",
          lambda c: (c["counts"]["cytotoxicity"]["picks_ind"] < 0.05
                     and c["counts"]["cytotoxicity"]["picks_cyto"] > 0.8
                     and c["counts"]["selection"]["picks_sel"] > 0.8),
          "For the specified baseline rates, the two exposure effects give different total "
          "counts. Counts constrain net growth. Separate proliferation and death rates require "
          "additional measurements."),

    Check("V9", "Induction with count measurements",
          "With synthetic induction data, joint model comparison selects a model containing an "
          "induction term in more than 90% of datasets.",
          lambda c: c["counts"]["induction"]["picks_beta"] > 0.9,
          "The comparison uses AICc across selection, induction, combined and "
          "differential-death models."),

    Check("V10", "Detection by follow-up",
          "After the matching time, the unsorted model trajectories separate by more than the "
          "pointwise threshold within 30 days.",
          lambda c: (c["design"]["detect_day"] is not None
                     and c["design"]["detect_day"] - MATCH_DAY < 30.0),
          "The calculation assumes a replicate standard deviation of 0.03 in marker fraction "
          "and three independent replicates per arm."),

    Check("V11", "Power of the residual-fraction contrast",
          "Power increases with the separation between initial marker fractions and with "
          "replication under the assigned noise model.",
          lambda c: (c["design"]["titration_20x_power"] < 0.2
                     and 0.4 < c["design"]["titration_100x_power"] < 0.7
                     and c["design"]["power_100x_n6"] > 0.8
                     and c["design"]["fold_for_80pct_power"]
                         > 1.2 * c["design"]["titration_100x_fold"]),
          "The calculation uses a Gaussian contrast of four independent means. Applying the "
          "result to an experiment requires a specified experimental unit and model "
          "comparison."),

    Check("V12", "Marker fraction after washout",
          "Fraction trajectories coincide after washout in the matched scenarios. Counts "
          "accumulated during exposure differ between selection and differential death.",
          lambda c: (c["washout"]["max_frac_gap"] < 1e-6
                     and c["washout"]["logn_gap_sel_cyt"] > 1.0),
          "All arms return immediately to the same baseline rates from the same fraction. This "
          "result assumes fully reversible exposure effects."),

    Check("V13", "FZD10 and WNT dependence of the basin",
          "In the Boolean model, blocking FZD10 or WNT under the exposure perturbation reduces "
          "the stem-like basin below its control value.",
          lambda c: (c["bool"]["arsenic_FZD10_kd"]["stem_basin"] < 1e-9
                     and c["bool"]["arsenic_WNT_inhib"]["stem_basin"] < 1e-9
                     and c["bool"]["control"]["stem_basin"] > 0.1),
          "Both simulated perturbations abolish the stem-like basin. The unperturbed network "
          "retains a stem-like basin."),

    Check("V14", "First passage from a restrained start",
          "No simulated trajectory reaches the stem-like state from the specified start under "
          "control or the exposure-supported perturbation. The exploratory comparator gives a "
          "finite first-passage time.",
          lambda c: (not np.isfinite(c["bool"]["control"]["mfpt"])
                     and not np.isfinite(c["bool"]["arsenic"]["mfpt"])
                     and np.isfinite(c["bool"]["unsupported"]["mfpt"])),
          "First-passage times count asynchronous node updates. A reported infinite mean "
          "indicates that no trajectory reached the readout within the simulation."),

    Check("V15", "Immortalisation basin and reachability",
          "The exposure enlarges the immortalisation basin while reachability from a TERT-off "
          "start remains zero.",
          lambda c: (c["bool"]["arsenic"]["immortal_basin"]
                     > 1.1 * c["bool"]["control"]["immortal_basin"]
                     and c["bool"]["arsenic"]["reach_immortal"] < 1e-9
                     and c["bool"]["control"]["reach_immortal"] < 1e-9),
          "Basin estimates include TERT-on starting states. From a TERT-off start the network "
          "cannot activate this readout."),

    Check("V16", "Induction profiles with a free fitness difference",
          "The combined-model interval usually includes zero under selection or differential "
          "death. Under genuine induction, longer follow-up increases detection.",
          lambda c: (c["fp_both"]["short"]["selection"]["ci_excl"] < 0.05
                     and c["fp_both"]["short"]["cytotoxicity"]["ci_excl"] < 0.05
                     and c["fp_both"]["extended"]["selection"]["ci_excl"] < 0.05
                     and c["fp_both"]["extended"]["cytotoxicity"]["ci_excl"] < 0.05
                     and c["fp_both"]["extended"]["induction"]["ci_excl"]
                         > c["fp_both"]["short"]["induction"]["ci_excl"]),
          "Allowing the fitness difference to vary lets the fitted model account for changes "
          "in fractions through either conversion or differential net growth."),

    Check("V17", "Dependence on the initial TERT state",
          "Under the exploratory perturbation, immortalisation is reachable from a TERT-on "
          "start and remains absent from a TERT-off start. The exposure-supported perturbation "
          "leaves it unreachable from either restrained start.",
          lambda c: (c["bool"]["unsupported"]["reach_immortal"] < 1e-9
                     and c["bool_tert_on"]["unsupported"]["reach_immortal"] > 0.9
                     and c["bool_tert_on"]["arsenic"]["reach_immortal"] < 1e-9),
          "MCF-10A cells are spontaneously immortalised. The two TERT starting states are "
          "modelling assumptions. With TERT on, the immortalisation readout can follow the "
          "stem-like state."),
]
