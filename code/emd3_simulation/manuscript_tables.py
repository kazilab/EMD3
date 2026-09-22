"""Emit every number the EMD3 manuscript inserts need, as one JSON file.

    python -m emd3_simulation.manuscript_tables

Writes `figures/output/emd3_manuscript_tables.json`. The docx generators read
ONLY from this file, so main text and supplementary cannot drift apart, and
neither can drift from the simulation: rerun `emd3_simulation.run` first, then
this.

Mirrors `emd2_simulation/manuscript_tables.py`. The one structural difference is
that parameter units and provenance are read from the trailing comments in
`model.py` rather than restated in a table here -- EMD2 hand-wrote them and the
two drifted; extracting them means the comment IS the documentation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path

from .conditions import CHECKS, DESIGNS, MATCH_DAY, TARGET_X, TRUTHS, WASHOUT_DAY
from .model import Exposure, Params

SRC = Path(__file__).parent / "model.py"

# Parameters the ensemble perturbs, and what each one rests on. Anything not
# listed is either structural (the wiring) or derived.
PARAM_BASIS = {
    "r_S": "Chosen. Stem-like proliferation rate.",
    "d_S": "Chosen. Stem-like death rate.",
    "r_N": "Chosen. Non-stem proliferation rate.",
    "d_N": "Chosen. Non-stem death rate.",
    "beta": "CHOSEN, not measured. Dedifferentiation NSCC → CSC — the EMD3 "
            "event itself. The whole identifiability question is whether a "
            "change in this parameter can be told apart from a change in the "
            "net fitness gap.",
    "gamma": "CHOSEN. Differentiation CSC → NSCC. Enters the baseline "
             "stationary fraction jointly with beta AND with the fitness gap "
             "a − b; β/(β+γ) alone pins it only when a = b.",
}

EXPOSURE_CHANNELS = {
    "d_beta": ("induction", "Additive INCREASE in the dedifferentiation rate: "
                            "Δβ > 0. This is exposure-induced conversion."),
    "d_fit": ("selection", "Additive change in the net fitness gap (a − b), "
                           "applied through r_S. Baseline conversion rates "
                           "unchanged (Δβ = 0); cells still convert at β."),
    "kill_N": ("cytotoxicity", "Additive death rate on NSCC only. Baseline "
                               "conversion rates unchanged (Δβ = 0) and "
                               "neither compartment divides faster."),
    "kill_S": ("(unused in the matched set)", "Additive death rate on CSC only."),
    "d_gamma": ("(unused in the matched set)", "Additive change in "
                                               "differentiation."),
}

STATE_GROUPS = [
    ("Population state (ODE)",
     [("N", "non-stem-like cell count (NSCC)"),
      ("S", "stem-like cell count (CSC)")]),
    ("Derived readouts",
     [("x = S/(N+S)", "stem-like FRACTION — the quantity a flow assay reports"),
      ("ln(N+S)", "ABSOLUTE COUNT on the NATURAL-log scale — the quantity "
                  "that breaks the degeneracy. model.simulate uses np.log, and "
                  "every count gap in the text is in natural-log units; this "
                  "row previously read log₁₀, which was a labelling error, not "
                  "a different calculation")]),
    ("Boolean layer (separate, discrete)",
     [("E2F2, FZD10, WNT, …", "network nodes; asynchronous update"),
      ("stem basin", "fraction of initial states reaching the stem attractor, "
                     "meaned over repeated asynchronous trajectories per state "
                     "— a PROBABILITY under update order, reported ± its Monte "
                     "Carlo standard error"),
      ("immortal basin", "same, for the immortal attractor; note this counts "
                         "states that already carry TERT and is NOT "
                         "reachability"),
      ("stem / immortal reached", "REACHABILITY from a tumour-suppressor-"
                                  "intact, telomerase-negative start — the "
                                  "statistic that carries the KCC9 claim")]),
]


def param_comments() -> dict:
    """Unit and meaning for each Params field, read from its trailing comment.

    `r_S: float = 0.55        # stem-like proliferation` -> "stem-like
    proliferation". Keeping this next to the value is the only way the two stay
    in step.
    """
    out: dict[str, str] = {}
    for line in SRC.read_text().splitlines():
        m = re.match(r"\s*([a-zA-Z_]\w*)\s*:\s*float\s*=\s*[^#]*#\s*(.+?)\s*$",
                     line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def _channel_value(truth, channel: str) -> float:
    """TRUTHS holds Exposure objects; the summary JSON holds plain dicts."""
    if isinstance(truth, dict):
        return float(truth.get(channel, 0.0))
    return float(getattr(truth, channel, 0.0))


def build(summary: dict) -> dict:
    p = Params()
    comments = param_comments()

    parameters = [
        {"name": f.name,
         "value": getattr(p, f.name),
         "meaning": comments.get(f.name, ""),
         "basis": PARAM_BASIS.get(f.name, "Chosen.")}
        for f in fields(Params)
    ]

    exposure_fields = {f.name for f in fields(Exposure)}
    exposures = [
        {"channel": k,
         "agent": agent,
         "meaning": meaning,
         "value_at_match": {a: _channel_value(TRUTHS[a], k)
                            for a in ("induction", "selection", "cytotoxicity")}}
        for k, (agent, meaning) in EXPOSURE_CHANNELS.items()
        if k in exposure_fields
    ]

    checks = [{"cid": c.cid, "edge": c.edge, "statement": c.statement,
               "source": c.source, "passed": summary["checks"].get(c.cid)}
              for c in CHECKS]

    fpr = summary["false_positive_rates"]
    # The same intervals profiled under the COMBINED structure. Reporting only
    # the induction-only column would state a false-positive rate as a property
    # of fraction data when it is a property of assuming fitness parity, so the
    # two travel together into the manuscript and neither can be quoted alone.
    fpr_both = summary["false_positive_rates_both"]
    mc = summary["model_choice"]

    return {
        "parameters": parameters,
        "exposures": exposures,
        "state_groups": STATE_GROUPS,
        "checks": checks,
        "matched_pair": summary["matched_pair"],
        "degeneracy": summary["degeneracy"],
        "false_positive_rates": fpr,
        "false_positive_rates_both": fpr_both,
        "model_choice": mc,
        "washout": summary["washout"],
        "truths": summary["truths"],
        "design": summary["design"],
        "designs_spec": {k: list(v) if not isinstance(v, dict) else v
                         for k, v in DESIGNS.items()},
        "boolean": summary["boolean"],
        "boolean_tert_on": summary["boolean_tert_on"],
        "ensemble": summary["ensemble"],
        "provenance": summary.get("provenance", {}),
        "n_datasets": (summary.get("provenance") or {}).get(
            "n_datasets_per_truth_and_design"),
        "target_x": TARGET_X,
        "match_day": MATCH_DAY,
        "washout_day": WASHOUT_DAY,
        "n_checks": len(CHECKS),
        "n_checks_passed": sum(1 for c in CHECKS
                               if summary["checks"].get(c.cid)),
    }


EXPECTED_CHECKS = {f"V{i}" for i in range(1, 18)}


def audit(summary: dict) -> list[str]:
    """Reasons this summary must not be turned into manuscript text.

    ``--n-ensemble 0`` writes a well-formed summary with the ``ensemble`` key
    simply ABSENT, and still prints "16/16 checks passed" and exits 0, because
    skipped work is omitted rather than failed. The manuscript would then quote
    nominal results with no robustness qualification -- and for this build that
    matters more than most, since two of its claims (V5, V6) are only marginal
    across the ensemble and one design output is void in over half the draws.

    So the gate tests for MISSING BLOCKS and the full check roster, not for
    ``all(passed)``, which would never fire on a partial run.
    """
    problems = []

    checks = summary.get("checks") or {}
    failed = sorted(cid for cid, ok in checks.items() if not ok)
    if failed:
        problems.append(f"validation check(s) did not pass: {', '.join(failed)}")
    missing = sorted(EXPECTED_CHECKS - set(checks))
    if missing:
        problems.append(
            f"check(s) absent from the summary, i.e. skipped rather than "
            f"passed: {', '.join(missing)}")

    if not (summary.get("provenance") or {}).get("n_datasets_per_truth_and_design"):
        problems.append("provenance block missing; the dataset count behind "
                        "every reported frequency is unrecorded")

    for block in ("matched_pair", "degeneracy", "false_positive_rates",
                  "false_positive_rates_both", "washout", "model_choice",
                  "design", "boolean", "boolean_tert_on", "ensemble"):
        if summary.get(block) is None:
            problems.append(f"{block!r} block missing from the summary "
                            "(rerun without --n-ensemble 0)")

    ens = summary.get("ensemble") or {}
    if ens:
        if not ens.get("n_ok"):
            problems.append("ensemble.n_ok is 0 or absent; no draw completed")
        if ens.get("audit") is None:
            problems.append("ensemble.audit missing; V5 and V6 are marginal "
                            "and cannot be reported without it")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/output")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the tables even if the run was partial or "
                         "failed; for debugging only, never for a manuscript")
    args = ap.parse_args()

    out = Path(args.outdir)
    summary = json.loads((out / "emd3_simulation_summary.json").read_text())

    problems = audit(summary)
    if problems:
        print("REFUSING to write manuscript tables from this summary:")
        for why in problems:
            print(f"  - {why}")
        print(f"\n  The summary in {out / 'emd3_simulation_summary.json'} is "
              "from a partial or failed run.\n  Rerun "
              "`python -m emd3_simulation.run` with no skip flags, then retry.")
        if not args.allow_incomplete:
            return 1
        print("\n  --allow-incomplete given; writing anyway. DO NOT PUBLISH THIS.")

    tables = build(summary)
    path = out / "emd3_manuscript_tables.json"
    path.write_text(json.dumps(tables, indent=2, default=float))
    print(f"wrote {path}  ({len(tables['parameters'])} parameters, "
          f"{tables['n_checks']} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
