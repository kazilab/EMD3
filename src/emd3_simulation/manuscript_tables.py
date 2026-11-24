"""Export the EMD3 parameter definitions, model checks and numerical results."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path

from .conditions import CHECKS, DESIGNS, MATCH_DAY, TARGET_X, TRUTHS, WASHOUT_DAY
from .model import Exposure, Params

SRC = Path(__file__).parent / "model.py"

PARAM_BASIS = {
    "r_S": "Assigned stem-like proliferation rate.",
    "d_S": "Assigned stem-like death rate.",
    "r_N": "Assigned non-stem-like proliferation rate.",
    "d_N": "Assigned non-stem-like death rate.",
    "beta": "Assigned conversion rate from non-stem-like to stem-like cells.",
    "gamma": "Assigned differentiation rate from stem-like to non-stem-like cells. "
             "The stationary stem-like fraction depends on both transition rates "
             "and the net growth difference between states.",
}

EXPOSURE_CHANNELS = {
    "d_beta": ("induction", "Increase in the conversion rate from non-stem-like to stem-like cells."),
    "d_fit": ("selection", "Increase in the net growth difference through stem-like proliferation."),
    "kill_N": ("cytotoxicity", "Additional non-stem-like cell death."),
    "kill_S": ("unused in matched scenarios", "Additional stem-like cell death."),
    "d_gamma": ("unused in matched scenarios", "Change in the differentiation rate."),
}

STATE_GROUPS = [
    ("Population state (ODE)",
     [("N", "non-stem-like cell count (NSCC)"),
      ("S", "stem-like cell count (CSC)")]),
    ("Derived readouts",
     [("x = S/(N+S)", "stem-like fraction"),
      ("ln(N+S)", "natural logarithm of total cell count")]),
    ("Boolean layer (separate, discrete)",
     [("E2F2, FZD10, WNT, …", "regulatory nodes updated asynchronously"),
      ("stem basin", "probability of reaching the stem-like readout across initial states and update orders"),
      ("immortal basin", "probability of reaching the immortalization readout across initial states and update orders"),
      ("stem / immortal reached", "probability of reaching each readout from a restrained starting state")]),
]


def param_comments() -> dict:
    """Read parameter definitions from the field comments in model.py."""
    out: dict[str, str] = {}
    for line in SRC.read_text().splitlines():
        m = re.match(r"\s*([a-zA-Z_]\w*)\s*:\s*float\s*=\s*[^#]*#\s*(.+?)\s*$",
                     line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def _channel_value(truth, channel: str) -> float:
    """Read an exposure channel from a scenario or its JSON record."""
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
         "basis": PARAM_BASIS.get(f.name, "Assigned.")}
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

    check_results = summary.get("checks", {})
    checks = [{"cid": c.cid, "edge": c.edge, "statement": c.statement,
               "source": c.source, "passed": check_results.get(c.cid)}
              for c in CHECKS]

    return {
        "parameters": parameters,
        "exposures": exposures,
        "state_groups": STATE_GROUPS,
        "checks": checks,
        "matched_pair": summary.get("matched_pair"),
        "degeneracy": summary.get("degeneracy"),
        "false_positive_rates": summary.get("false_positive_rates"),
        "false_positive_rates_both": summary.get("false_positive_rates_both"),
        "model_choice": summary.get("model_choice"),
        "washout": summary.get("washout"),
        "truths": summary.get("truths"),
        "design": summary.get("design"),
        "designs_spec": {k: list(v) if not isinstance(v, dict) else v
                         for k, v in DESIGNS.items()},
        "boolean": summary.get("boolean"),
        "boolean_tert_on": summary.get("boolean_tert_on"),
        "ensemble": summary.get("ensemble"),
        "provenance": summary.get("provenance", {}),
        "n_datasets": (summary.get("provenance") or {}).get(
            "n_datasets_per_truth_and_design"),
        "target_x": TARGET_X,
        "match_day": MATCH_DAY,
        "washout_day": WASHOUT_DAY,
        "n_checks": len(CHECKS),
        "n_checks_passed": sum(1 for c in CHECKS
                               if check_results.get(c.cid)),
    }


EXPECTED_CHECKS = {f"V{i}" for i in range(1, 18)}


def audit(summary: dict) -> list[str]:
    """List missing results and failed model checks in a completed run."""
    problems = []

    checks = summary.get("checks") or {}
    failed = sorted(cid for cid, ok in checks.items() if not ok)
    if failed:
        problems.append(f"failed model checks: {', '.join(failed)}")
    missing = sorted(EXPECTED_CHECKS - set(checks))
    if missing:
        problems.append(f"missing model checks: {', '.join(missing)}")

    if not (summary.get("provenance") or {}).get("n_datasets_per_truth_and_design"):
        problems.append("number of datasets per mechanism and schedule is missing")

    for block in ("matched_pair", "degeneracy", "false_positive_rates",
                  "false_positive_rates_both", "washout", "model_choice",
                  "design", "boolean", "boolean_tert_on", "ensemble"):
        if summary.get(block) is None:
            problems.append(f"missing results for {block}")

    ens = summary.get("ensemble") or {}
    if ens:
        if not ens.get("n_ok"):
            problems.append("no completed sensitivity draw recorded")
        if ens.get("audit") is None:
            problems.append("sensitivity results for the model checks are missing")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures/output")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="export available results from a partial run")
    args = ap.parse_args()

    out = Path(args.outdir)
    summary = json.loads((out / "emd3_simulation_summary.json").read_text())

    problems = audit(summary)
    if problems:
        print("Incomplete simulation summary:")
        for why in problems:
            print(f"  - {why}")
        if not args.allow_incomplete:
            return 1

    tables = build(summary)
    path = out / "emd3_manuscript_tables.json"
    path.write_text(json.dumps(tables, indent=2, default=float))
    print(f"wrote {path}  ({len(tables['parameters'])} parameters, "
          f"{tables['n_checks']} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
