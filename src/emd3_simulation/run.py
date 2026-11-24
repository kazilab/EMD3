"""Run the EMD3 synthetic-data simulation and its model checks."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from . import FIGURE_FORMATS, FIGURE_STEM
from . import boolean as bl
from .ensemble import run_ensemble
from . import design as dz
from . import inference as inf
from .conditions import CHECKS, DESIGNS, MATCH_DAY, TARGET_X, TRUTHS
from .model import (Exposure, Params, control_fixed_point, simulate,
                    simulate_washout)

BETA_STRUCTURES = ("ind", "both")  # models allowing an exposure-induced conversion rate


def false_positive_rates(p: Params, x0: float, n_draws: int,
                         structure: str = "ind") -> dict:
    """Summarize conversion-rate profile intervals by sampling schedule and mechanism."""
    out = {}
    for dname, t in DESIGNS.items():
        out[dname] = {}
        for tname, e in TRUTHS.items():
            excl, widths, points, censored = 0, [], [], 0
            los, his = [], []
            for s in range(n_draws):
                rng = np.random.default_rng(s)
                y = inf.observe(e.apply(p), t, x0, rng=rng)
                r = inf.beta_interval(p, t, y, x0, structure=structure)
                excl += r["excludes_zero"]
                censored += r["right_censored"]
                widths.append(r["width"])
                points.append(r["point"])
                los.append(r["lo"]); his.append(r["hi"])
            out[dname][tname] = {"ci_excl": excl / n_draws,
                                 "width": float(np.median(widths)),
                                 "point": float(np.median(points)),
                                 "lo": float(np.median(los)),
                                 "hi": float(np.median(his)),
                                 "right_censored": censored / n_draws,
                                 "structure": structure}
    return out


def count_comparison(p: Params, x0: float, n_draws: int) -> dict:
    """Model choice with fractions alone vs fractions plus absolute counts."""
    t = DESIGNS["extended"]
    out = {}
    for tname, e in TRUTHS.items():
        frac, cnt = collections.Counter(), collections.Counter()
        for s in range(n_draws):
            rng = np.random.default_rng(s)
            y = inf.observe(e.apply(p), t, x0, rng=rng)
            frac[inf.compare(p, t, y, x0)["best"]] += 1
            rng = np.random.default_rng(s)
            d = inf.observe_with_counts(e.apply(p), t, x0, rng=rng)
            cnt[inf.compare_with_counts(p, t, d, x0)["best"]] += 1
        out[tname] = {
            "frac_only": {k: v / n_draws for k, v in frac.items()},
            "with_counts": {k: v / n_draws for k, v in cnt.items()},
            "picks_ind": cnt["ind"] / n_draws,
            "picks_cyto": cnt["cyto"] / n_draws,
            "picks_sel": cnt["sel"] / n_draws,
            "picks_beta": sum(cnt[k] for k in BETA_STRUCTURES) / n_draws,
            "frac_picks_ind": frac["ind"] / n_draws,
            "frac_picks_beta": sum(frac[k] for k in BETA_STRUCTURES) / n_draws,
        }
    return out


def degeneracy_check(p: Params, x0: float) -> dict:
    """Show that a growth advantage and differential death give identical x(t).

    Matched so that (a - b) changes by the same amount through r_S in one case
    and through d_N in the other.
    """
    t = np.linspace(0.0, 60.0, 301)
    delta = 0.15
    x_sel = simulate(Exposure(d_fit=delta).apply(p), t, x0=x0)["x"]
    x_cyt = simulate(Exposure(kill_N=delta).apply(p), t, x0=x0)["x"]
    n_sel = simulate(Exposure(d_fit=delta).apply(p), t, x0=x0)["log_total"]
    n_cyt = simulate(Exposure(kill_N=delta).apply(p), t, x0=x0)["log_total"]
    return {"t": t, "x_sel": x_sel, "x_cyt": x_cyt,
            "max_abs_diff": float(np.max(np.abs(x_sel - x_cyt))),
            "logn_sel": n_sel, "logn_cyt": n_cyt,
            "logn_gap": float(np.abs(n_sel - n_cyt)[-1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=120)
    ap.add_argument("--n-ensemble", type=int, default=60,
                    help="sensitivity draws (0 to skip)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--outdir", default="figures/output")
    args = ap.parse_args()
    if args.n_draws < 1 or args.n_ensemble < 0:
        ap.error("--n-draws must be positive and --n-ensemble must be nonnegative")

    p = Params()
    x0 = control_fixed_point(p)
    print("Matching induction and selection to a common endpoint", flush=True)
    mp = dz.matched_pair(p, TARGET_X, MATCH_DAY)
    t_long = np.linspace(0.0, 120.0, 601)
    sep = dz.separation_over_time(mp, t_long)
    im = int(np.argmin(np.abs(t_long - MATCH_DAY)))
    pair = {"x_ind_at_match": float(sep["induction"][im]),
            "x_sel_at_match": float(sep["selection"][im]),
            "d_beta": mp.induction.d_beta, "d_fit": mp.selection.d_fit}
    deg = degeneracy_check(p, x0)
    unit = "dataset" if args.n_draws == 1 else "datasets"
    print(f"Fitting {args.n_draws} synthetic {unit} per mechanism and schedule", flush=True)
    fp = false_positive_rates(p, x0, args.n_draws)
    fp_both = false_positive_rates(p, x0, args.n_draws, structure="both")
    cnt = count_comparison(p, x0, args.n_draws)
    print("Calculating design power and washout trajectories", flush=True)
    detect = dz.detection_day(mp)
    tit20 = dz.purity_dependence(mp, eps_hi=0.020, day=7.0)
    tit100 = dz.purity_dependence(mp, eps_hi=0.100, day=7.0)
    need80 = dz.purity_fold_for_power(mp, 0.80, day=7.0)
    rep = dz.power_vs_replication(mp, day=7.0)
    t_on = np.linspace(0.0, MATCH_DAY, 401)
    t_off = np.linspace(0.0, 60.0, 601)
    wash = {k: simulate_washout(p, e, t_on, t_off, x0=x0)
            for k, e in TRUTHS.items()}
    xs = np.vstack([w["x_after"] for w in wash.values()])
    wsh = {"max_frac_gap": float(np.max(xs.max(axis=0) - xs.min(axis=0))),
           "logn_gap_sel_cyt": float(abs(wash["selection"]["log_total_at_washout"]
                                         - wash["cytotoxicity"]["log_total_at_washout"])),
           "x_at_washout": {k: w["x_at_washout"] for k, w in wash.items()},
           "log_total_at_washout": {k: w["log_total_at_washout"]
                                    for k, w in wash.items()}}
    print("Estimating Boolean basins and reachability", flush=True)
    boo, boo_on = {}, {}
    for name, forced in bl.PERTURBATIONS.items():
        a = bl.attractors(forced)
        m = bl.mean_first_passage(forced, n_runs=400, tert_start=0)
        boo[name] = {**{k: v for k, v in a.items() if k != "attractors"}, **m}
        boo_on[name] = bl.mean_first_passage(forced, n_runs=400, tert_start=1)
    ens = None
    if args.n_ensemble > 0:
        print(f"Running {args.n_ensemble} parameter-sensitivity draws", flush=True)
        ens = run_ensemble(n=args.n_ensemble)
        print(f"Completed {ens['n_ok']}/{ens['n_requested']} sensitivity draws")

    ctx = {"n_datasets": args.n_draws, "pair": pair, "fp": fp, "fp_both": fp_both, "counts": cnt,
           "degeneracy": deg, "bool": boo, "washout": wsh,
           "bool_tert_on": boo_on,
           "design": {"detect_day": detect,
                      "titration_20x": tit20["separable"],
                      "titration_100x": tit100["separable"],
                      "titration_20x_contrast": tit20["contrast"],
                      "titration_100x_contrast": tit100["contrast"],
                      "titration_20x_power": tit20["power"],
                      "titration_100x_power": tit100["power"],
                      "titration_20x_power_upper_tail": tit20["power_upper_tail"],
                      "titration_100x_power_upper_tail": tit100["power_upper_tail"],
                      "power_100x_n6": rep["grid"][100]["power_by_n_rep"][6],
                      "power_vs_replication": rep,
                      "n_rep": dz.N_REP,
                      "titration_20x_fold": tit20["purity_fold"],
                      "titration_100x_fold": tit100["purity_fold"],
                      "fold_for_80pct_power": (need80["fold"] if need80["feasible"]
                                               else float("inf")),
                      "eps_hi_for_80pct_power": need80.get("eps_hi"),
                      "titration_noise_floor": tit20["noise_floor"],
                      "sigma_bio": dz.SIGMA_BIO,
                      "detect_search_horizon_days": 200.0}}
    results = []
    for chk in CHECKS:
        try:
            ok = bool(chk.test(ctx))
        except Exception as exc:
            ok = False
            print(f"  !! {chk.cid} raised: {exc}")
        results.append((chk, ok))
        if not ok:
            print(f"Failed {chk.cid}: {chk.statement}")
    n_pass = sum(1 for _, o in results if o)
    print(f"{n_pass}/{len(results)} model checks passed")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import platform
    import scipy
    provenance = {
        "n_datasets_per_truth_and_design": args.n_draws,
        "n_ensemble_draws": args.n_ensemble,
        "n_replicates_per_timepoint": inf.N_REP,
        "sigma_fraction_per_replicate": inf.SIGMA_BIO,
        "sigma_logcount_per_replicate": inf.SIGMA_LOGN,
        "likelihood": "Gaussian on replicate means with log counts projected onto "
                      "orthonormal Helmert contrasts",
        "interval": "grid profile likelihood, 2 log-likelihood units",
        "model_selection": "AICc",
        "boolean_trajectories_per_start": boo["control"]["n_traj"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    summary = {
        "provenance": provenance,
        "matched_pair": pair,
        "degeneracy": {"max_abs_diff_fraction": deg["max_abs_diff"],
                       "log_count_gap_day60": deg["logn_gap"]},
        "false_positive_rates": {d: {t: fp[d][t] for t in TRUTHS}
                                 for d in DESIGNS},
        "false_positive_rates_both": {d: {t: fp_both[d][t] for t in TRUTHS}
                                      for d in DESIGNS},
        "washout": wsh,
        "truths": {k: {f: getattr(e, f) for f in
                       ("d_beta", "d_fit", "kill_N", "kill_S", "d_gamma")}
                   for k, e in TRUTHS.items()},
        "model_choice": {t: {"fractions_only": cnt[t]["frac_only"],
                             "with_counts": cnt[t]["with_counts"]}
                         for t in TRUTHS},
        "design": ctx["design"],
        "boolean_tert_on": {k: {kk: (None if not np.isfinite(vv) else vv)
                                if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
                            for k, v in boo_on.items()},
        "boolean": {k: {kk: (None if not np.isfinite(vv) else vv)
                        if isinstance(vv, float) else vv
                        for kk, vv in v.items()} for k, v in boo.items()},
        "checks": {c.cid: bool(o) for c, o in results},
    }
    if ens is not None:
        summary["ensemble"] = ens
    path = outdir / "emd3_simulation_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Simulation summary saved to {path}")

    # The tables are rebuilt separately from the new summary.
    (outdir / "emd3_manuscript_tables.json").unlink(missing_ok=True)
    for ext in FIGURE_FORMATS:
        (outdir / f"figure2e_emd3_simulation.{ext}").unlink(missing_ok=True)
    if args.no_figure:
        for ext in FIGURE_FORMATS:
            (outdir / f"{FIGURE_STEM}.{ext}").unlink(missing_ok=True)
        (outdir / f"{FIGURE_STEM}_legend.md").unlink(missing_ok=True)
    else:
        from .figure import make_figure
        make_figure(mp, sep, ctx, outdir)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
