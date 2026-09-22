"""Run the EMD3 simulation.

    python -m emd3_simulation.run [--n-draws 120] [--no-figure]

Exit status is non-zero if any validation check fails.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from . import boolean as bl
from .ensemble import run_ensemble
from . import design as dz
from . import inference as inf
from .conditions import CHECKS, DESIGNS, MATCH_DAY, TARGET_X, TRUTHS
from .model import (Exposure, Params, control_fixed_point, simulate,
                    simulate_washout)

BETA_STRUCTURES = ("ind", "both")     # structures that assert a non-zero beta


def false_positive_rates(p: Params, x0: float, n_draws: int,
                         structure: str = "ind") -> dict:
    """How often does a d_beta interval exclude zero, by design and by truth.

    ``structure`` selects which model is profiled, and that choice IS the
    result. Under ``ind`` the fit enforces fitness parity and every agent looks
    plastic; under ``both`` the fitness gap is free and only the genuinely
    plastic one does. V5 reads the first, V16 the second.
    """
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=120)
    ap.add_argument("--n-ensemble", type=int, default=60,
                    help="parameter-uncertainty draws; 0 to skip")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--outdir", default="figures/output")
    args = ap.parse_args()

    p = Params()
    x0 = control_fixed_point(p)
    bar = "=" * 78
    print(bar)
    print("EMD3 mechanistic simulation   KCC8 upstream -> EMD3 (KCC4) -> KCC9, KCC10")
    print("arsenic exemplar; the deliverable is IDENTIFIABILITY and DESIGN,")
    print("not a fit -- the published exposure studies report endpoint marker")
    print("fractions, not the longitudinal absolute counts this question needs")
    print(bar)

    # --- the matched pair --------------------------------------------------
    mp = dz.matched_pair(p, TARGET_X, MATCH_DAY)
    t_long = np.linspace(0.0, 120.0, 601)
    sep = dz.separation_over_time(mp, t_long)
    im = int(np.argmin(np.abs(t_long - MATCH_DAY)))
    pair = {"x_ind_at_match": float(sep["induction"][im]),
            "x_sel_at_match": float(sep["selection"][im]),
            "d_beta": mp.induction.d_beta, "d_fit": mp.selection.d_fit}
    print(f"\n{'-' * 78}\nTwo agents, one endpoint\n{'-' * 78}")
    print(f"  induction  d_beta = {pair['d_beta']:.4f}")
    print(f"  selection  d_fit  = {pair['d_fit']:.4f}")
    print(f"  both reach x = {pair['x_ind_at_match']:.4f} at day {MATCH_DAY:.0f}")
    print(f"\n  {'day':>5}{'induction':>12}{'selection':>12}{'difference':>12}")
    for d in (7, 14, 21, 30, 45, 60, 120):
        i = int(np.argmin(np.abs(t_long - d)))
        print(f"  {d:>5}{sep['induction'][i]:>12.4f}{sep['selection'][i]:>12.4f}"
              f"{sep['diff'][i]:>12.4f}")

    # --- degeneracy --------------------------------------------------------
    deg = degeneracy_check(p, x0)
    print(f"\n{'-' * 78}\nSelection and differential death are one model\n{'-' * 78}")
    print(f"  max |x_selection(t) - x_cytotoxicity(t)| over 60 days: "
          f"{deg['max_abs_diff']:.2e}")
    print(f"  same two arms, difference in log total cell count at day 60: "
          f"{deg['logn_gap']:.2f}")
    print("  -> the fraction cannot separate them; the count separates them cleanly")

    # --- false positives ---------------------------------------------------
    print(f"\n{'-' * 78}\nDoes a d_beta interval mean anything? "
          f"({args.n_draws} draws)\n{'-' * 78}")
    fp = false_positive_rates(p, x0, args.n_draws)
    print("  All three agents are matched to the SAME marker fraction at the")
    print("  same endpoint, so any difference below is inference, not dose.")
    print(f"\n  {'design':<12}{'truth':<16}{'excludes 0':>12}{'CI width':>11}"
          f"{'centre':>9}")
    for dname in DESIGNS:
        for tname in TRUTHS:
            r = fp[dname][tname]
            flag = "" if tname == "induction" else ("   <- FALSE POSITIVE"
                                                    if r["ci_excl"] > 0.2 else "")
            print(f"  {dname:<12}{tname:<16}{r['ci_excl'] * 100:>11.0f}%"
                  f"{r['width']:>11.4f}{r['point']:>9.4f}{flag}")
    print("\n  The rate saturates at 100%; what still moves with more data is the")
    print("  WIDTH (narrower) and the CENTRE (further from zero) -- V6.")

    # --- the same data, profiled under the COMBINED structure --------------
    # V16. The false positive above is a property of ENFORCING FITNESS PARITY,
    # not of fraction data. Free the fitness gap and the same marker fractions
    # put the interval back across zero for both non-plastic agents. What that
    # costs is power, and the power cost is the honest argument for counts.
    print(f"\n{'-' * 78}\nThe same fractions, profiled under the COMBINED "
          f"structure\n{'-' * 78}")
    fp_both = false_positive_rates(p, x0, args.n_draws, structure="both")
    print(f"  {'design':<12}{'truth':<16}{'excludes 0':>12}{'CI width':>11}"
          f"{'centre':>9}")
    for dname in DESIGNS:
        for tname in TRUTHS:
            r = fp_both[dname][tname]
            flag = ("" if tname == "induction"
                    else ("   <- correctly covers 0" if r["ci_excl"] < 0.05
                          else "   <- FALSE POSITIVE"))
            print(f"  {dname:<12}{tname:<16}{r['ci_excl'] * 100:>11.0f}%"
                  f"{r['width']:>11.4f}{r['point']:>9.4f}{flag}")
    print("\n  Freeing the fitness gap removes the false positive on fractions")
    print("  ALONE. The cost is power against a real induction agent")
    print(f"  ({fp_both['short']['induction']['ci_excl'] * 100:.0f}% at the short "
          f"design, {fp_both['extended']['induction']['ci_excl'] * 100:.0f}% at the "
          f"extended one),")
    print("  which is why counts are worth paying for -- not the false-positive")
    print("  rate, which belongs to the induction-only assumption.")

    # --- counts ------------------------------------------------------------
    print(f"\n{'-' * 78}\nModel choice: fractions alone vs fractions + counts"
          f"\n{'-' * 78}")
    cnt = count_comparison(p, x0, args.n_draws)
    for tname in TRUTHS:
        c = cnt[tname]
        f_ = ", ".join(f"{k} {v * 100:.0f}%"
                       for k, v in sorted(c["frac_only"].items(),
                                          key=lambda kv: -kv[1])[:3])
        w_ = ", ".join(f"{k} {v * 100:.0f}%"
                       for k, v in sorted(c["with_counts"].items(),
                                          key=lambda kv: -kv[1])[:3])
        print(f"  truth = {tname}")
        print(f"    fractions only : {f_}")
        print(f"    with counts    : {w_}")

    # --- design ------------------------------------------------------------
    print(f"\n{'-' * 78}\nWhat experiment separates them\n{'-' * 78}")
    detect = dz.detection_day(mp)
    tit20 = dz.purity_dependence(mp, eps_hi=0.020, day=7.0)
    tit100 = dz.purity_dependence(mp, eps_hi=0.100, day=7.0)
    print(f"  watch longer  : separable from day {detect:.0f} "
          f"({detect - MATCH_DAY:.0f} d past the endpoint), no sorting")
    need80 = dz.purity_fold_for_power(mp, 0.80, day=7.0)
    rep = dz.power_vs_replication(mp, day=7.0)
    print(f"  watch longer, no sorting : separable from day {detect:.0f}")
    print(f"\n  Sorting contrast at day 7 -- power over purity separation and")
    print(f"  replication (two-sided 5% test, per-replicate SD {dz.SIGMA_BIO}):")
    print("    " + f"{'separation':<12}{'dirty arm':>11}{'contrast':>10}"
          + "".join(f"{'n=' + str(n):>7}" for n in rep["reps"]))
    for fo in rep["folds"]:
        g = rep["grid"][fo]
        print("    " + f"{str(fo) + 'x':<12}{g['eps_hi'] * 100:>10.1f}%"
              f"{g['contrast']:>10.4f}"
              + "".join(f"{g['power_by_n_rep'][n]:>7.2f}" for n in rep["reps"]))
    if need80["feasible"]:
        print(f"\n  At n = {dz.N_REP} replicates, 80% power needs {need80['fold']:.0f}x "
              f"(dirty arm at {need80['eps_hi'] * 100:.1f}% stem-like).")
    print(f"  Holding 100x and raising replication to 6 per arm reaches "
          f"{rep['grid'][100]['power_by_n_rep'][6]:.2f}.")
    print("  Both are levers the experimenter controls; one combination is not")
    print("  a property of the design.")

    # --- washout -----------------------------------------------------------
    print(f"\n{'-' * 78}\nWashout: does withdrawing the agent help?\n{'-' * 78}")
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
    print(f"  {'arm':<16}{'x at washout':>14}{'log count':>12}")
    for k, w in wash.items():
        print(f"  {k:<16}{w['x_at_washout']:>14.4f}"
              f"{w['log_total_at_washout']:>12.3f}")
    print(f"\n  max spread across arms in x(t) AFTER washout: "
          f"{wsh['max_frac_gap']:.2e}")
    print("  -> the washout arm carries no information in the fraction. Every")
    print("     arm returns to the same parameters from the same fraction, so")
    print("     the trajectories coincide. The COUNT at washout separates them")
    print(f"     (selection vs cytotoxicity differ by {wsh['logn_gap_sel_cyt']:.2f} in log count).")

    # --- boolean -----------------------------------------------------------
    print(f"\n{'-' * 78}\nBoolean attractor layer\n{'-' * 78}")
    boo, boo_on = {}, {}
    for name, forced in bl.PERTURBATIONS.items():
        a = bl.attractors(forced)
        m = bl.mean_first_passage(forced, n_runs=400, tert_start=0)
        boo[name] = {**{k: v for k, v in a.items() if k != "attractors"}, **m}
        boo_on[name] = bl.mean_first_passage(forced, n_runs=400, tert_start=1)
    print(f"  {'condition':<20}{'stem basin':>16}{'stem reach':>12}"
          f"{'MFPT':>8}{'imm. basin':>16}{'imm. reach':>12}")
    for name, b in boo.items():
        mf = "inf" if not np.isfinite(b["mfpt"]) else f"{b['mfpt']:.0f}"
        sb = f"{b['stem_basin'] * 100:.1f}+-{b['stem_basin_se'] * 100:.1f}%"
        ib = f"{b['immortal_basin'] * 100:.1f}+-{b['immortal_basin_se'] * 100:.1f}%"
        print(f"  {name:<20}{sb:>16}{b['reach_frac'] * 100:>11.1f}%{mf:>8}"
              f"{ib:>16}{b['reach_immortal'] * 100:>11.1f}%")
    print(f"\n  Basins are means over {boo['control']['n_traj']} asynchronous "
          f"trajectories per starting state, +- Monte Carlo SE. Under")
    print("  asynchronous update a basin is a PROBABILITY, not a set: one "
          "trajectory")
    print("  per start (an earlier release) carries an MC error near +-1.2 "
          "points,")
    print("  which is the third significant figure of every number in this "
          "table.")
    print("\n  Read REACHABILITY, not basin size, for the KCC9 claim: exposure")
    print("  enlarges the immortal basin while immortal reachability stays at")
    print("  exactly zero, because nothing on the exposure path turns TERT on.")

    print(f"\n  Same network, TELOMERASE-POSITIVE start (V17). MCF-10A -- the")
    print("  exemplar -- is a spontaneously immortalised line, so the TERT-off")
    print("  start above is not a validated representation of it.")
    print(f"  {'condition':<20}{'stem reach':>12}{'imm. reach':>12}")
    for name, b in boo_on.items():
        print(f"  {name:<20}{b['reach_frac'] * 100:>11.1f}%"
              f"{b['reach_immortal'] * 100:>11.1f}%")
    print("\n  With TERT already on, IMMORTAL <- stem AND tert collapses to")
    print("  IMMORTAL <- stem: the unsupported perturbation now scores")
    print("  immortalisation in every run. The KCC9 separation holds for the")
    print("  SUPPORTED perturbation under either start -- arsenic never reaches")
    print("  the stem state at all -- but it is a property of the start state")
    print("  as well as of the wiring, and must be stated that way.")

    # --- ensemble ----------------------------------------------------------
    ens = None
    if args.n_ensemble > 0:
        print(f"\n{'-' * 78}")
        print(f"Parameter uncertainty ({args.n_ensemble} draws, agents re-matched "
              f"in each)\n{'-' * 78}")
        ens = run_ensemble(n=args.n_ensemble)
        print(f"  {ens['n_ok']}/{ens['n_requested']} draws completed"
              + (f"  ({ens['n_unmatched']} unmatchable)" if ens["n_unmatched"] else ""))
        print("  Bands are sensitivity ranges over rates that were chosen, not")
        print("  measured. Not posteriors or confidence intervals.\n")
        for cid, frac in ens["audit"].items():
            mark = ("robust" if frac >= 0.95
                    else "marginal" if frac >= 0.60 else "NOT ROBUST")
            print(f"    {cid:<6}{frac * 100:>6.0f}%   {mark}")
        dd = ens["detect_days_past_endpoint"]
        print(f"\n  Of draws where the pair EVER separates within 240 d: "
              f"{ens['frac_draws_separable_within_240d'] * 100:.0f}%")
        if dd:
            print(f"    days past endpoint until separable: {dd['p50']:.1f} "
                  f"[{dd['p5']:.1f}, {dd['p95']:.1f}]")
        print("  In the remaining draws the two mechanisms never separate by more")
        print("  than measurement noise at all -- the 'watch longer' design is the")
        print("  one output of this build that parameter uncertainty can void.")
        for k, v in ens["not_audited"].items():
            print(f"    {k} not audited -- {v}")

    # --- validation --------------------------------------------------------
    ctx = {"pair": pair, "fp": fp, "fp_both": fp_both, "counts": cnt,
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
    print(f"\n{bar}\nValidation battery\n{bar}")
    results = []
    for chk in CHECKS:
        try:
            ok = bool(chk.test(ctx))
        except Exception as exc:
            ok = False
            print(f"  !! {chk.cid} raised: {exc}")
        results.append((chk, ok))
        print(f"\n[{'PASS' if ok else 'FAIL'}] {chk.cid}  {chk.edge}")
        print(f"       {chk.statement}")
        print(f"       basis: {chk.source}")
    n_pass = sum(1 for _, o in results if o)
    print(f"\n  {n_pass}/{len(results)} checks passed")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Run provenance. Without it the saved JSON cannot be told apart from a
    # smaller diagnostic run, and the table generator has nothing to check the
    # dataset count against.
    import platform
    import scipy
    provenance = {
        "n_datasets_per_truth_and_design": args.n_draws,
        "n_ensemble_draws": args.n_ensemble,
        "n_replicates_per_timepoint": inf.N_REP,
        "sigma_fraction_per_replicate": inf.SIGMA_BIO,
        "sigma_logcount_per_replicate": inf.SIGMA_LOGN,
        "likelihood": "Gaussian on replicate means; log counts projected onto "
                      "orthonormal Helmert contrasts",
        "interval": "grid profile likelihood, 2 log-likelihood units",
        "model_selection": "AICc",
        "boolean_trajectories_per_start": bl.attractors.__defaults__[2],
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
    print(f"\n  wrote {path}")

    if not args.no_figure:
        try:
            from .figure import make_figure
            make_figure(mp, sep, ctx, outdir)
        except ImportError:
            print("  (figure module not present yet; skipped)")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
