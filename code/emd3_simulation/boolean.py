"""Boolean attractor layer: does the exposure make a stem-like state accessible?

A compact regulatory network for the axis the exemplar study implicates
(E2F2 -> FZD10 -> Wnt/beta-catenin), embedded in the tumour-suppressor context
that the published hepatocyte plasticity model uses (Hernandez-Magana 2024,
npj Syst Biol Appl 10:99 -- reference 47): p53, p21 and RB restraining a
proliferative stem-like attractor.

The point of this layer is NOT to show that some in-silico perturbation can
create a stem-like attractor -- almost any perturbation of a small network can.
It is to ask whether perturbing ONLY the nodes the exposure is experimentally
reported to touch changes the attractor structure, and by how much. The
manuscript is explicit that the ability of an arbitrary perturbation to
stabilise a stem-like attractor establishes nothing.

Update is asynchronous: one randomly chosen node per step. Synchronous update on
a network this small produces spurious limit cycles that are artefacts of the
update scheme rather than features of the biology.

Two consequences of asynchrony are handled explicitly rather than assumed away:

* **A basin is a probability, not a set.** The same initial state can reach
  different attractors depending on the update order, so ``attractors`` runs
  each start several times and reports a Monte Carlo standard error beside each
  basin fraction. A single trajectory per start carries an MC error near
  +-0.012, which is the third significant figure of every basin number.
* **A fixed point is absorbing.** When the full next-state map agrees with the
  current state no further update can change anything, so the trajectory is cut
  there. That is exact rather than a step budget, and it is what makes the
  repeated sampling affordable.

A third point is a limitation rather than a feature. Every reachability number
here is conditional on a CHOSEN START STATE as well as on the wiring, and the
KCC9 result is conditional on TERT starting off. The exemplar line, MCF-10A, is
spontaneously immortalised, so that start does not describe it; with TERT on,
``IMMORTAL <- stem AND tert`` collapses to ``IMMORTAL <- stem`` and stemness
scores as immortalisation one for one. ``mean_first_passage`` takes
``tert_start`` and the run reports both, because a result that holds only under
an unstated assumption should not be presented as a property of the network.
"""

from __future__ import annotations

import numpy as np

# TERT and IMMORTAL encode the KCC9 rule structurally rather than editorially.
# Nothing on the exposure path activates TERT, and IMMORTAL requires it, so a
# stem-like state can never by itself move the immortalisation readout. The
# refined KCC framework treats senescence/immortalisation as one axis and
# stemness as a different property; making that a wiring constraint means the
# model cannot accidentally score stemness as evidence of KCC9.
NODES = ["E2F2", "FZD10", "WNT", "MYC", "p53", "p21", "RB", "STEM",
         "TERT", "IMMORTAL"]
N = len(NODES)
IX = {n: i for i, n in enumerate(NODES)}


def _next(s: tuple) -> tuple:
    """Unforced next state, as a plain tuple of 0/1.

    This is the canonical statement of the biology and the only place the logic
    is written. ``rules`` wraps it for the array-based API; the trajectory loops
    call it directly, because building a dict and a numpy array per asynchronous
    step dominated the runtime and made repeated sampling unaffordable.
    ``test_fast_and_canonical_rules_agree`` pins the two together over all 2^N
    states.
    """
    e2f2, fzd10, wnt, myc, p53, p21, rb, stem, tert, _immortal = s
    return (
        1 - rb,                       # E2F2
        e2f2,                         # FZD10
        fzd10 & (1 - p21),            # WNT
        wnt & (1 - p53),              # MYC
        1 - myc,                      # p53
        p53,                          # p21
        p21 & (1 - myc),              # RB
        # the stem-like readout: Wnt-driven, MYC-supported, RB-restrained
        wnt & myc & (1 - rb),         # STEM
        tert,                         # TERT: no activator on the exposure path
        stem & tert,                  # IMMORTAL
    )


def rules(s: np.ndarray, forced: dict) -> np.ndarray:
    """Next-state function. ``forced`` pins nodes the exposure clamps."""
    out = np.array(_next(tuple(int(v) for v in s)), dtype=np.int8)
    for k, v in forced.items():
        out[IX[k]] = int(v)
    return out


def _pairs(forced: dict | None) -> tuple:
    return tuple((IX[k], int(v)) for k, v in (forced or {}).items())


def _next_forced(s: tuple, pairs: tuple) -> tuple:
    t = _next(s)
    if pairs:
        t = list(t)
        for i, v in pairs:
            t[i] = v
        t = tuple(t)
    return t


def _apply_forced(s: np.ndarray, forced: dict) -> np.ndarray:
    s = s.copy()
    for k, v in forced.items():
        s[IX[k]] = int(v)
    return s


def _settle(s: tuple, pairs: tuple, draws: np.ndarray) -> tuple:
    """Run one asynchronous trajectory and return the attractor it settles in.

    ``draws`` is the pre-drawn sequence of node indices; drawing them in one
    block rather than per step is a large part of the speedup. Steps on which
    the chosen node does not change still count, exactly as before, so first-
    passage step counts stay comparable with earlier releases.
    """
    seen = []
    for i in draws:
        nxt = _next_forced(s, pairs)
        if nxt == s:                      # fixed point: absorbing, exactly
            return (int("".join(map(str, s)), 2),)
        if nxt[i] != s[i]:
            s = s[:i] + (nxt[i],) + s[i + 1:]
        seen.append(int("".join(map(str, s)), 2))
    # Not a fixed point within the budget: fall back to the tail heuristic, so
    # a cyclic attractor introduced by a future rule change is still reported
    # rather than silently mis-assigned.
    return tuple(sorted(set(seen[-200:])))


def attractors(forced: dict | None = None, n_steps: int = 4000,
               seed: int = 0, n_traj: int = 32) -> dict:
    """Enumerate attractors and their basins over all 2^N initial states.

    Each of the 2^N starts is run ``n_traj`` times. Under asynchronous update a
    start does not have *an* attractor, it has a distribution over attractors,
    so the basin fraction is a mean over trajectories and carries Monte Carlo
    error. An earlier release ran one trajectory per start, which gave every
    basin fraction a binomial MC error near +-0.012 -- and those numbers were
    quoted to three significant figures in the manuscript. The returned
    ``*_basin_se`` are that error, and only states whose outcome actually
    depends on the update order contribute to it.

    ``n_traj = 32`` puts the MC error near +-0.002, so a basin quoted to one
    decimal place as a percentage is meaningful. The whole enumeration costs
    about a second per condition: cutting each trajectory at the first
    absorbing fixed point, rather than always running the full step budget,
    more than pays for the repeats.
    """
    pairs = _pairs(forced)
    rng = np.random.default_rng(seed)
    basins: dict = {}
    stem_hits = np.zeros(2 ** N)
    imm_hits = np.zeros(2 ** N)

    for code in range(2 ** N):
        s0 = tuple((code >> i) & 1 for i in range(N))
        if pairs:
            t = list(s0)
            for i, v in pairs:
                t[i] = v
            s0 = tuple(t)
        for _ in range(n_traj):
            tail = _settle(s0, pairs, rng.integers(N, size=n_steps))
            basins[tail] = basins.get(tail, 0) + 1
            stem_on = all((st >> (N - 1 - IX["STEM"])) & 1 for st in tail)
            imm_on = all((st >> (N - 1 - IX["IMMORTAL"])) & 1 for st in tail)
            stem_hits[code] += stem_on
            imm_hits[code] += imm_on

    total = (2 ** N) * n_traj
    out = []
    for tail, n in sorted(basins.items(), key=lambda kv: -kv[1]):
        # an attractor counts as stem-like if STEM is on throughout it
        stem_on = all((st >> (N - 1 - IX["STEM"])) & 1 for st in tail)
        imm_on = all((st >> (N - 1 - IX["IMMORTAL"])) & 1 for st in tail)
        out.append({"states": tail, "size": len(tail), "basin": n,
                    "basin_frac": n / total, "stem_like": bool(stem_on),
                    "immortal": bool(imm_on)})

    def _se(hits: np.ndarray) -> float:
        """MC standard error of the basin mean over update orders.

        Per start i the outcome is Bernoulli(p_i) over update order; the basin
        is the mean of p_i over the 2^N starts, which are enumerated
        exhaustively and so contribute no sampling error of their own. Starts
        that always land in the same place contribute nothing.
        """
        if n_traj < 2:
            return float("nan")
        ph = hits / n_traj
        return float(np.sqrt(np.sum(ph * (1.0 - ph) / (n_traj - 1)))
                     / (2 ** N))

    return {"attractors": out,
            "stem_basin": float(stem_hits.sum() / total),
            "immortal_basin": float(imm_hits.sum() / total),
            "stem_basin_se": _se(stem_hits),
            "immortal_basin_se": _se(imm_hits),
            "n_attractors": len(out),
            "n_traj": n_traj}


# The exposure-path perturbations the exemplar and plan call for, as clamps.
# FZD10 and WNT knockdowns test whether the arsenic effect runs through the
# pathway it is attributed to; p53 loss is included ONLY as the unsupported
# comparator and must never be presented as an exposure effect.
PERTURBATIONS = {
    "control": {},
    "arsenic": {"E2F2": 1},
    "arsenic_FZD10_kd": {"E2F2": 1, "FZD10": 0},
    "arsenic_WNT_inhib": {"E2F2": 1, "WNT": 0},
    "E2F2_kd": {"E2F2": 0},
    "unsupported": {"E2F2": 1, "p53": 0},
}


def mean_first_passage(forced: dict | None = None, n_runs: int = 300,
                       max_steps: int = 4000, seed: int = 1,
                       tert_start: int = 0) -> dict:
    """First-passage statistics into the stem-like and immortal readouts.

    From a restrained, non-stem, telomerase-negative start. Steps are
    asynchronous node updates, not time -- the quantity is comparable between
    conditions but has no physical unit, and should not be reported as one.

    **Both readouts are per-RUN rates, and the immortal one is now able to
    fire.** An earlier release incremented the immortal tally once per STEP and
    broke out of the loop the moment STEM was reached. Since IMMORTAL requires
    STEM, the tally could never fire: ``reach_immortal`` read exactly 0.0
    whatever the wiring did, so the check that guards the KCC9 claim -- the
    claim this build cares most about -- had no power at all. The two readouts
    are now recorded independently, per run, and the trajectory continues until
    both have been seen or the state is absorbed.
    ``test_the_immortal_guard_can_actually_fail`` rewires TERT onto the exposure
    path and asserts the statistic then reports a non-zero value. A guard that
    cannot fail is not a guard.
    """
    pairs = _pairs(forced)
    rng = np.random.default_rng(seed)
    hits, times, imm_hits, imm_times = 0, [], 0, []

    base = [0] * N
    base[IX["p53"]] = 1
    base[IX["p21"]] = 1
    base[IX["RB"]] = 1                   # a restrained, non-stem start
    # ``tert_start`` is a MODELLING CHOICE, not a property of the network, and
    # the KCC9 result turns on it. With TERT off, IMMORTAL <- stem AND tert can
    # never fire and immortalisation is unreachable whatever the exposure does.
    # With TERT already on, the same rule collapses to IMMORTAL <- stem, and
    # anything that reaches the stem-like state scores as immortalisation 1:1.
    # This matters for the exemplar: MCF-10A is a spontaneously IMMORTALISED
    # line, so a telomerase-negative start is not a validated representation of
    # it. Both starts are therefore reported. See V3, V15 and V17.
    base[IX["TERT"]] = int(tert_start)

    for _ in range(n_runs):
        s = list(base)
        for i, v in pairs:
            s[i] = v
        s = tuple(s)
        stem_step = imm_step = None
        for step, i in enumerate(rng.integers(N, size=max_steps)):
            if stem_step is None and s[IX["STEM"]] == 1:
                stem_step = step
            if imm_step is None and s[IX["IMMORTAL"]] == 1:
                imm_step = step
            if stem_step is not None and imm_step is not None:
                break
            nxt = _next_forced(s, pairs)
            if nxt == s:
                break                    # absorbed: neither readout can move
            if nxt[i] != s[i]:
                s = s[:i] + (nxt[i],) + s[i + 1:]
        if stem_step is not None:
            hits += 1
            times.append(stem_step)
        if imm_step is not None:
            imm_hits += 1
            imm_times.append(imm_step)

    return {"tert_start": int(tert_start),
            "reach_frac": hits / n_runs,
            "mfpt": float(np.mean(times)) if times else np.inf,
            # KCC9 discipline: from a start with no telomere maintenance, no
            # amount of stem-like drive can reach the immortalisation readout,
            # because TERT has no activator on the exposure path. This is 0
            # under every perturbation in PERTURBATIONS -- and, unlike the
            # earlier implementation, it is 0 because of the wiring and would
            # become non-zero if the wiring changed.
            "reach_immortal": imm_hits / n_runs,
            "mfpt_immortal": float(np.mean(imm_times)) if imm_times else np.inf,
            "n_runs": n_runs}


# Perturbations the exemplar study actually supports: arsenic raises E2F2, which
# drives FZD10. Nothing else is clamped -- adding p53 loss would certainly open
# the stem attractor, and would be exactly the arbitrary perturbation the
# manuscript warns against.
SUPPORTED = {"E2F2": 1}
UNSUPPORTED = {"E2F2": 1, "p53": 0}      # shown for contrast, not as evidence
