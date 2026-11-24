"""Asynchronous Boolean model of stem-like and immortalisation readouts.

The network represents the E2F2/FZD10/Wnt axis implicated in mammary
epithelium (Yang et al., 2026) using a hepatocyte plasticity network as a
structural template (Hernández-Magaña et al., 2024). One randomly selected
node updates at each step. Basin fractions average over all starting
configurations, while reachability uses a specified initial state.

The immortalisation readout requires both STEM and TERT. TERT retains
its initial value. Both TERT-off and TERT-on states are evaluated because
the MCF-10A exemplar is spontaneously immortalised.
"""

from __future__ import annotations

import numpy as np

NODES = ["E2F2", "FZD10", "WNT", "MYC", "p53", "p21", "RB", "STEM",
         "TERT", "IMMORTAL"]
N = len(NODES)
IX = {n: i for i, n in enumerate(NODES)}


def _next(s: tuple) -> tuple:
    """Return the unforced next-state values in NODES order."""
    e2f2, fzd10, wnt, myc, p53, p21, rb, stem, tert, _immortal = s
    return (
        1 - rb,                       # E2F2
        e2f2,                         # FZD10
        fzd10 & (1 - p21),            # WNT
        wnt & (1 - p53),              # MYC
        1 - myc,                      # p53
        p53,                          # p21
        p21 & (1 - myc),              # RB
        wnt & myc & (1 - rb),         # STEM
        tert,                         # TERT: no activator on the exposure path
        stem & tert,                  # IMMORTAL
    )


def rules(s: np.ndarray, forced: dict) -> np.ndarray:
    """Return the next-state values with the specified nodes held fixed."""
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
    """Return the absorbing fixed point, or the distinct states in the final 200 updates.

    The latter is a finite-trajectory approximation when the step budget is reached.
    """
    seen = []
    for i in draws:
        nxt = _next_forced(s, pairs)
        if nxt == s:                      # absorbing fixed point
            return (int("".join(map(str, s)), 2),)
        if nxt[i] != s[i]:
            s = s[:i] + (nxt[i],) + s[i + 1:]
        seen.append(int("".join(map(str, s)), 2))
    return tuple(sorted(set(seen[-200:])))


def attractors(forced: dict | None = None, n_steps: int = 4000,
               seed: int = 0, n_traj: int = 32) -> dict:
    """Estimate basin fractions over all starting states and random asynchronous updates.

    Each of the 2**N initial configurations contributes n_traj trajectories.
    For a readout to count as active, it must be on throughout the terminal
    state set. Monte Carlo errors quantify variability across update orders.
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
        stem_on = all((st >> (N - 1 - IX["STEM"])) & 1 for st in tail)
        imm_on = all((st >> (N - 1 - IX["IMMORTAL"])) & 1 for st in tail)
        out.append({"states": tail, "size": len(tail), "basin": n,
                    "basin_frac": n / total, "stem_like": bool(stem_on),
                    "immortal": bool(imm_on)})

    def _se(hits: np.ndarray) -> float:
        """Monte Carlo standard error across update orders, conditional on enumerating all starts."""
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


# The p53-loss condition is an exploratory comparator.
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
    """Estimate the probability and conditional mean time of reaching each readout.

    Each trajectory starts in a restrained, non-stem state with the specified
    TERT value. First-passage times use asynchronous update steps and are
    averaged over runs that reached the readout. If there are no hits, the
    reported mean is infinite.
    """
    pairs = _pairs(forced)
    rng = np.random.default_rng(seed)
    hits, times, imm_hits, imm_times = 0, [], 0, []

    base = [0] * N
    base[IX["p53"]] = 1
    base[IX["p21"]] = 1
    base[IX["RB"]] = 1                   # a restrained, non-stem start
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
            "reach_immortal": imm_hits / n_runs,
            "mfpt_immortal": float(np.mean(imm_times)) if imm_times else np.inf,
            "n_runs": n_runs}


SUPPORTED = {"E2F2": 1}
UNSUPPORTED = {"E2F2": 1, "p53": 0}      # exploratory comparator
