# EMD3 — data-confrontation record

Companion to [`PROVENANCE.md`](PROVENANCE.md), which records what the sources
contain. This file records what happened when the model was confronted with
them and with its own uncertainty — including the bug that was found in the
check this build exists to make.

---

# Pass 1 — the deliverable had to be reframed (2026-08-23)

The plan flagged EMD3 as carrying the highest data risk of the three, and the
risk materialised. The published exposure studies report **endpoint marker
fractions**; the question — induction or selection? — needs **longitudinal
composition with absolute counts**. No such series is publicly available for
arsenic and stemness.

Rather than dress a fit up out of endpoint data, the deliverable was reframed to
**identifiability plus experimental design**. Most checks in the battery pass by
*detecting a failure*, which is the intended shape of the result.

---

# Pass 2 — the negative control was under-dosed (2026-08-25)

## The bug

`conditions.py` declared three agents that are supposed to produce the **same
marker fraction at the same endpoint** by three different mechanisms. Two did.
The third did not:

| channel | hard-coded | `match_at` says | x(14) achieved |
|---|---|---|---|
| `d_beta` (induction) | 0.0324 | 0.0324 | 0.1999 |
| `d_fit` (selection) | 0.2042 | 0.2042 | 0.2000 |
| **`kill_N` (cytotoxicity)** | **0.1150** | **0.2042** | **0.1095** |

The cytotoxic magnitude had gone stale. It reached x = 0.11 against a target of
0.20 — a little over half the intended effect.

**The value should have been identical to the selection magnitude, and that is
provable rather than empirical.** `d_fit` acts on r_S and `kill_N` acts on d_N;
the observed fraction depends on the two net growth rates only through a − b, so
the same shift on either channel is the same perturbation. That the two matched
magnitudes coincide is V7 showing up in the constants themselves.

## Why it mattered

The cytotoxic agent is the **negative control** — the plan calls it "the single
most important check". Under-dosing it made the build understate its own
headline:

| design | under-dosed | matched |
|---|---|---|
| short | 26% | **100%** |
| extended | 90% | **100%** |

A purely cytotoxic agent, carrying no plasticity whatsoever, yields a Δβ
interval excluding zero in **every replicate at both designs**. The reported
26% was an artefact of a control that had been given half the intended dose.

## What it propped up

**V6 passed only because of the bug.** It asserted, strictly, that the
false-positive rate is higher in the extended design than the short one. With a
correctly matched control both are 100%, so there is no headroom for a strict
inequality and the check fails as written.

The *claim* survives — its stated basis was always that more data tightens an
interval around the wrong value — but it needed a statistic with headroom:

| design | median Δβ CI width | median centre |
|---|---|---|
| short | 0.0120 | 0.0193 |
| extended | **0.0100** | **0.0266** |

The interval narrows while its centre moves *further* from zero. **V6 is
restated on width and centre.** This is the same shape of correction as EMD2's
V8: a claim that was true, tested through a statistic that had quietly
saturated.

## The fix

The three agents are now **computed by `match_at` at import time** rather than
written down. A stale constant of this kind cannot recur, and a regression test
asserts all three reach the target and that the selection and cytotoxicity
magnitudes coincide.

## What the correction improved

Beyond the headline, the corrected control makes several outputs coherent that
were previously asymmetric for no reason:

- selection and cytotoxicity now return **identical** interval widths and
  centres, as two names for one model must;
- model choice on fractions alone becomes a clean coin flip between the two
  identical structures (≈55/45 either way) instead of an asymmetric three-way
  split;
- V8 strengthens: with counts, the matched cytotoxic agent is identified as
  cytotoxic in 99–100% of replicates.

---

# Pass 3 — three plan expectations that the model refutes

Each was implemented and each failed. All three are now checks that pass by
recording the failure.

## The washout arm carries no information (V12)

The plan lists a washout/reversal arm among the held-out conditions. It is
implemented, and on marker fractions it discriminates **nothing**: the three
arms follow post-washout trajectories identical to 1×10⁻¹².

The reason is structural, not numerical. After withdrawal every arm returns to
the **same baseline parameters**, and the matched arms are at the **same
fraction** when the agent is removed. The subsequent trajectory is then a
function of state and parameters alone, so the arms coincide by construction.

Only the **count accumulated during exposure** separates them — the arms differ
by ~2.9 in log total cell number at washout. Recorded so the experiment is not
run in the expectation that the fraction will resolve it.

## MFPT does not fall with dose (V14)

The plan's draft battery expects mean first-passage time into the stem-like
attractor to decrease with dose. In this network it **cannot**: MFPT is infinite
under both control and the supported arsenic perturbation, because the
perturbation enlarges the stem-like basin **without opening a route into it**
from a tumour-suppressor-intact start. That is V2 restated as a rate.

Only the *unsupported* perturbation (p53 loss) gives a finite MFPT. A build that
reported a dose-dependent MFPT here would have had to reach for a perturbation
the exposure is not shown to produce — precisely the move the manuscript warns
against.

## Pathway knockdown abolishes rather than restores (V13)

The plan asks whether reversing the perturbed node restores attractor structure.
It does not restore it, it removes it:

> Basin figures in this pass are the single-trajectory estimates superseded in
> Pass 5; the current values, with Monte Carlo error, are 19.6 ± 0.2% and
> 27.5 ± 0.2%. The abolition result is unaffected — the knockdowns give exactly
> zero.

| condition | stem-like basin |
|---|---|
| control | 19.9% |
| arsenic (E2F2 on) | 28.4% |
| arsenic + FZD10 knockdown | **0.0%** |
| arsenic + WNT inhibition | **0.0%** |
| E2F2 knockdown alone | 0.0% |

The knockdowns overshoot baseline to zero. Epistasis on this network is not a
simple reversal, and the distinction matters when interpreting a rescue
experiment: abolition of the stem basin is not evidence that the exposure effect
was reversed.

---

# Pass 4 — a misreading trap, and uncertainty

## The immortalisation basin moves; reachability does not (V15)

Arsenic raises the **immortal basin** from 10.2% to 14.3% (single-trajectory
estimates; Pass 5 supersedes these with 9.7 ± 0.1% and 13.6 ± 0.1%). Quoted
alone, that
reads as an exposure promoting immortalisation — a KCC9 claim this build exists
to refuse.

It is an artefact of what a basin measures. Every attractor in that basin is
entered only from a start state where telomere maintenance is **already on**,
and nothing on the exposure path can switch TERT on. **Immortal reachability
from a TERT-off start is exactly zero under every perturbation**, supported and
unsupported alike.

Reachability, not basin size, is the statistic that carries the KCC9 claim. V15
asserts both halves so the trap cannot be walked into.

## Parameter uncertainty

`ensemble.py` perturbs the six baseline rates (σ 0.25–0.50) and re-scores the
movable claims. The agents are **re-matched inside every draw** — holding the
magnitudes fixed while perturbing the rates would silently unmatch them, which
is the Pass 2 bug in another form.

Two results are **algebraic and are not sampled**: the selection /
differential-death degeneracy (x depends on the net rates only through a − b)
and the KCC9 wiring (a property of the network, not of the rates). They cannot
move at any parameter value.

Draws are matched to a fixed **fold increase over each draw's own control**
rather than to an absolute fraction. Across draws the unexposed stem-like
fraction ranges from a few per cent to over half, and "raise x to 0.20" is not
an exposure at all when the control already sits above it.

Across 60 draws (60/60 completed, none unmatchable):

| claim | holds in | verdict |
|---|---|---|
| V5 an induction-only Δβ interval is not evidence of plasticity | 73% | **marginal** |
| V6 more data makes the wrong answer more confident | 70% | **marginal** |
| V8 absolute counts break the degeneracy | 97% | robust |
| V12 washout carries no information | 100% | robust (algebraic) |

**The design output is the one thing uncertainty can void, and it is worse than
expected.** In only **48%** of draws do the induction and selection arms ever
separate by more than measurement noise **within 240 days**. When they do, they
do so quickly — 5.2 days past the endpoint, 5th–95th percentile 3.5–26.9 — but
in more than half the sampled parameter space the "watch longer" recommendation
simply does not work at any follow-up length.

That is a substantive qualification on this build's headline design output, and
it strengthens the case for the sort-based design, whose discriminating power
does not depend on the two asymptotes being far apart.

V5 and V6 at ~70% are reported as marginal rather than robust. Both are
statements about a *misspecified* fit, and under draws where the control
fraction sits high or the transition rates are fast the misspecification is less
severe, so the false positive is less reliable. The nominal result stands; its
generality does not.

---

# Pass 5 — a guard with no power, and three quieter errors (2026-09-21)

Prompted by an external review of the build. Four defects were found; one of
them is the same *class* of error as Pass 2, in the check this build cares most
about.

## The KCC9 reachability guard could not fail

`mean_first_passage` incremented the immortal tally once per **step** and broke
out of the loop the moment `STEM` was reached. `IMMORTAL` requires `STEM`, so
the tally could never fire: `reach_immortal` read exactly `0.0` **whatever the
wiring did**.

The claim it guards is true — nothing on the exposure path activates TERT, which
is verifiable by inspecting `rules` — but `V3` and `V15`, the checks that assert
it, and `test_exposure_can_never_reach_the_immortal_readout`, all had **no
power**. The number was right for the wrong reason, and the code comment beside
it ("if it is ever non-zero, the wiring constraint has leaked") was false.

Demonstrated by deliberately breaking the constraint — rewiring `TERT ← tert or
wnt` so the exposure path activates telomerase:

| | intact wiring | **leaky wiring** |
|---|---|---|
| immortal basin, unsupported | 50.0% | **100.0%** |
| trajectory reaches IMMORTAL by hand | no | **yes, step 58** |
| **`reach_immortal` as the build reported it** | 0.0000 | **0.0000** |

The fix records both readouts per run, independently of each other, and runs
until both are seen or the state is absorbed. On the leaky network the statistic
now reports `1.0000`. `test_the_immortal_guard_can_actually_fail` performs that
rewiring in-test, so a guard that goes blind again is itself detectable.

## The headline claim was broader than the evidence

`V5` reports that fitting an induction-only structure to a purely selective or
purely cytotoxic agent returns a Δβ interval excluding zero in every replicate.
True. But the README and the manuscript generalised it to *"a Δβ interval is not
evidence of plasticity"*, and the build's own `M_both` structure refutes that —
on fractions alone, with no counts:

| design | truth | induction-only | **gap free** |
|---|---|---|---|
| short | induction | 100% | 30% |
| short | **selection** | **100%** | **0%** |
| short | **cytotoxicity** | **100%** | **0%** |
| extended | induction | 100% | 91% |
| extended | **selection** | **100%** | **0%** |
| extended | **cytotoxicity** | **100%** | **0%** |

The false positive belongs to **assuming fitness parity**, not to fraction data.
What freeing the gap costs is **power** (30% at the short design), and that cost
— together with the selection/differential-death degeneracy, which no fraction
data can resolve — is the honest argument for absolute counts. The
false-positive rate is not.

`V5` now names the structure in its statement; **`V16` is new** and asserts the
bound. The recommendation is restated: a model comparison including
differential death, fitted to fractions and counts — *not an interval from a
structure that has already assumed fitness parity*. `beta_interval`'s grid is
now structure-dependent, because the `both` profile runs past the 0.12 that
suffices for `ind` and was right-censored in 23–31 of 40 draws on the old grid.

## Basins were single-sample estimates quoted to three significant figures

Under asynchronous update a basin is a **probability over update orders**, not a
set. `attractors` ran one trajectory per starting state, giving every basin a
binomial MC error near **±1.2 percentage points** — and the table reported
19.9% / 28.4% / 10.2% / 14.3%.

Now 32 trajectories per start, with the MC error reported beside each figure:

| | old (1 traj) | **new (32 traj)** |
|---|---|---|
| control stem basin | 19.9% | **19.6 ± 0.2%** |
| arsenic stem basin | 28.4% | **27.5 ± 0.2%** |
| control immortal basin | 10.2% | **9.7 ± 0.1%** |
| arsenic immortal basin | 14.3% | **13.6 ± 0.1%** |

The qualitative claims held: across 144 seed pairs of the old implementation
`V1` and `V15` never failed. The direction is many times its error. The third
digit never belonged there. Cutting each trajectory at the first absorbing fixed
point — exact, not a step budget — made 32 repeats **~40× faster** than the
single-sample version, so the precision is free.

## Three quieter errors

- **The purity-titration noise floor was understated by √2.** The statistic is a
  difference of differences over four measured means, so its standard error is
  σ√(4/n), not the σ√(2/n) of a two-arm contrast. Corrected 0.048 → **0.068**.
  Both verdicts stand: 20× fails (contrast 0.016) and 100× passes (0.073), but
  the 100× margin is thinner than the old figure suggested, and 50× fails under
  either threshold.
- **AICc counted a structural zero as data.** Counts enter as fold-change from
  the first timepoint, so the first count residual is identically zero. `n` is
  now `2·len(t) − 1`.
- **`d_fit` could be driven below `−r_S`**, i.e. to a negative proliferation
  rate. Bounds are now computed from the parameters rather than fixed at −1.0;
  it never bound at the nominal rates, but the ensemble perturbs them.

Two limits that were true but unstated are now in the README and Note S7: every
fit is given the baseline rates and control fixed point **exactly** (conservative
for the headline, but every interval here is narrower than one achievable in
practice), and the washout result assumes the exposure effect is **fully and
instantly reversible**, which is precisely the case a heritable plastic change
would violate.

---

# Pass 6 — two likelihood errors, a self-flattering audit, and a start-state assumption (2026-09-21)

A second external review. Every claim in it checked out. Two are errors in the
likelihood itself, which move numbers this build reports as results.

## The fraction likelihood overstated the noise by √3

`observe` returns a mean of `n_rep = 3` biological replicates, so its SD is
σ/√3 ≈ 0.0173. The fit divided residuals by the per-replicate σ = 0.0300. The
log-likelihood surface was flattened by a factor of 3 and every profile interval
was correspondingly too wide.

| | as shipped | **corrected** |
|---|---|---|
| CI width, cytotoxicity, short | 0.0120 | **0.0060** |
| CI width, cytotoxicity, extended | 0.0100 | **0.0040** |
| gap-free detection, induction, short | 30% | **82%** |
| gap-free detection, induction, extended | 91% | **100%** |
| induction-only false positive, all agents | 100% | **100%** |

The headline false positive is untouched, because there it is the
misspecification and not the noise that excludes zero. **But the argument
around it changes.** Pass 5 justified absolute counts partly on the power cost
of dropping the fitness-parity assumption. At the corrected noise that cost is
18 points at the short design and *nothing* at the extended one. The case for
counts now rests where it always should have: on the selection /
differential-death degeneracy, which no quantity of fraction data can resolve.

Both halves now go through `inference.se_mean`, so they cannot drift apart
again.

## The count likelihood ignored a correlation it had created itself

Log counts were compared as fold-change from the first timepoint, to stop the
fit buying agreement by rescaling the seeding density. But the first timepoint
is a *noisy observation*, not a known constant, so subtracting it injects −ε₀
into every differenced residual:

| | assumed by the fit | actual |
|---|---|---|
| Corr(dᵢ, dⱼ), i≠j | 0 | **0.50** |
| Var(dᵢ) | σ_n² | **2·se²** |
| scale | per-replicate σ_n | SE of a mean |

Three errors at once. The fix is to project the offset out onto an
**orthonormal (Helmert) contrast basis**: `H` has rows orthogonal to **1**, so
`H·v` is exactly invariant to an additive offset — the actual intent — and
because the rows are orthonormal, `H·noise` has covariance se²·I. Mean-centring
alone is *not* sufficient: the centred vector keeps covariance se²(I − J/n), so
its components stay correlated. Calibration, mean scaled RSS ÷ dof at the true
structure:

| | differenced | mean-centred | **contrasts** |
|---|---|---|---|
| truth = cytotoxicity | — | 0.93 | **0.99** |
| truth = selection | — | — | **1.00** |

Model choice moves: on fractions the comparison now prefers `both` in a small
minority of replicates rather than never; with counts, selection → `sel` 94%
(was 98%) and cytotoxicity → `cyto` 100% (unchanged).

## The uncertainty audit flattered itself, three ways

1. **Truncated intervals were compared as widths.** 240 of 960 cytotoxicity
   intervals were right-censored at the grid edge. A censored width is a lower
   bound, not a width. Draws containing any censored interval are now excluded
   from the V6 audit and counted: 13/60 dropped, V6 scored over **47**.
2. **The ensemble used a laxer detection bar than the headline.** 2σ/√3 =
   0.0346 against the nominal 1.96·σ·√(2/3) = 0.0480. Now imported from
   `design.py`. The separable fraction falls **48% → 45%** and the median
   detection moves **5.2 → 9.5 days** past the endpoint.
3. **V8's audit tested one third of V8.** It checked only that induction is not
   falsely chosen, never that selection is identified as growth or cytotoxicity
   as killing — the identification claim it was cited as supporting. Now asserts
   all three parts: **58/60 → 49/60**.

Denominators now travel with the audit (`audit_denominator`), because V6's is no
longer `n_ok`.

## Mechanism descriptions were wrong in a way that matters

The selection and cytotoxicity agents were described as involving "no state
conversion at all". They retain the **full baseline conversion machinery**
(β = 0.010, γ = 0.120) — cells convert in every arm, because conversion is a
baseline property of the tissue, not something an exposure switches on. They are
**Δβ = 0** agents, not β = 0 agents, and the question this build poses is
whether Δβ > 0 can be established, not whether β > 0.

Relatedly, V8's basis overstated what counts buy. Counts separate *these two
exposure channels* given a known baseline. They do not separate proliferation
from death in general: a total count constrains only the net rate r − d, so
equal increases in both are invisible to it.

## The sorting design was reported without its power

`separable` asks whether the TRUE contrast clears the critical value — which at
the boundary is 50% power. The 100× design was reported as working:

| separation | contrast | clears floor | **power** |
|---|---|---|---|
| 20× | 0.016 | no | **0.07** |
| 50× | 0.039 | no | 0.20 |
| **100×** | 0.073 | yes | **0.56** |
| **141×** | 0.107 | yes | **0.80** |

56% is a coin flip, not a design. `purity_fold_for_power` now reports the
separation that reaches 80%: ~141×, with the dirty arm at **14.1% stem-like**.
V11 is restated on power.

## The KCC9 result is conditional on the start state, and the exemplar breaks it

The build presented immortal unreachability as a property of the *wiring*. It is
a property of the wiring **and the chosen start state**:

| start | unsupported perturbation | stem reached | immortal reached |
|---|---|---|---|
| TERT off | p53 loss | 100% | **0%** |
| **TERT on** | p53 loss | 100% | **100%** |

With TERT already on, `IMMORTAL ← stem AND tert` collapses to `IMMORTAL ← stem`.
**MCF-10A — the exemplar in Yang 2026 — is a spontaneously immortalised line**,
so the telomerase-negative start is not a validated representation of it.

What survives either start is the claim that matters: the **supported**
perturbation never reaches immortalisation, because arsenic never reaches the
stem-like state at all. What does not survive is "the model cannot score
stemness as KCC9 even by accident". **V17** is new and asserts both halves;
`mean_first_passage` takes `tert_start` and the run reports both tables.

## Two labelling and drift issues

- The supplementary S2 table labelled the absolute count **log₁₀**. `simulate`
  uses `np.log`, and every count gap in the text is in natural-log units.
  Corrected to `ln(N+S)`.
- **The combined manuscript has drifted.** `hKCC_manuscript_v4.docx` still
  reports the Boolean basins as 19.9% → 28.4%; current output is 19.6 ± 0.2% →
  27.5 ± 0.2%. That file is hand-maintained rather than generated from
  `emd3_manuscript_tables.json`, so it is outside this build's reproducibility
  chain and has not been edited here. **It needs a manual pass** — see the note
  at the end of this file.

---

# Pass 7 — export, metadata and framing corrections (2026-09-22)

A third external review, against the frozen deposit. Its full text is in
`deposit_review_2026-09-22/`. The simulation itself needed no correction this
time; the errors were in what the simulation's output was turned into.

## The V6 denominator was wrong in both Word generators

The summary records `audit.V6 = 0.872` over `audit_denominator.V6 = 47`, with 13
draws excluded for boundary censoring. Both generators multiplied every audit
proportion by `n_ok = 60`, printing **52/60** for a result that is **41/47**.
The censoring fix of Pass 6 had gone into the JSON but not into the text that
reads it. Both now resolve the per-check denominator and state the exclusions.

## The figure caption described the previous figure

Pass 6 replaced panel e with a power curve and added a TERT-on overlay to panel
f, but the caption still described panel e as days of follow-up against
measurement precision, still claimed counts restore induction power (the
extended schedule reaches 100% without them), and still said immortal
reachability is zero in every condition. Panel c was titled "One dataset, two
structures" while plotting medians of interval endpoints across 120 datasets.
Caption and panel titles are rewritten; panel d's axis now reads "synthetic
datasets (%)" rather than "% of replicates".

## A cited DOI pointed to an unrelated paper

`10.1038/s41540-024-00429-2` is *Peptide hemolytic activity analysis using
visual data mining of similarity-based complex networks*, npj Syst Biol Appl
10:115. The intended source is Hernández-Magaña A, Bensussen A, Martínez-García
J C, Álvarez-Buylla E R, *A Boolean model explains phenotypic plasticity changes
underlying hepatic cancer stem cells emergence*, npj Syst Biol Appl **10**:99
(2024), **10.1038/s41540-024-00422-9**. Both DOIs were checked against CrossRef
before the change. The Yang title was a paraphrase and is now the registered
title; full author lists were added, and the Transcompp description now reads
"cell-culture state-transition data without arsenic-specific calibration"
rather than "untreated cell lines", which its hydrocortisone and cholera-toxin
conditions contradict.

## Mechanism descriptions were inconsistent with the model

The exposure-channel table still described selection and differential death as
involving no cell conversion. Both retain the full baseline rates; they differ
in **Δβ, not in β**. Table headers and panel labels now distinguish an interval
excluding Δβ = 0 from selection of an induction-capable structure, because AICc
model selection is not a significance test on Δβ.

## The sorting conclusion generalised a fixed-replication result

Power was reported at three replicates per arm only, and described as "a coin
flip, not a design". Power responds to two levers:

| separation | contrast | n = 3 | n = 4 | n = 6 | n = 8 |
|---|---|---|---|---|---|
| 20-fold | 0.0157 | 0.07 | 0.08 | 0.10 | 0.11 |
| 50-fold | 0.0387 | 0.20 | 0.25 | 0.35 | 0.45 |
| **100-fold** | 0.0727 | **0.56** | 0.68 | **0.84** | 0.93 |
| 200-fold | 0.1270 | 0.96 | 0.99 | 1.00 | 1.00 |

Holding the 100-fold separation and moving from three to six replicates per arm
reaches 84%. `design.power_vs_replication` makes this a deposited output rather
than a claim in prose, and the reported power is now the conventional two-sided
5% test; the directional upper tail alone is retained as
`power_upper_tail` and differs materially only where power is low (7.4% against
6.6% at 20-fold). V11 is restated on the trade-off.

## Smaller corrections

- The ensemble's V8 counted `ind` + `both` against a 0.2 threshold while the
  nominal V8 counted `ind` alone against 0.05, and the two were described as the
  same criterion. The ensemble now counts `ind` alone; with 8 inner draws the
  threshold is stated as "none of the eight".
- The 0.90 cap on the ensemble's fold target, and the 240-day search horizon,
  are now in the summary and the text. The claim that follow-up fails "at any
  length" is replaced by the tested horizon.
- The ensemble's scope is stated: it audits selected interval, identification,
  washout and follow-up criteria, not the combined-structure profiles, the
  sorting calculation or the Boolean layer.
- The state table labelled the absolute count log₁₀; it is a natural log.
- A `provenance` block records dataset counts, replicate count, noise settings,
  likelihood and package versions, and `manuscript_tables` now refuses a summary
  that lacks it.
- TERT being held fixed is described as a property of the specified network
  rather than a biological exclusion; β-catenin-associated hTERT regulation has
  been reported in MCF-10A.

## Framing

Development history has been moved out of the manuscripts into this file, which
is what it is for. The main text and supplement now lead with what each
measurement contributes and state limits once. `hKCC_manuscript_v5.docx`
rebuilds the two EMD3 paragraphs of the combined manuscript against the current
outputs; `v4` is left unchanged as the prior draft.

---

# Status after these passes

| | |
|---|---|
| checks | **17** (V1–V17; V16 added in Pass 5, V17 in Pass 6) |
| bugs found and fixed | **10** (under-dosed control; powerless KCC9 guard; single-sample basins; √2 noise floor; AICc sample count; √3 fraction-likelihood scaling; correlated count residuals; censored widths in the V6 audit; ensemble/nominal threshold mismatch; V8 audited at one third of its scope) |
| checks restated | **5** (V6 onto width and centre; V5 scoped to the induction-only structure; V11 onto power; V8's basis narrowed; V3 scoped to a TERT-negative start) |
| guards verified to be capable of failing | **1** (immortal reachability, by rewiring TERT in-test) |
| export/metadata defects fixed in Pass 7 | **6** (V6 denominator in both generators; figure caption; wrong DOI; channel descriptions; one-sided power; ensemble V8 criterion) |
| combined manuscript | `v5` rebuilds the EMD3 paragraphs; `v4` retained as the prior draft |
| plan expectations tested and refuted | **3** (washout, MFPT-vs-dose, reversal) |
| misreading traps guarded | **1** (immortal basin vs reachability) |
| model comparison criterion | AIC → **AICc** (5–7 points against 1–2 parameters) |
| claims reported as marginal under uncertainty | **2** (V5 at 73%, V6 at 70%) |
| design outputs uncertainty can void | **1** (watch-longer: works in 48% of draws) |

**Still open, in priority order:**

1. Calibrate the plasticity machinery on Transcompp's deposited sorted-fraction
   series. This is the single most valuable improvement available and would
   convert the baseline rates from chosen to estimated.
2. Locate or generate a longitudinal composition series with absolute counts for
   any exposure–stemness pair. If none exists, that is the argument for the
   designed experiment this build outputs.
3. The Boolean layer remains illustrative: node count, logic and update scheme
   all affect attractor structure, and no sensitivity analysis over alternative
   rule sets has been run.

---

# Combined manuscript

`hKCC_manuscript_v5.docx` (repository root) rebuilds the two EMD3 paragraphs
against the current outputs: 41/47 with its 13 exclusions, 27/60 within the
tested 240-day horizon, 19.6% → 27.5% basins, the replication trade-off, and the
initial-state qualifier on the immortalisation readout. Only those two
paragraphs differ from `v4`; media, tables and every other paragraph are
byte-identical.

`v4` is retained unchanged as the prior draft. If `v4` rather than `v5` is the
submission version, the two paragraphs need the same replacement there.
