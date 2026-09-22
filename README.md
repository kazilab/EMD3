# EMD3: minimal reproducibility deposit

An identifiability and experimental-design analysis of exposure-induced phenotypic plasticity, with arsenic in non-tumorigenic mammary epithelium as the exemplar ([Yang et al., *Food Chem Toxicol* 211:116003, 2026](https://doi.org/10.1016/j.fct.2026.116003); E2F2/FZD10 → Wnt/β-catenin). A two-type population model asks when a rise in the stem-like marker fraction can be attributed to **induction** rather than to **selection** or to **differential death**, and what experiment would decide it. A separate 10-node Boolean attractor layer asks whether perturbing only the nodes the exposure is reported to touch changes attractor structure.

Rates and observation models are specified for a methodological study rather than estimated from a dataset: the available exposure studies report endpoint marker fractions, and the questions posed here require longitudinal composition. Several of the predefined checks are negative controls and report the conditions under which an inference fails. Population and Boolean results are conditional model analyses, not independent experimental evidence of carcinogenic activity.

## Reproduce offline

Use Python 3.11 or later. From this archive's root:

```bash
python -m pip install -r code/emd3_simulation/requirements.txt
cd code
python -m emd3_simulation.run --outdir ../reproduced
python -m emd3_simulation.manuscript_tables --outdir ../reproduced
python -m pytest emd3_simulation/tests -q -p no:cacheprovider
```

No downloads are needed at any point. The full run takes roughly 10–15 minutes: it profiles Δβ under two structures across 120 seeded replicates at two designs, compares four model structures with and without absolute counts, enumerates the Boolean attractor landscape over all 2^10 starting states with 32 asynchronous update orders each, and re-scores the movable claims across 60 parameter draws. `--no-figure` skips figure export; `--n-draws 40 --n-ensemble 0` gives a fast diagnostic run, but partial results cannot generate publication tables — `manuscript_tables.py` refuses a summary with missing blocks rather than silently writing one.

A guided walkthrough that reproduces each headline result and regenerates the figure is in `EMD3_simulation_walkthrough.ipynb`.

Compare the two generated JSON files with `outputs/`. On the reviewed configuration both were **byte-identical** to the deposited copies when regenerated from an isolated extraction of this archive; floating-point results may vary slightly on other platforms. Reviewed software: Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1, pytest 8.4.2. `requirements-tested.txt` pins those; the core requirements specify minimum versions. The deposited run passed **17/17** internal checks and **36** regression tests, and the isolated extraction reproduced both.

## What each measurement contributes

Three exposure scenarios — increased dedifferentiation (Δβ = 0.0324 per day), increased stem-like proliferation and increased non-stem death (each 0.2042 per day) — reach the same stem-like fraction of 0.20 on day 14, so a single endpoint measurement is consistent with all three. All three retain the baseline bidirectional conversion rates; they differ in **Δβ, not in β**, and cells convert in every arm.

**Longitudinal fractions** separate induction from selection once relative fitness is allowed to vary. An induction-only structure, which enforces fitness parity, returns a Δβ interval excluding zero in every synthetic dataset including those generated with Δβ = 0; the combined structure covers zero for both Δβ = 0 scenarios while still detecting genuine induction in 120/120 datasets at the extended schedule. That contrast is a property of the parity constraint and is specific to it.

**Absolute counts** add information about the two specified proliferation and death channels, which are algebraically one model for the fraction: they agree to under 10⁻¹⁰ over 60 days while their log total counts differ by 9.0 natural-log units. A total count constrains the net rate, so identifying proliferation and death rates more generally requires separate division or death readouts.

**Initial-state analysis** establishes the conditions under which a Boolean reachability statement holds — see the limits below.

## Contents and provenance

| Path | Purpose |
|---|---|
| `code/emd3_simulation/model.py` | Two-type ODE, exposure channels, endpoint matching of the three agents |
| `code/emd3_simulation/inference.py` | Four competing structures, profile-likelihood intervals, AICc comparison with and without counts |
| `code/emd3_simulation/design.py` | Detection window and sort-and-re-emerge power |
| `code/emd3_simulation/boolean.py` | 10-node asynchronous attractor layer, basins with Monte Carlo error, reachability from two start states |
| `code/emd3_simulation/conditions.py` | The predefined V1–V17 validation battery, each with its evidentiary basis |
| `code/emd3_simulation/ensemble.py` | 60-draw sensitivity over the six baseline rates, agents re-matched inside each draw |
| `code/emd3_simulation/fit/PROVENANCE.md` | What each consulted source can and cannot support, and why none was acquired |
| `code/emd3_simulation/fit/FINDINGS.md` | Six passes of confrontation, including ten bugs found and fixed |
| `outputs/` | Reference simulation summary and manuscript-table JSONs |
| `data/` | Source manifest and reuse attribution; **no dataset is bundled or downloaded** |
| `EMD3_simulation_walkthrough.ipynb` | Guided reproduction of each result and the figure |
| `MANIFEST.sha256` | Checksums for every deposit file except the manifest itself |

The deposit excludes publication images, Word files and their generators, and caches. The figure regenerates from code. The Word toolchain remains in the working manuscript project and is unnecessary to reproduce the numerical analysis.

## Evidentiary limits

- **Nothing is fitted.** Baseline rates are plausible values for a mammary epithelial line, not measurements. The *relationships* — the degeneracy, the structure-dependence of the false positive, the design comparison — are properties of the model structure; the specific day counts are not.
- **Every fit is given the baseline rates and the control fixed point exactly**, estimating only the exposure effect. This is conservative for the headline, but every interval here is narrower than one achievable in practice.
- **The sorting contrast trades purity separation against replication.** At three replicates per arm a 20-fold separation reaches 0.07 power and a 100-fold separation 0.56; reaching 0.80 at that replicate count needs ~141-fold, placing the lower-purity arm at 14.1% stem-like, while holding 100-fold and raising replication to six per arm reaches 0.84. These are prospective benchmarks under a known-noise Gaussian contrast of four measured means at a two-sided 5% threshold; applying them to a single agent of unknown mechanism requires the observational units and model-discrimination rule to be specified. Extended follow-up crossed the separation threshold within the tested 240-day horizon in 27 of 60 parameter draws; the search was not extended beyond that horizon. Unbounded exponential growth is assumed throughout, so a long follow-up also requires density and passage assumptions that this model does not represent.
- **Counts separate the two specified exposure channels given a known baseline.** They do not separate proliferation from death in general: a total count constrains only the net rate `r − d`.
- **The Boolean layer is illustrative**, and every reachability figure is conditional on an imposed wiring *and* an imposed start state. TERT is held fixed in this network; that is a modelling choice, not a measured exclusion, and β-catenin-associated hTERT regulation has been reported in MCF-10A. Immortalisation is unreachable from a telomerase-negative start; from a telomerase-positive start — which MCF-10A, the exemplar line, is — the exploratory p53-loss perturbation reaches it in every run, while the arsenic-supported perturbation reaches it under neither. A TERT-on but STEM-off start also has IMMORTAL = 0 here, although an immortalised non-stem-like cell is biologically possible. No rule-set sensitivity analysis was run.
- **Two compartments only**, with no lineage information. Barcoding or live tracking answers a question this model cannot pose.
- **This is not a Bayesian analysis.** Δβ intervals are profile-likelihood intervals and model comparison uses AICc. Where the text says an interval "excludes zero" it means the profile interval, not a credible interval.
- **A stem-like state and an immortalisation readout are distinct.** In this network `TERT` has no activator on the exposure path, so the supported perturbation does not move the immortalisation readout. Mapping EMD3 to a KCC home remains conditional on independent epigenetic and immortalisation evidence.

Software: MIT. New derived results: CC BY 4.0. No third-party data are redistributed. See `LICENSE`, `LICENSE-DATA` and `data/THIRD_PARTY.md`.
