# Published-data sources for the EMD3 build

Retrieved 2026-08-25.

## The short version

**Nothing in this build is fitted to data, and no dataset was acquired.** That
is not an oversight deferred for later; it is the finding that determines what
the build is for. The published exposure studies report **endpoint marker
fractions**. The question EMD3 poses — is a change in the stem-like fraction
induction or selection? — requires **longitudinal composition with absolute
counts**. Those are different measurements, and the second does not exist
publicly for any arsenic–stemness exposure series we could locate.

So the deliverable is identifiability and experimental design, not a fit. Every
rate here is a plausible value for a mammary epithelial line, and the results
that matter are relationships between structures, not magnitudes.

## Consulted, and what each can and cannot support

| source | what it gives | what it cannot give |
|---|---|---|
| Yang Y *et al.*, *Food Chem Toxicol* **211**:116003 (2026) — ref 42 | The exemplar: arsenic raises stemness in MCF-10A through E2F2 → FZD10 → Wnt/β-catenin. Fixes which Boolean nodes may be clamped. | Longitudinal composition. Endpoint marker fractions only. |
| Hernández-Magaña A *et al.*, *npj Syst Biol Appl* **10**:99 (2024) — ref 47 | The published plasticity-network topology the Boolean layer follows. | A validated network for mammary epithelium; it is a hepatocyte model, used as a structural template. |
| Jagannathan N *et al.* — Transcompp (ref 44), <https://github.com/nsuhasj/Transcompp> | The estimation framework for state-transition rates from sorted-fraction time series; the method this build's inference layer is a stripped-down analogue of. | Exposure-specific rates. Transcompp's deposited series are cell-culture state-transition data without arsenic-specific calibration. |
| Yang *et al.*, *ACS Omega* (ref 43) | Cadmium in MCF-7 / HepG2, as a possible second agent. | Same limitation: endpoint fractions. |
| Schroeder *et al.*, *Front Oncol* **14**:1411295 (2024) — ref 41 | Primary human breast cells, non-tumour setting, as an external comparator. | Not a time series with counts. |

## Not acquired, and why

- **Transcompp's deposited datasets.** They would let the *plasticity machinery*
  be calibrated on real sorted-fraction series, which is the single most
  valuable improvement available to this build. Not attempted here; the baseline
  rates remain chosen. This is the analogue of digitising figure panels in EMD2
  and is the recommended next step.
- **Any arsenic longitudinal composition series with absolute counts.** We could
  not locate one. If it does not exist, that is itself the argument for the
  designed experiment this build outputs.
- **Cadmium (ref 43) and the Schroeder primary-cell comparator (ref 41)** as
  held-out arms. Both were listed in the EMD2-4 plan §3.7. Neither is
  implemented, because with endpoint fractions only they would add a second and
  third *unconstrained* parameter set rather than a test. Scoping them out is
  deliberate and is stated here rather than left as a silent gap.

## Three plan items this build tested and had to report as refuted

Recorded in `FINDINGS.md`, and noted here because they concern what the
*sources* can support:

1. **The washout/reversal arm does not discriminate** on marker fractions (V12).
2. **MFPT into the stem-like attractor does not fall with dose** (V14) — the
   supported perturbation enlarges the basin without opening a route into it.
3. **Pathway knockdown does not "restore" attractor structure** (V13); it
   abolishes the stem basin entirely, below the level the control network
   carries.

## One correction to a previous release

The three matched agents were hard-coded, and the cytotoxic magnitude had gone
stale — see `FINDINGS.md`. They are now computed from the model at import time.
