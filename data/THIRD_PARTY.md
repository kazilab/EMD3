# EMD3 model inputs and sources

All cell fractions and counts are synthetic. The population model uses assigned daily rates of division, death and state transition. The fitted conversion effect is the additive change in the transition rate from non-stem to stem-like cells (day⁻¹). A conversion-only fit assumes equal net growth in the two states, so differential death can produce an apparent conversion effect. The joint fit also estimates the difference in net growth.

The Boolean model uses a published hepatocyte network as a structural template. An arsenic study in MCF-10A mammary epithelial cells informs the choice of pathway perturbations. Its endpoint measurements do not determine the population rates. MCF-10A is spontaneously immortalised, so the model considers both TERT-on and TERT-off starting states. [Hernández-Magaña et al. (2024)](https://doi.org/10.1038/s41540-024-00422-9) · [Yang et al. (2026)](https://doi.org/10.1016/j.fct.2026.116003)

Transcompp provides a methodological reference for estimating transition rates from cell-state fractions. The source manifest lists each reference and its role. [Jagannathan et al. (2020)](https://doi.org/10.1093/bioinformatics/btaa021) · [Source manifest](SOURCES.tsv)

EMD3 distributes its simulation code under the [MIT license](../LICENSE).
