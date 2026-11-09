# Corrections

## 2026-11 — three-group pairwise statistics regenerated

### What was wrong

`data/derived/three_group_validation/statistics/sequence_three_group_pairwise.tsv`
was not produced by any step of this pipeline. Its p-values were computed on an
earlier feature set: for 21 of 84 features the group means stored inside that file
disagree with `RECOMPUTED_FEATURES_76_VARIANTS.tsv`, which is the certified feature
matrix every other output is built from.

The file was nevertheless read by the figure scripts (`SEQ_STATS`) and its p-values
were carried into Supplementary Table S6, while that table's group means came from
the current matrix. The result was a table that mixed two datasets.

The clearest symptom: `grp_Negative` and `tail_DE_fraction` are computed by the same
formula (`count(D)+count(E)` over tail length) and are bit-for-bit identical across
all 76 variants, yet the table reported p = 0.0231 for one and p = 3.14e-06 for the
other.

### What was done

`04_review_validation/C21_three_group_pairwise.py` regenerates the file from the
current feature matrix. The method was reverse-engineered from the superseded file
and reproduces it exactly (256/256 rows) when run on the old inputs:

* two-sided Mann-Whitney U per feature per pair
* Cohen's d with pooled SD, same formula as `01_clustering`
* Benjamini-Hochberg applied **separately within each pairwise comparison**

`data/derived/` is git-ignored, so the superseded file was never published here; it
existed only in local working copies. The gap this repository had was that no step
generated the file at all, even though the figure scripts read it. `C21` closes that
gap, so the statistics behind the manuscript figures and Supplementary Table S6 are
now reproducible from the published inputs.

### Effect

16 significance verdicts change. The largest are:

| Feature | Comparison | Was | Is |
|---|---|---|---|
| grp_Hydrophobic | Type 1-A vs Type 1-B | ns | *** |
| n_rtrr | Type 1-A vs Type 1-B | ns | *** |
| polycation_len | Type 1-A vs Type 1-B | *** | ** |
| grp_Negative / tail_DE_fraction | Type 1-A vs Type 1-B | * / *** | ** / ** |
| entropy_novel | Type 2 vs Type 1-B | * | ns |

Group means, cluster assignments, structural metrics (`AF2_DEFINITIVE.tsv`,
`all_comparisons.tsv`) and the feature-importance table
(`feature_importance_posthoc.tsv`) were checked against the feature matrix and are
unaffected. The regenerated values agree with `feature_importance_posthoc.tsv` on
83/83 shared features.

### Also corrected

* `n_rtrr` was displayed as "RTRR motif count" in `figure_03.py` and `figure_05.py`.
  The feature counts the RXRR pattern (`R[A-Z]RR`); the label now reads "RXRR".
* `docs/PIPELINE.md` stated that all variants fall in Das-Pappu region 3. Two
  Type 1-A variants (var9, var159) are in region 5.
