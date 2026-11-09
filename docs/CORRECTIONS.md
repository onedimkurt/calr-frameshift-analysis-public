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

19 of the 288 significance cells in Supplementary Table S6 change, in three
categories:

* **11 verdicts change** for comparisons that were testable in both versions.
* **5 comparisons are newly reported** (`uversky_llps` x 3, `KRKEE_n` x 2). The
  superseded file carried no rows for them.
* **3 comparisons are withdrawn** (`PKA_sites`). That feature is constant at 2
  across all 76 variants, so a rank test on it is undefined; the superseded file
  reported p = 0.2933 / 0.1350 / 1.0000, which the current matrix cannot produce.
  `CK2_sites` is likewise invariant and was already shown as not tested.

The 11 changed verdicts in full:

| Feature | Comparison | Was | Is |
|---|---|---|---|
| grp_Hydrophobic | Type 1-A vs Type 1-B | ns | *** |
| n_rtrr | Type 1-A vs Type 1-B | ns | *** |
| f_D | Type 1-A vs Type 1-B | ** | *** |
| f_T | Type 1-A vs Type 1-B | * | ** |
| grp_Negative | Type 1-A vs Type 1-B | * | ** |
| tail_DE_fraction | Type 1-A vs Type 1-B | *** | ** |
| polycation_len | Type 1-A vs Type 1-B | *** | ** |
| PKC_sites | Type 1-A vs Type 1-B | ** | * |
| f_A | Type 2 vs Type 1-A | ns | * |
| entropy_novel | Type 2 vs Type 1-B | * | ns |
| PKC_sites | Type 2 vs Type 1-B | *** | ** |

`grp_Negative` and `tail_DE_fraction` now agree, as they must: the two columns hold
identical data.

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

### Manuscript text reconciled against the tables

A line-by-line audit of every count, group mean, standard deviation, direction and
figure or table citation in the manuscript against the certified outputs turned up
four further discrepancies, none of which involved the regenerated statistics:

* The post-hoc count was given as "50 of 83 tested features". The pipeline's own
  `feature_importance_posthoc.tsv` and Supplementary Table S8 both hold 83 rows with
  51 significant (34 ***, 14 **, 3 *). The count is 51.
* The Type 1-A interface pTM standard deviation was given as 0.055 in the Figure 5D
  legend. `AF2_DEFINITIVE.tsv` and Supplementary Table S7 give 0.0681; 0.055 is the
  Type 1-B value repeated.
* Supplementary Figure S10 was cited for "mean inter-chain Calpha-Calpha distance
  profiles". No distance profile is computed anywhere in the pipeline;
  `figure_s_contact_comprehensive.py` plots contact region pairs, contact frequency,
  contact maps and hotspot residues. The sentence now describes contact frequencies.
  The reciprocal pattern it reports is unchanged and reproduces exactly: 723 / 239 /
  398 total contacts and 98.9% / 66.9% / 11.6% post-anchor shares for Type 1-A /
  Type 1-B / Type 2.
* Three feature directions and two approach-zone position lists had the groups
  reversed relative to the feature matrix: approach zone length is enriched in
  Type 1-A, while mean hydropathy and the charge-hydropathy disorder-boundary metric
  are enriched in Type 1-B.

Checks that passed unchanged: all 76 variant counts and group sizes, the Das-Pappu
region assignment (74 of 76 in region 3), 3 of 25 Type 1-A variants with f- < 0.10,
21/25 compact Type 1-A and 14/19 extended Type 1-B, 35/43 = 81.4% sequence-to-geometry
mapping, 35 of 76 variants with a model-stable compact/extended call, 32/32 KKRK
co-occurrence with Type 2, and 51 of 53 contacts at approach-zone position -1 carrying
a cysteine.
