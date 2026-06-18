# Pipeline documentation

End-to-end description of the analysis: what each script reads, what it writes, and how
its output was verified against the published results.

## Overview DAG

```
data/sequences/Table_S1_Sequences_76variants.tsv   (76 variant sequences)
data/sequences/sequence_metadata.tsv               (novel tails, frameshift positions)
data/sequences/CALR_WT_P27797.fasta                (WT reference)
            │
            ▼
[00] recompute_all_features.py
            │  → data/derived/RECOMPUTED_FEATURES_76_VARIANTS.tsv   (76 × 101 features)
            │
            ├──────────────────────────────┬───────────────────────────────┐
            ▼                               ▼                               ▼
[01] true_unsupervised_clustering.py   [01] type2_unsupervised_clustering   [02] af2_definitive.py
     → cluster_assignments.tsv              → cluster_assignments_et.tsv       (reads ColabFold PDB/JSON)
       (44 Type 1-like → 1-A / 1-B)           (Type 2 control)                 → AF2_DEFINITIVE.tsv
            │                                                                  → all_comparisons.tsv
            ▼                                                                       │
[00] build_definition_tables.py  → Table_S4 / Table_S5                              │
            │                                                                       │
            └───────────────┬───────────────────────────────────────────────────┘
                            ▼
[04] review-stage validation analyses (bootstrap, PCA, effect sizes, IDP, ALBATROSS, ...)
                            │
                            ▼
[figures] figure_01.py … figure_05.py
```

## Inputs

| File | Description | Source |
|------|-------------|--------|
| `data/sequences/Table_S1_Sequences_76variants.tsv` | 76 variant identifiers + full sequences | public databases (accessions in file) |
| `data/sequences/sequence_metadata.tsv` | novel C-terminal tails, frameshift positions | derived from sequences |
| `data/sequences/CALR_WT_P27797.fasta` | wild-type CALR (417 aa) | UniProt P27797 |
| `data/FEATURE_DICTIONARY.tsv` | feature name → definition mapping | this study |

The frameshift position used throughout is taken from `sequence_metadata.tsv`, which is
consistent with the protein-level nomenclature (e.g. p.K385fs → position 385).

## Per-script reference

### 00_feature_pipeline/recompute_all_features.py
- **Reads:** Table S1 (sequences), sequence_metadata (novel tails), WT fasta
- **Writes:** `data/derived/RECOMPUTED_FEATURES_76_VARIANTS.tsv` (101 features) + inventory
- **Computes:** composition, NCPR, FCR, SCD/κ (localCIDER), Kyte–Doolittle hydropathy,
  charge–hydropathy disorder metric, Grantham distances, approach-zone position charges,
  motif counts, WT C-domain features
- **Verification:** reproduces the published feature matrix exactly (0 differing cells, 76×108).

### 01_clustering/true_unsupervised_clustering.py
- **Reads:** feature matrix; Sγ–Sγ distances from AF2_DEFINITIVE (post-hoc validation only)
- **Writes:** `cluster_assignments.tsv`, feature-importance table, clustering figure
- **Method:** k-means (k=2, n_init=50, random_state=42) + Ward; subgroups labelled by mean
  Sγ–Sγ (lower = Type 1-A). Sγ–Sγ is NOT a clustering input.
- **Verification:** silhouette@k=2 = 0.213; 100% k-means/Ward agreement; 25/19 split;
  44/44 per-variant labels match published.

### 01_clustering/type2_unsupervised_clustering.py
- **Reads:** feature matrix (Type 2 variants)
- **Writes:** `cluster_assignments_et.tsv`, silhouette/gap tables, figure
- **Purpose:** control analysis showing Type 2 has no robust subgroups
- **Verification:** k=2 silhouette = 0.263; 100% partition agreement with published.

### 02_structural/af2_definitive.py
- **Reads:** ColabFold homodimer + monomer predictions (see Structural inputs below); Table S1;
  metadata; cluster assignments
- **Writes:** `AF2_DEFINITIVE.tsv` (76 × structural metrics), `all_comparisons.tsv`
- **Extracts:** inter-chain Sγ–Sγ and Cβ–Cβ distances, χ1 dihedrals, novel-tail contacts,
  pLDDT, pTM, ipTM, PAE (inter/intra)
- **Verification:** homodimer metrics reproduce published values exactly for all 76; overall
  ipTM = 0.187. Monomer pLDDT exact except var9 (recomputed value differs slightly from an
  earlier tabulated cell; the recomputed value is authoritative).

### 00_feature_pipeline/build_definition_tables.py
- **Reads:** feature matrix, cluster assignments, FEATURE_DICTIONARY.tsv
- **Writes:** Table S4 (feature definitions, 88 rows), Table S5 (structural metric definitions, 16 rows)

### 04_review_validation/
Review-stage analyses, each reading the certified derived files and writing to `data/derived/`:

| Script | Produces | Key result |
|--------|----------|-----------|
| C02_bootstrap_87.py | per-cluster Jaccard | 0.523 / 0.516 (soft boundary) |
| C02_C03_bootstrap_pcclust.py | bootstrap + PC clustering | ARI 1.000 on PCs |
| C03_C06_pca.py | PCA loadings | PC1 29.6%, PC2 15.6% |
| C04_C20_effectsizes_descriptives.py | effect sizes, group descriptives | — |
| C05_silhouette_gap.py | silhouette/gap curves | k=2 = 0.213 |
| C07_circularity*.py | circularity control | — |
| C08_C09_idp.py / C08_daspappu_region.py | f⁺/f⁻, Das–Pappu regions | all region 3 |
| C10_albatross.py | sequence-ensemble Rg/Re/ν | Rg 26.8/24.7/25.9 Å |
| C14_pae_figure.py | representative PAE matrices | — |
| C16_count_reconcile.py | structural count reconciliation | — |
| C18_retained_acidic.py | retained acidic C-domain charge | net −26/−26/−30 |
| C19_type2_register.py | Type 2 dimer register | — |

### figures/scripts/
`figure_01.py`–`figure_05.py` generate the manuscript figures from the certified derived
files. `figure_01A_schematic.py` draws the schematic panel.

## Structural inputs (large; archived separately)

The ColabFold predictions are not included in this repository due to size. They are archived
on Zenodo (see `data/README.md`). The structural scripts read them via environment variables:

| Variable | Default | Used by |
|----------|---------|---------|
| `CALR_MONO_DIR` | `data/structures/monomer_pdbs` | af2_definitive.py (monomer pLDDT) |
| `CALR_HOMODIMER_DIR` | `data/structures/homodimer_pdbs` | C19_type2_register.py |
| `CALR_PAE_GLOB` | `~/Downloads/C14_dimer_pae*.json` | C14_pae_figure.py |

The homodimer ZIP archives (`results_homodimers`, `crosslinks_homodimers`,
`results_monomers`) are referenced near the top of `af2_definitive.py`; point them at the
unpacked Zenodo download to re-run structural extraction.

## Determinism

All sequence and clustering steps are deterministic (`random_state=42`, `n_init=50`).
Bootstrap analyses use a fixed seed. Structural extraction is deterministic given the input PDBs.
