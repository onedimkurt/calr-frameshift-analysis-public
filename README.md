# CALR exon 9 frameshift variant analysis

Computational pipeline for the sequence-and-structure analysis of 76 *CALR* exon 9
frameshift variants in myeloproliferative neoplasms: sequence-feature computation,
unsupervised clustering of the Type 1–like variants, ColabFold homodimer structural
metric extraction, and the review-stage validation analyses.

This repository contains the **code and the small public inputs** required to reproduce
the analysis. The large structural prediction files (ColabFold PDB/PAE/JSON outputs) are
archived separately (see `data/README.md`).

## Installation

```bash
conda env create -f environment.yml
conda activate calr_env
```

Core dependencies (pinned in `environment.yml` / `requirements.txt`): Python 3.10.14,
numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1, biopython 1.87,
localCIDER 0.1.21, matplotlib. The sequence-ensemble step (`04_review_validation/C10_albatross.py`)
additionally requires `sparrow` (ALBATROSS); see that script for the install command.

## Repository layout

```
00_feature_pipeline/    raw sequences -> 101-feature matrix; feature/metric definition tables
01_clustering/          unsupervised clustering of Type 1-like variants; Type 2 control
02_structural/          ColabFold homodimer/monomer structural-metric extraction
04_review_validation/   bootstrap stability, PCA, effect sizes, IDP biophysics,
                        Das-Pappu regions, ALBATROSS, structural reconciliation
figures/scripts/        manuscript figure generation
data/sequences/         variant sequences, metadata, WT CALR reference (public inputs)
data/derived/           regenerated outputs (created by running the pipeline)
docs/PIPELINE.md        full pipeline DAG, per-script inputs/outputs, data provenance
```

## Reproducing the analysis

Run in order (each step writes to `data/derived/`):

```bash
# 1. Sequence feature matrix (from sequences only)
python 00_feature_pipeline/recompute_all_features.py

# 2. Clustering of the 44 Type 1-like variants + Type 2 control
python 01_clustering/true_unsupervised_clustering.py
python 01_clustering/type2_unsupervised_clustering.py

# 3. Structural metrics (requires the ColabFold outputs; see data/README.md)
python 02_structural/af2_definitive.py

# 4. Definition tables
python 00_feature_pipeline/build_definition_tables.py

# 5. Review-stage validation analyses
python 04_review_validation/C02_bootstrap_87.py    # cluster stability
python 04_review_validation/C08_daspappu_region.py # charge-regime classification
python 04_review_validation/C10_albatross.py       # sequence-ensemble dimensions
# ... (see docs/PIPELINE.md for the full list)

# 6. Figures
python figures/scripts/figure_01.py   # ... figure_05.py
```

Steps 1, 2, 4, 5 (sequence/clustering) run from the included public inputs.
Step 3 (structural) requires the archived ColabFold predictions — set the paths described
in `data/README.md`.

## Data provenance

Variant sequences were obtained from public sources (accession identifiers in
`data/sequences/Table_S1_Sequences_76variants.tsv`); the wild-type CALR reference is
UniProt P27797. See `docs/PIPELINE.md` for full details.

## Notes on reproducibility

- All sequence-derived and clustering results regenerate deterministically
  (fixed `random_state=42`).
- Structural-metric extraction reproduces the published homodimer metrics exactly from the
  archived PDB/JSON files.
- One variant's monomer pLDDT (var9) is recomputed from the deposited structure and differs
  slightly from an earlier tabulated value; the recomputed value is the one this pipeline produces.

## License

MIT (see `LICENSE`).
