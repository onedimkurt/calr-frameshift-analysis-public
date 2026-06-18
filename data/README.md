# Data

## Included in this repository (small, public inputs)

```
sequences/Table_S1_Sequences_76variants.tsv   76 variant identifiers + full sequences
sequences/sequence_metadata.tsv               novel C-terminal tails, frameshift positions
sequences/CALR_WT_P27797.fasta                wild-type CALR reference (UniProt P27797)
sequences/calr_full_sequences.fasta           variant full sequences (FASTA)
FEATURE_DICTIONARY.tsv                         feature name → definition mapping
```

These are sufficient to reproduce the sequence-feature, clustering, and review-validation
analyses end to end.

## Archived separately on Zenodo (large structural files)

The ColabFold structure predictions (monomer and homodimer PDB/PAE/JSON, ~several GB) are
not stored in this git repository. They are deposited on Zenodo:

> **Zenodo DOI:** `[DOI — fill in after deposit]`
> Access during peer review: `[reviewer token link — fill in]`

Contents of the Zenodo archive:
- `results_monomers*` — monomer predictions (per-variant rank_001 PDB + scores JSON)
- `results_homodimers*`, `crosslinks_homodimers*` — homodimer predictions
- (and the unpacked per-variant structure folders used by the structural scripts)

### Pointing the structural scripts at the downloaded data

After downloading and unpacking the Zenodo archive, set these environment variables (or edit
the path variables near the top of the relevant scripts):

```bash
export CALR_MONO_DIR=/path/to/unpacked/phase3_structures      # monomer pLDDT (af2_definitive.py)
export CALR_HOMODIMER_DIR=/path/to/unpacked/homodimer_pdbs    # C19_type2_register.py
export CALR_PAE_GLOB="/path/to/C14_dimer_pae*.json"           # C14_pae_figure.py
```

The homodimer ZIP archive paths are defined near the top of
`02_structural/af2_definitive.py` (`MONO_ZIP`, `HOMO_ZIP1`, `HOMO_ZIP2`); set `STRUCT_DIR`
to the directory containing the downloaded ZIPs to re-run structural extraction.

## Regenerated outputs

`data/derived/` is created when you run the pipeline and is not tracked in git
(see `.gitignore`). Every file in it is reproducible from the inputs above plus the
Zenodo-archived structures.
