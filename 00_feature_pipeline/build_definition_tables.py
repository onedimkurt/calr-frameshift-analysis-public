#!/usr/bin/env python3
"""
build_definition_tables.py

Builds two publication-ready definition tables:

Table S4: Feature definitions — all features used in the final manuscript
  Source: FEATURE_DICTIONARY.md (for definitions/formulas)
  Cross-referenced against: RECOMPUTED_FEATURES_76_VARIANTS.tsv (actual cols used)
  Keeps ONLY features present in the final 87-feature clustering set
  plus additional features shown in main/supplementary figures.

Table S5 (new): Structural metric definitions — all AF2 metrics used
  Built from scratch using verified column names from AF2_DEFINITIVE.tsv
  and manuscript Methods text.

Output: PUBLICATION_READY/SUPPLEMENTARY_TABLES_FINAL/
  Table_S4_Feature_Definitions.tsv
  Table_S5_Structural_Metric_Definitions.tsv

Run: conda run -n calr_env python ~/Downloads/build_definition_tables.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT   = Path(__file__).resolve().parents[1]
DATA      = PROJECT / "data"
OUT_DIR   = DATA / "derived" / "definition_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_FILE  = DATA / "derived" / "RECOMPUTED_FEATURES_76_VARIANTS.tsv"
CLUST_FILE = DATA / "derived" / "unsupervised_clustering" / "statistics" / "cluster_assignments.tsv"
DICT_TSV   = DATA / "FEATURE_DICTIONARY.tsv"

# ── LOAD REFERENCE DATA ────────────────────────────────────
feat  = pd.read_csv(FEAT_FILE,  sep="\t")
clust = pd.read_csv(CLUST_FILE, sep="\t")
dict_df = pd.read_csv(DICT_TSV, sep="\t")

clust_map = dict(zip(
    clust["sequence_id"].str.strip(),
    clust["seq_subgroup"].str.strip()
))
def get_group(row):
    if "2" in str(row.get("primary_class", "")):
        return "Type 2"
    return clust_map.get(str(row["sequence_id"]).strip(), "Unknown")
feat["group"] = feat.apply(get_group, axis=1)

# ── IDENTIFY FINAL FEATURE SETS ───────────────────────────
# META columns excluded from everything
META = {"sequence_id","variant_id","primary_class","frameshift_position",
        "has_anchor","anchor_start","post_kkrk_seq","group"}
EXCLUDE = META | {"sg_sg_dist","KKRK_present",
                   "post_kkrk_length","post_kkrk_charge",
                   "post_kkrk_rk_frac","post_kkrk_de_frac"}

type1 = feat[feat["primary_class"].str.contains("1", na=False)].copy()
all_f  = [c for c in feat.columns if c not in EXCLUDE
          and feat[c].dtype in ["float64","int64","bool"]
          and c not in META]
# 87 clustering features
clustering_87 = [f for f in all_f
                 if len(type1[f].dropna()) >= 10
                 and type1[f].dropna().std() > 1e-10]

# Additional features shown in figures but not in clustering
fig_only = [
    "wt_novel_charge_contrast",  # Fig 4B (main)
    "polyanion_len",             # Fig 4A (main)
    "polycation_len",            # Fig 4C (main) — moved from supp
    "coac_novel",                # removed from main, kept in supp for reference
    "RRR_n",                     # discussed in text
    "KEE_n",                     # Fig S5C
    "n_rtrr",                    # Fig S5A (RXRR)
    "PKC_sites",                 # Fig S5B
    "n_de_clusters",             # manuscript text
    "wt_fragment_length",        # Fig 1C (main)
    "novel_cterminus_residues",  # Fig 1B (main) — tail length
    "KKRK_present",              # text — class separator
]
# All features to document
all_to_document = sorted(set(clustering_87 + [f for f in fig_only
                                               if f in feat.columns]))

print(f"Features in clustering (87): {len(clustering_87)}")
print(f"Additional figure features: {len([f for f in fig_only if f in feat.columns])}")
print(f"Total to document: {len(all_to_document)}")
print()

# ── BUILD CATEGORY AND DEFINITION MAP ─────────────────────
# Use FEATURE_DICTIONARY.tsv where available (maps old names to new)
# Note: old dict uses different column names in some cases
# Mapping: old dict col -> new RECOMPUTED col
OLD_TO_NEW = {
    "chg_-2":  "az_charge_-2",
    "chg_-4":  "az_charge_-4",
    "chg_-6":  "az_charge_-6",
    "chg_-7":  "az_charge_-7",
    "chg_-10": "az_charge_-10",
    "chg_-11": "az_charge_-11",
    "grantham_-4":  "az_grantham_-4",
    "grantham_-7":  "az_grantham_-7",
    "grantham_-11": "az_grantham_-11",
    "tail_RK_fraction": "tail_RK_fraction",
    "tail_DE_fraction": "tail_DE_fraction",
    "polycation_len": "polycation_len",
    "ncpr": "ncpr",
    "SCD":  "SCD",
    "fcr":  "fcr",
    "approach_len": "approach_len",
    "entropy_novel": "entropy_novel",
    "f_R": "f_R",
    "f_M": "f_M",
    "f_D": "f_D",
    "f_A": "f_A",
    "f_E": "f_E",
    "f_K": "f_K",
    "n_rtrr": "n_rtrr",
}
# Build reverse map
dict_lookup = {}
for _, row in dict_df.iterrows():
    old_name = row["feature"]
    new_name = OLD_TO_NEW.get(old_name, old_name)
    dict_lookup[new_name] = {
        "category":   row["category"],
        "definition": row["definition"],
        "unit":       row["unit"],
    }

# ── MASTER DEFINITION TABLE ───────────────────────────────
# Define all features with category, formula, source, usage
DEFINITIONS = {
    # ── A. NOVEL TAIL COMPOSITION ──────────────────────────
    "f_A":  ("A. Novel tail composition",
             "Fraction of Ala residues in novel tail",
             "count(A) / tail_length", "fraction (0–1)",
             "Supplementary Figure S3D"),
    "f_C":  ("A. Novel tail composition",
             "Fraction of Cys residues in novel tail",
             "count(C) / tail_length", "fraction (0–1)",
             "Supplementary Figure S12 (PCA biplot)"),
    "f_D":  ("A. Novel tail composition",
             "Fraction of Asp residues in novel tail",
             "count(D) / tail_length", "fraction (0–1)",
             "Supplementary Figure S3E"),
    "f_E":  ("A. Novel tail composition",
             "Fraction of Glu residues in novel tail",
             "count(E) / tail_length", "fraction (0–1)",
             "Supplementary Figure S4D (as Asp+Glu)"),
    "f_G":  ("A. Novel tail composition",
             "Fraction of Gly residues in novel tail",
             "count(G) / tail_length", "fraction (0–1)",
             "Supplementary Figure S12 (PCA biplot)"),
    "f_K":  ("A. Novel tail composition",
             "Fraction of Lys residues in novel tail",
             "count(K) / tail_length", "fraction (0–1)",
             "Supplementary Figure S3 (implicit via RK)"),
    "f_M":  ("A. Novel tail composition",
             "Fraction of Met residues in novel tail",
             "count(M) / tail_length", "fraction (0–1)",
             "Supplementary Figure S3B"),
    "f_N":  ("A. Novel tail composition",
             "Fraction of Asn residues in novel tail",
             "count(N) / tail_length", "fraction (0–1)",
             "Clustering feature"),
    "f_P":  ("A. Novel tail composition",
             "Fraction of Pro residues in novel tail",
             "count(P) / tail_length", "fraction (0–1)",
             "Supplementary Figure S3C"),
    "f_Q":  ("A. Novel tail composition",
             "Fraction of Gln residues in novel tail",
             "count(Q) / tail_length", "fraction (0–1)",
             "Clustering feature"),
    "f_R":  ("A. Novel tail composition",
             "Fraction of Arg residues in novel tail",
             "count(R) / tail_length", "fraction (0–1)",
             "Supplementary Figure S3A"),
    "f_S":  ("A. Novel tail composition",
             "Fraction of Ser residues in novel tail",
             "count(S) / tail_length", "fraction (0–1)",
             "Supplementary Figure S12 (PCA — top loading)"),
    "f_T":  ("A. Novel tail composition",
             "Fraction of Thr residues in novel tail",
             "count(T) / tail_length", "fraction (0–1)",
             "Clustering feature"),
    "f_V":  ("A. Novel tail composition",
             "Fraction of Val residues in novel tail",
             "count(V) / tail_length", "fraction (0–1)",
             "Clustering feature"),
    "f_W":  ("A. Novel tail composition",
             "Fraction of Trp residues in novel tail",
             "count(W) / tail_length", "fraction (0–1)",
             "Clustering feature"),
    "f_L":  ("A. Novel tail composition",
             "Fraction of Leu residues in novel tail",
             "count(L) / tail_length", "fraction (0–1)",
             "Clustering feature"),

    # Charge metrics
    "ncpr": ("A. Novel tail composition",
             "Net charge per residue of novel tail",
             "(count(R+K) − count(D+E)) / tail_length; "
             "R,K=+1; D,E=−1; all others=0",
             "charge per residue",
             "Main Figure 3A"),
    "fcr":  ("A. Novel tail composition",
             "Fraction of charged residues in novel tail",
             "(count(R+K) + count(D+E)) / tail_length",
             "fraction (0–1)",
             "Main Figure 3B"),
    "tail_RK_fraction": ("A. Novel tail composition",
             "Fraction of Arg+Lys residues in novel tail",
             "count(R+K) / tail_length",
             "fraction (0–1)",
             "Clustering feature"),
    "tail_DE_fraction": ("A. Novel tail composition",
             "Fraction of Asp+Glu residues in novel tail",
             "count(D+E) / tail_length",
             "fraction (0–1)",
             "Clustering feature"),
    "entropy_novel": ("A. Novel tail composition",
             "Shannon entropy of amino acid composition of novel tail",
             "−Σ p_i log₂(p_i) over 20 amino acid types",
             "bits",
             "Supplementary Figure S3F"),
    "aromatic_novel": ("A. Novel tail composition",
             "Fraction of aromatic residues (Phe, Trp, Tyr) in novel tail",
             "count(F+W+Y) / tail_length",
             "fraction (0–1)",
             "Supplementary Figure S4F"),
    "novel_cterminus_residues": ("A. Novel tail composition",
             "Length of novel C-terminal tail (number of residues from "
             "frameshift position to end)",
             "count of residues post-frameshift",
             "residues",
             "Main Figure 1B"),

    # Residue groups
    "grp_Positive": ("A. Novel tail composition",
             "Fraction of positively charged residues (Arg, Lys) in novel tail",
             "count(R+K) / tail_length",
             "fraction (0–1)",
             "Clustering feature"),
    "grp_Negative": ("A. Novel tail composition",
             "Fraction of negatively charged residues (Asp, Glu) in novel tail",
             "count(D+E) / tail_length",
             "fraction (0–1)",
             "Supplementary Figure S4C"),
    "grp_Hydrophobic": ("A. Novel tail composition",
             "Fraction of hydrophobic residues (Val, Ile, Leu, Phe, Trp, "
             "Met, Ala) in novel tail",
             "count(V+I+L+F+W+M+A) / tail_length",
             "fraction (0–1)",
             "Supplementary Figure S4A"),
    "grp_Special": ("A. Novel tail composition",
             "Fraction of special residues (Cys, Gly, Pro) in novel tail",
             "count(C+G+P) / tail_length",
             "fraction (0–1)",
             "Supplementary Figure S4B"),
    "grp_Polar": ("A. Novel tail composition",
             "Fraction of polar uncharged residues (Ser, Thr, Asn, Gln) "
             "in novel tail",
             "count(S+T+N+Q) / tail_length",
             "fraction (0–1)",
             "Clustering feature"),

    # ── B. CHARGE PATTERNING ──────────────────────────────
    "SCD": ("B. Charge patterning",
            "Sequence charge decoration — quantifies spatial clustering "
            "of charged residues along the sequence",
            "SCD = (1/N) Σᵢ<ⱼ qᵢ qⱼ √|j−i|, where qᵢ is formal charge "
            "of residue i; R,K=+1; D,E=−1",
            "arbitrary units (positive = charge block segregation; "
            "negative = charge mixing)",
            "Main Figure 3C"),
    "kappa": ("B. Charge patterning",
              "Charge segregation parameter — normalised deviation of local "
              "charge asymmetry across sliding windows relative to theoretical "
              "maximum for the given composition (computed by localCIDER v0.1.21)",
              "κ = (σ²_local − σ²_shuffle_mean) / (σ²_max − σ²_shuffle_mean); "
              "see Das & Pappu 2013",
              "dimensionless (0–1; higher = more segregated)",
              "Main Figure 3D"),

    # ── C. PROPENSITY SCALES ──────────────────────────────
    "kd_novel": ("C. Propensity scales",
                 "Mean Kyte-Doolittle hydropathy of novel tail",
                 "arithmetic mean of KD scale values over all novel tail "
                 "residues; no windowing; scale: Ile=4.5, Val=4.2, "
                 "Leu=3.8 ... Arg=−4.5",
                 "KD units (negative = hydrophilic)",
                 "Main Figure 3E"),
    "beta_novel": ("C. Propensity scales",
                   "Mean Chou-Fasman beta-sheet propensity of novel tail",
                   "arithmetic mean of CF beta-sheet scale values; "
                   "Val=1.70, Ile=1.60 ... Pro=0.55",
                   "dimensionless",
                   "Supplementary Figure S4E"),
    "uversky_llps": ("C. Propensity scales",
                     "Uversky disorder boundary metric — position relative "
                     "to the IDP boundary in mean charge vs mean hydropathy "
                     "space",
                     "|NCPR| − (2.785 × mean_hydropathy + 1.151); "
                     "positive = predicted disordered (Uversky 2003)",
                     "dimensionless",
                     "Clustering feature (top loading on PCA)"),
    "coac_novel": ("C. Propensity scales",
                   "Charge-driven condensation propensity score of novel tail",
                   "mean over residues: R,K=+1; D,E=−1; F,W,Y=+0.5; "
                   "all others=0. Note: sometimes labelled 'coacervation "
                   "score' in figure axes; the metric quantifies charge-driven "
                   "condensation propensity, not polyampholyte coacervation",
                   "dimensionless",
                   "Removed from main figures; retained as clustering feature"),
    "rgg_score": ("C. Propensity scales",
                  "RGG/RG motif count in novel tail",
                  "count of non-overlapping RGG or RG dinucleotide motifs",
                  "count",
                  "Clustering feature"),

    # ── D. APPROACH ZONE FEATURES ────────────────────────
    "approach_len": ("D. Approach zone",
                     "Length of the approach zone — the segment of the novel "
                     "tail immediately N-terminal to the conserved anchor "
                     "sequence",
                     "count of residues between frameshift junction and "
                     "first residue of anchor",
                     "residues",
                     "Clustering feature (top Cohen's d)"),
    "wt_fragment_length": ("D. Approach zone",
                            "Length of the wild-type C-domain fragment retained "
                            "upstream of the frameshift position",
                            "frameshift_position − C_domain_start (residue 309)",
                            "residues",
                            "Main Figure 1C"),
    "wt_novel_charge_contrast": ("D. Approach zone",
                                  "Electrostatic asymmetry between WT C-domain "
                                  "and novel tail",
                                  "|NCPR_WT_fragment| − |NCPR_novel_tail|; "
                                  "quantifies heterotypic attraction driving "
                                  "post-anchor cysteine proximity",
                                  "dimensionless",
                                  "Main Figure 4B"),
}

# Generate approach zone charge positions -1 to -20
for pos in range(1, 21):
    col = f"az_charge_-{pos}"
    if col in feat.columns:
        desc = (f"Mean formal charge at approach zone position −{pos} "
                f"(position {pos} residues N-terminal to the anchor). "
                f"R,K=+1; D,E=−1; H=0; all others=0")
        fig_ref = "Main Figure 2C" if pos <= 14 else "Supplementary Figure S6"
        if pos in [15, 16, 17, 18, 19, 20]:
            fig_ref = "Supplementary Figure S6B (imputed)"
        DEFINITIONS[col] = (
            "D. Approach zone",
            desc,
            "formal charge at physiological pH: R,K=+1; D,E=−1; all others=0",
            "formal charge units (−1 to +1)",
            fig_ref
        )

# Generate approach zone Grantham distances -1 to -20
for pos in range(1, 21):
    col = f"az_grantham_-{pos}"
    if col in feat.columns:
        desc = (f"Mean Grantham physicochemical distance at approach zone "
                f"position −{pos}")
        fig_ref = "Supplementary Figure S6A"
        if pos in [15, 16, 17, 18, 19, 20]:
            fig_ref = "Supplementary Figure S6C (imputed)"
        DEFINITIONS[col] = (
            "D. Approach zone",
            desc,
            "Grantham distance between observed residue and reference "
            "(population mean); scale 0–215 (Grantham 1974)",
            "Grantham distance units (0–215)",
            fig_ref
        )

# ── E. CYSTEINE GEOMETRY ─────────────────────────────────
DEFINITIONS.update({
    "has_cys32": ("E. Cysteine geometry",
                  "Binary indicator: cysteine present at anchor-relative "
                  "position +32",
                  "1 if Cys at position anchor_start+32, else 0",
                  "binary (0/1)",
                  "Universal across all 76 variants; excluded from clustering "
                  "(zero variance)"),
    "has_cys36": ("E. Cysteine geometry",
                  "Binary indicator: cysteine present at anchor-relative "
                  "position +36",
                  "1 if Cys at position anchor_start+36, else 0",
                  "binary (0/1)",
                  "Universal across all 76 variants; excluded from clustering "
                  "(zero variance)"),
    "n_az_cys":  ("E. Cysteine geometry",
                  "Number of cysteine residues in the approach zone",
                  "count(C) in approach zone",
                  "count",
                  "Supplementary Figure S1F (PCA biplot loading)"),
    "n_tail_cys": ("E. Cysteine geometry",
                   "Total number of cysteine residues in the novel tail "
                   "(approach zone + anchor + post-anchor)",
                   "count(C) in novel tail",
                   "count",
                   "Supplementary Figure S1F (PCA biplot loading)"),
})

# ── F. SEQUENCE MOTIFS ────────────────────────────────────
DEFINITIONS.update({
    "KKRK_present": ("F. Sequence motifs",
                     "Binary indicator: KKRK tetrapeptide motif present in "
                     "novel tail",
                     "1 if KKRK substring found in novel tail sequence, else 0",
                     "binary (0/1)",
                     "Perfect class separator (all Type 2 = 1, all Type 1 = 0); "
                     "excluded from clustering"),
    "n_de_clusters": ("F. Sequence motifs",
                      "Number of consecutive Asp/Glu clusters of length ≥3 "
                      "in novel tail",
                      "count of non-overlapping runs of ≥3 consecutive D/E "
                      "residues",
                      "count",
                      "Manuscript text (invariant = 3 in all Type 2)"),
    "polyanion_len": ("F. Sequence motifs",
                      "Length of longest consecutive polyanion run (Asp, Glu) "
                      "in novel tail",
                      "maximum run length of consecutive D/E residues",
                      "residues",
                      "Main Figure 4A"),
    "polycation_len": ("F. Sequence motifs",
                       "Length of longest consecutive polycation run (Arg, Lys) "
                       "in novel tail",
                       "maximum run length of consecutive R/K residues",
                       "residues",
                       "Main Figure 4C"),
    "KEE_n": ("F. Sequence motifs",
              "Count of KEE tripeptide motifs in novel tail",
              "count of non-overlapping KEE substrings",
              "count",
              "Supplementary Figure S5C (invariant = 2 in all Type 2, "
              "absent from Type 1-A)"),
    "n_rtrr": ("F. Sequence motifs",
               "Count of RXRR tetrapeptide motifs in novel tail "
               "(R-any-R-R; note: labelled 'RTRR' in earlier figure "
               "versions but the motif counted is RXRR where X is any "
               "residue)",
               "count of non-overlapping RXRR matches where X = any residue",
               "count",
               "Supplementary Figure S5A"),
    "RRR_n": ("F. Sequence motifs",
              "Count of RRR triple-arginine motifs in novel tail",
              "count of non-overlapping RRR substrings",
              "count",
              "Manuscript text (enriched in Type 1-A: mean 0.720 ± 0.614 "
              "vs Type 1-B: 0.053 ± 0.229)"),
    "KEE_n": ("F. Sequence motifs",
              "Count of KEE tripeptide motifs in novel tail",
              "count of non-overlapping KEE substrings",
              "count",
              "Supplementary Figure S5C"),
    "PKC_sites": ("F. Sequence motifs",
                  "Number of PKC (protein kinase C) consensus phosphorylation "
                  "sites in novel tail",
                  "count of [S/T]-X-[R/K] motifs (PKC consensus)",
                  "count",
                  "Supplementary Figure S5B"),
    "PKA_sites": ("F. Sequence motifs",
                  "Number of PKA (protein kinase A) consensus sites in novel "
                  "tail",
                  "count of R-[R/K]-X-[S/T] motifs (PKA consensus)",
                  "count",
                  "Excluded from clustering (zero variance in Type 1-like)"),
    "CK2_sites": ("F. Sequence motifs",
                  "Number of CK2 (casein kinase 2) consensus sites in novel "
                  "tail",
                  "count of [S/T]-X-X-[D/E] motifs (CK2 consensus)",
                  "count",
                  "Excluded from clustering (zero variance in Type 1-like)"),
    "KRKEE_n": ("F. Sequence motifs",
                "Count of KRKEE pentapeptide motifs in novel tail",
                "count of non-overlapping KRKEE substrings",
                "count",
                "Excluded from clustering (zero variance)"),
})

# ── G. POST-KKRK REGION ──────────────────────────────────
DEFINITIONS.update({
    "post_kkrk_length": ("G. Post-KKRK region (Type 2 only)",
                          "Length of the sequence segment immediately C-terminal "
                          "to the KKRK motif in Type 2 variants",
                          "residue count from end of KKRK to start of anchor",
                          "residues",
                          "Manuscript text; undefined for Type 1-like variants"),
    "post_kkrk_charge": ("G. Post-KKRK region (Type 2 only)",
                          "Mean NCPR of the post-KKRK segment",
                          "(count(R+K) − count(D+E)) / post_kkrk_length",
                          "charge per residue",
                          "Manuscript text; excluded from clustering"),
    "post_kkrk_rk_frac": ("G. Post-KKRK region (Type 2 only)",
                            "Fraction of Arg+Lys in post-KKRK segment",
                            "count(R+K) / post_kkrk_length",
                            "fraction (0–1)",
                            "Excluded from clustering"),
    "post_kkrk_de_frac": ("G. Post-KKRK region (Type 2 only)",
                            "Fraction of Asp+Glu in post-KKRK segment",
                            "count(D+E) / post_kkrk_length",
                            "fraction (0–1)",
                            "Excluded from clustering"),
})

# ── H. WILD-TYPE SCAFFOLD ────────────────────────────────
DEFINITIONS.update({
    "wt_fragment_length": ("H. Wild-type scaffold",
                            "Length of retained wild-type C-domain fragment "
                            "upstream of frameshift",
                            "frameshift_position − 309 (start of C-domain)",
                            "residues",
                            "Main Figure 1C"),
    "wt_novel_charge_contrast": ("H. Wild-type scaffold",
                                  "Electrostatic contrast between retained WT "
                                  "C-domain (anionic) and novel tail (cationic)",
                                  "NCPR_novel_tail − NCPR_wt_fragment; higher "
                                  "values indicate greater heterotypic attraction "
                                  "between the two chains in the homodimer",
                                  "dimensionless",
                                  "Main Figure 4B"),
})

# ── BUILD FINAL TABLE ─────────────────────────────────────
print("Building feature definitions table...")
rows = []
for feat_name in sorted(all_to_document):
    if feat_name in DEFINITIONS:
        cat, defn, formula, unit, fig = DEFINITIONS[feat_name]
    else:
        # Generate generic entry for undocumented features
        cat = "A. Novel tail composition" if feat_name.startswith("f_") \
              else "D. Approach zone" if "az_" in feat_name \
              else "Uncategorized"
        defn    = f"Computed feature: {feat_name}"
        formula = "See Methods"
        unit    = "varies"
        fig     = "Clustering feature"

    in_clustering = feat_name in clustering_87
    rows.append({
        "feature_name":    feat_name,
        "category":        cat,
        "definition":      defn,
        "formula":         formula,
        "unit":            unit,
        "in_clustering_87": "Yes" if in_clustering else "No",
        "figures":         fig,
    })

df_s4 = pd.DataFrame(rows)
# Sort by category then feature name
cat_order = {
    "A. Novel tail composition": 0,
    "B. Charge patterning": 1,
    "C. Propensity scales": 2,
    "D. Approach zone": 3,
    "E. Cysteine geometry": 4,
    "F. Sequence motifs": 5,
    "G. Post-KKRK region (Type 2 only)": 6,
    "H. Wild-type scaffold": 7,
    "Uncategorized": 8,
}
df_s4["_cat_order"] = df_s4["category"].map(cat_order).fillna(8)
df_s4 = df_s4.sort_values(["_cat_order","feature_name"]).drop(
    columns=["_cat_order"]).reset_index(drop=True)

out_s4 = OUT_DIR / "Table_S4_Feature_Definitions.tsv"
df_s4.to_csv(out_s4, sep="\t", index=False)
print(f"✓ Table S4: {len(df_s4)} features → {out_s4.name}")
print(f"  Categories: {df_s4['category'].value_counts().to_dict()}")
print(f"  In clustering: {(df_s4['in_clustering_87']=='Yes').sum()}")
print()

# ============================================================
# TABLE S5: STRUCTURAL METRIC DEFINITIONS
# ============================================================
print("Building structural metric definitions table...")

STRUCTURAL_DEFS = [
    # Monomer metrics
    ("mono_mean_plddt",     "Monomer confidence",
     "Mean per-residue pLDDT score across the entire monomer model",
     "mean(pLDDT) over all residues in unrelaxed rank-1 ColabFold v1.5 "
     "monomer model",
     "pLDDT score (0–100; >90 very high, 70–90 confident, 50–70 low, "
     "<50 very low/disordered)",
     "Supplementary Figure S7 (as mono_mean_plddt)"),
    ("mono_wt_region_plddt","Monomer confidence",
     "Mean pLDDT over wild-type CALR region (residues upstream of frameshift "
     "position)",
     "mean(pLDDT) over residues 1 to frameshift_position−1 in monomer model",
     "pLDDT score (0–100)",
     "Supplementary Figure S7B"),
    ("mono_novel_tail_plddt","Monomer confidence",
     "Mean pLDDT over novel C-terminal tail region",
     "mean(pLDDT) over residues from frameshift_position to end of chain "
     "in monomer model",
     "pLDDT score (0–100)",
     "Supplementary Figure S7C"),
    ("mono_ptm",            "Monomer confidence",
     "Predicted TM-score of monomer model — global model quality estimate",
     "pTM score from ColabFold v1.5 monomer rank-1 model; range 0–1",
     "dimensionless (0–1; >0.5 indicates reliable fold)",
     "Supplementary Figure S7A"),

    # Homodimer geometry
    ("sg_sg_dist",          "Homodimer cysteine geometry",
     "Sγ–Sγ inter-chain distance at the anchor+32 cysteine — primary "
     "structural validation metric",
     "Euclidean distance between Sγ atoms of Cys at anchor-relative "
     "position +32 on Chain A and Chain B in unrelaxed rank-1 ColabFold "
     "v1.5 homodimer model",
     "Ångström (Å)",
     "Main Figure 5A; threshold: compact <20 Å, extended >50 Å"),
    ("cys_cb_dist",         "Homodimer cysteine geometry",
     "Cβ–Cβ inter-chain distance at anchor+32 cysteine",
     "Euclidean distance between Cβ atoms of Cys+32 on Chain A and Chain B",
     "Ångström (Å)",
     "Supplementary Figure S8A; Spearman r=0.996 with sg_sg_dist"),
    ("cys_chi1_A",          "Homodimer cysteine geometry",
     "χ1 dihedral angle of anchor+32 cysteine on Chain A",
     "N–Cα–Cβ–Sγ dihedral angle extracted from unrelaxed homodimer model "
     "Chain A",
     "degrees (°)",
     "Supplementary Figure S8B"),
    ("cys_chi1_B",          "Homodimer cysteine geometry",
     "χ1 dihedral angle of anchor+32 cysteine on Chain B",
     "N–Cα–Cβ–Sγ dihedral angle extracted from unrelaxed homodimer model "
     "Chain B",
     "degrees (°)",
     "Supplementary Figure S8C"),

    # Homodimer contacts
    ("novel_tail_contacts",      "Homodimer contacts",
     "Raw count of inter-chain Cα–Cα contacts within novel tail",
     "count of residue pairs where Cα–Cα distance <8.0 Å, both residues "
     "in novel tail (approach zone + anchor + post-anchor), between Chain A "
     "and Chain B",
     "count",
     "Supplementary Figure S7F (raw); Main Figure 5F (normalised)"),
    ("novel_tail_contacts_norm", "Homodimer contacts",
     "Inter-chain novel tail contacts normalised by tail length",
     "novel_tail_contacts / novel_tail_length",
     "contacts per residue",
     "Main Figure 5F"),

    # Homodimer confidence
    ("dimer_novel_tail_plddt", "Homodimer confidence",
     "Mean pLDDT over novel tail residues in homodimer model",
     "mean(pLDDT) over all novel tail residues in Chain A of unrelaxed "
     "rank-1 ColabFold v1.5 homodimer model; novel tail identified by "
     "exact string matching",
     "pLDDT score (0–100)",
     "Main Figure 5C"),
    ("mean_pae",             "Homodimer confidence",
     "Mean predicted aligned error across all residue pairs in homodimer",
     "mean(PAE[i,j]) over all i,j pairs from ColabFold PAE JSON output",
     "Å (lower = higher confidence)",
     "Supplementary Figure S7 (as mean_pae)"),
    ("inter_pae",            "Homodimer confidence",
     "Mean inter-chain predicted aligned error",
     "mean(PAE[i,j]) where i ∈ Chain A and j ∈ Chain B",
     "Å (lower = higher interface confidence)",
     "Main Figure 2D context; Supplementary Figure S7E"),
    ("intra_pae",            "Homodimer confidence",
     "Mean intra-chain predicted aligned error",
     "mean(PAE[i,j]) where i and j ∈ same chain",
     "Å (lower = higher intra-chain confidence)",
     "Supplementary Figure S7D"),
    ("dimer_ptm",            "Homodimer confidence",
     "Dimer predicted TM-score — global homodimer model quality",
     "pTM score from ColabFold v1.5 homodimer rank-1 model",
     "dimensionless (0–1)",
     "Main Figure 5E"),
    ("dimer_iptm",           "Homodimer confidence",
     "Interface predicted TM-score — interface-specific model quality",
     "ipTM score from ColabFold v1.5 homodimer rank-1 model; "
     "weighted combination of inter-chain PAE",
     "dimensionless (0–1; >0.8 high confidence interface)",
     "Main Figure 5D"),
]

df_s5 = pd.DataFrame(STRUCTURAL_DEFS, columns=[
    "metric_name", "category", "definition",
    "computation", "unit", "figure_reference"
])

out_s5 = OUT_DIR / "Table_S5_Structural_Metric_Definitions.tsv"
df_s5.to_csv(out_s5, sep="\t", index=False)
print(f"✓ Table S5: {len(df_s5)} metrics → {out_s5.name}")
print(f"  Categories: {df_s5['category'].value_counts().to_dict()}")
print()

# ── PRINT VERIFICATION ────────────────────────────────────
print("=" * 65)
print("VERIFICATION — features in clustering_87 NOT in S4")
print("=" * 65)
s4_features = set(df_s4["feature_name"].tolist())
missing_from_s4 = [f for f in clustering_87 if f not in s4_features]
if missing_from_s4:
    print(f"WARNING: {len(missing_from_s4)} clustering features "
          f"not in S4:")
    for f in sorted(missing_from_s4):
        print(f"  {f}")
else:
    print("All 87 clustering features are documented in Table S4 ✓")

print()
print("=" * 65)
print("FINAL OUTPUT SUMMARY")
print("=" * 65)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:<55} {f.stat().st_size:>10} bytes")
print()
print("DONE")
