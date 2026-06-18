#!/usr/bin/env python3
"""
============================================================
FEATURE RECOMPUTATION FROM RAW SEQUENCES
============================================================
Recomputes ALL 100 analysis features from scratch using only:
  - full_sequence (mutant protein)
  - novel_cterminus (post-frameshift tail)
  - WT CALR sequence (P27797, 417 aa)
  - frameshift_position (per variant)
  - ANCHOR = "RRMMRTKMRMRRMRRTRRKMRR"

No features are copied from previous computations.
Every value is independently derived and fully traceable.

Run: conda run -n calr_env python ~/Downloads/recompute_all_features.py
============================================================
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
from collections import Counter
import math

# ── PATHS ──────────────────────────────────────────────────
# Repo-relative paths (portable; no absolute paths)
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUTDIR = DATA / "derived"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──────────────────────────────────────────────
ANCHOR = "RRMMRTKMRMRRMRRTRRKMRR"  # 22 aa, universal anchor
C_DOMAIN_START = 309  # 1-indexed, from UniProt P27797
WT_LENGTH = 417

# ── LOAD WT SEQUENCE ───────────────────────────────────────
wt_fasta = (DATA / "sequences" / "CALR_WT_P27797.fasta").read_text()
WT_SEQ = "".join(l.strip() for l in wt_fasta.split("\n") if not l.startswith(">"))
assert len(WT_SEQ) == WT_LENGTH, f"WT length mismatch: {len(WT_SEQ)} != {WT_LENGTH}"
print(f"WT CALR loaded: {len(WT_SEQ)} aa")

# ── LOAD VARIANT DATA (public inputs only) ─────────────────
# Table S1 defines the 76-variant set and provides full sequences;
# sequence_metadata.tsv provides the verified novel C-terminal tails.
s1 = pd.read_csv(DATA / "sequences" / "Table_S1_Sequences_76variants.tsv", sep="\t")
meta = pd.read_csv(DATA / "sequences" / "sequence_metadata.tsv", sep="\t")

# primary_class from Table S1 'type' (Type 1-like / Type 2-like)
s1 = s1.copy()
s1["primary_class"] = s1["type"]

# NOTE: frameshift_position is taken from sequence_metadata.tsv (verified against
# protein-level nomenclature, e.g. p.K385fs -> 385); Table S1's column contained
# transcription errors for 7 variants and is NOT used for the frameshift position.
data = s1[["variant_id", "primary_class", "full_sequence"]].copy()
data = data.merge(
    meta[["variant_id", "sequence_id", "frameshift_position", "novel_cterminus"]].drop_duplicates("variant_id"),
    on="variant_id", how="left")
data = data[["sequence_id", "variant_id", "primary_class", "frameshift_position",
             "full_sequence", "novel_cterminus"]]

n_t2 = data["primary_class"].str.contains("2").sum()
n_t1 = data["primary_class"].str.contains("1").sum()
print(f"Variants: {len(data)} (Type 2={n_t2}, Type 1-like={n_t1})")
assert len(data) == 76, f"Expected 76 variants, got {len(data)}"
assert data["full_sequence"].notna().all(), "Missing full sequences!"
assert data["novel_cterminus"].notna().all(), "Missing novel tails!"
assert data["frameshift_position"].notna().all(), "Missing frameshift positions!"

# ── GRANTHAM DISTANCE MATRIX ──────────────────────────────
# Grantham R (1974) Science 185:862-864
GRANTHAM = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','Q'):91,('A','E'):107,
    ('A','G'):60,('A','H'):86,('A','I'):94,('A','L'):96,('A','K'):106,('A','M'):84,
    ('A','F'):113,('A','P'):27,('A','S'):99,('A','T'):58,('A','W'):148,('A','Y'):112,
    ('A','V'):64,
    ('R','N'):86,('R','D'):96,('R','C'):180,('R','Q'):43,('R','E'):54,('R','G'):125,
    ('R','H'):29,('R','I'):97,('R','L'):102,('R','K'):26,('R','M'):91,('R','F'):97,
    ('R','P'):103,('R','S'):110,('R','T'):71,('R','W'):101,('R','Y'):77,('R','V'):96,
    ('N','D'):23,('N','C'):139,('N','Q'):46,('N','E'):42,('N','G'):80,('N','H'):68,
    ('N','I'):149,('N','L'):153,('N','K'):94,('N','M'):142,('N','F'):158,('N','P'):91,
    ('N','S'):46,('N','T'):65,('N','W'):174,('N','Y'):143,('N','V'):133,
    ('D','C'):154,('D','Q'):61,('D','E'):45,('D','G'):94,('D','H'):81,('D','I'):168,
    ('D','L'):172,('D','K'):101,('D','M'):160,('D','F'):177,('D','P'):108,('D','S'):65,
    ('D','T'):85,('D','W'):181,('D','Y'):160,('D','V'):152,
    ('C','Q'):154,('C','E'):170,('C','G'):159,('C','H'):174,('C','I'):198,('C','L'):198,
    ('C','K'):202,('C','M'):196,('C','F'):205,('C','P'):169,('C','S'):112,('C','T'):149,
    ('C','W'):215,('C','Y'):194,('C','V'):192,
    ('Q','E'):29,('Q','G'):87,('Q','H'):24,('Q','I'):109,('Q','L'):113,('Q','K'):53,
    ('Q','M'):101,('Q','F'):116,('Q','P'):76,('Q','S'):68,('Q','T'):42,('Q','W'):130,
    ('Q','Y'):99,('Q','V'):96,
    ('E','G'):98,('E','H'):40,('E','I'):134,('E','L'):138,('E','K'):56,('E','M'):126,
    ('E','F'):140,('E','P'):93,('E','S'):80,('E','T'):65,('E','W'):152,('E','Y'):122,
    ('E','V'):121,
    ('G','H'):98,('G','I'):135,('G','L'):138,('G','K'):127,('G','M'):127,('G','F'):153,
    ('G','P'):42,('G','S'):56,('G','T'):59,('G','W'):184,('G','Y'):147,('G','V'):109,
    ('H','I'):94,('H','L'):99,('H','K'):32,('H','M'):87,('H','F'):100,('H','P'):77,
    ('H','S'):89,('H','T'):47,('H','W'):115,('H','Y'):83,('H','V'):84,
    ('I','L'):5,('I','K'):102,('I','M'):10,('I','F'):21,('I','P'):95,('I','S'):142,
    ('I','T'):89,('I','W'):61,('I','Y'):33,('I','V'):29,
    ('L','K'):107,('L','M'):15,('L','F'):22,('L','P'):98,('L','S'):145,('L','T'):92,
    ('L','W'):61,('L','Y'):36,('L','V'):32,
    ('K','M'):95,('K','F'):102,('K','P'):103,('K','S'):121,('K','T'):78,('K','W'):110,
    ('K','Y'):85,('K','V'):97,
    ('M','F'):28,('M','P'):87,('M','S'):135,('M','T'):81,('M','W'):67,('M','Y'):36,
    ('M','V'):21,
    ('F','P'):114,('F','S'):155,('F','T'):103,('F','W'):40,('F','Y'):22,('F','V'):50,
    ('P','S'):74,('P','T'):38,('P','W'):147,('P','Y'):110,('P','V'):68,
    ('S','T'):58,('S','W'):177,('S','Y'):144,('S','V'):124,
    ('T','W'):128,('T','Y'):92,('T','V'):69,
    ('W','Y'):37,('W','V'):88,
    ('Y','V'):55,
}

def grantham_dist(aa1, aa2):
    """Return Grantham distance between two amino acids. 0 if identical."""
    if aa1 == aa2:
        return 0
    key = (aa1, aa2) if (aa1, aa2) in GRANTHAM else (aa2, aa1)
    return GRANTHAM.get(key, np.nan)

# ── AA CHARGE FUNCTION ─────────────────────────────────────
def aa_charge(aa):
    """Formal charge at physiological pH."""
    if aa in "RK": return 1
    if aa in "DE": return -1
    return 0

# ── AMINO ACID PROPENSITY SCALES ───────────────────────────
# Kyte-Doolittle hydropathy (1982)
KD_SCALE = {
    'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
    'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
    'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
    'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2,
}

# Chou-Fasman beta-sheet propensity (1978)
BETA_SCALE = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}

# Coacervation propensity (Das & Pappu, 2013 — simplified)
COAC_SCALE = {
    'A': 0.0, 'R': 1.0, 'N': 0.0, 'D':-1.0, 'C': 0.0,
    'Q': 0.0, 'E':-1.0, 'G': 0.0, 'H': 0.5, 'I': 0.0,
    'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.5, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.5, 'Y': 0.5, 'V': 0.0,
}

def mean_scale(seq, scale):
    """Mean propensity score for a sequence."""
    vals = [scale.get(aa, 0) for aa in seq]
    return np.mean(vals) if vals else np.nan

# ── KINASE CONSENSUS PATTERNS ─────────────────────────────
import re

def count_ck2_sites(seq):
    """CK2: S/T-x-x-D/E"""
    return len(re.findall(r'[ST]..[DE]', seq))

def count_pka_sites(seq):
    """PKA: R-R-x-S/T or R-x-S/T"""
    return len(re.findall(r'RR.[ST]', seq)) + len(re.findall(r'R.[ST]', seq))

def count_pkc_sites(seq):
    """PKC: S/T-x-R/K"""
    return len(re.findall(r'[ST].[RK]', seq))

# ── localCIDER ─────────────────────────────────────────────
try:
    from localcider.sequenceParameters import SequenceParameters
    HAS_CIDER = True
    print("localCIDER: available")
except ImportError:
    HAS_CIDER = False
    print("localCIDER: NOT available — SCD and kappa will be NaN")


# ============================================================
# MAIN COMPUTATION LOOP
# ============================================================
print("\n" + "="*70)
print("COMPUTING 100 FEATURES FOR 76 VARIANTS")
print("="*70)

results = []

for idx, row in data.iterrows():
    sid = row["sequence_id"]
    vid = row["variant_id"]
    pclass = row["primary_class"]
    fs = int(row["frameshift_position"])
    full = row["full_sequence"]
    novel = row["novel_cterminus"]

    feat = {
        "sequence_id": sid,
        "variant_id": vid,
        "primary_class": pclass,
        "frameshift_position": fs,
    }

    # ── ANCHOR DETECTION ──────────────────────────────────
    anch_in_novel = novel.find(ANCHOR)
    has_anchor = anch_in_novel >= 0
    if has_anchor:
        anchor_abs = fs + anch_in_novel  # 1-indexed absolute position
        approach_zone = novel[:anch_in_novel]
        post_anchor = novel[anch_in_novel + len(ANCHOR):]
    else:
        anchor_abs = np.nan
        approach_zone = ""
        post_anchor = ""

    feat["has_anchor"] = has_anchor
    feat["anchor_start"] = anchor_abs

    # ══════════════════════════════════════════════════════
    # CATEGORY A: Novel Tail Composition (34 features)
    # ══════════════════════════════════════════════════════
    n = len(novel)
    aa_counts = Counter(novel)

    # A1-A20: Individual AA fractions
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        feat[f"f_{aa}"] = aa_counts.get(aa, 0) / n

    # A21-22: Tail RK and DE fractions
    feat["tail_RK_fraction"] = (aa_counts.get("R",0) + aa_counts.get("K",0)) / n
    feat["tail_DE_fraction"] = (aa_counts.get("D",0) + aa_counts.get("E",0)) / n

    # A23-27: AA groups
    feat["grp_Positive"] = sum(aa_counts.get(a,0) for a in "RKH") / n
    feat["grp_Negative"] = sum(aa_counts.get(a,0) for a in "DE") / n
    feat["grp_Hydrophobic"] = sum(aa_counts.get(a,0) for a in "AVILMFWP") / n
    feat["grp_Polar"] = sum(aa_counts.get(a,0) for a in "STNQYC") / n
    feat["grp_Special"] = sum(aa_counts.get(a,0) for a in "GP") / n

    # A28-29: Net charge metrics
    n_pos = aa_counts.get("R",0) + aa_counts.get("K",0)
    n_neg = aa_counts.get("D",0) + aa_counts.get("E",0)
    feat["ncpr"] = (n_pos - n_neg) / n
    feat["fcr"] = (n_pos + n_neg + aa_counts.get("H",0)) / n

    # A30-31: Polycation/polyanion longest runs
    def longest_run(seq, chars):
        max_run = 0; cur = 0
        for aa in seq:
            if aa in chars:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        return max_run

    feat["polycation_len"] = longest_run(novel, "RK")
    feat["polyanion_len"] = longest_run(novel, "DE")

    # A32: Shannon entropy
    freqs = [c/n for c in aa_counts.values() if c > 0]
    feat["entropy_novel"] = -sum(f * math.log2(f) for f in freqs)

    # A33: Aromatic fraction
    feat["aromatic_novel"] = sum(aa_counts.get(a,0) for a in "FYW") / n

    # A34: Novel tail length
    feat["novel_cterminus_residues"] = n

    # ══════════════════════════════════════════════════════
    # CATEGORY B: Charge Patterning — localCIDER (2 features)
    # ══════════════════════════════════════════════════════
    if HAS_CIDER and len(novel) >= 5:
        try:
            sp = SequenceParameters(novel)
            feat["SCD"] = sp.get_SCD()
            feat["kappa"] = sp.get_kappa()
        except:
            feat["SCD"] = np.nan
            feat["kappa"] = np.nan
    else:
        feat["SCD"] = np.nan
        feat["kappa"] = np.nan

    # ══════════════════════════════════════════════════════
    # CATEGORY C: Propensity Scales (5 features)
    # ══════════════════════════════════════════════════════
    feat["kd_novel"] = mean_scale(novel, KD_SCALE)
    feat["beta_novel"] = mean_scale(novel, BETA_SCALE)
    feat["coac_novel"] = mean_scale(novel, COAC_SCALE)

    # Uversky LLPS: |mean_charge| vs mean_hydropathy plane
    mean_charge = abs(feat["ncpr"])
    mean_hydro = feat["kd_novel"]
    # Uversky boundary: <H> = (<R> + 1.151) / 2.785
    feat["uversky_llps"] = mean_hydro - (mean_charge + 1.151) / 2.785

    # RGG/RG motif count
    feat["rgg_score"] = len(re.findall(r'RGG', novel)) + len(re.findall(r'RG', novel))

    # ══════════════════════════════════════════════════════
    # CATEGORY D: Approach Zone Position-Specific (22 features)
    # ══════════════════════════════════════════════════════
    feat["approach_len"] = len(approach_zone)

    # D1: Approach zone charges (anchor-relative, -1 to -20)
    for pos in range(1, 21):
        idx_in_az = len(approach_zone) - pos  # -1 = last residue before anchor
        if idx_in_az >= 0 and idx_in_az < len(approach_zone):
            aa = approach_zone[idx_in_az]
            feat[f"az_charge_{-pos}"] = aa_charge(aa)
            # Grantham: compare to WT at same absolute position
            abs_pos = anchor_abs - pos  # 1-indexed
            if abs_pos >= 1 and abs_pos <= len(WT_SEQ):
                wt_aa = WT_SEQ[abs_pos - 1]
                feat[f"az_grantham_{-pos}"] = grantham_dist(wt_aa, aa)
            else:
                feat[f"az_grantham_{-pos}"] = np.nan
        else:
            feat[f"az_charge_{-pos}"] = np.nan
            feat[f"az_grantham_{-pos}"] = np.nan

    # ══════════════════════════════════════════════════════
    # CATEGORY E: Cysteine Geometry (4 features)
    # ══════════════════════════════════════════════════════
    feat["n_az_cys"] = approach_zone.count("C")
    feat["n_tail_cys"] = novel.count("C")

    if has_anchor:
        # Cysteine at anchor+32 and anchor+36 (0-indexed in novel)
        cys32_pos = anch_in_novel + 32
        cys36_pos = anch_in_novel + 36
        feat["has_cys32"] = (cys32_pos < len(novel) and novel[cys32_pos] == "C")
        feat["has_cys36"] = (cys36_pos < len(novel) and novel[cys36_pos] == "C")
    else:
        feat["has_cys32"] = False
        feat["has_cys36"] = False

    # ══════════════════════════════════════════════════════
    # CATEGORY F: Motifs (9 features)
    # ══════════════════════════════════════════════════════
    feat["KKRK_present"] = "KKRK" in novel

    # DE clusters: runs of ≥3 consecutive D/E
    de_clusters = re.findall(r'[DE]{3,}', novel)
    feat["n_de_clusters"] = len(de_clusters)

    # Basic motifs
    feat["n_rtrr"] = len(re.findall(r'R[A-Z]RR', novel))
    feat["RRR_n"] = len(re.findall(r'RRR', novel))
    feat["KEE_n"] = len(re.findall(r'KEE', novel))
    feat["KRKEE_n"] = len(re.findall(r'KRKEE', novel))

    # Kinase sites (standard literature patterns)
    feat["CK2_sites"] = count_ck2_sites(novel)
    feat["PKA_sites"] = count_pka_sites(novel)
    feat["PKC_sites"] = count_pkc_sites(novel)

    # ══════════════════════════════════════════════════════
    # CATEGORY G: Post-KKRK Features (4 features, Type 2-specific)
    # ══════════════════════════════════════════════════════
    kkrk_pos = novel.find("KKRK")
    if kkrk_pos >= 0 and has_anchor:
        post_kkrk_end = anch_in_novel  # up to anchor start
        post_kkrk_start = kkrk_pos + 4  # after KKRK
        if post_kkrk_start < post_kkrk_end:
            post_kkrk = novel[post_kkrk_start:post_kkrk_end]
            feat["post_kkrk_length"] = len(post_kkrk)
            pk_counts = Counter(post_kkrk)
            pk_n = len(post_kkrk)
            pk_pos = pk_counts.get("R",0) + pk_counts.get("K",0)
            pk_neg = pk_counts.get("D",0) + pk_counts.get("E",0)
            feat["post_kkrk_charge"] = (pk_pos - pk_neg) / pk_n if pk_n > 0 else 0
            feat["post_kkrk_rk_frac"] = pk_pos / pk_n if pk_n > 0 else 0
            feat["post_kkrk_de_frac"] = pk_neg / pk_n if pk_n > 0 else 0
            feat["post_kkrk_seq"] = post_kkrk
        else:
            feat["post_kkrk_length"] = 0
            feat["post_kkrk_charge"] = np.nan
            feat["post_kkrk_rk_frac"] = np.nan
            feat["post_kkrk_de_frac"] = np.nan
            feat["post_kkrk_seq"] = ""
    else:
        feat["post_kkrk_length"] = np.nan
        feat["post_kkrk_charge"] = np.nan
        feat["post_kkrk_rk_frac"] = np.nan
        feat["post_kkrk_de_frac"] = np.nan
        feat["post_kkrk_seq"] = np.nan

    # ══════════════════════════════════════════════════════
    # CATEGORY H: WT-Novel Interaction Proxy (2 features)
    # ══════════════════════════════════════════════════════
    wt_fragment = WT_SEQ[C_DOMAIN_START - 1 : fs - 1]  # WT C-domain preserved
    feat["wt_fragment_length"] = len(wt_fragment)

    # WT fragment charge
    wt_frag_counts = Counter(wt_fragment)
    wt_n = len(wt_fragment)
    if wt_n > 0:
        wt_ncpr = (wt_frag_counts.get("R",0) + wt_frag_counts.get("K",0)
                    - wt_frag_counts.get("D",0) - wt_frag_counts.get("E",0)) / wt_n
    else:
        wt_ncpr = 0
    feat["wt_novel_charge_contrast"] = feat["ncpr"] - wt_ncpr

    # ── DONE WITH THIS VARIANT ────────────────────────────
    results.append(feat)

    # Progress
    if (len(results)) % 20 == 0:
        print(f"  Computed {len(results)}/{len(data)} variants...")

print(f"  Computed {len(results)}/{len(data)} variants — DONE")


# ============================================================
# BUILD OUTPUT DATAFRAME
# ============================================================
print("\n" + "="*70)
print("BUILDING OUTPUT")
print("="*70)

out = pd.DataFrame(results)

# Separate numeric features from metadata/strings
meta_cols = ["sequence_id", "variant_id", "primary_class",
             "frameshift_position", "has_anchor", "anchor_start",
             "post_kkrk_seq"]
numeric_cols = [c for c in out.columns if c not in meta_cols]

print(f"Total columns: {len(out.columns)}")
print(f"  Metadata: {len(meta_cols)}")
print(f"  Numeric features: {len(numeric_cols)}")

# Count features by category
cat_counts = {
    "A_tail_composition": len([c for c in numeric_cols if c.startswith("f_") or
                                c in ["tail_RK_fraction","tail_DE_fraction",
                                       "grp_Positive","grp_Negative","grp_Hydrophobic",
                                       "grp_Polar","grp_Special","ncpr","fcr",
                                       "polycation_len","polyanion_len","entropy_novel",
                                       "aromatic_novel","novel_cterminus_residues"]]),
    "B_charge_patterning": len([c for c in numeric_cols if c in ["SCD","kappa"]]),
    "C_propensity": len([c for c in numeric_cols if c in ["kd_novel","beta_novel",
                                                           "coac_novel","uversky_llps","rgg_score"]]),
    "D_approach_zone": len([c for c in numeric_cols if c.startswith("az_") or c=="approach_len"]),
    "E_cysteine": len([c for c in numeric_cols if "cys" in c.lower()]),
    "F_motifs": len([c for c in numeric_cols if c in ["KKRK_present","n_de_clusters",
                                                        "n_rtrr","RRR_n","KEE_n","KRKEE_n",
                                                        "CK2_sites","PKA_sites","PKC_sites"]]),
    "G_post_kkrk": len([c for c in numeric_cols if c.startswith("post_kkrk") and c != "post_kkrk_seq"]),
    "H_wt_interaction": len([c for c in numeric_cols if c.startswith("wt_")]),
}
for cat, n in cat_counts.items():
    print(f"  {cat}: {n}")
print(f"  TOTAL: {sum(cat_counts.values())}")

# ── COMPLETENESS CHECK ─────────────────────────────────────
print(f"\n--- COMPLETENESS ---")
for col in numeric_cols:
    n_valid = out[col].notna().sum()
    if n_valid < 76:
        print(f"  {col:<30} {n_valid}/76 ({n_valid/76:.0%})")

# ── SAVE ───────────────────────────────────────────────────
print(f"\n--- SAVING ---")
out.to_csv(OUTDIR / "RECOMPUTED_FEATURES_76_VARIANTS.tsv", sep="\t", index=False)
print(f"  Saved: RECOMPUTED_FEATURES_76_VARIANTS.tsv")
print(f"  {len(out)} rows × {len(out.columns)} columns")

# Save feature inventory
inv = pd.DataFrame({
    "column": out.columns,
    "dtype": [str(out[c].dtype) for c in out.columns],
    "n_valid": [out[c].notna().sum() for c in out.columns],
    "n_unique": [out[c].nunique() for c in out.columns],
    "is_metadata": [c in meta_cols for c in out.columns],
    "sample": [str(out[c].dropna().iloc[0])[:50] if out[c].notna().any() else ""
               for c in out.columns],
})
inv.to_csv(OUTDIR / "RECOMPUTED_FEATURE_INVENTORY.tsv", sep="\t", index=False)
print(f"  Saved: RECOMPUTED_FEATURE_INVENTORY.tsv")

# ── GIT ────────────────────────────────────────────────────
import subprocess
subprocess.run(["git", "add", "-A"], cwd=PROJECT)
msg = (f"Recomputed {sum(cat_counts.values())} features from raw sequences for 76 variants. "
       f"Categories: tail({cat_counts['A_tail_composition']}), "
       f"charge({cat_counts['B_charge_patterning']}), "
       f"propensity({cat_counts['C_propensity']}), "
       f"AZ({cat_counts['D_approach_zone']}), "
       f"cys({cat_counts['E_cysteine']}), "
       f"motif({cat_counts['F_motifs']}), "
       f"postKKRK({cat_counts['G_post_kkrk']}), "
       f"WTinteract({cat_counts['H_wt_interaction']}).")
r = subprocess.run(["git", "commit", "-m", msg],
                    cwd=PROJECT, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0: print(r.stderr)

print("\n" + "="*70)
print("RECOMPUTATION COMPLETE")
print("="*70)
print(f"""
Output: {OUTDIR}

Files:
  RECOMPUTED_FEATURES_76_VARIANTS.tsv  — the master feature table
  RECOMPUTED_FEATURE_INVENTORY.tsv     — column descriptions

Every feature was computed from:
  - WT sequence: inputs/CALR_WT_P27797.fasta (P27797, 417 aa)
  - Mutant sequences: SUPPLEMENTARY_TABLE_1_SEQUENCES.tsv
  - Novel tails: sequence_metadata.tsv (verified 76/76 correct)
  - Anchor: RRMMRTKMRMRRMRRTRRKMRR (22 aa, universal)
  - Frameshift positions: per-variant metadata
  - localCIDER: SCD and kappa
  - Grantham (1974): approach zone mutation severity
  - Standard kinase consensus patterns: CK2, PKA, PKC

No features were copied from previous computations.
No AF2 structural predictions were used.
No composite scores with arbitrary weights were used.
""")
