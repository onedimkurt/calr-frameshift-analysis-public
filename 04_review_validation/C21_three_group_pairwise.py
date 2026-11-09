#!/usr/bin/env python3
"""
C21 - Three-group pairwise statistics (regeneration)

Regenerates sequence_three_group_pairwise.tsv from the CURRENT feature matrix.

Why this exists
---------------
The file shipped as data/derived/three_group_validation/statistics/
sequence_three_group_pairwise.tsv is not produced by any step of the published
pipeline (see docs/PIPELINE.md). Its p-values were computed on an earlier
feature set: for 21 of 84 features its own stored group means disagree with
RECOMPUTED_FEATURES_76_VARIANTS.tsv. Supplementary Table S6 and the
significance brackets of Figures 1, 3, 4 and S3-S6 inherit those stale
p-values while quoting current means.

Method (reverse-engineered from the stale file and reproduced 256/256)
----------------------------------------------------------------------
  * two-sided Mann-Whitney U per feature per pair
  * Cohen's d with pooled SD  (same formula as 01_clustering)
  * Benjamini-Hochberg applied SEPARATELY WITHIN each pairwise comparison
  * sig: *** <0.001, ** <0.01, * <0.05, else ns

Output schema and `comparison` labels are byte-compatible with the file the
figure scripts read (FIGURES_FINAL/scripts/supplementary_batch.py, SEQ_STATS).

Usage
-----
    python C21_three_group_pairwise.py --features <matrix.tsv> \
        --clusters <cluster_assignments.tsv> --out <out.tsv>

Uses scipy when available. Without scipy it falls back to a tie-corrected
normal approximation; rerun with scipy before publishing the numbers.
"""
import argparse, csv, math, sys

COMPARISONS = [
    ("ET_vs_MF-strong",      "Type 2",   "Type 1-A"),
    ("ET_vs_MF-weak",        "Type 2",   "Type 1-B"),
    ("MF-strong_vs_MF-weak", "Type 1-A", "Type 1-B"),
]
ID_COLS = {"sequence_id", "variant_id", "primary_class", "has_anchor"}

try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v); ties = []; i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            rk[order[k]] = avg
        ties.append(j - i + 1); i = j + 1
    return rk, ties

def mann_whitney_p(a, b):
    """Two-sided p. scipy when present, else tie-corrected normal approximation."""
    if HAVE_SCIPY:
        return float(mannwhitneyu(a, b, alternative="two-sided")[1])
    v = list(a) + list(b); rk, ties = _ranks(v)
    n1, n2 = len(a), len(b); N = n1 + n2
    U = sum(rk[:n1]) - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    tie = sum(t ** 3 - t for t in ties)
    sd = math.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie / (N * (N - 1))))
    if sd == 0:
        return 1.0
    z = (abs(U - mu) - 0.5) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    m1 = sum(g1) / n1; m2 = sum(g2) / n2
    v1 = sum((x - m1) ** 2 for x in g1) / (n1 - 1) if n1 > 1 else 0.0
    v2 = sum((x - m2) ** 2 for x in g2) / (n2 - 1) if n2 > 1 else 0.0
    ps = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (m1 - m2) / ps if ps > 1e-10 else 0.0

def bh(pvals):
    """Benjamini-Hochberg step-up; identical to 01_clustering/true_unsupervised_clustering.py."""
    n = len(pvals)
    if n == 0:
        return []
    si = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    adj[si[-1]] = min(1.0, pvals[si[-1]])
    for i in range(n - 2, -1, -1):
        adj[si[i]] = min(pvals[si[i]] * n / (i + 1), adj[si[i + 1]], 1.0)
    return adj

def sig_label(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=3)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.features), delimiter="\t"))
    sub = {r["sequence_id"]: r.get("seq_subgroup", "").strip()
           for r in csv.DictReader(open(a.clusters), delimiter="\t")}

    groups = {"Type 2": [], "Type 1-A": [], "Type 1-B": []}
    for r in rows:
        pc = r.get("primary_class", "").strip()
        if pc == "Type 2-like":
            groups["Type 2"].append(r)
        else:
            g = sub.get(r["sequence_id"], "")
            if g in ("Type 1-A", "Type 1-B"):
                groups[g].append(r)
    sizes = {k: len(v) for k, v in groups.items()}
    print(f"group sizes: {sizes}")
    assert sizes == {"Type 2": 32, "Type 1-A": 25, "Type 1-B": 19}, "unexpected group sizes"

    hdr = list(rows[0].keys())
    feats = []
    for c in hdr:
        if c in ID_COLS or c == "seq_subgroup":
            continue
        vals = [r[c] for r in rows if r[c] not in ("", None)]
        if not vals:
            continue
        try:
            [float(v) for v in vals]
        except ValueError:
            continue
        feats.append(c)
    print(f"numeric features considered: {len(feats)}")
    if not HAVE_SCIPY:
        print("WARNING: scipy not found - using normal approximation. "
              "Rerun with scipy before publishing.", file=sys.stderr)

    out = []
    for label, g1, g2 in COMPARISONS:
        block = []
        for f in feats:
            v1 = [float(r[f]) for r in groups[g1] if r[f] not in ("", None)]
            v2 = [float(r[f]) for r in groups[g2] if r[f] not in ("", None)]
            if len(v1) < a.min_n or len(v2) < a.min_n:
                continue
            if len(set(v1 + v2)) == 1:
                continue
            p = mann_whitney_p(v1, v2)
            d = cohens_d(v1, v2)
            block.append({"feature": f, "comparison": label,
                          "mean_1": sum(v1) / len(v1), "mean_2": sum(v2) / len(v2),
                          "d": d, "abs_d": abs(d), "p_raw": p,
                          "n1": len(v1), "n2": len(v2)})
        adj = bh([b["p_raw"] for b in block])
        for b, q in zip(block, adj):
            b["p_BH"] = q
            b["sig"] = sig_label(q)
        print(f"  {label:24} features={len(block):3}  significant={sum(1 for b in block if b['sig']!='ns')}")
        out.extend(block)

    cols = ["feature", "comparison", "mean_1", "mean_2", "d", "abs_d",
            "p_raw", "n1", "n2", "p_BH", "sig"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"wrote {len(out)} rows -> {a.out}")

if __name__ == "__main__":
    main()
