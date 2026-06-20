"""
Independent verification of the manuscript's headline numbers.

Recomputes the key QY results from the rawest files on disk
(merged_for_aggregate.csv for geometry + color + chrom code,
lit_qy_curated.csv for curated FPbase QY) and prints each result next
to the value claimed in manuscript.md. Nothing here trusts any saved
intermediate or any prose; it goes back to per-structure data.

Run:  python scripts/verify_manuscript_numbers.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

GEO = "data/merged_for_aggregate.csv"
QY = "data/lit_qy_curated.csv"


def curated_qy(qy):
    # Prefer a direct FPbase/pdb match, fall back to the 99% seq-match recovery.
    q = qy["lit_qy_fpbase"].copy()
    q = q.fillna(qy["lit_qy_fpbase_recovered"])
    return q


def line(label, got, claimed):
    g = "  ".join(f"{x}" for x in got)
    print(f"  {label:40s} disk: {g:32s} | manuscript: {claimed}")


def crystal(sub, col):
    """Per-crystal Spearman of col vs log10(QY)."""
    s = sub.dropna(subset=[col, "qy_curated"])
    r, p = spearmanr(s[col], np.log10(s["qy_curated"]))
    return round(r, 3), f"n={len(s)}", f"p={p:.1e}"


def unique(sub, col):
    """Per-unique-FP Spearman: median col and median QY per fp_id, vs log10(QY)."""
    g = (sub.dropna(subset=[col, "qy_curated", "fp_id"])
            .groupby("fp_id").agg(x=(col, "median"), q=("qy_curated", "median")))
    r, p = spearmanr(g["x"], np.log10(g["q"]))
    return round(r, 3), f"n={len(g)}", f"p={p:.2f}"


def unique_medians(sub, col):
    """Per-unique-FP median values of col (group by fp_id; structures with no
    slug are kept as singletons keyed by pdb_id). No QY filter."""
    s = sub.dropna(subset=[col]).copy()
    s["grp"] = s["fp_id"].fillna(s["pdb_id"])
    return s.groupby("grp")[col].median().values


def main():
    geo = pd.read_csv(GEO)
    qy = pd.read_csv(QY)
    qy["qy_curated"] = curated_qy(qy)
    qy["fp_id"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])
    df = geo.merge(qy[["pdb_id", "qy_curated", "fp_id"]], on="pdb_id", how="left")
    df = df[df.qy_curated.notna() & (df.qy_curated > 0)]

    d = "d_exp_to_planar_deg"
    f = "f_allowed_folded"
    red = df[df.color_class == "red"]
    grn = df[df.color_class == "green"]

    # d_exp_to_planar uses the production occupancy-ranked + canonical-CD values;
    # NRQ/CRQ subfamilies use chrom_resname (the deposited acylimine code), the
    # same selector the manuscript's "(NRQ)" label refers to.
    print("=" * 86)
    print("HEADLINE: distance-to-planar vs curated QY (manuscript section 3.4 / Figure 3)")
    print("=" * 86)
    line("within-RED d_exp vs QY (per crystal)", crystal(red, d),
         "rho=-0.49, p=1.3e-4, n=56")
    line("within-RED d_exp vs QY (per unique)", unique(red, d),
         "rho=-0.34, p=0.03, n=38")
    line("within-NRQ d_exp vs QY (per unique)", unique(red[red.chrom_resname == "NRQ"], d),
         "rho=-0.45, p=0.03, n=25")
    line("within-CRQ d_exp vs QY (per crystal)", crystal(red[red.chrom_resname == "CRQ"], d),
         "rho=-0.68, p=0.02, n=12")
    line("within-GREEN d_exp vs QY (per unique)", unique(grn, d),
         "rho=-0.03, n=52 (null)")

    print()
    print("=" * 86)
    print("NULL: f_allowed vs QY within color class (manuscript section 3.3: should NOT predict)")
    print("=" * 86)
    line("within-GREEN f_allowed vs QY (per unique)", unique(grn, f),
         "rho=+0.07, p=0.62, n=52 (null)")
    line("within-RED f_allowed vs QY (per unique)", unique(red, f),
         "rho=+0.06, p=0.70, n=38 (null)")

    print()
    print("=" * 86)
    print("ROBUSTNESS: within-red d_exp vs QY across objections (manuscript section S8 table)")
    print("=" * 86)
    red_res = red.assign(resolution=pd.to_numeric(red["resolution"], errors="coerce"))
    line("Red <=2.5 A (per unique)", unique(red_res[red_res.resolution <= 2.5], d),
         "rho=-0.37, p=0.02, n=37")
    line("Red <=2.0 A (per unique)", unique(red_res[red_res.resolution <= 2.0], d),
         "rho=-0.34, p=0.07, n=30")
    # most-twisted member that DOES enter the regression (AausFP2/6S68 is QY=0 and
    # already absent); drops mscarlet3-h (8ZXH, d_exp 76.7 deg), the max-d_exp red.
    rcrys = red.dropna(subset=[d, "qy_curated"])
    rcrys = rcrys.drop(rcrys[d].idxmax())
    line("Red most-twisted member removed (per crystal)", crystal(rcrys, d),
         "rho=-0.49, p=2e-4, n=55")

    print()
    print("=" * 86)
    print("CAGE SIZE: NRQ vs CRQ within red, per unique FP (manuscript section S7)")
    print("=" * 86)
    # all structures of each subfamily (no QY filter), grouped per unique FP
    dfa = geo.merge(qy[["pdb_id", "fp_id"]], on="pdb_id", how="left")
    redall = dfa[dfa.color_class == "red"]
    nrq = unique_medians(redall[redall.chrom_resname == "NRQ"], f)
    crq = unique_medians(redall[redall.chrom_resname == "CRQ"], f)
    U, p = mannwhitneyu(nrq, crq, alternative="two-sided")
    line("NRQ vs CRQ f_allowed (per unique)",
         (f"NRQ={np.median(nrq):.4f} (n={len(nrq)})",
          f"CRQ={np.median(crq):.4f} (n={len(crq)})", f"p={p:.2f}"),
         "NRQ 0.0066 vs CRQ 0.0162, ~2.5x, p=0.04")

    print()
    print("=" * 86)
    print("ORTHOGONALITY: the two predictors are independent (manuscript: rho=-0.03)")
    print("=" * 86)
    both = geo[geo[d].notna() & geo[f].notna()]
    r, p = spearmanr(both[f], both[d])
    line("f_allowed vs d_exp (all structures)", (round(r, 3), f"n={len(both)}", f"p={p:.2f}"),
         "rho=-0.03 (orthogonal)")


if __name__ == "__main__":
    main()
