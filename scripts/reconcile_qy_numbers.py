"""
Full numeric reconciliation of every QY/ESP claim in manuscript.md against
ONE canonical QY source (lit_qy_curated.csv), merged with geometry + controls
from merged_for_aggregate.csv. Prints a claim-by-claim table:
  CLAIM | manuscript value | recomputed value | MATCH / MISMATCH

Canonical QY = lit_qy_fpbase, fallback lit_qy_fpbase_recovered (the merge
that reproduces the within-red headline of -0.52, n=56).
Unique-FP key = fpbase_name.

Run: python3 scripts/reconcile_qy_numbers.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, mannwhitneyu, f as fdist
import statsmodels.formula.api as smf

GEO = "data/merged_for_aggregate.csv"
QY = "data/lit_qy_curated.csv"


def load():
    geo = pd.read_csv(GEO)
    qy = pd.read_csv(QY)
    qy["qy"] = qy["lit_qy_fpbase"].fillna(qy["lit_qy_fpbase_recovered"])
    qy["fp"] = qy["fpbase_name"]
    df = geo.merge(qy[["pdb_id", "qy", "fp"]], on="pdb_id", how="left")
    return df


def verdict(got, claim, tol=0.03):
    """Compare a recomputed rho/median to the claimed value within tol."""
    try:
        return "MATCH" if abs(float(got) - float(claim)) <= tol else "*** MISMATCH ***"
    except Exception:
        return "?"


def row(label, got, claimed, status):
    print(f"  {label:46s} disk={got:<26s} ms={claimed:<22s} {status}")


def sp(df, a, b):
    s = df[[a, b]].dropna()
    if len(s) < 3:
        return np.nan, np.nan, len(s)
    r, p = spearmanr(s[a], s[b])
    return r, p, len(s)


def per_fp(df, value="median"):
    """One row per unique FP: median geometry, FP-level QY (first non-null)."""
    g = df.dropna(subset=["fp"]).groupby("fp")
    out = g.agg(
        d_exp_to_planar_deg=("d_exp_to_planar_deg", "median"),
        f_allowed_folded=("f_allowed_folded", "median"),
        qy=("qy", "median"),
        color_class=("color_class", "first"),
        chrom_resname=("chrom_resname", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
    ).reset_index()
    return out


def main():
    df = load()
    d, f = "d_exp_to_planar_deg", "f_allowed_folded"
    print("CANONICAL: lit_qy_curated (lit_qy_fpbase|recovered) x merged_for_aggregate")
    print(f"structures with curated QY: {df.qy.notna().sum()} | unique FPs: {df.loc[df.qy.notna(),'fp'].nunique()}\n")

    print("=" * 100)
    print("A. HEADLINE within-color d_exp_to_planar vs QY (per structure)")
    print("=" * 100)
    red = df[df.color_class == "red"]
    r, p, n = sp(red, d, "qy"); row("within-RED d_exp vs QY", f"rho={r:.2f} n={n} p={p:.1e}", "rho=-0.52 n=56", verdict(r, -0.52))
    for code, claim, cn in [("NRQ", -0.45, 38), ("CRQ", -0.48, 20)]:
        sub = red[red.chrom_resname == code]
        r, p, n = sp(sub, d, "qy")
        row(f"within-{code} d_exp vs QY", f"rho={r:.2f} n={n} p={p:.1e}", f"rho={claim} n={cn}", verdict(r, claim))
    grn = df[df.color_class == "green"]
    r, p, n = sp(grn, d, "qy"); row("within-GREEN d_exp vs QY", f"rho={r:.2f} n={n} p={p:.2f}", "rho=-0.10 n=146", verdict(r, -0.10))
    r, p, n = sp(df, d, "qy"); row("POOLED d_exp vs QY", f"rho={r:.2f} n={n} p={p:.1e}", "rho=-0.18 n=290", verdict(r, -0.18))

    print("\n" + "=" * 100)
    print("B. Same correlations, per UNIQUE FP")
    print("=" * 100)
    fp = per_fp(df)
    redf = fp[fp.color_class == "red"]
    r, p, n = sp(redf, d, "qy"); row("within-RED per-uFP", f"rho={r:.2f} n={n} p={p:.1e}", "rho=-0.46 n=37", verdict(r, -0.46))
    for code, claim, cn in [("NRQ", -0.50, 27), ("CRQ", -0.58, 9)]:
        sub = redf[redf.chrom_resname == code]
        r, p, n = sp(sub, d, "qy")
        row(f"within-{code} per-uFP", f"rho={r:.2f} n={n} p={p:.2f}", f"rho={claim} n={cn}", verdict(r, claim, tol=0.06))
    r, p, n = sp(fp, d, "qy"); row("POOLED per-uFP d_exp vs QY", f"rho={r:.2f} n={n} p={p:.1e}", "rho=-0.38", verdict(r, -0.38, tol=0.05))

    print("\n" + "=" * 100)
    print("C. NULL: f_allowed vs QY, and orthogonality")
    print("=" * 100)
    r, p, n = sp(df, f, "qy"); row("POOLED f_allowed vs QY", f"rho={r:.2f} n={n} p={p:.2f}", "rho=-0.05 p=0.39", verdict(r, -0.05))
    r, p, n = sp(df, f, d); row("f_allowed vs d_exp (orthogonality)", f"rho={r:.2f} n={n} p={p:.2f}", "rho=-0.03", verdict(r, -0.03))

    print("\n" + "=" * 100)
    print("D. per-unique-FP f_allowed_folded medians by color (section 3.3)")
    print("=" * 100)
    claims = {"blue": 0.027, "cyan": 0.005, "green": 0.013, "yellow": 0.012, "red": 0.006}
    allfp = per_fp(df.assign(fp=df["fp"].fillna(df["pdb_id"])))  # fall back to pdb if no FP name
    for c, claim in claims.items():
        sub = allfp[allfp.color_class == c][f].dropna()
        med = sub.median() if len(sub) else np.nan
        row(f"median f_allowed_folded [{c}]", f"{med:.4f} n={len(sub)}", f"{claim}", verdict(med, claim, tol=0.004))

    print("\n" + "=" * 100)
    print("E. NRQ vs CRQ f_allowed_folded (section 3.6, per unique FP)")
    print("=" * 100)
    for code, claim in [("NRQ", 0.0044), ("CRQ", 0.0147)]:
        sub = allfp[allfp.chrom_resname == code][f].dropna()
        med = sub.median() if len(sub) else np.nan
        row(f"median f_allowed_folded [{code}]", f"{med:.4f} n={len(sub)}", f"{claim}", verdict(med, claim, tol=0.003))

    print("\n" + "=" * 100)
    print("F. Multivariate OLS on log10(QY)  (canonical QY; sections 3.4 / 2.7)")
    print("=" * 100)
    m = df.dropna(subset=["qy", f, d, "chrom_contacts", "b_factor_ratio", "minor_axis", "d_centroid_to_planar_deg"]).copy()
    m = m[m.qy > 0]
    m["log_qy"] = np.log10(m.qy)
    m["cc"] = m.color_class.fillna("unknown")
    for col in [f, d, "d_centroid_to_planar_deg", "chrom_contacts", "b_factor_ratio", "minor_axis"]:
        m["z_" + col] = (m[col] - m[col].mean()) / m[col].std(ddof=0)
    M1 = smf.ols("log_qy ~ C(cc)", data=m).fit()
    M2 = smf.ols("log_qy ~ C(cc) + z_chrom_contacts + z_b_factor_ratio + z_minor_axis", data=m).fit()
    M3 = smf.ols("log_qy ~ C(cc) + z_chrom_contacts + z_b_factor_ratio + z_minor_axis + z_f_allowed_folded", data=m).fit()
    M3p = smf.ols("log_qy ~ C(cc) + z_chrom_contacts + z_b_factor_ratio + z_minor_axis + z_d_exp_to_planar_deg + z_d_centroid_to_planar_deg", data=m).fit()

    def ftest(r0, r1):
        ddf = int(r1.df_model - r0.df_model)
        F = ((r0.ssr - r1.ssr) / ddf) / (r1.ssr / r1.df_resid)
        return F, float(1 - fdist.cdf(F, ddf, r1.df_resid)), ddf, int(r1.df_resid)

    print(f"  n used = {int(M1.nobs)}   (manuscript: 290 for M0-M3; 249 for M3')")
    row("M1 R2", f"{M1.rsquared:.3f}", "0.647", verdict(M1.rsquared, 0.647))
    row("M2 R2", f"{M2.rsquared:.3f}", "0.656", verdict(M2.rsquared, 0.656))
    row("M3 R2 (+f_allowed)", f"{M3.rsquared:.3f}", "0.659", verdict(M3.rsquared, 0.659))
    F, p, a, b = ftest(M2, M3); row("partial F f_allowed (M3 vs M2)", f"F={F:.2f} p={p:.2f}", "F=1.7 p=0.19", verdict(F, 1.7, tol=0.8))
    row("M3' R2 (+d_exp+d_centroid)", f"{M3p.rsquared:.3f}", "0.631", verdict(M3p.rsquared, 0.631, tol=0.05))
    F, p, a, b = ftest(M2, M3p); row("partial F position (M3' vs M2)", f"F={F:.2f} p={p:.1e}", "F=6.4 p=0.002", verdict(F, 6.4, tol=2.0))
    if "z_d_exp_to_planar_deg" in M3p.params:
        row("coef z_d_exp p-value", f"p={M3p.pvalues['z_d_exp_to_planar_deg']:.1e}", "p=5e-4", "see disk")

    print("\nDONE. MISMATCH lines above are the manuscript numbers to correct.")


if __name__ == "__main__":
    main()
