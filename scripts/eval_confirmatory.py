"""
eval_confirmatory.py
====================

Confirmatory analyses requested by the external manuscript evaluation
("Critical Evaluation of the Fluorescent-Protein Steric-Scan Manuscript").

Asks addressed for the red-FP twist/QY result:
  (1) permutation-based Spearman p-values (asymptotic p is said to be unreliable
      at n~tens; we check whether it actually is here),
  (2) a robust trend estimate (Theil-Sen) with CI sign check,
  (3) leave-one-FP-out jackknife + named outlier-deletion grid ("three famous
      points?"),
  (4) permutation p on the phi-vs-tau (P-bond vs I-bond) decomposition.

Cohort is the SAME one the manuscript headline uses: canonical-CD twist
(data/d_exp_canonical.csv) and slug-based unique-FP grouping
(data/lit_qy_curated.csv), matching bootstrap_cis.py. This reproduces the
manuscript's red per-crystal rho=-0.49 (n=56) and per-unique rho=-0.34 (n=38).

Output: data/eval_confirmatory.csv  (+ console summary)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RNG = np.random.default_rng(20260618)
N_PERM = 20000


# ---------------------------------------------------------------- loaders ---
def load():
    can = {r["pdb_id"].upper(): r for r in csv.DictReader(open(DATA / "d_exp_canonical.csv"))}
    qy, slug = {}, {}
    for r in csv.DictReader(open(DATA / "lit_qy_curated.csv")):
        q = r["lit_qy_fpbase"] or r["lit_qy_fpbase_recovered"]
        try:
            q = float(q)
        except Exception:
            q = None
        qy[r["pdb_id"].upper()] = q
        slug[r["pdb_id"].upper()] = r["fpbase_slug"] or r["seq_match_slug"]
    meta = {r["pdb_id"].upper(): r for r in csv.DictReader(open(DATA / "merged_for_aggregate.csv"))}
    return can, qy, slug, meta


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def signed_dev(theta):
    """Signed deviation from nearest planar axis (0 or +/-180), in (-90, 90]."""
    return (np.asarray(theta, float) + 90.0) % 180.0 - 90.0


def build_red(can, qy, slug, meta, chrom=None):
    """Return per-crystal and per-unique dicts of arrays for the red cohort.

    Columns: d (canonical combined), Pdev=|dev phi_canon|, Idev=|dev tau|, logqy.
    """
    recs = []
    for p, m in meta.items():
        if (m.get("color_class") or "").lower() != "red":
            continue
        if chrom and m.get("chrom_resname") != chrom:
            continue
        q = qy.get(p)
        if q is None or q <= 0:
            continue
        c = can.get(p)
        if not c:
            continue
        d = fnum(c["d_canonical"])
        tau = fnum(c["tau"])
        phi = fnum(c["phi_canon"])
        if d is None:
            continue
        recs.append(dict(pdb=p, slug=slug.get(p) or p, d=d,
                         Pdev=abs(signed_dev(phi)) if phi is not None else np.nan,
                         Idev=abs(signed_dev(tau)) if tau is not None else np.nan,
                         logqy=np.log10(q)))
    cryst = pd.DataFrame(recs)

    # per-unique: median over crystals of each FP
    g = cryst.groupby("slug")
    uniq = pd.DataFrame({
        "d": g["d"].median(), "Pdev": g["Pdev"].median(),
        "Idev": g["Idev"].median(), "logqy": g["logqy"].median(),
    }).reset_index()
    return cryst, uniq


# ------------------------------------------------------------- statistics ---
def perm_spearman(x, y, n_perm=N_PERM):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, n, np.nan
    rho_obs, p_asym = stats.spearmanr(x, y)
    rx = stats.rankdata(x)
    cnt = sum(abs(stats.spearmanr(rx, RNG.permutation(y))[0]) >= abs(rho_obs)
              for _ in range(n_perm))
    return rho_obs, (cnt + 1) / (n_perm + 1), n, p_asym


def main():
    can, qy, slug, meta = load()
    cryst, uniq = build_red(can, qy, slug, meta)
    nrq_c, nrq_u = build_red(can, qy, slug, meta, chrom="NRQ")
    crq_c, crq_u = build_red(can, qy, slug, meta, chrom="CRQ")

    rows = []

    print("=" * 78)
    print("1. PERMUTATION-BASED SPEARMAN p  (canonical d_exp vs log10 QY)")
    print("   [should reproduce manuscript: per-crystal -0.49 n56, per-unique -0.34 n38]")
    print("=" * 78)
    for label, d in [("red per crystal", cryst), ("red per unique FP", uniq),
                     ("NRQ per unique", nrq_u), ("CRQ per crystal", crq_c)]:
        rho, p_perm, n, p_asym = perm_spearman(d["d"], d["logqy"])
        print(f"  {label:<20} n={n:<3} rho={rho:+.3f}  p_perm={p_perm:.4f}  (p_asym={p_asym:.4f})")
        rows.append(dict(analysis=f"perm_spearman::{label}", n=n, val=rho,
                         lo=p_perm, hi=p_asym))

    print()
    print("=" * 78)
    print("2. PERMUTATION p ON P-vs-I DECOMPOSITION (canonical, red)")
    print("=" * 78)
    for label, d in [("red per crystal", cryst), ("red per unique FP", uniq)]:
        for coord in ("Pdev", "Idev"):
            rho, p_perm, n, p_asym = perm_spearman(d[coord], d["logqy"])
            print(f"  {label:<20} {coord:<5} n={n:<3} rho={rho:+.3f}  p_perm={p_perm:.4f}")
            rows.append(dict(analysis=f"perm_{coord}::{label}", n=n, val=rho,
                             lo=p_perm, hi=p_asym))

    print()
    print("=" * 78)
    print("3. THEIL-SEN ROBUST SLOPE  (log10 QY vs canonical d_exp)")
    print("=" * 78)
    for label, d in [("red per crystal", cryst), ("red per unique FP", uniq)]:
        dd = d.dropna(subset=["d", "logqy"])
        slope, intercept, lo, hi = stats.theilslopes(dd["logqy"], dd["d"])
        ols = stats.linregress(dd["d"], dd["logqy"])
        print(f"  {label:<20} n={len(dd):<3} Theil-Sen slope={slope:+.4f} "
              f"[95% {lo:+.4f}, {hi:+.4f}]  (OLS slope={ols.slope:+.4f})")
        rows.append(dict(analysis=f"theilsen::{label}", n=len(dd), val=slope,
                         lo=lo, hi=hi))

    print()
    print("=" * 78)
    print("4. LEAVE-ONE-FP-OUT JACKKNIFE + OUTLIER-DELETION GRID (red per unique)")
    print("=" * 78)
    base = uniq.dropna(subset=["d", "logqy"]).copy()
    rho_full = stats.spearmanr(base["d"], base["logqy"])[0]
    print(f"  baseline red per-unique rho = {rho_full:+.3f}  (n={len(base)})")

    jk = []
    for sl in base["slug"]:
        d = base[base["slug"] != sl]
        jk.append((sl, stats.spearmanr(d["d"], d["logqy"])[0]))
    jk.sort(key=lambda t: t[1])
    print(f"  leave-one-FP-out rho range: [{jk[0][1]:+.3f} (drop {jk[0][0]}), "
          f"{jk[-1][1]:+.3f} (drop {jk[-1][0]})]")
    rows.append(dict(analysis="jackknife_min", n=len(base) - 1, val=jk[0][1],
                     lo=np.nan, hi=np.nan))
    rows.append(dict(analysis="jackknife_max", n=len(base) - 1, val=jk[-1][1],
                     lo=np.nan, hi=np.nan))

    def drop(d, needle):
        return d[~d["slug"].astype(str).str.contains(needle, case=False, na=False)]

    grid = {
        "drop mscarlet*": drop(base, "mscarlet"),
        "drop mrouge": drop(base, "mrouge"),
        "drop mscarlet+mrouge": drop(drop(base, "mscarlet"), "mrouge"),
        "drop most-twisted FP": base[base["d"] < base["d"].max()],
        "drop 3 most-twisted": base.sort_values("d").iloc[:-3],
    }
    for label, d in grid.items():
        if len(d) < 4:
            continue
        rho, p_perm, n, _ = perm_spearman(d["d"], d["logqy"])
        print(f"  {label:<22} n={n:<3} rho={rho:+.3f}  p_perm={p_perm:.4f}")
        rows.append(dict(analysis=f"grid::{label}", n=n, val=rho, lo=p_perm, hi=np.nan))

    pd.DataFrame(rows).to_csv(DATA / "eval_confirmatory.csv", index=False)
    print(f"\nSaved {DATA / 'eval_confirmatory.csv'}")


if __name__ == "__main__":
    main()
