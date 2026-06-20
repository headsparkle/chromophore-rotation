#!/usr/bin/env python3
"""
investigate_yellow_minus_ht.py
==============================

The previous wedge test surfaced one nominally significant
within-class hit: yellow FPs (n = 35) show a positive Spearman
correlation between QY and the negative-HT (slope -1, "bottom HT")
wedge fraction, rho = +0.388, p = 0.021. The direction matches
Megley et al. 2009's MD prediction. This script asks whether the
hit is robust enough to take seriously.

Probes
------
1) List the 35 yellow FPs with QY, negative-HT wedge fraction,
   and chromophore code. Look for obvious leverage points.
2) Plot QY vs neg-HT fraction; flag the leverage points.
3) Leave-one-out: re-compute Spearman rho with each FP dropped,
   record how far rho moves. If any single drop pushes rho below
   0.20 or above 0.55 the result is fragile.
4) Bootstrap: 2000 resamples of the 35-FP set, report 95 % CI on rho.
5) Vary wedge half-width (10, 15, 20 degrees).
6) Use Megley's exact 2009 slope -0.857 as the axis direction
   instead of the symbolic -1.
7) Project allowed-cell centres onto the -HT axis and take the
   std dev; correlate THAT with QY.
8) Stratify by chromophore code within yellow.
9) Benjamini-Hochberg FDR across all 16 within-class wedge tests
   from the qy_wedge_test.csv table.

No new manuscript edit until the user looks at the diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCAN_DIR = DATA_DIR / "scans"
FIG_DIR = PROJECT_ROOT / "figures"
MASTER_CSV = PROJECT_ROOT.parent / "gfp-barrel-geometry" / "data" / "merged_complete_data.csv"

TOL = 0.4
WEDGES_BASE = {
    "phi_axis":     (0.0, 1.0),
    "positive_HT":  (np.cos(np.deg2rad(+45)), np.sin(np.deg2rad(+45))),
    "negative_HT":  (np.cos(np.deg2rad(-45)), np.sin(np.deg2rad(-45))),
    "tau_axis":     (1.0, 0.0),
}
# 2009 slope -0.857 axis: direction vector (1, -0.857) normalised
m2009 = -0.857
u2009 = np.array([1.0, m2009]) / np.hypot(1.0, m2009)
WEDGES_EXTENDED = {
    **WEDGES_BASE,
    "neg_HT_2009_slope": (u2009[0], u2009[1]),
}


def compute_wedge_and_sigma(npz_path: Path, half_widths=(10, 15, 20)) -> dict:
    z = np.load(npz_path)
    tau_g = z["tau_grid"]; phi_g = z["phi_grid"]
    overlap = z["overlap_map"]
    allowed = overlap <= TOL
    n_all = int(allowed.sum())
    out = {"n_allowed": n_all}
    if n_all < 3:
        for k in WEDGES_EXTENDED:
            for h in half_widths:
                out[f"frac_{k}_{h}"] = np.nan
            out[f"sigma_{k}"] = np.nan
        return out
    T, P = np.meshgrid(tau_g, phi_g, indexing="ij")
    Ta, Pa = T[allowed], P[allowed]
    tau_c = float(Ta.mean()); phi_c = float(Pa.mean())
    Tc, Pc = Ta - tau_c, Pa - phi_c
    ang = np.degrees(np.arctan2(Pc, Tc))
    for k, (ux, uy) in WEDGES_EXTENDED.items():
        dir_deg = np.degrees(np.arctan2(uy, ux))
        diff = (ang - dir_deg) % 180.0
        axial = np.where(diff > 90.0, 180.0 - diff, diff)
        for h in half_widths:
            out[f"frac_{k}_{h}"] = float((axial <= h).sum()) / n_all
        proj = Tc * ux + Pc * uy
        out[f"sigma_{k}"] = float(proj.std())
    return out


def main() -> int:
    master = pd.read_csv(MASTER_CSV)
    curated = pd.read_csv(DATA_DIR / "lit_qy_curated.csv")
    summary = pd.read_csv(DATA_DIR / "scan_all_summary.csv")
    df = summary.merge(
        master[["pdb_id", "color_class", "chromophore_type"]],
        on="pdb_id", how="left",
    ).merge(
        curated[["pdb_id", "lit_qy_fpbase", "qy_provenance",
                 "seq_match_name", "fpbase_name"]],
        on="pdb_id", how="left",
    )
    df = df[(df["status"] == "ok") & df["lit_qy_fpbase"].notna()].copy()
    df["lit_qy"] = df["lit_qy_fpbase"]

    yellow = df[df["color_class"] == "yellow"].copy()
    print(f"[yellow] n = {len(yellow)} with scan + curated QY")
    if len(yellow) < 5:
        sys.exit("not enough yellow FPs to investigate")

    # Compute the expanded wedge / sigma table for each yellow FP.
    rows = []
    for _, r in yellow.iterrows():
        pdb = r["pdb_id"]
        npz = SCAN_DIR / f"scan_{pdb}_free.npz"
        if not npz.is_file():
            continue
        w = compute_wedge_and_sigma(npz)
        w.update({
            "pdb_id": pdb,
            "lit_qy": r["lit_qy"],
            "chromophore_type": r["chromophore_type"],
            "fp_name": r["seq_match_name"] if pd.notna(r.get("seq_match_name")) else r.get("fpbase_name"),
            "qy_provenance": r["qy_provenance"],
        })
        rows.append(w)
    yw = pd.DataFrame(rows)
    yw = yw[yw["n_allowed"] >= 6].copy()
    print(f"[yellow] {len(yw)} with n_allowed >= 6")

    # ----- (1) listing -----
    show = ["pdb_id", "fp_name", "chromophore_type", "lit_qy",
            "n_allowed", "frac_negative_HT_15", "sigma_negative_HT",
            "frac_neg_HT_2009_slope_15", "qy_provenance"]
    print("\n[(1) all yellow FPs sorted by neg-HT wedge fraction]")
    print(yw.sort_values("frac_negative_HT_15", ascending=False)[show].to_string(index=False))
    print(f"\n  unique chromophore_type values: "
          f"{yw['chromophore_type'].value_counts(dropna=False).to_dict()}")
    print(f"  qy_provenance breakdown: "
          f"{yw['qy_provenance'].value_counts(dropna=False).to_dict()}")

    # ----- (3) leave-one-out -----
    base_r, base_p = spearmanr(yw["lit_qy"], yw["frac_negative_HT_15"])
    print(f"\n[(3) baseline Spearman]  rho = {base_r:+.3f}  p = {base_p:.4f}  n = {len(yw)}")
    loo = []
    for i, r in yw.iterrows():
        sub = yw.drop(i)
        r_loo, p_loo = spearmanr(sub["lit_qy"], sub["frac_negative_HT_15"])
        loo.append((r["pdb_id"], r_loo, p_loo))
    loo_df = pd.DataFrame(loo, columns=["dropped_pdb", "rho", "p"])
    print(f"  leave-one-out rho range: [{loo_df['rho'].min():+.3f}, {loo_df['rho'].max():+.3f}]")
    print(f"  drops that flip rho below 0.20 or above 0.55:")
    fragile = loo_df[(loo_df["rho"] < 0.20) | (loo_df["rho"] > 0.55)]
    if len(fragile) == 0:
        print("    (none)")
    else:
        print(fragile.to_string(index=False))

    # ----- (4) bootstrap CI -----
    rng = np.random.default_rng(1)
    boot = []
    n = len(yw)
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        sub = yw.iloc[idx]
        if sub["frac_negative_HT_15"].nunique() < 3 or sub["lit_qy"].nunique() < 3:
            continue
        r, _ = spearmanr(sub["lit_qy"], sub["frac_negative_HT_15"])
        if np.isnan(r):
            continue
        boot.append(r)
    boot = np.array(boot)
    print(f"\n[(4) bootstrap rho 95 % CI from {len(boot)} resamples]")
    print(f"  CI: [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
    print(f"  fraction <= 0: {(boot <= 0).mean():.3f}")

    # ----- (5) vary wedge width -----
    print(f"\n[(5) wedge half-width sensitivity]")
    for h in (10, 15, 20):
        r, p = spearmanr(yw["lit_qy"], yw[f"frac_negative_HT_{h}"])
        print(f"  +-{h:>2d} deg wedge: rho = {r:+.3f}  p = {p:.4f}")

    # ----- (6) 2009 slope axis (-0.857) -----
    print(f"\n[(6) Megley 2009 slope -0.857 axis]")
    for h in (10, 15, 20):
        r, p = spearmanr(yw["lit_qy"], yw[f"frac_neg_HT_2009_slope_{h}"])
        print(f"  +-{h:>2d} deg wedge: rho = {r:+.3f}  p = {p:.4f}")

    # ----- (7) sigma along axes -----
    print(f"\n[(7) projection sigma along axes]")
    for k in ("phi_axis", "positive_HT", "negative_HT", "neg_HT_2009_slope", "tau_axis"):
        r, p = spearmanr(yw["lit_qy"], yw[f"sigma_{k}"])
        print(f"  sigma_{k:<20s}  rho = {r:+.3f}  p = {p:.4f}")

    # ----- (8) within-chromophore-code (chromophore_type) -----
    print(f"\n[(8) within chromophore type]")
    for code, sub in yw.groupby("chromophore_type"):
        if len(sub) < 5:
            print(f"  {code} (n={len(sub)}): too small, skip")
            continue
        r, p = spearmanr(sub["lit_qy"], sub["frac_negative_HT_15"])
        print(f"  {code} (n={len(sub)}): rho = {r:+.3f}  p = {p:.4f}")

    # ----- (9) FDR across all 16 within-class tests -----
    # Pull those numbers from the existing qy_wedge_test.csv.
    qwt = pd.read_csv(DATA_DIR / "qy_wedge_test.csv")
    qwt = qwt[qwt["n_allowed"] >= 6].dropna(subset=["lit_qy"])
    tests = []
    for cls in ("green", "red", "yellow", "cyan"):
        sub = qwt[qwt["color_class"] == cls]
        if len(sub) < 20:
            continue
        for k in ("phi_axis", "positive_HT", "negative_HT", "tau_axis"):
            r, p = spearmanr(sub["lit_qy"], sub[f"frac_{k}"])
            tests.append((cls, k, len(sub), r, p))
    fdr_df = pd.DataFrame(tests, columns=["class", "wedge", "n", "rho", "p"])
    fdr_df = fdr_df.sort_values("p").reset_index(drop=True)
    m = len(fdr_df)
    fdr_df["bh_thresh"] = (fdr_df.index + 1) / m * 0.05
    fdr_df["passes_BH"] = fdr_df["p"] <= fdr_df["bh_thresh"]
    print(f"\n[(9) Benjamini-Hochberg FDR across {m} within-class tests, alpha = 0.05]")
    print(fdr_df.to_string(index=False))
    fdr_df.to_csv(DATA_DIR / "yellow_minus_ht_fdr.csv", index=False)

    # ----- plot: QY vs neg-HT wedge fraction (yellow) -----
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(yw["lit_qy"], yw["frac_negative_HT_15"],
               c="#cc9900", edgecolor="black", s=60, zorder=3)
    # label any point with neg-HT fraction > 0.1 or qy < 0.4
    for _, r in yw.iterrows():
        if r["frac_negative_HT_15"] > 0.10 or r["lit_qy"] < 0.50:
            ax.annotate(
                f"{r['pdb_id']}",
                xy=(r["lit_qy"], r["frac_negative_HT_15"]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8, color="black",
            )
    ax.set_xlabel("FPbase QY (yellow FPs, n = {})".format(len(yw)))
    ax.set_ylabel(r"fraction of allowed cells in $\pm 15^{\circ}$ neg-HT wedge")
    ax.set_title(
        f"Within-yellow: neg-HT wedge fraction vs QY\n"
        f"Spearman rho = {base_r:+.3f}, p = {base_p:.3f}"
    )
    plt.tight_layout()
    out = FIG_DIR / "yellow_minus_ht_diagnostic.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
