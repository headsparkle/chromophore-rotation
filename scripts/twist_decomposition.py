"""
twist_decomposition.py
======================

The production QY predictor d_exp_to_planar collapses the two methine-bridge
torsions into one Euclidean distance:

    d_exp_to_planar = hypot( dev_tau , dev_phi )

where dev_x = torus distance from the deposited dihedral to the nearest planar
reference (0 or +/-180 deg). But the biology distinguishes the two bonds:
  tau (N2-CA2-CB2-CG2)  = I-bond, the excited-state cis-trans isomerization axis
  phi (CA2-CB2-CG2-CD1) = P-bond, the ground-state phenol-flip axis

This script asks whether QY correlates more strongly with a decomposed
coordinate than with the combined distance. Candidates tested:

  I_twist   = |dev_tau|                 (absolute tau deviation)
  P_twist   = |dev_phi|                 (absolute phi deviation)
  combined  = hypot(dev_tau, dev_phi)   (= d_exp_to_planar, reproduced)
  HT_signed = sdev_tau - sdev_phi       (signed hula-twist / counter-rotation)
  HT_abs    = |sdev_tau - sdev_phi|
  BP_signed = sdev_tau + sdev_phi       (signed bicycle-pedal / co-rotation)
  BP_abs    = |sdev_tau + sdev_phi|

sdev_x is the SIGNED deviation from the nearest planar axis, in (-90, 90].

For each coordinate we report the Spearman correlation with log10(QY) within
the red class (per unique FP and per crystal structure), the two red
subfamilies (NRQ, CRQ), green as a null, and a color-adjusted partial Spearman
across the full per-unique-FP set.

Deposited torsions are the occupancy-ranked (production) values from
data/d_exp_altloc_fixed.csv (tau_new, phi_new).

Output: data/twist_decomposition.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"


# --------------------------------------------------------------------------
# angle helpers (match recompute_distances.py conventions)
# --------------------------------------------------------------------------

def wrap180(a):
    """Wrap to (-180, 180]."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def signed_dev(theta):
    """Signed deviation from the nearest planar axis (0 or +/-180), in (-90, 90]."""
    # offset to nearest multiple of 180, mapped into (-90, 90]
    return (np.asarray(theta) + 90.0) % 180.0 - 90.0


def abs_dev(theta):
    return np.abs(signed_dev(theta))


# --------------------------------------------------------------------------
# build per-structure table with the corrected torsions + QY
# --------------------------------------------------------------------------

def build() -> pd.DataFrame:
    meta = pd.read_csv(DATA / "merged_for_aggregate.csv")
    fix = pd.read_csv(DATA / "d_exp_altloc_fixed.csv")  # tau_new, phi_new, d_new
    qy = pd.read_csv(DATA / "lit_qy_curated.csv")

    qy["canon_qy"] = qy["lit_qy_fpbase"].fillna(qy["lit_qy_fpbase_recovered"])
    qy["fp_id"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])

    df = meta[["pdb_id", "color_class", "chromophore_type",
               "tau_exp_deg", "phi_exp_deg", "d_exp_to_planar_deg"]].copy()
    df = df.merge(fix[["pdb_id", "tau_new", "phi_new", "d_new"]], on="pdb_id", how="left")
    df = df.merge(qy[["pdb_id", "canon_qy", "fp_id"]], on="pdb_id", how="left")

    # production (occupancy-ranked) torsions
    tau = df["tau_new"].values
    phi = df["phi_new"].values

    df["I_twist"] = abs_dev(tau)              # |dev_tau|, P-bond? no: I-bond
    df["P_twist"] = abs_dev(phi)              # |dev_phi|
    df["combined"] = np.hypot(abs_dev(tau), abs_dev(phi))
    sdt = signed_dev(tau)
    sdp = signed_dev(phi)
    df["HT_signed"] = sdt - sdp
    df["HT_abs"] = np.abs(sdt - sdp)
    df["BP_signed"] = sdt + sdp
    df["BP_abs"] = np.abs(sdt + sdp)

    df["color_class"] = df["color_class"].fillna("unknown").str.lower()
    return df


COORDS = ["I_twist", "P_twist", "combined", "HT_signed", "HT_abs",
          "BP_signed", "BP_abs"]


def aggregate_unique(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["fp_id", "canon_qy"]).copy()
    sub = sub[sub["canon_qy"] > 0]
    agg = {c: (c, "median") for c in COORDS}
    agg["canon_qy"] = ("canon_qy", "first")
    agg["color_class"] = ("color_class", "first")
    agg["chromophore_type"] = ("chromophore_type", "first")
    per = sub.groupby("fp_id").agg(**agg).reset_index()
    per["log_qy"] = np.log10(per["canon_qy"])
    return per


def spear(d: pd.DataFrame, coord: str):
    x = d[coord].values
    y = d["log_qy"].values
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    rho, p = stats.spearmanr(x[m], y[m])
    return rho, p, int(m.sum())


def partial_spear_coloradj(per: pd.DataFrame, coord: str):
    """Color-adjusted partial Spearman: rank-residualize coord and log_qy on
    color-class dummies, then Spearman the residuals."""
    import statsmodels.api as sm
    d = per.dropna(subset=[coord, "log_qy"]).copy()
    dummies = pd.get_dummies(d["color_class"], drop_first=True).astype(float)
    X = sm.add_constant(dummies.values, has_constant="add")
    rc = sm.OLS(stats.rankdata(d[coord].values), X).fit().resid
    ry = sm.OLS(stats.rankdata(d["log_qy"].values), X).fit().resid
    rho, p = stats.spearmanr(rc, ry)
    return rho, p, len(d)


def main():
    df = build()
    per = aggregate_unique(df)

    # sanity: combined reproduces d_exp_to_planar
    chk = df.dropna(subset=["combined", "d_new"])
    err = np.abs(chk["combined"].values - chk["d_new"].values).max()
    print(f"[sanity] max |combined - d_new| = {err:.4g}  (should be ~0)\n")

    def red_unique(d):
        return d[d["color_class"] == "red"]

    rows = []
    groups = {
        "red_unique": red_unique(per),
        "green_unique": per[per["color_class"] == "green"],
        "NRQ_unique": red_unique(per)[red_unique(per)["chromophore_type"] == "NRQ"],
        "CRQ_unique": red_unique(per)[red_unique(per)["chromophore_type"] == "CRQ"],
    }
    # per-crystal red (no aggregation)
    red_crystal = df.dropna(subset=["canon_qy"])
    red_crystal = red_crystal[(red_crystal["color_class"] == "red") &
                              (red_crystal["canon_qy"] > 0)].copy()
    red_crystal["log_qy"] = np.log10(red_crystal["canon_qy"])

    print(f"{'coord':<10} | {'red_uniq':>16} | {'red_cryst':>16} | "
          f"{'NRQ':>14} | {'CRQ':>14} | {'green(null)':>14} | {'coloradj_partial':>18}")
    print("-" * 130)
    for c in COORDS:
        ru = spear(groups["red_unique"], c)
        rc = spear(red_crystal, c)
        nq = spear(groups["NRQ_unique"], c)
        cq = spear(groups["CRQ_unique"], c)
        gn = spear(groups["green_unique"], c)
        pa = partial_spear_coloradj(per, c)
        rows.append({
            "coord": c,
            "red_uniq_rho": ru[0], "red_uniq_p": ru[1], "red_uniq_n": ru[2],
            "red_cryst_rho": rc[0], "red_cryst_p": rc[1], "red_cryst_n": rc[2],
            "NRQ_rho": nq[0], "NRQ_p": nq[1], "NRQ_n": nq[2],
            "CRQ_rho": cq[0], "CRQ_p": cq[1], "CRQ_n": cq[2],
            "green_rho": gn[0], "green_p": gn[1], "green_n": gn[2],
            "coloradj_rho": pa[0], "coloradj_p": pa[1], "coloradj_n": pa[2],
        })

        def fmt(t):
            return f"{t[0]:+.2f}(p={t[1]:.3f},n={t[2]})"
        print(f"{c:<10} | {fmt(ru):>16} | {fmt(rc):>16} | {fmt(nq):>14} | "
              f"{fmt(cq):>14} | {fmt(gn):>14} | {fmt(pa):>18}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "twist_decomposition.csv", index=False)
    print(f"\nSaved {DATA / 'twist_decomposition.csv'}")


if __name__ == "__main__":
    main()
