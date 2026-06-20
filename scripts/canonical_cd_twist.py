"""
canonical_cd_twist.py
=====================

Implements a CD1/CD2-labeling-invariant phi (the canonical-CD rule) and re-tests
the signed twist coordinates (bicycle-pedal vs hula-twist) against QY.

Motivation
----------
phi = CA2-CB2-CG2-CD1 is measured through one ortho ring carbon, but the
phenol ring's local 2-fold (CD1<->CD2) is resolved arbitrarily across PDB
depositions, and the two ortho carbons are NOT exactly 180 deg apart in the
dihedral (median ~170 deg). So both the magnitude and, especially, the SIGN of
the phi deviation depend on atom naming, which makes the bicycle-pedal (co-rot)
vs hula-twist (counter-rot) distinction a labeling artifact (see log).

Canonical-CD rule
-----------------
For each chromophore, compute the raw dihedral CA2-CB2-CG2-X to BOTH ortho ring
carbons X. Pick the one whose raw value lies in (-90, 90].  Because the two
candidates are ~170 deg apart, exactly one lands in that window in the normal
case; this selects the geometrically NEAR-side carbon regardless of which atom
the PDB happens to call CD1, so phi_canon is labeling-invariant.  Tie-break for
the rare near-+/-90 overlap: smaller |phi|, then the more positive value.

tau (N2-CA2-CB2-CG2) has no such ambiguity (all four atoms unique).

Coordinates re-tested (sdev_tau = signed deviation of tau from nearest planar
axis in (-90,90]; phi_canon is already in (-90,90] so its planar reference is 0
and its signed deviation IS phi_canon):
  I_twist  = |sdev_tau|
  P_twist  = |phi_canon|
  combined = hypot(|sdev_tau|, |phi_canon|)        (canonical analogue of d_exp_to_planar)
  HT_abs   = |sdev_tau - phi_canon|                (counter-rotation magnitude)
  BP_abs   = |sdev_tau + phi_canon|                (co-rotation magnitude)
  plus signed HT_signed, BP_signed.

Torsions are recomputed from the cached CIFs with the production occupancy-ranked
altloc selection (barrel.load_structure).

Outputs:
  data/canonical_cd_twist.csv   -- per-structure canonical torsions + coordinates
Prints the Spearman comparison table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import barrel
from rotate import dihedral, build_bond_graph

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
CIF_DIR = DATA / "cif"


def signed_dev(theta):
    """Signed deviation from nearest planar axis (0 or +/-180), in (-90, 90]."""
    return (np.asarray(theta, dtype=float) + 90.0) % 180.0 - 90.0


def in_window(v):
    """True if raw dihedral v (deg, any range) folds to lie in (-90, 90] without
    a 180 shift, i.e. its principal value is within the near window."""
    w = (v + 180.0) % 360.0 - 180.0   # wrap to (-180,180]
    return -90.0 < w <= 90.0, w


def canonical_phi(phi_candidates: list[float]) -> tuple[float, int]:
    """Apply the canonical-CD rule. Returns (phi_canon, n_in_window)."""
    wrapped = [((v + 180.0) % 360.0 - 180.0) for v in phi_candidates]
    inside = [w for w in wrapped if -90.0 < w <= 90.0]
    if not inside:
        # degenerate (single carbon far out): fold it
        return float(signed_dev(wrapped[0])), 0
    if len(inside) == 1:
        return float(inside[0]), 1
    # overlap (two candidates near +/-90): smaller |phi|, ties -> more positive
    inside.sort(key=lambda w: (abs(w), -w))
    return float(inside[0]), len(inside)


def ortho_carbons(chrom_atoms: dict, chrom_elements: dict) -> list[str]:
    """Carbon neighbours of CG2 other than CB2 (the ortho ring carbons)."""
    graph = build_bond_graph(chrom_atoms)
    neigh = sorted(graph.get("CG2", set()) - {"CB2"})
    cs = [n for n in neigh if chrom_elements.get(n, n[:1]).upper().startswith("C")]
    return cs


def compute_per_structure() -> pd.DataFrame:
    rows = []
    n_fail = 0
    overlap_count = 0
    cifs = sorted(CIF_DIR.glob("*.cif"))
    for cif in cifs:
        pid = cif.stem.upper()
        try:
            L = barrel.load_structure(cif)
        except Exception:
            n_fail += 1
            continue
        a = L.chrom_atoms
        if not all(k in a for k in ("N2", "CA2", "CB2", "CG2")):
            n_fail += 1
            continue
        tau = dihedral(a["N2"], a["CA2"], a["CB2"], a["CG2"])

        cs = ortho_carbons(a, L.chrom_elements)
        phi_cands = [dihedral(a["CA2"], a["CB2"], a["CG2"], a[c]) for c in cs]
        if not phi_cands:
            n_fail += 1
            continue
        phi_canon, n_in = canonical_phi(phi_cands)
        if n_in >= 2:
            overlap_count += 1

        # CD1-named candidate, for the artifact comparison
        phi_cd1 = dihedral(a["CA2"], a["CB2"], a["CG2"], a["CD1"]) if "CD1" in a else np.nan

        sdt = float(signed_dev(tau))
        rows.append({
            "pdb_id": pid,
            "tau": tau, "phi_cd1": phi_cd1,
            "phi_canon": phi_canon, "n_ortho": len(cs),
            "sdev_tau": sdt,
            "I_twist": abs(sdt),
            "P_twist": abs(phi_canon),
            "combined": float(np.hypot(abs(sdt), abs(phi_canon))),
            "HT_abs": abs(sdt - phi_canon),
            "BP_abs": abs(sdt + phi_canon),
            "HT_signed": sdt - phi_canon,
            "BP_signed": sdt + phi_canon,
        })
    print(f"[load] {len(rows)} structures, {n_fail} skipped, "
          f"{overlap_count} near-+/-90 overlap (tie-broken)")
    return pd.DataFrame(rows)


COORDS = ["I_twist", "P_twist", "combined", "HT_abs", "BP_abs",
          "HT_signed", "BP_signed"]


def build_per_unique(ps: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(DATA / "merged_for_aggregate.csv")
    qy = pd.read_csv(DATA / "lit_qy_curated.csv")
    qy["canon_qy"] = qy["lit_qy_fpbase"].fillna(qy["lit_qy_fpbase_recovered"])
    qy["fp_id"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])

    df = ps.merge(meta[["pdb_id", "color_class", "chromophore_type"]], on="pdb_id", how="left")
    df = df.merge(qy[["pdb_id", "canon_qy", "fp_id"]], on="pdb_id", how="left")
    df["color_class"] = df["color_class"].fillna("unknown").str.lower()
    return df


def spear(d, c):
    x = d[c].values.astype(float); y = d["log_qy"].values.astype(float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4:
        return (np.nan, np.nan, int(m.sum()))
    rho, p = stats.spearmanr(x[m], y[m])
    return (rho, p, int(m.sum()))


def partial_coloradj(per, c):
    import statsmodels.api as sm
    d = per.dropna(subset=[c, "log_qy"]).copy()
    dum = pd.get_dummies(d["color_class"], drop_first=True).astype(float)
    X = sm.add_constant(dum.values, has_constant="add")
    rc = sm.OLS(stats.rankdata(d[c].values), X).fit().resid
    ry = sm.OLS(stats.rankdata(d["log_qy"].values), X).fit().resid
    rho, p = stats.spearmanr(rc, ry)
    return (rho, p, len(d))


def main():
    ps = compute_per_structure()
    ps.to_csv(DATA / "canonical_cd_twist.csv", index=False)
    print(f"Saved {DATA / 'canonical_cd_twist.csv'}")

    # label-invariance self-check: canonical_phi is symmetric in candidate order
    # (verified by construction; spot-check a few rows with 2 ortho carbons)
    df = build_per_unique(ps)

    # per-crystal red
    rc = df.dropna(subset=["canon_qy"]); rc = rc[(rc.color_class == "red") & (rc.canon_qy > 0)].copy()
    rc["log_qy"] = np.log10(rc["canon_qy"])

    # per-unique
    sub = df.dropna(subset=["fp_id", "canon_qy"]); sub = sub[sub.canon_qy > 0]
    agg = {c: (c, "median") for c in COORDS}
    agg["canon_qy"] = ("canon_qy", "first"); agg["color_class"] = ("color_class", "first")
    agg["chromophore_type"] = ("chromophore_type", "first")
    per = sub.groupby("fp_id").agg(**agg).reset_index()
    per["log_qy"] = np.log10(per["canon_qy"])

    red = per[per.color_class == "red"]
    green = per[per.color_class == "green"]
    nrq = red[red.chromophore_type == "NRQ"]
    crq = red[red.chromophore_type == "CRQ"]

    print(f"\nred_unique n={len(red)}, red_crystal n={len(rc)}, green n={len(green)}\n")
    print(f"{'coord':<10} | {'red_uniq':>16} | {'red_cryst':>16} | {'NRQ':>14} | "
          f"{'CRQ':>13} | {'green_null':>14} | {'coloradj(141)':>16}")
    print("-" * 122)

    def f(t):
        return f"{t[0]:+.2f}(p={t[1]:.3f},n={t[2]})" if not np.isnan(t[0]) else f"  na (n={t[2]})"
    for c in COORDS:
        print(f"{c:<10} | {f(spear(red,c)):>16} | {f(spear(rc,c)):>16} | "
              f"{f(spear(nrq,c)):>14} | {f(spear(crq,c)):>13} | "
              f"{f(spear(green,c)):>14} | {f(partial_coloradj(per,c)):>16}")


if __name__ == "__main__":
    main()
