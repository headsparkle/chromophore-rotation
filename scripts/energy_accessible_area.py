"""
energy_accessible_area.py
=========================

Boltzmann-weighted "energy-accessible" torsional area, computed by reweighting
the already-stored rigid-scan grids (data/scans/scan_<pdb>_free.npz). Instead of
the pure-steric upper bound f_allowed = (clash-free cells)/(all cells), we weight
each (tau, phi) cell by the chromophore's own intrinsic torsional Boltzmann
factor and ask what fraction of that thermally populated ensemble is sterically
allowed by the barrel:

    f_E = sum_allowed exp(-V(tau,phi)/kT)  /  sum_all exp(-V(tau,phi)/kT)

with a single generic HBI torsional potential (separable, two-fold periodic):

    V(tau,phi) = 0.5*B_tau*(1 - cos 2 tau) + 0.5*B_phi*(1 - cos 2 phi)

B_tau (I-bond, methine C=C: high barrier) and B_phi (P-bond, phenol C-C single
bond: low barrier) are swept over plausible ranges because the exact QM HBI
barriers are an input to be fixed from the literature, not invented here. This
script reports f_E and its within-red Spearman correlation with QY for each
barrier set, against the pure-steric f_allowed and the d_exp_to_planar baselines.
"""
from __future__ import annotations
import csv
from statistics import median
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DATA = Path("data"); SCANS = DATA / "scans"
KT = 0.593  # kcal/mol at ~298 K
TOL = 0.4


def load_meta():
    qy, slug = {}, {}
    for r in csv.DictReader(open(DATA/"lit_qy_curated.csv")):
        q = r["lit_qy_fpbase"] or r["lit_qy_fpbase_recovered"]
        try: q = float(q)
        except Exception: q = None
        qy[r["pdb_id"].upper()] = q; slug[r["pdb_id"].upper()] = r["fpbase_slug"] or r["seq_match_slug"]
    meta = {r["pdb_id"].upper(): r for r in csv.DictReader(open(DATA/"merged_for_aggregate.csv"))}
    dexp = {r["pdb_id"].upper(): float(r["d_canonical"]) for r in csv.DictReader(open(DATA/"d_exp_canonical.csv"))}
    return qy, slug, meta, dexp


def f_E(pid, B_tau, B_phi):
    z = np.load(SCANS/f"scan_{pid}_free.npz")
    ov = z["overlap_map"]; tau = np.radians(z["tau_grid"]); phi = np.radians(z["phi_grid"])
    T, P = np.meshgrid(tau, phi, indexing="ij")
    V = 0.5*B_tau*(1-np.cos(2*T)) + 0.5*B_phi*(1-np.cos(2*P))
    w = np.exp(-V/KT)
    allowed = ov <= TOL
    return float((w[allowed].sum()) / w.sum())


def f_allowed(pid):
    z = np.load(SCANS/f"scan_{pid}_free.npz")
    return float((z["overlap_map"] <= TOL).mean())


def main():
    qy, slug, meta, dexp = load_meta()
    red = [p for p, m in meta.items()
           if (m.get("color_class") or "").lower() == "red"
           and (qy.get(p) or 0) > 0 and (SCANS/f"scan_{p}_free.npz").is_file()]

    def per_unique_corr(valfn):
        byfp = defaultdict(list)
        for p in red:
            byfp[slug.get(p) or p].append((valfn(p), qy[p]))
        xs = [median([a for a, _ in v]) for v in byfp.values()]
        ys = [median([b for _, b in v]) for v in byfp.values()]
        return spearmanr(xs, ys), len(xs)

    (r_fa, p_fa), n = per_unique_corr(f_allowed)
    (r_de, p_de), _ = per_unique_corr(lambda p: dexp[p])
    print(f"red per unique FP (n={n}):")
    print(f"  baseline  f_allowed (pure steric) vs QY : rho={r_fa:+.2f} p={p_fa:.3f}")
    print(f"  baseline  d_exp_to_planar         vs QY : rho={r_de:+.2f} p={p_de:.3f}")
    print(f"\n  energy-accessible f_E vs QY, by HBI barrier set (kcal/mol):")
    print(f"  {'B_tau':>6} {'B_phi':>6}   {'rho(f_E,QY)':>12}  {'p':>7}   {'median f_E (red)':>16}")
    for B_tau in (15.0, 30.0):
        for B_phi in (2.0, 4.0, 6.0, 10.0):
            (r, pv), _ = per_unique_corr(lambda p: f_E(p, B_tau, B_phi))
            med = median([f_E(p, B_tau, B_phi) for p in red])
            print(f"  {B_tau:6.0f} {B_phi:6.0f}   {r:+12.2f}  {pv:7.3f}   {med:16.3f}")


if __name__ == "__main__":
    main()
