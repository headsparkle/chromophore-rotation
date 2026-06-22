"""
leverage_control.py
===================

Family-wide control for the "I-bond is clamped" claim. Tests whether the
tau (I-bond) vs phi (P-bond) accessibility asymmetry is intrinsic to the
chromophore (lever-arm / swept-volume geometry) or imposed by the barrel.

For each of the 838 structures:
  1. Lever arms: mean/max perpendicular distance of the tau-moving atoms to
     the CA2-CB2 axis and of the phi-moving atoms to the CB2-CG2 axis.
     tau rotation swings the whole phenol on a long arm; phi spins the ring
     about ~its own long axis.
  2. NO-BARREL 1D scans (chromophore self-clash only): the paper's exact
     f_allowed_I and f_allowed_P_folded metrics (scan_1d definitions) but with
     the barrel removed. This is the intrinsic-geometry baseline.
Full-barrel f_allowed_I / f_allowed_P_folded are read from SD1 for comparison.

Output: data/leverage_control.csv
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from barrel import load_structure, build_cage
from rotate import build_bond_graph, side_reachable_from, measure_megley, is_phi_symmetric
from scan_1d import _scan_1d_tau, _scan_1d_phi

CIF = Path("../data/cif")
SD1 = Path("../data/supplementary_SD1_per_structure.csv")


def perp(axis_p, axis_dir, x):
    d = axis_dir / np.linalg.norm(axis_dir)
    v = x - axis_p
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def lever_arms(A, g):
    tau_m = sorted(side_reachable_from(g, "CB2", "CA2"))
    phi_m = sorted(side_reachable_from(g, "CG2", "CB2"))
    tp, td = A["CA2"], A["CB2"] - A["CA2"]
    pp, pd_ = A["CB2"], A["CG2"] - A["CB2"]
    tr = [perp(tp, td, A[n]) for n in tau_m]
    pr = [perp(pp, pd_, A[n]) for n in phi_m]
    return (np.mean(tr), np.max(tr), sum(tr),
            np.mean(pr), np.max(pr), sum(pr))


def no_barrel(L):
    return dataclasses.replace(
        L, extra_xyz=np.empty((0, 3)),
        extra_elements=np.empty((0,), dtype="<U2"))


def main():
    sd1 = pd.read_csv(SD1)
    rows = []
    fails = []
    for _, r in sd1.iterrows():
        pid = r["pdb_id"]
        cif = CIF / f"{pid}.cif"
        if not cif.exists():
            fails.append((pid, "no_cif")); continue
        try:
            L = load_structure(cif)
            A, g = L.chrom_atoms, build_bond_graph(L.chrom_atoms)
            tlm, tlx, tls, plm, plx, pls = lever_arms(A, g)
            tau_exp, phi_exp = measure_megley(A)
            sym = is_phi_symmetric(A)
            cage_nb = build_cage(no_barrel(L))
            _, fI_nb = _scan_1d_tau(A, cage_nb, phi_fixed=phi_exp)
            _, _, _, fP_nb = _scan_1d_phi(A, cage_nb, tau_fixed=tau_exp,
                                          phi_symmetric=sym)
            rows.append(dict(
                pdb_id=pid, color_class=r.get("color_class"),
                tau_lever_mean=tlm, tau_lever_max=tlx,
                phi_lever_mean=plm, phi_lever_max=plx,
                swept_ratio=tls / pls if pls else np.nan,
                fI_nobarrel=fI_nb, fP_folded_nobarrel=fP_nb,
                fI_full=r.get("f_allowed_I"),
                fP_folded_full=r.get("f_allowed_P_folded"),
            ))
        except Exception as e:  # noqa
            fails.append((pid, type(e).__name__))

    df = pd.DataFrame(rows)
    df.to_csv("../data/leverage_control.csv", index=False)
    print(f"Computed {len(df)} structures; {len(fails)} failures")
    if fails:
        from collections import Counter
        print("  fails:", dict(Counter(x for _, x in fails)))

    med = df.median(numeric_only=True)
    print("\n=== LEVER ARMS (family medians, A) ===")
    print(f"  tau-movers mean lever: {med.tau_lever_mean:.2f}  (max {med.tau_lever_max:.2f})")
    print(f"  phi-movers mean lever: {med.phi_lever_mean:.2f}  (max {med.phi_lever_max:.2f})")
    print(f"  swept-volume ratio tau/phi: {med.swept_ratio:.1f}x")

    print("\n=== ACCESSIBILITY: no-barrel (intrinsic) vs full-barrel ===")
    print(f"  I-bond (tau)  f_allowed:  no-barrel median {df.fI_nobarrel.median():.3f}"
          f"  |  full-barrel median {df.fI_full.median():.3f}")
    print(f"  P-bond (phi)  f_allowed:  no-barrel median {df.fP_folded_nobarrel.median():.3f}"
          f"  |  full-barrel median {df.fP_folded_full.median():.3f}")
    print(f"  fraction with tau FULLY open (fI_nobarrel==1.0): "
          f"{(df.fI_nobarrel>=0.999).mean()*100:.0f}%")
    print(f"  fraction where no-barrel tau >= no-barrel phi: "
          f"{(df.fI_nobarrel>=df.fP_folded_nobarrel).mean()*100:.0f}%")

    print("\n=== tau/phi accessibility RATIO: barrel vs lever-arm prediction ===")
    sub = df[(df.fP_folded_full > 0) & (df.fI_full >= 0)].copy()
    sub["barrel_ratio"] = sub.fP_folded_full / sub.fI_full.replace(0, np.nan)
    print(f"  median full-barrel phi/tau accessibility ratio: "
          f"{sub.barrel_ratio.median():.1f}x  (lever-arm swept ratio ~{med.swept_ratio:.1f}x)")
    print("\n  per color class (median full-barrel f_allowed):")
    for c, gg in df.groupby("color_class"):
        print(f"    {c:8} n={len(gg):3d}  tau_full={gg.fI_full.median():.3f}"
              f"  phi_full={gg.fP_folded_full.median():.3f}"
              f"  tau_nobarrel={gg.fI_nobarrel.median():.2f}")


if __name__ == "__main__":
    main()
