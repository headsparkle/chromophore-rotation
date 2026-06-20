#!/usr/bin/env python3
"""Recompute d_exp_to_planar with the canonical-CD rule and write the
authoritative data/d_exp_canonical.csv.

Cleanup applied
---------------
phi is the dihedral CA2-CB2-CG2-CD1, but the phenol ring's local 2-fold means
the deposited CD1 atom can be either ortho carbon, and the two are ~170 deg
apart (not exactly 180), so the deposited CD1 choice puts a ~10 deg labeling
noise into dev_phi (and hence d_exp_to_planar). The canonical-CD rule removes
it: compute phi to BOTH ortho carbons (occupancy-ranked atoms from
barrel.load_structure) and keep the one whose raw dihedral lies in (-90, 90] --
the geometrically near-side carbon, independent of atom naming. tau
(N2-CA2-CB2-CG2) is unambiguous.

This is the production cleanup of d_exp_to_planar (the deposited-geometry metric
only). It does NOT touch the rotational scan / f_allowed, which is invariant to a
uniform phi-reference shift. Mirrors recompute_altloc_fixed.py in spirit (a
targeted deposited-geometry recompute; the full scan is not rerun).

Output columns: pdb_id, tau, phi_canon, d_canonical, d_cd1_occ, cd1_was_far
  d_cd1_occ  = d_exp_to_planar from the occupancy-ranked CD1 (current production)
  cd1_was_far= True if the canonical near-side carbon is NOT the deposited CD1
Run from the project root.
"""
import csv
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "scripts")
from barrel import load_structure
from rotate import dihedral, build_bond_graph

CIF = Path("data/cif")
PLANAR = [(0, 0), (0, 180), (180, 0), (180, 180)]


def wrap(x):
    return ((x + 180.0) % 360.0) - 180.0


def d_planar(tau, phi):
    return min(math.hypot(wrap(tau - rt), wrap(phi - rp)) for rt, rp in PLANAR)


def canonical_phi(phi_candidates):
    """Near-side ortho carbon: the candidate whose wrapped raw value is in
    (-90, 90]. Tie (near +/-90): smaller |phi|, then more positive."""
    wrapped = [wrap(v) for v in phi_candidates]
    inside = [w for w in wrapped if -90.0 < w <= 90.0]
    if not inside:                         # single far-out carbon: fold it
        w = wrapped[0]
        return wrap(w - 180.0) if not (-90.0 < w <= 90.0) else w
    inside.sort(key=lambda w: (abs(w), -w))
    return inside[0]


def main():
    rows = []
    fails = 0
    n_flip = 0
    for cif in sorted(CIF.glob("*.cif")):
        pid = cif.stem.upper()
        try:
            L = load_structure(cif)
            a = L.chrom_atoms
            tau = dihedral(a["N2"], a["CA2"], a["CB2"], a["CG2"])
            graph = build_bond_graph(a)
            neigh = sorted(graph.get("CG2", set()) - {"CB2"})
            cs = [n for n in neigh
                  if L.chrom_elements.get(n, n[:1]).upper().startswith("C")]
            phi_cands = [dihedral(a["CA2"], a["CB2"], a["CG2"], a[c]) for c in cs]
            if not phi_cands:
                fails += 1
                continue
            phi_canon = canonical_phi(phi_cands)
            phi_cd1 = dihedral(a["CA2"], a["CB2"], a["CG2"], a["CD1"]) if "CD1" in a else float("nan")
        except Exception:
            fails += 1
            continue
        d_can = d_planar(tau, phi_canon)
        d_cd1 = d_planar(tau, phi_cd1) if not math.isnan(phi_cd1) else d_can
        far = (not math.isnan(phi_cd1)) and abs(wrap(phi_canon - wrap(phi_cd1))) > 1.0
        if far:
            n_flip += 1
        rows.append(dict(pdb_id=pid, tau=tau, phi_canon=phi_canon,
                         d_canonical=d_can, d_cd1_occ=d_cd1,
                         cd1_was_far=int(far)))

    out = Path("data/d_exp_canonical.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    deltas = [abs(r["d_canonical"] - r["d_cd1_occ"]) for r in rows]
    print(f"processed {len(rows)} CIFs ({fails} skipped)")
    print(f"CD1 was the far-side ortho carbon in {n_flip}/{len(rows)} "
          f"({100*n_flip/len(rows):.0f}%)")
    print(f"d_exp changed >0.05 deg: {sum(d > 0.05 for d in deltas)}  "
          f"max {max(deltas):.2f}  median {np.median(deltas):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
