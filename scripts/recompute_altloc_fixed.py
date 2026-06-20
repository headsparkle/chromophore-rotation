#!/usr/bin/env python3
"""Recompute (tau_exp, phi_exp) and d_exp_to_planar for every cached CIF using
the occupancy-ranked altloc selection now in barrel.load_structure, and compare
with the stored (last-altloc-wins) values. Writes data/d_exp_altloc_fixed.csv.
Run from the project root.
"""
import csv, sys, math
from pathlib import Path
sys.path.insert(0, "scripts")
from barrel import load_structure
from rotate import measure_megley

CIF = Path("data/cif")
PLANAR = [(0, 0), (0, 180), (180, 0), (180, 180)]


def wrap(x):
    return ((x + 180.0) % 360.0) - 180.0


def d_planar(tau, phi):
    return min(math.hypot(wrap(tau - rt), wrap(phi - rp)) for rt, rp in PLANAR)


def main():
    old = {}
    for r in csv.DictReader(open("data/scan_all_summary.csv")):
        try:
            old[r["pdb_id"].upper()] = (float(r["tau_exp_deg"]),
                                        float(r["phi_exp_deg"]),
                                        float(r["d_exp_to_planar_deg"]))
        except (ValueError, KeyError):
            pass

    rows = []
    fails = 0
    for cif in sorted(CIF.glob("*.cif")):
        pid = cif.stem.upper()
        try:
            loaded = load_structure(cif)
            tau, phi = measure_megley(loaded.chrom_atoms)
        except Exception:
            fails += 1
            continue
        d = d_planar(tau, phi)
        o = old.get(pid)
        rows.append(dict(pdb_id=pid, tau_new=tau, phi_new=phi, d_new=d,
                         tau_old=o[0] if o else "", phi_old=o[1] if o else "",
                         d_old=o[2] if o else ""))

    # validation: stored formula vs recomputed-from-old-tau/phi (non-altloc)
    formula_err = []
    changed = []
    for r in rows:
        if r["d_old"] == "":
            continue
        d_from_old = d_planar(r["tau_old"], r["phi_old"])
        formula_err.append(abs(d_from_old - r["d_old"]))
        dd = abs(r["d_new"] - r["d_old"])
        if dd > 0.05:
            changed.append((r["pdb_id"], r["tau_old"], r["tau_new"],
                            r["phi_old"], r["phi_new"], r["d_old"], r["d_new"]))

    with open("data/d_exp_altloc_fixed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"processed {len(rows)} CIFs ({fails} failed to load)")
    print(f"distance-formula max error vs stored (non-altloc check): "
          f"{max(formula_err):.4f}°  (mean {sum(formula_err)/len(formula_err):.4f})")
    print(f"structures whose d_exp_to_planar changed > 0.05°: {len(changed)}")
    for c in sorted(changed, key=lambda x: -abs(x[6]-x[5]))[:15]:
        print(f"  {c[0]}: tau {c[1]:.1f}->{c[2]:.1f}  phi {c[3]:.1f}->{c[4]:.1f}  "
              f"d {c[5]:.1f}->{c[6]:.1f}")


if __name__ == "__main__":
    main()
