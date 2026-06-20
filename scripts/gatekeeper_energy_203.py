#!/usr/bin/env python3
"""
gatekeeper_energy_203.py
========================

Energetic decomposition of the position-203 / chromophore-phenol interaction
for the green (Thr203) -> yellow (Tyr203) transition. This upgrades the
forward in-silico atom-deletion test (which measures only volume clearance)
into a non-bonded interaction energy, split into:

    E_vdW   : Lennard-Jones (van der Waals) -- the steric wall AND the
              dispersion attraction of a stacked aromatic ring.
    E_elec  : Coulomb (electrostatic) between point charges.

IMPORTANT framing. "pi-stacking" is NOT a separate force-field term. A
parallel aromatic stack stabilises through (i) LJ dispersion (the attractive
r^-6 part of E_vdW) and (ii) the electrostatic quadrupole term (part of
E_elec). So we report E_vdW and E_elec, and SEPARATELY characterise the
ring-ring GEOMETRY (centroid distance, inter-ring angle) to demonstrate that
the favourable interaction really is a parallel stack. The expected (and
correct) physics is that the Thr203 -> Tyr203 "wall -> clamp" transition is
dispersion-dominated.

Design (chosen 2026-06-20): WITHIN-SCAFFOLD in-silico T203Y.
  For every real green Thr203 scaffold we compute E(203 side chain <->
  chromophore phenol ring) as deposited (Thr203), then graft a real 1YFP
  Tyr203 side-chain rotamer onto the SAME backbone (reusing the Kabsch graft
  from forward_gatekeeper_test.py) and recompute. The chromophore is never
  touched, so DeltaDeltaE = E(Tyr203) - E(Thr203) isolates the pure T203Y
  effect. We then VALIDATE on real Tyr203 yellows (as deposited).

Parameters.
  LJ: AMBER parm10/ff14SB vdW by atom type:
        aromatic C (CA)  Rmin/2 = 1.9080 A, eps = 0.0860 kcal/mol
        aliphatic C (CT) Rmin/2 = 1.9080 A, eps = 0.1094 kcal/mol
        hydroxyl O (OH)  Rmin/2 = 1.7210 A, eps = 0.2104 kcal/mol
      combining: Rmin_ij = Rmin/2_i + Rmin/2_j ; eps_ij = sqrt(eps_i eps_j).
  Charges:
    - 203 side chain: full ff14SB heavy-atom-summed charges (ff14sb_charges).
    - Chromophore phenol ring: two TRANSPARENT bracketing models, reported
      side by side because the bright-state protonation is a real modelling
      choice:
        EN (neutral)  : mapped neutral ff14SB Tyr ring charges (net ~0; the
                        ring quadrupole, no monopole).
        EA (anionic)  : same ring with the phenolic proton removed and the
                        residual brought to a net -1 localised on the
                        phenolate O (the simplest fully-specified anion model;
                        conservative -- it concentrates charge at O). This is
                        the bright-state phenolate, consistent with the
                        project's ESP convention.
      The true electrostatics lie between EN and EA; vdW is the rigorous,
      protonation-free primary result.

Outputs:
  data/gatekeeper_energy_203.csv      (per scaffold; grafted T203Y)
  data/gatekeeper_energy_203_real.csv (real Tyr203 yellows, validation)
  figures/gatekeeper_energy_203.png   (paired Thr203 vs Tyr203 E_vdW + geometry)
And a printed summary.

Usage:
  python3 scripts/gatekeeper_energy_203.py            # full cohort + validation
  python3 scripts/gatekeeper_energy_203.py 1EMA 1GFL  # specific green scaffolds
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import load_structure  # noqa: E402
import ff14sb_charges as ff  # noqa: E402
from forward_gatekeeper_test import (  # noqa: E402
    DONOR, make_mutant_cif, res203_atoms,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
CIF = DATA / "cif"
FIG = PROJECT_ROOT / "figures"
FIG.mkdir(exist_ok=True)
AGG = DATA / "merged_for_aggregate.csv"

COUL = 332.0637  # kcal*A / (mol*e^2)

# AMBER parm10 vdW: atom-name -> (Rmin/2 [A], eps [kcal/mol]) per residue role
LJ = {"CA": (1.9080, 0.0860), "CT": (1.9080, 0.1094), "OH": (1.7210, 0.2104)}
RING = ["CG2", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]   # chromophore phenol
RING_C = ["CG2", "CD1", "CD2", "CE1", "CE2", "CZ"]       # carbons only (centroid)

# atom-type assignment for the three relevant residues
TYPE_CHROM = {"CG2": "CA", "CD1": "CA", "CD2": "CA", "CE1": "CA",
              "CE2": "CA", "CZ": "CA", "OH": "OH"}
TYPE_THR = {"CB": "CT", "OG1": "OH", "CG2": "CT"}
TYPE_TYR = {"CB": "CT", "CG": "CA", "CD1": "CA", "CD2": "CA", "CE1": "CA",
            "CE2": "CA", "CZ": "CA", "OH": "OH"}

# neutral ff14SB Tyr ring charges mapped onto chromophore phenol atom names
_tyr_q, _ = ff.resolve_residue_charges("TYR")
CHROM_Q_NEUTRAL = {
    "CG2": _tyr_q["CG"], "CD1": _tyr_q["CD1"], "CD2": _tyr_q["CD2"],
    "CE1": _tyr_q["CE1"], "CE2": _tyr_q["CE2"], "CZ": _tyr_q["CZ"],
    "OH": _tyr_q["OH"],
}
# anionic phenolate: remove phenolic proton (+0.3992 folded into OH), then
# localise the residual on the phenolate O to reach net -1 over the ring.
_HH = 0.3992
def _phenolate(qneutral: dict) -> dict:
    q = dict(qneutral)
    q["OH"] = q["OH"] - _HH                 # strip the folded phenol proton
    deficit = -1.0 - sum(q.values())        # bring net to -1
    q["OH"] = q["OH"] + deficit             # localise residual on O (conservative)
    return q
CHROM_Q_ANION = _phenolate(CHROM_Q_NEUTRAL)


def _ljpair(t1, t2, r):
    r1, e1 = LJ[t1]
    r2, e2 = LJ[t2]
    rmin = r1 + r2
    eps = np.sqrt(e1 * e2)
    a = (rmin / r) ** 6
    return eps * (a * a - 2.0 * a)


def interaction(res_xyz: dict, res_type: dict, res_q: dict,
                chrom_xyz: dict, chrom_q: dict):
    """Return (E_vdW, E_elec) between a 203 side chain and the chromophore
    phenol ring, both in kcal/mol."""
    e_vdw = 0.0
    e_el = 0.0
    for an, ax in res_xyz.items():
        if an not in res_type:
            continue
        for cn in RING:
            if cn not in chrom_xyz:
                continue
            r = float(np.linalg.norm(ax - chrom_xyz[cn]))
            if r < 0.5:
                continue
            e_vdw += _ljpair(res_type[an], TYPE_CHROM[cn], r)
            e_el += COUL * res_q.get(an, 0.0) * chrom_q.get(cn, 0.0) / r
    return e_vdw, e_el


def ring_geometry(res_xyz: dict, ring_atoms, chrom_xyz: dict):
    """centroid distance (A) and inter-ring plane angle (deg, 0 = parallel)
    between a residue's aromatic ring and the chromophore phenol ring."""
    rc = np.array([res_xyz[a] for a in ring_atoms if a in res_xyz])
    cc = np.array([chrom_xyz[a] for a in RING_C if a in chrom_xyz])
    if len(rc) < 3 or len(cc) < 3:
        return np.nan, np.nan

    def normal(pts):
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        return vt[2], c
    n1, c1 = normal(rc)
    n2, c2 = normal(cc)
    dist = float(np.linalg.norm(c1 - c2))
    ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0, 1))))
    return dist, ang


def chrom_ring_xyz(pid):
    L = load_structure(CIF / f"{pid.upper()}.cif")
    if not all(a in L.chrom_atoms for a in RING):
        return None
    return {a: L.chrom_atoms[a] for a in RING}


def sidechain_charges(resn):
    q, _ = ff.resolve_residue_charges(resn)
    return {k: v for k, v in q.items() if k not in ("N", "CA", "C", "O")}


def eval_scaffold(pid):
    """Within-scaffold T203Y. Returns a dict row or None."""
    chrom = chrom_ring_xyz(pid)
    if chrom is None:
        return None
    name, thr = res203_atoms(pid)
    if name != "THR" or thr is None:
        return None
    q_thr = sidechain_charges("THR")
    vdw_thr, _ = interaction(thr, TYPE_THR, q_thr, chrom, {})  # vdw model-free
    _, el_thr_n = interaction(thr, TYPE_THR, q_thr, chrom, CHROM_Q_NEUTRAL)
    _, el_thr_a = interaction(thr, TYPE_THR, q_thr, chrom, CHROM_Q_ANION)

    # graft Tyr203
    donor = res203_atoms(DONOR["TYR"][0])[1]
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / f"{pid}_TYR.cif"
        make_mutant_cif(pid, "TYR", donor, mp)
        s = gemmi.read_structure(str(mp))
        tyr = None
        for ch in s[0]:
            for r in ch:
                # the grafted residue: seqid 203 AND renamed TYR (avoids
                # grabbing a water/ligand that happens to be numbered 203)
                if r.seqid.num == 203 and r.name == "TYR":
                    tyr = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z])
                           for a in r if not a.element.is_hydrogen}
                    break
            if tyr is not None:
                break
    q_tyr = sidechain_charges("TYR")
    vdw_tyr, _ = interaction(tyr, TYPE_TYR, q_tyr, chrom, {})
    _, el_tyr_n = interaction(tyr, TYPE_TYR, q_tyr, chrom, CHROM_Q_NEUTRAL)
    _, el_tyr_a = interaction(tyr, TYPE_TYR, q_tyr, chrom, CHROM_Q_ANION)
    dist, ang = ring_geometry(tyr, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], chrom)

    return {
        "pdb_id": pid.upper(),
        "vdw_Thr203": round(vdw_thr, 3),
        "vdw_Tyr203": round(vdw_tyr, 3),
        "d_vdw_T203Y": round(vdw_tyr - vdw_thr, 3),
        "elec_Thr203_neutral": round(el_thr_n, 3),
        "elec_Tyr203_neutral": round(el_tyr_n, 3),
        "d_elec_T203Y_neutral": round(el_tyr_n - el_thr_n, 3),
        "elec_Thr203_anion": round(el_thr_a, 3),
        "elec_Tyr203_anion": round(el_tyr_a, 3),
        "d_elec_T203Y_anion": round(el_tyr_a - el_thr_a, 3),
        "tyr_stack_dist_A": round(dist, 2),
        "tyr_stack_angle_deg": round(ang, 1),
    }


def eval_real_tyr(pid):
    """Real Tyr203 yellow, as deposited: the 203<->phenol decomposition."""
    chrom = chrom_ring_xyz(pid)
    if chrom is None:
        return None
    name, tyr = res203_atoms(pid)
    if name != "TYR" or tyr is None:
        return None
    q_tyr = sidechain_charges("TYR")
    vdw, _ = interaction(tyr, TYPE_TYR, q_tyr, chrom, {})
    _, el_n = interaction(tyr, TYPE_TYR, q_tyr, chrom, CHROM_Q_NEUTRAL)
    _, el_a = interaction(tyr, TYPE_TYR, q_tyr, chrom, CHROM_Q_ANION)
    dist, ang = ring_geometry(tyr, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], chrom)
    return {
        "pdb_id": pid.upper(),
        "vdw_Tyr203": round(vdw, 3),
        "elec_Tyr203_neutral": round(el_n, 3),
        "elec_Tyr203_anion": round(el_a, 3),
        "tyr_stack_dist_A": round(dist, 2),
        "tyr_stack_angle_deg": round(ang, 1),
    }


def cohort(color, res203):
    out = []
    for r in csv.DictReader(open(AGG)):
        if r.get("color_class") != color:
            continue
        pid = r["pdb_id"].upper()
        if not (CIF / f"{pid}.cif").is_file():
            continue
        nm = res203_atoms(pid)[0]
        if nm == res203:
            out.append(pid)
    return sorted(set(out))


def med(xs):
    xs = sorted(x for x in xs if x == x)
    n = len(xs)
    return float("nan") if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def main(argv):
    greens = [a.upper() for a in argv] or cohort("green", "THR")
    print(f"[gk-energy] {len(greens)} green Thr203 scaffolds (within-scaffold T203Y)")
    rows = []
    for pid in greens:
        try:
            row = eval_scaffold(pid)
        except Exception as e:
            print(f"  {pid}: FAIL {type(e).__name__}: {e}")
            continue
        if row:
            rows.append(row)
    if rows:
        with open(DATA / "gatekeeper_energy_203.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  wrote data/gatekeeper_energy_203.csv  (n={len(rows)})")
        print(f"\n  WITHIN-SCAFFOLD T203Y medians (n={len(rows)}):")
        print(f"    E_vdW  Thr203 = {med([r['vdw_Thr203'] for r in rows]):+.2f}  "
              f"Tyr203 = {med([r['vdw_Tyr203'] for r in rows]):+.2f}  "
              f"dE = {med([r['d_vdw_T203Y'] for r in rows]):+.2f} kcal/mol")
        print(f"    E_elec(anion)  dE = {med([r['d_elec_T203Y_anion'] for r in rows]):+.2f}   "
              f"E_elec(neutral) dE = {med([r['d_elec_T203Y_neutral'] for r in rows]):+.2f} kcal/mol")
        print(f"    Tyr203 stack: dist = {med([r['tyr_stack_dist_A'] for r in rows]):.2f} A  "
              f"angle = {med([r['tyr_stack_angle_deg'] for r in rows]):.1f} deg")

    # validation: real Tyr203 yellows
    yellows = cohort("yellow", "TYR")
    print(f"\n[gk-energy] {len(yellows)} real Tyr203 yellows (validation)")
    yrows = []
    for pid in yellows:
        try:
            row = eval_real_tyr(pid)
        except Exception as e:
            print(f"  {pid}: FAIL {type(e).__name__}: {e}")
            continue
        if row:
            yrows.append(row)
    if yrows:
        with open(DATA / "gatekeeper_energy_203_real.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(yrows[0].keys()))
            w.writeheader(); w.writerows(yrows)
        print(f"  wrote data/gatekeeper_energy_203_real.csv  (n={len(yrows)})")
        print(f"    real Tyr203 E_vdW median = {med([r['vdw_Tyr203'] for r in yrows]):+.2f} kcal/mol  "
              f"stack dist = {med([r['tyr_stack_dist_A'] for r in yrows]):.2f} A  "
              f"angle = {med([r['tyr_stack_angle_deg'] for r in yrows]):.1f} deg")

    if rows:
        make_figure(rows, yrows)
    return 0


def make_figure(rows, yrows):
    """Two panels. Left: distribution of the within-scaffold T203Y vdW change
    (robust to the rigid-graft clash tail). Right: Tyr203/chromophore stacking
    geometry, with real Tyr203 yellows overlaid. A minority of grafts are
    unrelaxed-rotamer artifacts (steric clash -> huge +vdW, or a backbone that
    points the rigid rotamer away -> large centroid distance); the panels are
    clipped to the physical range and the artifact count is annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dvdw = np.array([r["d_vdw_T203Y"] for r in rows])
    tyr = np.array([r["vdw_Tyr203"] for r in rows])
    d = np.array([r["tyr_stack_dist_A"] for r in rows])
    a = np.array([r["tyr_stack_angle_deg"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6))

    # Left: histogram of dE_vdW(T203Y), clipped to [-8, 8]; clash tail annotated
    lo, hi = -8.0, 8.0
    n_clash = int((dvdw > hi).sum())
    clip = np.clip(dvdw, lo, hi)
    ax1.hist(clip, bins=np.linspace(lo, hi, 33), color="#e90", edgecolor="0.3")
    ax1.axvline(0, color="k", lw=0.8, ls=":")
    ax1.axvline(float(np.median(dvdw)), color="#06c", lw=1.6,
                label=f"median = {np.median(dvdw):+.2f}")
    ax1.set_xlabel(r"$\Delta E_{vdW}$ (Tyr203 - Thr203, kcal/mol)")
    ax1.set_ylabel("scaffolds")
    ax1.set_title(f"Within-scaffold T203Y vdW change\n"
                  f"{(dvdw < 0).mean()*100:.0f}% favorable; "
                  f"{n_clash} graft-clash outliers > +8 off-scale")
    ax1.legend(fontsize=8)

    # Right: stacking geometry, physical window only
    win = (d < 8) & (a < 50)
    sc = ax2.scatter(d[win], a[win], c=np.clip(tyr[win], -5, 5),
                     cmap="viridis", s=22, vmin=-5, vmax=5)
    if yrows:
        yd = np.array([r["tyr_stack_dist_A"] for r in yrows])
        ya = np.array([r["tyr_stack_angle_deg"] for r in yrows])
        m = (yd < 8) & (ya < 50)
        ax2.scatter(yd[m], ya[m], facecolors="none", edgecolors="red",
                    s=42, lw=1.3, label="real Tyr203 yellows")
        ax2.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc, ax=ax2, label="Tyr203 E$_{vdW}$ (kcal/mol)")
    ax2.set_xlabel("ring-centroid distance (A)")
    ax2.set_ylabel("inter-ring angle (deg, 0 = parallel)")
    n_off = int((~win).sum())
    ax2.set_title(f"Tyr203 / chromophore stacking geometry\n"
                  f"({n_off} far/failed grafts > 8 A off-scale)")
    fig.tight_layout()
    out = FIG / "gatekeeper_energy_203.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
