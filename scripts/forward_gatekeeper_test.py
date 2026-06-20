"""
forward_gatekeeper_test.py
==========================

Forward (causal) test of the position-203 gatekeeper claim. In green Thr203
scaffolds we replace residue 203 in silico with Tyr (the YFP T203Y change) and
His (the PA-GFP T203H change) and recompute the P-bond accessible fraction
(f_allowed_P_folded). The gatekeeper claim predicts the P-bond cage TIGHTENS
(f_allowed_P drops) when a bulkier side chain is placed at 203.

Mutagenesis: a real Tyr203 (1YFP) and His203 (3GJ1, PA-GFP) side-chain rotamer
is grafted onto each green backbone by Kabsch superposition of the residue-203
backbone (N, CA, C); the donor side chain (CB onward) is transplanted in its
native conformation. This uses experimentally observed rotamers rather than an
ab-initio rotamer search. The chromophore (residue 66) is untouched, so the
only change to the steric cage is at position 203.

Output: data/forward_gatekeeper_203.csv  (per scaffold: f_allowed_P_folded for
Thr203 / Tyr203 / His203) and a printed summary.
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import load_structure, build_cage
from rotate import measure_megley
from scan_1d import _scan_1d_phi

CIF = Path(__file__).resolve().parent.parent / "data" / "cif"
DONOR = {"TYR": ("1YFP", ["CB","CG","CD1","CD2","CE1","CE2","CZ","OH"]),
         "HIS": ("3GJ1", ["CB","CG","ND1","CD2","CE1","NE2"])}
BACKBONE = ["N","CA","C"]


def res203_atoms(pid, names=None):
    s = gemmi.read_structure(str(CIF / f"{pid.upper()}.cif"))
    for ch in s[0]:
        for r in ch:
            if r.seqid.num == 203:
                d = {a.name: np.array([a.pos.x,a.pos.y,a.pos.z]) for a in r
                     if not a.element.is_hydrogen}
                return r.name, d
    return None, None


def kabsch(D, A):
    """Rigid transform mapping donor points D onto acceptor points A."""
    Dc, Ac = D.mean(0), A.mean(0)
    H = (D-Dc).T @ (A-Ac)
    U,_,Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T
    return R, Ac, Dc   # x_new = R @ (x - Dc) + Ac


def make_mutant_cif(green_pid, target, donor_coords, out_path):
    """Graft donor side chain onto green residue 203; write mutant CIF."""
    name, gd = res203_atoms(green_pid)
    A = np.array([gd[n] for n in BACKBONE])
    D = np.array([donor_coords[n] for n in BACKBONE])
    R, Ac, Dc = kabsch(D, A)
    sidechain = DONOR[target][1]
    placed = {n: R @ (donor_coords[n]-Dc) + Ac for n in sidechain}

    s = gemmi.read_structure(str(CIF / f"{green_pid.upper()}.cif"))
    done = False
    for ch in s[0]:
        for r in ch:
            if r.seqid.num == 203 and not done:
                keep = {"N","CA","C","O"}
                idx = [i for i,a in enumerate(r) if a.name not in keep]
                for i in reversed(idx):
                    del r[i]
                for nm in sidechain:
                    a = gemmi.Atom(); a.name = nm
                    a.element = gemmi.Element(nm[0]); a.occ = 1.0; a.b_iso = 30.0
                    p = placed[nm]; a.pos = gemmi.Position(float(p[0]),float(p[1]),float(p[2]))
                    r.add_atom(a)
                r.name = target
                done = True
    s.setup_entities()
    s.make_mmcif_document().write_file(str(out_path))


def f_allowed_P(cif_path):
    loaded = load_structure(Path(cif_path))
    cage = build_cage(loaded)
    tau_exp, _ = measure_megley(loaded.chrom_atoms)
    _, _, _, ffP = _scan_1d_phi(loaded.chrom_atoms, cage,
                                tau_fixed=tau_exp, phi_symmetric=loaded.phi_symmetric)
    return ffP


def main(greens):
    donors = {t: res203_atoms(pid)[1] for t,(pid,_) in DONOR.items()}
    print(f"{'scaffold':9s} {'Thr203':>8s} {'Tyr203':>8s} {'His203':>8s}   tighten?")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for pid in greens:
            try:
                wt = f_allowed_P(CIF / f"{pid}.cif")
                muts = {}
                for t in ("TYR","HIS"):
                    mp = Path(td)/f"{pid}_{t}.cif"
                    make_mutant_cif(pid, t, donors[t], mp)
                    muts[t] = f_allowed_P(mp)
            except Exception as e:
                print(f"{pid:9s}  FAIL: {e}"); continue
            tighten = "yes" if (muts["TYR"]<wt and muts["HIS"]<wt) else "partial" if (muts["TYR"]<wt or muts["HIS"]<wt) else "NO"
            print(f"{pid:9s} {wt:8.4f} {muts['TYR']:8.4f} {muts['HIS']:8.4f}   {tighten}")
            rows.append((pid, wt, muts["TYR"], muts["HIS"]))
    if rows:
        import statistics as st
        wt=[r[1] for r in rows]; ty=[r[2] for r in rows]; hi=[r[3] for r in rows]
        print(f"\nmedian    {st.median(wt):8.4f} {st.median(ty):8.4f} {st.median(hi):8.4f}  (n={len(rows)})")
        with open(CIF.parent/"forward_gatekeeper_203.csv","w") as fh:
            fh.write("pdb_id,f_allowed_P_Thr203,f_allowed_P_Tyr203,f_allowed_P_His203\n")
            for r in rows: fh.write(f"{r[0]},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f}\n")
        print(f"wrote {CIF.parent/'forward_gatekeeper_203.csv'}")


if __name__ == "__main__":
    g = sys.argv[1:] or ["1EMA"]
    main([x.upper() for x in g])
