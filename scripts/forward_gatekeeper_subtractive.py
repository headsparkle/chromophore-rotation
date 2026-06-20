"""
forward_gatekeeper_subtractive.py
=================================

Clean forward (causal) test of the position-203 gatekeeper claim, using
SUBTRACTIVE mutations that need no rotamer placement or minimization (the
additive Thr->Tyr/His graft failed its control because a rigidly placed bulky
side chain clashes nonspecifically; see log). Here we simply DELETE atoms:

  203 -> Ala : remove the whole Thr203 side chain (OG1, CG2), keep N/CA/C/O/CB
  203 -> Ser : remove only the CG2 methyl (keep the gamma-OH = Ser OG)
  167 -> Ala : NEGATIVE CONTROL, remove the Ile167 side chain (CG1/CG2/CD1)

Prediction: removing the 203 wall OPENS the P-bond (phi) cage (f_allowed_P up),
and does so far more than removing a non-gatekeeper side chain (Ile167). Removing
only CG2 (Ser) should barely change it, because Thr203-OG1 is the dominant wall
atom. The chromophore (residue 66) is never touched; only the cage changes, and
only by atom removal, so no relaxation is required.

Output: data/forward_gatekeeper_subtractive.csv ; prints summary + control test.
"""
from __future__ import annotations
import sys, csv, tempfile
from pathlib import Path
import numpy as np
import gemmi
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import load_structure, build_cage
from rotate import measure_megley
from scan_1d import _scan_1d_tau, _scan_1d_phi

CIF = Path(__file__).resolve().parent.parent / "data" / "cif"


def resname_at(s, pos):
    for ch in s[0]:
        for r in ch:
            if r.seqid.num == pos:
                return r.name
    return None


def delete_atoms(pid, pos, names, out):
    s = gemmi.read_structure(str(CIF/f"{pid}.cif"))
    for ch in s[0]:
        for r in ch:
            if r.seqid.num == pos:
                for i in reversed([i for i,a in enumerate(r) if a.name in names]):
                    del r[i]
    s.setup_entities()
    s.make_mmcif_document().write_file(str(out))


def scans(cif):
    L = load_structure(Path(cif)); cage = build_cage(L)
    tau, phi = measure_megley(L.chrom_atoms)
    _, fI = _scan_1d_tau(L.chrom_atoms, cage, phi_fixed=phi)
    _, _, _, fP = _scan_1d_phi(L.chrom_atoms, cage, tau_fixed=tau,
                               phi_symmetric=L.phi_symmetric)
    return fI, fP


def green_thr203():
    slug = {r["pdb_id"].upper(): (r["fpbase_slug"] or r["seq_match_slug"])
            for r in csv.DictReader(open(CIF.parent/"lit_qy_curated.csv"))}
    greens = [r["pdb_id"].upper() for r in csv.DictReader(open(CIF.parent/"merged_for_aggregate.csv"))
              if (r.get("color_class") or "").lower() == "green"]
    seen, picks = set(), []
    for pid in greens:
        if not (CIF/f"{pid}.cif").is_file():
            continue
        try:
            s = gemmi.read_structure(str(CIF/f"{pid}.cif"))
        except Exception:
            continue
        if resname_at(s, 203) != "THR":
            continue
        k = slug.get(pid) or pid
        if k in seen:
            continue
        seen.add(k); picks.append((pid, resname_at(s, 167) == "ILE"))
    return picks


def main():
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for pid, has_ile167 in green_thr203():
            try:
                wI, wP = scans(CIF/f"{pid}.cif")
                pa = Path(td)/f"{pid}_A203.cif"; delete_atoms(pid,203,{"OG1","CG2"},pa); aI,aP = scans(pa)
                ps = Path(td)/f"{pid}_S203.cif"; delete_atoms(pid,203,{"CG2"},ps);       sI,sP = scans(ps)
                rec = {"pdb_id":pid, "dP_Ala203":aP-wP, "dI_Ala203":aI-wI,
                       "dP_Ser203":sP-wP, "dI_Ser203":sI-wI,
                       "dP_Ala167":"", "dI_Ala167":""}
                if has_ile167:
                    pc = Path(td)/f"{pid}_A167.cif"; delete_atoms(pid,167,{"CG1","CG2","CD1"},pc); cI,cP = scans(pc)
                    rec["dP_Ala167"]=cP-wP; rec["dI_Ala167"]=cI-wI
                rows.append(rec)
            except Exception:
                pass
    with open(CIF.parent/"forward_gatekeeper_subtractive.csv","w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id","dP_Ala203","dI_Ala203","dP_Ser203","dI_Ser203","dP_Ala167","dI_Ala167"])
        w.writeheader(); w.writerows(rows)

    def col(name):
        return np.array([r[name] for r in rows if r[name] != ""], dtype=float)
    print(f"green Thr203 scaffolds n={len(rows)}; Ile167 control n={sum(1 for r in rows if r['dP_Ala167']!='')}\n")
    for lbl,c in [("203->Ala (remove OG1+CG2)","Ala203"),
                  ("203->Ser (remove CG2 only)","Ser203"),
                  ("167->Ala  CONTROL        ","Ala167")]:
        dP=col(f"dP_{c}")
        print(f"{lbl}: median dP(phi)={np.median(dP):+.4f}  phi opens (dP>0) in {int((dP>0).sum())}/{len(dP)}")
    p = mannwhitneyu(col("dP_Ala203"), col("dP_Ala167"), alternative="greater").pvalue
    print(f"\nphi-opening 203->Ala > 167->Ala (Mann-Whitney): p={p:.2e}")


if __name__ == "__main__":
    main()
