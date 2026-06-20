"""
forward_gatekeeper_PI.py
========================

Phi-vs-tau differential of the position-203 forward mutation test. The absolute
direction of the f_allowed_P change under the in-silico Thr203->Tyr/His graft is
unreliable (see forward_gatekeeper_test.py: a fixed grafted rotamer removes the
Thr203-OG1 wall and does not reliably re-create the YFP pi-stacking wall). But
the SPECIFICITY question is robust to that: does mutating 203 perturb the P-bond
(phi) cage more than the I-bond (tau) cage? The gatekeeper claim predicts yes.

For each green Thr203 scaffold (one per unique FP) we compute f_allowed_I (tau
scan at phi_exp) and f_allowed_P_folded (phi scan at tau_exp) for the wild type
and for the Tyr203/His203 grafts, and compare |Delta f_allowed_P| with
|Delta f_allowed_I|.

Output: data/forward_gatekeeper_PI.csv ; prints the summary.
"""
from __future__ import annotations
import sys, csv, tempfile
from pathlib import Path
import numpy as np
import gemmi
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forward_gatekeeper_test as F
from barrel import load_structure, build_cage
from rotate import measure_megley
from scan_1d import _scan_1d_tau, _scan_1d_phi

CIF = Path(__file__).resolve().parent.parent / "data" / "cif"


def scans(cif_path):
    L = load_structure(Path(cif_path)); cage = build_cage(L)
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
            s = gemmi.read_structure(str(CIF/f"{pid}.cif")); r203 = None
            for ch in s[0]:
                for r in ch:
                    if r.seqid.num == 203:
                        r203 = r.name; break
                if r203:
                    break
        except Exception:
            continue
        if r203 != "THR":
            continue
        k = slug.get(pid) or pid
        if k in seen:
            continue
        seen.add(k); picks.append(pid)
    return picks


def main():
    picks = green_thr203()
    donors = {t: F.res203_atoms(pid)[1] for t, (pid, _) in F.DONOR.items()}
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for pid in picks:
            try:
                wI, wP = scans(CIF/f"{pid}.cif")
                rec = {"pdb_id": pid, "I_Thr": wI, "P_Thr": wP}
                for t in ("TYR", "HIS"):
                    mp = Path(td)/f"{pid}_{t}.cif"
                    F.make_mutant_cif(pid, t, donors[t], mp)
                    mI, mP = scans(mp)
                    rec[f"I_{t}"] = mI; rec[f"P_{t}"] = mP
                rows.append(rec)
            except Exception:
                pass
    with open(CIF.parent/"forward_gatekeeper_PI.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id","I_Thr","P_Thr","I_TYR","P_TYR","I_HIS","P_HIS"])
        w.writeheader(); w.writerows(rows)
    print(f"n scaffolds = {len(rows)}\n")
    for t in ("TYR", "HIS"):
        dI = np.array([r[f"I_{t}"]-r["I_Thr"] for r in rows])
        dP = np.array([r[f"P_{t}"]-r["P_Thr"] for r in rows])
        ng = int(np.sum(np.abs(dP) > np.abs(dI)))
        wstat, p = wilcoxon(np.abs(dP), np.abs(dI), alternative="greater")
        print(f"Thr203->{t}: mean|dI(tau)|={np.mean(np.abs(dI)):.4f}  "
              f"mean|dP(phi)|={np.mean(np.abs(dP)):.4f}  "
              f"|dP|>|dI| in {ng}/{len(rows)}  Wilcoxon p={p:.2e}")


if __name__ == "__main__":
    main()
