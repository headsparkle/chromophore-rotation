"""
heteroatom_sensitivity.py
=========================

Sensitivity rescan requested by the manuscript evaluation: the production cage
is "every non-water heavy atom," so bound ions, buffer molecules and
crystallographic ligands are part of the steric cage. The evaluation asks
whether removing them changes the result.

For every red-FP crystal structure we recompute f_allowed_folded with two cages:
  full      : production cage (all non-water, non-chromophore heavy atoms)
  polymer   : drop every NON-POLYMER het atom (gemmi entity_type != Polymer)
              and water; modified polymer residues (e.g. MSE) are kept.

We also report, per structure, how many het atoms were dropped and the minimum
distance from any dropped het atom to any chromophore atom -- a het atom can
only act as a cage wall if it is within ~ (r_vdw_sum) of the swept chromophore,
so large minimum distances mean the cage (and f_allowed) is unaffected.

d_exp_to_planar (the red twist/QY predictor) is pure chromophore geometry and is
NOT a function of the cage, so it is unchanged by definition; this rescan tests
only the cage-size metric and the "cage size uncorrelated with QY within red"
null.

Output: data/heteroatom_sensitivity.csv  (+ console summary)
"""

from __future__ import annotations

import csv
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy import stats

import barrel
from barrel import (
    LoadedStructure, WATER_RESNAMES, build_cage,
    _residue_has_bridge_pattern, _normalise_megley_atoms, _element_symbol,
    _DIHEDRAL_ATOMS,
)
from scan_all import scan_overlap_map, folded_allowed_fraction

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
CIF = DATA / "cif"
STEP = 5.0
TOL = barrel.DEFAULT_TOLERANCE_A


def load_polymer_only(cif_path: Path):
    """Like barrel.load_structure, but extra (cage) atoms are restricted to
    Polymer-entity residues; non-polymer het and water are dropped. Returns
    (LoadedStructure, n_het_dropped, min_dist_het_to_chrom)."""
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    pdb_id = structure.name.upper() or cif_path.stem.upper()

    chrom_atoms, chrom_elements, chrom_occ = {}, {}, {}
    extra_xyz, extra_elements = [], []
    het_xyz = []  # dropped non-polymer het heavy atoms (for distance report)

    model = structure[0]
    chrom_found = False
    chrom_resname = chrom_chain = ""
    chrom_seqid = 0
    for chain in model:
        for residue in chain:
            if residue.name in WATER_RESNAMES:
                continue
            take_as_chrom = (not chrom_found) and _residue_has_bridge_pattern(residue)
            is_polymer = residue.entity_type == gemmi.EntityType.Polymer
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], float)
                elem = _element_symbol(atom)
                if take_as_chrom:
                    if atom.name not in chrom_atoms or atom.occ > chrom_occ[atom.name]:
                        chrom_atoms[atom.name] = xyz
                        chrom_elements[atom.name] = elem
                        chrom_occ[atom.name] = atom.occ
                elif is_polymer:
                    extra_xyz.append(xyz)
                    extra_elements.append(elem)
                else:
                    het_xyz.append(xyz)
            if take_as_chrom:
                chrom_found = True
                chrom_resname = residue.name
                chrom_chain = chain.name
                chrom_seqid = residue.seqid.num

    if not chrom_found:
        raise RuntimeError(f"no bridge pattern in {cif_path}")
    _normalise_megley_atoms(chrom_atoms, chrom_elements)
    missing = [n for n in _DIHEDRAL_ATOMS if n not in chrom_atoms]
    if missing:
        raise RuntimeError(f"{chrom_resname} missing {missing}")

    loaded = LoadedStructure(
        pdb_id=pdb_id, chrom_resname=chrom_resname, chrom_chain=chrom_chain,
        chrom_seqid=chrom_seqid, chrom_atoms=chrom_atoms,
        chrom_elements=chrom_elements,
        extra_xyz=np.asarray(extra_xyz, float),
        extra_elements=np.asarray(extra_elements),
        phi_symmetric=barrel.is_phi_symmetric(chrom_atoms),
    )

    # min distance from any dropped het atom to any chromophore atom
    if het_xyz:
        c = np.array(list(chrom_atoms.values()), float)
        h = np.array(het_xyz, float)
        d = np.sqrt(((h[:, None, :] - c[None, :, :]) ** 2).sum(-1)).min()
        min_dist = float(d)
    else:
        min_dist = float("nan")
    return loaded, len(het_xyz), min_dist


def f_allowed_full(cif_path: Path) -> float:
    loaded = barrel.load_structure(cif_path)
    cage = build_cage(loaded)
    _, _, ov = scan_overlap_map(loaded, cage, STEP)
    return folded_allowed_fraction(ov, TOL)


def f_allowed_polymer(cif_path: Path):
    loaded, n_het, min_dist = load_polymer_only(cif_path)
    cage = build_cage(loaded)
    _, _, ov = scan_overlap_map(loaded, cage, STEP)
    return folded_allowed_fraction(ov, TOL), n_het, min_dist


def main():
    meta = {r["pdb_id"].upper(): r for r in csv.DictReader(open(DATA / "merged_for_aggregate.csv"))}
    qy = {}
    for r in csv.DictReader(open(DATA / "lit_qy_curated.csv")):
        q = r["lit_qy_fpbase"] or r["lit_qy_fpbase_recovered"]
        try:
            qy[r["pdb_id"].upper()] = float(q)
        except Exception:
            qy[r["pdb_id"].upper()] = None

    red = [p for p, m in meta.items() if (m.get("color_class") or "").lower() == "red"]
    rows = []
    for i, p in enumerate(sorted(red), 1):
        cif = CIF / f"{p}.cif"
        if not cif.exists():
            cif = CIF / f"{p.lower()}.cif"
        if not cif.exists():
            continue
        try:
            ff = f_allowed_full(cif)
            fp, n_het, min_dist = f_allowed_polymer(cif)
        except Exception as e:
            print(f"  [{i:>2}/{len(red)}] {p} FAIL {e}")
            continue
        rows.append(dict(pdb_id=p, f_full=ff, f_polymer=fp,
                         delta=fp - ff, n_het_dropped=n_het,
                         min_het_to_chrom_A=min_dist, qy=qy.get(p)))
        flag = ""
        if not np.isnan(min_dist) and min_dist < 5.0:
            flag = "  <-- het within 5 A of chromophore"
        print(f"  [{i:>2}/{len(red)}] {p}  f_full={ff:.4f}  f_poly={fp:.4f}  "
              f"d={fp-ff:+.4f}  het_dropped={n_het:<3} min_d={min_dist:.1f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "heteroatom_sensitivity.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(df)
    n_with_het = int((df["n_het_dropped"] > 0).sum())
    close = df[df["min_het_to_chrom_A"] < 5.0]
    changed = df[df["delta"].abs() > 1e-9]
    print(f"red structures scanned        : {n}")
    print(f"  with >=1 non-polymer het    : {n_with_het}")
    print(f"  het within 5 A of chromophore: {len(close)}")
    print(f"  f_allowed_folded changed     : {len(changed)}")
    if len(changed):
        print(f"  max |delta f_allowed|        : {df['delta'].abs().max():.4f}")
        print("  structures that changed:")
        for _, r in changed.sort_values('delta', key=abs, ascending=False).iterrows():
            print(f"    {r['pdb_id']}  f_full={r['f_full']:.4f} -> f_poly={r['f_polymer']:.4f} "
                  f"(d={r['delta']:+.4f}, min_d={r['min_het_to_chrom_A']:.1f} A)")
    # red cage-size/QY null under each cage
    sub = df.dropna(subset=["qy"])
    sub = sub[sub["qy"] > 0]
    if len(sub) >= 4:
        rf = stats.spearmanr(sub["f_full"], np.log10(sub["qy"]))
        rp = stats.spearmanr(sub["f_polymer"], np.log10(sub["qy"]))
        print(f"\n  red f_allowed/QY (per crystal, n={len(sub)}):")
        print(f"    full cage   : rho={rf[0]:+.3f} p={rf[1]:.3f}")
        print(f"    polymer cage: rho={rp[0]:+.3f} p={rp[1]:.3f}")
    print(f"\nSaved {DATA / 'heteroatom_sensitivity.csv'}")


if __name__ == "__main__":
    main()
