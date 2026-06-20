#!/usr/bin/env python3
"""
test_fix_his.py
===============

Standalone trial of the atom-name normalisation that will let us
scan histidine-derived and other non-canonical chromophores.

For each of the representative failed structures (1BFP/IIC,
1EMF/CSH, 1EMK/CCY, 2YE0/SWG), we:

1. Load the CIF directly and pull out the chromophore residue.
2. Apply a normalisation:
     - if CG2 is missing but CG is present, rename CG -> CG2.
     - if CD1 is missing, pick a substitute among the atoms bonded
       to (the new) CG2 that aren't CB2, preferring C over N.
3. Try to measure tau_megley and phi_megley and partition the
   moving / fixed atom sets.
4. Print the result. This is what `barrel.load_structure` will
   eventually do once the in-flight scan finishes and we patch
   the live code.

We do NOT touch `barrel.py` or `rotate.py` here; this is a sketch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotate import build_bond_graph, dihedral, side_reachable_from  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CIF_DIR = DATA_DIR / "cif"


def load_chrom_residue(pdb_id: str) -> tuple[str, dict[str, np.ndarray], dict[str, str]]:
    """Return (resname, atoms, elements) for the first residue with
    the CA2/CB2/N2 bridge pattern."""
    path = CIF_DIR / f"{pdb_id.upper()}.cif"
    s = gemmi.read_structure(str(path))
    for chain in s[0]:
        for res in chain:
            names = {a.name for a in res if not a.element.is_hydrogen}
            if {"CA2", "CB2", "N2"}.issubset(names):
                atoms = {}
                elements = {}
                for a in res:
                    if a.element.is_hydrogen:
                        continue
                    atoms[a.name] = np.array(
                        [a.pos.x, a.pos.y, a.pos.z], dtype=float
                    )
                    elements[a.name] = a.element.name.upper()
                return res.name, atoms, elements
    raise RuntimeError(f"no CA2/CB2/N2 residue in {pdb_id}")


def normalise_megley_atoms(
    atoms: dict[str, np.ndarray], elements: dict[str, str]
) -> tuple[str | None, str | None]:
    """
    In-place rename so the canonical Megley quadruple
        tau = N2 - CA2 - CB2 - CG2
        phi = CA2 - CB2 - CG2 - CD1
    can be evaluated even for non-Tyr-derived chromophores.

    Returns (cg2_source_name, cd1_source_name) for record-keeping.
    """
    cg2_source = None
    cd1_source = None

    # CG2 substitute: the unique CB2 neighbour other than CA2.
    if "CG2" not in atoms:
        graph = build_bond_graph(atoms)
        candidates = sorted(graph["CB2"] - {"CA2"})
        # Drop the imidazolinone atoms in case CB2 has unusual bonding.
        ring_candidates = [c for c in candidates if c not in {"N2", "C2"}]
        if not ring_candidates:
            return None, None
        # Prefer the literal name "CG" if present, else first alphabetic.
        chosen = "CG" if "CG" in ring_candidates else ring_candidates[0]
        atoms["CG2"] = atoms.pop(chosen)
        elements["CG2"] = elements.pop(chosen)
        cg2_source = chosen
    else:
        cg2_source = "CG2"

    # CD1 substitute: pick a CG2 neighbour (other than CB2). Prefer C, then N.
    if "CD1" not in atoms:
        graph = build_bond_graph(atoms)
        ring_neighbours = sorted(graph["CG2"] - {"CB2"})
        if not ring_neighbours:
            return cg2_source, None
        carbons = [n for n in ring_neighbours if n.startswith("C")]
        nitrogens = [n for n in ring_neighbours if n.startswith("N")]
        substitute = (carbons + nitrogens)[0]
        atoms["CD1"] = atoms.pop(substitute)
        elements["CD1"] = elements.pop(substitute)
        cd1_source = substitute
    else:
        cd1_source = "CD1"

    return cg2_source, cd1_source


def try_one(pdb_id: str) -> None:
    print(f"\n=== {pdb_id} ===")
    try:
        resname, atoms, elements = load_chrom_residue(pdb_id)
    except RuntimeError as e:
        print(f"  could not even find the bridge pattern: {e}")
        return

    print(f"  chromophore residue: {resname}, {len(atoms)} heavy atoms")
    print(f"  atom names: {sorted(atoms)}")

    cg2_src, cd1_src = normalise_megley_atoms(atoms, elements)
    print(f"  CG2 sourced from: {cg2_src}")
    print(f"  CD1 sourced from: {cd1_src}")

    required = ("N2", "CA2", "CB2", "CG2", "CD1")
    missing = [a for a in required if a not in atoms]
    if missing:
        print(f"  STILL missing after normalisation: {missing}")
        return

    tau = dihedral(atoms["N2"], atoms["CA2"], atoms["CB2"], atoms["CG2"])
    phi = dihedral(atoms["CA2"], atoms["CB2"], atoms["CG2"], atoms["CD1"])
    print(f"  tau_megley = {tau:+.2f}, phi_megley = {phi:+.2f}")

    graph = build_bond_graph(atoms)
    moving = side_reachable_from(graph, start="CB2", blocked="CA2")
    fixed = set(atoms) - moving
    print(f"  moving ({len(moving)}): {sorted(moving)}")
    print(f"  fixed  ({len(fixed)}): {sorted(fixed)}")


def main() -> int:
    for pdb_id in ("1EMA", "1BFP", "1EMF", "1EMK", "2YE0", "1KYP"):
        try_one(pdb_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
