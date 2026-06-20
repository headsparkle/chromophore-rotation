#!/usr/bin/env python3
"""
test_1ema.py
============

First sanity-check script for the chromophore-rotation project.

What it does
------------
1. Locates the 1EMA crystal structure (canonical avGFP). Tries, in order:
     - any path given on the command line
     - ~/Downloads/scop_gfp_structures/1EMA.cif
     - cached copy at data/1EMA.cif under the project root
     - downloads from RCSB if none of the above exist
2. Identifies the chromophore residue by 3-letter code (CRO, NRQ, ...).
3. Prints every atom of that residue with its xyz coordinates.
4. Computes a panel of candidate dihedrals (the same ones Luke's
   chromophore_torsions.py considers) and tells us which definitions
   reproduce the values stored in
   gfp-barrel-geometry/data/megley_dihedrals.csv
       1EMA  ->  tau_megley = -17.557, phi_megley = 13.024
5. Prints the matching identification so we lock in the convention
   before we start rotating anything.

This is intentionally a small, single-structure script. Once we know
the Megley convention we will reuse the dihedral function in the
rotation pipeline.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gemmi
except ImportError:
    sys.exit("ERROR: pip3 install gemmi")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MEGLEY_CSV = (
    PROJECT_ROOT.parent / "gfp-barrel-geometry" / "data" / "megley_dihedrals.csv"
)

# All chromophore 3-letter codes used in Luke's pipeline
CHROMOPHORE_RESIDUES = {
    "CRO", "CR2", "GYS", "SYG", "CRQ", "CRF", "CRW", "CRY",
    "NRQ", "NYG", "CH6", "CH7", "CRG", "CRU", "CRV", "CRS",
    "66A", "CR0", "GYC", "LYG", "TYG", "OHD", "SWG", "QYG",
    "CR7", "CR8", "CR9", "CRK", "RC7", "CFY", "PIA", "B2H",
}

# Candidate dihedral definitions to test. The naming follows
# chromophore_torsions.py in the gfp-barrel-geometry repo so we can
# cross-reference. The first one that matches the CSV value (within
# 0.5 deg) becomes our locked-in Megley convention.
CANDIDATE_DIHEDRALS = [
    # (label, atom1, atom2, atom3, atom4)
    ("tau_1   CA2-CB2-CG2-CD1",  "CA2", "CB2", "CG2", "CD1"),
    ("tau_2   CA2-CB2-CG2-CD2",  "CA2", "CB2", "CG2", "CD2"),
    ("tau_3   C1 -CA2-CB2-CG2",  "C1",  "CA2", "CB2", "CG2"),
    ("phi_1   CB2-CG2-CD1-CE1",  "CB2", "CG2", "CD1", "CE1"),
    ("phi_2   CB2-CG2-CD2-CE2",  "CB2", "CG2", "CD2", "CE2"),
    ("psi_1   N2 -CA2-CB2-CG2",  "N2",  "CA2", "CB2", "CG2"),
    ("psi_2   C2 -N2 -CA2-CB2",  "C2",  "N2",  "CA2", "CB2"),
    ("psi_3   O2 -C2 -N2 -CA2",  "O2",  "C2",  "N2",  "CA2"),
    ("chi_1   N2 -C2 -CA2-CB2",  "N2",  "C2",  "CA2", "CB2"),
    ("chi_2   CA2-CG2-CD1-CE1",  "CA2", "CG2", "CD1", "CE1"),
    ("oh_1    CD1-CE1-CZ -OH ",  "CD1", "CE1", "CZ",  "OH"),
]


def dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Signed dihedral angle in degrees, IUPAC convention."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    b2 /= np.linalg.norm(b2)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def find_or_fetch_1ema(cli_path: str | None) -> Path:
    """Return a path to a 1EMA file (cif or pdb), downloading if needed."""
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    candidates += [
        Path.home() / "Downloads" / "scop_gfp_structures" / "1EMA.cif",
        Path.home() / "Downloads" / "scop_gfp_structures" / "1ema.cif",
        DATA_DIR / "1EMA.cif",
        DATA_DIR / "1ema.pdb",
    ]
    for p in candidates:
        if p.is_file():
            print(f"[input] using {p}")
            return p

    # Fall back to RCSB. CIF is the modern default.
    url = "https://files.rcsb.org/download/1EMA.cif"
    out = DATA_DIR / "1EMA.cif"
    print(f"[input] no local copy found, downloading from {url}")
    urllib.request.urlretrieve(url, out)
    print(f"[input] saved to {out}")
    return out


def get_chromophore_atoms(structure_path: Path) -> tuple[str, dict[str, np.ndarray]]:
    """Return (residue 3-letter code, dict of atom name -> xyz)."""
    structure = gemmi.read_structure(str(structure_path))
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name in CHROMOPHORE_RESIDUES:
                    atoms = {
                        atom.name: np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z]
                        )
                        for atom in residue
                    }
                    print(
                        f"[chromophore] chain {chain.name} residue "
                        f"{residue.name} {residue.seqid.num} "
                        f"({len(atoms)} atoms)"
                    )
                    return residue.name, atoms
    raise RuntimeError("no chromophore residue found in structure")


def expected_megley_values(pdb_id: str) -> tuple[float | None, float | None]:
    """Look up the stored Megley values from the geometry-project CSV."""
    if not MEGLEY_CSV.is_file():
        print(f"[reference] {MEGLEY_CSV} not found, skipping cross-check")
        return None, None
    df = pd.read_csv(MEGLEY_CSV)
    row = df[df["pdb_id"].str.upper() == pdb_id.upper()]
    if row.empty:
        return None, None
    return float(row.iloc[0]["tau_megley"]), float(row.iloc[0]["phi_megley"])


def main(argv: list[str]) -> int:
    pdb_id = "1EMA"
    cif_arg = argv[1] if len(argv) > 1 else None

    structure_path = find_or_fetch_1ema(cif_arg)
    chrom_code, atoms = get_chromophore_atoms(structure_path)

    print("\n--- chromophore atoms (name : x, y, z in A) ---")
    for name in sorted(atoms):
        x, y, z = atoms[name]
        print(f"  {name:<5}  {x:>9.3f}  {y:>9.3f}  {z:>9.3f}")

    print("\n--- candidate dihedrals (deg) ---")
    computed: dict[str, float] = {}
    for label, a, b, c, d in CANDIDATE_DIHEDRALS:
        if all(name in atoms for name in (a, b, c, d)):
            ang = dihedral(atoms[a], atoms[b], atoms[c], atoms[d])
            computed[label] = ang
            print(f"  {label}  =  {ang:+8.3f}")
        else:
            missing = [n for n in (a, b, c, d) if n not in atoms]
            print(f"  {label}  =   N/A (missing atoms: {','.join(missing)})")

    tau_ref, phi_ref = expected_megley_values(pdb_id)
    if tau_ref is None:
        print("\n[reference] no CSV entry for 1EMA, cannot cross-check")
        return 0

    print(
        f"\n--- megley_dihedrals.csv expects "
        f"tau_megley = {tau_ref:+.3f}, phi_megley = {phi_ref:+.3f} ---"
    )

    tol = 0.5  # degrees
    tau_hit = [lbl for lbl, v in computed.items() if abs(v - tau_ref) < tol]
    phi_hit = [lbl for lbl, v in computed.items() if abs(v - phi_ref) < tol]

    print("Matches for tau_megley:")
    for lbl in tau_hit:
        print(f"  ✓ {lbl}  ({computed[lbl]:+.3f})")
    if not tau_hit:
        print("  (no match within 0.5 deg — convention may differ)")

    print("Matches for phi_megley:")
    for lbl in phi_hit:
        print(f"  ✓ {lbl}  ({computed[lbl]:+.3f})")
    if not phi_hit:
        print("  (no match within 0.5 deg — convention may differ)")

    if tau_hit and phi_hit:
        print(
            "\nLOCKED-IN MEGLEY CONVENTION:\n"
            f"  tau_megley := {tau_hit[0]}\n"
            f"  phi_megley := {phi_hit[0]}"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
