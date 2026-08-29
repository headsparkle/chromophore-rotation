"""
barrel.py
=========

Structure loading and steric-clash checking for the
chromophore-rotation pipeline.

What "clash" means here
-----------------------
We use the MolProbity convention. For two heavy atoms i and j with
Bondi van der Waals radii r_i and r_j, the **overlap** is

    overlap_ij = (r_i + r_j) - distance_ij    [Angstrom]

A pair is flagged as clashing when overlap_ij > 0.4 A
(equivalently, distance_ij < r_i + r_j - 0.4 A). For the rotamer
scan we report the **largest** overlap among all moving / cage atom
pairs; a rotamer is "allowed" when that maximum is at or below the
0.4 A tolerance.

What counts as the cage
-----------------------
- Static heavy atoms of the chromophore residue itself (these do
  not move with the scan but the moving phenol can still bump into
  them, especially the imidazolinone ring on a heavy tau twist).
- All heavy atoms of every other residue in the structure, on every
  chain. We do not restrict to the same chain because real
  protein-protein contacts (dimer interfaces, crystal-packing
  neighbours that contact the chromophore) are physical
  constraints.
- Crystallographic waters (HOH, WAT, DOD) are excluded - they are
  not part of the rigid cage and would noisily flag clashes that
  any small movement would relieve.
- Hydrogens are excluded; almost no structure in the dataset has
  them anyway.

The structure-vs-moving check is vectorised in numpy. For a typical
~250-residue FP this is ~8 moving atoms times ~1500 cage atoms per
rotamer, well under a millisecond per rotamer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gemmi

from rotate import (
    build_bond_graph,
    is_phi_symmetric,
    measure_megley,
    set_megley,
    side_reachable_from,
)


# Bondi (1964) van der Waals radii, Angstrom.
# https://doi.org/10.1021/j100785a001 - the standard reference set.
BONDI_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,   # selenomethionine
}

DEFAULT_TOLERANCE_A = 0.4
WATER_RESNAMES = {"HOH", "WAT", "DOD"}

CHROMOPHORE_RESIDUES = {
    "CRO", "CR2", "GYS", "SYG", "CRQ", "CRF", "CRW", "CRY",
    "NRQ", "NYG", "CH6", "CH7", "CRG", "CRU", "CRV", "CRS",
    "66A", "CR0", "GYC", "LYG", "TYG", "OHD", "SWG", "QYG",
    "CR7", "CR8", "CR9", "CRK", "RC7", "CFY", "PIA", "B2H",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedStructure:
    """Everything the rotamer scan needs from one structure."""

    pdb_id: str
    chrom_resname: str
    chrom_chain: str
    chrom_seqid: int
    chrom_atoms: dict[str, np.ndarray]
    chrom_elements: dict[str, str]
    extra_xyz: np.ndarray             # (M, 3) non-chromophore heavy atoms
    extra_elements: np.ndarray        # (M,) element symbols, upper-case
    phi_symmetric: bool = False       # True for phenol; False for His/Trp rings


def _element_symbol(atom: "gemmi.Atom") -> str:
    sym = atom.element.name.upper().strip()
    return sym or "C"


_BRIDGE_ATOMS = ("CA2", "CB2", "N2")
_DIHEDRAL_ATOMS = ("CA2", "CB2", "CG2", "CD1", "N2")


def _residue_has_bridge_pattern(residue: "gemmi.Residue") -> bool:
    """A residue qualifies as the FP chromophore if it carries the
    fused-tripeptide bridge atom set CA2 / CB2 / N2. Standard amino
    acids never use atom names with a numeric suffix, so this is
    unambiguous across the 69 chromophore three-letter codes in
    Luke's dataset. We do NOT require CG2 at this stage: some
    histidine- and cysteine-derived chromophores (CSH, CCY, SWG,
    ...) use CG instead. The CG2 / CD1 names are reconstructed by
    `_normalise_megley_atoms` below.
    """
    names = {a.name for a in residue if not a.element.is_hydrogen}
    return all(n in names for n in _BRIDGE_ATOMS)


def _normalise_megley_atoms(
    atoms: dict[str, np.ndarray], elements: dict[str, str]
) -> dict[str, str]:
    """In-place rename so the canonical Megley quadruple
        tau = N2  - CA2 - CB2 - CG2
        phi = CA2 - CB2 - CG2 - CD1
    can be evaluated for non-Tyr-derived chromophores.

    Rules
    -----
    * CG2 substitute: if CG2 is missing, look at the heavy-atom
      neighbours of CB2 in the chromophore bond graph; pick the
      unique one that isn't CA2. For CRO / CRF / NRQ the bond
      graph already exposes CG2; for CSH / CCY / SWG / ... it
      exposes CG, which we then rename to CG2.

    * CD1 substitute: if CD1 is missing, take a neighbour of the
      (renamed) CG2 atom that isn't CB2. Carbons are preferred
      over nitrogens; ties broken alphabetically. This gives CD2
      for histidine-derived imidazoles (IIC, CRG, CSH) and
      preserves CD1 wherever it already exists.

    Returns a dict {canonical_name -> source_name_in_cif} so callers
    can record which atom played which role.

    NOTE on phi and d_exp_to_planar: this routine keeps the deposited CD1
    where it exists, but the phenol ring's local two-fold makes the CD1/CD2
    assignment arbitrary across PDB depositions (the deposited CD1 is the
    far-side ortho carbon in ~30% of this dataset, and the two ortho
    dihedrals are ~170 deg apart, not exactly 180). The PRODUCTION
    d_exp_to_planar therefore uses a labeling-invariant phi measured through
    the canonical near-side ortho carbon (the one giving raw dihedral in
    (-90, 90]); see scripts/recompute_canonical_cd.py, which writes
    data/d_exp_canonical.csv (the authoritative d_exp source feeding
    merged_for_aggregate.csv and make_figure3.py). The live scan path here
    is left on the deposited CD1: f_allowed is invariant to a uniform
    phi-reference shift, so the cached scans remain valid.
    """
    from rotate import build_bond_graph as _bbg  # local to avoid cycle hint

    source = {n: n for n in atoms}

    if "CG2" not in atoms:
        graph = _bbg(atoms)
        cb2_neighbours = sorted(graph.get("CB2", set()) - {"CA2"})
        ring_candidates = [c for c in cb2_neighbours if c not in {"N2", "C2"}]
        if not ring_candidates:
            return source  # caller will raise on _DIHEDRAL_ATOMS check
        chosen = "CG" if "CG" in ring_candidates else ring_candidates[0]
        atoms["CG2"] = atoms.pop(chosen)
        elements["CG2"] = elements.pop(chosen)
        source["CG2"] = chosen
        source.pop(chosen, None)

    if "CD1" not in atoms:
        graph = _bbg(atoms)
        ring_neighbours = sorted(graph.get("CG2", set()) - {"CB2"})
        if not ring_neighbours:
            return source
        carbons = [n for n in ring_neighbours if n.startswith("C")]
        nitrogens = [n for n in ring_neighbours if n.startswith("N")]
        substitute = (carbons + nitrogens)[0]
        atoms["CD1"] = atoms.pop(substitute)
        elements["CD1"] = elements.pop(substitute)
        source["CD1"] = substitute
        source.pop(substitute, None)

    return source


def load_structure(cif_path: Path) -> LoadedStructure:
    """Read a structure and split into chromophore + everything else.

    The chromophore residue is found by atom-name pattern (presence
    of all of CA2, CB2, CG2, N2), not by 3-letter code, so this
    works across CRO, NRQ, CRF, IIC, ... without a maintained list.

    Raises RuntimeError if no chromophore residue is found or if
    the discovered residue lacks the CD1 atom that the Megley phi
    dihedral requires (e.g. histidine-derived BFP chromophores like
    IIC, where the imidazole ring has ND1 / NE2 instead of CD1).
    """
    structure = gemmi.read_structure(str(cif_path))
    pdb_id = structure.name.upper() or Path(cif_path).stem.upper()

    chrom_atoms: dict[str, np.ndarray] = {}
    chrom_elements: dict[str, str] = {}
    chrom_chain = ""
    chrom_seqid = 0
    chrom_resname = ""
    extra_xyz: list[np.ndarray] = []
    extra_elements: list[str] = []

    model = structure[0]
    chrom_found = False
    # Occupancy-ranked altloc selection for chromophore atoms: when an atom
    # name appears under several altloc labels, keep the highest-occupancy
    # copy (first-encountered, i.e. altloc 'A', wins ties). This makes the
    # deposited (tau_exp, phi_exp) deterministic rather than depending on
    # which conformer happens to be read last.
    chrom_occ: dict[str, float] = {}
    for chain in model:
        for residue in chain:
            if residue.name in WATER_RESNAMES:
                continue
            take_as_chrom = (
                (not chrom_found)
                and _residue_has_bridge_pattern(residue)
            )
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                xyz = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                )
                elem = _element_symbol(atom)
                if take_as_chrom:
                    if atom.name not in chrom_atoms or atom.occ > chrom_occ[atom.name]:
                        chrom_atoms[atom.name] = xyz
                        chrom_elements[atom.name] = elem
                        chrom_occ[atom.name] = atom.occ
                else:
                    extra_xyz.append(xyz)
                    extra_elements.append(elem)
            if take_as_chrom:
                chrom_found = True
                chrom_resname = residue.name
                chrom_chain = chain.name
                chrom_seqid = residue.seqid.num

    if not chrom_found:
        raise RuntimeError(
            f"no residue with the CA2/CB2/N2 bridge pattern in {cif_path}"
        )
    # Map CG -> CG2 and discover a CD1 substitute for non-Tyr rings.
    _normalise_megley_atoms(chrom_atoms, chrom_elements)
    missing = [n for n in _DIHEDRAL_ATOMS if n not in chrom_atoms]
    if missing:
        raise RuntimeError(
            f"{chrom_resname} chromophore in {cif_path.stem} is missing "
            f"atom(s) {missing} even after Megley-atom normalisation"
        )

    return LoadedStructure(
        pdb_id=pdb_id,
        chrom_resname=chrom_resname,
        chrom_chain=chrom_chain,
        chrom_seqid=chrom_seqid,
        chrom_atoms=chrom_atoms,
        chrom_elements=chrom_elements,
        extra_xyz=np.asarray(extra_xyz, dtype=float),
        extra_elements=np.asarray(extra_elements),
        phi_symmetric=is_phi_symmetric(chrom_atoms),
    )


# ---------------------------------------------------------------------------
# Cage assembly (everything the moving atoms can clash with)
# ---------------------------------------------------------------------------

def _lookup_radii(elements: np.ndarray) -> np.ndarray:
    radii = np.empty(len(elements), dtype=float)
    for i, e in enumerate(elements):
        radii[i] = BONDI_RADII.get(e.upper(), BONDI_RADII["C"])
    return radii


@dataclass(frozen=True)
class CageContext:
    """Fixed atoms a moving chromophore atom can clash against."""

    moving_names: tuple[str, ...]
    moving_elements: np.ndarray
    moving_radii: np.ndarray
    cage_xyz: np.ndarray         # static-CRO + extra
    cage_radii: np.ndarray
    exclude: np.ndarray          # (n_moving, n_cage) bool: True -> skip pair


def _graph_distance_le2(
    graph: dict[str, set[str]], source: str
) -> set[str]:
    """Atoms within 2 covalent bonds of `source` (excluding `source`)."""
    one_step = graph[source]
    two_step: set[str] = set()
    for n in one_step:
        two_step.update(graph[n])
    two_step.discard(source)
    return one_step | two_step


def build_cage(loaded: LoadedStructure) -> CageContext:
    """
    Determine which chromophore atoms move under a Megley scan and
    build the fixed cage of everything else, including the
    exclusion mask for 1,2 (covalent) and 1,3 (bond-angle) pairs.

    Exclusion rationale
    -------------------
    The bridge methylene CB2 sits ON the tau and phi rotation axes,
    so its bonded distance to CA2 (~1.34 A, a C=C) and its bond-angle
    distances to N2 and C2 (~2.4 A) are independent of the scan.
    Those pairs are below the vdW sum and would always be flagged
    as clashing. We exclude every pair that is 1, 2, or 3 bonds
    apart in the chromophore. The Megley-dihedral-defining 1,4
    pairs (e.g. CG2 - N2 for tau, CD1 - CA2 for phi) are *not*
    excluded; those distances genuinely depend on the rotamer and
    have to participate in the clash check.

    No moving chromophore atom is covalently bonded to any atom
    outside the chromophore residue, so no exclusions are needed
    for the `extra` part of the cage.
    """
    graph = build_bond_graph(loaded.chrom_atoms)
    moving_names = tuple(
        sorted(side_reachable_from(graph, start="CB2", blocked="CA2"))
    )
    moving_elem = np.array(
        [loaded.chrom_elements[n] for n in moving_names]
    )
    moving_radii = _lookup_radii(moving_elem)

    static_names = [
        n for n in loaded.chrom_atoms if n not in moving_names
    ]
    static_xyz = np.array(
        [loaded.chrom_atoms[n] for n in static_names], dtype=float
    )
    static_elem = np.array(
        [loaded.chrom_elements[n] for n in static_names]
    )
    if len(static_xyz) == 0:
        static_xyz = np.empty((0, 3), dtype=float)
        static_elem = np.empty((0,), dtype=str)

    cage_xyz = np.vstack([static_xyz, loaded.extra_xyz])
    cage_elem = np.concatenate([static_elem, loaded.extra_elements])
    cage_radii = _lookup_radii(cage_elem)

    # Build the 1,2 + 1,3 exclusion mask. The cage is laid out as
    # [static_chrom_atoms ... extra_atoms ...]; the extra block is
    # never excluded because no covalent bonds cross the chrom /
    # extra boundary.
    static_index = {name: j for j, name in enumerate(static_names)}
    exclude = np.zeros(
        (len(moving_names), len(cage_xyz)), dtype=bool
    )
    for i, m in enumerate(moving_names):
        nearby = _graph_distance_le2(graph, m)
        for s in nearby:
            j = static_index.get(s)
            if j is not None:
                exclude[i, j] = True

    return CageContext(
        moving_names=moving_names,
        moving_elements=moving_elem,
        moving_radii=moving_radii,
        cage_xyz=cage_xyz,
        cage_radii=cage_radii,
        exclude=exclude,
    )


# ---------------------------------------------------------------------------
# Overlap and clash check
# ---------------------------------------------------------------------------

def max_overlap(
    moving_xyz: np.ndarray,
    moving_radii: np.ndarray,
    cage_xyz: np.ndarray,
    cage_radii: np.ndarray,
    exclude: np.ndarray | None = None,
) -> tuple[float, tuple[int, int]]:
    """
    Return (largest overlap in A, (i_moving, j_cage)).
    Positive overlap means atoms penetrate; negative means a gap.
    Pairs flagged True in `exclude` are skipped.
    """
    diff = moving_xyz[:, None, :] - cage_xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    overlap = moving_radii[:, None] + cage_radii[None, :] - dist
    if exclude is not None:
        overlap = np.where(exclude, -np.inf, overlap)
    flat = int(np.argmax(overlap))
    i, j = divmod(flat, overlap.shape[1])
    return float(overlap[i, j]), (i, j)


def overlap_at(
    cage: CageContext,
    chrom_atoms_now: dict[str, np.ndarray],
) -> tuple[float, tuple[int, int]]:
    """Largest overlap for the current set of chromophore coords."""
    moving_xyz = np.array(
        [chrom_atoms_now[n] for n in cage.moving_names], dtype=float
    )
    return max_overlap(
        moving_xyz,
        cage.moving_radii,
        cage.cage_xyz,
        cage.cage_radii,
        exclude=cage.exclude,
    )


def is_allowed(
    cage: CageContext,
    chrom_atoms_now: dict[str, np.ndarray],
    tolerance_a: float = DEFAULT_TOLERANCE_A,
) -> bool:
    """Allowed when the largest atom-atom overlap is <= tolerance."""
    return overlap_at(cage, chrom_atoms_now)[0] <= tolerance_a


# ---------------------------------------------------------------------------
# Convenience: evaluate a (tau, phi) rotamer in one call
# ---------------------------------------------------------------------------

def overlap_for_megley(
    loaded: LoadedStructure,
    cage: CageContext,
    tau_target_deg: float,
    phi_target_deg: float,
) -> float:
    """Largest overlap (A) for the rotamer set to the given (tau, phi)."""
    rotated = set_megley(loaded.chrom_atoms, tau_target_deg, phi_target_deg)
    return overlap_at(cage, rotated)[0]


__all__ = [
    "BONDI_RADII",
    "DEFAULT_TOLERANCE_A",
    "CHROMOPHORE_RESIDUES",
    "LoadedStructure",
    "CageContext",
    "load_structure",
    "build_cage",
    "max_overlap",
    "overlap_at",
    "is_allowed",
    "overlap_for_megley",
]
