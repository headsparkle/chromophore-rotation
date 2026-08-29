"""
rotate.py
=========

Rigid driving of the two Megley torsions of a fluorescent-protein
chromophore.

Convention (verified against gfp-barrel-geometry/data/megley_dihedrals.csv
on 2026-06-10):

    tau_megley = N2  - CA2 - CB2 - CG2     (around CA2=CB2, "I-bond")
    phi_megley = CA2 - CB2 - CG2 - CD1     (around CB2-CG2, "P-bond")

Driving strategy
----------------
The imidazolinone half of the chromophore is fused into the protein
backbone (CA1 and CA3 are bonded to neighbouring residues). Rotating
the imidazolinone half would therefore move atoms that are
covalently anchored to the rest of the structure, producing
artefactual clashes. So for both Megley torsions we rotate the
bridge / phenol half:

    drive tau : rotate atoms reachable from CB2 (BFS, blocking CA2)
                around the CA2 - CB2 axis.
    drive phi : rotate atoms reachable from CG2 (BFS, blocking CB2)
                around the CB2 - CG2 axis.

The two rotations are applied sequentially. Order matters slightly
in the sense that the second rotation uses the post-first-rotation
positions of CB2 and CG2 as its axis, which is the natural choice
("rotate the phenol around its current single bond").

Topology is inferred from a distance-based bond graph (heavy atoms
within 1.8 A). This works without prior knowledge of the chromophore
3-letter code as long as the standard CRO-style names (CA2, CB2,
CG2, CD1, N2) are present. Non-standard codes that rename these
atoms will need an explicit mapping which can be added later.
"""

from __future__ import annotations

from collections import deque

import numpy as np


_BOND_CUTOFF_A = 1.8

_TAU_QUAD = ("N2", "CA2", "CB2", "CG2")
_PHI_QUAD = ("CA2", "CB2", "CG2", "CD1")


# ----------------------------------------------------------------------------
# Geometry primitives
# ----------------------------------------------------------------------------

def dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Signed dihedral angle (deg), IUPAC convention."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    b2_hat = b2 / np.linalg.norm(b2)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2_hat)
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit axis and an angle in degrees."""
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotate_points(
    coords: np.ndarray,
    axis_p1: np.ndarray,
    axis_p2: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Rotate Nx3 coords around the line through axis_p1 -> axis_p2."""
    R = rotation_matrix(axis_p2 - axis_p1, angle_deg)
    return axis_p1 + (coords - axis_p1) @ R.T


# ----------------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------------

def build_bond_graph(
    atoms: dict[str, np.ndarray],
    cutoff_a: float = _BOND_CUTOFF_A,
) -> dict[str, set[str]]:
    """Heavy-atom adjacency graph from inter-atom distances."""
    names = list(atoms)
    coords = np.asarray([atoms[n] for n in names])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    graph: dict[str, set[str]] = {n: set() for n in names}
    bonded = np.argwhere(dist < cutoff_a)
    for i, j in bonded:
        graph[names[i]].add(names[j])
    return graph


def side_reachable_from(
    graph: dict[str, set[str]],
    start: str,
    blocked: str,
) -> set[str]:
    """BFS from `start` without traversing through `blocked`."""
    if start not in graph or blocked not in graph:
        raise KeyError(
            f"atoms missing from graph: start={start!r}, blocked={blocked!r}"
        )
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if nbr == blocked or nbr in visited:
                continue
            visited.add(nbr)
            queue.append(nbr)
    if blocked in visited:
        raise RuntimeError(
            f"BFS leaked through blocked atom {blocked!r} - check connectivity"
        )
    return visited


# ----------------------------------------------------------------------------
# Driving the Megley torsions
# ----------------------------------------------------------------------------

def measure_megley(atoms: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return (tau_megley, phi_megley) of the supplied atom dict, in degrees."""
    tau = dihedral(*(atoms[n] for n in _TAU_QUAD))
    phi = dihedral(*(atoms[n] for n in _PHI_QUAD))
    return tau, phi


def _apply_rotation(
    atoms: dict[str, np.ndarray],
    movers: set[str],
    axis_p1: np.ndarray,
    axis_p2: np.ndarray,
    angle_deg: float,
) -> None:
    """In-place rotation of the named atoms around an axis."""
    if abs(angle_deg) < 1e-12:
        return
    names = list(movers)
    coords = np.asarray([atoms[n] for n in names])
    rotated = rotate_points(coords, axis_p1, axis_p2, angle_deg)
    for name, new_xyz in zip(names, rotated):
        atoms[name] = new_xyz


def set_megley(
    atoms: dict[str, np.ndarray],
    tau_target_deg: float,
    phi_target_deg: float,
    graph: dict[str, set[str]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Return a new atom dict with tau_megley == tau_target and
    phi_megley == phi_target. The input is not modified.
    """
    new = {k: np.array(v, dtype=float, copy=True) for k, v in atoms.items()}
    if graph is None:
        graph = build_bond_graph(new)

    # Drive tau: rotate the CB2 side around the CA2-CB2 axis.
    # Sign note: the IUPAC dihedral A-B-C-D is positive when D is CCW
    # from A looking along B -> C. A Rodrigues rotation of D about the
    # same B -> C axis by +theta moves D CW from that viewpoint (it is
    # CCW when looking from C back toward B). So increasing the
    # dihedral by Delta requires a Rodrigues rotation by -Delta.
    tau_now = dihedral(*(new[n] for n in _TAU_QUAD))
    d_tau = _wrap_signed_deg(tau_target_deg - tau_now)
    tau_side = side_reachable_from(graph, start="CB2", blocked="CA2")
    _apply_rotation(new, tau_side, new["CA2"], new["CB2"], -d_tau)

    # Drive phi: rotate the CG2 side around the now-current CB2-CG2 axis.
    phi_now = dihedral(*(new[n] for n in _PHI_QUAD))
    d_phi = _wrap_signed_deg(phi_target_deg - phi_now)
    phi_side = side_reachable_from(graph, start="CG2", blocked="CB2")
    _apply_rotation(new, phi_side, new["CB2"], new["CG2"], -d_phi)

    return new


def _wrap_signed_deg(angle: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a if a != -180.0 else 180.0


# ----------------------------------------------------------------------------
# Inspection helpers (used by the test driver)
# ----------------------------------------------------------------------------

def tau_phi_movers(atoms: dict[str, np.ndarray]) -> tuple[set[str], set[str]]:
    """Return (atoms moved when driving tau, atoms moved when driving phi)."""
    graph = build_bond_graph(atoms)
    return (
        side_reachable_from(graph, start="CB2", blocked="CA2"),
        side_reachable_from(graph, start="CG2", blocked="CB2"),
    )


def is_phi_symmetric(atoms: dict[str, np.ndarray]) -> bool:
    """Does a 180-degree rotation about the CB2-CG2 axis map the
    moving ring atoms onto themselves with matching elements?

    Test: for each moving atom we find its nearest neighbour in the
    180-degree-rotated copy of the moving set, and require that
    neighbour to share its element (first letter of the atom name).
    No distance threshold: a phenol ring will map every C to a
    same-element C (the worst real-crystal mismatch in this dataset
    is about 0.8 A for the para OH); an imidazole or indole ring
    will inevitably map a C to an N (or vice versa) because the
    ring is built with C and N in non-2-fold-equivalent positions.

    Returns True for Tyr-derived phenol rings (CRO, CRF, ...) where
    (tau, phi) and (tau, phi + 180) are the same physical
    configuration so f_allowed can legitimately be folded across the
    phi axis. Returns False for histidine-derived imidazoles (IIC,
    CRG, CSH) and tryptophan-derived indoles (SWG) where folding
    would double-count allowed area.
    """
    graph = build_bond_graph(atoms)
    moving = side_reachable_from(graph, start="CG2", blocked="CB2")
    cb2 = atoms["CB2"]
    cg2 = atoms["CG2"]
    R = rotation_matrix(cg2 - cb2, 180.0)
    rotated = {n: cb2 + R @ (atoms[n] - cb2) for n in moving}

    def first_letter(name: str) -> str:
        return name[0] if name else "?"

    for orig_name in moving:
        orig_xyz = atoms[orig_name]
        orig_elem = first_letter(orig_name)
        nearest_name = min(
            rotated,
            key=lambda rn: float(np.linalg.norm(orig_xyz - rotated[rn])),
        )
        if first_letter(nearest_name) != orig_elem:
            return False
    return True
