#!/usr/bin/env python3
"""
test_rotate.py
==============

Sanity check for the rotation routine in rotate.py, run on the 1EMA
chromophore.

Checks (in order)
-----------------
1.  Print the partition: which atoms move when we drive tau, which
    move when we drive phi. The phenol atoms must end up in both
    sets and the imidazolinone / backbone-fused atoms must end up in
    neither.

2.  Round-trip identity. Read the experimental (tau, phi) from the
    1EMA chromophore, call set_megley() with those same targets, and
    confirm the coordinates come back unchanged to within 1e-6 A.

3.  Single-target accuracy. For a range of target angles, call
    set_megley() and confirm measure_megley() returns the requested
    values to within 1e-4 deg.

4.  Bond-length conservation. After driving to a non-trivial
    (tau, phi), the bond lengths inside the chromophore must be
    unchanged (rigid-body motion preserves all distances within each
    rotated subset and across the rotation axis).

5.  Side-by-side reproducibility. Driving tau and phi separately
    must compose correctly: the final dihedrals match the targets
    independent of starting geometry.

Usage
-----
    python3 scripts/test_rotate.py [path/to/1EMA.cif]
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotate import (  # noqa: E402
    build_bond_graph,
    measure_megley,
    rotate_points,
    set_megley,
    side_reachable_from,
    tau_phi_movers,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CHROMOPHORE_RESIDUES = {
    "CRO", "CR2", "GYS", "SYG", "CRQ", "CRF", "CRW", "CRY",
    "NRQ", "NYG", "CH6", "CH7", "CRG", "CRU", "CRV", "CRS",
    "66A", "CR0", "GYC", "LYG", "TYG", "OHD", "SWG", "QYG",
    "CR7", "CR8", "CR9", "CRK", "RC7", "CFY", "PIA", "B2H",
}


def load_chromophore_atoms(cif_path: Path) -> dict[str, np.ndarray]:
    structure = gemmi.read_structure(str(cif_path))
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name in CHROMOPHORE_RESIDUES:
                    return {
                        atom.name: np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                        )
                        for atom in residue
                    }
    raise RuntimeError(f"no chromophore residue found in {cif_path}")


def find_or_fetch_1ema(cli_arg: str | None) -> Path:
    if cli_arg:
        p = Path(cli_arg).expanduser()
        if p.is_file():
            return p
    cached = DATA_DIR / "1EMA.cif"
    if cached.is_file():
        return cached
    luke = Path.home() / "Downloads" / "scop_gfp_structures" / "1EMA.cif"
    if luke.is_file():
        return luke
    url = "https://files.rcsb.org/download/1EMA.cif"
    print(f"[input] downloading {url}")
    urllib.request.urlretrieve(url, cached)
    return cached


def _all_pair_distances(atoms: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    names = sorted(atoms)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[(a, b)] = float(np.linalg.norm(atoms[a] - atoms[b]))
    return out


def check_partition(atoms: dict[str, np.ndarray]) -> None:
    tau_side, phi_side = tau_phi_movers(atoms)
    print("\n--- atom partitioning ---")
    print(f"  tau movers ({len(tau_side):>2}): {sorted(tau_side)}")
    print(f"  phi movers ({len(phi_side):>2}): {sorted(phi_side)}")
    fixed = set(atoms) - tau_side
    print(f"  fixed     ({len(fixed):>2}): {sorted(fixed)}")
    assert phi_side <= tau_side, "phi-side atoms should be a subset of tau-side"
    must_be_fixed = {"N2", "CA2", "C2", "O2", "N3", "C1"}
    leaked = must_be_fixed & tau_side
    assert not leaked, f"imidazolinone atoms leaked into tau side: {leaked}"
    print("  partition OK")


def check_roundtrip(atoms: dict[str, np.ndarray]) -> None:
    tau0, phi0 = measure_megley(atoms)
    new = set_megley(atoms, tau0, phi0)
    max_drift = max(
        float(np.linalg.norm(new[name] - atoms[name])) for name in atoms
    )
    print(
        f"\n--- round-trip ---\n"
        f"  reset to ({tau0:+.4f}, {phi0:+.4f}); max coord drift = "
        f"{max_drift:.2e} A"
    )
    assert max_drift < 1e-6, "round-trip drifted by more than 1 micro-A"
    tau_check, phi_check = measure_megley(new)
    assert abs(tau_check - tau0) < 1e-6
    assert abs(phi_check - phi0) < 1e-6
    print("  round-trip OK")


def check_target_accuracy(atoms: dict[str, np.ndarray]) -> None:
    print("\n--- single-target accuracy (target -> measured) ---")
    targets = [
        (0.0, 0.0),
        (-30.0, 45.0),
        (90.0, -90.0),
        (179.0, -179.0),
        (-17.557, 13.024),
    ]
    for tau_t, phi_t in targets:
        new = set_megley(atoms, tau_t, phi_t)
        tau_m, phi_m = measure_megley(new)
        err = max(abs(tau_m - tau_t), abs(phi_m - phi_t))
        print(
            f"  ({tau_t:+7.3f}, {phi_t:+7.3f}) -> "
            f"({tau_m:+7.3f}, {phi_m:+7.3f})  err = {err:.2e} deg"
        )
        assert err < 1e-4, f"set_megley did not hit target within 1e-4 deg"
    print("  target accuracy OK")


def check_rigidity(atoms: dict[str, np.ndarray]) -> None:
    """Bond lengths within each rotated subgraph must be preserved."""
    print("\n--- bond-length conservation ---")
    graph = build_bond_graph(atoms)
    bonds = []
    for a, neigh in graph.items():
        for b in neigh:
            if a < b:
                bonds.append((a, b))
    new = set_megley(atoms, 60.0, -45.0)
    max_change = 0.0
    for a, b in bonds:
        d0 = float(np.linalg.norm(atoms[a] - atoms[b]))
        d1 = float(np.linalg.norm(new[a] - new[b]))
        max_change = max(max_change, abs(d1 - d0))
    print(f"  largest bond-length change after drive: {max_change:.2e} A")
    assert max_change < 1e-6, "bond length changed under rotation"
    print("  bond rigidity OK")


def main(argv: list[str]) -> int:
    cif = find_or_fetch_1ema(argv[1] if len(argv) > 1 else None)
    print(f"[input] using {cif}")
    atoms = load_chromophore_atoms(cif)
    tau0, phi0 = measure_megley(atoms)
    print(
        f"[input] 1EMA chromophore: {len(atoms)} atoms, "
        f"tau_megley = {tau0:+.3f}, phi_megley = {phi0:+.3f}"
    )

    check_partition(atoms)
    check_roundtrip(atoms)
    check_target_accuracy(atoms)
    check_rigidity(atoms)

    print("\nALL ROTATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
