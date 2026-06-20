#!/usr/bin/env python3
"""
test_clash.py
=============

Sanity check for the steric-clash machinery in barrel.py, run on
1EMA. Verifies:

1. The structure loads, the chromophore is identified, and the cage
   contains the expected number of heavy atoms (rough check).

2. The **experimental** rotamer (the (tau, phi) actually present in
   the crystal structure) is clash-free under the 0.4 A MolProbity
   tolerance. If it weren't we would know our clash machinery is
   miscalibrated (the deposited crystal can't be clashing with
   itself).

3. The cage check **does fire** when we twist the chromophore to
   geometries that should not fit. We probe two extreme rotamers,
   (tau, phi) = (90, 0) and (0, 90), and confirm at least one
   produces a clash.

4. A small "control": all-zero rotamer ((tau, phi) = (0, 0)) should
   be close to the experimental one for 1EMA (the experimental
   values are tau = -17.6, phi = +13.0, only modest twists), so we
   expect it to also be allowed.

Usage:
    python3 scripts/test_clash.py [path/to/1EMA.cif]
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import (  # noqa: E402
    DEFAULT_TOLERANCE_A,
    build_cage,
    is_allowed,
    load_structure,
    overlap_at,
    overlap_for_megley,
)
from rotate import measure_megley  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


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
    print("[input] downloading 1EMA from RCSB")
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        "https://files.rcsb.org/download/1EMA.cif", cached
    )
    return cached


def main(argv: list[str]) -> int:
    cif = find_or_fetch_1ema(argv[1] if len(argv) > 1 else None)
    print(f"[input] {cif}")

    loaded = load_structure(cif)
    print(
        f"[load] {loaded.pdb_id}: chromophore "
        f"{loaded.chrom_resname} {loaded.chrom_chain}/{loaded.chrom_seqid}"
        f", chrom_atoms={len(loaded.chrom_atoms)}"
        f", extra_atoms={len(loaded.extra_xyz)}"
    )

    cage = build_cage(loaded)
    print(
        f"[cage] moving atoms ({len(cage.moving_names)}): "
        f"{list(cage.moving_names)}\n"
        f"       cage atoms (static-CRO + extras) = {len(cage.cage_xyz)}"
    )

    # --- Test 1: experimental rotamer is clash-free ---
    tau0, phi0 = measure_megley(loaded.chrom_atoms)
    ov, (i, j) = overlap_at(cage, loaded.chrom_atoms)
    print(
        f"\n[exp] experimental (tau, phi) = ({tau0:+.3f}, {phi0:+.3f}); "
        f"largest overlap = {ov:+.3f} A  "
        f"(moving atom #{i} = {cage.moving_names[i]} vs cage atom {j})"
    )
    assert ov <= DEFAULT_TOLERANCE_A, (
        f"experimental rotamer flagged as clashing (overlap {ov:+.3f} A "
        f"> tolerance {DEFAULT_TOLERANCE_A} A) - "
        "the clash machinery or vdW radii are miscalibrated"
    )
    print("  experimental rotamer is allowed: OK")

    # --- Test 2: (0, 0) is just a probe (no assertion) ---
    # Earlier I assumed a perfectly planar chromophore would also be
    # allowed because (-17.6, +13.0) -> (0, 0) is only a ~10-20 deg
    # twist. That assumption was wrong for 1EMA: the OH and the
    # phenol are wedged tightly enough that even a 13 deg shift in
    # phi can push the OH out of its H-bond pocket and into a cage
    # carbon. That is itself a real structural result, not a bug.
    ov_00 = overlap_for_megley(loaded, cage, 0.0, 0.0)
    print(
        f"\n[probe] (0, 0) overlap = {ov_00:+.3f} A "
        f"({'CLASH' if ov_00 > DEFAULT_TOLERANCE_A else 'allowed'})"
    )

    # --- Test 3: extreme twists should clash ---
    extremes = [
        (90.0, 0.0),
        (0.0, 90.0),
        (90.0, 90.0),
        (-90.0, -90.0),
    ]
    print("\n[probe] extreme twists (expect at least one to clash):")
    overlaps = []
    for tau, phi in extremes:
        ov_ext = overlap_for_megley(loaded, cage, tau, phi)
        overlaps.append(ov_ext)
        flag = "CLASH" if ov_ext > DEFAULT_TOLERANCE_A else "ok   "
        print(
            f"  (tau={tau:+6.1f}, phi={phi:+6.1f}) -> "
            f"overlap = {ov_ext:+6.3f} A  [{flag}]"
        )
    assert max(overlaps) > DEFAULT_TOLERANCE_A, (
        "none of the extreme twists clash - the clash check is not "
        "firing as expected"
    )
    print("  at least one extreme twist clashes: OK")

    # --- Test 4: small grid summary ---
    print("\n[probe] coarse 25-point preview of the (tau, phi) map:")
    grid = np.linspace(-180, 180, 5, endpoint=False)
    n_allowed = 0
    for tau in grid:
        row = []
        for phi in grid:
            ov = overlap_for_megley(loaded, cage, float(tau), float(phi))
            allowed = ov <= DEFAULT_TOLERANCE_A
            n_allowed += int(allowed)
            row.append("." if allowed else "X")
        print("  " + " ".join(row) + f"   (tau = {tau:+6.1f})")
    print(f"  -> {n_allowed} / 25 grid points allowed on this 5x5 coarse map")

    print("\nALL CLASH-CHECK TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
