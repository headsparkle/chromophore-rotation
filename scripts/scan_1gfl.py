#!/usr/bin/env python3
"""
scan_1gfl.py
============

Production scan of 1GFL (Yang, Moss, Phillips 1996, wt GFP dimer at
1.9 A) so we can compare apples-to-apples against:

  Baffour-Awuah & Zimmer 2004 (Chem. Phys.) - constrained AMBER*/MacroModel
     minimisation of the same 1GFL coordinates. Found that the cavity is
     complementary to a positively correlated hula twist (HT) with the
     minimum near (tau, phi) = (+45, +45) deg.

  Megley, Dickson, Maddalo, Chandler & Zimmer 2009 (J. Phys. Chem. B) -
     freely-rotating MD on the same 1GFL chain A. Found a negatively
     correlated HT cloud (bottom-HT) with phi vs tau best-fit slope
     -0.857, intercept -8.75, convex hull 2575 deg2, 80%-area 627 deg2.

The 1996 1GFL deposition predates the CRO ligand code, so the chromophore
is modelled as discrete SER65 / TYR66 / GLY67. Our pipeline finds a
chromophore residue by the CA2 / CB2 / N2 bridge pattern, which 1GFL
lacks. The preprocessor below copies the CIF into data/1GFL_cro.cif
with TYR66 chain A atoms renamed N -> N2, CA -> CA2, CB -> CB2,
CG -> CG2 (CD1 / CD2 / CE1 / CE2 / CZ / OH already match), and the
residue retagged CRO. SER65 and GLY67 remain as separate residues and
contribute to the steric cage like any other backbone, which matches
the way the 2004 and 2009 simulations treated 1GFL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import (  # noqa: E402
    DEFAULT_TOLERANCE_A,
    build_cage,
    load_structure,
    max_overlap,
)
from rotate import measure_megley, set_megley  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

PDB_ID = "1GFL"
STEP_DEG = 5.0
TARGET_CHAIN = "A"
TARGET_RES_SEQ = 66
TARGET_RES_NAME = "TYR"

TYR_TO_CRO = {
    "N": "N2",
    "CA": "CA2",
    "CB": "CB2",
    "CG": "CG2",
}


def make_cro_cif(src_cif: Path, out_cif: Path) -> None:
    """Rewrite 1GFL so chain A TYR66 atoms carry the CRO names that the
    barrel.load_structure bridge detector expects."""
    struct = gemmi.read_structure(str(src_cif))
    touched = False
    for model in struct:
        for chain in model:
            if chain.name != TARGET_CHAIN:
                continue
            for res in chain:
                if (
                    res.name == TARGET_RES_NAME
                    and res.seqid.num == TARGET_RES_SEQ
                ):
                    for atom in res:
                        if atom.name in TYR_TO_CRO:
                            atom.name = TYR_TO_CRO[atom.name]
                    res.name = "CRO"
                    touched = True
    if not touched:
        raise RuntimeError(
            f"could not find chain {TARGET_CHAIN} TYR{TARGET_RES_SEQ} in {src_cif}"
        )
    struct.make_mmcif_document().write_file(str(out_cif))


def grid(step: float) -> np.ndarray:
    return np.arange(-180.0, 180.0, step)


def scan_overlap_map(loaded, cage, step_deg: float):
    tau_grid = grid(step_deg)
    phi_grid = grid(step_deg)
    overlap_map = np.empty((len(tau_grid), len(phi_grid)), dtype=float)
    base_atoms = loaded.chrom_atoms
    moving_names = cage.moving_names
    moving_radii = cage.moving_radii
    cage_xyz = cage.cage_xyz
    cage_radii = cage.cage_radii
    exclude = cage.exclude
    t0 = time.perf_counter()
    for i, tau in enumerate(tau_grid):
        for j, phi in enumerate(phi_grid):
            atoms_now = set_megley(base_atoms, float(tau), float(phi))
            moving_xyz = np.array(
                [atoms_now[n] for n in moving_names], dtype=float
            )
            ov, _ = max_overlap(
                moving_xyz, moving_radii, cage_xyz, cage_radii, exclude
            )
            overlap_map[i, j] = ov
    dt = time.perf_counter() - t0
    print(f"[scan] {overlap_map.size} rotamers in {dt:.2f} s "
          f"({1000 * dt / overlap_map.size:.2f} ms each)")
    return tau_grid, phi_grid, overlap_map


def nearest_distance_deg(p_tau, p_phi, tau_grid, phi_grid, mask):
    if not mask.any():
        return float("nan")
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    dT = (T - p_tau + 180.0) % 360.0 - 180.0
    dP = (P - p_phi + 180.0) % 360.0 - 180.0
    return float(np.sqrt((dT * dT + dP * dP)[mask].min()))


def write_summary(pdb_id, step_deg, tau_grid, phi_grid, overlap_map,
                  tau_exp, phi_exp, tolerance_a, summary_csv):
    allowed = overlap_map <= tolerance_a
    n_allowed = int(allowed.sum())
    n_total = overlap_map.size
    cell = step_deg * step_deg
    A = n_allowed * cell
    f = A / (360.0 * 360.0)
    d = nearest_distance_deg(tau_exp, phi_exp, tau_grid, phi_grid, ~allowed)
    i_exp = int(np.argmin(np.abs(((tau_grid - tau_exp + 180) % 360) - 180)))
    j_exp = int(np.argmin(np.abs(((phi_grid - phi_exp + 180) % 360) - 180)))
    summary = {
        "pdb_id": pdb_id,
        "step_deg": step_deg,
        "tolerance_a": tolerance_a,
        "n_total": n_total,
        "n_allowed": n_allowed,
        "A_allowed_deg2": A,
        "f_allowed": f,
        "tau_exp_deg": tau_exp,
        "phi_exp_deg": phi_exp,
        "exp_cell_allowed": bool(allowed[i_exp, j_exp]),
        "exp_cell_overlap_a": float(overlap_map[i_exp, j_exp]),
        "d_exp_to_clash_deg": d,
        "max_overlap_a": float(overlap_map.max()),
        "min_overlap_a": float(overlap_map.min()),
    }
    keys = list(summary)
    with open(summary_csv, "w") as f_out:
        f_out.write(",".join(keys) + "\n")
        f_out.write(",".join(str(summary[k]) for k in keys) + "\n")
    return summary


def plot_map(pdb_id, tau_grid, phi_grid, overlap_map, tau_exp, phi_exp,
             tolerance_a, out_path):
    step = float(tau_grid[1] - tau_grid[0])
    extent = (
        tau_grid[0] - step / 2,
        tau_grid[-1] + step / 2,
        phi_grid[0] - step / 2,
        phi_grid[-1] + step / 2,
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    field = np.clip(overlap_map, -1.0, 3.0).T
    im = ax.imshow(
        field, origin="lower", extent=extent, cmap="RdYlGn_r",
        vmin=-1.0, vmax=3.0, aspect="equal", interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("max steric overlap (A)\n(green = clash-free)")
    Tcell, Pcell = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    ax.contour(Tcell, Pcell, overlap_map, levels=[tolerance_a],
               colors="black", linewidths=1.2)
    ax.plot(tau_exp, phi_exp, marker="o", markersize=10,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.5, linestyle="",
            label=f"experimental ({tau_exp:+.1f}, {phi_exp:+.1f})")
    ax.set_xlabel(r"$\tau_{\mathrm{megley}}$ (deg)")
    ax.set_ylabel(r"$\varphi_{\mathrm{megley}}$ (deg)")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title(f"{pdb_id} freely-rotating chromophore (tau, phi) scan, 5 deg grid")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    src_cif = DATA_DIR / f"{PDB_ID}.cif"
    cro_cif = DATA_DIR / f"{PDB_ID}_cro.cif"
    if not src_cif.is_file():
        sys.exit(f"missing {src_cif}")
    if not cro_cif.is_file():
        print(f"[prep] writing {cro_cif} with renamed TYR66 atoms")
        make_cro_cif(src_cif, cro_cif)
    print(f"[input] {cro_cif}")

    loaded = load_structure(cro_cif)
    cage = build_cage(loaded)
    tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)
    print(
        f"[load] {loaded.pdb_id}: {loaded.chrom_resname} "
        f"{loaded.chrom_chain}/{loaded.chrom_seqid}, "
        f"exp (tau, phi) = ({tau_exp:+.3f}, {phi_exp:+.3f}), "
        f"n_cage_atoms = {len(cage.cage_xyz)}"
    )

    tau_grid, phi_grid, overlap_map = scan_overlap_map(loaded, cage, STEP_DEG)

    npz_path = DATA_DIR / f"scan_{PDB_ID}_free.npz"
    np.savez_compressed(
        npz_path,
        tau_grid=tau_grid, phi_grid=phi_grid, overlap_map=overlap_map,
        tau_exp=tau_exp, phi_exp=phi_exp,
        tolerance_a=DEFAULT_TOLERANCE_A, step_deg=STEP_DEG, pdb_id=PDB_ID,
    )
    print(f"[save] grid -> {npz_path}")

    summary_csv = DATA_DIR / f"scan_{PDB_ID}_summary.csv"
    summary = write_summary(
        PDB_ID, STEP_DEG, tau_grid, phi_grid, overlap_map,
        tau_exp, phi_exp, DEFAULT_TOLERANCE_A, summary_csv,
    )
    print(f"[save] summary -> {summary_csv}")
    for k in (
        "n_allowed", "n_total", "A_allowed_deg2", "f_allowed",
        "exp_cell_allowed", "exp_cell_overlap_a", "d_exp_to_clash_deg",
        "max_overlap_a", "min_overlap_a",
    ):
        print(f"  {k:>22s} = {summary[k]}")

    fig_path = FIG_DIR / f"scan_{PDB_ID}_free.png"
    plot_map(
        PDB_ID, tau_grid, phi_grid, overlap_map,
        tau_exp, phi_exp, DEFAULT_TOLERANCE_A, fig_path,
    )
    print(f"[save] figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
