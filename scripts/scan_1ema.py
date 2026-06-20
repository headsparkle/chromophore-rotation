#!/usr/bin/env python3
"""
scan_1ema.py
============

Production run of the freely-rotating Megley (tau, phi) scan on
1EMA (avGFP). For every point on a regular 5 deg grid over
[-180, +180) x [-180, +180) (72 x 72 = 5184 rotamers) we drive the
chromophore to that (tau, phi) and record the largest steric
overlap against the rest of the structure. The freely-rotating
variant means we ignore the intrinsic torsional barrier of the
chromophore bonds; only collisions with the protein cage forbid a
rotamer.

Outputs
-------
- data/scan_1EMA_free.npz         (full overlap grid + axis arrays)
- data/scan_1EMA_summary.csv      (one row, scalar summary)
- figures/scan_1EMA_free.png      (heat map + allowed island + exp marker)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

PDB_ID = "1EMA"
STEP_DEG = 5.0


def grid(step: float) -> np.ndarray:
    return np.arange(-180.0, 180.0, step)


def scan_overlap_map(loaded, cage, step_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tau_grid, phi_grid, overlap_map [n_tau, n_phi])."""
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
          f"({1000*dt/overlap_map.size:.2f} ms each)")
    return tau_grid, phi_grid, overlap_map


def nearest_distance_deg(
    point_tau: float, point_phi: float,
    tau_grid: np.ndarray, phi_grid: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Shortest angular distance (deg) from (point) to any True grid cell.

    Angles are wrapped to (-180, 180]; grid cells are treated as
    centered samples.
    """
    if not mask.any():
        return float("nan")
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    dT = (T - point_tau + 180.0) % 360.0 - 180.0
    dP = (P - point_phi + 180.0) % 360.0 - 180.0
    d2 = dT * dT + dP * dP
    return float(np.sqrt(d2[mask].min()))


def write_summary(
    pdb_id: str,
    step_deg: float,
    tau_grid: np.ndarray,
    phi_grid: np.ndarray,
    overlap_map: np.ndarray,
    tau_exp: float,
    phi_exp: float,
    tolerance_a: float,
    summary_csv: Path,
) -> dict:
    allowed = overlap_map <= tolerance_a
    n_allowed = int(allowed.sum())
    n_total = overlap_map.size
    cell_area_deg2 = step_deg * step_deg
    A_allowed = n_allowed * cell_area_deg2
    f_allowed = A_allowed / (360.0 * 360.0)
    # Closest disallowed cell to the experimental point (deg)
    d_exp_to_edge = nearest_distance_deg(
        tau_exp, phi_exp, tau_grid, phi_grid, ~allowed
    )
    # Did the experimental point's own grid cell come out allowed?
    i_exp = int(np.argmin(np.abs(((tau_grid - tau_exp + 180) % 360) - 180)))
    j_exp = int(np.argmin(np.abs(((phi_grid - phi_exp + 180) % 360) - 180)))
    exp_cell_allowed = bool(allowed[i_exp, j_exp])
    exp_cell_overlap = float(overlap_map[i_exp, j_exp])

    summary = {
        "pdb_id": pdb_id,
        "step_deg": step_deg,
        "tolerance_a": tolerance_a,
        "n_total": n_total,
        "n_allowed": n_allowed,
        "A_allowed_deg2": A_allowed,
        "f_allowed": f_allowed,
        "tau_exp_deg": tau_exp,
        "phi_exp_deg": phi_exp,
        "exp_cell_allowed": exp_cell_allowed,
        "exp_cell_overlap_a": exp_cell_overlap,
        "d_exp_to_clash_deg": d_exp_to_edge,
        "max_overlap_a": float(overlap_map.max()),
        "min_overlap_a": float(overlap_map.min()),
    }
    keys = list(summary)
    with open(summary_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(summary[k]) for k in keys) + "\n")
    return summary


def plot_map(
    pdb_id: str,
    tau_grid: np.ndarray,
    phi_grid: np.ndarray,
    overlap_map: np.ndarray,
    tau_exp: float,
    phi_exp: float,
    tolerance_a: float,
    out_path: Path,
) -> None:
    step = float(tau_grid[1] - tau_grid[0])
    extent = (
        tau_grid[0] - step / 2,
        tau_grid[-1] + step / 2,
        phi_grid[0] - step / 2,
        phi_grid[-1] + step / 2,
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    # Clip absurd overlaps so colour scale stays informative.
    field = np.clip(overlap_map, -1.0, 3.0).T
    im = ax.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap="RdYlGn_r",
        vmin=-1.0,
        vmax=3.0,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("max steric overlap (A)\n(green = clash-free)")
    # Clash boundary at the 0.4 A tolerance.
    Tcell, Pcell = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    ax.contour(
        Tcell, Pcell, overlap_map,
        levels=[tolerance_a],
        colors="black", linewidths=1.2,
    )
    # Experimental point.
    ax.plot(
        tau_exp, phi_exp,
        marker="o", markersize=10,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="",
        label=f"experimental ({tau_exp:+.1f}, {phi_exp:+.1f})",
    )
    ax.set_xlabel(r"$\tau_{\mathrm{megley}}$ (deg)")
    ax.set_ylabel(r"$\varphi_{\mathrm{megley}}$ (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title(
        f"{pdb_id} freely-rotating chromophore (tau, phi) scan, "
        f"5 deg grid"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    cif = DATA_DIR / f"{PDB_ID}.cif"
    if not cif.is_file():
        sys.exit(f"missing {cif} - run test_1ema.py first to fetch it")
    print(f"[input] {cif}")

    loaded = load_structure(cif)
    cage = build_cage(loaded)
    tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)
    print(
        f"[load] {loaded.pdb_id}: {loaded.chrom_resname} "
        f"{loaded.chrom_chain}/{loaded.chrom_seqid}, "
        f"exp (tau, phi) = ({tau_exp:+.3f}, {phi_exp:+.3f})"
    )

    tau_grid, phi_grid, overlap_map = scan_overlap_map(
        loaded, cage, STEP_DEG
    )

    npz_path = DATA_DIR / f"scan_{PDB_ID}_free.npz"
    np.savez_compressed(
        npz_path,
        tau_grid=tau_grid,
        phi_grid=phi_grid,
        overlap_map=overlap_map,
        tau_exp=tau_exp,
        phi_exp=phi_exp,
        tolerance_a=DEFAULT_TOLERANCE_A,
        step_deg=STEP_DEG,
        pdb_id=PDB_ID,
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
