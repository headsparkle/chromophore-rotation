#!/usr/bin/env python3
"""
make_envelope.py  -  (tau, phi) sterically-accessible-space overlay for Figure 5.

Reuses the project's scan core (scan/barrel.py, scan/rotate.py) to reproduce,
for one structure, the same rigid (tau, phi) map shown in Figure 2: at every
5 deg grid point the chromophore is rebuilt and scored for steric overlap with
the barrel; cells with overlap <= 0.4 A (the MolProbity tolerance) are the
sterically allowed region.

Writes, for each input CIF:
    envelope_<pdbid>.png    heat map + allowed-region outline + deposited point
    envelope_<pdbid>.npz    tau_grid, phi_grid, overlap_map, tau_exp, phi_exp

This is pure NumPy (no GPU); it runs in a few seconds per structure.

Usage:
    python make_envelope.py structures/7y40.cif structures/6nqk.cif
    python make_envelope.py --step 5 structures/*.cif
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scan"))

from barrel import DEFAULT_TOLERANCE_A, build_cage, load_structure, max_overlap  # noqa: E402
from rotate import measure_megley, set_megley                                    # noqa: E402


def grid(step: float) -> np.ndarray:
    return np.arange(-180.0, 180.0, step)


def scan_overlap_map(loaded, cage, step_deg: float):
    tau_grid = grid(step_deg)
    phi_grid = grid(step_deg)
    omap = np.empty((len(tau_grid), len(phi_grid)), dtype=float)
    base = loaded.chrom_atoms
    for i, tau in enumerate(tau_grid):
        for j, phi in enumerate(phi_grid):
            atoms_now = set_megley(base, float(tau), float(phi))
            mxyz = np.array([atoms_now[n] for n in cage.moving_names], dtype=float)
            ov, _ = max_overlap(mxyz, cage.moving_radii, cage.cage_xyz,
                                cage.cage_radii, cage.exclude)
            omap[i, j] = ov
    return tau_grid, phi_grid, omap


def plot_envelope(pdb_id, tau_grid, phi_grid, omap, tau_exp, phi_exp,
                  tol, out_png):
    step = float(tau_grid[1] - tau_grid[0])
    extent = (tau_grid[0] - step / 2, tau_grid[-1] + step / 2,
              phi_grid[0] - step / 2, phi_grid[-1] + step / 2)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    field = np.clip(omap, -1.0, 3.0).T
    im = ax.imshow(field, origin="lower", extent=extent, cmap="RdYlGn_r",
                   vmin=-1.0, vmax=3.0, aspect="equal", interpolation="nearest")
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    ax.contour(T, P, omap, levels=[tol], colors="black", linewidths=1.3)
    ax.plot(tau_exp, phi_exp, marker="o", ms=9, mfc="white",
            mec="black", mew=1.5, ls="")
    for sx in (0.0, 180.0, -180.0):
        for sy in (0.0, 180.0, -180.0):
            ax.plot(sx, sy, marker="*", ms=9, color="black", ls="")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(r"$\tau$ (I-bond), deg")
    ax.set_ylabel(r"$\varphi$ (P-bond), deg")
    n_allowed = int((omap <= tol).sum())
    f_allowed = n_allowed * step * step / (360.0 * 360.0)
    ax.set_title("%s   f_allowed = %.4f" % (pdb_id, f_allowed), fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("max overlap (A); green = clash-free", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return f_allowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cifs", nargs="+", help="one or more structure .cif files")
    ap.add_argument("--step", type=float, default=5.0, help="grid step (deg)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOLERANCE_A,
                    help="clash tolerance (A); 0.4 = MolProbity (default)")
    ap.add_argument("--outdir", default=".", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for cif in args.cifs:
        cif = Path(cif)
        loaded = load_structure(cif)
        cage = build_cage(loaded)
        tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)
        pdb_id = getattr(loaded, "pdb_id", None) or cif.stem.upper()
        print("[%s] scanning at %.1f deg ..." % (pdb_id, args.step))
        t0 = time.perf_counter()
        tau_g, phi_g, omap = scan_overlap_map(loaded, cage, args.step)
        print("   %d rotamers in %.1f s" % (omap.size, time.perf_counter() - t0))
        png = outdir / ("envelope_%s.png" % pdb_id.lower())
        npz = outdir / ("envelope_%s.npz" % pdb_id.lower())
        f = plot_envelope(pdb_id, tau_g, phi_g, omap, tau_exp, phi_exp,
                          args.tol, str(png))
        np.savez(npz, tau_grid=tau_g, phi_grid=phi_g, overlap_map=omap,
                 tau_exp=tau_exp, phi_exp=phi_exp, tolerance_a=args.tol)
        print("   f_allowed = %.4f  ->  %s , %s" % (f, png.name, npz.name))


if __name__ == "__main__":
    main()
