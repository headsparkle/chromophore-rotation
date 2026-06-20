#!/usr/bin/env python3
"""
make_figS4_panel.py
===================

Assemble Figure S4: a 2x2 panel of the relaxed (cage-breathing) energy
surfaces for the four baseline FPs, from the per-structure grids written by
relaxed_energy_scan.py (data/relaxed_scans/relaxed_<PDB>.npz).

Output: figures/relaxed_dE_panel_S4.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SCANS = ROOT / "data" / "relaxed_scans"
OUT = ROOT / "figures" / "relaxed_dE_panel_S4.png"
TOL = 0.4

PANEL = [
    ("1EMA", "avGFP (green, CRO)"),
    ("5LK4", "mScarlet (bright red, NRQ)"),
    ("3KCS", "mCherry (dim/twisted red, NRQ)"),
    ("2YE0", "mTurquoise (cyan, SWG)"),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    im = None
    for ax, (pid, lab) in zip(axes.ravel(), PANEL):
        z = np.load(SCANS / f"relaxed_{pid}.npz")
        taus, phis, dE = z["tau_grid"], z["phi_grid"], z["dE"]
        ov_rigid, ov_relax = z["overlap_rigid"], z["overlap_relaxed"]
        extent = [phis[0], phis[-1], taus[0], taus[-1]]
        im = ax.imshow(np.minimum(dE, 25.0), origin="lower", extent=extent,
                       aspect="auto", cmap="viridis_r", vmin=0, vmax=25)
        P, T = np.meshgrid(phis, taus)
        ax.contour(P, T, (ov_rigid <= TOL).astype(float), levels=[0.5],
                   colors="white", linewidths=1.2)
        ax.contour(P, T, (ov_relax <= TOL).astype(float), levels=[0.5],
                   colors="cyan", linewidths=1.0, linestyles="--")
        ax.plot(float(z["phi_exp"]), float(z["tau_exp"]), "r*", ms=15, mec="k")
        ax.set_title(f"{pid}: {lab}", fontsize=10)
        ax.set_xlabel(r"$\varphi$ (P-bond, deg)")
        ax.set_ylabel(r"$\tau$ (I-bond, deg)")
    fig.subplots_adjust(right=0.88, hspace=0.32, wspace=0.25)
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax).set_label(
        r"relaxed $\Delta E$ (kcal/mol, capped at 25)")
    fig.suptitle("Relaxed (cage-breathing) energy surfaces. White = rigid-"
                 "allowed, cyan dashed = relaxed-allowed, star = deposited.",
                 fontsize=10, y=0.995)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
