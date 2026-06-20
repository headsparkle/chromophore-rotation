#!/usr/bin/env python3
"""
compare_1gfl_literature.py
==========================

Overlay literature predictions of (tau, phi) chromophore freedom in
wt GFP (1GFL) on top of our rigid-cage steric scan, and quantify how
much accessibility the rigid map gives in each direction.

Sources we are comparing against
--------------------------------
1) Baffour-Awuah & Zimmer 2004, Chem. Phys. 303, 7-11.
   Constrained AMBER*/MacroModel minimisation on 1GFL. Energy minimum
   sits at +tau, +phi ~ (+45, +45) deg (positively correlated hula
   twist). Plot as a single marker at (+45, +45).

2) Megley, Dickson, Maddalo, Chandler & Zimmer 2009,
   J. Phys. Chem. B 113, 302-308.
   Freely-rotating MD on the same 1GFL chain A, GFP-A form. From
   Table 1 of that paper:
       slope  = -0.857
       intercept = -8.75 deg
       tau_av = +5.62, phi_av = -13.57 deg   (cloud centre)
       convex hull area = 2575 deg^2
       80%-area = 627 deg^2
   The cloud is elongated along slope = -0.857 (a negatively correlated
   hula twist, "bottom HT"). We draw the best-fit line and an ellipse
   whose area equals the published 80%-area (627 deg^2), oriented along
   the slope-(-0.857) major axis. The semi-axes (22, 9) were picked to
   match Fig. 4 of the 2009 paper and give pi * 22 * 9 = 622 ~= 627.

3) Our 1GFL rigid steric scan, this project. The allowed region from
   scan_1GFL_free.npz is drawn as a filled contour with the 0.4 A
   tolerance edge in black.

Directional accessibility
-------------------------
For each of three direction unit vectors -
    positive HT  : ( cos45,  sin45)
    negative HT  : ( cos45, -sin45)
    phi-axis OBF : (0, 1)
we project the centres of all clash-free grid cells onto the vector,
mean-centre the projections, and report the standard deviation in
degrees. The bigger the deviation along a direction, the further the
chromophore can move along that direction without clashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

PDB_ID = "1GFL"
SCAN_NPZ = DATA_DIR / f"scan_{PDB_ID}_free.npz"

# 2004 Baffour-Awuah & Zimmer: positive HT minimum
POS_HT_2004 = (+45.0, +45.0)

# 2009 Megley et al. GFP-A (Table 1)
SLOPE_2009 = -0.857
INTERCEPT_2009 = -8.75
TAU_AV_2009 = 5.62
PHI_AV_2009 = -13.57
HULL_2009 = 2575.0       # deg^2, convex hull
AREA80_2009 = 627.0      # deg^2, 80%-area
ELLIPSE_SEMI = (22.0, 9.0)   # picked to match Fig. 4 + give area ~627


def main() -> int:
    if not SCAN_NPZ.is_file():
        sys.exit(f"missing {SCAN_NPZ} - run scan_1gfl.py first")
    z = np.load(SCAN_NPZ)
    tau_grid = z["tau_grid"]
    phi_grid = z["phi_grid"]
    overlap = z["overlap_map"]            # (n_tau, n_phi)
    tau_exp = float(z["tau_exp"])
    phi_exp = float(z["phi_exp"])
    tol = float(z["tolerance_a"])
    step = float(z["step_deg"])

    allowed = overlap <= tol
    cell = step * step
    A_allowed = float(allowed.sum()) * cell
    print(f"[load] 1GFL allowed = {allowed.sum()} cells = {A_allowed:.0f} deg^2")
    print(f"[load] experimental (tau, phi) = ({tau_exp:+.2f}, {phi_exp:+.2f})")

    # ------------------------------------------------------------------
    # Figure: scan map + literature overlay
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    extent = (
        tau_grid[0] - step / 2,
        tau_grid[-1] + step / 2,
        phi_grid[0] - step / 2,
        phi_grid[-1] + step / 2,
    )
    # Greyscale overlap so the literature markers pop.
    field = np.clip(overlap, -1.0, 3.0).T
    ax.imshow(
        field, origin="lower", extent=extent, cmap="Greys",
        vmin=-1.0, vmax=3.0, aspect="equal", interpolation="nearest",
        alpha=0.6,
    )
    # Allowed region filled green.
    ax.contourf(
        tau_grid, phi_grid, allowed.T.astype(float),
        levels=[0.5, 1.5], colors=["#5fbf5f"], alpha=0.55,
    )
    ax.contour(
        tau_grid, phi_grid, overlap.T,
        levels=[tol], colors="#1f8a1f", linewidths=1.6,
    )

    # 2009 best-fit line, drawn across the full range it could span.
    tau_line = np.linspace(-90, 90, 200)
    phi_line = SLOPE_2009 * tau_line + INTERCEPT_2009
    ax.plot(
        tau_line, phi_line, color="#1f4ea0", linewidth=2.0,
        linestyle="--",
        label=(
            r"2009 MD GFP-A fit: $\varphi = -0.857\,\tau - 8.75$"
            "\n(negative-HT, bottom-HT)"
        ),
    )

    # 2009 80%-area envelope as an ellipse, slope-aligned.
    angle_deg = float(np.degrees(np.arctan(SLOPE_2009)))
    ell = Ellipse(
        xy=(TAU_AV_2009, PHI_AV_2009),
        width=2 * ELLIPSE_SEMI[0], height=2 * ELLIPSE_SEMI[1],
        angle=angle_deg, facecolor="#1f4ea0", alpha=0.20,
        edgecolor="#1f4ea0", linewidth=1.5,
        label="2009 MD 80%-area (627 deg^2)",
    )
    ax.add_patch(ell)

    # 2004 positive-HT minimum.
    ax.plot(
        *POS_HT_2004, marker="*", markersize=22,
        markerfacecolor="#d04020", markeredgecolor="black",
        markeredgewidth=1.0, linestyle="",
        label="2004 MM minimum (+45, +45)\n(positive-HT)",
    )

    # Experimental point.
    ax.plot(
        tau_exp, phi_exp, marker="o", markersize=11,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.5, linestyle="",
        label=f"1GFL experimental ({tau_exp:+.1f}, {phi_exp:+.1f})",
    )

    # Axes around the chromophore-twisting region of interest.
    ax.set_xlim(-90, 90); ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="k", linewidth=0.4, alpha=0.4)
    ax.set_xlabel(r"$\tau_{\mathrm{megley}}$ (deg)  [I-bond, methine-imidazolinone]")
    ax.set_ylabel(r"$\varphi_{\mathrm{megley}}$ (deg)  [P-bond, methine-phenol]")
    ax.set_title(
        "1GFL chromophore (tau, phi) freedom: rigid scan vs. literature\n"
        f"green = rigid-cage clash-free ({A_allowed:.0f} deg^2)"
    )
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "scan_1GFL_lit_overlay.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[save] {out}")

    # ------------------------------------------------------------------
    # Directional accessibility quantification
    # ------------------------------------------------------------------
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    Ta = T[allowed]
    Pa = P[allowed]
    # Centre at the allowed-cloud centroid so spread reflects shape, not
    # offset.
    tau_c = float(Ta.mean())
    phi_c = float(Pa.mean())
    Tc = Ta - tau_c
    Pc = Pa - phi_c

    directions = {
        "positive-HT (+1 diag)":  (np.cos(np.deg2rad(+45)),  np.sin(np.deg2rad(+45))),
        "negative-HT (-1 diag)":  (np.cos(np.deg2rad(-45)),  np.sin(np.deg2rad(-45))),
        "phi-axis OBF (P-spin)":  (0.0, 1.0),
        "tau-axis OBF (I-swing)": (1.0, 0.0),
    }

    rows = []
    print("\n[directional accessibility around allowed-cloud centroid "
          f"({tau_c:+.2f}, {phi_c:+.2f})]")
    print(f"{'direction':<28s} {'sigma (deg)':>12s} {'range (deg)':>12s}")
    for name, (ux, uy) in directions.items():
        proj = Tc * ux + Pc * uy
        sigma = float(proj.std())
        ptp = float(proj.max() - proj.min())
        rows.append((name, ux, uy, sigma, ptp))
        print(f"{name:<28s} {sigma:>12.2f} {ptp:>12.2f}")

    # Wedge counts: how many allowed cells lie in a +-15 deg wedge
    # around each direction axis (bidirectional: a cell at +-180 deg
    # from the direction is also "along" it).
    ang = np.degrees(np.arctan2(Pc, Tc))
    half = 15.0
    print(f"\n[wedge accessibility, +-{half:.0f} deg cone around each direction]")
    print(f"{'direction':<28s} {'cells in wedge':>16s} {'deg^2':>10s}")
    for name, (ux, uy) in directions.items():
        dir_deg = np.degrees(np.arctan2(uy, ux))
        diff = (ang - dir_deg) % 180.0          # 0..180
        axial = np.where(diff > 90.0, 180.0 - diff, diff)  # 0..90
        mask = axial <= half
        n = int(mask.sum())
        rows.append((f"{name} (wedge n)", ux, uy, n, n * cell))
        print(f"{name:<28s} {n:>16d} {n*cell:>10.0f}")

    # Save numbers to CSV.
    out_csv = DATA_DIR / "scan_1GFL_directional.csv"
    with open(out_csv, "w") as f:
        f.write("metric,direction,ux,uy,value,units\n")
        # Re-emit the rows with units.
        for name, (ux, uy) in directions.items():
            proj = Tc * ux + Pc * uy
            f.write(f"sigma,{name},{ux:.4f},{uy:.4f},"
                    f"{proj.std():.4f},deg\n")
            f.write(f"range,{name},{ux:.4f},{uy:.4f},"
                    f"{(proj.max()-proj.min()):.4f},deg\n")
        for name, (ux, uy) in directions.items():
            dir_deg = np.degrees(np.arctan2(uy, ux))
            diff = (ang - dir_deg) % 180.0
            axial = np.where(diff > 90.0, 180.0 - diff, diff)
            mask = axial <= half
            f.write(f"wedge_cells_+-15deg,{name},{ux:.4f},{uy:.4f},"
                    f"{int(mask.sum())},count\n")
            f.write(f"wedge_area_deg2_+-15deg,{name},{ux:.4f},{uy:.4f},"
                    f"{int(mask.sum())*cell:.2f},deg2\n")
        f.write(f"allowed_centroid,tau_phi,,,{tau_c:.4f};{phi_c:.4f},deg\n")
        f.write(f"allowed_area,,,,{A_allowed:.2f},deg2\n")
        f.write(f"hull_2009_GFP_A,,,,{HULL_2009:.2f},deg2\n")
        f.write(f"area80_2009_GFP_A,,,,{AREA80_2009:.2f},deg2\n")
    print(f"\n[save] {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
