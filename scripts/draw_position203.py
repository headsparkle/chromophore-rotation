#!/usr/bin/env python3
"""
draw_position203.py
===================
Generates figures/fig_position203_gatekeeper.png using matplotlib only.
No PyMOL or other molecular viewer required.

View: looking along Y axis (screen_x = world_x, screen_y = world_z).
In 1EMA, the chromophore ring sits at X=29-33 and Thr203 at X=34-37,
so this view separates them clearly in the horizontal direction.

Run from the project root:
    python3 scripts/draw_position203.py
"""

import math
from pathlib import Path

import gemmi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

PROJECT  = Path(__file__).resolve().parent.parent
CIF_PATH = PROJECT / "data" / "cif" / "1EMA.cif"
OUT_PNG  = PROJECT / "figures" / "fig_position203_gatekeeper.png"
OUT_PNG.parent.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_CHROM  = "#007E74"   # teal for chromophore carbons
C_THR    = "#E07800"   # orange for Thr203 carbons
COL_N    = "#3333BB"
COL_O    = "#BB2222"
ARC_COL  = "#C03800"

BOND_W_CHROM = 3.5
BOND_W_THR   = 3.0
BOND_MAX     = 1.72    # Angstrom covalent bond detection
WATER_NAMES  = {"HOH", "WAT", "DOD"}

# ---------------------------------------------------------------------------
# Load 1EMA
# ---------------------------------------------------------------------------
st    = gemmi.read_structure(str(CIF_PATH))
model = st[0]

chrom_atoms, chrom_elem = {}, {}
thr_atoms,   thr_elem   = {}, {}

for chain in model:
    for res in chain:
        if chain.name == "A" and res.seqid.num == 66:
            for a in res:
                if not a.element.is_hydrogen:
                    chrom_atoms[a.name] = np.array([a.pos.x, a.pos.y, a.pos.z])
                    chrom_elem[a.name]  = a.element.name.upper()
        elif chain.name == "A" and res.seqid.num == 203:
            for a in res:
                if not a.element.is_hydrogen:
                    thr_atoms[a.name] = np.array([a.pos.x, a.pos.y, a.pos.z])
                    thr_elem[a.name]  = a.element.name.upper()

# ---------------------------------------------------------------------------
# Bond detection by distance
# ---------------------------------------------------------------------------
def bonds_from_coords(atoms: dict) -> list:
    names  = list(atoms.keys())
    coords = np.array([atoms[n] for n in names])
    pairs  = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.linalg.norm(coords[i] - coords[j]) < BOND_MAX:
                pairs.append((names[i], names[j]))
    return pairs

chrom_bonds = bonds_from_coords(chrom_atoms)
thr_bonds   = bonds_from_coords(thr_atoms)

# ---------------------------------------------------------------------------
# Projection: view along +Y, so screen = (X, Z)
# ---------------------------------------------------------------------------
def project(xyz: np.ndarray) -> np.ndarray:
    """Project 3D point to 2D screen (X, Z) coordinates."""
    return np.array([xyz[0], xyz[2]])

# Key atom positions
cb2  = chrom_atoms["CB2"]
cg2  = chrom_atoms["CG2"]
oh   = chrom_atoms["OH"]
og1  = thr_atoms["OG1"]

# Verify separation: chromophore and Thr203 should be well-separated in X
print(f"OH  screen position:  {project(oh)}")
print(f"OG1 screen position:  {project(og1)}")
print(f"OG1-OH 3D distance:   {np.linalg.norm(og1-oh):.2f} A")

# ---------------------------------------------------------------------------
# P-bond rotation arc
# Arc: sweep OH around the CB2-CG2 (P-bond) axis toward OG1.
# ---------------------------------------------------------------------------
p_axis = cg2 - cb2
p_axis = p_axis / np.linalg.norm(p_axis)

# Center of arc = foot of perpendicular from OH to the P-bond axis
proj_l  = np.dot(oh - cb2, p_axis)
arc_ctr = cb2 + proj_l * p_axis
radial  = oh - arc_ctr
arc_r   = np.linalg.norm(radial)
rad_u   = radial / arc_r
perp    = np.cross(p_axis, rad_u)   # tangent direction at angle=0

# Choose sweep direction that brings OH closer to OG1
test_pos = arc_ctr + arc_r * (math.cos(math.radians( 20)) * rad_u
                               + math.sin(math.radians( 20)) * perp)
test_neg = arc_ctr + arc_r * (math.cos(math.radians(-20)) * rad_u
                               + math.sin(math.radians(-20)) * perp)
sign = -1 if np.linalg.norm(test_neg - og1) < np.linalg.norm(test_pos - og1) else 1

ARC_DEG  = 72
n_arc    = 200
arc_angles  = np.linspace(0, math.radians(sign * ARC_DEG), n_arc)
arc_pts_3d  = np.array([
    arc_ctr + arc_r * (math.cos(a) * rad_u + math.sin(a) * perp)
    for a in arc_angles])
arc_pts_2d  = np.array([project(p) for p in arc_pts_3d])

# Tangent at arc tip (for arrowhead)
end_a  = math.radians(sign * ARC_DEG)
tang3d = sign * (-math.sin(end_a) * rad_u + math.cos(end_a) * perp)
tang2d = project(arc_ctr + tang3d) - project(arc_ctr)
tang2d = tang2d / np.linalg.norm(tang2d)

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_aspect("equal")
ax.axis("off")

def atom_col(elem, c_default):
    if elem == "N":
        return COL_N
    if elem == "O":
        return COL_O
    return c_default

# --- chromophore bonds (half-bond coloring)
for a1, a2 in chrom_bonds:
    p1 = project(chrom_atoms[a1])
    p2 = project(chrom_atoms[a2])
    mid = (p1 + p2) / 2
    ax.plot([p1[0], mid[0]], [p1[1], mid[1]],
            color=atom_col(chrom_elem[a1], C_CHROM),
            lw=BOND_W_CHROM, solid_capstyle="round", zorder=3)
    ax.plot([mid[0], p2[0]], [mid[1], p2[1]],
            color=atom_col(chrom_elem[a2], C_CHROM),
            lw=BOND_W_CHROM, solid_capstyle="round", zorder=3)

# chromophore heteroatom spheres
for name, xyz in chrom_atoms.items():
    elem = chrom_elem[name]
    if elem in ("N", "O"):
        p = project(xyz)
        ax.plot(p[0], p[1], "o",
                color=atom_col(elem, C_CHROM),
                ms=8, mec="white", mew=0.8, zorder=4)

# --- Thr203 bonds
for a1, a2 in thr_bonds:
    p1 = project(thr_atoms[a1])
    p2 = project(thr_atoms[a2])
    mid = (p1 + p2) / 2
    ax.plot([p1[0], mid[0]], [p1[1], mid[1]],
            color=atom_col(thr_elem[a1], C_THR),
            lw=BOND_W_THR, solid_capstyle="round", zorder=3)
    ax.plot([mid[0], p2[0]], [mid[1], p2[1]],
            color=atom_col(thr_elem[a2], C_THR),
            lw=BOND_W_THR, solid_capstyle="round", zorder=3)

# Thr203 heteroatom spheres; OG1 highlighted larger
for name, xyz in thr_atoms.items():
    elem = thr_elem[name]
    p = project(xyz)
    if name == "OG1":
        ax.plot(p[0], p[1], "o", color=C_THR,
                ms=16, mec="#804000", mew=1.5, zorder=5)
    elif elem in ("N", "O"):
        ax.plot(p[0], p[1], "o",
                color=atom_col(elem, C_THR),
                ms=8, mec="white", mew=0.8, zorder=4)

# --- dashed contact line OG1 -- OH
p_og1_2d = project(og1)
p_oh_2d  = project(oh)
ax.plot([p_og1_2d[0], p_oh_2d[0]], [p_og1_2d[1], p_oh_2d[1]],
        color="#996600", lw=1.8, ls=(0, (5, 3)),
        alpha=0.90, zorder=6, solid_capstyle="round")

dist_og1_oh = np.linalg.norm(og1 - oh)
mid_contact = (p_og1_2d + p_oh_2d) / 2
ax.text(mid_contact[0], mid_contact[1] + 0.22,
        f"{dist_og1_oh:.2f} Å",
        fontsize=10, color="#805000", ha="center", va="bottom",
        fontweight="bold", zorder=7)

# --- P-bond rotation arc
ax.plot(arc_pts_2d[:, 0], arc_pts_2d[:, 1],
        color=ARC_COL, lw=2.8, zorder=7, solid_capstyle="round")

# Arrowhead at tip
tip = arc_pts_2d[-1]
ax.annotate("", xy=tip, xytext=tip - 0.0015 * tang2d,
            arrowprops=dict(arrowstyle="-|>",
                            color=ARC_COL,
                            lw=2.2,
                            mutation_scale=20),
            zorder=8)

# Label: P-bond rotation -- place to the left of arc midpoint
arc_mid_2d = arc_pts_2d[n_arc // 2]
# offset away from the chromophore (downward in this view)
ax.text(arc_mid_2d[0] - 0.80, arc_mid_2d[1] - 0.35,
        "P-bond\nrotation",
        fontsize=10, color=ARC_COL,
        ha="center", va="top", fontstyle="italic", fontweight="bold",
        zorder=9)

# --- labels
# OG1 label: to the right and above
ax.text(p_og1_2d[0] + 0.25, p_og1_2d[1] + 0.20,
        "Thr203-OG1\n(gatekeeper)",
        fontsize=11, color=C_THR, fontweight="bold",
        va="bottom", ha="left", zorder=9)

# Chromophore OH label: below and left of OH
ax.text(p_oh_2d[0] - 0.20, p_oh_2d[1] - 0.20,
        "Chrom. OH",
        fontsize=9.5, color=COL_O,
        va="top", ha="right", zorder=9)

# Label the phenol ring centre
ring_ctr = project((chrom_atoms["CE1"] + chrom_atoms["CE2"] +
                     chrom_atoms["CZ"]  + chrom_atoms["CD1"] +
                     chrom_atoms["CD2"] + chrom_atoms["CG2"]) / 6)
ax.text(ring_ctr[0], ring_ctr[1] + 0.30,
        "Phenol\nring",
        fontsize=8.5, color=C_CHROM, ha="center", va="bottom",
        alpha=0.85, zorder=9)

# --- legend
handles = [
    mpatches.Patch(color=C_CHROM, label="Chromophore (CRO, res. 66)"),
    mpatches.Patch(color=C_THR,   label="Thr203 (T203)"),
    mpatches.Patch(color=ARC_COL, label="P-bond rotation arc"),
    plt.Line2D([0], [0], color="#996600", lw=2, ls="--",
               label=f"OG1···OH contact ({dist_og1_oh:.2f} Å)"),
]
ax.legend(handles=handles, loc="upper left",
          fontsize=9, framealpha=0.90, edgecolor="gray",
          handlelength=1.8)

ax.set_title("Position 203 as P-bond gatekeeper — avGFP (1EMA)\n"
             "Thr203-OG1 directly contacts the chromophore phenol-OH",
             fontsize=11, pad=10)

# Tight layout with some padding
ax.autoscale_view()
ax.margins(0.18)
plt.tight_layout()
fig.savefig(str(OUT_PNG), dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
