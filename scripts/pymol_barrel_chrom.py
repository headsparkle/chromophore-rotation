#!/usr/bin/env python3
"""
pymol_barrel_chrom.py
=====================
avGFP (1EMA) structural panel: upright beta-barrel (gray transparent),
chromophore two-ring system (teal ball-and-stick: 5-membered imidazolinone
ring + methine bridge + 6-membered phenol ring), and Thr203 side chain
(orange/red ball-and-stick).

Backbone stubs on the chromophore residue (CA1/CB1/CG1/OG1/N1 from Ser65
and CA3/C3/O3 from Gly67) are excluded -- only the conjugated ring atoms
C1, N2, CA2, C2, N3, O2, CB2, CG2, CD1, CD2, CE1, CE2, CZ, OH are shown.

Run:
    /tmp/bin/micromamba run -n pymol pymol -cq scripts/pymol_barrel_chrom.py

Outputs:
    figures/fig_barrel_chrom.png   (1800 x 2400 portrait, 300 dpi)
    figures/fig_barrel_chrom.pse
"""

import sys
from pathlib import Path

try:
    from pymol import cmd
except ImportError:
    sys.exit("Run via:  pymol -cq scripts/pymol_barrel_chrom.py")

PROJECT    = Path("/Users/mzim/Documents/Projects/chromophore-rotation")
FIGURE_DIR = PROJECT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Atom names that form the conjugated ring system (5-membered + bridge + phenol)
RING_ATOMS = "name C1+N2+CA2+C2+N3+O2+CB2+CG2+CD1+CD2+CE1+CE2+CZ+OH"


def make_figure():
    cmd.reinitialize()

    # ---- load
    cmd.load(str(PROJECT / "data" / "cif" / "1EMA.cif"), "gfp")
    cmd.remove("hydro")
    cmd.remove("solvent")

    # ---- selections
    cmd.select("chrom",    "gfp and resi 66  and chain A")
    cmd.select("chrom_rings", f"chrom and ({RING_ATOMS})")
    cmd.select("thr203",   "gfp and resi 203 and chain A")
    cmd.select("thr203_sc", "thr203 and (name CB or name OG1 or name CG2)")

    # ---- hide everything first
    cmd.hide("everything")
    cmd.hide("nonbonded")

    # Barrel: gray transparent cartoon
    cmd.show("cartoon", "gfp")
    cmd.color("gray85", "gfp")
    cmd.set("cartoon_transparency", 0.60)
    cmd.color("gray85", "thr203 and backbone")   # prevent orange cartoon at Thr203

    # Chromophore rings: teal ball-and-stick (ring atoms only, no backbone stubs)
    cmd.show("sticks",  "chrom_rings")
    cmd.show("spheres", "chrom_rings")
    cmd.color("teal", "chrom_rings and elem C")
    cmd.color("red",  "chrom_rings and elem O")
    cmd.color("blue", "chrom_rings and elem N")
    cmd.set("stick_radius", 0.28,  "chrom_rings")
    cmd.set("sphere_scale", 0.38,  "chrom_rings")

    # Thr203 side chain: orange ball-and-stick
    cmd.show("sticks",  "thr203_sc")
    cmd.show("spheres", "thr203_sc")
    cmd.color("tv_orange", "thr203_sc and elem C")
    cmd.color("red",       "thr203_sc and elem O")
    cmd.set("stick_radius", 0.28, "thr203_sc")
    cmd.set("sphere_scale", 0.55, "thr203_sc")

    # ---- view: barrel upright, beta-sheets filling frame
    cmd.orient("gfp")
    cmd.turn("x", 90)    # top-down -> side-on
    cmd.turn("y", 30)    # bring Thr203 toward camera
    cmd.turn("z", 90)    # stand barrel upright
    # Zoom to the beta-sheet atoms so the strands just reach the image edges
    cmd.zoom("gfp and ss s", 1)

    # ---- render settings
    cmd.bg_color("white")
    cmd.set("ray_shadows",    0)
    cmd.set("antialias",      2)
    cmd.set("ray_trace_mode", 0)
    cmd.set("ambient",        0.45)
    cmd.set("direct",         0.55)
    cmd.set("specular",       0.10)
    cmd.set("shininess",      10)
    cmd.set("depth_cue",      0)
    cmd.set("fog_start",      0.90)

    # ---- render
    out_png = str(FIGURE_DIR / "fig_barrel_chrom.png")
    out_pse = str(FIGURE_DIR / "fig_barrel_chrom.pse")
    cmd.ray(1800, 2400)
    cmd.png(out_png, dpi=300)
    print(f"PNG saved: {out_png}")
    cmd.save(out_pse)
    print(f"PSE saved: {out_pse}")


make_figure()
