#!/usr/bin/env python3
"""
pymol_position203.py
====================
Structural panel for avGFP (1EMA): whole beta-barrel (gray transparent
cartoon), chromophore (teal ball-and-stick), and Thr203 side chain
(orange ball-and-stick, labeled).

Run:
    /tmp/bin/micromamba run -n pymol pymol -cq scripts/pymol_position203.py

Outputs:
    figures/fig_position203_gatekeeper.png   (2400 x 1800, 300 dpi)
    figures/fig_position203_gatekeeper.pse   (session for manual view tuning)
"""

import sys
from pathlib import Path

try:
    from pymol import cmd
except ImportError:
    sys.exit("Run via:  pymol -cq scripts/pymol_position203.py")

PROJECT    = Path("/Users/mzim/Documents/Projects/chromophore-rotation")
FIGURE_DIR = PROJECT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def make_figure():
    cmd.reinitialize()

    # ---- load structure
    cmd.load(str(PROJECT / "data" / "cif" / "1EMA.cif"), "gfp")
    cmd.remove("hydro")
    cmd.remove("solvent")

    # ---- named selections
    cmd.select("chrom",  "gfp and resi 66  and chain A")
    cmd.select("thr203", "gfp and resi 203 and chain A")

    # ---- representations
    cmd.hide("everything")
    cmd.hide("nonbonded")

    # Whole barrel: gray transparent cartoon
    cmd.show("cartoon", "gfp")
    cmd.color("gray85", "gfp")
    cmd.set("cartoon_transparency", 0.60)
    # Keep Thr203 backbone gray so it doesn't color the cartoon orange
    cmd.color("gray85", "thr203 and backbone")

    # Chromophore: teal ball-and-stick, thick enough to read at whole-barrel scale
    cmd.show("sticks", "chrom")
    cmd.color("teal", "chrom and elem C")
    cmd.color("red",  "chrom and elem O")
    cmd.color("blue", "chrom and elem N")
    cmd.set("stick_radius", 0.30, "chrom")
    cmd.show("spheres", "chrom")
    cmd.set("sphere_scale", 0.40, "chrom")

    # Thr203 side chain (CB, OG1, CG2): orange ball-and-stick
    cmd.select("thr203_sc", "thr203 and (name CB or name OG1 or name CG2)")
    cmd.show("sticks",  "thr203_sc")
    cmd.show("spheres", "thr203_sc")
    cmd.color("tv_orange", "thr203_sc and elem C")
    cmd.color("red",       "thr203_sc and elem O")
    cmd.set("stick_radius", 0.28,  "thr203_sc")
    cmd.set("sphere_scale", 0.65,  "thr203_sc")

    # ---- view: whole barrel, side-on so the cylinder is visible
    # orient("gfp") places the barrel axis along Z (top-down view).
    # Two 90-deg turns bring the barrel axis horizontal (side view).
    cmd.orient("gfp")
    cmd.turn("x", 90)    # tilt from top-down to side-on
    cmd.turn("y", 30)    # rotate to bring Thr203 toward camera
    cmd.turn("z", 90)    # stand the barrel upright
    cmd.zoom("gfp", 1)   # zoom in

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

    # ---- render and save
    out_png = str(FIGURE_DIR / "fig_position203_gatekeeper.png")
    out_pse = str(FIGURE_DIR / "fig_position203_gatekeeper.pse")
    cmd.ray(1800, 2400)
    cmd.png(out_png, dpi=300)
    print(f"PNG saved: {out_png}")
    cmd.save(out_pse)
    print(f"PSE saved: {out_pse}")


make_figure()
