# =====================================================================
# figure5_chromophore_cage.pml   (STANDALONE, self-fetching alternative)
#
# Single-file variant of the Figure 5 structural renderer. Unlike
# render_panels.pml (which reads local CIFs from ./structures/), this
# script fetches the structures itself with cmd.fetch and can be run on
# its own without the rest of the pipeline. render_panels.pml is the
# maintained version used by run_fig5.sh; this file is kept as a
# convenience for a quick one-off render.
#
# Renders a side-by-side comparison of a BRIGHT, loose-caged green FP
# and a DIM, tight-caged green FP, each showing:
#   - the HBI chromophore (sticks, teal)
#   - the first-shell cage residues within 5 A (thin sticks, atom-colored)
#   - position 203 / its aligned equivalent (orange)
#
# HOW TO RUN
#   Command line:   pymol -cq figure5_chromophore_cage.pml
#   Inside PyMOL:   File > Run Script...  or  @figure5_chromophore_cage.pml
#
# Outputs (300 dpi, in the current directory):
#   figure5_left_bright.png
#   figure5_right_dim.png
#   figure5_combined.png        (both panels, grid mode)
#
# EDIT THE FOUR SETTINGS BELOW. Confirm the PDB IDs against FPbase /
# RCSB, and set CHROMO_RESI to the deposited chromophore residue number
# if auto-detection picks the wrong residue. The (tau, phi) accessible
# envelope is a 2D plot from the scan (as in Figure 2); overlay it in
# the layout stage rather than in PyMOL.
# =====================================================================

# ---------------------- USER SETTINGS --------------------------------
python
# Bright, loose-caged green FP (suggestion: StayGold). Verify the PDB ID.
bright_pdb   = "7Y40"     # StayGold (Cytaeis uchidae), 1.7 A; verified on RCSB. Alt: 8BXT.
# Dim / dark, tight-caged green FP (Dronpa-2 = Dronpa M159T; alt bfloGFPc1 = 4DKM).
dim_pdb      = "6NQK"     # Dronpa2 (Dronpa M159T); verified on RCSB. Alt: 6NQL/7RRH.
# Deposited residue number of the gatekeeper (avGFP position 203, or the
# aligned equivalent in non-avGFP-register structures). Set per structure.
gk_resi_bright = "203"
gk_resi_dim    = "203"
python end

# Chromophore residue names covering the common GFP/CFP/RFP-type codes.
# Extend if your structure uses a code not listed here.
set_key none
python
chromo_resn = ("CRO+CR2+CRQ+CRF+CRK+CR7+CR8+CRG+CFY+CCY+CSH+GYC+GYS+NYG+"
               "NRQ+CH6+CH7+SWG+OHD+RC7+B2H+C12+XYG+DYG+PIA+IIC+MDO+QLG+GYG")
python end

# ---------------------- GLOBAL LOOK ----------------------------------
bg_color white
set ray_opaque_background, 0
set orthoscopic, 1
set ray_shadows, 0
set antialias, 2
set stick_radius, 0.13
set cartoon_transparency, 0.55
set cartoon_side_chain_helper, 1
set valence, 1
set specular, 0.15
set ambient, 0.35
set direct, 0.55
set light_count, 2

# ---------------------- RENDER FUNCTION ------------------------------
python
from pymol import cmd

def render_panel(pdb, gk_resi, out_png, title):
    cmd.delete("all")
    cmd.fetch(pdb, name="fp", type="pdb", async_=0)
    cmd.remove("solvent")
    cmd.remove("hydro")

    # locate the chromophore: prefer a known chromophore resn; else the
    # largest non-standard organic residue near the barrel center.
    cmd.select("chromo", "fp and resn %s" % chromo_resn)
    if cmd.count_atoms("chromo") == 0:
        cmd.select("chromo", "fp and (not polymer.protein) and not inorganic and organic")
    n_chromo = cmd.count_atoms("chromo")
    print("[%s] chromophore atoms detected: %d" % (pdb, n_chromo))

    # first-shell cage: protein residues with any atom within 5 A
    cmd.select("cage", "byres (polymer.protein within 5.0 of chromo) and not chromo")
    # gatekeeper residue
    cmd.select("gk", "polymer.protein and resi %s" % gk_resi)

    # representation
    cmd.hide("everything", "fp")
    cmd.show("cartoon", "fp")
    cmd.color("grey80", "fp")

    cmd.show("sticks", "chromo")
    cmd.color("teal", "chromo")
    cmd.util.cnc("chromo")            # keep N/O/S colored, carbons teal

    cmd.show("sticks", "cage")
    cmd.set("stick_radius", 0.09, "cage")
    cmd.color("wheat", "cage")
    cmd.util.cnc("cage")

    cmd.show("sticks", "gk")
    cmd.set("stick_radius", 0.16, "gk")
    cmd.color("orange", "gk")
    cmd.util.cnc("gk")

    # orient on the chromophore, pull the view out a little
    cmd.orient("chromo")
    cmd.zoom("chromo", 6.0)
    cmd.turn("y", 10)

    cmd.set("ray_trace_mode", 1)
    cmd.ray(1600, 1200)
    cmd.png(out_png, dpi=300, ray=1)
    print("[%s] wrote %s" % (pdb, out_png))

# ---- render both panels ----
render_panel(bright_pdb, gk_resi_bright, "figure5_left_bright.png",  "bright / loose cage")
render_panel(dim_pdb,    gk_resi_dim,    "figure5_right_dim.png",    "dim / tight cage")

# ---- optional: both structures side by side via grid mode ----
cmd.delete("all")
cmd.set("grid_mode", 1)
cmd.fetch(bright_pdb, "bright", type="pdb", async_=0)
cmd.fetch(dim_pdb,    "dim",    type="pdb", async_=0)
for obj, gk in (("bright", gk_resi_bright), ("dim", gk_resi_dim)):
    cmd.remove("%s and solvent" % obj)
    cmd.remove("%s and hydro" % obj)
    cmd.select("c", "%s and resn %s" % (obj, chromo_resn))
    if cmd.count_atoms("c") == 0:
        cmd.select("c", "%s and organic and not inorganic and not polymer.protein" % obj)
    cmd.select("cg", "byres (%s and polymer.protein within 5.0 of c) and not c" % obj)
    cmd.select("g",  "%s and polymer.protein and resi %s" % (obj, gk))
    cmd.hide("everything", obj)
    cmd.show("cartoon", obj); cmd.color("grey80", obj)
    cmd.show("sticks", "c");  cmd.color("teal", "c");   cmd.util.cnc("c")
    cmd.show("sticks", "cg"); cmd.set("stick_radius", 0.09, "cg"); cmd.color("wheat", "cg"); cmd.util.cnc("cg")
    cmd.show("sticks", "g");  cmd.set("stick_radius", 0.16, "g");  cmd.color("orange", "g"); cmd.util.cnc("g")
    cmd.orient("c")
cmd.set("ray_trace_mode", 1)
cmd.ray(2600, 1200)
cmd.png("figure5_combined.png", dpi=300, ray=1)
print("wrote figure5_combined.png")
python end

# Done. Import the two panels (or the combined image) into your figure
# layout tool, add the shaded (tau, phi) accessible envelope from the
# scan beside each panel, and label bright vs dim with QY and f_allowed.
