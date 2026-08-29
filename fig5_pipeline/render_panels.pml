# =====================================================================
# render_panels.pml  -  structural panels for Figure 5
#
# Renders one high-resolution PNG per structure:
#   panel_bright.png  (StayGold, loose cage, bright)
#   panel_dim.png     (Dronpa-2, tight cage, dim)
# each showing the HBI chromophore (teal), the first-shell cage
# residues within 5 A (wheat thin sticks), and position 203 / its
# aligned equivalent (orange).
#
# Runs head-less. PyMOL's ray tracer is CPU-multithreaded; it does not
# use the GPU, but it runs fine on a GPU compute node. Use xvfb-run if
# your build needs an X/GL context even in -cq mode (see run_fig5.sh).
#
#   pymol -cq render_panels.pml
#
# Structure files are read from ./structures/<pdbid>.cif (see
# fetch_structures.sh). Edit the four settings below if you change the
# structures or the gatekeeper residue numbers.
# =====================================================================

python
BRIGHT_CIF   = "structures/7y40.cif"   # StayGold (verified on RCSB)
DIM_CIF      = "structures/6nqk.cif"   # Dronpa-2 = Dronpa M159T (verified)
GK_RESI_BR   = "203"                   # gatekeeper residue number in the bright structure
GK_RESI_DIM  = "203"                   # gatekeeper residue number in the dim structure
DPI          = 300
W, H         = 1800, 1500              # pixels per panel
# chromophore residue-name codes (GFP/CFP/RFP types); extend if needed
CHROMO_RESN = ("CRO+CR2+CRQ+CRF+CRK+CR7+CR8+CRG+CFY+CCY+CSH+GYC+GYS+NYG+"
               "NRQ+CH6+CH7+SWG+OHD+RC7+B2H+C12+XYG+DYG+PIA+IIC+MDO+QLG+GYG")
python end

# ---- global appearance ----
bg_color white
set ray_opaque_background, 0
set orthoscopic, 1
set ray_shadows, 0
set antialias, 2
set ray_trace_mode, 1
set stick_radius, 0.13
set cartoon_transparency, 0.55
set cartoon_side_chain_helper, 1
set valence, 1
set specular, 0.15
set ambient, 0.38
set direct, 0.55
set light_count, 2

python
from pymol import cmd

def render(cif, gk_resi, out_png):
    cmd.delete("all")
    cmd.load(cif, "fp")
    cmd.remove("solvent")
    cmd.remove("hydro")

    cmd.select("chromo", "fp and resn %s" % CHROMO_RESN)
    if cmd.count_atoms("chromo") == 0:
        # fallback: non-protein organic residue (the matured chromophore)
        cmd.select("chromo", "fp and organic and not inorganic and not polymer.protein")
    print("[%s] chromophore atoms: %d" % (cif, cmd.count_atoms("chromo")))

    cmd.select("cage", "byres (polymer.protein within 5.0 of chromo) and not chromo")
    cmd.select("gk", "polymer.protein and resi %s" % gk_resi)

    cmd.hide("everything", "fp")
    cmd.show("cartoon", "fp");    cmd.color("grey80", "fp")
    cmd.show("sticks", "chromo"); cmd.color("teal", "chromo");  cmd.util.cnc("chromo")
    cmd.show("sticks", "cage");   cmd.set("stick_radius", 0.09, "cage")
    cmd.color("wheat", "cage");   cmd.util.cnc("cage")
    cmd.show("sticks", "gk");     cmd.set("stick_radius", 0.16, "gk")
    cmd.color("orange", "gk");    cmd.util.cnc("gk")

    cmd.orient("chromo")
    cmd.zoom("chromo", 6.0)
    cmd.turn("y", 10)
    cmd.ray(W, H)
    cmd.png(out_png, dpi=DPI, ray=1)
    print("wrote %s" % out_png)

render(BRIGHT_CIF, GK_RESI_BR, "panel_bright.png")
render(DIM_CIF,    GK_RESI_DIM, "panel_dim.png")
python end
