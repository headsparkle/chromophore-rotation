#!/usr/bin/env python3
"""
compose_fig5.py  -  assemble the final Figure 5.

Lays out, for the bright and the dim protein:
    [ structural panel ]  [ (tau, phi) accessible-space envelope ]
with a header line per protein (name, QY, f_allowed) and (a)/(b) labels.

Inputs (produced by render_panels.pml and make_envelope.py):
    panel_bright.png     panel_dim.png
    envelope_7y40.png    envelope_6nqk.png   (optional; omit to skip insets)

Output:
    figure5.png   figure5.pdf   (300 dpi, publication-ready)

Edit the METADATA block if you change the structures.
Usage:  python compose_fig5.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --------------------------- METADATA --------------------------------
ROWS = [
    dict(tag="(a)  bright, loose cage",
         name="StayGold  (PDB 7Y40)",
         meta="QY = 0.93     f_allowed = 0.033",
         panel="panel_bright.png",
         envelope="envelope_7y40.png"),
    dict(tag="(b)  dim, tight cage",
         name="Dronpa-2  (PDB 6NQK)",
         meta="QY = 0.33     f_allowed = 0.0001",
         panel="panel_dim.png",
         envelope="envelope_6nqk.png"),
]
CAPTION = ("Figure 5. A bright, loose-caged and a dim, tight-caged green fluorescent protein "
           "share the same phenol chromophore, so cage volume alone does not report brightness. "
           "Chromophore, teal; first-shell cage residues within 5 A, wheat; position 203, orange. "
           "Right of each: the sterically accessible (tau, phi) region from the rigid scan "
           "(green = clash-free; black outline = 0.4 A allowed boundary; open circle = deposited "
           "geometry; stars = planar references).")
# ---------------------------------------------------------------------


def _imshow(ax, path):
    ax.axis("off")
    p = Path(path)
    if p.exists():
        ax.imshow(mpimg.imread(str(p)))
    else:
        ax.text(0.5, 0.5, "missing:\n%s" % p.name, ha="center", va="center",
                fontsize=9, color="crimson", transform=ax.transAxes)


def main():
    nrows = len(ROWS)
    fig = plt.figure(figsize=(11.0, 4.6 * nrows))
    # 2 columns: structural panel (wider) + envelope
    gs = fig.add_gridspec(nrows, 2, width_ratios=[1.55, 1.0],
                          hspace=0.28, wspace=0.06,
                          left=0.02, right=0.98, top=0.95, bottom=0.10)
    for i, row in enumerate(ROWS):
        axL = fig.add_subplot(gs[i, 0])
        _imshow(axL, row["panel"])
        axL.set_title("%s\n%s        %s" % (row["tag"], row["name"], row["meta"]),
                      fontsize=11, loc="left")
        axR = fig.add_subplot(gs[i, 1])
        _imshow(axR, row["envelope"])
    fig.text(0.02, 0.015, CAPTION, fontsize=8.5, wrap=True, va="bottom")
    for ext in ("png", "pdf"):
        out = "figure5.%s" % ext
        fig.savefig(out, dpi=300)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
