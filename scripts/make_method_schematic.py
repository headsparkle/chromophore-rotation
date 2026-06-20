"""
make_method_schematic.py
========================

Method-schematic figure for the rigid (tau, phi) scan and the two geometric
metrics it yields, drawn on the real avGFP (1EMA) scan:
  - the clash-free (sterically allowed) island in (tau, phi) space; its area as
    a fraction of the full 360x360 deg space is f_allowed;
  - the four planar reference points (0,0), (0,180), (180,0), (180,180);
  - the deposited chromophore point (tau_exp, phi_exp) and the arrow to the
    nearest planar reference, whose length is d_exp_to_planar.

Output: figures/pub_figures/fig_method_schematic.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import load_structure, build_cage, max_overlap, DEFAULT_TOLERANCE_A
from rotate import set_megley, measure_megley

CIF = Path(__file__).resolve().parent.parent / "data" / "cif"
OUT = Path(__file__).resolve().parent.parent / "figures" / "pub_figures" / "fig_method_schematic"
GRID = np.arange(-180.0, 180.0, 5.0)


def overlap_map(pid):
    L = load_structure(CIF / f"{pid}.cif"); cage = build_cage(L)
    tau_exp, phi_exp = measure_megley(L.chrom_atoms)
    ov = np.empty((len(GRID), len(GRID)))
    for i, tau in enumerate(GRID):
        for j, phi in enumerate(GRID):
            atoms = set_megley(L.chrom_atoms, float(tau), float(phi))
            mv = np.array([atoms[n] for n in cage.moving_names])
            ov[i, j], _ = max_overlap(mv, cage.moving_radii, cage.cage_xyz,
                                      cage.cage_radii, cage.exclude)
    return ov, tau_exp, phi_exp


def nearest_planar(tau, phi):
    def snap(a):
        return min((0.0, 180.0, -180.0), key=lambda c: abs(((a - c + 180) % 360) - 180))
    return snap(tau), snap(phi)


def main(pid="1EMA"):
    ov, tau_exp, phi_exp = overlap_map(pid)
    allowed = ov <= DEFAULT_TOLERANCE_A
    f_allowed = allowed.mean()

    fig, ax = plt.subplots(figsize=(6.4, 5.9))
    # steric-overlap landscape: low overlap = open (green), high = clash (red),
    # diverging around the 0.4 A allowed cutoff
    edges = np.arange(-180.0, 181.0, 5.0)
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=DEFAULT_TOLERANCE_A, vmin=-1.0, vmax=3.0)
    pc = ax.pcolormesh(edges, edges, np.clip(ov, -1.0, 3.0).T, cmap="RdYlGn_r",
                       norm=norm, shading="flat", zorder=0)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("max steric overlap (Å)", fontsize=8); cb.ax.tick_params(labelsize=7)
    # outline the sterically allowed island(s): overlap <= 0.4 A
    ctr, cph = np.meshgrid(GRID + 2.5, GRID + 2.5, indexing="ij")
    ax.contour(ctr, cph, ov, levels=[DEFAULT_TOLERANCE_A], colors="#114d11",
               linewidths=1.6, zorder=2)

    # four planar references
    refs = [(0, 0), (0, 180), (180, 0), (180, 180),
            (0, -180), (-180, 0), (-180, 180), (180, -180), (-180, -180)]
    rx = [r[0] for r in refs]; ry = [r[1] for r in refs]
    ax.scatter(rx, ry, marker="*", s=200, color="black", zorder=5,
               label="planar references (0, ±180)")

    # deposited point + arrow to nearest planar reference (= d_exp_to_planar)
    pt, pp = nearest_planar(tau_exp, phi_exp)
    ax.scatter([tau_exp], [phi_exp], s=90, color="#c11f2f", edgecolors="white",
               linewidths=0.8, zorder=6, label="deposited (τ$_{exp}$, φ$_{exp}$)")
    ax.annotate("", xy=(pt, pp), xytext=(tau_exp, phi_exp),
                arrowprops=dict(arrowstyle="->", color="#c11f2f", lw=2.0), zorder=6)
    ax.text((tau_exp + pt) / 2 + 6, (phi_exp + pp) / 2 + 10,
            "$d_{\\mathrm{exp\\_to\\_planar}}$", color="#c11f2f", fontsize=10, zorder=7)

    ax.text(0.03, 0.04,
            f"green outline = allowed (overlap $\\leq$ 0.4 Å)\n"
            f"$f_{{\\mathrm{{allowed}}}}$ = allowed area / total = {f_allowed:.3f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb"))

    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel(r"$\tau$  (I-bond, N2-CA2-CB2-CG2)  [deg]")
    ax.set_ylabel(r"$\phi$  (P-bond, CA2-CB2-CG2-CD1)  [deg]")
    ax.set_title(f"Rigid ($\\tau$, $\\phi$) scan and its two metrics ({pid}, avGFP)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(str(OUT) + ".png", dpi=300)
    fig.savefig(str(OUT) + ".pdf")
    print(f"wrote {OUT}.png/.pdf   (f_allowed={f_allowed:.3f}, "
          f"tau_exp={tau_exp:.1f}, phi_exp={phi_exp:.1f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1EMA")
