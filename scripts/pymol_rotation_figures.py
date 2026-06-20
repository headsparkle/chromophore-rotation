"""Render two figures of the 1EMA chromophore driven through the
Megley torsions using PyMOL.

Each figure overlays five copies of the chromophore at 0, 15, 30, 45
and 60 deg of added rotation about the chosen Megley axis:

    tau (I-bond): dihedral N2 - CA2 - CB2 - CG2
    phi (P-bond): dihedral CA2 - CB2 - CG2 - CD1

The 0 deg copy is the crystallographic 1EMA chromophore. Higher copies
are the same chromophore with the named dihedral rotated by the indicated
amount; the side of the bond that moves is the phenol (CG2 onward), so
the imidazolinone half overlays exactly across all five copies.

Usage:
    python3 scripts/pymol_rotation_figures.py [path/to/1ema.cif]

Defaults to data/1EMA.cif relative to the project root. Writes:
    figures/fig_phi_rotation_0_60_1EMA.png
    figures/fig_tau_rotation_0_60_1EMA.png
"""

from __future__ import annotations

import os
import sys

from pymol import cmd


ANGLES = [0, 15, 30, 45, 60]
# Five-step viridis-like gradient (dark purple -> teal -> yellow-green).
COLORS_HEX = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']


def _to_pymol_color(name: str, hex_rgb: str) -> str:
    r = int(hex_rgb[1:3], 16) / 255.0
    g = int(hex_rgb[3:5], 16) / 255.0
    b = int(hex_rgb[5:7], 16) / 255.0
    cmd.set_color(name, [r, g, b])
    return name


def _isolate_chromophore(src_path: str, base_name: str = 'cro') -> None:
    """Load 1EMA, keep only the chromophore residue, drop hydrogens."""
    cmd.delete('all')
    cmd.load(src_path, 'full')
    # 1EMA's chromophore is CRO at residue 66, chain A.
    cmd.create(base_name, 'full and resn CRO and chain A')
    cmd.delete('full')
    cmd.remove(f'{base_name} and hydro')
    cmd.alter(base_name, 'segi=""')
    cmd.rebuild()


def _make_rotated_copies(base_name: str, dihedral_atoms, angles):
    """Create one copy per angle, set the named dihedral to (current +
    angle) on that copy, return the list of object names."""
    a1, a2, a3, a4 = dihedral_atoms
    sel0 = lambda an: f'{base_name} and name {an}'
    current = cmd.get_dihedral(sel0(a1), sel0(a2), sel0(a3), sel0(a4))

    names = []
    for ang in angles:
        nm = f'{base_name}_{ang:02d}'
        cmd.create(nm, base_name)
        cmd.set_dihedral(
            f'{nm} and name {a1}', f'{nm} and name {a2}',
            f'{nm} and name {a3}', f'{nm} and name {a4}',
            current + ang,
        )
        names.append(nm)
    cmd.delete(base_name)
    return names


def _stylize(names):
    cmd.bg_color('white')
    cmd.hide('everything')
    cmd.show('sticks')
    cmd.set('stick_radius', 0.16)
    cmd.set('stick_ball', 1)
    cmd.set('stick_ball_ratio', 1.8)
    cmd.set('ambient', 0.35)
    cmd.set('specular', 0.25)
    cmd.set('ray_shadows', 0)
    cmd.set('antialias', 2)
    cmd.set('ray_trace_mode', 1)
    cmd.set('ray_trace_color', 'gray30')

    for nm, hex_col in zip(names, COLORS_HEX):
        col = _to_pymol_color(f'col_{nm}', hex_col)
        cmd.color(col, nm)
        cmd.util.cnc(nm)  # heteroatoms (N, O) get standard colors
        cmd.color(col, f'{nm} and elem C')


def _orient_for_view(names, axis_atom_a: str, axis_atom_b: str):
    """Look perpendicular to the rotation axis with the axis horizontal,
    so the rotation is clearly visible in the image. We use the first
    (planar) copy as the reference for the orientation."""
    ref = names[0]
    cmd.orient(ref)
    # Rotate so the named rotation-axis bond is horizontal in the view.
    # PyMOL's `orient` aligns principal axes; the chromophore's long
    # axis is already nearly horizontal, but we nudge slightly so the
    # rotation is visible out of the page.
    cmd.turn('x', -10)
    cmd.turn('y', 5)
    cmd.zoom('visible', 1.5)


def make_figure(which: str, src_path: str, out_path: str) -> None:
    if which == 'phi':
        dihedral = ('CA2', 'CB2', 'CG2', 'CD1')
        axis = ('CB2', 'CG2')
        base = 'phi'
    elif which == 'tau':
        dihedral = ('N2', 'CA2', 'CB2', 'CG2')
        axis = ('CA2', 'CB2')
        base = 'tau'
    else:
        raise ValueError(which)

    _isolate_chromophore(src_path, base_name=base)
    names = _make_rotated_copies(base, dihedral, ANGLES)
    _stylize(names)
    _orient_for_view(names, *axis)

    cmd.png(out_path, width=1600, height=1200, dpi=300, ray=1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    default_src = os.path.join(project_root, 'data', '1EMA.cif')

    src = sys.argv[1] if len(sys.argv) > 1 else default_src
    if not os.path.exists(src):
        sys.stderr.write(f'Input not found: {src}\n')
        sys.exit(1)

    fig_dir = os.path.join(project_root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_phi = os.path.join(fig_dir, 'fig_phi_rotation_0_60_1EMA.png')
    out_tau = os.path.join(fig_dir, 'fig_tau_rotation_0_60_1EMA.png')

    make_figure('phi', src, out_phi)
    make_figure('tau', src, out_tau)
    print('Wrote:', out_phi)
    print('Wrote:', out_tau)


if __name__ == '__main__':
    main()
