#!/usr/bin/env python3
"""
relaxed_energy_scan.py
======================

Relaxed (cage-breathing) energy scan of the two Megley torsions for a
small panel of baseline fluorescent proteins. This is the "soft-core LJ
+ 5 A shell relaxation" complement to the rigid steric clash scan
(scan_all.py / barrel.py): instead of a binary clash / no-clash verdict
at each (tau, phi), it reports a continuous steric energy DeltaE(tau, phi)
after letting the side chains within 5 A of the chromophore relax.

Motivation
----------
The production analysis scores the rebuilt chromophore against a *rigid*
cage (hard Bondi spheres, 0.4 A MolProbity tolerance). The obvious
referee objection is that a real barrel breathes, so a rigid clash may
overstate how forbidden a rotamer is. This script tests that on a few
representative structures by replacing the rigid wall with a relaxable
one and asking two questions at every grid point:

  1. DeltaE(tau, phi): the steric energy after relaxation, relative to
     the global minimum of the relaxed surface. This is the "energy
     barrier" map the rigid scan cannot give.
  2. Does the gatekeeper survive breathing? i.e. after relaxation, is
     the rotamer still sterically excluded (relaxed Bondi overlap still
     above tolerance)? If the high-energy ridges of DeltaE coincide with
     the rigid clash walls, the rigid f_allowed is structurally sound.

What this model is (and is NOT)
-------------------------------
This is a RESTRAINED SOFT-SPHERE relaxation, not a full ff14SB / CHARMM
molecular-dynamics minimization. The chromophore is a non-standard
residue covalently fused into the backbone; deriving bonded force-field
parameters for 60+ chromophore three-letter codes is the heavyweight
"future work" the manuscript explicitly defers. Because the chromophore
is held RIGID at each (tau, phi), its internal bonded terms are constant
and irrelevant; only NONBONDED interactions with the cage matter. So the
model here is:

    E(shell) =  sum  LJ_softcore(r_ij)              (inter-residue only)
              + 0.5 * K_TETHER * sum |x_i - x0_i|^2  (shell atoms)

  * LJ between heavy atoms of DIFFERENT residues only (intra-residue
    bonded pairs are excluded; with no bonded terms they would otherwise
    blow a side chain apart). The pair minimum r_min,ij is the Bondi vdW
    sum (same radii the rigid scan uses), well depth EPS0.
  * A soft-core (linear-capped) repulsion bounds the force when a rotated
    chromophore atom deeply penetrates a shell atom at the start.
  * A harmonic tether of every relaxable atom to its crystal position
    stands in for the bonded network: it lets side chains breathe a few
    tenths of an Angstrom (the intended cage flexibility) without flying
    apart, and keeps intra-residue geometry near crystal.

Only side-chain heavy atoms of STANDARD amino acids whose residue has an
atom within 5 A of the chromophore are relaxed; backbone (N, CA, C, O)
and all non-standard residues (partner chromophores, ligands) stay fixed.

Outputs (per PDB, in data/relaxed_scans/)
-----------------------------------------
  relaxed_<PDB>.npz : tau_grid, phi_grid, E_relaxed, dE (=E-min),
                      overlap_rigid, overlap_relaxed, exp angles, params.
  figures/relaxed_dE_<PDB>.png : DeltaE heatmap + experimental point +
                      rigid- and relaxed-allowed contours.
And one summary row per PDB appended to data/relaxed_scan_summary.csv.

Usage:
    python3 scripts/relaxed_energy_scan.py                 # default panel
    python3 scripts/relaxed_energy_scan.py 1EMA            # one structure
    python3 scripts/relaxed_energy_scan.py --step 15       # coarser grid
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gemmi
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import (  # noqa: E402
    BONDI_RADII,
    DEFAULT_TOLERANCE_A,
    build_cage,
    load_structure,
)
from rotate import measure_megley, set_megley  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif"
OUT_DIR = DATA_DIR / "relaxed_scans"
FIG_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = DATA_DIR / "relaxed_scan_summary.csv"

# Default baseline panel (see log.md 2026-06-20). mRouge (QY 0.018) is not
# in the 838-structure cohort; 3KCS mCherry is the dim/twisted red exemplar
# (deposited at tau ~ -90, a fully twisted I-bond).
DEFAULT_PANEL = {
    "1EMA": "green avGFP/EGFP (CRO) reference",
    "5LK4": "bright red mScarlet (NRQ), QY 0.70, near-planar",
    "3KCS": "dim/twisted red mCherry (NRQ), QY 0.22, tau~-90",
    "2YE0": "indole cyan mTurquoise (SWG), QY 0.84",
}

# --- model parameters -------------------------------------------------------
EPS0 = 0.10          # kcal/mol, generic heavy-atom vdW well depth
K_TETHER = 10.0      # kcal/mol/A^2, harmonic restraint to crystal position
SHELL_RADIUS_A = 5.0 # residues with an atom within this of the chromophore relax
NEIGHBOR_CUT_A = 8.0 # fixed-env atoms beyond this of chrom+shell never interact
LJ_CUT_A = 6.0       # LJ pair cutoff during minimisation
SOFTCORE_FRAC = 0.80 # below SOFTCORE_FRAC * r_min the repulsion goes linear
KT = 0.593           # kcal/mol at 298 K (for the energy-accessible fraction)
DE_KT_MULT = 5.0     # "energy-allowed" if dE < DE_KT_MULT * KT (~3 kcal/mol)

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC",
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}
WATER = {"HOH", "WAT", "DOD"}


def radius_of(elem: str) -> float:
    return BONDI_RADII.get(elem.upper(), BONDI_RADII["C"])


# ---------------------------------------------------------------------------
# Environment (non-chromophore) atoms, with residue identity
# ---------------------------------------------------------------------------

@dataclass
class EnvAtoms:
    xyz: np.ndarray        # (N, 3)
    radii: np.ndarray      # (N,)
    res_id: np.ndarray     # (N,) integer residue index (per (chain, seqid))
    is_shell: np.ndarray   # (N,) bool, relaxable side-chain atom
    res_key: list          # length = n residues, for debugging


def load_env(cif_path: Path, chrom_chain: str, chrom_seqid: int,
             chrom_xyz: np.ndarray) -> EnvAtoms:
    """All heavy, non-water atoms except the one chromophore residue,
    tagged with a per-residue integer id and a shell flag (relaxable
    side-chain atom of a standard residue within SHELL_RADIUS_A of the
    chromophore)."""
    st = gemmi.read_structure(str(cif_path))
    model = st[0]
    xyz, radii, res_id, names, resnames = [], [], [], [], []
    res_key: list = []
    key_to_id: dict = {}
    for chain in model:
        for res in chain:
            if res.name in WATER:
                continue
            if chain.name == chrom_chain and res.seqid.num == chrom_seqid:
                continue  # the chromophore itself
            key = (chain.name, res.seqid.num, res.name)
            if key not in key_to_id:
                key_to_id[key] = len(res_key)
                res_key.append(key)
            rid = key_to_id[key]
            for atom in res:
                if atom.element.is_hydrogen:
                    continue
                xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                radii.append(radius_of(atom.element.name))
                res_id.append(rid)
                names.append(atom.name)
                resnames.append(res.name)
    xyz = np.asarray(xyz, dtype=float)
    radii = np.asarray(radii, dtype=float)
    res_id = np.asarray(res_id, dtype=int)
    names = np.asarray(names)
    resnames = np.asarray(resnames)

    # shell = standard-aa side-chain atom whose residue has any atom <= 5 A
    dmin = np.linalg.norm(xyz[:, None, :] - chrom_xyz[None, :, :], axis=-1).min(axis=1)
    near_res = set(res_id[dmin <= SHELL_RADIUS_A].tolist())
    is_std = np.array([rn in STD_AA for rn in resnames])
    is_side = np.array([nm not in BACKBONE for nm in names])
    in_near_res = np.array([rid in near_res for rid in res_id])
    is_shell = is_std & is_side & in_near_res
    return EnvAtoms(xyz, radii, res_id, is_shell, res_key)


# ---------------------------------------------------------------------------
# Soft-core Lennard-Jones (linear-capped below SOFTCORE_FRAC * r_min)
# ---------------------------------------------------------------------------

def _lj_e_and_dedr(r, rmin):
    """Vectorised LJ energy and dE/dr for pair minima rmin, well depth EPS0,
    with a linear soft core below SOFTCORE_FRAC*rmin to bound the force."""
    rc = SOFTCORE_FRAC * rmin
    # standard 12-6 with minimum at r = rmin, depth EPS0
    def lj(rr):
        a = (rmin / rr) ** 6
        e = EPS0 * (a * a - 2.0 * a)
        de = EPS0 * (-12.0 * a * a + 12.0 * a) / rr   # dE/dr
        return e, de
    e_full, de_full = lj(np.maximum(r, 1e-6))
    e_cap, de_cap = lj(rc)
    soft = r < rc
    e = np.where(soft, e_cap + de_cap * (r - rc), e_full)
    de = np.where(soft, de_cap, de_full)
    return e, de


# ---------------------------------------------------------------------------
# Per-structure relaxation context
# ---------------------------------------------------------------------------

@dataclass
class RelaxContext:
    pdb_id: str
    chrom_resname: str
    # chromophore
    chrom_names: list
    chrom_radii: np.ndarray
    moving_names: tuple
    static_mask: np.ndarray       # over chrom atoms: True = static (imidazolinone)
    # environment partitioned into shell (movers) and fixed
    shell_xyz0: np.ndarray        # (S,3) crystal positions of movers
    shell_radii: np.ndarray
    shell_resid: np.ndarray
    fixed_xyz: np.ndarray         # (F,3)
    fixed_radii: np.ndarray
    fixed_resid: np.ndarray
    # rigid-scan handles
    loaded: object
    cage: object
    tau_exp: float
    phi_exp: float


def build_context(pdb_id: str) -> RelaxContext:
    cif = CIF_DIR / f"{pdb_id.upper()}.cif"
    loaded = load_structure(cif)
    cage = build_cage(loaded)
    tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)
    chrom_names = list(loaded.chrom_atoms.keys())
    chrom_radii = np.array([radius_of(loaded.chrom_elements[n]) for n in chrom_names])
    static_mask = np.array([n not in cage.moving_names for n in chrom_names])
    chrom_xyz = np.array([loaded.chrom_atoms[n] for n in chrom_names])

    env = load_env(cif, loaded.chrom_chain, loaded.chrom_seqid, chrom_xyz)
    # keep only fixed-env atoms near the chromophore/shell (others never interact)
    near_pts = np.vstack([chrom_xyz, env.xyz[env.is_shell]])
    dmin_env = np.linalg.norm(env.xyz[:, None, :] - near_pts[None, :, :], axis=-1).min(axis=1)
    keep_fixed = (~env.is_shell) & (dmin_env <= NEIGHBOR_CUT_A)

    return RelaxContext(
        pdb_id=pdb_id.upper(),
        chrom_resname=loaded.chrom_resname,
        chrom_names=chrom_names,
        chrom_radii=chrom_radii,
        moving_names=cage.moving_names,
        static_mask=static_mask,
        shell_xyz0=env.xyz[env.is_shell].copy(),
        shell_radii=env.radii[env.is_shell],
        shell_resid=env.res_id[env.is_shell],
        fixed_xyz=env.xyz[keep_fixed],
        fixed_radii=env.radii[keep_fixed],
        fixed_resid=env.res_id[keep_fixed],
        loaded=loaded,
        cage=cage,
        tau_exp=tau_exp,
        phi_exp=phi_exp,
    )


# ---------------------------------------------------------------------------
# Energy + gradient for one (tau, phi)
# ---------------------------------------------------------------------------

def _pair_energy_grad(shell_xyz, shell_radii, shell_resid,
                      other_xyz, other_radii, other_resid, want_grad):
    """LJ energy and gradient on shell atoms from a set of 'other' atoms.
    Pairs in the same residue (same resid, finite) are excluded. The
    chromophore is given resid = -1 so it never matches a real residue."""
    diff = shell_xyz[:, None, :] - other_xyz[None, :, :]      # (S,O,3)
    dist = np.linalg.norm(diff, axis=-1)                      # (S,O)
    rmin = shell_radii[:, None] + other_radii[None, :]
    same_res = shell_resid[:, None] == other_resid[None, :]
    mask = (~same_res) & (dist <= LJ_CUT_A)
    if not mask.any():
        return 0.0, np.zeros_like(shell_xyz)
    r = np.where(mask, dist, np.inf)
    e_pair, de_pair = _lj_e_and_dedr(r, rmin)
    e_pair = np.where(mask, e_pair, 0.0)
    energy = float(e_pair.sum())
    if not want_grad:
        return energy, None
    de_pair = np.where(mask, de_pair, 0.0)
    rsafe = np.where(mask, dist, 1.0)
    coeff = (de_pair / rsafe)[..., None]                     # (S,O,1)
    grad = (coeff * diff).sum(axis=1)                        # (S,3)
    return energy, grad


def make_objective(ctx: RelaxContext, chrom_xyz_now: np.ndarray):
    """Closure returning E and analytic gradient for the shell positions
    flattened to 1D, at a fixed chromophore geometry."""
    chrom_resid = np.full(len(chrom_xyz_now), -1, dtype=int)
    S = len(ctx.shell_xyz0)

    def fun(x):
        sx = x.reshape(S, 3)
        e_c, g_c = _pair_energy_grad(sx, ctx.shell_radii, ctx.shell_resid,
                                     chrom_xyz_now, ctx.chrom_radii, chrom_resid, True)
        e_f, g_f = _pair_energy_grad(sx, ctx.shell_radii, ctx.shell_resid,
                                     ctx.fixed_xyz, ctx.fixed_radii, ctx.fixed_resid, True)
        e_s, g_s = _pair_energy_grad(sx, ctx.shell_radii, ctx.shell_resid,
                                     sx, ctx.shell_radii, ctx.shell_resid, True)
        # shell-shell counted twice (both directions); halve energy, gradient ok
        e_s *= 0.5
        dx = sx - ctx.shell_xyz0
        e_t = 0.5 * K_TETHER * float((dx * dx).sum())
        g_t = K_TETHER * dx
        e = e_c + e_f + e_s + e_t
        g = g_c + g_f + g_s + g_t
        return e, g.ravel()

    return fun


def relaxed_overlap(ctx: RelaxContext, chrom_xyz_now: np.ndarray,
                    shell_xyz: np.ndarray) -> float:
    """Largest Bondi overlap (A) of moving chromophore atoms against the
    relaxed cage (static chromophore + relaxed shell + fixed env), using
    the same 1,2/1,3 exclusions vs static-chromophore that the rigid scan
    uses (via cage.exclude)."""
    cage = ctx.cage
    mov_idx = [ctx.chrom_names.index(n) for n in ctx.moving_names]
    mov_xyz = chrom_xyz_now[mov_idx]
    mov_r = ctx.chrom_radii[np.array(mov_idx)]
    # static chromophore part: reuse cage.exclude (first nstatic columns)
    nstatic = int(ctx.static_mask.sum())
    stat_xyz = cage.cage_xyz[:nstatic]
    stat_r = cage.cage_radii[:nstatic]
    best = -np.inf
    # vs static chrom (with exclusions)
    d = np.linalg.norm(mov_xyz[:, None, :] - stat_xyz[None, :, :], axis=-1)
    ov = mov_r[:, None] + stat_r[None, :] - d
    ov = np.where(cage.exclude[:, :nstatic], -np.inf, ov)
    best = max(best, float(ov.max()) if ov.size else -np.inf)
    # vs relaxed shell + fixed env (no exclusions; different residues)
    env_xyz = np.vstack([shell_xyz, ctx.fixed_xyz])
    env_r = np.concatenate([ctx.shell_radii, ctx.fixed_radii])
    d = np.linalg.norm(mov_xyz[:, None, :] - env_xyz[None, :, :], axis=-1)
    ov = mov_r[:, None] + env_r[None, :] - d
    best = max(best, float(ov.max()) if ov.size else -np.inf)
    return best


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def grid(step: float) -> np.ndarray:
    return np.arange(-180.0, 180.0, step)


def scan_arrays(ctx: RelaxContext, step_deg: float, verbose: bool = False):
    """Run the (tau, phi) grid for one structure and return
    (taus, phis, E_relaxed, ov_rigid, ov_relax). Uses the current module
    EPS0 / K_TETHER (read at call time), so a caller may monkeypatch those
    for a parameter sweep. Builds no files."""
    t0 = time.perf_counter()
    taus = grid(step_deg)
    phis = grid(step_deg)
    nT, nP = len(taus), len(phis)
    E = np.full((nT, nP), np.nan)
    ov_rigid = np.full((nT, nP), np.nan)
    ov_relax = np.full((nT, nP), np.nan)

    base = ctx.loaded.chrom_atoms
    for i, tau in enumerate(taus):
        for j, phi in enumerate(phis):
            atoms_now = set_megley(base, float(tau), float(phi))
            chrom_xyz_now = np.array([atoms_now[n] for n in ctx.chrom_names])
            # rigid overlap (no relaxation) for direct comparison
            ov_rigid[i, j] = relaxed_overlap(ctx, chrom_xyz_now, ctx.shell_xyz0)
            # relax the shell
            fun = make_objective(ctx, chrom_xyz_now)
            res = minimize(fun, ctx.shell_xyz0.ravel(), jac=True,
                           method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-7, "gtol": 1e-5})
            shell_relaxed = res.x.reshape(ctx.shell_xyz0.shape)
            E[i, j] = res.fun
            ov_relax[i, j] = relaxed_overlap(ctx, chrom_xyz_now, shell_relaxed)
        if verbose:
            print(f"    {ctx.pdb_id} tau row {i+1}/{nT} (tau={tau:+.0f})  "
                  f"elapsed={(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return taus, phis, E, ov_rigid, ov_relax


def run_structure(pdb_id: str, step_deg: float, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    ctx = build_context(pdb_id)
    taus, phis, E, ov_rigid, ov_relax = scan_arrays(ctx, step_deg, verbose)
    dE = E - np.nanmin(E)
    npz = OUT_DIR / f"relaxed_{ctx.pdb_id}.npz"
    np.savez_compressed(
        npz, tau_grid=taus, phi_grid=phis,
        E_relaxed=E.astype(np.float32), dE=dE.astype(np.float32),
        overlap_rigid=ov_rigid.astype(np.float32),
        overlap_relaxed=ov_relax.astype(np.float32),
        tau_exp=ctx.tau_exp, phi_exp=ctx.phi_exp, step_deg=step_deg,
        pdb_id=ctx.pdb_id, chrom_resname=ctx.chrom_resname,
        eps0=EPS0, k_tether=K_TETHER, shell_radius_a=SHELL_RADIUS_A,
        n_shell_atoms=len(ctx.shell_xyz0),
    )

    tol = DEFAULT_TOLERANCE_A
    f_rigid = float((ov_rigid <= tol).mean())
    f_relax = float((ov_relax <= tol).mean())
    f_energy = float((dE < DE_KT_MULT * KT).mean())
    secs = time.perf_counter() - t0
    if verbose:
        print(f"  [{ctx.pdb_id}] done in {secs/60:.1f} min  "
              f"f_rigid={f_rigid:.3f} f_relax={f_relax:.3f} "
              f"f_energy={f_energy:.3f}  shell={len(ctx.shell_xyz0)} atoms", flush=True)
    make_figure(ctx, npz)
    return {
        "pdb_id": ctx.pdb_id,
        "chrom_resname": ctx.chrom_resname,
        "note": DEFAULT_PANEL.get(ctx.pdb_id, ""),
        "step_deg": step_deg,
        "n_shell_atoms": len(ctx.shell_xyz0),
        "tau_exp_deg": round(ctx.tau_exp, 2),
        "phi_exp_deg": round(ctx.phi_exp, 2),
        "f_allowed_rigid": round(f_rigid, 4),
        "f_allowed_relaxed": round(f_relax, 4),
        "f_energy_accessible": round(f_energy, 4),
        "relaxed_minus_rigid_area_pct": round(100 * (f_relax - f_rigid), 2),
        "eps0_kcal": EPS0,
        "k_tether": K_TETHER,
        "scan_seconds": round(secs, 1),
    }


def make_figure(ctx: RelaxContext, npz_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    taus, phis = z["tau_grid"], z["phi_grid"]
    dE = z["dE"]
    ov_rigid, ov_relax = z["overlap_rigid"], z["overlap_relaxed"]
    tol = DEFAULT_TOLERANCE_A
    extent = [phis[0], phis[-1], taus[0], taus[-1]]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    capped = np.minimum(dE, 25.0)  # cap colour scale at 25 kcal/mol
    im = ax.imshow(capped, origin="lower", extent=extent, aspect="auto",
                   cmap="viridis_r", vmin=0, vmax=25)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"relaxed $\Delta E$  (kcal/mol, capped at 25)")
    # rigid-allowed boundary (white) and relaxed-allowed boundary (cyan)
    P, T = np.meshgrid(phis, taus)
    ax.contour(P, T, (ov_rigid <= tol).astype(float), levels=[0.5],
               colors="white", linewidths=1.4)
    ax.contour(P, T, (ov_relax <= tol).astype(float), levels=[0.5],
               colors="cyan", linewidths=1.0, linestyles="--")
    ax.plot(z["phi_exp"], z["tau_exp"], "r*", ms=15, mec="k",
            label="experimental")
    ax.set_xlabel(r"$\varphi_{\mathrm{megley}}$  (P-bond, deg)")
    ax.set_ylabel(r"$\tau_{\mathrm{megley}}$  (I-bond, deg)")
    ax.set_title(f"{str(z['pdb_id'])}  ({str(z['chrom_resname'])})  "
                 f"relaxed steric energy\nwhite = rigid-allowed, "
                 f"cyan dashed = relaxed-allowed")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"relaxed_dE_{str(z['pdb_id'])}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def append_summary(row: dict) -> None:
    fields = list(row.keys())
    new = not SUMMARY_CSV.is_file()
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_ids", nargs="*", help="PDB ids (default: baseline panel)")
    ap.add_argument("--step", type=float, default=10.0, help="grid step in deg")
    args = ap.parse_args()

    pdb_ids = [p.upper() for p in args.pdb_ids] or list(DEFAULT_PANEL)
    print(f"[relaxed_scan] panel = {pdb_ids}, step = {args.step} deg")
    for pid in pdb_ids:
        print(f"[relaxed_scan] === {pid}: {DEFAULT_PANEL.get(pid,'')} ===", flush=True)
        row = run_structure(pid, args.step)
        append_summary(row)
    print("[relaxed_scan] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
