#!/usr/bin/env python3
"""
relaxed_param_sweep.py
======================

Sensitivity of the relaxed (cage-breathing) energy scan to its two free
parameters before any quantitative claim is made:

  EPS0     -- the generic heavy-atom LJ well depth (kcal/mol)
  K_TETHER -- the harmonic restraint to the crystal position (kcal/mol/A^2)

These two set the balance between the LJ push (which opens area) and the
tether stiffness (which resists breathing), so absolute DeltaE magnitudes
and absolute allowed fractions WILL move with them. What must be robust for
the manuscript story is:

  (1) the bright/dim ORDERING across the panel (the dim, twisted red mCherry
      keeps the roomiest cage at every parameter setting);
  (2) the surface SHAPE -- the relaxed energy minimum stays on the deposited
      geometry, and breathing stays sub-Angstrom (it is a local relaxation,
      not a rearrangement);
  (3) the gatekeeper survives -- the relaxed allowed region never balloons.

We sweep a 3 x 3 grid of (EPS0, K_TETHER) over the four baselines on a 15 deg
torsional grid (coarser than the 10 deg production scan; we are testing
parameter trends in aggregate metrics, not making the final figure) and
write one row per (PDB, EPS0, K_TETHER) to data/relaxed_param_sweep.csv.

Run is parallel across cores. Set BLAS threads to 1 to avoid oversubscription:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python3 scripts/relaxed_param_sweep.py
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relaxed_energy_scan as R  # noqa: E402


PANEL = list(R.DEFAULT_PANEL)
EPS0_VALS = [0.05, 0.10, 0.20]
K_VALS = [2.0, 10.0, 20.0]
STEP = 15.0
OUT_CSV = R.DATA_DIR / "relaxed_param_sweep.csv"

FIELDS = [
    "pdb_id", "note", "eps0", "k_tether", "step_deg",
    "f_rigid", "f_relax", "f_energy",
    "dE_at_exp", "tau_min", "phi_min", "dE_max",
    "shell_rms_disp_A", "shell_max_disp_A", "is_default",
]


def _nearest_cell(grid_vals, target):
    return int(np.argmin(np.abs(((grid_vals - target + 180) % 360) - 180)))


def run_combo(task):
    pdb, eps0, k = task
    R.EPS0 = eps0
    R.K_TETHER = k
    ctx = R.build_context(pdb)
    taus, phis, E, ov_rigid, ov_relax = R.scan_arrays(ctx, STEP, verbose=False)
    dE = E - np.nanmin(E)
    tol = R.DEFAULT_TOLERANCE_A

    ie = _nearest_cell(taus, ctx.tau_exp)
    je = _nearest_cell(phis, ctx.phi_exp)
    imin, jmin = np.unravel_index(np.nanargmin(dE), dE.shape)

    # breathing magnitude at the experimental geometry
    atoms_exp = R.set_megley(ctx.loaded.chrom_atoms,
                             float(ctx.tau_exp), float(ctx.phi_exp))
    chrom_xyz = np.array([atoms_exp[n] for n in ctx.chrom_names])
    fun = R.make_objective(ctx, chrom_xyz)
    res = R.minimize(fun, ctx.shell_xyz0.ravel(), jac=True, method="L-BFGS-B",
                     options={"maxiter": 200, "ftol": 1e-7, "gtol": 1e-5})
    disp = res.x.reshape(ctx.shell_xyz0.shape) - ctx.shell_xyz0
    per_atom = np.sqrt((disp * disp).sum(axis=1))

    return {
        "pdb_id": ctx.pdb_id,
        "note": R.DEFAULT_PANEL[pdb].split(",")[0],
        "eps0": eps0,
        "k_tether": k,
        "step_deg": STEP,
        "f_rigid": round(float((ov_rigid <= tol).mean()), 4),
        "f_relax": round(float((ov_relax <= tol).mean()), 4),
        "f_energy": round(float((dE < R.DE_KT_MULT * R.KT).mean()), 4),
        "dE_at_exp": round(float(dE[ie, je]), 3),
        "tau_min": float(taus[imin]),
        "phi_min": float(phis[jmin]),
        "dE_max": round(float(np.nanmax(dE)), 1),
        "shell_rms_disp_A": round(float(np.sqrt((per_atom ** 2).mean())), 3),
        "shell_max_disp_A": round(float(per_atom.max()), 3),
        "is_default": int(eps0 == 0.10 and k == 10.0),
    }


def main() -> int:
    tasks = [(p, e, k) for p in PANEL for e in EPS0_VALS for k in K_VALS]
    print(f"[sweep] {len(tasks)} tasks "
          f"({len(PANEL)} PDB x {len(EPS0_VALS)} eps0 x {len(K_VALS)} k), "
          f"step={STEP} deg", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_combo, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(f"  [{n:>2d}/{len(tasks)}] {row['pdb_id']} "
                  f"eps0={row['eps0']} k={row['k_tether']}  "
                  f"f_relax={row['f_relax']} f_energy={row['f_energy']} "
                  f"dE@exp={row['dE_at_exp']} rms_disp={row['shell_rms_disp_A']}A  "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    rows.sort(key=lambda r: (r["pdb_id"], r["eps0"], r["k_tether"]))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {OUT_CSV} in {(time.perf_counter()-t0)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
