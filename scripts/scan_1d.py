"""
scan_1d.py
==========

P/I decomposition of the Megley rotational freedom.

For each structure we run two 1D scans:

  I-bond scan: fix phi at phi_exp, step tau over [-180, +180) in 5 deg.
               Counts how many tau positions are sterically allowed.
               f_allowed_I = n_allowed_tau / 72

  P-bond scan: fix tau at tau_exp, step phi over [-180, +180) in 5 deg.
               Counts how many phi positions are sterically allowed.
               f_allowed_P = n_allowed_phi / 72
               f_allowed_P_folded: for phi-symmetric (Tyr66) chromophores,
               fold the phi grid to [0, 180) -- a position is allowed if
               EITHER phi_j OR phi_j + 180 is allowed.

We also run both scans with the fixed bond set to 0 deg (planar reference)
to allow sensitivity checks:
  f_allowed_I_at0: tau scan with phi = 0
  f_allowed_P_at0: phi scan with tau = 0

Motivation (Boxer / Romei et al. Science 2020):
  The I-bond (CA2=CB2, tau) is the excited-state isomerization axis --
  its rotational freedom at the Franck-Condon geometry controls the S1
  barrier height and hence the radiationless decay rate.
  The P-bond (CB2-CG2, phi) is a ground-state degree of freedom --
  phenol ring flip. These two axes are mechanistically distinct
  (opposite charge-transfer directions), so comparing f_allowed_I vs
  f_allowed_P vs f_allowed (2D) quantifies how much coupling there is.

Output: data/scan_1d_summary.csv
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import (
    DEFAULT_TOLERANCE_A,
    build_cage,
    load_structure,
    max_overlap,
)
from rotate import measure_megley, set_megley

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CIF_DIR      = DATA_DIR / "cif"
OUT_CSV      = DATA_DIR / "scan_1d_summary.csv"
MERGED_CSV   = DATA_DIR / "merged_for_aggregate.csv"

STEP_DEG  = 5.0
TAU_GRID  = np.arange(-180.0, 180.0, STEP_DEG)   # 72 points
PHI_GRID  = np.arange(-180.0, 180.0, STEP_DEG)   # 72 points
N_GRID    = len(TAU_GRID)                          # 72


FIELDS = [
    "pdb_id", "status", "error",
    "chrom_resname", "phi_symmetric",
    "tau_exp_deg", "phi_exp_deg",
    # I-bond (tau) scans
    "f_allowed_I",          # tau scan, phi fixed at phi_exp
    "f_allowed_I_at0",      # tau scan, phi fixed at 0
    "n_allowed_I",
    "n_allowed_I_at0",
    # P-bond (phi) scans
    "f_allowed_P",          # phi scan, tau fixed at tau_exp
    "f_allowed_P_folded",   # folded across 180 for phi-symmetric
    "f_allowed_P_at0",      # phi scan, tau fixed at 0
    "f_allowed_P_at0_folded",
    "n_allowed_P",
    "n_allowed_P_folded",
    "n_allowed_P_at0",
    "n_allowed_P_at0_folded",
    # Cross-check: product of 1D fracs vs 2D f_allowed_folded from merged
    "f_product_folded",     # f_allowed_I * f_allowed_P_folded
    "scan_seconds",
]


def _scan_1d_tau(base_atoms, cage, phi_fixed: float) -> tuple[int, float]:
    """Scan tau with phi fixed. Return (n_allowed, f_allowed_I)."""
    moving_names  = cage.moving_names
    moving_radii  = cage.moving_radii
    cage_xyz      = cage.cage_xyz
    cage_radii    = cage.cage_radii
    exclude       = cage.exclude
    n_allowed = 0
    for tau in TAU_GRID:
        atoms_now   = set_megley(base_atoms, float(tau), float(phi_fixed))
        moving_xyz  = np.array([atoms_now[n] for n in moving_names], dtype=float)
        ov, _       = max_overlap(moving_xyz, moving_radii, cage_xyz, cage_radii, exclude)
        if ov <= DEFAULT_TOLERANCE_A:
            n_allowed += 1
    return n_allowed, n_allowed / N_GRID


def _scan_1d_phi(base_atoms, cage, tau_fixed: float,
                  phi_symmetric: bool) -> tuple[int, int, float, float]:
    """
    Scan phi with tau fixed.
    Returns (n_allowed, n_allowed_folded, f_allowed_P, f_allowed_P_folded).
    For phi-symmetric chromophores, n_allowed_folded counts the 36 unique
    phi positions [0, 180) where EITHER phi_j OR phi_j+180 is allowed.
    """
    moving_names  = cage.moving_names
    moving_radii  = cage.moving_radii
    cage_xyz      = cage.cage_xyz
    cage_radii    = cage.cage_radii
    exclude       = cage.exclude

    allowed_phi = np.zeros(N_GRID, dtype=bool)
    for j, phi in enumerate(PHI_GRID):
        atoms_now  = set_megley(base_atoms, float(tau_fixed), float(phi))
        moving_xyz = np.array([atoms_now[n] for n in moving_names], dtype=float)
        ov, _      = max_overlap(moving_xyz, moving_radii, cage_xyz, cage_radii, exclude)
        allowed_phi[j] = (ov <= DEFAULT_TOLERANCE_A)

    n_allowed = int(allowed_phi.sum())
    f_allowed = n_allowed / N_GRID

    if phi_symmetric and N_GRID % 2 == 0:
        half = N_GRID // 2
        # fold: allowed if either the first half or the second half is allowed
        folded = allowed_phi[:half] | allowed_phi[half:]
        n_folded = int(folded.sum())
        f_folded = n_folded / half
    else:
        n_folded = n_allowed
        f_folded = f_allowed

    return n_allowed, n_folded, f_allowed, f_folded


def process_one(pdb_id: str) -> dict:
    cif_path = CIF_DIR / f"{pdb_id.upper()}.cif"
    loaded   = load_structure(cif_path)
    cage     = build_cage(loaded)
    tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)
    phi_sym  = loaded.phi_symmetric

    t0 = time.perf_counter()

    # I-bond scans
    n_I_exp,  f_I_exp  = _scan_1d_tau(loaded.chrom_atoms, cage, phi_fixed=phi_exp)
    n_I_at0,  f_I_at0  = _scan_1d_tau(loaded.chrom_atoms, cage, phi_fixed=0.0)

    # P-bond scans
    n_P_exp, nf_P_exp, f_P_exp, ff_P_exp = _scan_1d_phi(
        loaded.chrom_atoms, cage, tau_fixed=tau_exp, phi_symmetric=phi_sym)
    n_P_at0, nf_P_at0, f_P_at0, ff_P_at0 = _scan_1d_phi(
        loaded.chrom_atoms, cage, tau_fixed=0.0,     phi_symmetric=phi_sym)

    elapsed = time.perf_counter() - t0

    return {
        "pdb_id":                pdb_id,
        "status":                "ok",
        "error":                 "",
        "chrom_resname":         loaded.chrom_resname,
        "phi_symmetric":         phi_sym,
        "tau_exp_deg":           tau_exp,
        "phi_exp_deg":           phi_exp,
        "f_allowed_I":           f_I_exp,
        "f_allowed_I_at0":       f_I_at0,
        "n_allowed_I":           n_I_exp,
        "n_allowed_I_at0":       n_I_at0,
        "f_allowed_P":           f_P_exp,
        "f_allowed_P_folded":    ff_P_exp,
        "f_allowed_P_at0":       f_P_at0,
        "f_allowed_P_at0_folded":ff_P_at0,
        "n_allowed_P":           n_P_exp,
        "n_allowed_P_folded":    nf_P_exp,
        "n_allowed_P_at0":       n_P_at0,
        "n_allowed_P_at0_folded":nf_P_at0,
        "f_product_folded":      f_I_exp * ff_P_exp,
        "scan_seconds":          elapsed,
    }


def fail_row(pdb_id: str, err: str) -> dict:
    row = {k: "" for k in FIELDS}
    row.update(pdb_id=pdb_id, status="fail", error=err[:300])
    return row


def main():
    df = pd.read_csv(MERGED_CSV)
    pdb_ids = sorted(df["pdb_id"].astype(str).str.upper().unique())
    print(f"Loaded {len(pdb_ids)} structures from merged_for_aggregate.csv")

    # Resume-safe
    done: set[str] = set()
    new_file = not OUT_CSV.is_file()
    if not new_file:
        try:
            ex = pd.read_csv(OUT_CSV)
            done = set(ex["pdb_id"].astype(str).str.upper())
            print(f"{len(done)} already done, {len(pdb_ids) - len(done)} to go")
        except Exception:
            done = set()

    fh = open(OUT_CSV, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file or len(done) == 0:
        writer.writeheader()
        fh.flush()

    n_ok = n_fail = n_skip = 0
    t_start = time.perf_counter()
    for k, pdb_id in enumerate(pdb_ids, 1):
        if pdb_id in done:
            n_skip += 1
            continue
        try:
            row = process_one(pdb_id)
            n_ok += 1
        except Exception:
            row = fail_row(pdb_id, traceback.format_exc(limit=1).strip())
            n_fail += 1
        writer.writerow(row)
        fh.flush()
        if n_ok % 50 == 0 or k <= 3:
            elapsed = time.perf_counter() - t_start
            done_so_far = n_ok + n_fail
            rate = done_so_far / elapsed if elapsed > 0 else 1
            eta  = (len(pdb_ids) - n_skip - done_so_far) / rate
            print(f"  [{k:>4d}/{len(pdb_ids)}]  ok={n_ok}  fail={n_fail}  "
                  f"eta={eta/60:.1f} min", flush=True)
    fh.close()
    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed/60:.1f} min: ok={n_ok} fail={n_fail} skip={n_skip}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
