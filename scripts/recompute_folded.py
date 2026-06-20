#!/usr/bin/env python3
"""
recompute_folded.py
===================

Walk every successfully-scanned structure, detect whether its
chromophore ring has 180-degree phi-symmetry, and overwrite the
`f_allowed_folded` column accordingly:

- Phenol rings (CRO, CRF, ...): keep the existing 180-deg-folded
  value, which collapses the geometrically equivalent
  (tau, phi) and (tau, phi + 180) into one cell.
- Imidazole / indole rings (IIC, CRG, CSH, SWG, ...): replace the
  folded value with the unfolded `f_allowed`, because folding
  double-counts allowed area for asymmetric rings.

Adds a new boolean column `phi_symmetric` so downstream analysis
can distinguish the two cases at a glance.

The CIF cache and the npz overlap grids are reused; no re-scan is
needed. Runtime ~ minutes for ~838 structures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import DEFAULT_TOLERANCE_A, load_structure  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif"
SCAN_DIR = DATA_DIR / "scans"
SUMMARY_CSV = DATA_DIR / "scan_all_summary.csv"


def folded_min_phi(overlap_map: np.ndarray) -> np.ndarray:
    """Collapse the phi axis from 360 deg to 180 deg by taking the
    min overlap of (tau, phi) and (tau, phi + 180)."""
    n_phi = overlap_map.shape[1]
    if n_phi % 2 != 0:
        raise ValueError("phi grid must have an even number of cells")
    half = n_phi // 2
    return np.minimum(overlap_map[:, :half], overlap_map[:, half:])


def main() -> int:
    df = pd.read_csv(SUMMARY_CSV)
    print(f"[load] {len(df)} rows total ({(df['status']=='ok').sum()} ok)")

    new_folded = df["f_allowed_folded"].astype(float).copy()
    sym_col = pd.Series(np.nan, index=df.index, dtype=object)

    n_changed = 0
    n_kept = 0
    n_unfolded = 0
    n_skipped = 0
    for idx, row in df.iterrows():
        if row["status"] != "ok":
            n_skipped += 1
            continue
        pdb = str(row["pdb_id"]).upper()
        cif = CIF_DIR / f"{pdb}.cif"
        npz = SCAN_DIR / f"scan_{pdb}_free.npz"
        if not cif.is_file() or not npz.is_file():
            n_skipped += 1
            continue

        loaded = load_structure(cif)
        sym_col.at[idx] = bool(loaded.phi_symmetric)

        with np.load(npz) as z:
            overlap_map = np.asarray(z["overlap_map"], dtype=float)

        if loaded.phi_symmetric:
            folded = folded_min_phi(overlap_map)
            f = float((folded <= DEFAULT_TOLERANCE_A).sum() / folded.size)
            n_kept += 1
        else:
            allowed = overlap_map <= DEFAULT_TOLERANCE_A
            f = float(allowed.sum() / allowed.size)
            n_unfolded += 1

        if abs(float(new_folded.at[idx]) - f) > 1e-9:
            n_changed += 1
        new_folded.at[idx] = f

    df["f_allowed_folded"] = new_folded
    df["phi_symmetric"] = sym_col
    df.to_csv(SUMMARY_CSV, index=False)

    print(f"[done] phenol-fold kept    : {n_kept}")
    print(f"[done] asymmetric unfolded : {n_unfolded}")
    print(f"[done] rows changed        : {n_changed}")
    print(f"[done] rows skipped        : {n_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
