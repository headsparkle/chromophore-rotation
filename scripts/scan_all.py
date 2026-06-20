#!/usr/bin/env python3
"""
scan_all.py
===========

Run the freely-rotating Megley (tau, phi) scan on every fluorescent
protein crystal structure in Luke Begg's curated 908-FP dataset that
has a mature chromophore (has_chromophore == True in the master CSV).

The runner is resume-safe and incremental:

- CIF files are cached in `data/cif/<pdb>.cif` and downloaded from
  RCSB only on first encounter.
- Per-structure overlap grids are saved as compressed npz in
  `data/scans/scan_<pdb>_free.npz` (about 20 kB each compressed).
- One summary row per structure is appended to
  `data/scan_all_summary.csv` as soon as the structure finishes,
  so killing the process and re-running picks up where it stopped.
- PDB ids already present in the summary CSV are skipped.
- Per-structure errors are caught and recorded as a row with
  status=fail; they do not abort the run.

Usage:
    python3 scripts/scan_all.py                # full run
    python3 scripts/scan_all.py --limit 20     # first 20 structures only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import (  # noqa: E402
    DEFAULT_TOLERANCE_A,
    build_cage,
    load_structure,
    max_overlap,
)
from rotate import measure_megley, set_megley  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif"
SCAN_DIR = DATA_DIR / "scans"
CIF_DIR.mkdir(parents=True, exist_ok=True)
SCAN_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = (
    PROJECT_ROOT.parent / "gfp-barrel-geometry" / "data" / "merged_complete_data.csv"
)
OUT_CSV = DATA_DIR / "scan_all_summary.csv"

STEP_DEG = 5.0

SUMMARY_FIELDS = [
    "pdb_id",
    "status",
    "error",
    "chrom_resname",
    "n_chrom_atoms",
    "n_cage_atoms",
    "tau_exp_deg",
    "phi_exp_deg",
    "exp_cell_overlap_a",
    "exp_cell_allowed",
    "n_allowed",
    "n_total",
    "A_allowed_deg2",
    "f_allowed",
    "f_allowed_folded",
    "max_overlap_a",
    "min_overlap_a",
    "d_exp_to_clash_deg",
    "scan_seconds",
]


def download_cif(pdb_id: str, timeout: float = 30.0) -> Path:
    """Return a local CIF path, downloading from RCSB if needed."""
    out = CIF_DIR / f"{pdb_id.upper()}.cif"
    if out.is_file() and out.stat().st_size > 1024:
        return out
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    tmp = out.with_suffix(".cif.partial")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    tmp.write_bytes(data)
    tmp.rename(out)
    return out


def grid(step: float) -> np.ndarray:
    return np.arange(-180.0, 180.0, step)


def scan_overlap_map(loaded, cage, step_deg: float):
    tau_grid = grid(step_deg)
    phi_grid = grid(step_deg)
    overlap_map = np.empty((len(tau_grid), len(phi_grid)), dtype=float)
    base_atoms = loaded.chrom_atoms
    moving_names = cage.moving_names
    moving_radii = cage.moving_radii
    cage_xyz = cage.cage_xyz
    cage_radii = cage.cage_radii
    exclude = cage.exclude
    for i, tau in enumerate(tau_grid):
        for j, phi in enumerate(phi_grid):
            atoms_now = set_megley(base_atoms, float(tau), float(phi))
            moving_xyz = np.array(
                [atoms_now[n] for n in moving_names], dtype=float
            )
            ov, _ = max_overlap(
                moving_xyz, moving_radii, cage_xyz, cage_radii, exclude
            )
            overlap_map[i, j] = ov
    return tau_grid, phi_grid, overlap_map


def nearest_distance_deg(tau, phi, tau_grid, phi_grid, mask) -> float:
    if not mask.any():
        return float("nan")
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    dT = (T - tau + 180.0) % 360.0 - 180.0
    dP = (P - phi + 180.0) % 360.0 - 180.0
    d2 = dT * dT + dP * dP
    return float(np.sqrt(d2[mask].min()))


def folded_allowed_fraction(
    overlap_map: np.ndarray, tolerance_a: float
) -> float:
    """
    Allowed fraction after folding phi -> (phi mod 180), which collapses
    the two phenol-flip degenerate islands onto one. The grid has 72
    phi columns at 5 deg; we OR the first 36 columns with the last 36
    cell-by-cell so a rotamer is allowed if EITHER (tau, phi) or
    (tau, phi+180) is sterically allowed (i.e. the symmetric copy).
    """
    n_tau, n_phi = overlap_map.shape
    if n_phi % 2 != 0:
        return float("nan")
    half = n_phi // 2
    folded = np.minimum(overlap_map[:, :half], overlap_map[:, half:])
    allowed = folded <= tolerance_a
    return float(allowed.sum() / folded.size)


def summarise(
    pdb_id: str,
    loaded,
    cage,
    tau_grid: np.ndarray,
    phi_grid: np.ndarray,
    overlap_map: np.ndarray,
    tau_exp: float,
    phi_exp: float,
    scan_seconds: float,
) -> dict:
    allowed = overlap_map <= DEFAULT_TOLERANCE_A
    n_allowed = int(allowed.sum())
    n_total = overlap_map.size
    cell_area = STEP_DEG * STEP_DEG
    A_allowed = n_allowed * cell_area
    f_allowed = A_allowed / (360.0 * 360.0)
    i_exp = int(np.argmin(np.abs(((tau_grid - tau_exp + 180) % 360) - 180)))
    j_exp = int(np.argmin(np.abs(((phi_grid - phi_exp + 180) % 360) - 180)))
    exp_cell_overlap = float(overlap_map[i_exp, j_exp])
    exp_cell_allowed = bool(allowed[i_exp, j_exp])
    d_edge = nearest_distance_deg(
        tau_exp, phi_exp, tau_grid, phi_grid, ~allowed
    )
    return {
        "pdb_id": pdb_id,
        "status": "ok",
        "error": "",
        "chrom_resname": loaded.chrom_resname,
        "n_chrom_atoms": len(loaded.chrom_atoms),
        "n_cage_atoms": int(len(cage.cage_xyz)),
        "tau_exp_deg": tau_exp,
        "phi_exp_deg": phi_exp,
        "exp_cell_overlap_a": exp_cell_overlap,
        "exp_cell_allowed": exp_cell_allowed,
        "n_allowed": n_allowed,
        "n_total": n_total,
        "A_allowed_deg2": A_allowed,
        "f_allowed": f_allowed,
        "f_allowed_folded": folded_allowed_fraction(
            overlap_map, DEFAULT_TOLERANCE_A
        ),
        "max_overlap_a": float(overlap_map.max()),
        "min_overlap_a": float(overlap_map.min()),
        "d_exp_to_clash_deg": d_edge,
        "scan_seconds": scan_seconds,
    }


def fail_row(pdb_id: str, err: str) -> dict:
    row = {k: "" for k in SUMMARY_FIELDS}
    row["pdb_id"] = pdb_id
    row["status"] = "fail"
    row["error"] = err
    return row


def open_summary_writer(path: Path):
    """Return (file_handle, csv_writer, set_of_already_done_pdbs)."""
    done: set[str] = set()
    if path.is_file():
        try:
            existing = pd.read_csv(path)
            done = set(existing["pdb_id"].astype(str).str.upper())
        except Exception:
            done = set()
    new_file = not path.is_file() or len(done) == 0
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()
    return f, writer, done


def load_pdb_list(limit: int | None) -> list[str]:
    df = pd.read_csv(MASTER_CSV)
    df = df[df["has_chromophore"] == True]  # noqa: E712
    pdb_ids = sorted(df["pdb_id"].astype(str).str.upper().unique())
    if limit is not None:
        pdb_ids = pdb_ids[:limit]
    return pdb_ids


def process_one(pdb_id: str) -> dict:
    cif = download_cif(pdb_id)
    loaded = load_structure(cif)
    cage = build_cage(loaded)
    tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)
    t0 = time.perf_counter()
    tau_grid, phi_grid, overlap_map = scan_overlap_map(
        loaded, cage, STEP_DEG
    )
    scan_seconds = time.perf_counter() - t0
    npz_path = SCAN_DIR / f"scan_{pdb_id.upper()}_free.npz"
    np.savez_compressed(
        npz_path,
        tau_grid=tau_grid,
        phi_grid=phi_grid,
        overlap_map=overlap_map.astype(np.float32),
        tau_exp=tau_exp,
        phi_exp=phi_exp,
        tolerance_a=DEFAULT_TOLERANCE_A,
        step_deg=STEP_DEG,
        pdb_id=pdb_id.upper(),
    )
    return summarise(
        pdb_id, loaded, cage, tau_grid, phi_grid, overlap_map,
        tau_exp, phi_exp, scan_seconds,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N PDB ids (alphabetical)",
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="suppress per-structure progress lines",
    )
    args = ap.parse_args()

    pdb_ids = load_pdb_list(args.limit)
    fh, writer, done = open_summary_writer(OUT_CSV)
    print(
        f"[scan_all] {len(pdb_ids)} candidate PDBs from {MASTER_CSV.name}, "
        f"{len(done)} already done, "
        f"{len(pdb_ids) - len(done & set(pdb_ids))} to do"
    )

    started_at = time.perf_counter()
    n_ok = 0
    n_fail = 0
    n_skip = 0
    for k, pdb_id in enumerate(pdb_ids, start=1):
        if pdb_id in done:
            n_skip += 1
            continue
        try:
            row = process_one(pdb_id)
            n_ok += 1
        except urllib.error.HTTPError as e:
            row = fail_row(pdb_id, f"http {e.code}")
            n_fail += 1
        except Exception:
            row = fail_row(pdb_id, traceback.format_exc(limit=1).strip()[:300])
            n_fail += 1
        writer.writerow(row)
        fh.flush()
        if not args.quiet and (k % 10 == 0 or k <= 5):
            elapsed = time.perf_counter() - started_at
            todo = len(pdb_ids) - k
            print(
                f"  [{k:>4d}/{len(pdb_ids)}] {pdb_id}  "
                f"ok={n_ok} fail={n_fail} skip={n_skip}  "
                f"elapsed={elapsed/60:.1f} min  "
                f"eta={(elapsed/(k - n_skip)) * todo / 60:.1f} min"
                if (k - n_skip) > 0
                else f"  [{k:>4d}/{len(pdb_ids)}] {pdb_id}"
            )
    fh.close()
    elapsed = time.perf_counter() - started_at
    print(
        f"[scan_all] done in {elapsed/60:.1f} min: "
        f"ok={n_ok} fail={n_fail} skip={n_skip}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
