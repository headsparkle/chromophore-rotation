#!/usr/bin/env python3
"""
recompute_distances.py
======================

Add four columns to `data/scan_all_summary.csv` derived from the
saved per-structure (tau, phi) overlap grids:

- tau_centroid_deg, phi_centroid_deg : circular centroid of the
  sterically clash-free cells in (tau, phi) space.

- d_centroid_to_planar_deg : angular distance from that centroid
  to the nearest "planar" reference. Planar is defined per
  structure by snapping each of the experimental (tau, phi)
  components to the nearest of {0, +/-180}. This handles both
  Tyr-derived chromophores (planar at (0, 0)) and our histidine-
  derived chromophores (planar at (0, +/-180) under the CD1-
  substitute convention; see methods 2.2) without per-code
  bookkeeping. The metric measures how far the *centre of the
  allowed island* sits from the maximally emissive (fully planar)
  geometry. Because the centroid is set by the cage, this is a
  cage-intrinsic quantity.

- d_exp_to_planar_deg : angular distance from the experimental
  rotamer to the same planar reference. This measures how
  twisted the crystallographic chromophore is at the static
  level. For most FP structures this is small (a few degrees).

- d_exp_to_centroid_deg : angular distance from the experimental
  rotamer to the centroid of the allowed island. This measures
  how far the deposited chromophore sits from where the cage
  geometry alone would place it. Triangulates with the previous
  two: planar, centroid, and exp form a triangle on the (tau, phi)
  torus.

Distances are angular on the (-180, 180] torus, so a value of 0
deg means the centroid (or the experimental rotamer) is exactly at
the planar reference, and the maximum possible value is
sqrt(180^2 + 180^2) = 254.6 deg.

Runtime is fast (~ seconds for 838 structures). Idempotent on
re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCAN_DIR = DATA_DIR / "scans"
SUMMARY_CSV = DATA_DIR / "scan_all_summary.csv"
TOLERANCE_A = 0.4


def circular_mean_deg(x_deg: np.ndarray) -> float:
    """Circular mean on the (-180, 180] torus, in degrees."""
    rad = np.radians(x_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def torus_dist_deg(a: float, b: float) -> float:
    """Angular distance between two angles on (-180, 180] torus."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def snap_to_planar(exp_deg: float) -> float:
    """Snap a single dihedral component to the nearest planar
    reference, i.e. the closer of 0 deg or +/-180 deg."""
    cand = [0.0, 180.0, -180.0]
    return min(cand, key=lambda c: torus_dist_deg(exp_deg, c))


def centroid_of_allowed(
    overlap_map: np.ndarray,
    tau_grid: np.ndarray,
    phi_grid: np.ndarray,
    tolerance_a: float = TOLERANCE_A,
) -> tuple[float, float]:
    """Circular centroid of clash-free cells; (nan, nan) if none."""
    allowed = overlap_map <= tolerance_a
    if not allowed.any():
        return float("nan"), float("nan")
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    return circular_mean_deg(T[allowed]), circular_mean_deg(P[allowed])


def torus_distance_2d(
    a_tau: float, a_phi: float, b_tau: float, b_phi: float
) -> float:
    """Euclidean distance between two angular points, with each axis
    measured on the (-180, 180] torus."""
    return float(
        np.hypot(torus_dist_deg(a_tau, b_tau), torus_dist_deg(a_phi, b_phi))
    )


def main() -> int:
    df = pd.read_csv(SUMMARY_CSV)
    print(f"[load] {len(df)} rows ({(df['status']=='ok').sum()} ok)")

    n_done = 0
    n_skipped = 0
    n_no_allowed = 0
    for idx, row in df.iterrows():
        if row["status"] != "ok":
            n_skipped += 1
            continue
        npz_path = SCAN_DIR / f"scan_{str(row['pdb_id']).upper()}_free.npz"
        if not npz_path.is_file():
            n_skipped += 1
            continue
        with np.load(npz_path) as z:
            overlap_map = np.asarray(z["overlap_map"], dtype=float)
            tau_grid = np.asarray(z["tau_grid"], dtype=float)
            phi_grid = np.asarray(z["phi_grid"], dtype=float)

        tau_exp = float(row["tau_exp_deg"])
        phi_exp = float(row["phi_exp_deg"])

        # Planar reference for this structure
        planar_tau = snap_to_planar(tau_exp)
        planar_phi = snap_to_planar(phi_exp)

        # Centroid of the allowed island
        tau_c, phi_c = centroid_of_allowed(overlap_map, tau_grid, phi_grid)
        if np.isnan(tau_c):
            n_no_allowed += 1
            df.at[idx, "tau_centroid_deg"] = np.nan
            df.at[idx, "phi_centroid_deg"] = np.nan
            df.at[idx, "d_centroid_to_planar_deg"] = np.nan
            df.at[idx, "d_exp_to_centroid_deg"] = np.nan
        else:
            df.at[idx, "tau_centroid_deg"] = tau_c
            df.at[idx, "phi_centroid_deg"] = phi_c
            df.at[idx, "d_centroid_to_planar_deg"] = torus_distance_2d(
                tau_c, phi_c, planar_tau, planar_phi
            )
            df.at[idx, "d_exp_to_centroid_deg"] = torus_distance_2d(
                tau_exp, phi_exp, tau_c, phi_c
            )

        df.at[idx, "d_exp_to_planar_deg"] = torus_distance_2d(
            tau_exp, phi_exp, planar_tau, planar_phi
        )
        df.at[idx, "planar_tau_ref_deg"] = planar_tau
        df.at[idx, "planar_phi_ref_deg"] = planar_phi
        n_done += 1

    df.to_csv(SUMMARY_CSV, index=False)
    print(f"[done] updated: {n_done}, skipped: {n_skipped}, "
          f"no allowed island: {n_no_allowed}")
    # Quick summary of the new columns
    ok = df[df["status"] == "ok"]
    print(
        "\nNew column summaries (across ok rows):\n"
        f"  d_centroid_to_planar_deg  "
        f"median = {ok['d_centroid_to_planar_deg'].median():.2f},  "
        f"IQR = [{ok['d_centroid_to_planar_deg'].quantile(0.25):.2f}, "
        f"{ok['d_centroid_to_planar_deg'].quantile(0.75):.2f}]\n"
        f"  d_exp_to_planar_deg       "
        f"median = {ok['d_exp_to_planar_deg'].median():.2f},  "
        f"IQR = [{ok['d_exp_to_planar_deg'].quantile(0.25):.2f}, "
        f"{ok['d_exp_to_planar_deg'].quantile(0.75):.2f}]\n"
        f"  d_exp_to_centroid_deg     "
        f"median = {ok['d_exp_to_centroid_deg'].median():.2f},  "
        f"IQR = [{ok['d_exp_to_centroid_deg'].quantile(0.25):.2f}, "
        f"{ok['d_exp_to_centroid_deg'].quantile(0.75):.2f}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
