#!/usr/bin/env python3
"""
recompute_relaxed.py
====================

Re-threshold every saved (tau, phi) overlap grid at the relaxed
tolerance T_relaxed = 0.65 A. The relaxed criterion treats the
closest contact pair in each rotamer as being free to relax by
0.25 A, which is what the user's chromophore-photophysics
literature suggests is a reasonable soft tolerance for short cage
breathing. The closest contact is by construction also the
maximum overlap on the grid, so a 0.25 A pad on the closest
contact is operationally identical to globally raising the
MolProbity tolerance from 0.4 A to 0.65 A.

For each structure we compute and append to
`data/scan_all_summary.csv`:

- n_allowed_065, A_allowed_deg2_065, f_allowed_065 :
  raw allowed-cell counts and area at the relaxed tolerance.
- f_allowed_folded_065 :
  the phi-folded variant where the chromophore ring is
  2-fold symmetric (uses the same per-structure phi_symmetric
  flag we already store).
- d_centroid_to_planar_065_deg :
  angular distance from the relaxed-allowed centroid to the
  per-structure planar reference.
- major_axis_angle_deg :
  principal-axis orientation of the relaxed-allowed region in
  the (tau, phi) plane, in degrees, in (-90, 90]. Zero means
  elongated along the tau axis (constant phi); +-90 means
  elongated along phi; **+45 deg means elongated along
  tau = phi (the hula-twist / coupled-rotation axis)**.
- major_axis_aspect_ratio :
  sqrt(major eigenvalue / minor eigenvalue) of the (tau, phi)
  point cloud of allowed cells, a measure of how elongated
  the allowed region is. A circular blob gives 1.0; a thin
  strip gives a large number.

The slope-1 prediction from the Megley / Maddalo photophysics
work [4, 5] is that the chromophore's torsional PES has its
minimum elongated along tau = phi (the hula-twist coordinate),
so a cage that accommodates the chromophore's preferred motion
should produce allowed regions clustered around
major_axis_angle_deg = +45 deg (or -45 deg, depending on the
sign of the coupled motion).
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

TOL_STRICT = 0.4
TOL_RELAXED = 0.65
STEP_DEG = 5.0


def folded_min_phi(overlap_map: np.ndarray) -> np.ndarray:
    n_phi = overlap_map.shape[1]
    half = n_phi // 2
    return np.minimum(overlap_map[:, :half], overlap_map[:, half:])


def circular_mean_deg(x_deg: np.ndarray) -> float:
    rad = np.radians(x_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def torus_dist_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def snap_to_planar(exp_deg: float) -> float:
    return min([0.0, 180.0, -180.0], key=lambda c: torus_dist_deg(exp_deg, c))


def torus_distance_2d(a_tau, a_phi, b_tau, b_phi):
    return float(np.hypot(
        torus_dist_deg(a_tau, b_tau),
        torus_dist_deg(a_phi, b_phi),
    ))


def principal_orientation(
    allowed: np.ndarray, tau_grid: np.ndarray, phi_grid: np.ndarray,
    exp_tau: float, exp_phi: float,
) -> tuple[float, float]:
    """
    PCA on the allowed cells. Coordinates are unwrapped about the
    experimental rotamer so the island is locally contiguous.

    Returns (major_axis_angle_deg in (-90, 90], aspect_ratio).
    """
    if not allowed.any():
        return float("nan"), float("nan")
    T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    ts = T[allowed].astype(float)
    ps = P[allowed].astype(float)
    if len(ts) < 3:
        return float("nan"), float("nan")

    # Unwrap each axis around the experimental rotamer (modular shift
    # to (-180, +180] centered on exp). This keeps the local island
    # contiguous even when it straddles +/-180.
    def unwrap_around(values: np.ndarray, anchor: float) -> np.ndarray:
        shifted = ((values - anchor + 180.0) % 360.0) - 180.0
        return shifted + anchor

    ts_u = unwrap_around(ts, exp_tau)
    ps_u = unwrap_around(ps, exp_phi)

    pts = np.column_stack([ts_u - ts_u.mean(), ps_u - ps_u.mean()])
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major[1], major[0])))
    # Fold to (-90, 90] because a line has 180-deg ambiguity.
    if angle_deg > 90:
        angle_deg -= 180.0
    elif angle_deg <= -90:
        angle_deg += 180.0
    if eigvals[1] <= 1e-12:
        aspect = float("inf")
    else:
        aspect = float(np.sqrt(eigvals[0] / eigvals[1]))
    return angle_deg, aspect


def main() -> int:
    df = pd.read_csv(SUMMARY_CSV)
    n_ok = (df["status"] == "ok").sum()
    print(f"[load] {len(df)} rows ({n_ok} ok), relaxing tolerance "
          f"{TOL_STRICT} A -> {TOL_RELAXED} A")

    # New columns initialised
    new_cols = [
        "n_allowed_065", "A_allowed_deg2_065",
        "f_allowed_065", "f_allowed_folded_065",
        "d_centroid_to_planar_065_deg",
        "major_axis_angle_deg", "major_axis_aspect_ratio",
    ]
    for c in new_cols:
        df[c] = np.nan

    n_done = 0
    n_empty = 0
    for idx, row in df.iterrows():
        if row["status"] != "ok":
            continue
        pdb = str(row["pdb_id"]).upper()
        npz = SCAN_DIR / f"scan_{pdb}_free.npz"
        if not npz.is_file():
            continue
        with np.load(npz) as z:
            overlap_map = np.asarray(z["overlap_map"], dtype=float)
            tau_grid = np.asarray(z["tau_grid"], dtype=float)
            phi_grid = np.asarray(z["phi_grid"], dtype=float)

        cell_area = STEP_DEG * STEP_DEG
        allowed = overlap_map <= TOL_RELAXED
        n_allow = int(allowed.sum())
        A_allow = n_allow * cell_area
        f_allow = A_allow / (360.0 * 360.0)

        df.at[idx, "n_allowed_065"] = n_allow
        df.at[idx, "A_allowed_deg2_065"] = A_allow
        df.at[idx, "f_allowed_065"] = f_allow

        # Folded fraction (phi 180-deg fold only if Tyr-symmetric)
        phi_sym = bool(row.get("phi_symmetric", False))
        if phi_sym:
            folded = folded_min_phi(overlap_map)
            f_folded = float((folded <= TOL_RELAXED).sum() / folded.size)
        else:
            f_folded = f_allow
        df.at[idx, "f_allowed_folded_065"] = f_folded

        # Centroid distance to planar at the relaxed tolerance
        if not allowed.any():
            n_empty += 1
        else:
            T, P = np.meshgrid(tau_grid, phi_grid, indexing="ij")
            tau_c = circular_mean_deg(T[allowed])
            phi_c = circular_mean_deg(P[allowed])
            planar_tau = snap_to_planar(float(row["tau_exp_deg"]))
            planar_phi = snap_to_planar(float(row["phi_exp_deg"]))
            df.at[idx, "d_centroid_to_planar_065_deg"] = torus_distance_2d(
                tau_c, phi_c, planar_tau, planar_phi,
            )

        # Major-axis orientation of the relaxed-allowed region
        angle_deg, aspect = principal_orientation(
            allowed, tau_grid, phi_grid,
            float(row["tau_exp_deg"]),
            float(row["phi_exp_deg"]),
        )
        df.at[idx, "major_axis_angle_deg"] = angle_deg
        df.at[idx, "major_axis_aspect_ratio"] = aspect
        n_done += 1

    df.to_csv(SUMMARY_CSV, index=False)
    print(f"[done] {n_done} structures processed, "
          f"{n_empty} with no allowed cells at {TOL_RELAXED} A")

    ok = df[df["status"] == "ok"]
    print("\n--- relaxed (0.65 A) vs strict (0.4 A) f_allowed_folded ---")
    print(f"strict   median = {ok['f_allowed_folded'].median():.4f}, "
          f"IQR = [{ok['f_allowed_folded'].quantile(0.25):.4f}, "
          f"{ok['f_allowed_folded'].quantile(0.75):.4f}]")
    print(f"relaxed  median = {ok['f_allowed_folded_065'].median():.4f}, "
          f"IQR = [{ok['f_allowed_folded_065'].quantile(0.25):.4f}, "
          f"{ok['f_allowed_folded_065'].quantile(0.75):.4f}]")
    print(f"ratio    median(relaxed) / median(strict) = "
          f"{ok['f_allowed_folded_065'].median() / max(1e-9, ok['f_allowed_folded'].median()):.2f}x")
    print("\n--- major-axis orientation of the relaxed allowed region ---")
    ang = ok["major_axis_angle_deg"].dropna()
    print(f"  n with defined orientation: {len(ang)}")
    print(f"  median angle: {ang.median():+.1f} deg")
    print(f"  IQR: [{ang.quantile(0.25):+.1f}, {ang.quantile(0.75):+.1f}] deg")
    print(f"  fraction within +/-15 deg of +45 (slope ~ +1, hula): "
          f"{((ang > 30) & (ang < 60)).sum() / len(ang) * 100:.1f} %")
    print(f"  fraction within +/-15 deg of -45 (slope ~ -1, anti-hula): "
          f"{((ang > -60) & (ang < -30)).sum() / len(ang) * 100:.1f} %")
    print(f"  fraction within +/-15 deg of 0 (tau axis): "
          f"{((ang > -15) & (ang < 15)).sum() / len(ang) * 100:.1f} %")
    print(f"  fraction within +/-15 deg of +/-90 (phi axis): "
          f"{(((ang > 75) & (ang <= 90)) | ((ang >= -90) & (ang < -75))).sum() / len(ang) * 100:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
