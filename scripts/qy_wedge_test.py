#!/usr/bin/env python3
"""
qy_wedge_test.py
================

For every FP scan in data/scans/scan_<pdb>_free.npz, compute the
fraction of allowed cells that fall in three +-15 deg axial wedges
around (a) the phi axis (pure phenol-spin / P-bond OBF), (b) the
positive-HT diagonal (+1 slope, Baffour-Awuah 2004's "complementary"
direction), and (c) the negative-HT diagonal (-1 slope, Megley 2009's
"bottom HT" direction). A fourth wedge along the tau axis is logged
but is essentially never populated.

Compare wedge fractions between the top-10 and bottom-10 quantum-yield
structures, both pooled and restricted to the green color class so
chromophore chemistry doesn't dominate.

Outputs
-------
- data/qy_wedge_test.csv          per-PDB wedge fractions + QY + class
- figures/qy_wedge_test.png       grouped bar chart of wedge fractions
- prints Mann-Whitney summary for each wedge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCAN_DIR = DATA_DIR / "scans"
FIG_DIR = PROJECT_ROOT / "figures"

MASTER_CSV = PROJECT_ROOT.parent / "gfp-barrel-geometry" / "data" / "merged_complete_data.csv"
SUMMARY_CSV = DATA_DIR / "scan_all_summary.csv"
CURATED_CSV = DATA_DIR / "lit_qy_curated.csv"

TOL = 0.4  # standard tolerance
HALF_WEDGE_DEG = 15.0
WEDGES = {
    "phi_axis":     (0.0, 1.0),                                       # P-bond spin
    "positive_HT":  (np.cos(np.deg2rad(+45)), np.sin(np.deg2rad(+45))),
    "negative_HT":  (np.cos(np.deg2rad(-45)), np.sin(np.deg2rad(-45))),
    "tau_axis":     (1.0, 0.0),
}


def wedge_fractions(npz_path: Path) -> dict:
    """Return wedge-cell fraction (of allowed cells) for each direction
    in WEDGES, plus the allowed-cell count and centroid."""
    z = np.load(npz_path)
    tau_g = z["tau_grid"]; phi_g = z["phi_grid"]
    overlap = z["overlap_map"]
    allowed = overlap <= TOL
    n_all = int(allowed.sum())
    out = {"n_allowed": n_all}
    if n_all < 3:
        for k in WEDGES:
            out[f"frac_{k}"] = np.nan
            out[f"n_{k}"] = 0
        out["tau_c"] = np.nan; out["phi_c"] = np.nan
        return out
    T, P = np.meshgrid(tau_g, phi_g, indexing="ij")
    Ta, Pa = T[allowed], P[allowed]
    tau_c = float(Ta.mean()); phi_c = float(Pa.mean())
    Tc, Pc = Ta - tau_c, Pa - phi_c
    ang = np.degrees(np.arctan2(Pc, Tc))
    for k, (ux, uy) in WEDGES.items():
        dir_deg = np.degrees(np.arctan2(uy, ux))
        diff = (ang - dir_deg) % 180.0
        axial = np.where(diff > 90.0, 180.0 - diff, diff)
        n_in = int((axial <= HALF_WEDGE_DEG).sum())
        out[f"n_{k}"] = n_in
        out[f"frac_{k}"] = n_in / n_all
    out["tau_c"] = tau_c; out["phi_c"] = phi_c
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qy-source", choices=("keyword", "fpbase"), default="fpbase",
        help=("'keyword' = original master CSV lit_qy column (sloppy "
              "keyword matches, 0.79 pin); 'fpbase' = curated "
              "lit_qy_fpbase from data/lit_qy_curated.csv"),
    )
    args = ap.parse_args()

    master = pd.read_csv(MASTER_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    df = summary.merge(
        master[["pdb_id", "lit_qy", "color_class"]],
        on="pdb_id", how="left",
    )
    if args.qy_source == "fpbase":
        if not CURATED_CSV.is_file():
            sys.exit(
                f"missing {CURATED_CSV} - run curate_qy_from_fpbase.py first"
            )
        curated = pd.read_csv(CURATED_CSV)[["pdb_id", "lit_qy_fpbase"]]
        df = df.merge(curated, on="pdb_id", how="left")
        df["lit_qy"] = df["lit_qy_fpbase"]
        print("[qy] using FPbase-curated lit_qy")
    else:
        print("[qy] using keyword-matched lit_qy")
    df = df[(df["status"] == "ok") & df["lit_qy"].notna()].copy()
    print(f"[merge] {len(df)} FPs with scan + QY")

    # Compute wedge fractions for every scan in the working set.
    rows = []
    for _, r in df.iterrows():
        pdb = r["pdb_id"]
        npz = SCAN_DIR / f"scan_{pdb}_free.npz"
        if not npz.is_file():
            continue
        w = wedge_fractions(npz)
        w["pdb_id"] = pdb
        w["lit_qy"] = r["lit_qy"]
        w["color_class"] = r["color_class"]
        w["A_allowed_deg2"] = r["A_allowed_deg2"]
        rows.append(w)
    work = pd.DataFrame(rows)
    # Drop structures with too-small allowed regions for wedge stats.
    work_full = work.copy()
    work = work[work["n_allowed"] >= 6].copy()
    print(f"[wedge] {len(work)} FPs with n_allowed >= 6")

    out_csv = DATA_DIR / "qy_wedge_test.csv"
    work_full.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}")

    # Helper: report two groups
    def compare(group_top: pd.DataFrame, group_bot: pd.DataFrame, label: str):
        print(f"\n=== {label} ===")
        print(f"  n_top = {len(group_top)}, n_bot = {len(group_bot)}")
        print(f"  QY range top: [{group_top['lit_qy'].min():.3f}, {group_top['lit_qy'].max():.3f}]")
        print(f"  QY range bot: [{group_bot['lit_qy'].min():.3f}, {group_bot['lit_qy'].max():.3f}]")
        print(f"  color_class top: {group_top['color_class'].value_counts().to_dict()}")
        print(f"  color_class bot: {group_bot['color_class'].value_counts().to_dict()}")
        for k in WEDGES:
            col = f"frac_{k}"
            t = group_top[col].dropna()
            b = group_bot[col].dropna()
            if len(t) < 2 or len(b) < 2:
                continue
            try:
                stat, p = mannwhitneyu(t, b, alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), float("nan")
            print(
                f"  frac_{k:<11s}  top med={t.median():.3f}  "
                f"bot med={b.median():.3f}  delta={t.median()-b.median():+.3f}  "
                f"MWU p={p:.3g}"
            )

    # Tie-broken QY ranking: primary by lit_qy, secondary by f_allowed
    # so we don't shuffle the huge cluster at 0.79 arbitrarily.
    work = work.sort_values(["lit_qy", "A_allowed_deg2"], ascending=[True, True]).reset_index(drop=True)
    pooled_bot = work.head(10)
    pooled_top = work.tail(10)
    compare(pooled_top, pooled_bot, "pooled top-10 vs bottom-10 by QY")

    green = work[work["color_class"] == "green"].copy()
    green_bot = green.head(10)
    green_top = green.tail(10)
    compare(green_top, green_bot, "within-green top-10 vs bottom-10 by QY")

    # Spearman correlation across the full QY-labelled set, and within
    # each color class with at least 30 members. This avoids the
    # top/bottom-10 split entirely and uses all 238 FPs.
    def spearman_block(subset: pd.DataFrame, label: str):
        if len(subset) < 10:
            return
        print(f"\n--- Spearman rho(QY, wedge fraction) on {label} (n = {len(subset)}) ---")
        for k in WEDGES:
            col = f"frac_{k}"
            x = subset["lit_qy"]
            y = subset[col]
            mask = x.notna() & y.notna()
            if mask.sum() < 10:
                continue
            r, p = spearmanr(x[mask], y[mask])
            print(f"  frac_{k:<11s}  rho = {r:+.3f}   p = {p:.3g}")

    spearman_block(work, "pooled")
    for cls in ("green", "red", "cyan", "yellow", "blue"):
        sub = work[work["color_class"] == cls]
        if len(sub) >= 30:
            spearman_block(sub, f"color_class = {cls}")

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    pairs = [
        ("pooled bot-10", pooled_bot),
        ("pooled top-10", pooled_top),
        ("green bot-10",  green_bot),
        ("green top-10",  green_top),
    ]
    wedge_keys = ["phi_axis", "positive_HT", "negative_HT", "tau_axis"]
    xs = np.arange(len(wedge_keys))
    w = 0.18
    colors = ["#cc4444", "#cc7744", "#44aacc", "#4477cc"]
    for i, (lab, grp) in enumerate(pairs):
        means = [grp[f"frac_{k}"].mean() for k in wedge_keys]
        ax.bar(xs + (i - 1.5) * w, means, width=w, label=lab, color=colors[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(wedge_keys)
    ax.set_ylabel(r"mean fraction of allowed cells in $\pm 15^\circ$ wedge")
    ax.set_title("Directional accessibility vs QY (top-10 vs bottom-10, pooled and within green)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.axhline(15 / 90, color="k", linewidth=0.5, linestyle=":",
               label="uniform expectation (15/90)")
    plt.tight_layout()
    out_png = FIG_DIR / "qy_wedge_test.png"
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[save] {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
