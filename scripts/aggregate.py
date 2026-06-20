#!/usr/bin/env python3
"""
aggregate.py
============

Cross-reference our per-structure (tau, phi) scan results with
Luke Begg's master FP catalogue and produce the headline analyses
for the manuscript:

- f_allowed by color class (box plot + summary table)
- f_allowed vs literature quantum yield, brightness, EC
- f_allowed vs chromophore-cage flexibility (B-factor ratio,
  chromophore B-factor)
- f_allowed vs barrel geometry (eccentricity, minor axis,
  convex area, barrel length) - relating to Luke's JCIM findings
- A correlation matrix and a per-color summary CSV

Outputs
-------
- data/aggregate_results.csv   one-line-per-color summary plus
                                ranked correlation table
- data/merged_for_aggregate.csv merged scan + master rows used
- figures/aggregate/*.png       all the plots

This script is safe to run repeatedly. It reads the current state
of scan_all_summary.csv; partial scans are fine for previewing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "aggregate"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCAN_CSV = DATA_DIR / "scan_all_summary.csv"
MASTER_CSV = (
    PROJECT_ROOT.parent / "gfp-barrel-geometry" / "data" / "merged_complete_data.csv"
)

COLOR_PALETTE = {
    "green": "#22aa33",
    "yellow": "#d4b400",
    "cyan": "#1e90c8",
    "blue": "#3050d0",
    "orange": "#e07020",
    "red": "#cc2030",
    "unknown": "#888888",
}
COLOR_ORDER = ["blue", "cyan", "green", "yellow", "orange", "red"]

# Megley-scan response variables we will report on
PRIMARY_RESPONSE = "f_allowed_folded"
RESPONSE_LABEL = {
    "f_allowed": "f_allowed (raw)",
    "f_allowed_folded": "f_allowed (180-deg folded)",
    "A_allowed_deg2": "allowed area (deg^2)",
    "d_exp_to_clash_deg": "exp distance to clash (deg)",
    "exp_cell_overlap_a": "exp-rotamer overlap (A)",
    "min_overlap_a": "global min overlap (A)",
    "d_centroid_to_planar_deg": "allowed-island centroid distance to planar (deg)",
    "d_exp_to_planar_deg": "exp rotamer distance to planar (deg)",
}

# Predictors we want to regress / correlate against
PREDICTORS = [
    ("lit_qy", "quantum yield", "negative ρ expected (more freedom -> more quenching)"),
    ("lit_brightness", "brightness (QY x EC / 1000)", "negative ρ expected"),
    ("lit_ec", "extinction coefficient", "weak / no expectation"),
    ("b_factor_ratio", "B-factor ratio (chrom / barrel)", "positive ρ expected (more flex -> more allowed)"),
    ("chrom_b_factor", "chromophore mean B (A^2)", "positive ρ expected"),
    ("barrel_b_factor", "barrel mean B (A^2)", "weak / mixed expectation"),
    ("eccentricity", "barrel eccentricity (Begg et al.)", "Luke: red FPs more elliptical"),
    ("minor_axis", "barrel minor axis (A)", "negative ρ plausible (smaller -> tighter)"),
    ("major_axis", "barrel major axis (A)", "weak / mixed"),
    ("convex_area", "barrel cross-section area (A^2)", "negative ρ plausible (smaller -> tighter)"),
    ("barrel_length", "barrel length (A)", "weak / mixed"),
    ("chrom_contacts", "chrom barrel-atom contacts", "negative ρ expected (more contacts -> tighter)"),
    ("resolution", "crystal resolution (A)", "control for refinement quality"),
    ("em_max", "emission wavelength (nm)", "Luke: red-shifted -> tighter barrel"),
    ("stokes_shift", "Stokes shift em_max - ex_max (nm)", "positive ρ expected (more freedom -> more excited-state relaxation)"),
    ("d_centroid_to_planar_deg", "allowed-island centroid distance to planar (deg)", "scan-derived; included for self-consistency"),
    ("d_exp_to_planar_deg", "exp rotamer distance to planar (deg)", "scan-derived; included for self-consistency"),
]


# ----------------------------------------------------------------------------
# Loading & merging
# ----------------------------------------------------------------------------

def load_merged() -> pd.DataFrame:
    if not SCAN_CSV.is_file():
        sys.exit(f"missing {SCAN_CSV} - run scan_all.py first")
    scans = pd.read_csv(SCAN_CSV)
    scans = scans[scans["status"] == "ok"].copy()
    scans["pdb_id"] = scans["pdb_id"].astype(str).str.upper()
    master = pd.read_csv(MASTER_CSV)
    master["pdb_id"] = master["pdb_id"].astype(str).str.upper()
    merged = scans.merge(master, on="pdb_id", how="left", suffixes=("", "_m"))
    # Normalise color class
    if "color_class" in merged.columns:
        merged["color_class"] = (
            merged["color_class"].fillna("unknown").astype(str).str.lower()
        )
        merged.loc[~merged["color_class"].isin(COLOR_ORDER), "color_class"] = "unknown"
    else:
        merged["color_class"] = "unknown"
    return merged


# ----------------------------------------------------------------------------
# Per-color summary
# ----------------------------------------------------------------------------

def per_color_summary(merged: pd.DataFrame, response: str) -> pd.DataFrame:
    rows = []
    for c in COLOR_ORDER + ["unknown"]:
        sub = merged[merged["color_class"] == c]
        if len(sub) == 0:
            continue
        rows.append({
            "color_class": c,
            "n": len(sub),
            f"mean_{response}": float(sub[response].mean()),
            f"median_{response}": float(sub[response].median()),
            f"std_{response}": float(sub[response].std()),
            "mean_lit_qy": float(sub["lit_qy"].mean()) if "lit_qy" in sub else np.nan,
            "mean_lit_brightness": float(sub["lit_brightness"].mean()) if "lit_brightness" in sub else np.nan,
            "mean_b_factor_ratio": float(sub["b_factor_ratio"].mean()) if "b_factor_ratio" in sub else np.nan,
            "mean_eccentricity": float(sub["eccentricity"].mean()) if "eccentricity" in sub else np.nan,
        })
    return pd.DataFrame(rows)


def plot_box_by_color(merged: pd.DataFrame, response: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    classes = [c for c in COLOR_ORDER if (merged["color_class"] == c).any()]
    data = [merged.loc[merged["color_class"] == c, response].dropna() for c in classes]
    bp = ax.boxplot(
        data, labels=classes, patch_artist=True, showfliers=True
    )
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(COLOR_PALETTE.get(c, "#888888"))
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel(RESPONSE_LABEL.get(response, response))
    ax.set_xlabel("color class")
    ax.set_title(f"{RESPONSE_LABEL.get(response, response)} by FP color class")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pairwise correlations & scatter plots
# ----------------------------------------------------------------------------

@dataclass
class CorrResult:
    response: str
    predictor: str
    n: int
    rho: float
    p_value: float
    note: str


def spearman_summary(
    merged: pd.DataFrame, response: str
) -> pd.DataFrame:
    rows = []
    for col, label, note in PREDICTORS:
        if col not in merged.columns:
            continue
        sub = merged[[response, col]].dropna()
        if len(sub) < 20:
            continue
        rho, p = stats.spearmanr(sub[response], sub[col])
        rows.append(CorrResult(response, col, len(sub), float(rho), float(p), note).__dict__)
    return pd.DataFrame(rows).sort_values("p_value")


def spearman_by_color(
    merged: pd.DataFrame, response: str, min_n: int = 10
) -> pd.DataFrame:
    """Spearman correlation of every predictor against the response,
    stratified by color class. Catches Simpson's-paradox situations
    where the global rho differs in sign from the within-class rho.

    Returns a long-format frame with one row per (predictor, color).
    The global rho (color_class = 'ALL') is included on top.
    """
    rows: list[dict] = []
    colors = ["ALL"] + COLOR_ORDER
    for col, label, _ in PREDICTORS:
        if col not in merged.columns:
            continue
        for c in colors:
            if c == "ALL":
                sub = merged[[response, col]].dropna()
            else:
                sub = merged.loc[
                    merged["color_class"] == c, [response, col]
                ].dropna()
            if len(sub) < min_n:
                rows.append({
                    "predictor": col,
                    "color_class": c,
                    "n": len(sub),
                    "rho": np.nan,
                    "p_value": np.nan,
                })
                continue
            rho, p = stats.spearmanr(sub[response], sub[col])
            rows.append({
                "predictor": col,
                "color_class": c,
                "n": len(sub),
                "rho": float(rho),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)


def plot_stratified_forest(
    by_color: pd.DataFrame, response: str, out: Path,
) -> None:
    """Forest plot: for each predictor, the global rho and the
    per-color rho stacked vertically. Easy to spot Simpson cases
    (global sign differs from within-class sign)."""
    rows = by_color.dropna(subset=["rho"]).copy()
    predictors = list(rows["predictor"].unique())
    fig, ax = plt.subplots(
        figsize=(8.5, max(4, 0.55 * len(predictors) + 1))
    )
    yticks = []
    ylabels = []
    classes_present = ["ALL"] + [
        c for c in COLOR_ORDER if (rows["color_class"] == c).any()
    ]
    n_classes = len(classes_present)
    # Place each predictor in its own block
    for i, pred in enumerate(predictors):
        block = rows[rows["predictor"] == pred]
        for j, c in enumerate(classes_present):
            r = block[block["color_class"] == c]
            if r.empty:
                continue
            rho = float(r["rho"].iloc[0])
            n = int(r["n"].iloc[0])
            p = float(r["p_value"].iloc[0])
            y = i * (n_classes + 1) + j
            color = "#222222" if c == "ALL" else COLOR_PALETTE.get(c, "#888")
            marker = "D" if c == "ALL" else "o"
            ax.scatter(
                rho, y, s=30 + 0.4 * np.sqrt(n) * 5,
                color=color, marker=marker, alpha=0.85,
                edgecolors="black" if c == "ALL" else "none",
                linewidths=1.0 if c == "ALL" else 0,
            )
            # Annotate n and significance
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            ax.text(
                rho + 0.02, y, f"{c} n={n}{sig}",
                fontsize=6, va="center",
            )
        yticks.append(i * (n_classes + 1) + (n_classes - 1) / 2)
        ylabels.append(pred)

    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.5, color="grey", linewidth=0.3, linestyle=":")
    ax.axvline(-0.5, color="grey", linewidth=0.3, linestyle=":")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel(f"Spearman rho with {response}")
    ax.set_xlim(-0.9, 0.9)
    ax.set_title(
        f"Stratified Spearman rho ({response})\n"
        f"diamond = all colors; circle = within color class"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def find_simpsons_flips(by_color: pd.DataFrame) -> pd.DataFrame:
    """Find (predictor, color_class) pairs where the within-color rho
    has a different sign than the global rho AND both are
    significant at p < 0.05 — i.e. cases where pooling across colors
    would mislead."""
    out = []
    for pred, grp in by_color.dropna(subset=["rho"]).groupby("predictor"):
        glob = grp[grp["color_class"] == "ALL"]
        if glob.empty:
            continue
        rho_g = float(glob["rho"].iloc[0])
        p_g = float(glob["p_value"].iloc[0])
        for _, r in grp[grp["color_class"] != "ALL"].iterrows():
            if r["p_value"] is None or np.isnan(r["p_value"]):
                continue
            if (
                np.sign(r["rho"]) != np.sign(rho_g)
                and r["p_value"] < 0.05
                and p_g < 0.05
            ):
                out.append({
                    "predictor": pred,
                    "color_class": r["color_class"],
                    "n_color": int(r["n"]),
                    "rho_color": float(r["rho"]),
                    "p_color": float(r["p_value"]),
                    "rho_global": rho_g,
                    "p_global": p_g,
                })
    return pd.DataFrame(out)


def plot_scatter(
    merged: pd.DataFrame, response: str, predictor: str,
    predictor_label: str, out: Path,
) -> tuple[float, float, int]:
    sub = merged[[response, predictor, "color_class"]].dropna()
    if len(sub) < 5:
        return float("nan"), float("nan"), len(sub)
    rho, p = stats.spearmanr(sub[response], sub[predictor])
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in COLOR_ORDER + ["unknown"]:
        s = sub[sub["color_class"] == c]
        if len(s) == 0:
            continue
        ax.scatter(
            s[predictor], s[response],
            s=20, alpha=0.6,
            color=COLOR_PALETTE.get(c, "#888888"),
            edgecolors="none",
            label=f"{c} (n={len(s)})",
        )
    ax.set_xlabel(predictor_label)
    ax.set_ylabel(RESPONSE_LABEL.get(response, response))
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.set_title(
        f"{response} vs {predictor}\n"
        f"Spearman rho = {rho:+.3f}  p = {p:.2e}  n = {len(sub)} {sig}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9, loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return rho, p, len(sub)


# ----------------------------------------------------------------------------
# Distribution and global health
# ----------------------------------------------------------------------------

def plot_histogram(merged: pd.DataFrame, response: str, out: Path) -> None:
    sub = merged[response].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(sub, bins=40, alpha=0.85, color="#5099c0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel(RESPONSE_LABEL.get(response, response))
    ax.set_ylabel("number of FP structures")
    ax.set_title(
        f"Distribution of {RESPONSE_LABEL.get(response, response)} "
        f"across {len(sub)} structures"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_correlation_matrix(merged: pd.DataFrame, out: Path) -> None:
    cols = [PRIMARY_RESPONSE] + [
        c for c, _, _ in PREDICTORS if c in merged.columns
    ]
    sub = merged[cols].dropna()
    if len(sub) < 20:
        return
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.5 + 0.5 * len(cols), 0.5 + 0.5 * len(cols)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title(f"Spearman correlations (n={len(sub)} structures with full data)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> int:
    merged = load_merged()
    n_total = len(merged)
    print(f"[load] {n_total} OK rows merged with master CSV")
    if n_total == 0:
        sys.exit("no OK rows yet, nothing to aggregate")

    # Persist the merged frame for downstream
    out_merged = DATA_DIR / "merged_for_aggregate.csv"
    merged.to_csv(out_merged, index=False)
    print(f"[save] merged frame -> {out_merged}")

    # Per-color summary
    pc = per_color_summary(merged, PRIMARY_RESPONSE)
    print("\n[per-color summary]")
    print(pc.to_string(index=False))
    pc.to_csv(DATA_DIR / "aggregate_per_color.csv", index=False)

    # Distributions
    plot_histogram(merged, PRIMARY_RESPONSE, FIG_DIR / "hist_f_allowed_folded.png")
    plot_histogram(merged, "f_allowed", FIG_DIR / "hist_f_allowed.png")
    plot_box_by_color(merged, PRIMARY_RESPONSE, FIG_DIR / "box_by_color.png")
    plot_box_by_color(merged, "min_overlap_a", FIG_DIR / "box_min_overlap_by_color.png")

    # Per-predictor scatter + Spearman test
    corrs = spearman_summary(merged, PRIMARY_RESPONSE)
    print(f"\n[Spearman correlations of {PRIMARY_RESPONSE}]")
    print(corrs[["predictor", "n", "rho", "p_value", "note"]].to_string(index=False))
    corrs.to_csv(DATA_DIR / "aggregate_correlations.csv", index=False)

    # Per-color stratified Spearman
    by_color = spearman_by_color(merged, PRIMARY_RESPONSE)
    by_color.to_csv(DATA_DIR / "aggregate_correlations_by_color.csv", index=False)
    print(f"\n[stratified Spearman by color class] ({len(by_color)} rows -> aggregate_correlations_by_color.csv)")
    # Wide-format preview
    wide = by_color.pivot(index="predictor", columns="color_class", values="rho")
    wide = wide[[c for c in (["ALL"] + COLOR_ORDER) if c in wide.columns]]
    print(wide.round(3).to_string())
    plot_stratified_forest(
        by_color, PRIMARY_RESPONSE,
        FIG_DIR / f"forest_stratified_{PRIMARY_RESPONSE}.png",
    )

    flips = find_simpsons_flips(by_color)
    if not flips.empty:
        print("\n[Simpson's-paradox candidates: within-color sign differs from global, both p<0.05]")
        print(flips.to_string(index=False))
        flips.to_csv(DATA_DIR / "aggregate_simpsons_flips.csv", index=False)

    for col, label, _ in PREDICTORS:
        if col not in merged.columns:
            continue
        out = FIG_DIR / f"scatter_{PRIMARY_RESPONSE}_vs_{col}.png"
        plot_scatter(merged, PRIMARY_RESPONSE, col, label, out)

    # Big correlation matrix
    plot_correlation_matrix(merged, FIG_DIR / "correlation_matrix.png")

    # Write a single combined results CSV
    out_results = DATA_DIR / "aggregate_results.csv"
    pc.assign(__section="per_color").to_csv(out_results, index=False)
    with open(out_results, "a") as f:
        f.write("\n\n--- spearman correlations against f_allowed_folded ---\n")
    corrs.to_csv(out_results, mode="a", index=False)
    print(f"\n[save] results -> {out_results}")
    print(f"[save] figures -> {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
