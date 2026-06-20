#!/usr/bin/env python3
"""
make_pub_figures.py
===================

Produce the six manuscript figures with consistent publication
styling. Writes to `figures/pub_figures/`. Idempotent:
re-running overwrites the existing PNGs.

Figure 1. 1EMA freely-rotating (tau, phi) allowed map.
Figure 2. (a) Distribution of f_allowed across 838 FPs.
          (b) f_allowed by color class.
Figure 3. (a) f_allowed vs chromophore-cage contact count.
          (b) f_allowed vs barrel minor axis.
Figure 4. Partial regression of log10(QY) on f_allowed_folded
          after color class and geometric controls.
Figure 5. NRQ vs CRQ within red FPs: f_allowed vs barrel
          minor axis, illustrating the within-color Simpson flip.
Figure 6. Forest plot of stratified Spearman rho.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PUB_DIR = PROJECT_ROOT / "figures" / "pub_figures"
PUB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_PALETTE = {
    "blue": "#3050d0",
    "cyan": "#1e90c8",
    "green": "#22aa33",
    "yellow": "#d4b400",
    "orange": "#e07020",
    "red": "#cc2030",
    "unknown": "#888888",
}
COLOR_ORDER = ["blue", "cyan", "green", "yellow", "orange", "red"]


def panel_label(ax, text: str, dx: float = -0.18, dy: float = 1.02) -> None:
    ax.text(
        dx, dy, text,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="right",
    )


def save(fig, name: str) -> None:
    out_png = PUB_DIR / f"{name}.png"
    out_pdf = PUB_DIR / f"{name}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  -> {out_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1. 1EMA tau-phi map
# ---------------------------------------------------------------------------

def figure_1() -> None:
    print("Figure 1: 1EMA allowed map")
    npz = np.load(DATA_DIR / "scan_1EMA_free.npz")
    tau_grid = npz["tau_grid"]
    phi_grid = npz["phi_grid"]
    overlap_map = np.asarray(npz["overlap_map"], dtype=float)
    tau_exp = float(npz["tau_exp"])
    phi_exp = float(npz["phi_exp"])
    tolerance = float(npz["tolerance_a"])
    step = float(tau_grid[1] - tau_grid[0])

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    extent = (
        tau_grid[0] - step / 2, tau_grid[-1] + step / 2,
        phi_grid[0] - step / 2, phi_grid[-1] + step / 2,
    )
    field = np.clip(overlap_map, -1.0, 3.0).T
    im = ax.imshow(
        field, origin="lower", extent=extent,
        cmap="RdYlGn_r", vmin=-1.0, vmax=3.0,
        aspect="equal", interpolation="nearest",
    )
    Tcell, Pcell = np.meshgrid(tau_grid, phi_grid, indexing="ij")
    ax.contour(
        Tcell, Pcell, overlap_map,
        levels=[tolerance], colors="black", linewidths=1.0,
    )
    ax.plot(
        tau_exp, phi_exp, marker="o", markersize=9,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.2, linestyle="",
    )
    ax.annotate(
        f"experimental\n({tau_exp:+.1f}°, {phi_exp:+.1f}°)",
        xy=(tau_exp, phi_exp), xytext=(60, 80),
        textcoords="offset points",
        fontsize=8, ha="left",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
    )
    ax.set_xlabel(r"$\tau_{\mathrm{megley}}$ (°)")
    ax.set_ylabel(r"$\varphi_{\mathrm{megley}}$ (°)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label("max steric overlap (Å)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("1EMA (avGFP): allowed (τ, φ) island")
    save(fig, "fig1_1ema_map")


# ---------------------------------------------------------------------------
# Figure 2. Distribution + per-color box
# ---------------------------------------------------------------------------

def figure_2(df: pd.DataFrame) -> None:
    print("Figure 2: f_allowed distribution and per-color box")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # Panel a: histogram
    vals = df["f_allowed_folded"].dropna()
    ax1.hist(
        vals, bins=40, color="#5099c0",
        edgecolor="black", linewidth=0.4, alpha=0.85,
    )
    ax1.set_xlabel(r"$f_{\mathrm{allowed}}$ (folded)")
    ax1.set_ylabel("number of FP structures")
    ax1.text(
        0.97, 0.92,
        f"n = {len(vals)}\nmedian = {vals.median():.3f}\nIQR = "
        f"[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="grey", linewidth=0.4, pad=4),
    )
    ax1.grid(alpha=0.25, linewidth=0.4)
    panel_label(ax1, "a")

    # Panel b: per-color box
    classes = [c for c in COLOR_ORDER if (df["color_class"] == c).any()]
    data = [
        df.loc[df["color_class"] == c, "f_allowed_folded"].dropna()
        for c in classes
    ]
    bp = ax2.boxplot(
        data, tick_labels=classes, patch_artist=True, showfliers=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.4,
                        markerfacecolor="grey", markeredgecolor="none"),
    )
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(COLOR_PALETTE[c])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.6)
    for cap in bp["caps"]:
        cap.set_linewidth(0.6)
    ax2.set_xlabel("color class")
    ax2.set_ylabel(r"$f_{\mathrm{allowed}}$ (folded)")
    # Annotate n above each box
    ymax = max(d.max() for d in data) * 1.06
    for i, (c, d) in enumerate(zip(classes, data)):
        ax2.text(i + 1, ymax, f"n={len(d)}", ha="center",
                 fontsize=7, color="black")
    ax2.set_ylim(top=ymax * 1.08)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.4)
    panel_label(ax2, "b")

    fig.tight_layout()
    save(fig, "fig2_distribution_and_color")


# ---------------------------------------------------------------------------
# Figure 3. f_allowed vs chrom_contacts and vs minor_axis
# ---------------------------------------------------------------------------

def _scatter_by_color(
    ax, df: pd.DataFrame, x: str, y: str, x_label: str, y_label: str,
) -> tuple[float, float, int]:
    sub = df[[x, y, "color_class"]].dropna()
    rho, p = stats.spearmanr(sub[x], sub[y])
    for c in COLOR_ORDER:
        s = sub[sub["color_class"] == c]
        if len(s) == 0:
            continue
        ax.scatter(
            s[x], s[y], s=12, alpha=0.55,
            color=COLOR_PALETTE[c], edgecolors="none",
            label=f"{c} (n={len(s)})",
        )
    unk = sub[~sub["color_class"].isin(COLOR_ORDER)]
    if len(unk):
        ax.scatter(
            unk[x], unk[y], s=10, alpha=0.35,
            color="#aaaaaa", edgecolors="none",
            label=f"unknown (n={len(unk)})",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linewidth=0.4)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.text(
        0.97, 0.95,
        f"ρ = {rho:+.3f}{sig}\np = {p:.1e}\nn = {len(sub)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="grey", linewidth=0.4, pad=4),
    )
    return float(rho), float(p), len(sub)


def figure_3(df: pd.DataFrame) -> None:
    print("Figure 3: f_allowed vs chrom_contacts and vs minor_axis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _scatter_by_color(
        ax1, df, "chrom_contacts", "f_allowed_folded",
        "chromophore-cage contact count",
        r"$f_{\mathrm{allowed}}$ (folded)",
    )
    panel_label(ax1, "a")
    _scatter_by_color(
        ax2, df, "minor_axis", "f_allowed_folded",
        "barrel minor axis (Å)",
        r"$f_{\mathrm{allowed}}$ (folded)",
    )
    panel_label(ax2, "b")
    ax2.legend(loc="upper left", fontsize=7, ncol=1, markerscale=1.5)
    fig.tight_layout()
    save(fig, "fig3_geometric_correlates")


# ---------------------------------------------------------------------------
# Figure 4. Partial regression QY on f_allowed_folded
# ---------------------------------------------------------------------------

def figure_4(df: pd.DataFrame) -> None:
    """Per-unique-FP f_allowed vs curated QY by colour class.

    Three panels (cyan, green, red) showing that the within-family cage-size
    signal exists only in cyan.  Curated FPbase QY required (canon_qy column).
    """
    print("Figure 4: per-uFP f_allowed vs curated QY by colour class")
    needed = ["fp_id", "canon_qy", "f_allowed_folded", "color_class"]
    sub = df[needed].dropna(subset=["canon_qy", "f_allowed_folded", "fp_id"]).copy()
    sub = sub[sub["canon_qy"] > 0]

    # One row per unique FP per colour class (median f_allowed, first QY)
    def make_ufp(color: str) -> pd.DataFrame:
        s = sub[sub["color_class"] == color]
        return (
            s.groupby("fp_id")
            .agg(f_allowed=("f_allowed_folded", "median"),
                 qy=("canon_qy", "first"))
            .reset_index()
        )

    cyan_ufp  = make_ufp("cyan")
    green_ufp = make_ufp("green")
    red_ufp   = make_ufp("red")

    def _rho_str(ufp: pd.DataFrame) -> str:
        if len(ufp) < 4:
            return "n too small"
        r, p = stats.spearmanr(ufp["f_allowed"], ufp["qy"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        return f"rho = {r:+.2f}{sig}\nn = {len(ufp)}"

    panels = [
        ("cyan",  cyan_ufp,  "CFP / cyan class"),
        ("green", green_ufp, "GFP / green class"),
        ("red",   red_ufp,   "RFP / red class"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, (color, ufp, title) in zip(axes, panels):
        ax.scatter(
            ufp["f_allowed"], ufp["qy"],
            s=40, alpha=0.85,
            color=COLOR_PALETTE[color],
            edgecolors="black", linewidths=0.4,
        )
        if len(ufp) >= 4:
            r, _ = stats.spearmanr(ufp["f_allowed"], ufp["qy"])
            if abs(r) > 0.15:
                slope, intercept = np.polyfit(ufp["f_allowed"], ufp["qy"], 1)
                xs = np.linspace(ufp["f_allowed"].min(), ufp["f_allowed"].max(), 40)
                ax.plot(xs, intercept + slope * xs,
                        color="black", linewidth=0.9, linestyle="--")
        ax.set_xlabel(r"$f_{\mathrm{allowed}}$ per unique FP (median)")
        ax.set_ylabel("curated quantum yield")
        ax.set_title(title)
        ax.text(
            0.97, 0.97, _rho_str(ufp),
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="grey", linewidth=0.4, pad=4),
        )
        ax.grid(alpha=0.25, linewidth=0.4)

    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    panel_label(axes[2], "c")
    fig.tight_layout()
    save(fig, "fig4_partial_regression_qy")


# ---------------------------------------------------------------------------
# Figure 5. NRQ vs CRQ in red
# ---------------------------------------------------------------------------

def figure_5(df: pd.DataFrame) -> None:
    print("Figure 5: NRQ vs CRQ inside red")
    red = df[df["color_class"] == "red"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    palette = {
        "NRQ": "#cc2030", "CRQ": "#cc7090", "CH6": "#dd9020",
        "CRU": "#aa4040",
    }
    # Plot non-highlighted subtypes grey first so they sit behind
    other = red[~red["chromophore_type"].isin(palette)]
    if len(other):
        ax.scatter(
            other["minor_axis"], other["f_allowed_folded"],
            s=18, alpha=0.4, color="#cccccc", edgecolors="none",
            label=f"other red ({len(other)})",
        )
    for ct in ["NRQ", "CRQ", "CH6", "CRU"]:
        s = red[red["chromophore_type"] == ct]
        if len(s) == 0:
            continue
        ax.scatter(
            s["minor_axis"], s["f_allowed_folded"],
            s=38, alpha=0.8,
            color=palette[ct],
            edgecolors="black", linewidths=0.4,
            label=f"{ct} (n={len(s)})",
        )

    # Subfamily centroids
    for ct, c in palette.items():
        s = red[red["chromophore_type"] == ct]
        if len(s) < 3:
            continue
        mx = s["minor_axis"].median()
        my = s["f_allowed_folded"].median()
        ax.plot(
            mx, my, marker="x", markersize=10,
            markeredgewidth=2.0, color=c,
        )
        ax.annotate(
            f"{ct} median",
            xy=(mx, my), xytext=(8, 8),
            textcoords="offset points",
            fontsize=7, color=c,
        )

    ax.set_xlabel("barrel minor axis (Å)")
    ax.set_ylabel(r"$f_{\mathrm{allowed}}$ (folded)")
    ax.set_title("Red FPs: NRQ (DsRed-lineage) vs CRQ (GFP-like)")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.0)
    ax.grid(alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    save(fig, "fig5_nrq_vs_crq")


# ---------------------------------------------------------------------------
# Figure 6. Stratified Spearman forest plot
# ---------------------------------------------------------------------------

def figure_6() -> None:
    print("Figure 6: stratified Spearman forest plot")
    by_color = pd.read_csv(DATA_DIR / "aggregate_correlations_by_color.csv")
    # Drop predictors with no valid global rho
    valid_pred = (
        by_color[by_color["color_class"] == "ALL"]
        .dropna(subset=["rho"])
        ["predictor"].tolist()
    )
    by_color = by_color[by_color["predictor"].isin(valid_pred)].copy()
    # Sort predictors by absolute global rho descending
    order = (
        by_color[by_color["color_class"] == "ALL"]
        .assign(absrho=lambda d: d["rho"].abs())
        .sort_values("absrho", ascending=True)
        ["predictor"].tolist()
    )
    classes_present = ["ALL"] + [
        c for c in COLOR_ORDER
        if (by_color["color_class"] == c).any()
    ]
    n_classes = len(classes_present)

    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(order) + 1.5))
    yticks, ylabels = [], []
    for i, pred in enumerate(order):
        block = by_color[by_color["predictor"] == pred]
        for j, c in enumerate(classes_present):
            r = block[block["color_class"] == c]
            if r.empty or pd.isna(r["rho"].iloc[0]):
                continue
            rho = float(r["rho"].iloc[0])
            n = int(r["n"].iloc[0])
            p = float(r["p_value"].iloc[0])
            y = i * (n_classes + 1.0) + j
            if c == "ALL":
                color = "#222222"
                marker = "D"
                size = 55
                edge_w = 0.8
            else:
                color = COLOR_PALETTE.get(c, "#888")
                marker = "o"
                size = max(10, 18 + 0.5 * np.sqrt(n))
                edge_w = 0
            ax.scatter(
                rho, y, s=size, color=color, marker=marker,
                edgecolors="black" if c == "ALL" else "none",
                linewidths=edge_w,
                alpha=0.95 if c == "ALL" else 0.85,
            )
            if p < 0.05:
                ax.text(
                    rho, y + 0.35, "*", color=color,
                    fontsize=8, ha="center", va="bottom",
                )
        yticks.append(i * (n_classes + 1.0) + (n_classes - 1) / 2)
        ylabels.append(pred)

    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.5, color="grey", linewidth=0.3, linestyle=":")
    ax.axvline(-0.5, color="grey", linewidth=0.3, linestyle=":")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(r"Spearman ρ with $f_{\mathrm{allowed}}$ (folded)")
    ax.set_xlim(-0.9, 0.9)
    # Custom legend
    legend_handles = []
    for c in classes_present:
        if c == "ALL":
            legend_handles.append(plt.Line2D(
                [0], [0], marker="D", color="#222222", linestyle="",
                markersize=7, markeredgecolor="black",
                markeredgewidth=0.8, label="all colors",
            ))
        else:
            legend_handles.append(plt.Line2D(
                [0], [0], marker="o", color=COLOR_PALETTE[c],
                linestyle="", markersize=6, label=c,
            ))
    ax.legend(
        handles=legend_handles, loc="lower right",
        fontsize=7, ncol=2, columnspacing=0.8,
    )
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    save(fig, "fig6_stratified_forest")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure S1. Stokes shift inside green vs across all colors
# ---------------------------------------------------------------------------

def figure_7(df: pd.DataFrame) -> None:
    """Red-class QY vs d_exp_to_planar, using curated FPbase QY."""
    print("Figure 7: red QY vs distance-to-planar (curated QY)")
    red = df[
        (df["color_class"] == "red")
        & df["canon_qy"].notna()
        & df["d_exp_to_planar_deg"].notna()
    ].copy()
    red = red[red["canon_qy"] > 0]
    red["log_qy"] = np.log10(red["canon_qy"])

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    palette = {
        "NRQ": "#cc2030", "CRQ": "#cc7090",
        "CH6": "#dd9020", "CRU": "#aa4040",
    }
    other = red[~red["chromophore_type"].isin(palette)]
    if len(other):
        ax.scatter(
            other["d_exp_to_planar_deg"], other["canon_qy"],
            s=18, alpha=0.4, color="#cccccc", edgecolors="none",
            label=f"other red (n={len(other)})",
        )
    for ct in ["NRQ", "CRQ", "CH6", "CRU"]:
        s = red[red["chromophore_type"] == ct]
        if len(s) == 0:
            continue
        ax.scatter(
            s["d_exp_to_planar_deg"], s["canon_qy"],
            s=42, alpha=0.85, color=palette[ct],
            edgecolors="black", linewidths=0.4,
            label=f"{ct} (n={len(s)})",
        )

    valid = red.dropna(subset=["d_exp_to_planar_deg", "log_qy"])
    slope, intercept = np.polyfit(
        valid["d_exp_to_planar_deg"], valid["log_qy"], 1
    )
    xs = np.linspace(
        valid["d_exp_to_planar_deg"].min(),
        valid["d_exp_to_planar_deg"].max(),
        50,
    )
    ax.plot(xs, 10 ** (intercept + slope * xs),
            color="black", linewidth=1.0,
            label="OLS in log(QY) (visual only)")

    def _rho_line(sub: pd.DataFrame, label: str) -> str:
        if len(sub) < 4:
            return f"{label}  n too small\n"
        r, p = stats.spearmanr(sub["d_exp_to_planar_deg"], sub["log_qy"])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        return f"{label}  rho={r:+.2f}{stars}  p={p:.2f}  n={len(sub)}\n"

    stat_text = (
        _rho_line(valid, "all red")
        + _rho_line(red[red["chromophore_type"] == "CRQ"], "CRQ    ")
        + _rho_line(red[red["chromophore_type"] == "NRQ"], "NRQ    ")
    ).rstrip()
    ax.text(
        0.97, 0.97, stat_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, family="monospace",
        bbox=dict(facecolor="white", edgecolor="grey", linewidth=0.4, pad=4),
    )
    ax.set_xlabel("exp rotamer distance to planar (deg)")
    ax.set_ylabel("curated quantum yield (FPbase)")
    ax.legend(loc="upper right", fontsize=7, markerscale=1.0,
              bbox_to_anchor=(1.0, 0.7))
    ax.grid(alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    save(fig, "fig7_red_qy_vs_d_planar")


def figure_s1(df: pd.DataFrame) -> None:
    print("Figure S1: Stokes shift within green vs pooled across colors")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Panel a: all colors, showing the misleading negative pooled rho
    sub_all = df[["stokes_shift", "f_allowed_folded", "color_class"]].dropna()
    rho_g, p_g = stats.spearmanr(sub_all["stokes_shift"], sub_all["f_allowed_folded"])
    for c in COLOR_ORDER:
        s = sub_all[sub_all["color_class"] == c]
        if len(s) == 0:
            continue
        ax1.scatter(
            s["stokes_shift"], s["f_allowed_folded"],
            s=12, alpha=0.5, color=COLOR_PALETTE[c],
            edgecolors="none",
            label=f"{c} (n={len(s)})",
        )
    # Add a robust regression line (Theil-Sen via numpy polyfit on rank-transformed)
    x = sub_all["stokes_shift"].rank().values
    y = sub_all["f_allowed_folded"].rank().values
    # Skip line fit; just show rho instead.
    ax1.set_xlabel("Stokes shift (nm)")
    ax1.set_ylabel(r"$f_{\mathrm{allowed}}$ (folded)")
    sig_g = "**" if p_g < 0.01 else ("*" if p_g < 0.05 else "")
    ax1.text(
        0.97, 0.95,
        f"all colors\nρ = {rho_g:+.3f}{sig_g}\np = {p_g:.1e}\nn = {len(sub_all)}",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="grey", linewidth=0.4, pad=4),
    )
    ax1.set_xlim(0, max(sub_all["stokes_shift"].max() * 1.05, 100))
    ax1.set_ylim(bottom=-0.005)
    ax1.grid(alpha=0.25, linewidth=0.4)
    ax1.legend(loc="upper right", fontsize=7, markerscale=1.5,
               bbox_to_anchor=(0.7, 1.0))
    panel_label(ax1, "a")

    # Panel b: green only - mechanistic positive correlation visible
    sub_green = df.loc[
        df["color_class"] == "green",
        ["stokes_shift", "f_allowed_folded"],
    ].dropna()
    rho_gr, p_gr = stats.spearmanr(sub_green["stokes_shift"], sub_green["f_allowed_folded"])
    ax2.scatter(
        sub_green["stokes_shift"], sub_green["f_allowed_folded"],
        s=18, alpha=0.55, color=COLOR_PALETTE["green"],
        edgecolors="none",
    )
    # OLS line on raw data (visual only; the test is Spearman)
    slope, intercept = np.polyfit(
        sub_green["stokes_shift"], sub_green["f_allowed_folded"], 1
    )
    xs = np.linspace(
        sub_green["stokes_shift"].min(),
        sub_green["stokes_shift"].max(),
        50,
    )
    ax2.plot(xs, intercept + slope * xs, color="black", linewidth=1.0)
    sig_gr = "**" if p_gr < 0.01 else ("*" if p_gr < 0.05 else "")
    ax2.text(
        0.97, 0.95,
        f"green only\nρ = {rho_gr:+.3f}{sig_gr}\np = {p_gr:.3f}\n"
        f"n = {len(sub_green)}",
        transform=ax2.transAxes, ha="right", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="grey", linewidth=0.4, pad=4),
    )
    ax2.set_xlabel("Stokes shift (nm)")
    ax2.set_ylabel(r"$f_{\mathrm{allowed}}$ (folded)")
    ax2.set_xlim(0, sub_green["stokes_shift"].max() * 1.08)
    ax2.set_ylim(bottom=-0.005)
    ax2.grid(alpha=0.25, linewidth=0.4)
    panel_label(ax2, "b")

    fig.tight_layout()
    save(fig, "figS1_stokes_within_green")


def main() -> int:
    df = pd.read_csv(DATA_DIR / "merged_for_aggregate.csv")
    df["color_class"] = (
        df["color_class"].fillna("unknown").astype(str).str.lower()
    )

    # Merge curated FPbase QY (canonical source for all QY figures)
    qy = pd.read_csv(DATA_DIR / "lit_qy_curated.csv")
    qy["canon_qy"] = qy["lit_qy_fpbase"].fillna(qy["lit_qy_fpbase_recovered"])
    qy["fp_id"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])
    df = df.merge(qy[["pdb_id", "canon_qy", "fp_id"]], on="pdb_id", how="left")

    figure_1()
    figure_2(df)
    figure_3(df)
    figure_4(df)
    figure_5(df)
    figure_6()
    figure_7(df)
    figure_s1(df)
    print(f"\nAll figures saved to {PUB_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
