#!/usr/bin/env python3
"""Scatter of deposited chromophore Megley torsions for every FP structure:
phi_megley (P-bond, CA2-CB2-CG2-CD1) vs tau_megley (I-bond, N2-CA2-CB2-CG2).
Coloured by color class. Run from the project root.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "../gfp-barrel-geometry/data/megley_dihedrals.csv"
OUT = "figures/phi_vs_tau_all_chromophores_folded"
REF_PDB = "1EMA"


def fold_phi(p):
    """Collapse the phenol 180-degree symmetry: map phi into (-90, 90]."""
    return ((p + 90.0) % 180.0) - 90.0

COLORS = {
    "green": "#2ca02c",
    "red": "#d62728",
    "yellow": "#e6c700",
    "cyan": "#17becf",
    "orange": "#ff7f0e",
    "blue": "#1f77b4",
    "": "#999999",
}
LABELS = {"": "unclassified"}


def main():
    rows = list(csv.DictReader(open(SRC)))
    series = {}
    ref = None
    for r in rows:
        try:
            t = float(r["tau_megley"])
            p = fold_phi(float(r["phi_megley"]))
        except ValueError:
            continue
        c = r["color_class"].strip()
        series.setdefault(c, [[], []])
        series[c][0].append(t)
        series[c][1].append(p)
        if r["pdb_id"].strip().upper() == REF_PDB:
            ref = (t, p)

    fig, ax = plt.subplots(figsize=(11, 6))
    # draw unclassified first (background), then colours, brightest last
    order = ["", "orange", "blue", "cyan", "yellow", "red", "green"]
    for c in order:
        if c not in series:
            continue
        xs, ys = series[c]
        ax.scatter(xs, ys, s=18, alpha=0.7, linewidths=0.3, edgecolors="white",
                   color=COLORS.get(c, "#333333"),
                   label=f"{LABELS.get(c, c)} (n={len(xs)})")

    if ref is not None:
        ax.scatter([ref[0]], [ref[1]], marker="*", s=320, color="black",
                   edgecolors="white", linewidths=0.8, zorder=6,
                   label=f"avGFP 1EMA ({ref[0]:.0f}, {ref[1]:.0f})")
        ax.annotate("1EMA", ref, textcoords="offset points", xytext=(10, 8),
                    fontsize=9, fontweight="bold", zorder=6)

    # AausFP2 (6S68): crosslinked, non-rotatable chromophore excluded from the
    # plotted cohort; tau_exp/phi_exp come from the scan summary table
    aaus = None
    for r in csv.DictReader(open("data/scan_all_summary.csv")):
        if r["pdb_id"].strip().upper() == "6S68":
            aaus = (float(r["tau_exp_deg"]), fold_phi(float(r["phi_exp_deg"])))
            break
    if aaus is not None:
        ax.scatter([aaus[0]], [aaus[1]], marker="D", s=130, color="#c000c0",
                   edgecolors="white", linewidths=0.8, zorder=6,
                   label=f"AausFP2 6S68 ({aaus[0]:.0f}, {aaus[1]:.0f}); crosslinked, QY=0")
        ax.annotate("AausFP2", aaus, textcoords="offset points", xytext=(10, 6),
                    fontsize=9, fontweight="bold", color="#7a007a", zorder=6)

    for v in (-90, 0, 90):
        ax.axvline(v, color="0.85", lw=0.8, zorder=0)
    for v in (-45, 0, 45):
        ax.axhline(v, color="0.85", lw=0.8, zorder=0)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-90, 91, 45))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\tau$ (I-bond, N2-CA2-CB2-CG2)  [deg]")
    ax.set_ylabel(r"$\phi$ (P-bond, CA2-CB2-CG2-CD1), folded to (-90, 90]  [deg]")
    ax.set_title("Deposited chromophore torsions "
                 f"(n={sum(len(v[0]) for v in series.values())}, phenol symmetry folded)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=1.0)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=200)
    fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png", "and .pdf")


if __name__ == "__main__":
    main()
