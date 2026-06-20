#!/usr/bin/env python3
"""Figure S1 (supplementary, Section S1): deposited chromophore Megley torsions
phi vs tau for the full 838-structure cohort, phenol symmetry folded, with the
Ong et al. 2011 hula-twist diagonal (slope -0.85) and the avGFP / AausFP2
reference points. Publication styling: no title, panel label, vector output.

tau and phi are the production occupancy-ranked + canonical-CD values
(data/d_exp_canonical.csv: columns tau, phi_canon), so this map is consistent
with the d_exp_to_planar pipeline. phi_canon is already on the near-side ortho
carbon in (-90, 90]; fold_phi is idempotent on it.
Run from the project root.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCAN = "data/d_exp_canonical.csv"
COLORSRC = "../gfp-barrel-geometry/data/merged_complete_data.csv"
OUT = "figures/pub_figures/figS3_phi_vs_tau"

COLORS = {
    "green": "#2ca02c", "red": "#d62728", "yellow": "#e6c700",
    "cyan": "#17becf", "orange": "#ff7f0e", "blue": "#1f77b4", "": "#b0b0b0",
}
LABELS = {"": "unclassified"}
ORDER = ["", "orange", "blue", "cyan", "yellow", "red", "green"]


def fold_phi(p):
    return ((p + 90.0) % 180.0) - 90.0


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    color_of = {}
    for r in csv.DictReader(open(COLORSRC)):
        color_of[r["pdb_id"].strip().upper()] = r.get("color_class", "").strip()

    series, ref, aaus = {}, None, None
    n = 0
    for r in csv.DictReader(open(SCAN)):
        t, p = fnum(r["tau"]), fnum(r["phi_canon"])
        if t is None or p is None:
            continue
        pid = r["pdb_id"].strip().upper()
        p = fold_phi(p)
        c = color_of.get(pid, "")
        series.setdefault(c, [[], []])
        series[c][0].append(t)
        series[c][1].append(p)
        n += 1
        if pid == "1EMA":
            ref = (t, p)
        if pid == "6S68":
            aaus = (t, p)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Ong 2011 hula-twist diagonal, phi = -0.85 * tau, over the cluster region
    xs = [-80, 80]
    ax.plot(xs, [-0.85 * x for x in xs], "--", color="0.35", lw=1.4, zorder=2,
            label="Ong 2011 diagonal (slope -0.85)")

    for c in ORDER:
        if c not in series:
            continue
        x, y = series[c]
        ax.scatter(x, y, s=16, alpha=0.7, linewidths=0.3, edgecolors="white",
                   color=COLORS.get(c, "#333333"),
                   label=f"{LABELS.get(c, c)} (n={len(x)})", zorder=3)

    if ref is not None:
        ax.scatter(*ref, marker="*", s=300, color="black", edgecolors="white",
                   linewidths=0.8, zorder=6, label=f"avGFP 1EMA")
        ax.annotate("1EMA", ref, textcoords="offset points", xytext=(9, 7),
                    fontsize=9, fontweight="bold", zorder=6)
    if aaus is not None:
        ax.scatter(*aaus, marker="D", s=120, color="#c000c0", edgecolors="white",
                   linewidths=0.8, zorder=6, label="AausFP2 6S68 (QY=0)")
        ax.annotate("AausFP2", aaus, textcoords="offset points", xytext=(9, 5),
                    fontsize=9, fontweight="bold", color="#7a007a", zorder=6)

    for v in (-90, 0, 90):
        ax.axvline(v, color="0.9", lw=0.8, zorder=0)
    for v in (-45, 0, 45):
        ax.axhline(v, color="0.9", lw=0.8, zorder=0)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-90, 91, 45))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\tau$  (I-bond, N2-CA2-CB2-CG2)  [deg]")
    ax.set_ylabel(r"$\phi$  (P-bond, CA2-CB2-CG2-CD1), folded  [deg]")
    ax.text(0.012, 0.96, "S1", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92, markerscale=1.0,
              ncol=1)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".pdf")
    print(f"wrote {OUT}.png/.pdf  (n={n})")


if __name__ == "__main__":
    main()
