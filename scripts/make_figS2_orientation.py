#!/usr/bin/env python3
"""Regenerate the major-axis orientation figure (manuscript Figure S3; file
figS2_major_axis_distribution) with a clean polar panel whose axis labels sit
outside the bars (the φ-axis label was previously covered). Run from project root.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/pub_figures/figS2_major_axis_distribution"
BAR = "#3b78b5"
REF = "0.55"


def main():
    ang = np.array([float(r["major_axis_angle_deg"])
                    for r in csv.DictReader(open("data/scan_all_summary.csv"))
                    if r.get("major_axis_angle_deg") not in (None, "")])
    n = len(ang)

    fig = plt.figure(figsize=(11, 4.6))

    # ---- (a) linear histogram ----
    axh = fig.add_subplot(1, 2, 1)
    axh.hist(ang, bins=np.arange(-90, 91, 3), color=BAR, edgecolor="white",
             linewidth=0.3)
    refs = [(45, "+45° hula", "#2ca02c"), (-45, "−45° anti-hula", "#9467bd"),
            (90, "+90° φ axis", "#d62728"), (-90, "−90°", "#d62728"),
            (0, "0° τ axis", "#ff7f0e")]
    for x, lab, c in refs:
        axh.axvline(x, color=c, ls="--", lw=1.0, label=lab)
    axh.set_xlim(-90, 90)
    axh.set_xticks(range(-90, 91, 45))
    axh.set_xlabel("major-axis angle of allowed region (°)")
    axh.set_ylabel("number of FP structures")
    axh.set_title(f"(a) Major-axis orientation (n = {n})", fontsize=10, loc="left")
    axh.legend(fontsize=7, loc="upper left", framealpha=0.9)
    for s in ("top", "right"):
        axh.spines[s].set_visible(False)

    # ---- (b) polar rose (orientation is axial: plot θ and θ+180) ----
    axp = fig.add_subplot(1, 2, 2, projection="polar")
    edges = np.arange(-90, 91, 5)
    counts, _ = np.histogram(ang, bins=edges)
    centers = np.deg2rad((edges[:-1] + edges[1:]) / 2)
    w = np.deg2rad(5)
    for th in (centers, centers + np.pi):
        axp.bar(th, counts, width=w, color=BAR, edgecolor="white",
                linewidth=0.2, align="center")
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    rmax = counts.max()
    axp.set_ylim(0, rmax * 1.05)

    # reference directions
    for deg in (0, 45, 90, 135):
        for d in (deg, deg + 180):
            axp.plot([np.deg2rad(d)] * 2, [0, rmax * 1.02], color=REF, lw=0.8,
                     ls="--", zorder=1)
    axp.set_thetagrids(range(0, 360, 45), labels=[""] * 8)
    axp.set_rlabel_position(22)
    axp.tick_params(labelsize=7)
    axp.grid(alpha=0.3, linewidth=0.4)

    # axis labels placed outside the bars
    rlab = rmax * 1.20
    lab = {0: ("τ axis", "#ff7f0e"), 90: ("φ axis", "#d62728"),
           45: ("+45°", "#2ca02c"), 135: ("−45°", "#9467bd")}
    for deg, (txt, c) in lab.items():
        axp.text(np.deg2rad(deg), rlab, txt, color=c, ha="center", va="center",
                 fontsize=8, fontweight="bold")
    # short corner label so nothing collides with the top φ-axis label
    axp.annotate("(b) Polar view (axial)", xy=(0, 1.08), xycoords="axes fraction",
                 ha="left", va="bottom", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".pdf")
    print(f"wrote {OUT}.png/.pdf  (n={n}, peak bin={counts.max()})")


if __name__ == "__main__":
    main()
