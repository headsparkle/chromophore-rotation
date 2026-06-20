#!/usr/bin/env python3
"""Figure 3: red-FP quantum yield is associated with chromophore resting twist.
d_exp_to_planar vs QY (log scale). MAIN panel = the primary per-unique-FP
analysis (one median point per FP; NRQ filled circles / CRQ open squares /
other gray triangles, OLS-in-log regression, AausFP2 QY=0 endpoint labeled).
INSET = the per-crystal-structure view (all crystals, pseudoreplicated, gray),
shown for completeness. Run from the project root.
"""
import csv
from statistics import median
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data"
OUT = "figures/pub_figures/fig3_red_resting_qy"
CRIMSON = "#c11f2f"
GRAY = "#9a9a9a"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_red():
    """Return list of red-FP rows with d_exp, canon_qy, fp_id, chromophore_type.

    d_exp uses the production occupancy-ranked + canonical-CD values
    (data/d_exp_canonical.csv, column d_canonical), not the legacy
    last-conformer or CD1-labeling-dependent columns.
    """
    qy = {}
    for r in csv.DictReader(open(f"{DATA}/lit_qy_curated.csv")):
        q = fnum(r["lit_qy_fpbase"])
        if q is None:
            q = fnum(r["lit_qy_fpbase_recovered"])
        fp = r["fpbase_slug"] or r["seq_match_slug"]
        qy[r["pdb_id"].upper()] = (q, fp)
    dexp = {r["pdb_id"].upper(): fnum(r["d_canonical"])
            for r in csv.DictReader(open(f"{DATA}/d_exp_canonical.csv"))}
    rows = []
    for r in csv.DictReader(open(f"{DATA}/merged_for_aggregate.csv")):
        if (r.get("color_class") or "").strip().lower() != "red":
            continue
        d = dexp.get(r["pdb_id"].upper())
        if d is None:
            continue
        q, fp = qy.get(r["pdb_id"].upper(), (None, None))
        rows.append(dict(pdb_id=r["pdb_id"], d=d, qy=q, fp=fp,
                         ct=(r.get("chrom_resname") or "").strip()))
    return rows


def main():
    from collections import Counter
    red = load_red()
    pos = [r for r in red if r["qy"] is not None and r["qy"] > 0]

    # per-crystal view (support / inset)
    dx = np.array([r["d"] for r in pos])
    logq = np.log10([r["qy"] for r in pos])
    rho_c, p_c = stats.spearmanr(dx, logq)
    print(f"inset (per crystal): n={len(pos)}  rho={rho_c:+.2f}  p={p_c:.1e}")

    # per-unique-FP analysis (primary / main), subfamily = mode of chrom_resname
    byfp = {}
    for r in pos:
        if r["fp"]:
            byfp.setdefault(r["fp"], []).append(r)
    uni = []
    for v in byfp.values():
        sub = Counter(x["ct"] for x in v).most_common(1)[0][0]
        uni.append(dict(d=median([x["d"] for x in v]),
                        qy=median([x["qy"] for x in v]),
                        sub=sub if sub in ("NRQ", "CRQ") else "other"))
    ufp_d = np.array([u["d"] for u in uni])
    ufp_logq = np.log10([u["qy"] for u in uni])
    rho_u, p_u = stats.spearmanr(ufp_d, ufp_logq)
    print(f"main (per unique FP): n={len(uni)}  rho={rho_u:+.2f}  p={p_u:.2f}")

    # AausFP2 (6S68): QY=0 endpoint, fetched directly (excluded from the regression)
    aaus = None
    for r in csv.DictReader(open(f"{DATA}/merged_for_aggregate.csv")):
        if r["pdb_id"].upper() == "6S68":
            aaus = fnum(r["d_exp_to_planar_deg"])
            break
    print("AausFP2:", f"d_exp={aaus:.1f}" if aaus is not None else "missing")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    # ---- MAIN panel: per-unique-FP medians, by subfamily ----
    def pick(sub):
        return ([u["d"] for u in uni if u["sub"] == sub],
                [u["qy"] for u in uni if u["sub"] == sub])
    od, oq = pick("other")
    cd, cq = pick("CRQ")
    nd, nq = pick("NRQ")
    ax.scatter(od, oq, marker="^", s=40, color=GRAY, edgecolors="none", alpha=0.75,
               label="other red")
    ax.scatter(cd, cq, marker="s", s=52, facecolors="none", edgecolors=CRIMSON,
               linewidths=1.4, label="CRQ")
    ax.scatter(nd, nq, marker="o", s=48, color=CRIMSON, edgecolors="white",
               linewidths=0.5, label="NRQ")

    sl, ic = np.polyfit(ufp_d, ufp_logq, 1)
    xmax = max(ufp_d.max(), (aaus if aaus is not None else 46)) + 3
    xs = np.linspace(0, xmax, 100)
    ax.plot(xs, 10 ** (ic + sl * xs), color="black", lw=1.1,
            label="OLS on log10(QY)")

    ax.set_yscale("log")
    ax.set_ylim(0.012, 1.05)
    ax.set_xlim(-2, xs.max())

    # AausFP2 (QY=0): off the log axis -> mark at the floor with a label
    if aaus is not None:
        yfloor = 0.0135
        ax.scatter([aaus], [yfloor], marker="x", s=70, color="black", zorder=6,
                   linewidths=1.8)
        ax.annotate("AausFP2\n(QY = 0)", (aaus, yfloor), textcoords="offset points",
                    xytext=(-6, 16), fontsize=8, fontweight="bold", ha="center",
                    color="black")

    ax.set_title(f"Red FPs: resting twist vs quantum yield\n"
                 f"per unique FP, primary: "
                 f"$\\rho$ = {rho_u:+.2f}, p = {p_u:.2f}, n = {len(uni)}",
                 fontsize=10)
    ax.set_xlabel("chromophore distance from planar, $d_{\\mathrm{exp}}$ (deg)")
    ax.set_ylabel("quantum yield (FPbase)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25, linewidth=0.4, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- INSET: per-crystal-structure view (support, pseudoreplicated), gray ----
    axi = ax.inset_axes([0.62, 0.55, 0.35, 0.30])
    axi.scatter(dx, np.array([r["qy"] for r in pos]), s=10, color=GRAY,
                edgecolors="none", alpha=0.7)
    sl2, ic2 = np.polyfit(dx, logq, 1)
    xs2 = np.linspace(dx.min(), dx.max(), 50)
    axi.plot(xs2, 10 ** (ic2 + sl2 * xs2), color="black", lw=0.9)
    axi.set_yscale("log")
    axi.set_title(f"per crystal structure (n={len(pos)})\n$\\rho$={rho_c:+.2f}",
                  fontsize=7)
    axi.tick_params(labelsize=6)
    axi.set_xlabel("$d_{\\mathrm{exp}}$ (deg)", fontsize=6, labelpad=1)

    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png/.pdf")


if __name__ == "__main__":
    main()
