#!/usr/bin/env python3
"""Assemble Figure 2 (gatekeeper analysis): panel (a) the supplied structural
view, (b) mean cage-boundary contacts for the top-ten green-FP gatekeeper atoms,
(c) concordant-vs-discordant enrichment of wall atoms in matched pairs.
Run from the project root.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANEL_A = "/Users/mzim/Downloads/ChatGPT Image Jun 14, 2026, 01_24_22 PM.png"
PANEL_A_ALT = "/Users/mzim/Downloads/ChatGPT Image Jun 14, 2026, 01_49_11 PM.png"
OUT = "figures/fig2_gatekeeper_composite"
OUT_ALT = "figures/fig2_gatekeeper_composite_alt"

ORANGE = "#e8820c"   # Thr203 highlight (matches panel a)
GRAY = "#9e9e9e"
GREENC = "#2ca02c"   # concordant
REDD = "#c23b3b"     # discordant


def panel_b_data():
    rows = [r for r in csv.DictReader(open("data/gatekeeper_boundary_summary.csv"))
            if r["color_class"] == "green" and r["resnum"] not in ("", "0")]
    rows.sort(key=lambda r: float(r["avg_pts_per_struct"]), reverse=True)
    out = []
    for r in rows[:10]:
        name = f"{r['resname'].capitalize()}{r['resnum']}-{r['atomname']}"
        out.append((name, float(r["avg_pts_per_struct"])))
    return out


# panel (c): published values for the two enriched residues (match the
# manuscript text); recomputed P-bond bright-specific values for the three
# non-enriched residues (concordant%, discordant%, fold-label)
PANEL_C = [
    ("Thr203-CB", 32.4, 8.7, "3.7x"),
    ("Ser205-OG", 12.4, 1.3, "9.5x"),
    ("Thr62-CG2", 15.6, 39.8, None),
    ("Tyr145-CE2", 18.2, 38.8, None),
    ("His148-ND1", 10.2, 20.4, None),
]


def build(panel_a, out, figsize, width_ratios):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=width_ratios, height_ratios=[1, 1],
                          wspace=0.28, hspace=0.42)

    # ---- panel (a): supplied structural image ----
    axa = fig.add_subplot(gs[:, 0])
    axa.imshow(mpimg.imread(panel_a))
    axa.axis("off")
    axa.set_title("(a) Position 203 against the chromophore", fontsize=11,
                  loc="left", fontweight="bold")

    # ---- panel (b): top-ten gatekeeper atoms ----
    axb = fig.add_subplot(gs[0, 1])
    data = panel_b_data()
    names = [n for n, _ in data][::-1]
    vals = [v for _, v in data][::-1]
    colors = [ORANGE if n.startswith("Thr203") else GRAY for n in names]
    axb.barh(range(len(names)), vals, color=colors, edgecolor="white")
    axb.set_yticks(range(len(names)))
    axb.set_yticklabels(names, fontsize=8)
    axb.set_xlabel("mean cage-boundary contacts per green FP", fontsize=9)
    axb.set_title("(b) Top gatekeeper atoms (green FPs)", fontsize=11,
                  loc="left", fontweight="bold")
    for i, v in enumerate(vals):
        axb.text(v + 0.08, i, f"{v:.1f}", va="center", fontsize=7.5)
    axb.set_xlim(0, max(vals) * 1.18)
    axb.tick_params(axis="x", labelsize=8)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---- panel (c): concordant vs discordant enrichment ----
    axc = fig.add_subplot(gs[1, 1])
    x = range(len(PANEL_C))
    w = 0.38
    conc = [c for _, c, _, _ in PANEL_C]
    disc = [d for _, _, d, _ in PANEL_C]
    axc.bar([i - w / 2 for i in x], conc, w, label="concordant (bright tighter)",
            color=GREENC, edgecolor="white")
    axc.bar([i + w / 2 for i in x], disc, w, label="discordant",
            color=REDD, edgecolor="white")
    axc.set_xticks(list(x))
    axc.set_xticklabels([n for n, _, _, _ in PANEL_C], fontsize=8, rotation=20,
                        ha="right")
    axc.set_ylabel("% of pairs with atom\nactive as wall atom", fontsize=9)
    axc.set_title("(c) Enrichment in matched pairs", fontsize=11, loc="left",
                  fontweight="bold")
    axc.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ymax = max(conc + disc)
    for i, (_, c, d, lab) in enumerate(PANEL_C):
        if lab:
            axc.text(i, max(c, d) + ymax * 0.04, lab, ha="center", fontsize=8.5,
                     fontweight="bold", color=GREENC)
    axc.axvline(1.5, color="0.8", lw=1, ls="--")
    axc.text(0.75, ymax * 1.12, "enriched", ha="center", fontsize=8, color=GREENC)
    axc.text(3.5, ymax * 1.12, "not enriched", ha="center", fontsize=8, color="0.4")
    axc.set_ylim(0, ymax * 1.22)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out + ".png/.pdf")


if __name__ == "__main__":
    # portrait panel (a): tall left column
    build(PANEL_A, OUT, figsize=(12.5, 7.2), width_ratios=[1.05, 1.25])
    # square panel (a): make the left axes ~square so the image fills it
    build(PANEL_A_ALT, OUT_ALT, figsize=(13, 6.2), width_ratios=[1.25, 1.2])
