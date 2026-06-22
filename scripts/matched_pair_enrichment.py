"""
matched_pair_enrichment.py
==========================

Reproduces the matched-pair gatekeeper enrichment reported in the Results
(Figure 3c) and Supplementary Section S6.

For a given gatekeeper atom, the enrichment is the fraction of CONCORDANT
P-bond pairs (the brighter member has the tighter P-bond cage) in which that
atom is a "bright-specific" wall, present in the brighter member's cage and
absent in the dimmer member's, divided by the same fraction among DISCORDANT
pairs. A fold > 1 means the atom is preferentially a wall in the brighter,
tighter-caged member.

Two cohorts are reported:
  - all_pairs   : chemistry-pooled. These are the inflated 3.7-fold (Thr203-CB)
                  and 9.6-fold (Ser205-OG) values the text flags as inflated by
                  cyan/green pairs (Thr203 walling an indole, not a phenol).
  - phenol_only : both members carry a phenol chromophore (green, yellow,
                  orange, or red). This is the value the text uses:
                  Thr203-CB 2.2-fold (Fisher p ~ 0.002); Ser205-OG falls to a
                  non-significant ~3.9-fold on only seven pairs (p ~ 0.27).
  Controls Thr62-CG2 / Tyr145-CE2 / His148-ND1 are not enriched (fold 0.4-0.8).

Inputs (project data/):
  gatekeeper_boundary.csv   per-structure 2D boundary wall atoms (one row per
                            wall point); protein wall-atom sets are built from it
  matched_pairs.csv         >=80% identity, dQY>=0.10 pairs + concordance flags
Output:
  data/matched_pair_enrichment.csv

Run from anywhere; paths are resolved relative to this file.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PHENOL = {"green", "yellow", "orange", "red"}
ATOMS = [
    (203, "CB", "Thr203-CB"),
    (205, "OG", "Ser205-OG"),
    (62, "CG2", "Thr62-CG2"),
    (145, "CE2", "Tyr145-CE2"),
    (148, "ND1", "His148-ND1"),
]


def main():
    b = pd.read_csv(DATA / "gatekeeper_boundary.csv").drop_duplicates(
        ["pdb_id", "tau_deg", "phi_deg", "resnum", "atomname"])
    mp = pd.read_csv(DATA / "matched_pairs.csv")

    # one representative structure per unique FP, and its protein wall-atom set
    fp2pdb = b.drop_duplicates("fp_id").set_index("fp_id")["pdb_id"].to_dict()
    bp = b[~b["is_chrom"].astype(bool)]
    walls = {pid: set(zip(g["resnum"], g["atomname"]))
             for pid, g in bp.groupby("pdb_id")}

    def has_wall(fp, key):
        pid = fp2pdb.get(fp)
        return pid in walls and key in walls[pid]

    def enrichment(key, phenol_only):
        df = mp[mp["f_allowed_P_folded_concordant"].notna()]
        if phenol_only:
            df = df[df["color_bright"].isin(PHENOL)
                    & df["color_dim"].isin(PHENOL)]

        def bright_specific(sub):
            n = h = 0
            for _, r in sub.iterrows():
                fb, fd = r["fp_id_bright"], r["fp_id_dim"]
                if fp2pdb.get(fb) not in walls or fp2pdb.get(fd) not in walls:
                    continue
                n += 1
                if has_wall(fb, key) and not has_wall(fd, key):
                    h += 1
            return h, n

        hc, nc = bright_specific(df[df["f_allowed_P_folded_concordant"] == 1])
        hd, nd = bright_specific(df[df["f_allowed_P_folded_concordant"] == 0])
        cc = hc / nc if nc else np.nan
        dd = hd / nd if nd else np.nan
        fold = cc / dd if dd else np.inf
        _, p = fisher_exact([[hc, nc - hc], [hd, nd - hd]])
        return dict(conc_hit=hc, conc_n=nc, conc_pct=100 * cc,
                    disc_hit=hd, disc_n=nd, disc_pct=100 * dd,
                    fold=fold, fisher_p=p)

    rows = []
    for cohort, ponly in [("all_pairs", False), ("phenol_only", True)]:
        print(f"\n=== {cohort} ===")
        for rn, at, label in ATOMS:
            r = enrichment((rn, at), ponly)
            r.update(cohort=cohort, atom=label)
            rows.append(r)
            print(f"  {label:11} conc {r['conc_hit']:3d}/{r['conc_n']:<3d} = "
                  f"{r['conc_pct']:4.1f}%   disc {r['disc_hit']:2d}/{r['disc_n']:<3d} = "
                  f"{r['disc_pct']:4.1f}%   fold {r['fold']:.1f}   "
                  f"Fisher p = {r['fisher_p']:.3g}")

    out = pd.DataFrame(rows)[["cohort", "atom", "conc_hit", "conc_n", "conc_pct",
                              "disc_hit", "disc_n", "disc_pct", "fold", "fisher_p"]]
    out.to_csv(DATA / "matched_pair_enrichment.csv", index=False)
    print("\nwrote data/matched_pair_enrichment.csv")


if __name__ == "__main__":
    main()
