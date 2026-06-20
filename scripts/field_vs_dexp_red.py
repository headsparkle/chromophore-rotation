"""
field_vs_dexp_red.py
====================

Test of the proposed electrostatic -> geometry mechanistic link across the full
red set: does the electric field along the chromophore C-O bond at the phenolate
oxygen (the Park-Rhee quantity) predict d_exp_to_planar (the resting twist)?

Compares the cheap Coulomb proxy (scan_all_summary.csv: efield_along_co_V_per_A,
simple and ff14 charges) against the higher-fidelity validated reference field
(esp_validation.csv: ealong_vac / ealong_scr), per crystal and per unique FP.

Finding (see log 2026-06-17): the proxy shows a significant correlation
(rho ~ +0.34) but it COLLAPSES under the validated reference (rho ~ +0.18,
p ~ 0.28, n=38 unique) -- the same proxy-unreliability seen in S4. The
dataset-level field->geometry link is therefore not established at the validated
level, so the geometric d_exp_to_planar remains model-free and the electrostatic
mechanism is cited from QM/MM (Pieri et al.) rather than claimed here.
"""
from __future__ import annotations
import csv
from statistics import median
from collections import defaultdict
from scipy.stats import spearmanr

DATA = "data"


def load():
    sa = {r["pdb_id"].upper(): r for r in csv.DictReader(open(f"{DATA}/scan_all_summary.csv"))}
    val = {r["pdb_id"].upper(): r for r in csv.DictReader(open(f"{DATA}/esp_validation.csv"))}
    can = {r["pdb_id"].upper(): float(r["d_canonical"]) for r in csv.DictReader(open(f"{DATA}/d_exp_canonical.csv"))}
    meta = {r["pdb_id"].upper(): r for r in csv.DictReader(open(f"{DATA}/merged_for_aggregate.csv"))}
    slug = {r["pdb_id"].upper(): (r["fpbase_slug"] or r["seq_match_slug"])
            for r in csv.DictReader(open(f"{DATA}/lit_qy_curated.csv"))}
    return sa, val, can, meta, slug


def fnum(x):
    try: return float(x)
    except Exception: return None


def main():
    sa, val, can, meta, slug = load()
    reds = [p for p, m in meta.items() if (m.get("color_class") or "").lower() == "red" and p in can]

    def crystal(getter):
        xs, ys = [], []
        for p in reds:
            x = getter(p)
            if x is not None:
                xs.append(x); ys.append(can[p])
        return (*spearmanr(xs, ys), len(xs))

    def unique(getter):
        byfp = defaultdict(list)
        for p in reds:
            x = getter(p)
            if x is not None:
                byfp[slug.get(p) or p].append((x, can[p]))
        xs = [median([a for a, _ in v]) for v in byfp.values()]
        ys = [median([b for _, b in v]) for v in byfp.values()]
        return (*spearmanr(xs, ys), len(xs))

    getters = {
        "proxy E.(C->O) simple": lambda p: fnum(sa[p].get("efield_along_co_V_per_A")) if p in sa else None,
        "proxy E.(C->O) ff14":   lambda p: fnum(sa[p].get("efield_along_co_V_per_A_ff14")) if p in sa else None,
        "reference E.(C->O) vac": lambda p: fnum(val[p].get("ealong_vac")) if p in val else None,
        "reference E.(C->O) scr": lambda p: fnum(val[p].get("ealong_scr")) if p in val else None,
    }
    print("field along C-O at O_P  vs  d_exp_to_planar (red FPs)")
    for name, g in getters.items():
        rc, pc, nc = crystal(g); ru, pu, nu = unique(g)
        print(f"  {name:24s}  crystal rho={rc:+.2f} p={pc:.3f} n={nc}  |  unique rho={ru:+.2f} p={pu:.3f} n={nu}")


if __name__ == "__main__":
    main()
