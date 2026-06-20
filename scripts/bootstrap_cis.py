"""
bootstrap_cis.py
================

Bootstrap 95% confidence intervals for every headline Spearman rho in the paper.
Resampling is over the analysis unit (unique FPs for per-unique correlations,
crystal structures for the per-crystal ones), 10000 resamples, percentile CI.
Spearman is rank-based so raw QY is used (== log10 QY for ranks).

Prints rho [CI] n for each, and writes data/bootstrap_cis.csv.
"""
from __future__ import annotations
import csv
from statistics import median
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

DATA = "data"
RNG = np.random.default_rng(20260617)
NBOOT = 10000


def load():
    d = {r["pdb_id"].upper(): float(r["d_canonical"])
         for r in csv.DictReader(open(f"{DATA}/d_exp_canonical.csv"))}
    qy, slug = {}, {}
    for r in csv.DictReader(open(f"{DATA}/lit_qy_curated.csv")):
        q = r["lit_qy_fpbase"] or r["lit_qy_fpbase_recovered"]
        try:
            q = float(q)
        except Exception:
            q = None
        qy[r["pdb_id"].upper()] = q
        slug[r["pdb_id"].upper()] = r["fpbase_slug"] or r["seq_match_slug"]
    meta = {r["pdb_id"].upper(): r for r in csv.DictReader(open(f"{DATA}/merged_for_aggregate.csv"))}
    s1 = {r["pdb_id"].upper(): r for r in csv.DictReader(open(f"{DATA}/scan_1d_summary.csv"))}
    hb = {r["pdb_id"].upper(): r for r in csv.DictReader(open(f"{DATA}/hb_contacts.csv"))}
    return d, qy, slug, meta, s1, hb


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def boot_ci(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rho = spearmanr(x, y)[0]
    n = len(x)
    rs = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = RNG.integers(0, n, n)
        rs[b] = spearmanr(x[idx], y[idx])[0]
    lo, hi = np.nanpercentile(rs, [2.5, 97.5])
    return rho, lo, hi, n


def main():
    d, qy, slug, meta, s1, hb = load()

    def rows(color=None, chrom=None, need_qy=True, tyr66=False):
        out = []
        for p, m in meta.items():
            if color and (m.get("color_class") or "").lower() != color:
                continue
            if chrom and m.get("chrom_resname") != chrom:
                continue
            q = qy.get(p)
            if need_qy and (q is None or q <= 0):
                continue
            out.append((p, m, q, slug.get(p)))
        return out

    def per_unique(rws, xget):
        byfp = defaultdict(list)
        for p, m, q, sl in rws:
            xv = xget(p, m)
            if xv is None or q is None:
                continue
            byfp[sl or p].append((xv, q))
        xs = [median([a for a, _ in v]) for v in byfp.values()]
        ys = [median([b for _, b in v]) for v in byfp.values()]
        return xs, ys

    def per_crystal(rws, xget):
        xs, ys = [], []
        for p, m, q, sl in rws:
            xv = xget(p, m)
            if xv is None:
                continue
            xs.append(xv); ys.append(q)
        return xs, ys

    dexp = lambda p, m: d.get(p)
    fall = lambda p, m: fnum(m.get("f_allowed_folded"))
    fallP = lambda p, m: fnum(s1.get(p, {}).get("f_allowed_P_folded"))
    bfac = lambda p, m: fnum(m.get("b_factor_ratio"))
    npol = lambda p, m: fnum(hb.get(p, {}).get("n_polar_O2_32"))

    specs = []
    specs.append(("red d_exp/QY per crystal", per_crystal(rows("red"), dexp)))
    specs.append(("red d_exp/QY per unique",  per_unique(rows("red"), dexp)))
    specs.append(("NRQ d_exp/QY per unique",  per_unique(rows("red", "NRQ"), dexp)))
    specs.append(("CRQ d_exp/QY per crystal", per_crystal(rows("red", "CRQ"), dexp)))
    specs.append(("green d_exp/QY per unique",per_unique(rows("green"), dexp)))
    specs.append(("cyan f_allowed/QY per unique",  per_unique(rows("cyan"), fall)))
    specs.append(("cyan f_allowed_P/QY per unique",per_unique(rows("cyan"), fallP)))
    specs.append(("b_factor_ratio/QY per unique",  per_unique(rows(), bfac)))
    specs.append(("n_polar_O2_32/QY per unique",   per_unique(rows(), npol)))

    out = []
    print(f"{'correlation':32s} {'rho':>6s}  {'95% CI':>18s}  n")
    for name, (xs, ys) in specs:
        rho, lo, hi, n = boot_ci(xs, ys)
        print(f"{name:32s} {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]  n={n}")
        out.append((name, round(rho, 3), round(lo, 3), round(hi, 3), n))
    with open(f"{DATA}/bootstrap_cis.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["correlation", "rho", "ci_lo", "ci_hi", "n"])
        w.writerows(out)
    print(f"\nwrote {DATA}/bootstrap_cis.csv")


if __name__ == "__main__":
    main()
