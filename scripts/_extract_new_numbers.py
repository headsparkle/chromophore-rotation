#!/usr/bin/env python3
import csv
from statistics import median
import numpy as np
from scipy import stats

# ---- matched-pair signtest table (look for permutation column) ----
print("=== matched_pairs_signtest.csv ===")
rows = list(csv.DictReader(open("data/matched_pairs_signtest.csv")))
print("cols:", list(rows[0].keys()))
for r in rows:
    print({k: r[k] for k in r})

# ---- LOOCV improved-protein count (M_base vs M_full abs error) ----
print("\n=== LOOCV improved count ===")
pr = list(csv.DictReader(open("data/loocv_predictions.csv")))
cols = pr[0].keys()
print("pred cols:", list(cols))

# ---- NRQ / CRQ / green with NEW d_exp ----
print("\n=== subfamilies (NEW d_exp) ===")
dexp = {r["pdb_id"].upper(): float(r["d_new"])
        for r in csv.DictReader(open("data/d_exp_altloc_fixed.csv"))}
meta = {r["pdb_id"].upper(): r for r in csv.DictReader(open("data/merged_for_aggregate.csv"))}
qy = {}
for r in csv.DictReader(open("data/lit_qy_curated.csv")):
    q = r["lit_qy_fpbase"] or r["lit_qy_fpbase_recovered"]
    try: q = float(q)
    except: q = None
    qy[r["pdb_id"].upper()] = (q, r["fpbase_slug"] or r["seq_match_slug"])

def corr(ct=None, color="red", unique=True):
    rows = []
    for pid, m in meta.items():
        if (m.get("color_class") or "").lower() != color: continue
        if ct and (m.get("chromophore_type") or "") != ct: continue
        if pid not in dexp: continue
        q, fp = qy.get(pid, (None, None))
        if q is None or q <= 0: continue
        rows.append((dexp[pid], q, fp))
    if unique:
        byfp = {}
        for d, q, fp in rows:
            if fp: byfp.setdefault(fp, []).append((d, q))
        d = np.array([median([x[0] for x in v]) for v in byfp.values()])
        lq = np.log10([median([x[1] for x in v]) for v in byfp.values()])
        n = len(byfp)
    else:
        d = np.array([r[0] for r in rows]); lq = np.log10([r[1] for r in rows]); n = len(rows)
    rho, p = stats.spearmanr(d, lq)
    return f"n={n} rho={rho:+.2f} p={p:.3f}"

print("NRQ per-unique:", corr(ct="NRQ"))
print("NRQ per-crystal:", corr(ct="NRQ", unique=False))
print("CRQ per-crystal:", corr(ct="CRQ", unique=False))
print("CRQ per-unique:", corr(ct="CRQ"))
print("green per-unique:", corr(color="green"))
