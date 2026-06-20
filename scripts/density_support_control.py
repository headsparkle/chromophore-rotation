"""
density_support_control.py
==========================

Refinement-restraint control, done the refinement-INDEPENDENT way.

Instead of trusting a re-refinement (PDB-REDO mishandles the exotic chromophore
residues; see redo_refinement_control.py), ask directly: is the DEPOSITED
chromophore geometry supported by the experimental electron density?

Pulls per-residue real-space fit (RSCC, RSR) for the chromophore residue from
the official wwPDB validation reports, then tests:

  (1) Distribution of chromophore RSCC/RSR across the red cohort.
  (2) Does deposited twist (d_exp_to_planar) track POOR density fit? If the twist
      were refinement noise, high-twist chromophores should fit density worse
      (low RSCC / high RSR).
  (3) Does the within-red d_exp/QY correlation SURVIVE among well-fit chromophores
      (RSCC >= 0.9, full single-conformer occupancy)? Survival = the twist that
      predicts QY is data-supported, so the QY result is not a restraint artifact.

Output: data/chrom_density_support.csv
"""
from __future__ import annotations

import gzip
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
VAL_DIR = DATA / "wwpdb_val"
VAL_URL = "https://files.rcsb.org/pub/pdb/validation_reports/{mid}/{pid}/{pid}_validation.xml.gz"


def fetch_validation(pid: str) -> bytes | None:
    VAL_DIR.mkdir(exist_ok=True)
    cache = VAL_DIR / f"{pid.lower()}_validation.xml"
    if cache.exists() and cache.stat().st_size > 200:
        return cache.read_bytes()
    pl = pid.lower()
    url = VAL_URL.format(mid=pl[1:3], pid=pl)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dens-control/1.0"})
        raw = gzip.decompress(urllib.request.urlopen(req, timeout=30).read())
        cache.write_bytes(raw)
        return raw
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None


def chrom_density(xml: bytes, resname: str):
    """Return (rscc, rsr, occ, n_altloc) for the chromophore residue, picking the
    highest-occupancy conformer (matches the production occupancy-ranked altloc)."""
    root = ET.fromstring(xml)
    cands = []
    for sg in root.iter("ModelledSubgroup"):
        if sg.get("resname") != resname:
            continue
        rscc = sg.get("rscc")
        rsr = sg.get("rsr")
        occ = sg.get("avgoccu")
        cands.append((
            float(rscc) if rscc else np.nan,
            float(rsr) if rsr else np.nan,
            float(occ) if occ else np.nan,
            sg.get("altcode") or " ",
        ))
    if not cands:
        return None
    # distinct altloc conformers of the chromophore
    alts = sorted({c[3] for c in cands if c[3] not in (" ", "")})
    n_alt = len(alts)
    # highest-occupancy conformer
    best = max(cands, key=lambda c: (c[2] if not np.isnan(c[2]) else -1))
    return best[0], best[1], best[2], n_alt


def main():
    red = pd.read_csv("/tmp/red_cohort.csv")
    rows = []
    for _, r in red.iterrows():
        pid = r["pdb_id"]
        xml = fetch_validation(pid)
        dens = chrom_density(xml, r["chrom_resname"]) if xml else None
        rows.append({
            "pdb_id": pid, "fp_id": r["fp_id"], "chrom_resname": r["chrom_resname"],
            "qy_curated": r["qy_curated"], "resolution": r["resolution"],
            "d_exp": r["d_exp_to_planar_deg"],
            "rscc": dens[0] if dens else np.nan,
            "rsr": dens[1] if dens else np.nan,
            "occ": dens[2] if dens else np.nan,
            "n_altloc": dens[3] if dens else np.nan,
        })
        d = rows[-1]
        print(f"  {pid} {r['chrom_resname']:>4}  d_exp={d['d_exp']:6.1f}  "
              f"RSCC={d['rscc'] if d['rscc']==d['rscc'] else float('nan'):.3f}  "
              f"RSR={d['rsr'] if d['rsr']==d['rsr'] else float('nan'):.3f}  "
              f"occ={d['occ']}  altloc={d['n_altloc']}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "chrom_density_support.csv", index=False)
    got = df.rscc.notna().sum()
    print(f"\nDensity metrics recovered for {got}/{len(df)} chromophores")
    print(f"Saved {DATA/'chrom_density_support.csv'}")

    d = df.dropna(subset=["rscc"]).copy()
    print("\n" + "=" * 78)
    print("(1) Chromophore real-space fit across the red cohort")
    print("=" * 78)
    print(f"  RSCC: median {d.rscc.median():.3f}, IQR {d.rscc.quantile(.25):.3f}-{d.rscc.quantile(.75):.3f}, "
          f"min {d.rscc.min():.3f}")
    print(f"  RSR : median {d.rsr.median():.3f}")
    print(f"  chromophores with RSCC>=0.9: {(d.rscc>=0.9).sum()}/{len(d)}")
    print(f"  modeled with alt-confs (n_altloc>=2): {(d.n_altloc>=2).sum()}")
    print(f"  occupancy<0.9 at best conformer: {(d.occ<0.9).sum()}")

    print("\n" + "=" * 78)
    print("(2) Does deposited twist track POOR density fit? (noise hypothesis)")
    print("=" * 78)
    for col, sign in [("rscc", "neg=>twist is noise"), ("rsr", "pos=>twist is noise")]:
        rho, p = stats.spearmanr(d.d_exp, d[col])
        print(f"  Spearman(d_exp, {col.upper():4s}) = {rho:+.3f} (p={p:.3f}, n={len(d)})   [{sign}]")

    def crystal(s):
        s = s.dropna(subset=["d_exp", "qy_curated"])
        rho, p = stats.spearmanr(s.d_exp, np.log10(s.qy_curated))
        return rho, p, len(s)

    def unique(s):
        g = s.dropna(subset=["d_exp", "qy_curated", "fp_id"]).groupby("fp_id").agg(
            x=("d_exp", "median"), q=("qy_curated", "median"))
        rho, p = stats.spearmanr(g.x, np.log10(g.q))
        return rho, p, len(g)

    print("\n" + "=" * 78)
    print("(3) Within-red d_exp/QY among WELL-FIT chromophores (data-supported twist)")
    print("=" * 78)
    full = df.dropna(subset=["d_exp", "qy_curated"])
    wellfit = d[(d.rscc >= 0.9) & ((d.n_altloc < 2) | d.n_altloc.isna()) & (d.occ >= 0.9)]
    rscc09 = d[d.rscc >= 0.9]
    for label, s in [("all red (baseline)", full),
                     ("RSCC>=0.9", rscc09),
                     ("RSCC>=0.9 & single-conf & occ>=0.9", wellfit)]:
        cr, cu = crystal(s), unique(s)
        print(f"  {label:36s} per-crystal rho={cr[0]:+.3f} p={cr[1]:.3f} n={cr[2]}"
              f"   per-unique rho={cu[0]:+.3f} p={cu[1]:.3f} n={cu[2]}")


if __name__ == "__main__":
    main()
