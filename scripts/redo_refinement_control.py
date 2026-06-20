"""
redo_refinement_control.py
==========================

Refinement-restraint control for the within-red d_exp_to_planar / QY result.

Concern (referee-anticipated): the deposited chromophore dihedral is shaped by
the depositor's geometry-restraint dictionary, so "more twisted red chromophores
are dimmer" might be an artifact of refinement protocol / resolution rather than
real ground-state geometry.

Control: PDB-REDO independently re-refines every entry against the ORIGINAL
structure factors using a single, modern, consistent restraint model. If the
within-red d_exp/QY correlation survives recomputation on PDB-REDO geometry,
the twist is data-supported, not a restraint artifact.

Tests:
  (a) agreement between deposited and PDB-REDO chromophore twist;
  (b) does the deposited->REDO twist change grow with worse resolution?
  (c) re-run the within-red d_exp/QY Spearman on PDB-REDO geometry.

Recompute uses the SAME canonical-CD routine as the production pipeline
(barrel.load_structure + rotate.dihedral + canonical_phi), so deposited and
REDO numbers are produced by identical code.

Outputs: data/redo_d_exp.csv
"""
from __future__ import annotations

import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import barrel
import rotate
from canonical_cd_twist import signed_dev, canonical_phi, ortho_carbons

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
CIF_DIR = DATA / "cif"
REDO_DIR = DATA / "redo"
REDO_URL = "https://pdb-redo.eu/db/{pid}/{pid}_final.cif"


def fetch_redo(pid: str) -> Path | None:
    """Download the PDB-REDO final mmCIF for pid (lowercase). Returns path or None."""
    REDO_DIR.mkdir(exist_ok=True)
    out = REDO_DIR / f"{pid.lower()}_final.cif"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = REDO_URL.format(pid=pid.lower())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "redo-control/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1000:
            return None
        out.write_bytes(data)
        return out
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def twist_from_cif(cif: Path):
    """(tau, phi_canon, d_exp_to_planar) via the production canonical-CD rule, or None."""
    try:
        L = barrel.load_structure(cif)
    except Exception:
        return None
    a = L.chrom_atoms
    if not all(k in a for k in ("N2", "CA2", "CB2", "CG2")):
        return None
    tau = rotate.dihedral(a["N2"], a["CA2"], a["CB2"], a["CG2"])
    cs = ortho_carbons(a, L.chrom_elements)
    phi_cands = [rotate.dihedral(a["CA2"], a["CB2"], a["CG2"], a[c]) for c in cs]
    if not phi_cands:
        return None
    phi_canon, _ = canonical_phi(phi_cands)
    d = float(np.hypot(abs(signed_dev(tau)), abs(phi_canon)))
    return float(tau), float(phi_canon), d


def spearman_log(d: pd.DataFrame, col: str):
    s = d.dropna(subset=[col, "qy_curated"])
    if len(s) < 4:
        return (np.nan, np.nan, len(s))
    rho, p = stats.spearmanr(s[col], np.log10(s["qy_curated"]))
    return (rho, p, len(s))


def per_unique(d: pd.DataFrame, col: str):
    g = (d.dropna(subset=[col, "qy_curated", "fp_id"])
           .groupby("fp_id").agg(x=(col, "median"), q=("qy_curated", "median")))
    if len(g) < 4:
        return (np.nan, np.nan, len(g))
    rho, p = stats.spearmanr(g["x"], np.log10(g["q"]))
    return (rho, p, len(g))


def main():
    red = pd.read_csv("/tmp/red_cohort.csv")
    print(f"Red cohort: {len(red)} crystals, {red.fp_id.nunique()} unique FPs\n")

    rows = []
    n_redo_ok = 0
    for _, r in red.iterrows():
        pid = r["pdb_id"]
        dep = twist_from_cif(CIF_DIR / f"{pid}.cif")
        redo_path = fetch_redo(pid)
        redo = twist_from_cif(redo_path) if redo_path else None
        if redo is not None:
            n_redo_ok += 1
        rows.append({
            "pdb_id": pid, "fp_id": r["fp_id"], "chrom_resname": r["chrom_resname"],
            "qy_curated": r["qy_curated"], "resolution": r["resolution"],
            "d_dep_stored": r["d_exp_to_planar_deg"],
            "d_dep_recomp": dep[2] if dep else np.nan,
            "d_redo": redo[2] if redo else np.nan,
            "tau_dep": dep[0] if dep else np.nan, "tau_redo": redo[0] if redo else np.nan,
            "phi_dep": dep[1] if dep else np.nan, "phi_redo": redo[1] if redo else np.nan,
            "redo_available": redo is not None,
        })
        tag = "ok" if redo is not None else "MISSING"
        print(f"  {pid}  dep={rows[-1]['d_dep_recomp']:6.2f}  "
              f"redo={rows[-1]['d_redo'] if redo else float('nan'):6.2f}  redo:{tag}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "redo_d_exp.csv", index=False)
    print(f"\nPDB-REDO available for {n_redo_ok}/{len(df)} red crystals")
    print(f"Saved {DATA/'redo_d_exp.csv'}")

    # sanity: recompute reproduces stored deposited values
    chk = df.dropna(subset=["d_dep_stored", "d_dep_recomp"])
    mad = (chk.d_dep_stored - chk.d_dep_recomp).abs().max()
    print(f"\n[sanity] deposited recompute vs stored: max abs diff = {mad:.4f} deg "
          f"(should be ~0)")

    have = df[df.redo_available].copy()
    have["abs_delta"] = (have.d_redo - have.d_dep_recomp).abs()

    print("\n" + "=" * 78)
    print("(a) DEPOSITED vs PDB-REDO chromophore twist agreement")
    print("=" * 78)
    rho, p = stats.spearmanr(have.d_dep_recomp, have.d_redo)
    pear = np.corrcoef(have.d_dep_recomp, have.d_redo)[0, 1]
    print(f"  Spearman rho = {rho:.3f} (p={p:.1e}), Pearson r = {pear:.3f}, n={len(have)}")
    print(f"  median |Δd_exp| deposited->REDO = {have.abs_delta.median():.2f} deg")
    print(f"  mean   |Δd_exp|                 = {have.abs_delta.mean():.2f} deg")

    print("\n" + "=" * 78)
    print("(b) Does the twist CHANGE grow at worse resolution? (restraint-sensitivity)")
    print("=" * 78)
    hr = have.assign(resolution=pd.to_numeric(have.resolution, errors="coerce")).dropna(subset=["resolution"])
    rho, p = stats.spearmanr(hr.resolution, hr.abs_delta)
    print(f"  Spearman(|Δd_exp|, resolution) = {rho:.3f} (p={p:.3f}, n={len(hr)})")
    print("  (positive+significant would indicate low-res twists are restraint-driven)")

    print("\n" + "=" * 78)
    print("(c) WITHIN-RED d_exp / QY correlation recomputed on PDB-REDO geometry")
    print("=" * 78)
    print(f"  {'':22s}{'deposited':>22s}{'PDB-REDO':>22s}")

    def fmt(t):
        return f"rho={t[0]:+.3f} p={t[1]:.3f} n={t[2]}"

    for label, fn in [("per crystal", spearman_log), ("per unique FP", per_unique)]:
        dep_stat = fn(have.rename(columns={"d_dep_recomp": "d"}), "d")
        rd_stat = fn(have.rename(columns={"d_redo": "d"}), "d")
        print(f"  {label:22s}{fmt(dep_stat):>22s}{fmt(rd_stat):>22s}")


if __name__ == "__main__":
    main()
