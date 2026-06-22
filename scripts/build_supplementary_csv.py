"""
build_supplementary_csv.py
==========================

Assemble the machine-readable master tables promised in Methods S9 and asked
for by the evaluation's submission checklist:

  data/supplementary_SD1_per_structure.csv  -- one row per crystal structure
  data/supplementary_SD2_per_unique_fp.csv   -- one row per unique fluorescent
                                                 protein (replicate crystals
                                                 collapsed by median)

Every column is pulled from the rawest on-disk source; no prose is trusted.
Sources:
  merged_for_aggregate.csv : geometry, cage, B-factor, barrel shape, spectra
  scan_1d_summary.csv      : 1-D I-bond / P-bond accessible fractions
  hb_contacts.csv          : polar-contact counts at OH and O2
  d_exp_canonical.csv      : canonical-CD resting twist (headline d_exp)
  lit_qy_curated.csv       : curated FPbase QY + provenance + unique-FP slug
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"


def up(df, col="pdb_id"):
    df[col] = df[col].astype(str).str.upper()
    return df


def main():
    meta = up(pd.read_csv(DATA / "merged_for_aggregate.csv"))
    s1 = up(pd.read_csv(DATA / "scan_1d_summary.csv"))
    hb = up(pd.read_csv(DATA / "hb_contacts.csv"))
    can = up(pd.read_csv(DATA / "d_exp_canonical.csv"))
    qy = up(pd.read_csv(DATA / "lit_qy_curated.csv"))

    # curated QY + provenance + unique-FP id (slug)
    qy["qy_curated"] = qy["lit_qy_fpbase"].fillna(qy["lit_qy_fpbase_recovered"])
    qy["fp_slug"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])
    qy["fp_name"] = qy["fpbase_name"].fillna(qy["seq_match_name"])
    qy = qy[["pdb_id", "qy_curated", "qy_provenance", "fp_slug", "fp_name",
             "seq_match_identity"]]

    # base: only structures that carry a scanned chromophore
    df = meta[meta.get("has_chromophore").astype(str).str.lower().isin(["true", "1", "1.0"])].copy() \
        if "has_chromophore" in meta.columns else meta.copy()

    cols_meta = {
        "pdb_id": "pdb_id",
        "color_class": "color_class",
        "chrom_resname": "chromophore_code",
        "chromophore_type": "chromophore_type",
        "tau_exp_deg": "tau_exp_deg",
        "phi_exp_deg": "phi_exp_deg",
        "f_allowed": "f_allowed",
        "f_allowed_folded": "f_allowed_folded",
        "f_allowed_folded_065": "f_allowed_folded_065",
        "A_allowed_deg2": "A_allowed_deg2",
        "d_exp_to_planar_deg": "d_exp_to_planar_deg",
        "d_centroid_to_planar_deg": "d_centroid_to_planar_deg",
        "d_exp_to_centroid_deg": "d_exp_to_centroid_deg",
        "b_factor_ratio": "b_factor_ratio",
        "minor_axis": "barrel_minor_axis",
        "eccentricity": "barrel_eccentricity",
        "barrel_length": "barrel_length",
        "resolution": "resolution_A",
        "r_factor": "r_factor",
        "ex_max": "ex_max_nm",
        "em_max": "em_max_nm",
        "stokes_shift": "stokes_shift_nm",
        "n_cage_atoms": "n_cage_atoms",
        "chrom_contacts": "chrom_cage_contacts",
    }
    keep = [c for c in cols_meta if c in df.columns]
    out = df[keep].rename(columns={k: cols_meta[k] for k in keep})

    out = out.merge(
        s1[["pdb_id", "f_allowed_I", "f_allowed_P_folded"]], on="pdb_id", how="left")
    out = out.merge(
        hb[["pdb_id", "n_polar_O2_32", "n_polar_OH_32"]], on="pdb_id", how="left")
    out = out.merge(
        can[["pdb_id", "tau", "phi_canon", "d_canonical"]].rename(columns={
            "tau": "tau_canonical_deg", "phi_canon": "phi_canonical_deg",
            "d_canonical": "d_exp_to_planar_canonical_deg"}),
        on="pdb_id", how="left")
    out = out.merge(qy, on="pdb_id", how="left")

    # tidy ordering: identity first
    front = ["pdb_id", "fp_slug", "fp_name", "color_class",
             "chromophore_code", "chromophore_type"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest].sort_values("pdb_id").reset_index(drop=True)
    out.to_csv(DATA / "supplementary_SD1_per_structure.csv", index=False)
    print(f"SD1 per-structure: {len(out)} rows, {out.shape[1]} cols "
          f"-> supplementary_SD1_per_structure.csv")

    # ---- SD2: one row per unique FP PER COLOR CLASS -------------------------
    # Color-first grouping (consistent with the per-color QY analysis and
    # Table 2): the grouping key is (fp_slug, color_class), so a photoconvertible
    # FP with distinct color states (e.g. mEos4b green/red) contributes one row
    # per state. This is what gives the per-class counts used throughout (red 38)
    # and keeps SD2 consistent with the recomputed analysis.
    grp = out.dropna(subset=["fp_slug"]).copy()
    grp["color_class"] = grp["color_class"].fillna("unknown")
    num = grp.select_dtypes(include=[np.number]).columns.tolist()
    cat_first = ["fp_name", "chromophore_code", "chromophore_type",
                 "qy_provenance"]
    agg = {c: "median" for c in num}
    for c in cat_first:
        if c in grp.columns:
            agg[c] = "first"
    keys = ["fp_slug", "color_class"]
    sd2 = grp.groupby(keys).agg(agg)
    sd2.insert(0, "n_structures", grp.groupby(keys).size())
    sd2.insert(1, "pdb_ids", grp.groupby(keys)["pdb_id"].apply(
        lambda s: ";".join(sorted(s))))
    sd2 = sd2.reset_index()
    # order columns: id, name, class, counts, then metrics
    lead = ["fp_slug", "fp_name", "color_class", "chromophore_code",
            "chromophore_type", "n_structures", "pdb_ids", "qy_curated",
            "qy_provenance"]
    lead = [c for c in lead if c in sd2.columns]
    sd2 = sd2[lead + [c for c in sd2.columns if c not in lead]]
    sd2 = sd2.sort_values(["fp_slug", "color_class"]).reset_index(drop=True)
    sd2.to_csv(DATA / "supplementary_SD2_per_unique_fp.csv", index=False)
    print(f"SD2 per-unique-FP: {len(sd2)} rows, {sd2.shape[1]} cols "
          f"-> supplementary_SD2_per_unique_fp.csv")
    n_qy = int(sd2["qy_curated"].notna().sum())
    print(f"  unique FPs with curated QY: {n_qy}")


if __name__ == "__main__":
    main()
