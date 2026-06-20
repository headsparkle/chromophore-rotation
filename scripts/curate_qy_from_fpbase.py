#!/usr/bin/env python3
"""
curate_qy_from_fpbase.py
========================

Build a curated quantum-yield table for the FPs in our 838-structure
dataset by joining against the FPbase protein registry.

The existing `lit_qy` column in
`../gfp-barrel-geometry/data/merged_complete_data.csv` was produced
by keyword matching: any structure whose name contained "gfp" got
wt avGFP's 0.79, etc. This pinned 263 of 322 labelled rows at the
same value and made any QY-vs-geometry analysis meaningless past
the chromophore-chemistry class.

FPbase (https://www.fpbase.org) is the curated registry of
fluorescent proteins, with citations and PDB cross-references for
each entry. Their JSON API lists every protein with `pdb` (list of
PDB IDs) and `states[]` (each carrying `qy`, `ex_max`, `em_max`,
`ext_coeff`, `pka`, `lifetime`, ...). We build a PDB -> FPbase
lookup, pick the most informative state per protein, and emit one
row per PDB in our dataset. Rows without an FPbase PDB match are
left NA (strict curation per user request).

Inputs
------
- data/fpbase_proteins.json
    Downloaded via:
      curl -L 'https://www.fpbase.org/api/proteins/?format=json'
          -o data/fpbase_proteins.json
- data/scan_all_summary.csv  (838 FPs we actually scanned)

Outputs
-------
- data/lit_qy_curated.csv with columns
    pdb_id, fpbase_slug, fpbase_name, lit_qy_fpbase,
    ex_max_fpbase, em_max_fpbase, ext_coeff_fpbase,
    fpbase_state, fpbase_doi
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FPBASE_JSON = DATA_DIR / "fpbase_proteins.json"
SCAN_SUMMARY = DATA_DIR / "scan_all_summary.csv"
OUT_CSV = DATA_DIR / "lit_qy_curated.csv"


def pick_state(states: list[dict]) -> dict | None:
    """Choose the state to report as the protein's photophysical
    summary. Prefer the one named 'default'; otherwise the first
    state with a non-null qy; otherwise the first state."""
    if not states:
        return None
    for s in states:
        if (s.get("name") or "").lower() == "default":
            return s
    for s in states:
        if s.get("qy") is not None:
            return s
    return states[0]


def main() -> int:
    if not FPBASE_JSON.is_file():
        sys.exit(f"missing {FPBASE_JSON} - download from FPbase first")
    proteins = json.loads(FPBASE_JSON.read_text())
    print(f"[load] FPbase has {len(proteins)} protein records")

    # pdb_id (upper) -> dict of curated fields. If a PDB ID maps to
    # multiple FPbase entries (rare but possible for paralogues), keep
    # the first; we log the collision.
    lookup: dict[str, dict] = {}
    collisions: list[tuple[str, str, str]] = []
    n_with_pdb = 0
    for p in proteins:
        pdbs = p.get("pdb") or []
        if not pdbs:
            continue
        n_with_pdb += 1
        state = pick_state(p.get("states", []))
        if state is None:
            continue
        row = {
            "fpbase_slug": p.get("slug"),
            "fpbase_name": p.get("name"),
            "lit_qy_fpbase": state.get("qy"),
            "ex_max_fpbase": state.get("ex_max"),
            "em_max_fpbase": state.get("em_max"),
            "ext_coeff_fpbase": state.get("ext_coeff"),
            "fpbase_state": state.get("name"),
            "fpbase_doi": p.get("doi"),
        }
        for pdb in pdbs:
            key = pdb.upper().strip()
            if key in lookup:
                collisions.append((key, lookup[key]["fpbase_name"], row["fpbase_name"]))
                continue
            lookup[key] = row
    print(f"[fpbase] {n_with_pdb} entries carry at least one PDB ID")
    print(f"[fpbase] {len(lookup)} unique PDB -> FPbase mappings")
    if collisions:
        print(f"[fpbase] {len(collisions)} PDB collisions (kept first):")
        for c in collisions[:10]:
            print(f"    {c[0]}: {c[1]} <- ignored {c[2]}")

    # Join against the scanned dataset.
    summary = pd.read_csv(SCAN_SUMMARY)
    pdbs_scanned = summary["pdb_id"].astype(str).str.upper().unique().tolist()
    print(f"[scan] {len(pdbs_scanned)} scanned PDB ids")

    rows = []
    n_hit = 0
    n_qy = 0
    for pdb in pdbs_scanned:
        hit = lookup.get(pdb)
        row = {"pdb_id": pdb}
        if hit is None:
            for col in ("fpbase_slug", "fpbase_name", "lit_qy_fpbase",
                        "ex_max_fpbase", "em_max_fpbase",
                        "ext_coeff_fpbase", "fpbase_state", "fpbase_doi"):
                row[col] = None
        else:
            n_hit += 1
            row.update(hit)
            if hit["lit_qy_fpbase"] is not None:
                n_qy += 1
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n[merge] matched {n_hit} / {len(pdbs_scanned)} scanned PDBs to FPbase")
    print(f"[merge] of those, {n_qy} have a curated QY")
    print(f"[save] {OUT_CSV}")
    print()
    print("QY distribution (curated):")
    print(df["lit_qy_fpbase"].describe())
    print(f"unique QY values: {df['lit_qy_fpbase'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
