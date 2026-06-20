"""
gatekeeper_analysis.py
======================

For every FP structure, identify the barrel residue(s) that impose
the first steric clash when the chromophore is driven outward along
each Megley torsion.  The resulting "gatekeeper" atoms are the cage
walls that physically restrict I-bond (tau) and P-bond (phi) rotation.

Analysis
--------
Four 1-D sweeps per structure:
  tau_plus   : tau increased from tau_exp until first clash
  tau_minus  : tau decreased
  phi_plus   : phi increased from phi_exp until first clash
  phi_minus  : phi decreased

At the first clashing grid step (5-deg resolution), max_overlap
returns the (i_moving, j_cage) index pair.  j_cage indexes the
combined cage array:

    cage_xyz = [static_chrom_atoms | extra (barrel) atoms]

Static chromophore atoms are labeled as (chrom_resname, atomname,
"self_clash").  Barrel atoms carry their actual (resnum, resname,
atomname).  We report protein clashes separately from self-clashes
because self-clashes are intrinsic to the chromophore geometry and
cannot be removed by mutation.

Boundary analysis (2-D)
-----------------------
We also scan the full 72x72 (tau, phi) grid.  Any disallowed point
that has at least one allowed neighbour in the grid is a "boundary"
point.  The clashing cage atom at each boundary point is the residue
that draws that particular wall of the cage.  This gives a complete
picture of every residue that contributes to the steric envelope --
not just the first one hit from the experimental geometry.

Outputs
-------
  data/gatekeeper_hits.csv        per-structure, per-sweep first-clash atom
  data/gatekeeper_boundary.csv    all boundary-point clash atoms (full 2-D)
  data/gatekeeper_summary.csv     aggregated frequency by color class
  figures/aggregate/gatekeeper_phi_heatmap.png
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barrel import (
    DEFAULT_TOLERANCE_A,
    WATER_RESNAMES,
    build_cage,
    load_structure,
    max_overlap,
)
from rotate import measure_megley, set_megley, side_reachable_from, build_bond_graph

PROJECT = Path(__file__).resolve().parent.parent
DATA    = PROJECT / "data"
CIF_DIR = DATA / "cif"
FIG_DIR = PROJECT / "figures" / "aggregate"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STEP_DEG = 5.0
GRID     = np.arange(-180.0, 180.0, STEP_DEG)   # 72 points
N_GRID   = len(GRID)


# ---------------------------------------------------------------------------
# Labeled structure loader
# ---------------------------------------------------------------------------

def load_labeled(cif_path: Path):
    """Load a structure and return (LoadedStructure, cage_labels).

    cage_labels is a list of dicts, one per cage atom, in the same
    order as cage_xyz inside CageContext.  Cage layout:
        indices 0 .. n_static-1 : static (non-moving) chromophore atoms
        indices n_static ..     : barrel / extra atoms

    Each dict has keys: chain, resnum, resname, atomname, is_chrom.
    """
    loaded = load_structure(cif_path)

    # Re-read the raw gemmi structure to collect barrel atom labels.
    # We must iterate in exactly the same order as load_structure does,
    # skipping waters and hydrogens, and skipping the first chromophore
    # residue (identified by bridge-atom pattern).
    from barrel import _residue_has_bridge_pattern, _element_symbol

    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]

    extra_labels: list[dict] = []
    chrom_found = False
    for chain in model:
        for residue in chain:
            if residue.name in WATER_RESNAMES:
                continue
            take_as_chrom = (not chrom_found) and _residue_has_bridge_pattern(residue)
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                if not take_as_chrom:
                    extra_labels.append({
                        "chain":    chain.name,
                        "resnum":   residue.seqid.num,
                        "resname":  residue.name,
                        "atomname": atom.name,
                        "is_chrom": False,
                    })
            if take_as_chrom:
                chrom_found = True

    # Build the cage to learn which chromophore atoms are static.
    cage = build_cage(loaded)
    graph = build_bond_graph(loaded.chrom_atoms)
    moving = set(side_reachable_from(graph, "CB2", "CA2"))
    static_names = [n for n in loaded.chrom_atoms if n not in moving]

    # Static chromophore labels come first in cage_xyz.
    static_labels = [
        {
            "chain":    loaded.chrom_chain,
            "resnum":   loaded.chrom_seqid,
            "resname":  loaded.chrom_resname,
            "atomname": n,
            "is_chrom": True,
        }
        for n in static_names
    ]

    cage_labels = static_labels + extra_labels
    assert len(cage_labels) == len(cage.cage_xyz), (
        f"Label count mismatch: {len(cage_labels)} labels vs "
        f"{len(cage.cage_xyz)} cage atoms in {cif_path.stem}"
    )
    return loaded, cage, cage_labels


def label_at(cage_labels: list[dict], j: int) -> dict:
    if j < 0 or j >= len(cage_labels):
        return {"chain": "?", "resnum": -1, "resname": "?",
                "atomname": "?", "is_chrom": False}
    return cage_labels[j]


# ---------------------------------------------------------------------------
# 1-D first-clash sweep
# ---------------------------------------------------------------------------

def sweep_first_clash(loaded, cage, cage_labels,
                      tau_start: float, phi_start: float,
                      fix_phi: bool, direction: int) -> dict:
    """Sweep tau (if not fix_phi) or phi starting from experimental value.

    direction: +1 or -1 (step direction).
    Returns a dict with the first-clash atom label and step count.
    """
    step = STEP_DEG * direction
    result = {"steps_to_clash": -1, "is_chrom": None,
              "chain": None, "resnum": None, "resname": None, "atomname": None}

    for k in range(1, N_GRID + 1):
        if fix_phi:
            tau_try = tau_start + k * step
            phi_try = phi_start
        else:
            tau_try = tau_start
            phi_try = phi_start + k * step

        atoms_now  = set_megley(loaded.chrom_atoms, tau_try, phi_try)
        moving_xyz = np.array([atoms_now[n] for n in cage.moving_names], dtype=float)
        ov, (i_mv, j_cage) = max_overlap(
            moving_xyz, cage.moving_radii,
            cage.cage_xyz, cage.cage_radii,
            cage.exclude,
        )
        if ov > DEFAULT_TOLERANCE_A:
            lbl = label_at(cage_labels, j_cage)
            result.update({
                "steps_to_clash": k,
                "overlap_a":      ov,
                "moving_atom":    cage.moving_names[i_mv],
                **{k2: lbl[k2] for k2 in ("chain", "resnum", "resname",
                                           "atomname", "is_chrom")},
            })
            return result
    return result   # no clash found in full circle


# ---------------------------------------------------------------------------
# 2-D boundary analysis
# ---------------------------------------------------------------------------

def boundary_clashes(loaded, cage, cage_labels,
                     tau_grid=GRID, phi_grid=GRID) -> list[dict]:
    """Full 2-D scan; return clash labels at boundary (wall) points only."""
    n_tau = len(tau_grid)
    n_phi = len(phi_grid)
    allowed = np.zeros((n_tau, n_phi), dtype=bool)
    clash_j = np.full((n_tau, n_phi), -1, dtype=int)  # cage index of worst clash

    for ii, tau in enumerate(tau_grid):
        for jj, phi in enumerate(phi_grid):
            atoms_now  = set_megley(loaded.chrom_atoms, float(tau), float(phi))
            moving_xyz = np.array([atoms_now[n] for n in cage.moving_names], dtype=float)
            ov, (_, j_cage) = max_overlap(
                moving_xyz, cage.moving_radii,
                cage.cage_xyz, cage.cage_radii,
                cage.exclude,
            )
            if ov <= DEFAULT_TOLERANCE_A:
                allowed[ii, jj] = True
            else:
                clash_j[ii, jj] = j_cage

    # Find boundary: disallowed cells with >= 1 allowed neighbour (4-connected)
    rows = []
    for ii in range(n_tau):
        for jj in range(n_phi):
            if allowed[ii, jj]:
                continue
            neighbours = [
                allowed[(ii - 1) % n_tau, jj],
                allowed[(ii + 1) % n_tau, jj],
                allowed[ii, (jj - 1) % n_phi],
                allowed[ii, (jj + 1) % n_phi],
            ]
            if any(neighbours):
                j = int(clash_j[ii, jj])
                lbl = label_at(cage_labels, j)
                rows.append({
                    "tau_deg": float(tau_grid[ii]),
                    "phi_deg": float(phi_grid[jj]),
                    **{k2: lbl[k2] for k2 in ("chain", "resnum", "resname",
                                               "atomname", "is_chrom")},
                })
    return rows


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_pdb(pdb_id: str):
    cif_path = CIF_DIR / f"{pdb_id.upper()}.cif"
    loaded, cage, cage_labels = load_labeled(cif_path)
    tau_exp, phi_exp = measure_megley(loaded.chrom_atoms)

    sweeps = {}
    for name, fix_phi, direction in [
        ("tau_plus",  True,  +1),
        ("tau_minus", True,  -1),
        ("phi_plus",  False, +1),
        ("phi_minus", False, -1),
    ]:
        r = sweep_first_clash(loaded, cage, cage_labels,
                              tau_exp, phi_exp, fix_phi, direction)
        sweeps[name] = r

    bnd = boundary_clashes(loaded, cage, cage_labels)

    return {
        "pdb_id":    pdb_id,
        "tau_exp":   tau_exp,
        "phi_exp":   phi_exp,
        "sweeps":    sweeps,
        "boundary":  bnd,
    }


def main():
    # Run on per-unique-FP representative structures
    merged  = pd.read_csv(DATA / "merged_for_aggregate.csv")
    qydf    = pd.read_csv(DATA / "lit_qy_curated.csv")
    qydf["fp_id"] = qydf["fpbase_slug"].fillna(qydf["seq_match_slug"])
    merged  = merged.merge(qydf[["pdb_id", "fp_id"]], on="pdb_id", how="left")

    # Pick one representative PDB per fp_id (lowest b_factor_ratio = best crystal)
    merged["b_factor_ratio"] = pd.to_numeric(merged["b_factor_ratio"], errors="coerce")
    rep = (
        merged.dropna(subset=["fp_id"])
        .sort_values("b_factor_ratio")
        .groupby("fp_id", as_index=False)
        .first()
    )
    # Include structures without fp_id too (matched by canonical_cohort)
    no_fp = merged[merged["fp_id"].isna() & merged["canonical_cohort"]].copy()
    no_fp["fp_id"] = no_fp["pdb_id"]  # treat as their own representative
    targets = pd.concat([rep, no_fp], ignore_index=True).drop_duplicates("pdb_id")

    print(f"Running gatekeeper analysis on {len(targets)} representative structures")

    # Resume support
    hits_path = DATA / "gatekeeper_hits.csv"
    bnd_path  = DATA / "gatekeeper_boundary.csv"
    done: set[str] = set()
    if hits_path.is_file():
        ex = pd.read_csv(hits_path)
        done = set(ex["pdb_id"].str.upper())
        print(f"  {len(done)} already done, {len(targets)-len(done)} remaining")

    sweep_fields = ["pdb_id", "color_class", "fp_id",
                    "tau_exp", "phi_exp",
                    "sweep",
                    "steps_to_clash", "overlap_a", "moving_atom",
                    "chain", "resnum", "resname", "atomname", "is_chrom"]
    bnd_fields   = ["pdb_id", "color_class", "fp_id",
                    "tau_deg", "phi_deg",
                    "chain", "resnum", "resname", "atomname", "is_chrom"]

    hits_fh = open(hits_path, "a", newline="")
    bnd_fh  = open(bnd_path,  "a", newline="")
    hits_w  = csv.DictWriter(hits_fh, fieldnames=sweep_fields, extrasaction="ignore")
    bnd_w   = csv.DictWriter(bnd_fh,  fieldnames=bnd_fields,   extrasaction="ignore")
    if len(done) == 0:
        hits_w.writeheader(); bnd_w.writeheader()
        hits_fh.flush(); bnd_fh.flush()

    n_ok = n_fail = 0
    t0 = time.perf_counter()

    for _, row in targets.iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        if pdb_id in done:
            continue
        cif_path = CIF_DIR / f"{pdb_id}.cif"
        if not cif_path.is_file():
            n_fail += 1
            continue
        try:
            result = process_pdb(pdb_id)
            meta = {
                "pdb_id":       pdb_id,
                "color_class":  row.get("color_class", ""),
                "fp_id":        row.get("fp_id", ""),
                "tau_exp":      result["tau_exp"],
                "phi_exp":      result["phi_exp"],
            }
            for sweep_name, sr in result["sweeps"].items():
                hits_w.writerow({**meta, "sweep": sweep_name, **sr})
            for br in result["boundary"]:
                bnd_w.writerow({**meta, **br})
            hits_fh.flush(); bnd_fh.flush()
            n_ok += 1
        except Exception:
            tb = traceback.format_exc(limit=2)
            print(f"  FAIL {pdb_id}: {tb.strip()}", file=sys.stderr)
            n_fail += 1

        if (n_ok + n_fail) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (n_ok + n_fail) / elapsed if elapsed > 0 else 1
            eta  = (len(targets) - len(done) - n_ok - n_fail) / rate
            print(f"  [{n_ok+n_fail}/{len(targets)-len(done)}]  ok={n_ok}"
                  f"  fail={n_fail}  eta={eta/60:.1f} min", flush=True)

    hits_fh.close(); bnd_fh.close()
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed/60:.1f} min: ok={n_ok} fail={n_fail}")

    # ---------------------------------------------------------------------------
    # Aggregate: gatekeeper frequency table
    # ---------------------------------------------------------------------------
    aggregate()


def aggregate():
    hits = pd.read_csv(DATA / "gatekeeper_hits.csv")
    bnd  = pd.read_csv(DATA / "gatekeeper_boundary.csv")

    # Keep only protein (non-self) clashes for engineering relevance
    protein_hits = hits[hits["is_chrom"] == False].copy()
    protein_bnd  = bnd[bnd["is_chrom"]  == False].copy()

    # Drop unclassified color classes
    for df in (protein_hits, protein_bnd):
        df["color_class"] = df["color_class"].fillna("unknown").str.lower()

    # --- 1. Sweep summary: frequency per (color_class, sweep, resnum, resname, atomname) ---
    sweep_grp = (
        protein_hits
        .groupby(["color_class", "sweep", "resnum", "resname", "atomname"])
        .size()
        .reset_index(name="n_structures")
    )
    # Denominator: structures per color_class * sweep combo
    denom = (
        hits.groupby(["color_class", "sweep"])
        .size()
        .reset_index(name="n_total")
    )
    sweep_grp = sweep_grp.merge(denom, on=["color_class", "sweep"])
    sweep_grp["frac"] = sweep_grp["n_structures"] / sweep_grp["n_total"]
    sweep_grp = sweep_grp.sort_values(
        ["color_class", "sweep", "frac"], ascending=[True, True, False]
    )
    sweep_grp.to_csv(DATA / "gatekeeper_summary.csv", index=False)
    print(f"Saved {DATA / 'gatekeeper_summary.csv'}")

    # --- 2. Boundary summary: how often each residue defines a cage wall ---
    bnd_grp = (
        protein_bnd
        .groupby(["color_class", "resnum", "resname", "atomname"])
        .size()
        .reset_index(name="n_boundary_pts")
    )
    n_struct_per_class = (
        bnd.groupby("color_class")["pdb_id"].nunique()
        .reset_index(name="n_structures")
    )
    bnd_grp = bnd_grp.merge(n_struct_per_class, on="color_class")
    bnd_grp["avg_pts_per_struct"] = bnd_grp["n_boundary_pts"] / bnd_grp["n_structures"]
    bnd_grp = bnd_grp.sort_values(
        ["color_class", "avg_pts_per_struct"], ascending=[True, False]
    )
    bnd_grp.to_csv(DATA / "gatekeeper_boundary_summary.csv", index=False)
    print(f"Saved {DATA / 'gatekeeper_boundary_summary.csv'}")

    # --- Print top gatekeepers per color class ---
    print_summary(sweep_grp, bnd_grp)

    # --- Figure ---
    plot_phi_gatekeepers(protein_bnd)


def print_summary(sweep_grp: pd.DataFrame, bnd_grp: pd.DataFrame):
    colors = ["cyan", "green", "yellow", "red"]

    print("\n" + "="*70)
    print("TOP PHI (P-BOND) GATEKEEPERS BY COLOR CLASS (boundary analysis)")
    print("="*70)
    for color in colors:
        sub = bnd_grp[
            (bnd_grp["color_class"] == color)
        ].nlargest(10, "avg_pts_per_struct")
        if sub.empty:
            continue
        print(f"\n--- {color.upper()} (n={sub['n_structures'].iloc[0]} structures) ---")
        print(f"{'ResNum':>8} {'ResName':>8} {'Atom':>6}   {'avg_pts/struct':>15}")
        for _, r in sub.iterrows():
            print(f"{int(r['resnum']) if pd.notna(r['resnum']) else '?':>8} "
                  f"{r['resname']:>8} {r['atomname']:>6}   "
                  f"{r['avg_pts_per_struct']:>15.1f}")

    print("\n" + "="*70)
    print("TOP TAU (I-BOND) FIRST-CLASH GATEKEEPERS BY COLOR CLASS")
    print("="*70)
    for color in colors:
        sub = sweep_grp[
            (sweep_grp["color_class"] == color) &
            (sweep_grp["sweep"].str.startswith("tau"))
        ].nlargest(10, "frac")
        if sub.empty:
            continue
        n_tot = sub["n_total"].iloc[0] if len(sub) else "?"
        print(f"\n--- {color.upper()} (n~{n_tot} sweep evaluations) ---")
        print(f"{'Sweep':>12} {'ResNum':>8} {'ResName':>8} {'Atom':>6}   {'frac':>6}")
        for _, r in sub.iterrows():
            print(f"{r['sweep']:>12} "
                  f"{int(r['resnum']) if pd.notna(r['resnum']) else '?':>8} "
                  f"{r['resname']:>8} {r['atomname']:>6}   "
                  f"{r['frac']:>6.2%}")

    print("\n" + "="*70)
    print("TOP PHI (P-BOND) FIRST-CLASH GATEKEEPERS BY COLOR CLASS")
    print("="*70)
    for color in colors:
        sub = sweep_grp[
            (sweep_grp["color_class"] == color) &
            (sweep_grp["sweep"].str.startswith("phi"))
        ].nlargest(10, "frac")
        if sub.empty:
            continue
        n_tot = sub["n_total"].iloc[0] if len(sub) else "?"
        print(f"\n--- {color.upper()} (n~{n_tot} sweep evaluations) ---")
        print(f"{'Sweep':>12} {'ResNum':>8} {'ResName':>8} {'Atom':>6}   {'frac':>6}")
        for _, r in sub.iterrows():
            print(f"{r['sweep']:>12} "
                  f"{int(r['resnum']) if pd.notna(r['resnum']) else '?':>8} "
                  f"{r['resname']:>8} {r['atomname']:>6}   "
                  f"{r['frac']:>6.2%}")


def plot_phi_gatekeepers(protein_bnd: pd.DataFrame):
    """Bar chart of top phi-gatekeeper residues per color class."""
    COLOR_PALETTE = {
        "green": "#22aa33", "yellow": "#d4b400", "cyan": "#1e90c8",
        "red": "#cc2030",
    }
    colors = ["cyan", "green", "yellow", "red"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Phi (P-bond) gatekeeper residues by color class\n"
                 "(boundary of allowed region, protein atoms only)", fontsize=11)

    for ax, color in zip(axes, colors):
        sub = protein_bnd[protein_bnd["color_class"] == color].copy()
        if sub.empty:
            ax.set_title(color); ax.axis("off"); continue
        n_struct = sub["pdb_id"].nunique()

        # Aggregate by residue label (resnum + resname + atomname)
        sub["label"] = (sub["resnum"].astype(str) + " " +
                        sub["resname"] + "\n" + sub["atomname"])
        grp = (
            sub.groupby("label")["pdb_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(12)
        )
        frac = grp / n_struct

        ax.barh(range(len(frac)), frac.values[::-1],
                color=COLOR_PALETTE.get(color, "#888888"), alpha=0.85)
        ax.set_yticks(range(len(frac)))
        ax.set_yticklabels(frac.index[::-1], fontsize=8)
        ax.set_xlabel("Fraction of structures\nwhere residue is a cage wall", fontsize=8)
        ax.set_title(f"{color} (n={n_struct})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "gatekeeper_phi_heatmap.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
