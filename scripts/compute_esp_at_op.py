#!/usr/bin/env python3
"""
compute_esp_at_op.py
====================

For every PDB structure in our scan, compute the electrostatic
potential and electric field at the chromophore's phenolate oxygen
(OP) using a simple Coulomb summation over the surrounding
residues. The OP atom is the OH atom of Tyr-derived chromophores
(CRO, CRQ, NRQ, CRF and relatives); non-Tyr chromophores
(His-derived IIC/CRG, Trp-derived SWG/CRW and others) lack an
equivalent atom and are marked NA.

Motivation
----------
Park and Rhee (J. Am. Chem. Soc. 2016, 138, 13619) showed that
avGFP's high quantum yield is dominated by the protein's
electric field at the phenolate oxygen, opposing the charge
migration that would accompany P-bond twisting. The relevant
quantity is the field magnitude and its component along the
C(phenolate)-O bond. We compute a structural proxy for both
quantities across the 838-FP cohort and add them as columns
to scan_all_summary.csv.

Method (deliberately simple, for reproducibility across a
heterogeneous PDB cohort)
-------------------------
- For each non-chromophore residue within 12 A of OP, place
  point charges on selected heavy atoms from the SIMPLE_CHARGES
  dictionary below.
- Side chains: full +/-1 e on ionizable groups (Asp, Glu, Lys,
  Arg) split across symmetric atoms; small dipoles on polar
  side chains (Tyr, Ser, Thr, Asn, Gln, Cys, Met, Trp).
  Histidine is left neutral (epsilon-tautomer, no charge).
- Backbone: standard amide dipole charges (+0.3 on N, -0.4
  on O, +0.1 on C).
- Water (HOH/WAT/DOD) excluded for v1; Park-Rhee's critical
  Wat241 in avGFP is structure-specific and would require
  per-PDB curation.
- Output: esp_at_op_V (volts), efield_mag_V_per_A (V/Å),
  efield_along_co_V_per_A (V/Å, signed; positive = field
  vector pointing in the C->O direction).

Caveats
-------
- This is a low-resolution proxy. Park-Rhee use ff14SB partial
  charges, explicit waters, and PROPKA protonation per structure.
  We use standard protonation states uniformly. Relative
  comparisons across the dataset should still be informative.
- ESP at a single point is a scalar projection of the full 3D
  field map; the field along the C-O bond is the mechanistically
  relevant quantity per Park-Rhee Figure 6/7.

Output columns
--------------
- esp_at_op_V : electrostatic potential at OP, V
- efield_mag_V_per_A : |E| at OP, V/A
- efield_along_co_V_per_A : signed E . (n_C->O), V/A
- esp_n_residues_within_12a : sanity count

Tractable runtime is a few minutes for the full cohort.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

from ff14sb_charges import (
    resolve_residue_charges,
    TIP3P_O_CHARGE,
    TIP3P_H_CHARGE,
    TIP3P_OH_BOND_A,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif"
SUMMARY_CSV = DATA_DIR / "scan_all_summary.csv"

# Coulomb constant in V * Angstrom per elementary charge.
# k = 1 / (4 pi eps0) ~ 14.3996 V * A * e^-1
K_VA_PER_E = 14.3996

# Cutoff: distance within which we sum charges, in Angstrom.
CUTOFF_A = 12.0

WATER_RESNAMES = {"HOH", "WAT", "DOD"}

# Standard chromophore residue codes we treat as the chromophore.
# This list comes from barrel.py and the per-PDB scan results;
# the OH atom is present only for Tyr-derived members.
CHROMOPHORE_RESIDUES = {
    "CRO", "CR2", "GYS", "SYG", "CRQ", "CRF", "CRW", "CRY",
    "NRQ", "CRG", "CSY", "CFY", "MFC", "OFY", "CR5", "CRK",
    "CRO", "CSH", "GYC", "GYG", "X9Q", "C12", "C99", "PIA",
    "SWG", "CCY", "B2H", "CR7", "CR8", "CCY", "IIC", "QYG",
    "EYG", "5ZA", "RYG", "SYG", "TYG", "C2G", "AYG", "GH6",
    "1UB", "CR2", "CR2", "I2C", "MDO", "NYG", "QLG", "X9Q",
    "CH6", "ZH6", "MYG", "C2G", "CYG", "CH7", "OYG", "MDP",
    "MJ7", "CO8", "CWR", "DYG", "GLG", "RDO", "NPB", "TRG",
}

# Simplified partial-charge dictionary. Residue -> list of
# (atom_name, partial_charge_in_e). Charges are intentionally
# coarse so that the procedure is stable across the cohort.
SIMPLE_CHARGES = {
    # --- Ionizable side chains (full unit charge, split for symmetry) ---
    "ASP": [("OD1", -0.5), ("OD2", -0.5)],
    "GLU": [("OE1", -0.5), ("OE2", -0.5)],
    "LYS": [("NZ", +1.0)],
    "ARG": [("NH1", +0.5), ("NH2", +0.5)],
    # --- Polar side chains (dipole approximation) ---
    "HIS": [],  # neutral epsilon-tautomer; no net charge
    "TYR": [("OH", -0.5), ("CZ", +0.5)],
    "SER": [("OG", -0.4), ("CB", +0.4)],
    "THR": [("OG1", -0.4), ("CB", +0.4)],
    "ASN": [("OD1", -0.5), ("ND2", +0.5)],
    "GLN": [("OE1", -0.5), ("NE2", +0.5)],
    "CYS": [("SG", -0.2), ("CB", +0.2)],
    "MET": [("SD", -0.2), ("CG", +0.2)],
    "TRP": [("NE1", -0.3), ("CD1", +0.3)],
    # --- Non-polar side chains: no contribution ---
    # ALA, GLY, VAL, LEU, ILE, PHE, PRO, MSE not in dict
}

# Backbone amide dipole (applied to ALL standard amino acids
# whose residue name is in the SIMPLE_CHARGES dict OR in
# NONPOLAR_AMINO_ACIDS below).
BACKBONE_CHARGES = [("N", +0.30), ("O", -0.40), ("C", +0.10)]

NONPOLAR_AMINO_ACIDS = {
    "ALA", "GLY", "VAL", "LEU", "ILE", "PHE", "PRO", "MSE",
}

# Standard one-letter aas for the backbone charge map. Any residue
# in this set gets backbone amide charges applied.
STANDARD_AA = set(SIMPLE_CHARGES.keys()) | NONPOLAR_AMINO_ACIDS


def is_tyr_derived_chromophore(residue: gemmi.Residue) -> bool:
    """Return True if the chromophore residue has a OH atom on a
    phenol ring (the standard Tyr-derived case)."""
    has_oh = any(a.name == "OH" for a in residue)
    has_cz = any(a.name == "CZ" for a in residue)
    return has_oh and has_cz


def _residue_has_bridge_pattern(residue: gemmi.Residue) -> bool:
    names = {a.name for a in residue}
    return "CA2" in names and "CB2" in names and "N2" in names


def find_chromophore_residue(structure: gemmi.Structure):
    """Return the first chromophore residue (using the CA2/CB2/N2
    pattern as in barrel.py) or None."""
    for chain in structure[0]:
        for residue in chain:
            if residue.name in WATER_RESNAMES:
                continue
            if _residue_has_bridge_pattern(residue):
                return chain, residue
    return None, None


def find_op_position(residue: gemmi.Residue):
    """Return (op_position, cz_position) as np.array(3,) tuples
    if both atoms exist on the residue, else (None, None)."""
    op = None
    cz = None
    for a in residue:
        if a.name == "OH":
            op = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)
        elif a.name == "CZ":
            cz = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)
    if op is None or cz is None:
        return None, None
    return op, cz


NEAR_SHELL_A = 6.0
WATER_NEAR_OP_A = 4.0


def _add_charge_contribution(
    esp_acc: dict,
    efield_acc: dict,
    op_xyz: np.ndarray,
    r_i: np.ndarray,
    q: float,
    shell_cutoff: float = NEAR_SHELL_A,
):
    """Add a single point charge's contribution to the running
    sums. esp_acc and efield_acc are 2-element dicts with keys
    'near' and 'far'."""
    dr = op_xyz - r_i
    d2 = float(dr @ dr)
    if d2 < 1e-6:
        return
    d = d2 ** 0.5
    dV = K_VA_PER_E * q / d
    dE = K_VA_PER_E * q * dr / (d2 * d)
    key = "near" if d < shell_cutoff else "far"
    esp_acc[key] += dV
    efield_acc[key] += dE


def compute_field_at_point_ff14(
    structure: gemmi.Structure,
    op_xyz: np.ndarray,
    chrom_chain_name: str,
    chrom_seqid: int,
    cutoff_a: float = CUTOFF_A,
    include_waters: bool = True,
) -> dict:
    """ff14SB-charged Coulomb summation with shell decomposition
    and explicit-water inclusion.

    Returns a dict with keys:
      esp_near_V, esp_far_V, esp_water_V
      efield_near_xyz, efield_far_xyz, efield_water_xyz (3-vectors V/A)
      n_residues_near, n_residues_far, n_waters
    """
    cutoff2 = cutoff_a * cutoff_a
    esp_acc = {"near": 0.0, "far": 0.0}
    efield_acc = {"near": np.zeros(3), "far": np.zeros(3)}
    esp_water = 0.0
    efield_water = np.zeros(3)
    n_near = 0
    n_far = 0
    n_waters = 0
    water_near2 = WATER_NEAR_OP_A * WATER_NEAR_OP_A

    for chain in structure[0]:
        for residue in chain:
            # Waters: include only if very close to OP, model as
            # oriented H-bond donor.
            if residue.name in WATER_RESNAMES:
                if not include_waters:
                    continue
                # Find the water O atom.
                ow = None
                for atom in residue:
                    if atom.element.is_hydrogen:
                        continue
                    if atom.name in ("O", "OW", "OH2"):
                        ow = np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z],
                            dtype=float,
                        )
                        break
                if ow is None:
                    continue
                d2 = float(((op_xyz - ow) @ (op_xyz - ow)))
                if d2 > water_near2:
                    continue
                # Place O with TIP3P O charge, then a single virtual H
                # (sum of both TIP3Ps) on the line from O to OP.
                d = d2 ** 0.5
                op_unit = (op_xyz - ow) / d
                # Virtual H position: 0.957 A from O toward OP.
                # The two H atoms of water are 1.515 A apart and the
                # H-O-H bisector points toward OP when the water
                # H-bonds to OP. The combined dipole projection is
                # approximated as a single +2*0.417 = +0.834 charge
                # placed at the bisector position 0.6 A from O (the
                # projection of each H onto the bisector for tetrahedral
                # geometry: 0.957 * cos(52.25 deg) = 0.586 A).
                h_pos = ow + 0.586 * op_unit
                # O contribution
                dr = op_xyz - ow
                dV = K_VA_PER_E * TIP3P_O_CHARGE / d
                dE = K_VA_PER_E * TIP3P_O_CHARGE * dr / (d2 * d)
                esp_water += dV
                efield_water += dE
                # Single-virtual-H contribution (charge = 2 * H_TIP3P)
                dr_h = op_xyz - h_pos
                d_h2 = float(dr_h @ dr_h)
                d_h = d_h2 ** 0.5
                q_h = 2.0 * TIP3P_H_CHARGE
                if d_h > 1e-6:
                    esp_water += K_VA_PER_E * q_h / d_h
                    efield_water += K_VA_PER_E * q_h * dr_h / (d_h2 * d_h)
                n_waters += 1
                continue
            # Skip the chromophore residue itself.
            if (
                chain.name == chrom_chain_name
                and residue.seqid.num == chrom_seqid
                and _residue_has_bridge_pattern(residue)
            ):
                continue
            charges, _ = resolve_residue_charges(residue.name, normalise=True)
            if not charges:
                continue
            placed_near = False
            placed_far = False
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                q = charges.get(atom.name)
                if q is None:
                    continue
                r_i = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                )
                dr = op_xyz - r_i
                d2 = float(dr @ dr)
                if d2 > cutoff2 or d2 < 1e-6:
                    continue
                d = d2 ** 0.5
                dV = K_VA_PER_E * q / d
                dE = K_VA_PER_E * q * dr / (d2 * d)
                if d < NEAR_SHELL_A:
                    esp_acc["near"] += dV
                    efield_acc["near"] += dE
                    placed_near = True
                else:
                    esp_acc["far"] += dV
                    efield_acc["far"] += dE
                    placed_far = True
            if placed_near:
                n_near += 1
            elif placed_far:
                n_far += 1

    return {
        "esp_near_V": esp_acc["near"],
        "esp_far_V": esp_acc["far"],
        "esp_water_V": esp_water,
        "efield_near_xyz": efield_acc["near"],
        "efield_far_xyz": efield_acc["far"],
        "efield_water_xyz": efield_water,
        "n_residues_near": n_near,
        "n_residues_far": n_far,
        "n_waters": n_waters,
    }


def compute_field_at_point(
    structure: gemmi.Structure,
    op_xyz: np.ndarray,
    chrom_chain_name: str,
    chrom_seqid: int,
    cutoff_a: float = CUTOFF_A,
) -> tuple[float, np.ndarray, int]:
    """Sum point-charge contributions from non-chromophore residues
    to the potential and electric field at op_xyz.

    Returns
    -------
    esp_V : scalar potential at op_xyz in volts
    efield_V_per_A : (3,) array, electric field in V/A
    n_residues_used : number of residues with at least one charge
                      placed within the cutoff
    """
    cutoff2 = cutoff_a * cutoff_a
    esp = 0.0
    efield = np.zeros(3, dtype=float)
    n_used = 0
    for chain in structure[0]:
        for residue in chain:
            if residue.name in WATER_RESNAMES:
                continue
            # Skip the chromophore residue itself.
            if (
                chain.name == chrom_chain_name
                and residue.seqid.num == chrom_seqid
                and _residue_has_bridge_pattern(residue)
            ):
                continue
            # Build the (atom_name -> partial_charge) for this
            # residue: backbone amide for any standard AA, plus
            # side-chain charges from the dict.
            charges = {}
            if residue.name in STANDARD_AA:
                for name, q in BACKBONE_CHARGES:
                    charges[name] = q
            for name, q in SIMPLE_CHARGES.get(residue.name, []):
                charges[name] = charges.get(name, 0.0) + q
            if not charges:
                continue
            placed = False
            for atom in residue:
                if atom.name not in charges:
                    continue
                if atom.element.is_hydrogen:
                    continue
                r_i = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
                )
                dr = op_xyz - r_i
                d2 = float(dr @ dr)
                if d2 > cutoff2 or d2 < 1e-6:
                    continue
                d = d2 ** 0.5
                q = charges[atom.name]
                # Potential: V = k * q / r
                esp += K_VA_PER_E * q / d
                # Field at op_xyz due to charge at r_i:
                # E = k * q * (op - r_i) / r^3 (V/A)
                efield += K_VA_PER_E * q * dr / (d2 * d)
                placed = True
            if placed:
                n_used += 1
    return esp, efield, n_used


_NA_FF14_FIELDS = {
    "esp_at_op_V_ff14": np.nan,
    "efield_mag_V_per_A_ff14": np.nan,
    "efield_along_co_V_per_A_ff14": np.nan,
    "esp_near_V_ff14": np.nan,
    "esp_far_V_ff14": np.nan,
    "esp_water_V_ff14": np.nan,
    "efield_along_co_near_V_per_A_ff14": np.nan,
    "efield_along_co_far_V_per_A_ff14": np.nan,
    "efield_along_co_water_V_per_A_ff14": np.nan,
    "esp_n_residues_near_ff14": 0,
    "esp_n_residues_far_ff14": 0,
    "esp_n_waters_ff14": 0,
}


def _na_dict(err: str) -> dict:
    d = {
        "esp_at_op_V": np.nan,
        "efield_mag_V_per_A": np.nan,
        "efield_along_co_V_per_A": np.nan,
        "esp_n_residues_within_12a": 0,
        "esp_error": err,
    }
    d.update(_NA_FF14_FIELDS)
    return d


def process_pdb(pdb_id: str) -> dict:
    """Returns a dict of new column values for this PDB. Marks NA
    on missing CIF, non-Tyr chromophore, or other failure."""
    cif_path = CIF_DIR / f"{pdb_id.upper()}.cif"
    if not cif_path.is_file():
        return _na_dict("no_cif")
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as exc:
        return _na_dict(f"read_failed:{exc.__class__.__name__}")
    chrom_chain, chrom_res = find_chromophore_residue(st)
    if chrom_res is None:
        return _na_dict("no_chromophore")
    if not is_tyr_derived_chromophore(chrom_res):
        return _na_dict("non_tyr_chromophore")
    op, cz = find_op_position(chrom_res)
    if op is None:
        return _na_dict("no_op_cz")
    # Unit vector from CZ to OP (C-phenolate to O-phenolate)
    co_vec = op - cz
    co_norm = float(np.linalg.norm(co_vec))
    if co_norm < 1e-6:
        return _na_dict("co_zero")
    co_unit = co_vec / co_norm
    esp, efield, n_used = compute_field_at_point(
        st, op, chrom_chain.name, chrom_res.seqid.num,
    )
    e_mag = float(np.linalg.norm(efield))
    e_along = float(efield @ co_unit)
    # ff14SB version with shells and waters
    ff = compute_field_at_point_ff14(
        st, op, chrom_chain.name, chrom_res.seqid.num,
    )
    esp_total_ff = ff["esp_near_V"] + ff["esp_far_V"] + ff["esp_water_V"]
    efield_total_ff = (
        ff["efield_near_xyz"] + ff["efield_far_xyz"] + ff["efield_water_xyz"]
    )
    e_mag_ff = float(np.linalg.norm(efield_total_ff))
    e_along_ff = float(efield_total_ff @ co_unit)
    e_along_near = float(ff["efield_near_xyz"] @ co_unit)
    e_along_far = float(ff["efield_far_xyz"] @ co_unit)
    e_along_water = float(ff["efield_water_xyz"] @ co_unit)
    return {
        # v1 (formal charges, no waters, no shells) - kept for comparison
        "esp_at_op_V": float(esp),
        "efield_mag_V_per_A": e_mag,
        "efield_along_co_V_per_A": e_along,
        "esp_n_residues_within_12a": n_used,
        "esp_error": "",
        # v2 (ff14SB + waters + shell decomposition)
        "esp_at_op_V_ff14": float(esp_total_ff),
        "efield_mag_V_per_A_ff14": e_mag_ff,
        "efield_along_co_V_per_A_ff14": e_along_ff,
        "esp_near_V_ff14": float(ff["esp_near_V"]),
        "esp_far_V_ff14": float(ff["esp_far_V"]),
        "esp_water_V_ff14": float(ff["esp_water_V"]),
        "efield_along_co_near_V_per_A_ff14": e_along_near,
        "efield_along_co_far_V_per_A_ff14": e_along_far,
        "efield_along_co_water_V_per_A_ff14": e_along_water,
        "esp_n_residues_near_ff14": ff["n_residues_near"],
        "esp_n_residues_far_ff14": ff["n_residues_far"],
        "esp_n_waters_ff14": ff["n_waters"],
    }


def main() -> int:
    df = pd.read_csv(SUMMARY_CSV)
    n_total = len(df)
    n_ok = (df["status"] == "ok").sum()
    print(f"[load] {n_total} rows ({n_ok} ok); cutoff = {CUTOFF_A} A")
    new_cols = [
        # v1
        "esp_at_op_V",
        "efield_mag_V_per_A",
        "efield_along_co_V_per_A",
        "esp_n_residues_within_12a",
        # v2 (ff14SB + waters + shells)
        "esp_at_op_V_ff14",
        "efield_mag_V_per_A_ff14",
        "efield_along_co_V_per_A_ff14",
        "esp_near_V_ff14",
        "esp_far_V_ff14",
        "esp_water_V_ff14",
        "efield_along_co_near_V_per_A_ff14",
        "efield_along_co_far_V_per_A_ff14",
        "efield_along_co_water_V_per_A_ff14",
        "esp_n_residues_near_ff14",
        "esp_n_residues_far_ff14",
        "esp_n_waters_ff14",
    ]
    for c in new_cols:
        df[c] = np.nan
    df["esp_error"] = ""
    n_done = 0
    n_skipped = 0
    n_nontyr = 0
    n_no_cif = 0
    for idx, row in df.iterrows():
        if row["status"] != "ok":
            n_skipped += 1
            continue
        result = process_pdb(str(row["pdb_id"]))
        for k in new_cols + ["esp_error"]:
            df.at[idx, k] = result[k]
        err = result["esp_error"]
        if err == "non_tyr_chromophore":
            n_nontyr += 1
        elif err == "no_cif":
            n_no_cif += 1
        elif err == "":
            n_done += 1
        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{n_total}] computed = {n_done}, "
                  f"non-tyr = {n_nontyr}, no-CIF = {n_no_cif}")
    df.to_csv(SUMMARY_CSV, index=False)
    ok_with_esp = df[df["esp_at_op_V"].notna()]
    print(f"\n[done] computed ESP for {n_done} PDBs; "
          f"non-Tyr chromophores skipped: {n_nontyr}; "
          f"no CIF: {n_no_cif}; status!=ok skipped: {n_skipped}\n")
    print("Summary of new columns (Tyr-derived chromophores only):")
    for c in new_cols:
        s = ok_with_esp[c]
        if c.startswith("esp_n"):
            print(f"  {c:36s} median = {s.median():.0f}  "
                  f"IQR = [{s.quantile(0.25):.0f}, {s.quantile(0.75):.0f}]")
        else:
            print(f"  {c:36s} median = {s.median():+.3f}  "
                  f"IQR = [{s.quantile(0.25):+.3f}, "
                  f"{s.quantile(0.75):+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
