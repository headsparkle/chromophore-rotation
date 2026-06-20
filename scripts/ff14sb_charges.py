"""
ff14sb_charges.py
=================

AMBER ff14SB partial charges for the 20 standard amino acids,
with all hydrogen charges summed onto their parent heavy atom.

This module exists because almost no PDB X-ray structure in our
848-FP cohort has hydrogens. Computing the electrostatic
potential at the chromophore phenolate oxygen using ff14SB
charges requires the H contributions to be carried somewhere,
and the conventional approach when working from heavy-atom-only
PDBs is to fold each H charge onto its directly bonded heavy
atom. The resulting heavy-atom partial charge captures the
net dipole of that atom + its hydrogens. This is the same
approximation made by tools such as PDB2PQR when atoms are
missing.

Numbers are derived from the canonical ff14SB residue files
(e.g. amber14/protein.ff14SB.xml in OpenMM, or the GAFF /
parm10 source files in AmberTools), with each H charge added to
its bonded heavy atom.

Histidine is provided in three protonation states:
- HID (delta-protonated, neutral)
- HIE (epsilon-protonated, neutral; the most common form at
       physiological pH and our default)
- HIP (doubly protonated, +1)

For "HIS" with unknown tautomer we default to HIE.

Each entry is a dict mapping atom name -> partial charge in
elementary-charge units.
"""

# Backbone charges. Apply to every standard residue except PRO
# (which has no NH proton; the N charge is different).
BACKBONE_STANDARD = {
    "N": -0.4157 + 0.2719,   # N + H summed
    "CA": 0.0337 + 0.0823,   # CA + HA
    "C": 0.5973,
    "O": -0.5679,
}

BACKBONE_PRO = {
    "N": -0.2548,            # No NH proton in Pro
    "CA": -0.0266 + 0.0641,  # CA + HA
    "C": 0.5896,
    "O": -0.5748,
}

# Glycine has 2 HAs and no CB
BACKBONE_GLY = {
    "N": -0.4157 + 0.2719,
    "CA": -0.0252 + 2 * 0.0698,
    "C": 0.5973,
    "O": -0.5679,
}

# Side-chain charges, with H charges summed onto their parent
# heavy atom. Backbone (N/CA/C/O) is handled separately.

SIDECHAIN = {
    "ALA": {
        "CB": -0.1825 + 3 * 0.0603,
    },
    "ARG": {
        "CB": -0.0007 + 2 * 0.0327,
        "CG": 0.0390 + 2 * 0.0285,
        "CD": 0.0486 + 2 * 0.0687,
        "NE": -0.5295 + 0.3456,
        "CZ": 0.8076,
        "NH1": -0.8627 + 2 * 0.4478,
        "NH2": -0.8627 + 2 * 0.4478,
    },
    "ASN": {
        "CB": -0.2041 + 2 * 0.0797,
        "CG": 0.7130,
        "OD1": -0.5931,
        "ND2": -0.9191 + 2 * 0.4196,
    },
    "ASP": {
        "CB": -0.0303 + 2 * (-0.0122),
        "CG": 0.7994,
        "OD1": -0.8014,
        "OD2": -0.8014,
    },
    "CYS": {
        "CB": -0.1231 + 2 * 0.1112,
        "SG": -0.3119 + 0.1933,
    },
    "GLN": {
        "CB": -0.0036 + 2 * 0.0171,
        "CG": -0.0645 + 2 * 0.0352,
        "CD": 0.6951,
        "OE1": -0.6086,
        "NE2": -0.9407 + 2 * 0.4251,
    },
    "GLU": {
        "CB": 0.0398 + 2 * (-0.0173),
        "CG": 0.0560 + 2 * (-0.0211),
        "CD": 0.8054,
        "OE1": -0.8188,
        "OE2": -0.8188,
    },
    "GLY": {},  # no side chain; backbone uses BACKBONE_GLY
    "HID": {  # delta-protonated
        "CB": -0.0462 + 2 * 0.0402,
        "CG": -0.0266,
        "ND1": -0.3811 + 0.3649,
        "CE1": 0.2057 + 0.1392,
        "NE2": -0.5727,
        "CD2": 0.1292 + 0.1147,
    },
    "HIE": {  # epsilon-protonated (default neutral form)
        "CB": -0.0074 + 2 * 0.0367,
        "CG": 0.1868,
        "ND1": -0.5432,
        "CE1": 0.1635 + 0.1435,
        "NE2": -0.2795 + 0.3339,
        "CD2": -0.2207 + 0.1862,
    },
    "HIP": {  # doubly protonated (+1)
        "CB": -0.0414 + 2 * 0.0810,
        "CG": -0.0012,
        "ND1": -0.1513 + 0.3866,
        "CE1": -0.0170 + 0.2681,
        "NE2": -0.1718 + 0.3911,
        "CD2": -0.1141 + 0.2317,
    },
    "ILE": {
        "CB": 0.1303 + 0.0187,
        "CG1": -0.0430 + 2 * 0.0236,
        "CG2": -0.3204 + 3 * 0.0882,
        "CD1": -0.0660 + 3 * 0.0186,
    },
    "LEU": {
        "CB": -0.1102 + 2 * 0.0457,
        "CG": 0.3531 + (-0.0361),
        "CD1": -0.4121 + 3 * 0.1000,
        "CD2": -0.4121 + 3 * 0.1000,
    },
    "LYS": {
        "CB": -0.0479 + 2 * 0.0651,
        "CG": 0.0187 + 2 * 0.0103,
        "CD": -0.0143 + 2 * 0.0288,
        "CE": -0.0117 + 2 * 0.0997,
        "NZ": -0.3854 + 3 * 0.3338,
    },
    "MET": {
        "CB": 0.0342 + 2 * 0.0241,
        "CG": 0.0018 + 2 * 0.0440,
        "SD": -0.2737,
        "CE": -0.0536 + 3 * 0.0684,
    },
    "PHE": {
        "CB": -0.0343 + 2 * 0.0295,
        "CG": 0.0118,
        "CD1": -0.1256 + 0.1330,
        "CD2": -0.1256 + 0.1330,
        "CE1": -0.1704 + 0.1430,
        "CE2": -0.1704 + 0.1430,
        "CZ": -0.1072 + 0.1297,
    },
    "PRO": {
        "CB": -0.0070 + 2 * 0.0253,
        "CG": 0.0189 + 2 * 0.0213,
        "CD": 0.0192 + 2 * 0.0391,
    },
    "SER": {
        "CB": 0.2117 + 2 * 0.0352,
        "OG": -0.6546 + 0.4275,
    },
    "THR": {
        "CB": 0.3654 + 0.0043,
        "OG1": -0.6761 + 0.4102,
        "CG2": -0.2438 + 3 * 0.0642,
    },
    "TRP": {
        "CB": -0.0050 + 2 * 0.0339,
        "CG": -0.1415,
        "CD1": -0.1638 + 0.2062,
        "NE1": -0.3418 + 0.3412,
        "CE2": 0.1380,
        "CZ2": -0.2601 + 0.1572,
        "CH2": -0.1134 + 0.1417,
        "CZ3": -0.1972 + 0.1447,
        "CE3": -0.2387 + 0.1700,
        "CD2": 0.1243,
    },
    "TYR": {
        "CB": -0.0152 + 2 * 0.0295,
        "CG": -0.0011,
        "CD1": -0.1906 + 0.1699,
        "CD2": -0.1906 + 0.1699,
        "CE1": -0.2341 + 0.1656,
        "CE2": -0.2341 + 0.1656,
        "CZ": 0.3226,
        "OH": -0.5579 + 0.3992,
    },
    "VAL": {
        "CB": -0.0875 + 0.0969,
        "CG1": -0.3192 + 3 * 0.0791,
        "CG2": -0.3192 + 3 * 0.0791,
    },
    # Selenomethionine: treat as methionine with SE in place of SD
    "MSE": {
        "CB": 0.0342 + 2 * 0.0241,
        "CG": 0.0018 + 2 * 0.0440,
        "SE": -0.2737,  # same magnitude as S in MET
        "CE": -0.0536 + 3 * 0.0684,
    },
}


# Formal charges (used for the per-residue normalisation step).
FORMAL_CHARGES = {
    "ARG": +1, "LYS": +1, "HIP": +1,
    "ASP": -1, "GLU": -1,
    # All others: 0 (default in the function)
}


# Map any residue name to (resolved sidechain key, backbone key).
# HIS without tautomer info defaults to HIE.
def resolve_residue_charges(resname: str, normalise: bool = True):
    """Returns (charges, total) where:
    - charges is a dict atom_name -> partial charge (heavy-atom-summed)
      that combines backbone and side-chain contributions for this
      residue.
    - total is the residue's formal charge.
    Returns ({}, 0.0) for non-standard residues.

    If normalise is True (default), per-residue residual relative to
    the formal charge is distributed uniformly across the heavy atoms.
    This corrects for small inconsistencies in the heavy-atom-summed
    dictionary (per-residue backbone tweaks in ff14SB are not all
    captured by the three default backbones above) while preserving
    the relative dipole structure of each residue. The correction is
    typically smaller than 0.02 e per atom.
    """
    rn = resname.upper()
    if rn == "HIS":
        rn = "HIE"
    if rn not in SIDECHAIN:
        return {}, 0.0
    sc = SIDECHAIN[rn]
    if rn == "GLY":
        bb = BACKBONE_GLY
    elif rn == "PRO":
        bb = BACKBONE_PRO
    else:
        bb = BACKBONE_STANDARD
    charges = {**bb, **sc}
    target = FORMAL_CHARGES.get(rn, 0.0)
    if normalise and charges:
        current = sum(charges.values())
        correction = (target - current) / len(charges)
        charges = {k: v + correction for k, v in charges.items()}
    return charges, float(target)


# Total net charge of each residue (sanity check; standard
# residues should give 0 or +/- 1).
def _net_charge(resname: str, normalise: bool = True) -> float:
    charges, _ = resolve_residue_charges(resname, normalise=normalise)
    if not charges:
        return float("nan")
    return sum(charges.values())


# TIP3P water charges (used for explicit waters within 4 A of OP).
TIP3P_O_CHARGE = -0.834
TIP3P_H_CHARGE = +0.417
TIP3P_OH_BOND_A = 0.9572  # angstrom


__all__ = [
    "SIDECHAIN",
    "BACKBONE_STANDARD",
    "BACKBONE_PRO",
    "BACKBONE_GLY",
    "resolve_residue_charges",
    "TIP3P_O_CHARGE",
    "TIP3P_H_CHARGE",
    "TIP3P_OH_BOND_A",
]


if __name__ == "__main__":
    print("Sanity check: net residue charges")
    print(f"{'residue':>8s}  {'raw':>8s}  {'normalised':>10s}  {'max abs correction':>20s}")
    for rn in ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
               "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET",
               "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"]:
        raw = _net_charge(rn, normalise=False)
        norm = _net_charge(rn, normalise=True)
        charges, target = resolve_residue_charges(rn, normalise=True)
        raw_charges, _ = resolve_residue_charges(rn, normalise=False)
        deltas = [(charges[k] - raw_charges[k]) for k in charges]
        max_delta = max(abs(d) for d in deltas) if deltas else 0.0
        print(f"  {rn:>8s}  {raw:+8.4f}  {norm:+10.4f}  "
              f"{max_delta:+20.4f}")
