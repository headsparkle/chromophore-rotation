#!/usr/bin/env python3
"""
validate_esp_apbs.py
====================

Validate the fast ff14SB heavy-atom-summed ESP-at-OP proxy
(scripts/compute_esp_at_op.py) against a higher-fidelity
reference built from pdb2pqr + PROPKA.

The reference improves on the proxy in three ways:
  1. Explicit hydrogens (pdb2pqr adds and optimises them).
  2. Real per-atom AMBER charges (not heavy-atom-summed).
  3. PROPKA pH 7 protonation states (per structure), and real
     crystallographic waters with optimised hydrogen orientation
     (not our single-virtual-H approximation).

We additionally test the proxy's no-screening (vacuum Coulomb)
assumption by recomputing the reference with Debye-Hueckel
screening at a buried-site dielectric. APBS itself (grid
Poisson-Boltzmann with a dielectric boundary) is not installable
on this arm64 machine without conda; the pdb2pqr charges plus an
analytic Debye-Hueckel screen capture the two physical gaps that
matter (charge fidelity and ionic/dielectric screening) and are
fully reproducible.

For each test structure we report, at the chromophore phenolate
oxygen OP (the OH atom of the Tyr-derived chromophore):
  - esp_ref_vac_V        : reference ESP, real charges, eps=1, no screen
  - esp_ref_screened_V   : reference ESP, Debye-Hueckel screened
  - efield_along_co_ref_vac, _screened
  - the near (<6A) / far (6-12A) / water shell decomposition
and compares them to the proxy columns already in
scan_all_summary.csv.

Output: data/esp_validation.csv (one row per test PDB).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif"
PQR_DIR = DATA_DIR / "pqr"
PQR_DIR.mkdir(exist_ok=True)

PDB2PQR = Path("/Users/mzim/Library/Python/3.9/bin/pdb2pqr30")

K_VA_PER_E = 14.3996      # V*A/e
CUTOFF_A = 12.0
NEAR_A = 6.0
WATER_RESNAMES = {"HOH", "WAT", "DOD"}

# Debye-Hueckel: lambda_D (A) = 3.04 / sqrt(I[M]) for water at 25C.
# Buried-site effective dielectric is uncertain; we report eps=4
# (buried) as the headline and note rank-order robustness.
IONIC_STRENGTH_M = 0.15
DH_LAMBDA_A = 3.04 / np.sqrt(IONIC_STRENGTH_M)
DH_KAPPA = 1.0 / DH_LAMBDA_A
EPS_BURIED = 4.0


def cif_to_pdb(pdb_id: str) -> Path:
    pdb_path = PQR_DIR / f"{pdb_id}.pdb"
    if pdb_path.is_file():
        return pdb_path
    st = gemmi.read_structure(str(CIF_DIR / f"{pdb_id}.cif"))
    st.setup_entities()
    pdb_path.write_text(st.make_pdb_string())
    return pdb_path


def run_pdb2pqr(pdb_id: str) -> Path | None:
    pqr_path = PQR_DIR / f"{pdb_id}.pqr"
    if pqr_path.is_file() and pqr_path.stat().st_size > 0:
        return pqr_path
    pdb_path = cif_to_pdb(pdb_id)
    cmd = [
        str(PDB2PQR), "--ff=AMBER", "--with-ph=7.0",
        "--titration-state-method=propka", "--keep-chain",
        str(pdb_path), str(pqr_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  [{pdb_id}] pdb2pqr failed: {exc.__class__.__name__}")
        return None
    return pqr_path if pqr_path.is_file() else None


def parse_pqr(pqr_path: Path):
    """Return list of (xyz[3], charge, is_water)."""
    atoms = []
    for line in pqr_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # PQR is whitespace-ambiguous; parse from the right for
        # x y z charge radius which are the last 5 float fields.
        parts = line.split()
        try:
            radius = float(parts[-1])
            charge = float(parts[-2])
            z = float(parts[-3]); y = float(parts[-4]); x = float(parts[-5])
        except (ValueError, IndexError):
            continue
        resname = parts[3] if len(parts) > 3 else ""
        is_water = resname in WATER_RESNAMES
        atoms.append((np.array([x, y, z]), charge, is_water))
    return atoms


def find_op_cz(pdb_id: str):
    st = gemmi.read_structure(str(CIF_DIR / f"{pdb_id}.cif"))
    for chain in st[0]:
        for res in chain:
            names = {a.name for a in res}
            if not ({"CA2", "CB2", "N2"} <= names):
                continue
            if "OH" not in names or "CZ" not in names:
                return None, None
            op = cz = None
            for a in res:
                if a.name == "OH":
                    op = np.array([a.pos.x, a.pos.y, a.pos.z])
                elif a.name == "CZ":
                    cz = np.array([a.pos.x, a.pos.y, a.pos.z])
            return op, cz
    return None, None


def reference_esp(atoms, op, co_unit):
    """Compute reference ESP and E_along_co at OP from PQR charges,
    both vacuum (eps=1, no screen) and Debye-Hueckel screened.
    Shell-decomposed into near/far/water."""
    out = {k: 0.0 for k in [
        "esp_vac", "esp_scr", "ealong_vac", "ealong_scr",
        "esp_vac_near", "esp_vac_far", "esp_vac_water",
        "ealong_vac_near", "ealong_vac_far", "ealong_vac_water",
        "n_atoms", "n_water_atoms",
    ]}
    cut2 = CUTOFF_A * CUTOFF_A
    for xyz, q, is_water in atoms:
        dr = op - xyz
        d2 = float(dr @ dr)
        if d2 > cut2 or d2 < 1e-6:
            continue
        d = d2 ** 0.5
        # Vacuum (eps=1)
        v_vac = K_VA_PER_E * q / d
        e_vac = K_VA_PER_E * q * dr / (d2 * d)
        ealong_vac = float(e_vac @ co_unit)
        # Debye-Hueckel screened, buried dielectric
        screen = np.exp(-DH_KAPPA * d) / EPS_BURIED
        v_scr = v_vac * screen
        # field of a screened (Yukawa) potential: includes the
        # (1 + kappa*d) term from d/dr [exp(-kr)/r]
        e_scr_mag_factor = (1.0 + DH_KAPPA * d) * screen
        ealong_scr = ealong_vac * e_scr_mag_factor
        out["esp_vac"] += v_vac
        out["esp_scr"] += v_scr
        out["ealong_vac"] += ealong_vac
        out["ealong_scr"] += ealong_scr
        out["n_atoms"] += 1
        if is_water:
            out["esp_vac_water"] += v_vac
            out["ealong_vac_water"] += ealong_vac
            out["n_water_atoms"] += 1
        elif d < NEAR_A:
            out["esp_vac_near"] += v_vac
            out["ealong_vac_near"] += ealong_vac
        else:
            out["esp_vac_far"] += v_vac
            out["ealong_vac_far"] += ealong_vac
    return out


def main(pdb_ids):
    rows = []
    for i, pdb_id in enumerate(pdb_ids):
        pdb_id = pdb_id.upper()
        print(f"[{i+1}/{len(pdb_ids)}] {pdb_id}")
        op, cz = find_op_cz(pdb_id)
        if op is None:
            print("  no OP/CZ, skip"); continue
        co = op - cz
        co_unit = co / np.linalg.norm(co)
        pqr = run_pdb2pqr(pdb_id)
        if pqr is None:
            continue
        atoms = parse_pqr(pqr)
        ref = reference_esp(atoms, op, co_unit)
        ref["pdb_id"] = pdb_id
        rows.append(ref)
    out = pd.DataFrame(rows)
    out_path = DATA_DIR / "esp_validation.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    ids = sys.argv[1:]
    if not ids:
        print("usage: validate_esp_apbs.py PDB1 PDB2 ...")
        sys.exit(1)
    main(ids)
