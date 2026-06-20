"""
count_hb_contacts.py

For each FP structure in merged_for_aggregate.csv, count the number of
polar heavy atoms (N and O from non-chromophore residues) within a
distance cutoff of the two key chromophore atoms:

  OH  -- phenolate oxygen (para-OH of tyrosyl-derived ring)
  O2  -- imidazolinone carbonyl oxygen (exocyclic C=O of 5-membered ring)

These two atoms flank the methine bridge from opposite sides:
  phenolate (OH) <-- P-bond --> methine bridge <-- I-bond --> imidazolinone (O2)

Motivation: Romei et al. (Science 2020, Boxer group) showed that within a
fixed FP scaffold, electrostatic environment at these two ends determines
which isomerization pathway dominates (P-bond ring flip vs I-bond cis-trans).
A polar contact asymmetry between the two ends is a proxy for electrostatic
asymmetry.

Outputs:
  data/hb_contacts.csv  -- per-structure n_polar_OH, n_polar_O2, delta_polar

Columns:
  pdb_id             -- 4-letter PDB code
  chrom_resname      -- residue name of the chromophore
  n_polar_OH_35      -- polar heavy-atom contacts within 3.5 Ang of OH
  n_polar_O2_35      -- polar heavy-atom contacts within 3.5 Ang of O2
  delta_polar_35     -- n_polar_O2_35 - n_polar_OH_35
  n_polar_OH_32      -- tighter cutoff (3.2 Ang) for likely H-bond partners
  n_polar_O2_32      -- same
  delta_polar_32     -- tighter asymmetry
  oh_found           -- bool: OH atom present in chromophore
  o2_found           -- bool: O2 atom present in chromophore
"""

import os, sys
import pandas as pd
import numpy as np
import gemmi

CIF_DIR = "/Users/mzim/Documents/Projects/chromophore-rotation/data/cif"
MERGED   = "/Users/mzim/Documents/Projects/chromophore-rotation/data/merged_for_aggregate.csv"
OUT_CSV  = "/Users/mzim/Documents/Projects/chromophore-rotation/data/hb_contacts.csv"

CUTOFF_WIDE   = 3.5  # Ang -- all polar contacts
CUTOFF_TIGHT  = 3.2  # Ang -- likely direct H-bond partners

POLAR_ELEMENTS = {"N", "O"}

# chromophore residue names recognised across the 838-structure dataset
CHROM_RESNAMES = {
    "CRO","CR2","NRQ","PIA","GYS","CRQ","CRF","GYC","5SQ","CH6",
    "CR7","CCY","GYS","CRG","CJO","CRK","CZO","DYG","MFC","CSH",
    "0YG","7R0","7R6","A1BE5","A1IJF","B2H","BF6","BJO","C12",
    "CFY","CH7","CIV","CQ1","CQ2","CR0","CRU","FHE","GMO","IEY",
    "IIC","JBY","KXV","KY4","KY7","KZV","KZY","KZ1","KZ4","KZ7",
    "KZG","LKE","M3V","NRP","NYG","OFM","OHD","OIM","QCA","QFG",
    "QLG","QYG","QYX","RC7","SWG","TUK","VUB","X9Q","XXY","XYG",
    "4M9",
}


def process_structure(pdb_id, chrom_resname):
    cif_path = os.path.join(CIF_DIR, pdb_id + ".cif")
    if not os.path.exists(cif_path):
        return None

    try:
        st = gemmi.read_structure(cif_path)
    except Exception as e:
        print(f"  {pdb_id}: read error -- {e}", file=sys.stderr)
        return None

    model = st[0]

    # Collect chromophore atoms (there may be multiple chains/residues if
    # the asymmetric unit has multiple copies -- take first)
    chrom_res = None
    for chain in model:
        for res in chain:
            if res.name == chrom_resname:
                chrom_res = res
                break
        if chrom_res is not None:
            break

    if chrom_res is None:
        return None

    # Get positions of OH and O2
    oh_pos = None
    o2_pos = None
    for atom in chrom_res:
        if atom.name == "OH":
            oh_pos = atom.pos
        elif atom.name == "O2":
            o2_pos = atom.pos

    # Build neighbor search over ALL non-chromophore heavy atoms
    # We exclude the chromophore residue itself (by sequence number)
    chrom_seqid = chrom_res.seqid
    chrom_chain = None
    for chain in model:
        for res in chain:
            if res.seqid == chrom_seqid and res.name == chrom_resname:
                chrom_chain = chain.name
                break
        if chrom_chain is not None:
            break

    ns = gemmi.NeighborSearch(model, st.cell, CUTOFF_WIDE + 0.5).populate()

    # Helper: count polar contacts, excluding the chromophore residue itself
    def count_around(pos, cutoff):
        if pos is None:
            return 0
        results = ns.find_atoms(pos, "\0", radius=cutoff)
        count = 0
        for item in results:
            if item.element.name not in POLAR_ELEMENTS:
                continue
            cra = item.to_cra(model)
            if cra.residue.name in CHROM_RESNAMES:
                continue
            count += 1
        return count

    n_oh_35 = count_around(oh_pos, CUTOFF_WIDE)
    n_o2_35 = count_around(o2_pos, CUTOFF_WIDE)
    n_oh_32 = count_around(oh_pos, CUTOFF_TIGHT)
    n_o2_32 = count_around(o2_pos, CUTOFF_TIGHT)

    return {
        "pdb_id":         pdb_id,
        "chrom_resname":  chrom_resname,
        "n_polar_OH_35":  n_oh_35,
        "n_polar_O2_35":  n_o2_35,
        "delta_polar_35": n_o2_35 - n_oh_35,
        "n_polar_OH_32":  n_oh_32,
        "n_polar_O2_32":  n_o2_32,
        "delta_polar_32": n_o2_32 - n_oh_32,
        "oh_found":       oh_pos is not None,
        "o2_found":       o2_pos is not None,
    }


def main():
    df = pd.read_csv(MERGED)
    print(f"Loaded {len(df)} structures from merged_for_aggregate.csv")

    rows = []
    n_ok = 0
    n_skip = 0
    for i, row in df.iterrows():
        pdb_id      = row["pdb_id"]
        chrom_rname = row["chrom_resname"]
        result = process_structure(pdb_id, chrom_rname)
        if result is None:
            n_skip += 1
            continue
        rows.append(result)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  processed {n_ok} structures...", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nDone. {n_ok} ok, {n_skip} skipped.")
    print(f"Output: {OUT_CSV}")

    # Quick sanity: mean contacts per site
    print(f"\nMean n_polar_OH_35: {out.n_polar_OH_35.mean():.2f}")
    print(f"Mean n_polar_O2_35: {out.n_polar_O2_35.mean():.2f}")
    print(f"Mean delta_35:      {out.delta_polar_35.mean():.2f}")
    print(f"OH found: {out.oh_found.sum()} / {len(out)}")
    print(f"O2 found: {out.o2_found.sum()} / {len(out)}")


if __name__ == "__main__":
    main()
