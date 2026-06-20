#!/usr/bin/env python3
"""
recover_qy_by_sequence.py
=========================

For each PDB in our scanned set, extract the protein sequence from
the deposited CIF and look for an FPbase entry with at least 99 %
identity. When found, inherit FPbase's curated QY (and ex/em max,
extinction coefficient) for that PDB. This recovers the FPs we
couldn't reach by direct PDB-ID match against FPbase.

Method
------
1. Sequence extraction. Use gemmi to read each CIF, walk chain A (or
   whichever chain holds the chromophore residue), and emit the
   one-letter sequence of standard amino acids. Non-standard residues
   - including the chromophore (CRO etc.), waters, ligands - are
   skipped. The chromophore is replaced by a single 'X' to preserve
   register; subsequent identity is computed ignoring 'X' positions
   on either side.

2. Alignment. BioPython PairwiseAligner with BLOSUM62, global
   alignment with end-gap weight 0 (so N/C-terminal His tags and
   linker stubs don't penalise). Identity = matches / shorter sequence
   length (after stripping any 'X' chromophore tokens).

3. Threshold. Inherit QY when best-match identity >= 0.99.

Output
------
Overwrites data/lit_qy_curated.csv to add columns:
    qy_provenance:
        - 'pdb_match'      : already matched via PDB ID -> FPbase
        - 'seq_match_99'   : newly recovered by >=99 % sequence
        - 'unmatched'      : no FPbase QY available
    seq_match_slug, seq_match_identity, seq_match_n_aligned:
        diagnostics on the sequence hit (NaN if pdb_match or unmatched)

This script is slow on the first run because every CIF is opened and
every sequence is aligned against ~1000 FPbase records. We cache
extracted PDB sequences in data/pdb_sequences.csv so re-runs only
re-do the alignment step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif"

SUMMARY_CSV = DATA_DIR / "scan_all_summary.csv"
CURATED_CSV = DATA_DIR / "lit_qy_curated.csv"
FPBASE_JSON = DATA_DIR / "fpbase_proteins.json"
SEQ_CACHE = DATA_DIR / "pdb_sequences.csv"

IDENTITY_THRESHOLD = 0.99


# 20 standard amino acids; gemmi gives 3-letter; we keep a manual table
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",   # selenomethionine -> Met
}

# Same chromophore set as barrel.py - these residues become 'X'.
CHROM_RESNAMES = {
    "CRO", "CR2", "GYS", "SYG", "CRQ", "CRF", "CRW", "CRY",
    "NRQ", "NYG", "CH6", "CH7", "CRG", "CRU", "CRV", "CRS",
    "66A", "CR0", "GYC", "LYG", "TYG", "OHD", "SWG", "QYG",
    "CR7", "CR8", "CR9", "CRK", "RC7", "CFY", "PIA", "B2H",
    "IIC",
}


def extract_sequence(cif_path: Path) -> str:
    """Return the one-letter sequence of chain A (or whichever chain
    holds a chromophore) with the chromophore replaced by 'X'.

    Falls back to the first protein chain if no chromophore is found.
    Non-standard non-chromophore residues are dropped silently.
    """
    struct = gemmi.read_structure(str(cif_path))
    if not len(struct):
        return ""
    model = struct[0]

    # Find the chain containing the chromophore.
    chrom_chain = None
    for chain in model:
        for res in chain:
            if res.name in CHROM_RESNAMES:
                chrom_chain = chain
                break
        if chrom_chain is not None:
            break
    if chrom_chain is None:
        # Fallback: first chain with any standard amino acids.
        for chain in model:
            for res in chain:
                if res.name in THREE_TO_ONE:
                    chrom_chain = chain
                    break
            if chrom_chain is not None:
                break
    if chrom_chain is None:
        return ""

    parts: list[str] = []
    for res in chrom_chain:
        if res.name in THREE_TO_ONE:
            parts.append(THREE_TO_ONE[res.name])
        elif res.name in CHROM_RESNAMES:
            parts.append("X")
        # else: ions, waters, ligands - skip silently
    return "".join(parts)


def load_or_build_pdb_seqs(pdb_ids: list[str]) -> pd.DataFrame:
    """Cached: per-PDB one-letter sequence with 'X' at the chromophore."""
    if SEQ_CACHE.is_file():
        cache = pd.read_csv(SEQ_CACHE).set_index("pdb_id")["seq"].to_dict()
    else:
        cache = {}

    rows = []
    t0 = time.perf_counter()
    miss = 0
    for pdb in pdb_ids:
        if pdb in cache and isinstance(cache[pdb], str) and cache[pdb]:
            rows.append({"pdb_id": pdb, "seq": cache[pdb]})
            continue
        cif = CIF_DIR / f"{pdb}.cif"
        if not cif.is_file():
            rows.append({"pdb_id": pdb, "seq": ""})
            miss += 1
            continue
        try:
            seq = extract_sequence(cif)
        except Exception as e:
            print(f"  [skip] {pdb}: {e}")
            seq = ""
        rows.append({"pdb_id": pdb, "seq": seq})
    df = pd.DataFrame(rows)
    df.to_csv(SEQ_CACHE, index=False)
    dt = time.perf_counter() - t0
    print(f"[seq] {len(df)} PDB sequences ready in {dt:.1f}s "
          f"({miss} missing CIFs)")
    return df


def fpbase_seqs() -> pd.DataFrame:
    import json
    proteins = json.loads(FPBASE_JSON.read_text())
    rows = []
    for p in proteins:
        seq = p.get("seq") or ""
        if not seq or len(seq) < 100:  # too short to be an FP barrel
            continue
        # Pick a state with non-null QY; otherwise default state
        state = None
        for s in p.get("states") or []:
            if s.get("qy") is not None:
                state = s
                break
        if state is None and p.get("states"):
            state = p["states"][0]
        rows.append({
            "fpbase_slug": p.get("slug"),
            "fpbase_name": p.get("name"),
            "fpbase_seq": seq.upper(),
            "fpbase_qy": (state or {}).get("qy"),
            "fpbase_ex": (state or {}).get("ex_max"),
            "fpbase_em": (state or {}).get("em_max"),
            "fpbase_ec": (state or {}).get("ext_coeff"),
        })
    return pd.DataFrame(rows)


def _count_matches(aln, first: str, second: str) -> int:
    """Walk aligned blocks and count columns where the two sequences
    have identical characters.

    `first` is the sequence passed as the first arg to aligner.align,
    `second` is the sequence passed as the second arg. BioPython's
    aln.coordinates is a (2, n_break+1) array; row 0 indexes into
    `first`, row 1 indexes into `second`. Between consecutive
    breakpoints either both advance equally (aligned block) or one
    stays put (a gap, contributes zero matches).
    """
    import numpy as np
    coords = np.asarray(aln.coordinates)
    n = 0
    for k in range(coords.shape[1] - 1):
        a0, a1 = int(coords[0, k]), int(coords[0, k + 1])
        b0, b1 = int(coords[1, k]), int(coords[1, k + 1])
        if (a1 - a0) == (b1 - b0) and (a1 - a0) > 0:
            for ac, bc in zip(first[a0:a1], second[b0:b1]):
                if ac == bc:
                    n += 1
    return n


def best_match(query: str, db: list[tuple[str, str, float]],
               aligner: PairwiseAligner) -> tuple[int, float, int]:
    """Return (best_idx, best_identity, best_n_aligned).
    db: list of (slug, seq, qy). We compute identity = exact_matches /
    min(len(query), len(target)) where matches are counted from the
    top-scoring alignment.

    Pre-filter on length: if |len_q - len_t| / len_q > 0.05, skip.
    """
    q = query.replace("X", "")
    if len(q) < 100:
        return -1, 0.0, 0
    best_idx, best_id, best_n = -1, 0.0, 0
    for i, (slug, seq, _qy) in enumerate(db):
        t = seq.replace("X", "")
        if not t:
            continue
        shorter = min(len(q), len(t))
        if shorter < 100:
            continue
        if abs(len(q) - len(t)) / shorter > 0.05:
            continue
        try:
            aln = aligner.align(q, t)[0]
        except Exception:
            continue
        n_match = _count_matches(aln, q, t)  # aligned in same order as align(q, t)
        ident = n_match / shorter
        if ident > best_id:
            best_id, best_idx, best_n = ident, i, n_match
            if best_id >= 0.999:
                break  # essentially identical, no need to keep searching
    return best_idx, best_id, best_n


def main() -> int:
    if not FPBASE_JSON.is_file():
        sys.exit(f"missing {FPBASE_JSON}")
    if not CURATED_CSV.is_file():
        sys.exit(f"missing {CURATED_CSV} - run curate_qy_from_fpbase.py first")

    cur = pd.read_csv(CURATED_CSV)
    cur["pdb_id"] = cur["pdb_id"].astype(str).str.upper()
    print(f"[load] {len(cur)} PDBs in curated table; "
          f"{cur['lit_qy_fpbase'].notna().sum()} already have QY")

    fp = fpbase_seqs()
    print(f"[fpbase] {len(fp)} FPbase records with seq>=100 aa")
    fp_with_qy = fp[fp["fpbase_qy"].notna()].reset_index(drop=True)
    print(f"[fpbase] {len(fp_with_qy)} of those have a non-null QY")

    pdbs = cur["pdb_id"].tolist()
    seqs = load_or_build_pdb_seqs(pdbs)
    cur = cur.merge(seqs, on="pdb_id", how="left")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    db = [
        (r["fpbase_slug"], r["fpbase_seq"], r["fpbase_qy"])
        for _, r in fp_with_qy.iterrows()
    ]

    provenance = []
    seq_slug = []
    seq_ident = []
    seq_naln = []
    seq_qy = []
    seq_ex = []
    seq_em = []
    seq_ec = []
    seq_name = []

    fp_with_qy_indexed = fp_with_qy.set_index("fpbase_slug")

    t0 = time.perf_counter()
    n_seq_hits = 0
    for i, row in cur.iterrows():
        if pd.notna(row["lit_qy_fpbase"]):
            provenance.append("pdb_match")
            seq_slug.append(None); seq_ident.append(np.nan); seq_naln.append(0)
            seq_qy.append(np.nan); seq_ex.append(np.nan); seq_em.append(np.nan)
            seq_ec.append(np.nan); seq_name.append(None)
            continue
        q = row.get("seq") or ""
        if not q or len(q.replace("X", "")) < 100:
            provenance.append("unmatched")
            seq_slug.append(None); seq_ident.append(np.nan); seq_naln.append(0)
            seq_qy.append(np.nan); seq_ex.append(np.nan); seq_em.append(np.nan)
            seq_ec.append(np.nan); seq_name.append(None)
            continue
        idx, ident, n_aln = best_match(q, db, aligner)
        if idx >= 0 and ident >= IDENTITY_THRESHOLD:
            slug, _, qy = db[idx]
            rec = fp_with_qy_indexed.loc[slug]
            provenance.append("seq_match_99")
            seq_slug.append(slug)
            seq_ident.append(ident)
            seq_naln.append(n_aln)
            seq_qy.append(qy)
            seq_ex.append(rec["fpbase_ex"])
            seq_em.append(rec["fpbase_em"])
            seq_ec.append(rec["fpbase_ec"])
            seq_name.append(rec["fpbase_name"])
            n_seq_hits += 1
        else:
            provenance.append("unmatched")
            seq_slug.append(None); seq_ident.append(np.nan); seq_naln.append(0)
            seq_qy.append(np.nan); seq_ex.append(np.nan); seq_em.append(np.nan)
            seq_ec.append(np.nan); seq_name.append(None)
        if (i + 1) % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"  [match] {i+1}/{len(cur)}, {n_seq_hits} new seq hits, "
                  f"{dt:.1f}s elapsed")
    dt = time.perf_counter() - t0
    print(f"[match] done in {dt:.1f}s, {n_seq_hits} new sequence-based hits")

    cur["qy_provenance"] = provenance
    cur["seq_match_slug"] = seq_slug
    cur["seq_match_name"] = seq_name
    cur["seq_match_identity"] = seq_ident
    cur["seq_match_n_aligned"] = seq_naln
    # Recovered QY goes into a parallel column AND backfills lit_qy_fpbase
    cur["lit_qy_fpbase_recovered"] = seq_qy
    cur["ex_max_fpbase_recovered"] = seq_ex
    cur["em_max_fpbase_recovered"] = seq_em
    cur["ext_coeff_fpbase_recovered"] = seq_ec

    fill_mask = cur["lit_qy_fpbase"].isna() & cur["lit_qy_fpbase_recovered"].notna()
    cur.loc[fill_mask, "lit_qy_fpbase"] = cur.loc[fill_mask, "lit_qy_fpbase_recovered"]
    cur.loc[fill_mask, "ex_max_fpbase"] = cur.loc[fill_mask, "ex_max_fpbase_recovered"]
    cur.loc[fill_mask, "em_max_fpbase"] = cur.loc[fill_mask, "em_max_fpbase_recovered"]
    cur.loc[fill_mask, "ext_coeff_fpbase"] = cur.loc[fill_mask, "ext_coeff_fpbase_recovered"]

    # Don't ship the chunky per-PDB seq into the curated csv.
    if "seq" in cur.columns:
        cur = cur.drop(columns=["seq"])

    cur.to_csv(CURATED_CSV, index=False)
    print(f"[save] {CURATED_CSV}")
    print()
    print("provenance counts:")
    print(cur["qy_provenance"].value_counts())
    print()
    n_qy = cur["lit_qy_fpbase"].notna().sum()
    print(f"final PDBs with curated QY: {n_qy} "
          f"(of {len(cur)} scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
