"""
check_avgfp_numbering.py
========================

Independent test of the manuscript claim that GFP-family depositors use
avGFP numbering (Ser65/Tyr66/Gly67, Thr203, ...), so the gatekeeper tally
can read deposited PDB residue numbers without structural realignment.

Method
------
Reference = 1EMA (avGFP/S65T), whose deposited numbering is canonical:
chromophore (CRO) at 66, Thr203, His148, Ser205, Glu222.

For each of the 536 gatekeeper representative structures:
  1. Parse the chain bearing the chromophore (residue with CA2/CB2/N2).
  2. Build the chain's one-letter sequence + a parallel array of the
     DEPOSITED residue numbers (read from the CIF ATOM records). The
     fused chromophore residue is written as a single 'Y' anchor.
  3. Global-align that sequence to the 1EMA reference (BLOSUM62).
  4. For each avGFP anchor (66, 148, 203, 205, 222), find the target
     residue that aligns to that reference column and read its deposited
     number. offset = deposited_number - avGFP_number.

If depositors used avGFP numbering, offset == 0 (esp. at 203, the
gatekeeper site). Non-zero offsets flag structures whose numbering is
NOT in avGFP register -> the cases the manuscript must not pool.

Writes data/avgfp_numbering_check.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import gemmi
from Bio.Align import PairwiseAligner, substitution_matrices

CIF_DIR = Path("data/cif")
REP_CSV = Path("data/gatekeeper_hits.csv")
REF_PDB = "1EMA"
ANCHORS = [66, 148, 203, 205, 222]      # avGFP numbers to probe
BRIDGE = ("CA2", "CB2", "N2")

aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.mode = "global"


def one_letter(resname: str) -> str | None:
    info = gemmi.find_tabulated_residue(resname)
    if info and info.is_amino_acid():
        c = info.one_letter_code.upper()
        return c if c.isalpha() else "X"
    return None


def chain_with_chromophore(model):
    """Return (chain, chrom_seqid) for the first chain holding a residue
    with the CA2/CB2/N2 bridge."""
    for chain in model:
        for res in chain:
            names = {a.name for a in res}
            if all(b in names for b in BRIDGE):
                return chain, res.seqid.num
    return None, None


def seq_and_numbers(chain, chrom_seqid):
    """One-letter sequence + deposited-number array for one chain.
    The chromophore residue is emitted as a single 'Y' anchor."""
    seq, nums = [], []
    seen = set()
    for res in chain:
        n = res.seqid.num
        if n in seen:          # skip altloc / microhetero duplicates
            continue
        if res.seqid.num == chrom_seqid:
            seq.append("Y"); nums.append(n); seen.add(n); continue
        c = one_letter(res.name)
        if c is None:
            continue           # water / ligand / ion
        seq.append(c); nums.append(n); seen.add(n)
    return "".join(seq), np.array(nums)


def build_reference():
    st = gemmi.read_structure(str(CIF_DIR / f"{REF_PDB}.cif"))
    chain, chrom = chain_with_chromophore(st[0])
    seq, nums = seq_and_numbers(chain, chrom)
    num_to_idx = {int(n): i for i, n in enumerate(nums)}
    return seq, nums, num_to_idx


def aligned_partner(aln, ref_idx):
    """Given an alignment (target vs ref), return the target sequence
    index aligned to reference index ref_idx, or None if ref_idx is in
    a gap. aln.aligned is (target_blocks, ref_blocks)."""
    tblocks, rblocks = aln.aligned
    for (t0, t1), (r0, r1) in zip(tblocks, rblocks):
        if r0 <= ref_idx < r1:
            return t0 + (ref_idx - r0)
    return None


def main():
    ref_seq, ref_nums, ref_num_to_idx = build_reference()
    for a in ANCHORS:
        assert a in ref_num_to_idx, f"avGFP anchor {a} missing in reference"
    print(f"Reference {REF_PDB}: {len(ref_seq)} residues; "
          f"anchor residues = "
          + ", ".join(f"{a}:{ref_seq[ref_num_to_idx[a]]}" for a in ANCHORS))

    reps = pd.read_csv(REP_CSV)
    pdb_ids = sorted(reps["pdb_id"].unique())
    color = dict(zip(reps["pdb_id"], reps["color_class"]))

    rows = []
    failures = []
    for pid in pdb_ids:
        cif = CIF_DIR / f"{pid}.cif"
        if not cif.exists():
            failures.append((pid, "no_cif")); continue
        try:
            st = gemmi.read_structure(str(cif))
            chain, chrom = chain_with_chromophore(st[0])
            if chain is None:
                failures.append((pid, "no_chromophore")); continue
            tseq, tnums = seq_and_numbers(chain, chrom)
            if len(tseq) < 50:
                failures.append((pid, "short_chain")); continue
            aln = aligner.align(tseq, ref_seq)[0]
            row = {"pdb_id": pid, "color_class": color.get(pid),
                   "chrom_dep_num": int(chrom), "n_res": len(tseq),
                   "aln_score": aln.score}
            for a in ANCHORS:
                ti = aligned_partner(aln, ref_num_to_idx[a])
                if ti is None:
                    row[f"dep_{a}"] = np.nan
                    row[f"off_{a}"] = np.nan
                    row[f"aa_{a}"] = "-"
                else:
                    dep = int(tnums[ti])
                    row[f"dep_{a}"] = dep
                    row[f"off_{a}"] = dep - a
                    row[f"aa_{a}"] = tseq[ti]
            rows.append(row)
        except Exception as e:  # noqa
            failures.append((pid, f"err:{type(e).__name__}"))

    df = pd.DataFrame(rows)
    out = Path("data/avgfp_numbering_check.csv")
    df.to_csv(out, index=False)
    print(f"\nChecked {len(df)} structures; {len(failures)} failures")
    if failures:
        from collections import Counter
        print("  failure reasons:", dict(Counter(r for _, r in failures)))

    # ---- summary ----
    print("\n=== Chromophore deposited number (avGFP expects 66) ===")
    print(df["chrom_dep_num"].value_counts().head(10).to_string())

    for a in ANCHORS:
        off = df[f"off_{a}"]
        ok = (off == 0).sum()
        n = off.notna().sum()
        print(f"\n=== avGFP {a} ({ref_seq[ref_num_to_idx[a]]}): "
              f"deposited number == {a} in {ok}/{n} "
              f"({100*ok/n:.1f}%) ===")
        nz = df[(off != 0) & off.notna()]
        if len(nz):
            print("  offset distribution:",
                  dict(off[off != 0].value_counts().head(8)))
            print(f"  {len(nz)} structures off-register at {a}; "
                  "by color:",
                  dict(nz["color_class"].value_counts()))

    # residue identity at 203 (should be T in green, Y in T203Y yellow)
    print("\n=== Identity at the residue aligned to avGFP-203, by color ===")
    aa203 = df.groupby("color_class")["aa_203"].value_counts()
    print(aa203.to_string())


if __name__ == "__main__":
    main()
