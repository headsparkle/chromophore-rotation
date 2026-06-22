#!/usr/bin/env bash
# Assemble the core code+data deposit archive for Zenodo/GitHub.
# Excludes the large/regenerable directories (cif, pqr, scans, scans_25deg) and
# backup/intermediate CSVs. Run from the project root.
set -euo pipefail
cd "$(dirname "$0")"

OUT="chromophore-rotation-deposit"
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/scripts" "$OUT/data"

# all analysis code
cp scripts/*.py "$OUT/scripts/"

# core curated/derived data tables (the reproducibility set; no backups, no bulk)
CORE_CSVS=(
  supplementary_SD1_per_structure.csv
  supplementary_SD2_per_unique_fp.csv
  merged_for_aggregate.csv
  scan_all_summary.csv
  scan_1d_summary.csv
  d_exp_canonical.csv
  canonical_cd_twist.csv
  d_exp_altloc_fixed.csv
  lit_qy_curated.csv
  pdb_sequences.csv
  hb_contacts.csv
  gatekeeper_summary.csv
  gatekeeper_hits.csv
  gatekeeper_boundary_summary.csv
  matched_pairs.csv
  matched_pairs_permutation.csv
  loocv_results.csv
  loocv_predictions.csv
  bootstrap_cis.csv
  twist_decomposition.csv
  eval_confirmatory.csv
  heteroatom_sensitivity.csv
  relaxed_param_sweep.csv
  gatekeeper_energy_203.csv
  gatekeeper_energy_203_real.csv
  avgfp_numbering_check.csv
  leverage_control.csv
)
for f in "${CORE_CSVS[@]}"; do
  if [ -f "data/$f" ]; then cp "data/$f" "$OUT/data/"; else echo "WARN missing data/$f"; fi
done

# small relaxed-scan grids that back Figure S4 (4 files, ~80 kB). The bulk
# data/scans is excluded as regenerable, but these are tiny and back a figure.
if [ -d data/relaxed_scans ]; then
  mkdir -p "$OUT/data/relaxed_scans"
  cp data/relaxed_scans/*.npz "$OUT/data/relaxed_scans/"
fi

cp DEPOSIT_README.md MANIFEST_deposit.txt "$OUT/" 2>/dev/null || true

zip -rq "$OUT.zip" "$OUT"
echo "built $OUT.zip"
du -sh "$OUT.zip"
echo "files: $(find "$OUT" -type f | wc -l)"
