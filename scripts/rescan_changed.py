#!/usr/bin/env python3
"""Re-run the 2D scan (scan_all) and 1D scans (scan_1d) for only the structures
whose chromophore conformer changed under the occupancy-ranked altloc fix, and
patch the corresponding rows in scan_all_summary.csv / scan_1d_summary.csv.
Structures not in the changed set are provably identical and are left untouched.
Run from the project root.
"""
import csv, sys, time
sys.path.insert(0, "scripts")
import scan_all
import scan_1d

CHANGED = [l.strip().upper() for l in open("data/_altloc_changed_pdbs.txt") if l.strip()]


def patch(csv_path, process_one):
    rows = list(csv.DictReader(open(csv_path)))
    fields = rows[0].keys()
    by_id = {r["pdb_id"].upper(): r for r in rows}
    t0 = time.perf_counter()
    for i, pid in enumerate(CHANGED, 1):
        try:
            new = process_one(pid)
        except Exception as e:
            print(f"  FAIL {pid}: {e}")
            continue
        old = by_id.get(pid, {})
        by_id[pid] = {c: (new[c] if c in new else old.get(c, "")) for c in fields}
        if i % 10 == 0:
            print(f"  {i}/{len(CHANGED)} ({time.perf_counter()-t0:.0f}s)")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        # preserve original ordering, updated in place
        seen = set()
        for r in rows:
            pid = r["pdb_id"].upper()
            w.writerow(by_id[pid]); seen.add(pid)
    print(f"patched {csv_path} ({len(CHANGED)} structures)")


print(f"rescanning {len(CHANGED)} changed structures (2D)...")
patch("data/scan_all_summary.csv", scan_all.process_one)
print("rescanning 1D...")
patch("data/scan_1d_summary.csv", scan_1d.process_one)
print("done")
