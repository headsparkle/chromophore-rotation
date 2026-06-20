#!/usr/bin/env python3
"""
compare_grid_density.py
=======================

Compare f_allowed_folded between the 5-degree (scan_all_summary.csv) and
2.5-degree (scan_25deg_summary.csv) grids across all structures where both
are available.  Prints the Spearman rho and a brief per-class breakdown
for the S8 robustness section.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parent.parent / "data"

s5  = pd.read_csv(DATA / "scan_all_summary.csv")
s25 = pd.read_csv(DATA / "scan_25deg_summary.csv")
mfa = pd.read_csv(DATA / "merged_for_aggregate.csv")[["pdb_id", "color_class"]]

s5  = s5[s5["status"] == "ok"][["pdb_id", "f_allowed_folded"]].rename(
    columns={"f_allowed_folded": "f5"})
s25 = s25[s25["status"] == "ok"][["pdb_id", "f_allowed_folded"]].rename(
    columns={"f_allowed_folded": "f25"})

df = s5.merge(s25, on="pdb_id").merge(mfa, on="pdb_id", how="left")
df = df.dropna(subset=["f5", "f25"])

rho, p = spearmanr(df["f5"], df["f25"])
print(f"Overall  n={len(df)}  rho={rho:.4f}  p={p:.3g}")

for cc in ["cyan", "green", "yellow", "red", "blue"]:
    sub = df[df["color_class"] == cc]
    if len(sub) < 5:
        continue
    r, pv = spearmanr(sub["f5"], sub["f25"])
    print(f"  {cc:<8s} n={len(sub):>3d}  rho={r:.4f}  p={pv:.3g}")

# Mean absolute difference in f_allowed_folded
df["abs_diff"] = (df["f5"] - df["f25"]).abs()
print(f"\nMean |f5 - f25|: {df['abs_diff'].mean():.6f}")
print(f"Max  |f5 - f25|: {df['abs_diff'].max():.6f}")
print(f"Structures where rank changes by > 50: "
      f"{((df['f5'].rank() - df['f25'].rank()).abs() > 50).sum()}")
