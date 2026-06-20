"""
permutation_signtest.py
=======================

Scaffold-level permutation test for the matched-pair concordance statistic.

Background
----------
The matched-pair sign test (scripts/matched_pairs.py) reports a naive binomial
p-value, but the 886 pairs at >= 80% identity are drawn from only 141 unique
scaffolds, so they are NOT independent. To get a valid significance test we
permute the quantum-yield labels among the 141 unique FPs (the structural
metric values stay attached to their protein), re-derive the matched pairs
under the shuffled labels, and recompute the concordance fraction. The null
distribution of that fraction is what the observed value is tested against.

This corresponds to the Methods statement: "all inferential claims rest on the
scaffold-level permutation p-values (QY labels shuffled among unique FPs)."

Statistic
---------
For each metric: concordance fraction = n_concordant / n_untied, where a pair is
concordant if the brighter (higher-QY) member has the metric in the predicted
direction. Ties in the metric are excluded (NaN). Pairs are defined by sequence
identity >= MIN_IDENTITY (fixed across permutations -- it depends only on
sequence, not QY) AND |delta QY| >= MIN_DELTA_QY (re-evaluated each permutation,
since it depends on the shuffled QY labels).

p-value: one-sided, p = (#{frac_perm >= frac_obs} + 1) / (N_PERM + 1).

Outputs
-------
  data/matched_pairs_permutation.csv  -- one row per metric: observed fraction,
                                         null mean, permutation p, N_PERM

This script reuses build_dataset() and the aligner from matched_pairs.py so the
per-FP dataset and identity definition are identical to the sign-test pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matched_pairs as mp   # same directory; reuse dataset + aligner + constants

PROJECT = Path(__file__).resolve().parent.parent
DATA    = PROJECT / "data"

N_PERM = 5000        # matches Methods text
SEED   = 20260616


def precompute_identity(per: pd.DataFrame) -> np.ndarray:
    """Symmetric n x n matrix of global sequence identities (longer-seq norm)."""
    aligner = mp.make_aligner()
    seqs = per["seq"].fillna("").values
    n    = len(per)
    ident = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = mp.seq_identity(seqs[i], seqs[j], aligner)
            ident[i, j] = ident[j, i] = v
    return ident


def concordance_fractions(per: pd.DataFrame, ident: np.ndarray,
                          qy: np.ndarray, min_identity: float) -> dict:
    """
    Return {metric: (n_untied, n_concordant)} for one QY label vector.

    Pairs: identity >= min_identity AND |qy_i - qy_j| >= MIN_DELTA_QY.
    Concordant: (metric_bright - metric_dim) * direction > 0, ties excluded.
    """
    n = len(per)
    metric_vals = {m: per[col].values.astype(float)
                   for m, (col, _) in mp.METRICS.items()}
    dirs = {m: d for m, (_, d) in mp.METRICS.items()}

    counts = {m: [0, 0] for m in mp.METRICS}   # [n_untied, n_concordant]

    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        if ident[i, j] < min_identity:
            continue
        if abs(qy[i] - qy[j]) < mp.MIN_DELTA_QY:
            continue
        if qy[i] >= qy[j]:
            bi, di = i, j
        else:
            bi, di = j, i
        for m in mp.METRICS:
            vb, vd = metric_vals[m][bi], metric_vals[m][di]
            if np.isnan(vb) or np.isnan(vd) or vb == vd:
                continue
            counts[m][0] += 1
            if (vb - vd) * dirs[m] > 0:
                counts[m][1] += 1
    return counts


def main():
    per = mp.build_dataset()
    n = len(per)
    print(f"Per-unique-FP dataset: {n} proteins")

    print("Precomputing sequence-identity matrix ...", flush=True)
    ident = precompute_identity(per)

    qy_obs = per["canon_qy"].values.astype(float)

    # Observed statistic (sanity-check against matched_pairs_signtest.csv)
    obs = concordance_fractions(per, ident, qy_obs, mp.MIN_IDENTITY)
    print("\nObserved concordance (should match matched_pairs_signtest.csv >=80%):")
    for m, (nu, nc) in obs.items():
        print(f"  {m:<20} {nc}/{nu} = {nc/nu:.4f}")

    # Permutations
    rng = np.random.default_rng(SEED)
    perm_frac = {m: np.empty(N_PERM) for m in mp.METRICS}
    print(f"\nRunning {N_PERM} permutations (seed={SEED}) ...", flush=True)
    for k in range(N_PERM):
        qy_p = rng.permutation(qy_obs)
        c = concordance_fractions(per, ident, qy_p, mp.MIN_IDENTITY)
        for m, (nu, nc) in c.items():
            perm_frac[m][k] = nc / nu if nu > 0 else np.nan
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{N_PERM}", flush=True)

    rows = []
    print("\n=== Permutation results (identity >= 80%) ===")
    for m in mp.METRICS:
        nu, nc = obs[m]
        f_obs  = nc / nu
        null   = perm_frac[m]
        ge     = int(np.sum(null >= f_obs - 1e-12))
        p_perm = (ge + 1) / (N_PERM + 1)
        rows.append({
            "metric":         m,
            "n_untied":       nu,
            "n_concordant":   nc,
            "frac_observed":  f_obs,
            "null_mean":      float(np.nanmean(null)),
            "null_sd":        float(np.nanstd(null)),
            "n_perm_ge_obs":  ge,
            "n_perm":         N_PERM,
            "p_permutation":  p_perm,
        })
        print(f"  {m:<20} obs={f_obs:.4f}  null={np.nanmean(null):.4f}"
              f"+-{np.nanstd(null):.4f}  ge={ge}  p={p_perm:.4g}")

    out = DATA / "matched_pairs_permutation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
