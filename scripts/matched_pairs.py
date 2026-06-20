"""
matched_pairs.py
================

Systematic matched-pair analysis across the 141 per-unique-FP dataset.

For each pair of FPs that share a close sequence scaffold (global identity
>= MIN_IDENTITY) but differ in QY by >= MIN_DELTA_QY, we ask whether the
brighter member has the structurally "correct" value for each of the five
QY metrics:

  d_exp_to_planar      : brighter should be LOWER  (closer to flat)
  b_factor_ratio       : brighter should be LOWER  (more rigid)
  n_polar_O2_32        : brighter should be HIGHER (more O2 contacts)
  f_allowed_P_folded   : brighter should be LOWER  (tighter P-bond cage)
  f_allowed_folded     : brighter should be LOWER  (tighter 2D cage)

A "concordant" pair is one where the brighter member has the predicted
direction for a given metric. We report:
  - n pairs, n concordant, concordance fraction
  - Two-sided binomial sign test (H0: concordance = 0.5)
  - Same analysis at a stricter identity threshold (MIN_IDENTITY_STRICT)

Sequence identity is computed by global pairwise alignment (BioPython
PairwiseAligner, match=1 mismatch=0 no gap penalties) normalised by the
length of the longer sequence. FPs whose sequences differ in length by
more than MAX_LEN_DIFF_FRAC are excluded (likely fusion proteins).

Outputs:
  data/matched_pairs.csv           -- all pairs that pass filters
  data/matched_pairs_signtest.csv  -- sign-test table, one row per metric
  figures/aggregate/matched_pairs_concordance.png
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from Bio.Align import PairwiseAligner

PROJECT = Path(__file__).resolve().parent.parent
DATA    = PROJECT / "data"
FIG_DIR = PROJECT / "figures" / "aggregate"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_IDENTITY        = 0.80   # primary threshold
MIN_IDENTITY_STRICT = 0.90   # sensitivity check
MIN_DELTA_QY        = 0.10   # minimum QY difference in the pair
MAX_LEN_DIFF_FRAC   = 0.20   # skip pairs where seqs differ by >20% length

# metric name -> (column, direction: +1 means brighter should be HIGHER, -1 means LOWER)
METRICS = {
    "d_exp_to_planar":    ("d_exp",              -1),
    "b_factor_ratio":     ("b_factor_ratio",     -1),
    "n_polar_O2_32":      ("n_polar_O2_32",      +1),
    "f_allowed_P_folded": ("f_allowed_P_folded", -1),
    "f_allowed_folded":   ("f_allowed_folded",   -1),
}


# ---------------------------------------------------------------------------
# Build per-unique-FP dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    df   = pd.read_csv(DATA / "merged_for_aggregate.csv")
    qydf = pd.read_csv(DATA / "lit_qy_curated.csv")
    hb   = pd.read_csv(DATA / "hb_contacts.csv")
    s1   = pd.read_csv(DATA / "scan_1d_summary.csv")
    seqs = pd.read_csv(DATA / "pdb_sequences.csv")

    qydf["canon_qy"] = qydf["lit_qy_fpbase"].fillna(qydf["lit_qy_fpbase_recovered"])
    qydf["fp_id"]    = qydf["fpbase_slug"].fillna(qydf["seq_match_slug"])

    df = df.merge(qydf[["pdb_id", "canon_qy", "fp_id", "fpbase_name"]], on="pdb_id", how="left")
    df = df.merge(hb[["pdb_id", "n_polar_O2_32"]], on="pdb_id", how="left")
    df = df.merge(s1[["pdb_id", "f_allowed_P_folded"]], on="pdb_id", how="left")
    df = df.merge(seqs, on="pdb_id", how="left")

    cols = [
        "fp_id", "fpbase_name", "canon_qy", "color_class", "match_name",
        "d_exp_to_planar_deg", "b_factor_ratio", "n_polar_O2_32",
        "f_allowed_P_folded", "f_allowed_folded", "seq",
    ]
    sub = df[cols].dropna(subset=["fp_id", "canon_qy"]).copy()
    sub = sub[sub["canon_qy"] > 0]

    per = sub.groupby("fp_id").agg(
        canon_qy           = ("canon_qy",            "first"),
        fpbase_name        = ("fpbase_name",          "first"),
        color_class        = ("color_class",          "first"),
        match_name         = ("match_name",           "first"),
        d_exp              = ("d_exp_to_planar_deg",  "median"),
        b_factor_ratio     = ("b_factor_ratio",        "median"),
        n_polar_O2_32      = ("n_polar_O2_32",         "median"),
        f_allowed_P_folded = ("f_allowed_P_folded",    "median"),
        f_allowed_folded   = ("f_allowed_folded",      "median"),
        seq                = ("seq",                   "first"),
    ).reset_index()
    per["color_class"] = per["color_class"].fillna("unknown").str.lower()
    return per


# ---------------------------------------------------------------------------
# Sequence identity
# ---------------------------------------------------------------------------

def make_aligner() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode            = "global"
    a.match_score     = 1
    a.mismatch_score  = 0
    a.open_gap_score  = 0
    a.extend_gap_score = 0
    return a


def seq_identity(s1: str, s2: str, aligner: PairwiseAligner) -> float:
    """Global identity normalised by the longer sequence."""
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    if abs(l1 - l2) / max(l1, l2) > MAX_LEN_DIFF_FRAC:
        return 0.0        # length-mismatch signals fusion protein -- skip
    score = aligner.score(s1, s2)
    return float(score) / max(l1, l2)


# ---------------------------------------------------------------------------
# Find matched pairs
# ---------------------------------------------------------------------------

def find_pairs(per: pd.DataFrame,
               min_identity: float) -> pd.DataFrame:
    aligner = make_aligner()
    ids   = per["fp_id"].values
    seqs  = per["seq"].fillna("").values
    n     = len(per)

    rows = []
    total_pairs = n * (n - 1) // 2
    checked = 0
    print(f"  Computing pairwise identities for {n} proteins "
          f"({total_pairs} pairs) ...", flush=True)

    for i, j in itertools.combinations(range(n), 2):
        ident = seq_identity(seqs[i], seqs[j], aligner)
        checked += 1
        if checked % 2000 == 0:
            print(f"    {checked}/{total_pairs} pairs checked ...", flush=True)
        if ident < min_identity:
            continue

        ri = per.iloc[i]
        rj = per.iloc[j]
        dqy = abs(ri["canon_qy"] - rj["canon_qy"])
        if dqy < MIN_DELTA_QY:
            continue

        # bright = higher QY
        if ri["canon_qy"] >= rj["canon_qy"]:
            bright, dim = ri, rj
        else:
            bright, dim = rj, ri

        row = {
            "fp_id_bright":  bright["fp_id"],
            "fp_id_dim":     dim["fp_id"],
            "name_bright":   bright.get("fpbase_name", bright["fp_id"]),
            "name_dim":      dim.get("fpbase_name", dim["fp_id"]),
            "color_bright":  bright["color_class"],
            "color_dim":     dim["color_class"],
            "qy_bright":     bright["canon_qy"],
            "qy_dim":        dim["canon_qy"],
            "delta_qy":      dqy,
            "seq_identity":  ident,
        }
        for metric, (col, direction) in METRICS.items():
            vb = bright[col]
            vd = dim[col]
            row[f"{metric}_bright"]    = vb
            row[f"{metric}_dim"]       = vd
            row[f"{metric}_delta"]     = vb - vd          # positive = bright higher
            if pd.isna(vb) or pd.isna(vd) or vb == vd:
                row[f"{metric}_concordant"] = np.nan      # ties and missing excluded
            else:
                # concordant: bright has the predicted direction relative to dim
                # direction=-1 (brighter=lower): concordant when vb < vd, i.e. (vb-vd)*direction > 0
                row[f"{metric}_concordant"] = int((vb - vd) * direction > 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Found {len(df)} pairs at identity >= {min_identity:.0%}")
    return df


# ---------------------------------------------------------------------------
# Sign tests
# ---------------------------------------------------------------------------

def sign_test_table(pairs: pd.DataFrame, threshold_label: str) -> pd.DataFrame:
    rows = []
    n_total = len(pairs)
    for metric, (col, direction) in METRICS.items():
        conc_col = f"{metric}_concordant"
        valid = pairs[conc_col].dropna()   # ties already NaN
        n_untied = len(valid)
        n_conc   = int(valid.sum())
        frac     = n_conc / n_untied if n_untied > 0 else np.nan
        result   = stats.binomtest(n_conc, n_untied, p=0.5, alternative="two-sided")
        rows.append({
            "threshold":      threshold_label,
            "metric":         metric,
            "direction":      "brighter = lower" if direction == -1 else "brighter = higher",
            "n_pairs_total":  n_total,
            "n_untied":       n_untied,
            "n_concordant":   n_conc,
            "frac_concordant":frac,
            "binom_p":        result.pvalue,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

COLOR_PAIR = {"concordant": "#2a7aca", "discordant": "#cc3333"}

METRIC_LABELS = {
    "d_exp_to_planar":    "d_exp\n(brighter = flatter)",
    "b_factor_ratio":     "B-factor ratio\n(brighter = more rigid)",
    "n_polar_O2_32":      "n_polar_O2\n(brighter = more contacts)",
    "f_allowed_P_folded": "f_allowed_P\n(brighter = tighter cage)",
    "f_allowed_folded":   "f_allowed (2D)\n(brighter = tighter cage)",
}


def plot_concordance(st80: pd.DataFrame, st90: pd.DataFrame, out: Path):
    metrics = list(METRICS.keys())
    n_met   = len(metrics)
    x80     = st80.set_index("metric")
    x90     = st90.set_index("metric")

    fig, axes = plt.subplots(1, n_met, figsize=(13, 4.5), sharey=True)
    fig.suptitle("Matched-pair concordance: does brighter = predicted direction?",
                 fontsize=11)

    for ax, met in zip(axes, metrics):
        r80 = x80.loc[met]
        r90 = x90.loc[met] if met in x90.index else None

        n80 = int(r80["n_untied"])
        c80 = int(r80["n_concordant"])
        d80 = n80 - c80
        p80 = r80["binom_p"]

        bars = ax.bar([0, 1], [c80, d80],
                      color=[COLOR_PAIR["concordant"], COLOR_PAIR["discordant"]],
                      alpha=0.85, width=0.6, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Concordant", "Discordant"], fontsize=8)
        ax.set_title(METRIC_LABELS.get(met, met), fontsize=8.5)
        ax.axhline(n80 / 2, color="k", lw=1.0, ls="--", zorder=2)

        # Annotate with numbers and p-value
        for bar, val in zip(bars, [c80, d80]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=9)

        p_str = f"p = {p80:.3f}" if p80 >= 0.001 else f"p = {p80:.2e}"
        frac  = r80["frac_concordant"]
        ax.text(0.5, 0.97, f"{frac:.0%} concordant\n{p_str}\n(n={n80})",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"))

        ax.set_ylim(0, max(c80, d80) * 1.35)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    axes[0].set_ylabel("Number of pairs (identity >= 80%)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per = build_dataset()
    print(f"Per-unique-FP dataset: {len(per)} proteins with QY")

    # Primary analysis: identity >= 80%
    print("\nPrimary analysis (identity >= 80%):")
    pairs80 = find_pairs(per, MIN_IDENTITY)

    # Strict sensitivity: identity >= 90%
    print("\nSensitivity (identity >= 90%):")
    pairs90 = find_pairs(per, MIN_IDENTITY_STRICT)

    # Sign tests
    st80 = sign_test_table(pairs80, ">=80%")
    st90 = sign_test_table(pairs90, ">=90%")
    combined = pd.concat([st80, st90], ignore_index=True)

    # Save
    pairs80.to_csv(DATA / "matched_pairs.csv", index=False)
    combined.to_csv(DATA / "matched_pairs_signtest.csv", index=False)
    print(f"\nSaved {DATA / 'matched_pairs.csv'} ({len(pairs80)} pairs)")
    print(f"Saved {DATA / 'matched_pairs_signtest.csv'}")

    # Print results
    print("\n=== Sign test results (identity >= 80%) ===")
    fmt = "{:<22} {:>8} {:>9} {:>13} {:>18} {:>12}"
    print(fmt.format("metric", "n_total", "n_untied", "n_concordant", "frac_concordant", "binom_p"))
    for _, row in st80.iterrows():
        print(fmt.format(
            row["metric"],
            int(row["n_pairs_total"]),
            int(row["n_untied"]),
            int(row["n_concordant"]),
            f"{row['frac_concordant']:.3f}",
            f"{row['binom_p']:.4g}",
        ))

    print("\n=== Sign test results (identity >= 90%) ===")
    for _, row in st90.iterrows():
        print(fmt.format(
            row["metric"],
            int(row["n_pairs_total"]),
            int(row["n_untied"]),
            int(row["n_concordant"]),
            f"{row['frac_concordant']:.3f}",
            f"{row['binom_p']:.4g}",
        ))

    # Figure
    plot_concordance(st80, st90, FIG_DIR / "matched_pairs_concordance.png")

    # Print a few exemplary pairs for the manuscript
    print("\n=== Top 10 exemplary pairs (identity >= 90%, largest delta_qy) ===")
    if len(pairs90) > 0:
        top = pairs90.nlargest(10, "delta_qy")[
            ["name_bright", "name_dim", "qy_bright", "qy_dim",
             "seq_identity", "d_exp_to_planar_concordant",
             "b_factor_ratio_concordant", "n_polar_O2_32_concordant",
             "f_allowed_P_folded_concordant"]
        ]
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
