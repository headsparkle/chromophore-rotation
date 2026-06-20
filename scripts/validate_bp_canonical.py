"""
validate_bp_canonical.py
========================

Validation of the canonical co-rotation coordinate BP_abs (= |sdev_tau + phi_canon|,
from canonical_cd_twist.py) against the published d_exp_to_planar, through the
paper's two generalization tests: LOOCV (S5) and matched pairs (S6).

Question: the raw red Spearman favours BP_abs (-0.56) over d_exp_to_planar (-0.36).
Does that advantage survive the confound-controlled tests, or is it between-scaffold
/ outlier driven?

Outputs:
  data/bp_validation_summary.csv

Verdict (see log 2026-06-16): NOT superior. The LOOCV CV-R2 gain is carried by ~5
very dim, very twisted reds (drop them and it vanishes); the within-scaffold
matched-pair concordance on red-red pairs is comparable-to-slightly-worse than
d_exp_to_planar. The raw red Spearman advantage is between-scaffold.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import loocv_validation as L
import matched_pairs as mp

DATA = Path(__file__).resolve().parent.parent / "data"


def bp_per_fp() -> pd.DataFrame:
    ps = pd.read_csv(DATA / "canonical_cd_twist.csv")
    qy = pd.read_csv(DATA / "lit_qy_curated.csv")
    qy["fp_id"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])
    m = ps.merge(qy[["pdb_id", "fp_id"]], on="pdb_id").dropna(subset=["fp_id"])
    return m.groupby("fp_id")["BP_abs"].median().rename("BP_abs").reset_index()


def loocv_part(rows: list):
    per = L.build_dataset().merge(bp_per_fp(), on="fp_id", how="left").dropna(subset=["BP_abs"])
    y = per["log_qy"].values
    specs = {
        "M_base": ["b_factor_ratio", "n_polar_O2_32"],
        "M_dexp": ["b_factor_ratio", "n_polar_O2_32", "d_exp_to_planar_deg"],
        "M_bp":   ["b_factor_ratio", "n_polar_O2_32", "BP_abs"],
        "M_both": ["b_factor_ratio", "n_polar_O2_32", "d_exp_to_planar_deg", "BP_abs"],
    }
    preds = {}
    for name, quant in specs.items():
        X = L.make_X(per, {"color": True, "quant": quant})
        preds[name] = L.loocv_one_model(X, y, 0.0)
        met = L.cv_metrics(y, preds[name])
        rows.append({"test": "LOOCV_full141", "model": name,
                     "cv_r2": met["cv_r2"], "rho": met["spearman_rho"],
                     "rmse": met["rmse"], "n": len(per)})

    # robustness: drop the k proteins whose M_dexp->M_bp residual drop is largest
    drop = np.argsort(np.abs(y - preds["M_dexp"]) - np.abs(y - preds["M_bp"]))[::-1]
    for k in (0, 5):
        keep = np.ones(len(per), bool); keep[drop[:k]] = False
        r2 = 1 - np.sum((y[keep] - preds["M_bp"][keep]) ** 2) / np.sum((y[keep] - y[keep].mean()) ** 2)
        rows.append({"test": f"LOOCV_M_bp_drop_top{k}", "model": "M_bp",
                     "cv_r2": r2, "rho": np.nan, "rmse": np.nan, "n": int(keep.sum())})

    red = per[per["color_class"] == "red"]
    for c in ("d_exp_to_planar_deg", "BP_abs"):
        rho, p = stats.spearmanr(red[c], red["log_qy"])
        rows.append({"test": "red_unique_spearman", "model": c,
                     "cv_r2": np.nan, "rho": rho, "rmse": p, "n": len(red)})


def matched_part(rows: list):
    per = mp.build_dataset().merge(bp_per_fp(), on="fp_id", how="left")
    mp.METRICS = {"d_exp_to_planar": ("d_exp", -1),
                  "BP_abs": ("BP_abs", -1),
                  "f_allowed_folded": ("f_allowed_folded", -1)}
    pairs = mp.find_pairs(per, mp.MIN_IDENTITY)
    red = pairs[(pairs.color_bright == "red") & (pairs.color_dim == "red")]
    for label, sub in (("matched_ALL", pairs), ("matched_RED_RED", red)):
        for met in mp.METRICS:
            c = sub[f"{met}_concordant"].dropna()
            nun, nc = len(c), int(c.sum())
            p = stats.binomtest(nc, nun, 0.5).pvalue if nun else np.nan
            rows.append({"test": label, "model": met, "cv_r2": np.nan,
                         "rho": nc / nun if nun else np.nan, "rmse": p, "n": nun})


def main():
    rows: list = []
    loocv_part(rows)
    matched_part(rows)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "bp_validation_summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {DATA / 'bp_validation_summary.csv'}")
    print("\nVerdict: BP_abs NOT superior to d_exp_to_planar under confound control "
          "(see matched_RED_RED + LOOCV drop-top5).")


if __name__ == "__main__":
    main()
