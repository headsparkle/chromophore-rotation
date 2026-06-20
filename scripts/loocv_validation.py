"""
loocv_validation.py
===================

Leave-one-protein-out cross-validation (LOOCV) for quantum yield prediction.
Dataset: one row per unique FP (147 max; drops to ~141 after QY filtering).
Response: log10(canon_qy).

Models compared:
  M_color   : color_class dummies only
  M_base    : color + b_factor_ratio + n_polar_O2_32
  M_full    : M_base + d_exp_to_planar_deg   <-- focal comparison
  M_all5    : M_full + f_allowed_P_folded    <-- all five metrics

For each model:
  - LOOCV predicted log10(QY) for every held-out protein
  - CV R-squared (1 - SS_res / SS_tot on held-out predictions)
  - Spearman rho (predicted vs. observed log10 QY, held-out)
  - RMSE on held-out set (in log10 QY units)

Also:
  - Paired comparison of |error_full| vs |error_base| (Wilcoxon signed-rank)
  - A scatter plot: predicted vs. observed log10 QY for M_full

Outputs:
  data/loocv_results.csv       -- per-model LOOCV metrics
  data/loocv_predictions.csv   -- per-protein held-out predictions for all models
  figures/aggregate/loocv_predicted_vs_observed.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
DATA    = PROJECT / "data"
FIG_DIR = PROJECT / "figures" / "aggregate"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLOR_PALETTE = {
    "green": "#22aa33", "yellow": "#d4b400", "cyan": "#1e90c8",
    "blue": "#3050d0",  "orange": "#e07020", "red":  "#cc2030",
    "unknown": "#888888",
}
COLOR_ORDER = ["blue", "cyan", "green", "yellow", "orange", "red", "unknown"]


# ---------------------------------------------------------------------------
# Build per-unique-FP dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    df  = pd.read_csv(DATA / "merged_for_aggregate.csv")
    qy  = pd.read_csv(DATA / "lit_qy_curated.csv")
    hb  = pd.read_csv(DATA / "hb_contacts.csv")
    s1  = pd.read_csv(DATA / "scan_1d_summary.csv")

    qy["canon_qy"] = qy["lit_qy_fpbase"].fillna(qy["lit_qy_fpbase_recovered"])
    qy["fp_id"]    = qy["fpbase_slug"].fillna(qy["seq_match_slug"])

    df = df.merge(qy[["pdb_id", "canon_qy", "fp_id"]], on="pdb_id", how="left")
    df = df.merge(hb[["pdb_id", "n_polar_O2_32", "oh_found"]], on="pdb_id", how="left")
    df = df.merge(s1[["pdb_id", "f_allowed_I", "f_allowed_P_folded"]], on="pdb_id", how="left")

    cols = [
        "fp_id", "canon_qy", "color_class",
        "f_allowed_folded", "d_exp_to_planar_deg",
        "b_factor_ratio", "n_polar_O2_32",
        "f_allowed_I", "f_allowed_P_folded",
        "chrom_contacts", "minor_axis",
    ]
    sub = df[cols].dropna(subset=["fp_id", "canon_qy"]).copy()
    sub = sub[sub["canon_qy"] > 0]

    grp = sub.groupby("fp_id")
    per = grp.agg(
        canon_qy           = ("canon_qy",           "first"),
        color_class        = ("color_class",         "first"),
        f_allowed_folded   = ("f_allowed_folded",    "median"),
        d_exp_to_planar_deg= ("d_exp_to_planar_deg", "median"),
        b_factor_ratio     = ("b_factor_ratio",      "median"),
        n_polar_O2_32      = ("n_polar_O2_32",       "median"),
        f_allowed_I        = ("f_allowed_I",          "median"),
        f_allowed_P_folded = ("f_allowed_P_folded",  "median"),
        chrom_contacts     = ("chrom_contacts",       "median"),
        minor_axis         = ("minor_axis",           "median"),
    ).reset_index()

    per["log_qy"] = np.log10(per["canon_qy"])
    per["color_class"] = per["color_class"].fillna("unknown").str.lower()
    return per


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

QUANT_COLS = ["b_factor_ratio", "n_polar_O2_32", "d_exp_to_planar_deg",
              "f_allowed_P_folded"]

MODEL_SPECS = {
    "M_color": {
        "color": True,
        "quant": [],
    },
    "M_base": {
        "color": True,
        "quant": ["b_factor_ratio", "n_polar_O2_32"],
    },
    "M_full": {
        "color": True,
        "quant": ["b_factor_ratio", "n_polar_O2_32", "d_exp_to_planar_deg"],
    },
    "M_all5": {
        "color": True,
        "quant": ["b_factor_ratio", "n_polar_O2_32", "d_exp_to_planar_deg",
                  "f_allowed_P_folded"],
    },
}


def make_X(df: pd.DataFrame, spec: dict) -> np.ndarray:
    parts = []
    if spec["color"]:
        dummies = pd.get_dummies(
            df["color_class"].astype(pd.CategoricalDtype(
                categories=COLOR_ORDER, ordered=False)),
            drop_first=True,   # green is reference (largest class)
        ).astype(float)
        parts.append(dummies.values)
    if spec["quant"]:
        parts.append(df[spec["quant"]].values.astype(float))
    if not parts:
        return np.ones((len(df), 1))
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------------------

def loocv_one_model(X: np.ndarray, y: np.ndarray, alpha: float = 0.0
                    ) -> np.ndarray:
    """Return per-sample LOOCV predictions. alpha=0 => OLS."""
    loo  = LeaveOneOut()
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in loo.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te       = X[test_idx]
        # standardise quant cols (refit scaler on training fold only)
        # For simplicity: Ridge(alpha) handles regularisation; scaler applied.
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        if alpha == 0.0:
            m = LinearRegression().fit(X_tr_s, y_tr)
        else:
            m = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        pred[test_idx] = m.predict(X_te_s)
    return pred


def cv_metrics(y_obs: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    rho, p_rho = stats.spearmanr(y_obs, y_pred)
    rmse   = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    return {"cv_r2": r2, "spearman_rho": rho, "spearman_p": p_rho, "rmse": rmse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per = build_dataset()
    print(f"Per-unique-FP dataset: {len(per)} proteins")
    print(per["color_class"].value_counts().to_string())

    y = per["log_qy"].values

    results   = []
    all_preds = per[["fp_id", "canon_qy", "log_qy", "color_class"]].copy()

    for name, spec in MODEL_SPECS.items():
        X    = make_X(per, spec)
        pred = loocv_one_model(X, y, alpha=0.0)
        met  = cv_metrics(y, pred)
        met["model"]   = name
        met["n"]       = len(per)
        met["n_params"] = X.shape[1]
        results.append(met)
        all_preds[f"pred_{name}"] = pred
        print(f"\n{name}: CV R2={met['cv_r2']:.3f}  rho={met['spearman_rho']:.3f}"
              f"  p={met['spearman_p']:.4g}  RMSE={met['rmse']:.3f}")

    # Paired Wilcoxon: |err_base| vs |err_full|
    err_base = np.abs(y - all_preds["pred_M_base"].values)
    err_full = np.abs(y - all_preds["pred_M_full"].values)
    stat_wx, p_wx = stats.wilcoxon(err_base, err_full, alternative="greater")
    print(f"\nWilcoxon |err_base| > |err_full|: W={stat_wx:.1f}  p={p_wx:.4g}")

    # Save CSV outputs
    res_df = pd.DataFrame(results)[
        ["model", "n", "n_params", "cv_r2", "spearman_rho", "spearman_p", "rmse"]
    ]
    res_df.to_csv(DATA / "loocv_results.csv", index=False)
    all_preds.to_csv(DATA / "loocv_predictions.csv", index=False)
    print(f"\nSaved {DATA / 'loocv_results.csv'}")
    print(f"Saved {DATA / 'loocv_predictions.csv'}")

    # ---- Figure: predicted vs observed for M_full, colour-coded by class ----
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for cls in COLOR_ORDER:
        mask = per["color_class"] == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            y[mask],
            all_preds["pred_M_full"].values[mask],
            color=COLOR_PALETTE.get(cls, "#888888"),
            label=cls, s=40, alpha=0.82, zorder=3,
        )
    lo = min(y.min(), all_preds["pred_M_full"].values.min()) - 0.05
    hi = max(y.max(), all_preds["pred_M_full"].values.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=1)

    m_met = [r for r in results if r["model"] == "M_full"][0]
    ax.text(0.04, 0.96,
            f"CV $R^2$ = {m_met['cv_r2']:.3f}\n"
            f"Spearman $\\rho$ = {m_met['spearman_rho']:.3f}\n"
            f"RMSE = {m_met['rmse']:.3f} (log$_{{10}}$ units)",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_xlabel("Observed log$_{10}$(QY)")
    ax.set_ylabel("LOOCV predicted log$_{10}$(QY)")
    ax.set_title("Leave-one-protein-out cross-validation (M_full)")
    ax.legend(title="Color class", fontsize=8, title_fontsize=8,
              loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_fig = FIG_DIR / "loocv_predicted_vs_observed.png"
    fig.savefig(out_fig, dpi=200)
    print(f"Saved {out_fig}")

    # Print summary table
    print("\nModel comparison (LOOCV, 141 unique FPs, OLS, log10 QY):")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWilcoxon test (M_base vs M_full absolute errors): W={stat_wx:.1f}, p={p_wx:.4g}")


if __name__ == "__main__":
    main()
