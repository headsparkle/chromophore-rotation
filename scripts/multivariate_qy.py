#!/usr/bin/env python3
"""
multivariate_qy.py
==================

Test whether f_allowed_folded adds explanatory power for the
literature quantum yield after controlling for the obvious
confounders we have already identified in the aggregation:

- color class            : controls for chromophore family chemistry
                            (green vs red vs blue vs ...).
- chrom_contacts         : direct geometric proxy for cage tightness.
- b_factor_ratio         : Luke's headline predictor; behaves as a
                            red-FP marker in our dataset.
- minor_axis             : barrel cross-sectional tightness.

Strategy
--------
Nested OLS models with increasing predictor sets:

    M0  log_qy ~ 1                                          (null)
    M1  log_qy ~ color_class                                (chemistry)
    M2  M1 + chrom_contacts + b_factor_ratio + minor_axis   (controls)
    M3  M2 + f_allowed_folded                               (target)

We report:
- R-squared, adjusted R-squared, AIC for each model.
- An F test of M3 vs M2 (does f_allowed add information?).
- The coefficient on f_allowed_folded in M3 with 95 % CI and p.
- A partial regression plot for f_allowed_folded.

We use **log10(lit_qy)** because QY is bounded on (0, 1] and
heavily right-skewed; log10 stabilises variance and gives an
interpretable coefficient (change in log10 QY per unit
f_allowed_folded). All quantitative predictors are standardised
(zero mean, unit variance) before fitting so coefficients can be
compared on a single scale.

We also run a green-only subset model because green dominates the
QY data (~200 of 290 rows) and the cross-color signal is the most
likely source of confounding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures" / "aggregate"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV = DATA_DIR / "merged_for_aggregate.csv"

TARGET_PREDICTOR = "f_allowed_folded"
CONTROL_PREDICTORS = ["chrom_contacts", "b_factor_ratio", "minor_axis"]
COLOR_COL = "color_class"
RESPONSE_RAW = "lit_qy"


def standardize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    needed = [RESPONSE_RAW, TARGET_PREDICTOR, *CONTROL_PREDICTORS, COLOR_COL]
    sub = df[needed].copy()
    sub = sub.dropna(subset=[RESPONSE_RAW, TARGET_PREDICTOR, *CONTROL_PREDICTORS])
    sub = sub[sub[RESPONSE_RAW] > 0]  # log requires positive
    sub["log_qy"] = np.log10(sub[RESPONSE_RAW])
    for col in [TARGET_PREDICTOR, *CONTROL_PREDICTORS]:
        sub[f"z_{col}"] = standardize(sub[col])
    # Color as categorical; reference = green (largest class)
    sub[COLOR_COL] = sub[COLOR_COL].fillna("unknown")
    sub[COLOR_COL] = pd.Categorical(
        sub[COLOR_COL],
        categories=["green", "blue", "cyan", "yellow", "orange", "red", "unknown"],
    )
    return sub


def fit_models(df: pd.DataFrame) -> dict[str, sm.regression.linear_model.RegressionResults]:
    z_target = f"z_{TARGET_PREDICTOR}"
    z_controls = [f"z_{c}" for c in CONTROL_PREDICTORS]
    formulas = {
        "M0": "log_qy ~ 1",
        "M1": f"log_qy ~ C({COLOR_COL})",
        "M2": f"log_qy ~ C({COLOR_COL}) + " + " + ".join(z_controls),
        "M3": f"log_qy ~ C({COLOR_COL}) + " + " + ".join(z_controls) + f" + {z_target}",
    }
    return {name: smf.ols(f, data=df).fit() for name, f in formulas.items()}


def model_table(models: dict) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        rows.append({
            "model": name,
            "n": int(m.nobs),
            "k_params": int(m.df_model),
            "r2": float(m.rsquared),
            "adj_r2": float(m.rsquared_adj),
            "aic": float(m.aic),
            "bic": float(m.bic),
        })
    return pd.DataFrame(rows)


def f_test_nested(restricted, full) -> tuple[float, float, int, int]:
    """Compare nested OLS via F test. Returns (F, p, df_diff, df_resid)."""
    rss_r = float(restricted.ssr)
    rss_f = float(full.ssr)
    df_diff = int(full.df_model - restricted.df_model)
    df_resid = int(full.df_resid)
    F = ((rss_r - rss_f) / df_diff) / (rss_f / df_resid)
    from scipy import stats as _st
    p = float(1 - _st.f.cdf(F, df_diff, df_resid))
    return F, p, df_diff, df_resid


def report_target(model, name: str) -> dict:
    z_target = f"z_{TARGET_PREDICTOR}"
    if z_target not in model.params.index:
        return {}
    coef = float(model.params[z_target])
    ci_lo, ci_hi = (float(x) for x in model.conf_int().loc[z_target])
    p = float(model.pvalues[z_target])
    return {
        "model": name,
        "coef_z_f_allowed": coef,
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "p_value": p,
    }


def plot_partial_regression(model, df: pd.DataFrame, out: Path) -> None:
    z_target = f"z_{TARGET_PREDICTOR}"
    # Residuals of log_qy ~ everything except f_allowed
    z_controls = [f"z_{c}" for c in CONTROL_PREDICTORS]
    formula_y = f"log_qy ~ C({COLOR_COL}) + " + " + ".join(z_controls)
    formula_x = f"{z_target} ~ C({COLOR_COL}) + " + " + ".join(z_controls)
    res_y = smf.ols(formula_y, data=df).fit().resid
    res_x = smf.ols(formula_x, data=df).fit().resid
    fig, ax = plt.subplots(figsize=(6.5, 5))
    palette = {
        "green": "#22aa33", "yellow": "#d4b400", "cyan": "#1e90c8",
        "blue": "#3050d0", "orange": "#e07020", "red": "#cc2030",
        "unknown": "#888888",
    }
    for c, sub_df in df.groupby(COLOR_COL, observed=True):
        idx = sub_df.index
        ax.scatter(
            res_x.loc[idx], res_y.loc[idx],
            s=22, alpha=0.7, color=palette.get(str(c), "#888"),
            edgecolors="none",
            label=f"{c} (n={len(idx)})",
        )
    # OLS line on the residuals
    slope, intercept = np.polyfit(res_x, res_y, 1)
    xs = np.linspace(res_x.min(), res_x.max(), 50)
    ax.plot(xs, intercept + slope * xs, color="black", linewidth=1.5,
            label=f"partial slope = {slope:+.3f}")
    ax.set_xlabel(f"residual {z_target} (after C, controls)")
    ax.set_ylabel("residual log10(QY) (after C, controls)")
    ax.set_title("Partial regression: log QY on f_allowed_folded\n"
                 "after color and chrom_contacts / b_factor_ratio / minor_axis")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    raw = pd.read_csv(MERGED_CSV)
    df = prepare(raw)
    print(f"[load] {len(df)} structures with QY and all predictors")
    print(f"  color breakdown: {dict(df[COLOR_COL].value_counts())}")

    models = fit_models(df)
    print("\n[nested OLS, response = log10(lit_qy)]")
    print(model_table(models).to_string(index=False))

    print("\n[F test: M3 vs M2 — does f_allowed_folded add anything?]")
    F, p, ddf, rdf = f_test_nested(models["M2"], models["M3"])
    print(f"  F({ddf}, {rdf}) = {F:.3f}   p = {p:.4f}")

    print("\n[F test: M2 vs M1 — does adding chrom_contacts, b_factor_ratio, minor_axis help?]")
    F2, p2, ddf2, rdf2 = f_test_nested(models["M1"], models["M2"])
    print(f"  F({ddf2}, {rdf2}) = {F2:.3f}   p = {p2:.4f}")

    print("\n[Coefficient on standardized f_allowed_folded in each model that includes it]")
    targets = [report_target(models["M3"], "M3 (full)")]
    pd.DataFrame(targets).to_string(index=False)
    print(pd.DataFrame(targets).to_string(index=False))

    print("\n[full M3 summary (coefficients, p-values, CIs)]")
    print(models["M3"].summary().tables[1])

    # Partial regression plot
    out_pr = FIG_DIR / "partial_regression_qy_on_f_allowed.png"
    plot_partial_regression(models["M3"], df, out_pr)
    print(f"\n[save] partial regression plot -> {out_pr}")

    # Green-only subset
    green = df[df[COLOR_COL] == "green"].copy()
    print(f"\n[green-only sensitivity check, n = {len(green)}]")
    if len(green) >= 30:
        z_controls = [f"z_{c}" for c in CONTROL_PREDICTORS]
        z_target = f"z_{TARGET_PREDICTOR}"
        m_g_ctrl = smf.ols("log_qy ~ " + " + ".join(z_controls), data=green).fit()
        m_g_full = smf.ols("log_qy ~ " + " + ".join(z_controls) + f" + {z_target}", data=green).fit()
        print(model_table({"green_controls": m_g_ctrl, "green_full": m_g_full}).to_string(index=False))
        Fg, pg, ddfg, rdfg = f_test_nested(m_g_ctrl, m_g_full)
        print(f"  green-only F test: F({ddfg}, {rdfg}) = {Fg:.3f}, p = {pg:.4f}")
        tg = report_target(m_g_full, "green-only M3")
        print(f"  green-only coef on z_{TARGET_PREDICTOR}: {tg['coef_z_f_allowed']:+.4f}"
              f"  (95% CI [{tg['ci_lo_95']:+.4f}, {tg['ci_hi_95']:+.4f}], p = {tg['p_value']:.4f})")

    # Save summary
    out_csv = DATA_DIR / "multivariate_qy_results.csv"
    model_table(models).to_csv(out_csv, index=False)
    print(f"\n[save] model table -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
