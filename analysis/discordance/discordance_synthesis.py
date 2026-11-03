#!/usr/bin/env python3
"""Computes HbA1c-CGM window agreement, GMI/eA1c conversion sensitivity, and
adjusted marker associations with HbA1c-GMI discordance.

Primary inference uses no-diabetes-range participants, comparing laboratory
HbA1c against glucose-management-indicator (GMI) estimates computed from CGM
mean glucose (Bergenstal et al. 2018). Full-range conversion fits are exported
separately as explicitly labeled all-lab sensitivities, because comparison with
the published GMI/ADAG regression direction (Nathan et al. 2008) requires that
range. All biomarkers come from the same canonical panel.

Inputs:
  - OUTPUTS/canonical_panel/... canonical lab-CGM panel tables (symmetric and
    preceding windows), produced by analysis/canonical_panel/build_panel.py.

Outputs (OUTPUTS/discordance/hba1c_cgm_discordance_synthesis/):
  - primary_symmetric_90d_pairs.parquet, primary_preceding_90d_pairs.parquet,
    first_pairs_all_windows.parquet, full_range_symmetric_90d_pairs.parquet
    (participant-level; private)
  - window_agreement_summary.csv, common_user_window_sensitivity.csv,
    conversion_formula_sensitivity.csv, cgm_metric_correlations.csv,
    incremental_cgm_metrics.csv, adjusted_marker_associations.csv,
    discordance_group_profiles.csv,
    conversion_formula_sensitivity_all_lab_linked.csv,
    window_agreement_summary_all_lab_linked.csv (aggregate summaries)
  - REPORT_HBA1C_CGM_DISCORDANCE_SYNTHESIS.md, discordance_synthesis.metadata.json

Usage: python analysis/discordance/discordance_synthesis.py
  [--panel PATH] [--preceding-panel PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from cgmpaper.publication_panels import DEFAULT_PANEL, DEFAULT_PRECEDING, PUBLICATION, load_panel, write_analysis_metadata

OUT = PUBLICATION / 'discordance/hba1c_cgm_discordance_synthesis'
GMI_INTERCEPT = 3.31
GMI_SLOPE = .02392
MIN_SENSOR_DAYS = 14
PANEL_TABLES = {"symmetric_90d": DEFAULT_PANEL, "preceding_90d": DEFAULT_PRECEDING}
PRIMARY_WINDOW, SENSITIVITY_WINDOW = "symmetric_90d", "preceding_90d"
WINDOWS = {"symmetric_90d": "Symmetric ±90 days", "preceding_90d": "Preceding −89…0 days"}
LAB_TYPES = ["fasting_glucose", "fasting_insulin", "triglycerides", "hdl_cholesterol", "apo_b", "hscrp", "uric_acid", "alt", "ast"]
LAB_LABELS = {name: name.replace("_", " ") for name in LAB_TYPES + ["homa_ir", "bmi", "age_bucket"]}


def load_panel_pairs(tables=None, cohort="no_diabetes_range"):
    frames = []
    for name, path in (tables or PANEL_TABLES).items():
        expected = "symmetric" if name == PRIMARY_WINDOW else "preceding"
        can = load_panel(path, cohort, required=["anchor_date", "n_readings", "sd_glucose", "tir_70_140", "tbr_70", "tar_180", "tar_140"], expected_window=expected)
        data = can[["USER_ID", "anchor_date", "a1c", "n_dense_days", "n_readings", "mean_glucose", "sd_glucose", "tir_70_140", "tbr_70", "tar_180", "tar_140"]].rename(
            columns={"anchor_date": "a1c_date", "n_dense_days": "sensor_days", "sd_glucose": "mean_daily_sd", "tir_70_140": "titr_70_140"})
        data["a1c_date"] = pd.to_datetime(data.a1c_date)
        data["window"] = name
        data["analysis_cohort"] = cohort
        for column in data.columns.difference(["USER_ID", "a1c_date", "window", "analysis_cohort"]):
            data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frames.append(data)
    pairs = pd.concat(frames, ignore_index=True)
    pairs["gmi"] = gmi(pairs.mean_glucose)
    pairs["a1c_minus_gmi"] = pairs.a1c - pairs.gmi
    pairs["abs_a1c_minus_gmi"] = pairs.a1c_minus_gmi.abs()
    return pairs


def load_covariates(primary, path=DEFAULT_PANEL, cohort="no_diabetes_range"):
    can = load_panel(path, cohort, required=["sex", "age_bucket", "bmi", "homa_ir", *LAB_TYPES])
    cols = ["USER_ID", "sex", "age_bucket", "bmi", "homa_ir", *LAB_TYPES]
    return primary.merge(can[cols], on="USER_ID", how="left", validate="one_to_one")


def gmi(mean_glucose: pd.Series) -> pd.Series:
    return GMI_INTERCEPT + GMI_SLOPE * mean_glucose


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return pd.Series(out, index=p_values.index)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[np.where(valid)[0]] = restored
    return pd.Series(out, index=p_values.index)


def agreement_rows(df: pd.DataFrame, window: str, cohort: str) -> dict:
    x = df["a1c"].to_numpy(float)
    y = df["gmi"].to_numpy(float)
    delta = x - y
    r, p = stats.pearsonr(x, y) if len(df) >= 3 and np.std(x) > 0 and np.std(y) > 0 else (np.nan, np.nan)
    return {
        "window": window,
        "cohort": cohort,
        "n": len(df),
        "sensor_days_median": df["sensor_days"].median(),
        "pearson_r": r,
        "pearson_p": p,
        "mean_a1c_minus_gmi": np.mean(delta),
        "sd_a1c_minus_gmi": np.std(delta, ddof=1) if len(delta) > 1 else np.nan,
        "median_a1c_minus_gmi": np.median(delta),
        "mae_a1c_minus_gmi": np.mean(np.abs(delta)),
        "pct_abs_ge_0_3": 100 * np.mean(np.abs(delta) >= 0.3),
        "pct_abs_ge_0_5": 100 * np.mean(np.abs(delta) >= 0.5),
        "pct_abs_ge_0_8": 100 * np.mean(np.abs(delta) >= 0.8),
        "pct_a1c_higher_ge_0_5": 100 * np.mean(delta >= 0.5),
        "pct_gmi_higher_ge_0_5": 100 * np.mean(delta <= -0.5),
    }


def summarize_agreement(first_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, d in first_pairs.groupby("window"):
        cohorts = {
            "all": d,
            "a1c_normal_lt_5_7": d[d["a1c"] < 5.7],
            "a1c_elevated_ge_5_7": d[d["a1c"] >= 5.7],
        }
        for cohort, sub in cohorts.items():
            rows.append(agreement_rows(sub, window, cohort))
    return pd.DataFrame(rows)


def common_user_window_sensitivity(first_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Users present in BOTH panel tables. Returns the compact summary and
    the full agreement rows (window suffixed `_common_users`, three cohorts), so
    the two window constructions can be compared on identical people."""
    counts = first_pairs.groupby("USER_ID")["window"].nunique()
    common_ids = set(counts[counts == len(PANEL_TABLES)].index)
    common = first_pairs[first_pairs["USER_ID"].isin(common_ids)]
    same_anchor = (
        common.pivot(index="USER_ID", columns="window", values="a1c_date")
        .apply(lambda r: r.nunique() == 1, axis=1).mean()
    )
    rows, full = [], []
    for window in WINDOWS:
        d = common[common["window"] == window]
        a1c, gmi_values = d["a1c"].to_numpy(float), d["gmi"].to_numpy(float)
        rows.append(
            {
                "window": window,
                "n_common_users": len(d),
                "pct_same_anchor_date": 100 * float(same_anchor),
                "pearson_r": stats.pearsonr(a1c, gmi_values).statistic if len(d) >= 3 and np.std(a1c) > 0 and np.std(gmi_values) > 0 else np.nan,
                "mean_a1c_minus_gmi": (a1c - gmi_values).mean(),
                "sd_a1c_minus_gmi": (a1c - gmi_values).std(ddof=1),
                "pct_abs_ge_0_5": 100 * (np.abs(a1c - gmi_values) >= 0.5).mean(),
            }
        )
        for cohort, sub in {
            "all": d,
            "a1c_normal_lt_5_7": d[d["a1c"] < 5.7],
            "a1c_elevated_ge_5_7": d[d["a1c"] >= 5.7],
        }.items():
            full.append(agreement_rows(sub, f"{window}_common_users", cohort))
    return pd.DataFrame(rows), pd.DataFrame(full)


def calibration_sensitivity(primary: pd.DataFrame) -> pd.DataFrame:
    """Compare conversion-equation bias with cohort-calibrated residual scatter."""
    d = primary.copy()
    d["adag_ea1c"] = (d["mean_glucose"] + 46.7) / 28.7
    x = sm.add_constant(d[["mean_glucose"]], has_constant="add")
    fit = sm.OLS(d["a1c"], x).fit()
    d["internal_calibrated_a1c"] = fit.predict(x)
    rows = []
    for method, estimate in {
        "GMI_Bergenstal_2018": d["gmi"],
        "eA1c_ADAG_2008": d["adag_ea1c"],
        "internal_linear_calibration": d["internal_calibrated_a1c"],
    }.items():
        delta = d["a1c"] - estimate
        rows.append(
            {
                "conversion": method,
                "fit_scope": "in_sample_descriptive_calibration" if method == "internal_linear_calibration" else "fixed_published_formula_evaluation",
                "n": len(d),
                "mean_a1c_minus_estimate": delta.mean(),
                "sd_a1c_minus_estimate": delta.std(ddof=1),
                "mae": delta.abs().mean(),
                "pct_abs_ge_0_5": 100 * (delta.abs() >= 0.5).mean(),
                "estimate_intercept": (
                    GMI_INTERCEPT if method.startswith("GMI")
                    else (46.7 / 28.7 if method.startswith("eA1c") else fit.params["const"])
                ),
                "estimate_slope_per_mgdl": (
                    GMI_SLOPE if method.startswith("GMI")
                    else (1 / 28.7 if method.startswith("eA1c") else fit.params["mean_glucose"])
                ),
            }
        )
    return pd.DataFrame(rows)


def incremental_cgm_metrics(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether dynamic CGM features add HbA1c information beyond mean glucose."""
    metrics = ["mean_glucose", "mean_daily_sd", "tar_140", "tar_180", "titr_70_140", "tbr_70"]
    corr_rows = []
    for metric in metrics:
        sub = primary[["a1c", metric]].dropna()
        pearson = stats.pearsonr(sub["a1c"], sub[metric])
        spearman = stats.spearmanr(sub["a1c"], sub[metric])
        corr_rows.append(
            {
                "metric": metric,
                "n": len(sub),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )

    model_cols = ["a1c", "mean_glucose", "mean_daily_sd", "tar_140", "tbr_70"]
    d = primary[model_cols].dropna().copy()
    d["mean_glucose_sq"] = np.square(d["mean_glucose"])
    base_cols = ["mean_glucose", "mean_glucose_sq"]
    full_cols = base_cols + ["mean_daily_sd", "tar_140", "tbr_70"]
    base_fit = sm.OLS(d["a1c"], sm.add_constant(d[base_cols], has_constant="add")).fit()
    full_fit = sm.OLS(d["a1c"], sm.add_constant(d[full_cols], has_constant="add")).fit()
    f_stat, p_value, df_diff = full_fit.compare_f_test(base_fit)

    cv = KFold(n_splits=min(10, len(d)), shuffle=True, random_state=20260710)
    pred_base = cross_val_predict(LinearRegression(), d[base_cols], d["a1c"], cv=cv)
    pred_full = cross_val_predict(LinearRegression(), d[full_cols], d["a1c"], cv=cv)
    tss = np.square(d["a1c"] - d["a1c"].mean()).sum()
    cv_r2_base = 1 - np.square(d["a1c"] - pred_base).sum() / tss
    cv_r2_full = 1 - np.square(d["a1c"] - pred_full).sum() / tss
    nested = pd.DataFrame(
        [
            {
                "n": len(d),
                "base_features": "+".join(base_cols),
                "added_features": "mean_daily_sd+tar_140+tbr_70",
                "r2_base": base_fit.rsquared,
                "r2_full": full_fit.rsquared,
                "delta_r2": full_fit.rsquared - base_fit.rsquared,
                "partial_f": f_stat,
                "partial_f_df": df_diff,
                "partial_f_p": p_value,
                "cv_r2_base": cv_r2_base,
                "cv_r2_full": cv_r2_full,
                "delta_cv_r2": cv_r2_full - cv_r2_base,
            }
        ]
    )
    return pd.DataFrame(corr_rows), nested


def adjusted_associations(data: pd.DataFrame) -> pd.DataFrame:
    candidates = ["bmi", "age_bucket"] + LAB_TYPES + ["homa_ir"]
    rows = []
    for candidate in candidates:
        if candidate not in data.columns:
            continue
        adjustment = ["gmi", "age_bucket", "bmi", "sex"]
        adjustment = [x for x in adjustment if x != candidate]
        cols = ["a1c_minus_gmi", candidate] + adjustment
        sub = data[cols].copy()
        for col in ["a1c_minus_gmi", candidate] + [x for x in adjustment if x != "sex"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 75 or sub[candidate].nunique() < 4:
            continue
        # Winsorize laboratory predictors to reduce single-value leverage.
        lo, hi = sub[candidate].quantile([0.01, 0.99])
        sub[candidate] = sub[candidate].clip(lo, hi)
        candidate_sd = sub[candidate].std(ddof=1)
        if not np.isfinite(candidate_sd) or candidate_sd == 0:
            continue
        sub["candidate_z"] = (sub[candidate] - sub[candidate].mean()) / candidate_sd
        x_parts = [sub[["candidate_z"]]]
        for covariate in adjustment:
            if covariate == "sex":
                x_parts.append(pd.get_dummies(sub[["sex"]], drop_first=True, dtype=float))
            else:
                values = sub[covariate].astype(float)
                sd = values.std(ddof=1)
                if sd > 0:
                    x_parts.append(((values - values.mean()) / sd).rename(covariate).to_frame())
        x = sm.add_constant(pd.concat(x_parts, axis=1), has_constant="add").astype(float)
        fit = sm.OLS(sub["a1c_minus_gmi"].astype(float), x).fit(cov_type="HC3")
        beta = fit.params["candidate_z"]
        se = fit.bse["candidate_z"]
        rows.append(
            {
                "marker": candidate,
                "label": LAB_LABELS.get(candidate, candidate),
                "n": len(sub),
                "beta_a1c_pct_per_marker_sd": beta,
                "ci_low": beta - 1.96 * se,
                "ci_high": beta + 1.96 * se,
                "p_value": fit.pvalues["candidate_z"],
                "model_r2": fit.rsquared,
                "marker_sd": candidate_sd,
                "adjusted_for": "+".join(adjustment),
            }
        )
    out = pd.DataFrame(rows, columns=["marker", "label", "n", "beta_a1c_pct_per_marker_sd", "ci_low", "ci_high", "p_value", "model_r2", "marker_sd", "adjusted_for"])
    out["q_value_bh"] = bh_adjust(out["p_value"])
    return out.sort_values(["q_value_bh", "p_value"]).reset_index(drop=True)


def discordance_profiles(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["discordance_group"] = np.select(
        [d["a1c_minus_gmi"] >= 0.5, d["a1c_minus_gmi"] <= -0.5],
        ["A1c higher by >=0.5", "GMI higher by >=0.5"],
        default="Within 0.5",
    )
    markers = [
        "a1c",
        "gmi",
        "mean_glucose",
        "a1c_minus_gmi",
        "age_bucket",
        "bmi",
        "fasting_glucose",
        "fasting_insulin",
        "homa_ir",
        "triglycerides",
        "hdl_cholesterol",
        "apo_b",
        "hemoglobin",
        "mcv",
        "rdw",
        "ferritin",
    ]
    markers = [x for x in markers if x in d.columns]
    rows = []
    for group, sub in d.groupby("discordance_group"):
        row = {"discordance_group": group, "n": len(sub), "pct": 100 * len(sub) / len(d)}
        for marker in markers:
            row[f"{marker}_n"] = sub[marker].notna().sum()
            row[f"{marker}_median"] = sub[marker].median()
        rows.append(row)
    order = ["GMI higher by >=0.5", "Within 0.5", "A1c higher by >=0.5"]
    return pd.DataFrame(rows).set_index("discordance_group").reindex(order).reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--preceding-panel", type=Path, default=DEFAULT_PRECEDING)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    tables = {PRIMARY_WINDOW: args.panel, SENSITIVITY_WINDOW: args.preceding_panel}
    pairs = load_panel_pairs(tables)
    primary = pairs.loc[pairs.window.eq(PRIMARY_WINDOW)].copy()
    sensitivity = pairs.loc[pairs.window.eq(SENSITIVITY_WINDOW)].copy()
    common, common_full = common_user_window_sensitivity(pairs)
    agreement = pd.concat([summarize_agreement(pairs), common_full], ignore_index=True)
    agreement["analysis_cohort"] = "no_diabetes_range"
    calibration = calibration_sensitivity(primary)
    calibration["analysis_cohort"] = "no_diabetes_range"
    correlations, incremental = incremental_cgm_metrics(primary)
    data = load_covariates(primary, args.panel)
    associations, profiles = adjusted_associations(data), discordance_profiles(data)
    full_pairs = load_panel_pairs(tables, "all_lab_linked")
    full_primary = full_pairs.loc[full_pairs.window.eq(PRIMARY_WINDOW)].copy()
    full_calibration = calibration_sensitivity(full_primary)
    full_calibration["analysis_cohort"] = "all_lab_linked_full_range_sensitivity"
    full_agreement = summarize_agreement(full_pairs)
    full_agreement["analysis_cohort"] = "all_lab_linked_full_range_sensitivity"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in [("primary_symmetric_90d_pairs.parquet", primary),
                        ("primary_preceding_90d_pairs.parquet", sensitivity),
                        ("first_pairs_all_windows.parquet", pairs),
                        ("full_range_symmetric_90d_pairs.parquet", full_primary)]:
        path = args.output_dir / name
        frame.to_parquet(path, index=False)
        outputs.append(path)
    for name, frame in [("window_agreement_summary.csv", agreement),
                        ("common_user_window_sensitivity.csv", common),
                        ("conversion_formula_sensitivity.csv", calibration),
                        ("cgm_metric_correlations.csv", correlations),
                        ("incremental_cgm_metrics.csv", incremental),
                        ("adjusted_marker_associations.csv", associations),
                        ("discordance_group_profiles.csv", profiles),
                        ("conversion_formula_sensitivity_all_lab_linked.csv", full_calibration),
                        ("window_agreement_summary_all_lab_linked.csv", full_agreement)]:
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    report = args.output_dir / "REPORT_HBA1C_CGM_DISCORDANCE_SYNTHESIS.md"
    lines = ["# Corrected canonical HbA1c–CGM synthesis", "",
             f"Primary no-diabetes-range populations: symmetric n={len(primary):,}; preceding n={len(sensitivity):,}. Full symmetric panel sensitivity: n={len(full_primary):,}.", "",
             "The window-agreement CSV retains `cohort=all` as the all-participants-within-selected-population row; `analysis_cohort` explicitly identifies no_diabetes_range. Common-user rows compare identical participants; their anchors may differ under independently selected windows.", "",
             "The internal calibration regresses HbA1c on glucose. GMI uses that direction; the ADAG eA1c conversion algebraically inverts the published glucose-on-HbA1c equation. The internally fitted line is an in-sample calibration: its zero mean residual is algebraic and is not external validation or a biological noise floor. Full-range all-lab fits are separate sensitivity files for comparison with published conversion equations. They must not be mixed with the no-diabetes primary segmented curve.", "",
             "| Population / window | N | Pearson r | Mean HbA1c − GMI | SD | Absolute difference ≥0.5 (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in agreement.loc[agreement.cohort.eq("all")].itertuples():
        lines.append(f"| {r.window} | {r.n:,} | {r.pearson_r:.3f} | {r.mean_a1c_minus_gmi:.3f} | {r.sd_a1c_minus_gmi:.3f} | {r.pct_abs_ge_0_5:.1f} |")
    lines += ["", "All marker associations use canonical anchor-band values; HOMA is the canonical same-day laboratory pair, never independently selected constituent dates. Associations use HC3 intervals and the documented 1st/99th-percentile descriptive predictor clipping. Nested CGM models share complete cases and folds; OOF R² uses pooled held-out errors against the full outcome mean.", "",
              "Sensor-day repeat stability and unnormalized CBC/iron/renal ±90-day queries are not part of this analysis. No raw database access is required."]
    report.write_text("\n".join(lines)+"\n")
    outputs.append(report)
    write_analysis_metadata(args.output_dir / "discordance_synthesis.metadata.json", script=__file__,
                            panels=tables, outputs=outputs, details={
                                "primary_cohort": "no_diabetes_range", "full_range_sensitivity": "all_lab_linked",
                                "participants_symmetric": len(primary), "participants_preceding": len(sensitivity),
                                "participants_full_symmetric": len(full_primary),
                                "repeat_and_unnormalized_labs_included": False,
                                "mean_daily_sd_alias": "Canonical pooled window SD, not mean of daily SD",
                                "internal_calibration": "in-sample descriptive; not external prediction or irreducible biological noise",
                                "full_range_conversion_output": "conversion_formula_sensitivity_all_lab_linked.csv"})
    print(f"Corrected synthesis: symmetric={len(primary):,}, preceding={len(sensitivity):,}, full-range sensitivity={len(full_primary):,}")


if __name__ == "__main__":
    main()
