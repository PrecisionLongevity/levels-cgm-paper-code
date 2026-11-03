#!/usr/bin/env python3
"""Qualify meal-shape measurement and canonical-window biomarker associations.

The primary participant traits use at least 30 corrected scored meals within
inclusive ±90 local days of the canonical HbA1c anchor, excluding anyone with
an accepted diabetes-range lab ever. Mean glucose, iAUC, amplitude and shape
models share exactly the same participants per outcome. HC3 coefficient CIs
and in-sample R² describe associations; this is not predictive validation.

Anchors and same-food reliability describe all corrected scored meals. Named
food cells use one nonblank macro item, normalized by lowercase/trim, in a
single-log meal cluster. Whole-tenure traits are an optional, labeled sensitivity.

Inputs
    OUTPUTS/meals/... scored-meal and canonical-panel files produced by
    analysis/meals/build_meals.py and analysis/canonical_panel/build_panel.py,
    loaded via analysis/meals/_shape_analysis.py helpers.

Outputs
    OUTPUTS/meals/meal_shape_biomarker_models.csv,
    meal_shape_biomarker_models_window90.csv,
    meal_shape_biomarker_models_whole_tenure.csv (optional sensitivity),
    meal_shape_anchor_correlations.csv, meal_shape_reliability.csv,
    meal_shape_trait_splithalf.csv, meal_shape_qualification_summary.md,
    and meal_shape_qualification.metadata.json.

Usage
    python analysis/meals/meal_shape_qualification.py [--models-only]
        [--whole-tenure-sensitivity] [--output-dir PATH] [--basis-dir PATH]
        [--canonical-panel PATH] [--meal-substrate PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from analysis.meals import _shape_analysis as shared

OUT = shared.DEFAULT_OUT
PANEL = shared.DEFAULT_PANEL
COMPONENTS = ["fpc1", "fpc2", "fpc3"]
MIN_MEALS_TRAIT = shared.MIN_MEALS
SB_TARGET = .70
OUTCOMES = shared.OUTCOMES
DEMOGRAPHICS = ["bmi", "sex_male", "age_bucket"]
SHAPE_TERMS = ["fpc2_mean", "fpc3_mean"]
PRIMARY_ADJUSTMENT = "mean_glucose_fpc1_only"


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def icc_1(x1, x2):
    x = np.column_stack([x1, x2])
    if len(x) < 2:
        return np.nan
    grand, rows = x.mean(), x.mean(axis=1)
    between = 2 * ((rows - grand)**2).sum() / (len(x)-1)
    within = ((x-rows[:, None])**2).sum() / len(x)
    return float((between-within)/(between+within)) if between+within > 0 else np.nan


def spearman_brown_k(r1, target=SB_TARGET):
    return float(target*(1-r1)/(r1*(1-target))) if r1 > 0 else np.inf


def model_specs():
    return {
        "mean_glucose_only": ["mean_glucose"],
        "mean_glucose_fpc1_amplitude": ["mean_glucose", "fpc1_mean"],
        "mean_glucose_iauc_amplitude": ["mean_glucose", "iauc_mean"],
        PRIMARY_ADJUSTMENT: ["mean_glucose", "fpc1_mean"] + SHAPE_TERMS,
        "mean_glucose_iauc_only": ["mean_glucose", "iauc_mean"] + SHAPE_TERMS,
    }


def fit_biomarker_models(meals, panel=None, *, min_n=100):
    panel = shared.load_panel(PANEL) if panel is None else panel
    traits = shared.user_traits(meals)
    merged = traits.merge(panel, on="USER_ID", validate="one_to_one")
    primary = model_specs()
    maximal = shared.matched_models(merged, primary, min_n=min_n)
    # Each demographic model has its exact participant-matched CGM counterpart.
    complete = merged.dropna(subset=DEMOGRAPHICS)
    sensitivity = {label + "_covcomplete": terms for label, terms in primary.items()}
    sensitivity |= {
        "mean_glucose_fpc1_age_sex_bmi": primary[PRIMARY_ADJUSTMENT] + DEMOGRAPHICS,
        "mean_glucose_iauc_age_sex_bmi": primary["mean_glucose_iauc_only"] + DEMOGRAPHICS,
    }
    covariate = shared.matched_models(complete, sensitivity, min_n=min_n)
    return pd.concat([maximal, covariate], ignore_index=True)


def same_food_reliability(meals):
    sf = meals.loc[meals.N_LOGS_IN_CLUSTER.eq(1) & meals.food_key.notna()].copy()
    sf = sf.sort_values(["USER_ID", "food_key", "MEAL_TS", "MEAL_EVENT_ID"], kind="stable")
    sf["rep"] = sf.groupby(["USER_ID", "food_key"]).cumcount()
    first = sf.loc[sf.rep.eq(0)].set_index(["USER_ID", "food_key"])
    second = sf.loc[sf.rep.eq(1)].set_index(["USER_ID", "food_key"])
    cells = first.join(second, lsuffix="_1", rsuffix="_2", how="inner").reset_index()
    # All metrics use identical finite pairs, including the iAUC amplitude anchor.
    metrics = COMPONENTS + ["IAUC_2H"]
    cells = cells.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[f"{c}_{rep}" for c in metrics for rep in (1, 2)])
    rows = []
    for metric in metrics:
        a, b = f"{metric}_1", f"{metric}_2"
        multi = cells.groupby("USER_ID")[a].transform("size").ge(2)
        dem = cells.loc[multi]
        d1 = dem[a]-dem.groupby("USER_ID")[a].transform("mean")
        d2 = dem[b]-dem.groupby("USER_ID")[b].transform("mean")
        r = pearson(d1, d2)
        rows.append({"metric": metric, "n_cells": len(cells), "n_users": cells.USER_ID.nunique(),
                     "n_cells_demeaned": len(dem), "n_users_demeaned": dem.USER_ID.nunique(),
                     "raw_first_second_r": pearson(cells[a], cells[b]), "participant_demeaned_r": r,
                     "icc_1_two_obs": icc_1(cells[a].to_numpy(), cells[b].to_numpy()),
                     "sb_repeats_to_070": spearman_brown_k(r)})
    return pd.DataFrame(rows)


def trait_split_half(meals):
    meals = meals.replace([np.inf, -np.inf], np.nan).dropna(subset=COMPONENTS+["IAUC_2H"])
    meals = meals.loc[meals.groupby("USER_ID").MEAL_EVENT_ID.transform("size").ge(MIN_MEALS_TRAIT)].copy()
    meals = meals.sort_values(["USER_ID", "MEAL_TS", "MEAL_EVENT_ID"], kind="stable")
    meals["half"] = meals.groupby("USER_ID").cumcount() % 2
    halves = meals.groupby(["USER_ID", "half"])[COMPONENTS].mean().unstack()
    return pd.DataFrame([{"component": c, "n_users": len(halves),
                          "split_half_r": pearson(halves[c][0], halves[c][1])} for c in COMPONENTS])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--whole-tenure-sensitivity", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--basis-dir", type=Path, default=shared.DEFAULT_OUT)
    parser.add_argument("--canonical-panel", type=Path, default=PANEL)
    parser.add_argument("--meal-substrate", type=Path, default=shared.DEFAULT_SUBSTRATE)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = shared.load_panel(args.canonical_panel)
    meals = shared.load_scored_meals(args.basis_dir, args.meal_substrate)
    window = shared.window_meals(meals, panel)
    models = fit_biomarker_models(window, panel)
    models.to_csv(args.output_dir / "meal_shape_biomarker_models.csv", index=False)
    models.to_csv(args.output_dir / "meal_shape_biomarker_models_window90.csv", index=False)
    if args.whole_tenure_sensitivity:
        eligible = shared.user_traits(window).USER_ID
        sensitivity = fit_biomarker_models(meals.loc[meals.USER_ID.isin(eligible)], panel)
        sensitivity["trait_source"] = "whole_tenure_same_users_sensitivity"
        sensitivity.to_csv(args.output_dir / "meal_shape_biomarker_models_whole_tenure.csv", index=False)
    if not args.models_only:
        rows = []
        for component in COMPONENTS:
            for anchor in ["IAUC_2H", "PEAK_RISE_2H", "TIME_TO_PEAK_MIN", "RETURN_10_MIN", "AUC_SHAPE_RATIO"]:
                pair = meals[[component, anchor]].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append({"component": component, "anchor": anchor, "n": len(pair),
                             "pearson_r": pearson(pair[component], pair[anchor])})
        pd.DataFrame(rows).to_csv(args.output_dir / "meal_shape_anchor_correlations.csv", index=False)
        same_food_reliability(meals).to_csv(args.output_dir / "meal_shape_reliability.csv", index=False)
        trait_split_half(window).to_csv(args.output_dir / "meal_shape_trait_splithalf.csv", index=False)
    shared.write_metadata(args.output_dir, "meal_shape_qualification.metadata.json", panel=args.canonical_panel,
                          basis_dir=args.basis_dir, meal_substrate=args.meal_substrate,
                          n_scored_meals=len(meals), n_canonical_window_meals=len(window),
                          n_trait_participants=len(shared.user_traits(window)), models_only=args.models_only,
                          reliability_scope="all corrected scored meals; shared finite same-food pairs",
                          food_identity="single-log cluster, exactly one total macro row with a nonblank name; lower(trim(FOOD_ITEM_NAME))",
                          whole_tenure_sensitivity=args.whole_tenure_sensitivity,
                          model_scale="within-common-outcome-sample standardization; HC3; in-sample R²",
                          multiplicity="BH over FPC2/FPC3 across four outcomes within each specification; exploratory families")
    lines = ["# Corrected meal-shape qualification", "",
             "Primary traits: at least 30 corrected meals within ±90 local days of the canonical anchor; no-diabetes-range participants.",
             "Mean, mean+FPC1, mean+iAUC and amplitude+shape comparisons share one participant set per outcome. Demographic sensitivity uses its own matched complete subset.",
             "All reported R² values are in-sample association summaries. Alternative bases and macro adjustment are internal robustness analyses.",
             "Anchors and same-food reliability use all corrected scored meals; trait split-half uses the canonical window. No component meaning or effect direction is imposed.",
             "", models.round(4).to_string(index=False)]
    (args.output_dir / "meal_shape_qualification_summary.md").write_text("\n".join(lines)+"\n")
    print(f"Scored meals {len(meals):,}; canonical-window meals {len(window):,}; coefficient rows {len(models):,}")


if __name__ == "__main__":
    main()
