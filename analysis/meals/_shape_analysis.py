"""Common canonical-window meal substrate and matched descriptive meal-shape models.

Builds the shared meal trait substrate used by meal-shape analyses: restricts meals
to the participant-local meal day within an inclusive +/-90-day window of the
canonical anchor date, joins corrected meal-PCA scores (fpc1-fpc3) with meal
metadata (IAUC, baseline, cluster size, food key), and aggregates to per-participant
trait means over a minimum number of scored meals. Provides `matched_models`, which
fits HC3-robust OLS association models between shape traits and metabolic outcomes
(HOMA-IR, A1c, triglycerides, fasting insulin; log-transformed where noted) on a
common complete-case sample per outcome, with FDR control across shape terms.

Inputs: canonical lab/CGM panel (analysis/canonical_panel/build_panel.py), meal
PCA scores and metadata (analysis/meals/meal_fpca.py), and the meal shape
substrate (analysis/meals/build_meals.py).

This is a helper module; it is imported by the meal-shape analysis producers
rather than run directly."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.cross_validation import validate_publication_panel

PUBLICATION = OUTPUTS
DEFAULT_OUT = PUBLICATION / 'meals'
DEFAULT_PANEL = PUBLICATION / "canonical_panel/canonical_lab_cgm_panel.csv"
DEFAULT_TRACES = DEFAULT_OUT / "meal_traces_observed.parquet"
DEFAULT_SUBSTRATE = DEFAULT_OUT / "meal_shape_substrate.parquet"
DEFINITION_VERSION = "canonical-window-meal-shape"
OUTCOMES = [("homa_ir", True), ("a1c", False), ("triglycerides", True), ("fasting_insulin", True)]
MIN_MEALS = 30
WINDOW_DAYS = 90
KEYS = ["USER_ID", "MEAL_EVENT_ID"]


def refit_module():
    spec = importlib.util.spec_from_file_location("meal_fpca_for_shape_analysis", Path(__file__).with_name("meal_fpca.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_basis(out=DEFAULT_OUT):
    out = Path(out)
    metadata = json.loads((out / "meal_fpca_metadata.json").read_text())
    if metadata.get("definition_version") != refit_module().DEFINITION_VERSION:
        raise ValueError("Shape analysis requires the corrected self-contained meal PCA")
    return metadata


def load_panel(path=DEFAULT_PANEL, cohort="no_diabetes_range"):
    metadata = validate_publication_panel(path)
    frame = pd.read_csv(path, dtype={"USER_ID": str})
    if frame.USER_ID.duplicated().any():
        raise ValueError("Canonical shape models require one anchor per participant")
    if cohort == "no_diabetes_range":
        frame = frame.loc[frame.no_diabetes_range.astype(str).str.lower().eq("true")].copy()
    elif cohort != "all":
        raise ValueError("Unknown shape cohort")
    frame["anchor_date"] = pd.to_datetime(frame.anchor_date).dt.normalize()
    frame["sex_male"] = frame.sex.astype(str).str.lower().map({"male": 1., "female": 0.})
    frame.attrs["canonical_metadata"] = metadata
    return frame


def window_meals(meals, panel, *, whole_tenure=False):
    """Use the participant-local meal day and inclusive ±90-day anchor bounds."""
    fields = ["USER_ID", "anchor_date", "tz_offset_hours"]
    frame = meals.merge(panel[fields], on="USER_ID", how="inner", validate="many_to_one")
    if not whole_tenure:
        utc = pd.to_datetime(frame.MEAL_TS, utc=True).dt.tz_convert(None)
        local_day = (utc + pd.to_timedelta(frame.tz_offset_hours, unit="h")).dt.normalize()
        difference = (local_day - frame.anchor_date).dt.days
        frame = frame.loc[difference.between(-WINDOW_DAYS, WINDOW_DAYS)].copy()
    return frame.drop(columns=["anchor_date", "tz_offset_hours"])


def validate_meal_substrate(path=DEFAULT_SUBSTRATE):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    gates = (metadata.get("all_food_logs_used") is True,
             metadata.get("matched_shape_iauc_baseline") is True,
             metadata.get("shape_end_minutes") == 240,
             metadata.get("post_meal_isolation_minutes", 0) >= 240,
             metadata.get("noninternal_only") is True)
    if not all(gates):
        raise ValueError("Primary shape substrate requires all-food isolation through 4h, matched area/shape baseline, and noninternal participants")
    return metadata


def load_scored_meals(out=DEFAULT_OUT, meal_substrate=DEFAULT_SUBSTRATE):
    """Load restricted corrected meal metadata; primary analysis fails closed."""
    validate_basis(out)
    validate_meal_substrate(meal_substrate)
    scores = pd.read_parquet(Path(out) / "meal_fpca_scores.parquet")
    metadata = pd.read_parquet(meal_substrate)
    if scores.duplicated(KEYS).any() or metadata.duplicated(KEYS).any():
        raise ValueError("Duplicate scored or substrate meal keys")
    required = KEYS + ["MEAL_TS", "IAUC_2H", "BASELINE_PRE", "N_LOGS_IN_CLUSTER", "food_key"]
    if set(required) - set(metadata):
        raise ValueError("Corrected meal substrate is missing required metadata fields")
    frame = scores.merge(metadata, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Scored meals do not match the corrected meal substrate")
    return frame.drop(columns="_merge")


def completed_substrate(out=DEFAULT_OUT, traces=DEFAULT_TRACES):
    """Exactly reuse the meal PCA trace completion, centering and keyed score order."""
    validate_basis(out)
    frame, centered, _ = refit_module().complete_traces(pd.read_parquet(traces))
    scores = pd.read_parquet(Path(out) / "meal_fpca_scores.parquet")
    if not frame[KEYS].equals(scores[KEYS]):
        raise ValueError("Completed traces and scores must share exact keyed row order")
    return frame, centered, scores


def user_traits(meals, components=("fpc1", "fpc2", "fpc3"), *, min_meals=MIN_MEALS):
    """Use a common meal set for all compared participant trait means."""
    required = list(components) + ["IAUC_2H"]
    finite = meals.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    aggregation = {f"{column}_mean": (column, "mean") for column in components}
    aggregation |= {"iauc_mean": ("IAUC_2H", "mean"), "n_meals": ("MEAL_EVENT_ID", "size")}
    traits = finite.groupby("USER_ID").agg(**aggregation).reset_index()
    return traits.loc[traits.n_meals.ge(min_meals)].copy()


def matched_models(frame, specs, *, label_column="adjustment", shape_terms=("fpc2_mean", "fpc3_mean"), min_n=100):
    """HC3 association models with one common complete sample per outcome.

    R² is explicitly in-sample. This function makes no predictive-validation
    claim for an unsupervised basis fitted on a shared cohort.
    """
    columns = sorted({column for values in specs.values() for column in values})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    rows = []
    for outcome, log_transform in OUTCOMES:
        sample = frame.dropna(subset=columns + [outcome]).copy()
        if log_transform:
            sample = sample.loc[sample[outcome].gt(0)].copy()
        if len(sample) < min_n:
            continue
        if sample.USER_ID.duplicated().any():
            raise ValueError("Shape association sample must contain one row per participant")
        y = np.log(sample[outcome]) if log_transform else sample[outcome]
        if y.std() == 0:
            continue
        y = (y - y.mean()) / y.std()
        for label, predictors in specs.items():
            x = sample[predictors].astype(float)
            x = (x - x.mean()) / x.std().replace(0, 1)
            x = sm.add_constant(x, has_constant="add")
            fit = sm.OLS(y, x).fit(cov_type="HC3")
            for term in predictors:
                rows.append({label_column: label, "outcome": outcome, "term": term,
                             "n": len(sample), "beta": float(fit.params[term]),
                             "ci_low": float(fit.conf_int().loc[term, 0]),
                             "ci_high": float(fit.conf_int().loc[term, 1]), "p": float(fit.pvalues[term]),
                             "r_squared_in_sample": float(fit.rsquared),
                             "cohort": "canonical_no_diabetes_range", "trait_source": "canonical_window90",
                             "definition_version": DEFINITION_VERSION})
    result = pd.DataFrame(rows, columns=[label_column, "outcome", "term", "n", "beta", "ci_low", "ci_high", "p",
                                        "r_squared_in_sample", "cohort", "trait_source", "definition_version"])
    result["q_fdr_shape_family"] = np.nan
    for label in specs:
        mask = result[label_column].eq(label) & result.term.isin(shape_terms) & np.isfinite(result.p)
        if mask.any():
            result.loc[mask, "q_fdr_shape_family"] = stats.false_discovery_control(result.loc[mask, "p"])
    return result


def write_metadata(out, name, *, panel=DEFAULT_PANEL, basis_dir=DEFAULT_OUT, meal_substrate=DEFAULT_SUBSTRATE, **details):
    document = {"definition_version": DEFINITION_VERSION,
                "canonical_panel_metadata": validate_publication_panel(panel),
                "basis_metadata": validate_basis(basis_dir),
                "meal_substrate_metadata": validate_meal_substrate(meal_substrate),
                "primary_trait_window": "participant-local meal day within inclusive ±90 days of canonical anchor",
                "minimum_scored_meals_per_trait": MIN_MEALS,
                "cohort": "no_diabetes_range",
                "iauc_definition": "corrected positive 0–120-minute area with the same premeal baseline as the trace PCA; see meal substrate provenance",
                "shape_scope": "shared-cohort internal robustness; descriptive associations, not independent replication",
                **details}
    (Path(out) / name).write_text(json.dumps(document, indent=2, default=str) + "\n")
