"""Canonical laboratory value normalization, unit conversion, and daily aggregation.

Converts raw lab draws to canonical units, validates against physiological bounds,
applies the fasting-insulin exclusion rule described below, collapses same-day
duplicate draws to a median, and derives ever-diabetes-range flags per participant.
Has no I/O: it operates on an already-loaded raw lab DataFrame.

Fasting insulin drawn before 2023-03-01 is excluded by default (treated as missing,
which also removes the same-day HOMA-IR): in that period insulin was reported
inconsistently with later draws (a large fraction of pre-2023 monthly draws are
exactly 1.9 uIU/mL and no other value below 2 occurs, whereas from March 2023
values below 2 are reported continuously and 1.9 is rare). Because the 1.9 results
carry information (same-day glucose, lipids, and later draws of the same
participants are consistent with genuinely low insulin), removing only the 1.9
values would selectively remove the lowest values, so the whole pre-2023 period is
excluded instead. Alternative rules (floor_code_only, keep_all) exist only for the
sensitivity analysis in analysis/canonical_panel/insulin_value_sensitivity.py and
for unit tests; production runs use the default exclude_period rule.

Returned daily/flag frames are participant-level data and should stay private;
only the audit dict is aggregate.

Inputs: a raw lab DataFrame with columns USER_ID, TYPE, DRAW_DATE, VALUE, UNIT
(as read from the study database by the caller).

Usage: imported by analysis/canonical_panel/build_panel.py and
analysis/canonical_panel/insulin_value_sensitivity.py; not run directly.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


LAB_DEFINITION_VERSION = "canonical-labs"
# Fasting insulin drawn before 1 March 2023 is excluded (treated as missing, which also removes same-day HOMA-IR).
# In that period insulin was reported inconsistently with later draws: 38-71% of monthly draws from July to December
# 2022 are exactly 1.9 uIU/mL and no other value below 2 occurs, whereas from March 2023 values below 2 are reported
# continuously and 1.9 is 0.5% of draws. The 1.9 results carry information (same-day glucose, lipids and later draws
# of the same participants are those of low insulin), so removing only them would remove the lowest values
# selectively; the whole period is excluded instead.
# INSULIN_RULE alternatives exist only for the sensitivity analysis (analysis/canonical_panel/insulin_value_sensitivity.py).
INSULIN_EXCLUDE_BEFORE = pd.Timestamp("2023-03-01")
INSULIN_FLOOR_CODE = 1.9
# CGM_PAPER_INSULIN_RULE exists only so unit tests with synthetic pre-2023 dates (and their subprocesses) can test the
# normalization mechanics; production runs never set it.
INSULIN_RULE = os.environ.get("CGM_PAPER_INSULIN_RULE", "exclude_period")   # "exclude_period" | "floor_code_only" | "keep_all"
CANONICAL_UNITS = {
    "a1c": "%", "fasting_glucose": "mg/dL", "fasting_insulin": "uIU/mL",
    "triglycerides": "mg/dL", "hdl_cholesterol": "mg/dL", "ldl_cholesterol": "mg/dL",
    "total_cholesterol": "mg/dL", "apo_b": "mg/dL", "hscrp": "mg/L",
    "uric_acid": "mg/dL", "alt": "U/L", "ast": "U/L",
}
BOUNDS = {
    "a1c": (3.0, 20.0), "fasting_glucose": (30.0, 600.0),
    "fasting_insulin": (0.5, 500.0), "triglycerides": (10.0, 5000.0),
    "hdl_cholesterol": (5.0, 200.0), "ldl_cholesterol": (10.0, 500.0),
    "total_cholesterol": (50.0, 800.0), "apo_b": (20.0, 250.0),
    "hscrp": (0.01, 100.0), "uric_acid": (1.0, 20.0),
    "alt": (1.0, 500.0), "ast": (1.0, 500.0),
}
# Explicit unit whitelist: normalized value = reported value * multiplier + offset.
# Do not infer units from magnitude or silently repair malformed vendor text.
UNIT_TRANSFORMS = {
    "a1c": {"%": (1.0, 0.0), "%oftotalhgb": (1.0, 0.0), "mmol/mol": (0.09148, 2.152)},
    "fasting_glucose": {"mg/dl": (1.0, 0.0), "mmol/l": (18.0182, 0.0)},
    "fasting_insulin": {key: (1.0, 0.0) for key in ("uiu/ml", "uu/ml", "u[iu]/ml", "mu/l", "mciu/ml")},
    "triglycerides": {"mg/dl": (1.0, 0.0), "mmol/l": (88.57, 0.0)},
    "apo_b": {"mg/dl": (1.0, 0.0), "g/l": (100.0, 0.0)},
    "hscrp": {"mg/l": (1.0, 0.0), "mg/dl": (10.0, 0.0)},
    "uric_acid": {"mg/dl": (1.0, 0.0)},
}
for _kind in ("hdl_cholesterol", "ldl_cholesterol", "total_cholesterol"):
    UNIT_TRANSFORMS[_kind] = {"mg/dl": (1.0, 0.0), "mg/dl(calc)": (1.0, 0.0), "mmol/l": (38.67, 0.0)}
for _kind in ("alt", "ast"):
    UNIT_TRANSFORMS[_kind] = {key: (1.0, 0.0) for key in ("u/l", "iu/l", "unit/l", "units/l", "iunit/l")}


def round_a1c(values):
    """Half-up tenths; 1e-6 percentage-point tolerance absorbs float32 noise."""
    return np.floor(np.asarray(values, dtype=float) * 10.0 + 0.5 + 1e-5) / 10.0


def normalize_labs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean all supported analytes, collapse dates, and preserve ever-high flags.

    Conversion and individual-row validity gates precede any daily aggregation.
    A1c days whose normalized raw spread exceeds 0.30001 percentage points are
    excluded as anchors/endpoints. Ever-diabetes flags use every accepted raw
    record, including values on such conflicting days; median aggregation cannot
    erase an otherwise valid diabetes-range observation.
    """
    d = raw[["USER_ID", "TYPE", "DRAW_DATE", "VALUE", "UNIT"]].copy().reset_index(drop=True)
    d["DRAW_DATE"] = pd.to_datetime(d["DRAW_DATE"], errors="coerce").dt.normalize()
    d["VALUE"] = pd.to_numeric(d["VALUE"], errors="coerce").astype(float)
    unit = (d["UNIT"].fillna("").astype(str).str.strip().str.lower()
            .str.replace("µ", "u", regex=False).str.replace("μ", "u", regex=False)
            .str.replace(r"\s+", "", regex=True))
    transforms = [UNIT_TRANSFORMS.get(kind, {}).get(key) for kind, key in zip(d["TYPE"], unit)]
    known = d["TYPE"].isin(UNIT_TRANSFORMS)
    unit_ok = pd.Series([value is not None for value in transforms], index=d.index)
    dated = (d["USER_ID"].notna() & d["USER_ID"].astype(str).str.strip().ne("")
             & d["DRAW_DATE"].notna() & np.isfinite(d["VALUE"]))
    multipliers = np.array([value[0] if value else np.nan for value in transforms])
    offsets = np.array([value[1] if value else np.nan for value in transforms])
    d["VALUE"] = d["VALUE"] * multipliers + offsets
    in_range = pd.Series([BOUNDS.get(kind, (np.inf, -np.inf))[0] <= value <=
                          BOUNDS.get(kind, (np.inf, -np.inf))[1]
                          for kind, value in zip(d["TYPE"], d["VALUE"])], index=d.index)
    early_insulin = d["TYPE"].eq("fasting_insulin") & d["DRAW_DATE"].lt(INSULIN_EXCLUDE_BEFORE)
    if INSULIN_RULE == "exclude_period":
        insulin_fill = early_insulin
    elif INSULIN_RULE == "floor_code_only":
        insulin_fill = early_insulin & (d["VALUE"] - INSULIN_FLOOR_CODE).abs().lt(1e-4)
    elif INSULIN_RULE == "keep_all":
        insulin_fill = early_insulin & False
    else:
        raise ValueError(f"unknown INSULIN_RULE {INSULIN_RULE!r}")
    accepted = known & unit_ok & dated & in_range & ~insulin_fill
    converted = unit_ok & dated & ((multipliers != 1.0) | (offsets != 0.0))
    audit = {
        "lab_definition_version": LAB_DEFINITION_VERSION,
        "input_rows": int(len(d)), "unsupported_analyte_rows": int((~known).sum()),
        "unknown_or_missing_unit_rows": int((known & ~unit_ok).sum()),
        "invalid_id_date_or_value_rows": int((known & unit_ok & ~dated).sum()),
        "out_of_range_rows_after_conversion": int((known & unit_ok & dated & ~in_range).sum()),
        "insulin_rule": INSULIN_RULE,
        "fasting_insulin_rows_before_2023_03_excluded": int((known & unit_ok & dated & in_range & insulin_fill).sum()),
        "accepted_raw_rows_before_conflicts": int(accepted.sum()),
        "converted_rows_before_range_gate": int(converted.sum()),
        "accepted_converted_rows": int((converted & accepted).sum()),
        "converted_rows_by_analyte": {str(k): int(v) for k, v in d.loc[converted & accepted, "TYPE"].value_counts().sort_index().items()},
        "unknown_unit_rows_by_analyte": {str(k): int(v) for k, v in d.loc[known & ~unit_ok, "TYPE"].value_counts().sort_index().items()},
        "canonical_units": CANONICAL_UNITS.copy(),
    }
    d = d.loc[accepted].copy()
    is_a1c = d["TYPE"].eq("a1c")
    d["_diabetes"] = ((is_a1c & (round_a1c(d["VALUE"]) >= 6.5)) |
                      (d["TYPE"].eq("fasting_glucose") & d["VALUE"].ge(126.0)))
    flags = d.groupby("USER_ID", as_index=False)["_diabetes"].any().rename(columns={"_diabetes": "diabetes_range_ever"})
    daily = (d.groupby(["USER_ID", "DRAW_DATE", "TYPE"])["VALUE"]
             .agg(value="median", min="min", max="max", n_rows="size").reset_index())
    conflict = daily["TYPE"].eq("a1c") & (daily["max"] - daily["min"]).gt(0.30001)
    high_on_day = ((daily["TYPE"].eq("a1c") & (round_a1c(daily["max"]) >= 6.5)) |
                   (daily["TYPE"].eq("fasting_glucose") & daily["max"].ge(126.0)))
    conflict_high = set(daily.loc[conflict & high_on_day, "USER_ID"])
    retained_high = set(daily.loc[~conflict & high_on_day, "USER_ID"])
    audit.update({
        "a1c_conflict_days_excluded": int(conflict.sum()),
        "rows_on_conflicting_a1c_days": int(daily.loc[conflict, "n_rows"].sum()),
        "duplicate_daily_analyte_groups": int(daily["n_rows"].gt(1).sum()),
        "diabetes_range_participants": int(flags["diabetes_range_ever"].sum()),
        "diabetes_flags_only_from_conflicting_a1c_days": len(conflict_high - retained_high),
        "diabetes_flags_use_accepted_raw_records_before_conflict_exclusion": True,
    })
    daily = daily.loc[~conflict].copy()
    a1c = daily["TYPE"].eq("a1c")
    daily.loc[a1c, "value"] = round_a1c(daily.loc[a1c, "value"])
    audit["retained_daily_values_by_analyte"] = {str(k): int(v) for k, v in daily["TYPE"].value_counts().sort_index().items()}
    return daily.reset_index(drop=True), flags.reset_index(drop=True), audit
