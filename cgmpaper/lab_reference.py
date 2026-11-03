"""Normalized laboratory references and deterministic first/last event endpoints.

Provides shared helpers for reference/longitudinal lab populations, which are
distinct from the lab-anchored CGM panel built in analysis/canonical_panel/build_panel.py.
They share the same laboratory QC and normalization (see lab_normalization.py) but
never substitute lifetime CGM data for fasting labs.

Key computations:
- load_normalized_labs: pulls normalized daily lab values, ever-high flags, and QC
  audit information for non-internal accounts from the labs/users tables.
- same_day_indices: pivots unique normalized daily labs to one row per
  participant/date and derives same-date HOMA-IR, TyG, TG:HDL ratio, and
  HOMA-beta (undefined for fasting glucose <=63 mg/dL).
- first_last_daily: selects each group's first and last dated row (requiring a
  minimum follow-up and at least two measurements) without combining values
  across dates.

Inputs: database tables `labs` and `users`, read via LEVELS_DB.
Outputs: none directly; these are shared helpers used by analysis modules that
write their own outputs under OUTPUTS/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .lab_normalization import BOUNDS, LAB_DEFINITION_VERSION, normalize_labs

REFERENCE_VERSION = "normalized-lab-reference"
DERIVED_BOUNDS = {"homa_ir": (0.05, 100.0), "tyg": (4.0, 15.0), "tg_hdl_ratio": (0.05, 100.0)}


def load_normalized_labs(con, types=None):
    """Return normalized daily values, ever-high flags, and aggregate QC.

    Select only explicitly non-internal accounts. Include both diabetes-screening
    analytes even for a narrower requested endpoint so cohort flags stay uniform.
    """
    wanted = sorted(set(types or BOUNDS) | {"a1c", "fasting_glucose"})
    if not set(wanted) <= set(BOUNDS):
        raise ValueError("Unsupported normalized laboratory analyte")
    placeholders = ",".join("?" for _ in wanted)
    raw = con.execute(f"""
        SELECT l.USER_ID, l.TYPE, l.DRAW_DATE, l.VALUE, l.UNIT
        FROM labs l JOIN users u ON u.USER_ID=l.USER_ID
        WHERE u.IS_INTERNAL_USER=FALSE AND l.TYPE IN ({placeholders})
    """, wanted).fetchdf()
    daily, flags, audit = normalize_labs(raw)
    audit["account_rule"] = "IS_INTERNAL_USER=false; missing account status excluded"
    return daily, flags, audit


def same_day_indices(daily: pd.DataFrame) -> pd.DataFrame:
    """Pivot unique normalized daily labs; form same-date HOMA/TyG/TG:HDL.

    Derived bounds match the canonical builder. HOMA-beta is optional description,
    undefined for glucose <=63; it never gates availability of valid HOMA-IR.
    """
    keys = ["USER_ID", "DRAW_DATE", "TYPE"]
    if daily.duplicated(keys).any():
        raise ValueError("Same-day indices require one normalized value per participant/date/analyte")
    if daily.empty:
        wide = daily[["USER_ID", "DRAW_DATE"]].copy()
    else:
        wide = daily.pivot(index=["USER_ID", "DRAW_DATE"], columns="TYPE", values="value").reset_index()
    wide.columns.name = None
    for analyte in ("fasting_glucose", "fasting_insulin", "triglycerides", "hdl_cholesterol"):
        if analyte not in wide:
            wide[analyte] = np.nan
    wide["homa_ir"] = wide.fasting_glucose * wide.fasting_insulin / 405.0
    wide["tyg"] = np.log(wide.triglycerides * wide.fasting_glucose / 2.0)
    wide["tg_hdl_ratio"] = wide.triglycerides / wide.hdl_cholesterol
    for name, (low, high) in DERIVED_BOUNDS.items():
        wide.loc[~wide[name].between(low, high), name] = np.nan
    denominator = (wide.fasting_glucose - 63.0).where(wide.fasting_glucose.gt(63.0))
    wide["homa_beta"] = 360.0 * wide.fasting_insulin / denominator
    return wide.sort_values(["USER_ID", "DRAW_DATE"], kind="stable").reset_index(drop=True)


def first_last_daily(events: pd.DataFrame, date_column: str, *, min_days: int = 30,
                     group_columns=None) -> pd.DataFrame:
    """Select complete first/last rows, preserving null cells on those rows.

    Input must already have one normalized row per date and group. Unlike groupby
    first/last, this cannot combine values from different dates by skipping nulls.
    """
    if min_days < 0:
        raise ValueError("min_days must be nonnegative")
    groups = list(group_columns or ["USER_ID"])
    data = events.copy()
    data[date_column] = pd.to_datetime(data[date_column]).dt.normalize()
    if data[date_column].isna().any() or data.duplicated(groups + [date_column]).any():
        raise ValueError("First/last selection requires unique, dated normalized daily rows")
    data = data.sort_values(groups + [date_column], kind="stable")
    counts = data.groupby(groups, as_index=False).size().rename(columns={"size": "n_measurements"})
    first = data.drop_duplicates(groups, keep="first")
    last = data.drop_duplicates(groups, keep="last")
    pairs = first.merge(last, on=groups, suffixes=("_first", "_last"), validate="one_to_one")
    pairs["days_between"] = (pairs[f"{date_column}_last"] - pairs[f"{date_column}_first"]).dt.days
    pairs = pairs.merge(counts, on=groups, validate="one_to_one")
    return pairs.loc[pairs.days_between.ge(min_days) & pairs.n_measurements.ge(2)].reset_index(drop=True)


def write_lab_metadata(path: Path, *, audit: dict, outputs: list[Path], details: dict) -> None:
    """Write aggregate run metadata describing the laboratory processing."""
    metadata = {
        "reference_version": REFERENCE_VERSION,
        "lab_definition_version": LAB_DEFINITION_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "laboratory_processing": audit,
        **details,
    }
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
