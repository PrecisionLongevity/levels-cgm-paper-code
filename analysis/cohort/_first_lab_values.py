#!/usr/bin/env python3
"""First valid daily laboratory values per participant for the whole-cohort reference.

For each analyte, takes the earliest accepted DRAW_DATE per participant after shared
lab QC, restricted to non-internal accounts, using same-day medians before selecting
the first date. This descriptive reference is distinct from the matched lab-CGM
analytic cohort (built in analysis/canonical_panel/build_panel.py).

eGFR (creatinine) is normalized separately: only values reported in an indexed
mL/min/1.73 m^2 unit are accepted (unindexed mL/min is not assumed to be
body-surface-area normalized), with accepted values restricted to the 15-200 range,
then reduced to a daily median before taking the earliest date.

Inputs: `labs` and `users` tables via LEVELS_DB (DB).

Outputs:
- OUTPUTS/cohort/first_lab_values_per_user.csv - participant-level, one row per
  USER_ID with first-available value per analyte (private; not for release).
- OUTPUTS/cohort/first_lab_values_per_user.metadata.json - audit counts and rules.

Usage: imported by analysis/cohort/cohort_reference.py; also runnable directly as
`python analysis/cohort/_first_lab_values.py [--database PATH] [--output-dir DIR]`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402

from cgmpaper.lab_reference import load_normalized_labs, write_lab_metadata

DB = database_path()
OUT_DIR = OUTPUTS / 'cohort'
ANALYTES = ("uric_acid", "hscrp", "fasting_glucose", "egfr_creatinine")
# Only indexed eGFR at the same mL/min/1.73 m² scale. Unindexed mL/min is not
# silently assumed to mean body-surface-area normalization.
EGFR_UNITS = {"ml/min/1.73", "ml/min/1.73m2", "ml/min/1.73m^2", "ml/min/{1.73_m2}",
              "ml/min/1.73m²", "ml/min/1.73_m2"}


def normalize_egfr(raw: pd.DataFrame):
    d = raw[["USER_ID", "DRAW_DATE", "VALUE", "UNIT"]].copy()
    d["DRAW_DATE"] = pd.to_datetime(d.DRAW_DATE, errors="coerce").dt.normalize()
    d["VALUE"] = pd.to_numeric(d.VALUE, errors="coerce")
    unit = d.UNIT.fillna("").astype(str).str.lower().str.replace(r"\s+", "", regex=True)
    unit_ok = unit.isin(EGFR_UNITS)
    valid = (d.USER_ID.notna() & d.USER_ID.astype(str).str.strip().ne("") & d.DRAW_DATE.notna()
             & np.isfinite(d.VALUE) & d.VALUE.between(15.0, 200.0))
    audit = {"input_rows": len(d), "unknown_or_missing_unit_rows": int((~unit_ok).sum()),
             "invalid_value_date_or_id_rows": int((unit_ok & ~valid).sum()),
             "accepted_raw_rows": int((unit_ok & valid).sum()),
             "canonical_unit": "mL/min/1.73m2", "validity_bounds": [15.0, 200.0]}
    daily = (d.loc[unit_ok & valid].groupby(["USER_ID", "DRAW_DATE"], as_index=False)
             .VALUE.median().rename(columns={"VALUE": "value"}))
    daily["TYPE"] = "egfr_creatinine"
    return daily, audit


def build_first_values(con):
    daily, _, audit = load_normalized_labs(con, types=ANALYTES[:-1])
    daily = daily[daily.TYPE.isin(ANALYTES)]
    raw = con.execute("""
        SELECT l.USER_ID,l.DRAW_DATE,l.VALUE,l.UNIT FROM labs l
        JOIN users u ON u.USER_ID=l.USER_ID
        WHERE u.IS_INTERNAL_USER=FALSE AND l.TYPE='egfr_creatinine'
    """).fetchdf()
    egfr, egfr_audit = normalize_egfr(raw)
    daily = pd.concat([daily[["USER_ID", "DRAW_DATE", "TYPE", "value"]], egfr], ignore_index=True)
    first = daily.sort_values(["USER_ID", "TYPE", "DRAW_DATE"], kind="stable").drop_duplicates(["USER_ID", "TYPE"])
    if first.empty:
        wide = pd.DataFrame(columns=["USER_ID", *ANALYTES])
    else:
        wide = first.pivot(index="USER_ID", columns="TYPE", values="value").reset_index()
        wide.columns.name = None
        for name in ANALYTES:
            if name not in wide:
                wide[name] = np.nan
        wide = wide[["USER_ID", *ANALYTES]].sort_values("USER_ID").reset_index(drop=True)
    audit["egfr"] = egfr_audit
    return wide, audit


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    with duckdb.connect(str(args.database), read_only=True) as con:
        wide, audit = build_first_values(con)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "first_lab_values_per_user.csv"
    wide.to_csv(path, index=False)
    write_lab_metadata(path.with_suffix(".metadata.json"), audit=audit, outputs=[path], details={
        "population": "All explicitly non-internal accounts with accepted analyte observations; no CGM requirement",
        "selection": "Earliest accepted DRAW_DATE per analyte after unit normalization and same-day median",
        "no_diabetes_restriction": False,
        "rows": len(wide), "available_by_analyte": {name: int(wide[name].notna().sum()) for name in ANALYTES},
        "egfr_rule": "Indexed mL/min/1.73m2 whitelist, range 15–200; daily median then earliest date",
        "egfr_units_normalized": sorted(EGFR_UNITS),
    })
    print(f"Wrote corrected first-lab reference: {len(wide):,} participants")
    for name in ANALYTES:
        print(f"  {name}: {wide[name].notna().sum():,}")


if __name__ == "__main__":
    main()
