#!/usr/bin/env python3
"""Compute descriptive associations between CGM glucose metrics and recorded daily context.

Uses the first dense seven-day window per participant from the repeatability cohort
(analysis/reliability/repeatability.py) and merges nearby context logs (food/nutrient,
steps, sleep) and nearest BMI with the glucose metrics computed over that window, then
fits descriptive (adjusted and unadjusted) models relating glucose axes to each context
variable.

Definitions
- Population: participants with a verified first dense seven-day CGM window, as
  identified by the repeatability analysis.
- Context variables: logged nutrient intake, step counts, sleep, and nearest-available
  BMI, each matched to the glucose window.
- Interpretation: these are descriptive associations with recorded context; ad hoc
  logging does not measure total intake or causal effects. Alcohol values are
  text-derived inferred serving units, not measured standard drinks.

Inputs
- OUTPUTS/reliability/repeatability_analysis.metadata.json and
  OUTPUTS/reliability/cache/* (seven_day_windows.parquet, seven_day_cgm_features.parquet,
  repeatability_features.metadata.json), produced by analysis/reliability/repeatability.py.
- Database tables read via LEVELS_DB (context logs, BMI).

Outputs (OUTPUTS/reliability/)
- context_axis_associations.csv, context_mean_reference.csv, context_axis_scales.csv,
  context_raw_correlations.csv, context_coverage.csv
- context_analysis.metadata.json
- cache/context_daily.parquet, cache/context_primary.parquet (participant-level, private)

Usage
    python analysis/reliability/context.py [--results-root PATH]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.reliability import _context_analysis as context


def verified_primary(folder):
    cache = folder / 'cache'
    windows = pd.read_parquet(cache / 'seven_day_windows.parquet')
    features = pd.read_parquet(cache / 'seven_day_cgm_features.parquet')
    primary = windows.merge(features, on='window_id', validate='one_to_one')
    primary = primary.loc[primary.window_number.eq(1)].copy()
    if primary.USER_ID.duplicated().any():
        raise ValueError('First-week population differs from repeatability cohort')
    if not (pd.to_datetime(primary.window_end) - pd.to_datetime(primary.window_start)).dt.days.eq(7).all():
        raise ValueError('Expected exclusive-end seven-day windows')
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=OUTPUTS)
    args = parser.parse_args()
    folder = args.results_root / 'reliability'
    cache = folder / 'cache'
    primary = verified_primary(folder)
    print(f'Using {len(primary):,} first dense weeks', flush=True)
    db = database_path()
    with duckdb.connect(str(db), read_only=True) as con:
        con.execute("SET threads=3; SET memory_limit='3GB'")
        con.register('primary_windows', primary)
        daily = context.daily_context(con)
        bmi = context.nearest_bmi(con)
    observed = context.summarize_context(daily)
    results, raw, scales, joined = context.context_models(primary, observed, bmi)
    if results.duplicated(['axis', 'context', 'adjustment']).any():
        raise ValueError('Incomplete model inventory')
    daily.to_parquet(cache / 'context_daily.parquet', index=False)
    joined.to_parquet(cache / 'context_primary.parquet', index=False)
    outputs = {
        'context_axis_associations.csv': results.loc[results.axis.ne('mean_glucose')],
        'context_mean_reference.csv': results.loc[results.axis.eq('mean_glucose')],
        'context_axis_scales.csv': scales,
        'context_raw_correlations.csv': raw,
    }
    coverage = []
    for column in context.CONTEXTS + ['nutrient_days', 'step_days', 'sleep_days', 'midpoint_days', 'paired_bmi']:
        values = joined[column].replace([np.inf, -np.inf], np.nan).dropna()
        coverage.append(dict(variable=column, n_observed=len(values), n_missing=len(joined)-len(values),
                             median=values.median(), p25=values.quantile(.25), p75=values.quantile(.75)))
    outputs['context_coverage.csv'] = pd.DataFrame(coverage)
    for name, frame in outputs.items():
        frame.to_csv(folder / name, index=False)
    metadata = dict(
        n_first_week=len(primary), n_context_dates=len(daily),
        status_counts=results.status.value_counts().to_dict(),
        interpretation='Descriptive associations with recorded context; ad hoc logging does not measure total intake or causal effects.',
        alcohol='Text-derived inferred serving units, not measured standard drinks.',
    )
    (folder / 'context_analysis.metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(n_first_week=len(primary), model_status=metadata['status_counts'],
                         n_range=[int(results.n_adjusted.min()), int(results.n_adjusted.max())]), indent=2))


if __name__ == '__main__':
    main()
