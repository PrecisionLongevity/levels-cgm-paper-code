"""Supplementary Table 5: definitions of the CGM measures and their cohort medians.

Median (interquartile range) of each measure across participants (or meals), in the population over which the paper
summarizes it:
  whole record        every participant's full record (average glucose, SD, CV, time 70-140 mg/dL)
  lab-linked panel    canonical primary cohort, no diabetes-range laboratory value (MODD, MAGE, spike measures)
  logged meals        isolated logged meals with all 30 five-minute cells observed, participants with any
                      diabetes-range laboratory value excluded (meal peak rise, 2-h iAUC)
  >=30 logged meals   participants with >= 30 logged meals and a carbohydrate-sensitivity slope, same exclusion

Inputs (analysis outputs; the meal rows also need the study database through LEVELS_DB, for the diabetes-range roster)
  outputs/cohort/cgm_reference.parquet                         (analysis/cohort/cohort_reference.py)
  outputs/canonical_panel/canonical_lab_cgm_panel.csv          (analysis/canonical_panel/build_panel.py)
  outputs/clinical/mage_standard/mage_distribution.csv         (analysis/clinical/mage_standard.py)
  outputs/meals/spike_facets/spike_facets_participants.parquet (analysis/meals/spike_facets.py)
  outputs/clinical/spike_width/width_summary.json              (analysis/clinical/spike_width.py)
  outputs/meals/meal_shape_substrate.parquet, meal_traces_observed.parquet   (analysis/meals/build_meals.py)
  outputs/meal_populations/_cache/broad_mixed_participants.parquet           (analysis/meals/meal_populations.py)
Source data
  source_data/stable5_cgm_measures.csv
Outputs
  stable5_cgm_measures.md / .csv

Usage
  python tables/stable5_cgm_measures.py source-data      (needs LEVELS_DB)
  python tables/stable5_cgm_measures.py draw
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, REPO, TABLES_OUT, database_path  # noqa: E402

NAME = 'stable5_cgm_measures'
TITLE = 'Supplementary Table 5 | CGM measures: definitions and cohort values'
MIN_MEALS_CARB = 30

# measure, definition, unit, population label, decimals
ROWS = [
    ('Average glucose', 'average of the observed 15-minute cells of the record', 'mg/dL', 'whole record', 1),
    ('SD', 'standard deviation of the same cells (spread around the average)', 'mg/dL', 'whole record', 1),
    ('CV', 'SD divided by the average, × 100', '%', 'whole record', 1),
    ('MODD', 'mean absolute difference between consecutive days at the same clock time', 'mg/dL', 'lab-linked panel', 1),
    ('MAGE', 'mean amplitude of glycemic excursions: mean of rises and falls larger than 1 SD (moving-average algorithm; '
             'Methods)', 'mg/dL', 'lab-linked panel', 1),
    ('Time 70–140 mg/dL', 'share of observed time with glucose between 70 and 140 mg/dL', '% of time', 'whole record', 1),
    ('Spike frequency', 'spikes per eligible day; a spike is a peak ≥30 mg/dL above the lowest value of the preceding 90 min',
     'spikes per day', 'lab-linked panel', 1),
    ('Spike height', "a participant's mean rise across their spikes: peak minus its preceding 90-min low", 'mg/dL',
     'lab-linked panel', 1),
    ('Spike width', 'minutes above half the rise of each spike, participant mean over resolved spikes', 'min',
     'lab-linked panel', 1),
    ('Meal peak rise', 'highest value 0–120 min after a logged meal minus the pre-meal baseline', 'mg/dL',
     'isolated logged meals', 0),
    ('2-h iAUC', 'area between the curve and the baseline, 0–120 min, counting only area above it', 'mg/dL·min',
     'isolated logged meals', 0),
    ('Carbohydrate sensitivity', "a person's peak rise per gram of carbohydrate, adjusted for fiber, fat, protein",
     'mg/dL per g', '≥30 logged meals', 2),
]


def quartiles(values):
    s = pd.Series(values, dtype=float).dropna()
    return dict(n=int(len(s)), median=float(s.median()), q1=float(s.quantile(.25)), q3=float(s.quantile(.75)))


def diabetes_range_users():
    """Participants with any accepted diabetes-range laboratory value (analysis/validation/diabetes_range_exclusion.py)."""
    spec = importlib.util.spec_from_file_location('diabetes_range_exclusion', REPO / 'analysis/validation/diabetes_range_exclusion.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with duckdb.connect(str(database_path()), read_only=True) as con:
        return mod.diabetes_range_users(con)


def make_source_data():
    v = {}
    ref = pd.read_parquet(OUTPUTS / 'cohort/cgm_reference.parquet', columns=['mean_glucose', 'sd_glucose', 'cv_glucose', 'tir_70_140'])
    for col, measure in (('mean_glucose', 'Average glucose'), ('sd_glucose', 'SD'), ('cv_glucose', 'CV'), ('tir_70_140', 'Time 70–140 mg/dL')):
        v[measure] = dict(quartiles(ref[col]), unit_n='participants')
    panel = pd.read_csv(OUTPUTS / 'canonical_panel/canonical_lab_cgm_panel.csv', usecols=['no_diabetes_range', 'modd'], low_memory=False)
    v['MODD'] = dict(quartiles(panel.loc[panel.no_diabetes_range == True, 'modd']), unit_n='participants')  # noqa: E712
    mage = pd.read_csv(OUTPUTS / 'clinical/mage_standard/mage_distribution.csv')
    m = mage.loc[mage.population.str.startswith('primary cohort') & mage.measure.str.startswith('standard MAGE')].iloc[0]
    v['MAGE'] = dict(n=int(m.n), median=float(m['median']), q1=float(m.q1), q3=float(m.q3), unit_n='participants')
    facets = pd.read_parquet(OUTPUTS / 'meals/spike_facets/spike_facets_participants.parquet', columns=['all_frequency', 'all_size'])
    v['Spike frequency'] = dict(quartiles(facets.all_frequency), unit_n='participants')
    v['Spike height'] = dict(quartiles(facets.all_size), unit_n='participants')
    width = json.loads((OUTPUTS / 'clinical/spike_width/width_summary.json').read_text())
    w = width['descriptives']['spike_width']
    v['Spike width'] = dict(n=int(width['n_participants_with_width']), median=w['median'], q1=w['q1'], q3=w['q3'], unit_n='participants')
    excluded = diabetes_range_users()
    cells = [f'g{i}' for i in range(30)]
    meals = pd.read_parquet(OUTPUTS / 'meals/meal_shape_substrate.parquet', columns=['USER_ID', 'MEAL_EVENT_ID', 'PEAK_RISE_2H', 'IAUC_2H'])
    traces = pd.read_parquet(OUTPUTS / 'meals/meal_traces_observed.parquet', columns=['USER_ID', 'MEAL_EVENT_ID', *cells])
    meals = meals.merge(traces, on=['USER_ID', 'MEAL_EVENT_ID'], validate='one_to_one')
    full = np.isfinite(meals[cells].to_numpy(float)).all(axis=1)
    keep = ~meals.USER_ID.isin(excluded) & full & np.isfinite(meals.PEAK_RISE_2H) & np.isfinite(meals.IAUC_2H)
    v['Meal peak rise'] = dict(quartiles(meals.loc[keep, 'PEAK_RISE_2H']), unit_n='meals')
    v['2-h iAUC'] = dict(quartiles(meals.loc[keep, 'IAUC_2H']), unit_n='meals')
    carb = pd.read_parquet(OUTPUTS / 'meal_populations/_cache/broad_mixed_participants.parquet', columns=['USER_ID', 'acs', 'n_meals'])
    carb = carb.loc[~carb.USER_ID.isin(excluded) & carb.n_meals.ge(MIN_MEALS_CARB)]
    v['Carbohydrate sensitivity'] = dict(quartiles(carb.acs), unit_n='participants')
    df = pd.DataFrame([dict(measure=measure, definition=definition, unit=unit, summarized_over=population, n=v[measure]['n'],
                            n_unit=v[measure]['unit_n'], median=v[measure]['median'], q1=v[measure]['q1'], q3=v[measure]['q3'],
                            decimals=dec) for measure, definition, unit, population, dec in ROWS])
    return [source_data.write(NAME, df)]


def fmt(x, dec):
    return f'{x:,.{dec}f}'


def draw(out_dir):
    d = source_data.read(NAME)
    t = pd.DataFrame([(r.measure, r.definition, r.unit,
                       f'{fmt(r.median, r.decimals)} ({fmt(r.q1, r.decimals)} to {fmt(r.q3, r.decimals)})',
                       f'{r.summarized_over}; {int(r.n):,} {r.n_unit}') for r in d.itertuples()],
                     columns=['Measure', 'Definition', 'Unit', 'Cohort median (IQR)', 'Summarized over'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
