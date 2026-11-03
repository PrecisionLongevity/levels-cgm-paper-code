"""Supplementary Table 7: age and sex sensitivity.

A  % difference in HOMA-IR and fasting insulin per SD of each measure, in the same participants without and with age
   decade and sex (unadjusted, and at the same average glucose); MAGE is the standard moving-average MAGE.
B  variance in log fasting insulin and log HOMA-IR explained by average glucose and by HbA1c, unadjusted and beyond age
   and sex (participant bootstrap 95% CI).

Inputs
  outputs/metabolic/unadjusted_insulin_markers/age_sex_sensitivity.csv          A
  outputs/clinical/mage_standard/age_sex_sensitivity_mage_standard.csv          A, MAGE rows
  outputs/metabolic/hba1c_unadjusted_r2/age_sex_r2_sensitivity.csv              B
Source data
  source_data/stable7a_age_sex.csv, stable7b_variance_explained.csv
Outputs
  stable7_age_sex.md, stable7_age_sex_a.csv, stable7_age_sex_b.csv

Usage
  python tables/stable7_age_sex.py source-data
  python tables/stable7_age_sex.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'stable7_age_sex'
TITLE = 'Supplementary Table 7 | Age and sex sensitivity'
MARKER = {'homa_ir': 'HOMA-IR', 'fasting_insulin': 'Fasting insulin'}
MEASURE = {'mean_glucose': 'Average glucose', 'sd_glucose': 'SD', 'cv_glucose': 'CV', 'modd': 'MODD', 'mage': 'MAGE',
           'all_frequency': 'Spike frequency', 'all_size': 'Spike height', 'premeal_baseline': 'Pre-meal glucose',
           'peak_rise': 'Meal peak rise', 'iauc_2h': 'Meal 2-h iAUC'}
COMPARISON = {'unadjusted vs age+sex': 'unadjusted → + age, sex', 'mean glucose vs mean glucose+age+sex': 'same average glucose → + age, sex'}
SCOPE = {'all_primary': 'whole primary cohort', 'hba1c_normal_lt57': 'HbA1c <5.7%'}
A_COLS = ['n_without', 'estimate_without', 'ci_low_without', 'ci_high_without', 'estimate_with', 'ci_low_with', 'ci_high_with', 'change_points']


def make_source_data():
    a = pd.read_csv(OUTPUTS / 'metabolic/unadjusted_insulin_markers/age_sex_sensitivity.csv')
    a = a.loc[a.in_paper_set & a.marker.isin(list(MARKER)) & a.measure.isin(list(MEASURE))].copy()
    m = pd.read_csv(OUTPUTS / 'clinical/mage_standard/age_sex_sensitivity_mage_standard.csv')
    m = m.loc[m.measure.eq('mage') & m.marker.isin(list(MARKER))].set_index(['comparison', 'marker'])
    for i, r in a.loc[a.measure == 'mage'].iterrows():            # standard MAGE replaces the producer's MAGE rows
        a.loc[i, A_COLS] = m.loc[(r.comparison, r.marker), A_COLS].to_numpy()
    ta = pd.DataFrame(dict(comparison=a.comparison.map(COMPARISON), marker=a.marker.map(MARKER), measure=a.measure.map(MEASURE),
                           n_participants=a.n_without.astype(int), pct_per_sd_without_age_sex=a.estimate_without, ci_low_without=a.ci_low_without,
                           ci_high_without=a.ci_high_without, pct_per_sd_with_age_sex=a.estimate_with, ci_low_with=a.ci_low_with,
                           ci_high_with=a.ci_high_with, change_points=a.change_points))
    r = pd.read_csv(OUTPUTS / 'metabolic/hba1c_unadjusted_r2/age_sex_r2_sensitivity.csv')
    r = r.loc[r.scope.isin(list(SCOPE))]
    tb = pd.DataFrame(dict(outcome=r.outcome.map(MARKER), scope=r.scope.map(SCOPE), n_participants=r.n, r2_average_glucose_unadjusted=r.unadjusted_r2_cgm,
                           r2_hba1c_unadjusted=r.unadjusted_r2_hba1c, delta_r2_average_glucose_beyond_age_sex=r.delta_r2_cgm_beyond_age_sex,
                           delta_r2_average_glucose_ci_low=r.delta_r2_cgm_ci_low, delta_r2_average_glucose_ci_high=r.delta_r2_cgm_ci_high,
                           delta_r2_hba1c_beyond_age_sex=r.delta_r2_hba1c_beyond_age_sex, delta_r2_hba1c_ci_low=r.delta_r2_hba1c_ci_low,
                           delta_r2_hba1c_ci_high=r.delta_r2_hba1c_ci_high, difference=r.diff_cgm_minus_hba1c, difference_ci_low=r.diff_ci_low,
                           difference_ci_high=r.diff_ci_high))
    return [source_data.write('stable7a_age_sex', ta), source_data.write('stable7b_variance_explained', tb)]


def sg(v, d=1):
    return ('+' if v >= 0 else '−') + f'{abs(v):.{d}f}'


def draw(out_dir):
    a, b = source_data.read('stable7a_age_sex'), source_data.read('stable7b_variance_explained')
    ta = pd.DataFrame([(r.comparison, r.marker, r.measure, f'{r.n_participants:,}',
                        f'{sg(r.pct_per_sd_without_age_sex)} ({sg(r.ci_low_without)} to {sg(r.ci_high_without)})',
                        f'{sg(r.pct_per_sd_with_age_sex)} ({sg(r.ci_low_with)} to {sg(r.ci_high_with)})', sg(r.change_points, 2)) for r in a.itertuples()],
                      columns=['Comparison', 'Marker', 'Measure', 'n', '% per SD, without age/sex (95% CI)', '% per SD, with age/sex (95% CI)', 'change (points)'])
    tb = pd.DataFrame([(r.outcome, r.scope, f'{r.n_participants:,}', f'{100 * r.r2_average_glucose_unadjusted:.1f}', f'{100 * r.r2_hba1c_unadjusted:.1f}',
                        f'{100 * r.delta_r2_average_glucose_beyond_age_sex:.1f} ({100 * r.delta_r2_average_glucose_ci_low:.1f}–{100 * r.delta_r2_average_glucose_ci_high:.1f})',
                        f'{100 * r.delta_r2_hba1c_beyond_age_sex:.1f} ({100 * r.delta_r2_hba1c_ci_low:.1f}–{100 * r.delta_r2_hba1c_ci_high:.1f})',
                        f'{sg(100 * r.difference)} ({sg(100 * r.difference_ci_low)} to {sg(100 * r.difference_ci_high)})') for r in b.itertuples()],
                      columns=['Outcome', 'Scope', 'n', 'R² average glucose, unadjusted (%)', 'R² HbA1c, unadjusted (%)',
                               'ΔR² average glucose beyond age/sex (%)', 'ΔR² HbA1c beyond age/sex (%)', 'difference (points)'])
    return tables.write([('A Per-SD associations', ta), ('B Variance explained', tb)], out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
