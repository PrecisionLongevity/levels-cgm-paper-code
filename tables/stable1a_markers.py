"""Supplementary Table 1a: average glucose and HbA1c against other laboratory markers of metabolic health.

For each marker (log-transformed; value nearest the anchor HbA1c draw), in the same participants: variance explained by
average glucose and by HbA1c alone, their difference (paired participant bootstrap 95% CI), and % difference per SD of
each (OLS with age decade and sex; HC3 95% CI), also with BMI.

Inputs
  outputs/clinical/markers/markers_avg_vs_a1c.csv    (analysis/clinical/markers.py)
Source data
  source_data/stable1a_markers.csv
Outputs
  stable1a_markers.md / .csv

Usage
  python tables/stable1a_markers.py source-data
  python tables/stable1a_markers.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'stable1a_markers'
TITLE = 'Supplementary Table 1a | Average glucose and HbA1c against other laboratory markers of metabolic health'
COLS = {'label': 'marker', 'n': 'n_participants', 'r2_avg': 'r2_average_glucose', 'r2_a1c': 'r2_hba1c', 'r2_diff': 'r2_difference',
        'r2_diff_lo': 'r2_difference_ci_low', 'r2_diff_hi': 'r2_difference_ci_high', 'avg_pct': 'average_glucose_pct_per_sd',
        'avg_lo': 'average_glucose_ci_low', 'avg_hi': 'average_glucose_ci_high', 'a1c_pct': 'hba1c_pct_per_sd', 'a1c_lo': 'hba1c_ci_low',
        'a1c_hi': 'hba1c_ci_high', 'avg_bmi_pct': 'average_glucose_with_bmi_pct_per_sd', 'avg_bmi_lo': 'average_glucose_with_bmi_ci_low',
        'avg_bmi_hi': 'average_glucose_with_bmi_ci_high', 'n_bmi': 'n_participants_with_bmi'}


def make_source_data():
    t = pd.read_csv(OUTPUTS / 'clinical/markers/markers_avg_vs_a1c.csv')
    return [source_data.write(NAME, t[list(COLS)].rename(columns=COLS))]


def draw(out_dir):
    d = source_data.read(NAME)
    f1 = lambda a, b, c: f'{a:.1f} ({b:.1f} to {c:.1f})'  # noqa: E731
    t = pd.DataFrame([(r.marker, f'{r.n_participants:,}', f'{r.r2_average_glucose:.3f}', f'{r.r2_hba1c:.3f}',
                       f'{r.r2_difference:.3f} ({r.r2_difference_ci_low:.3f} to {r.r2_difference_ci_high:.3f})',
                       f1(r.average_glucose_pct_per_sd, r.average_glucose_ci_low, r.average_glucose_ci_high),
                       f1(r.hba1c_pct_per_sd, r.hba1c_ci_low, r.hba1c_ci_high),
                       f1(r.average_glucose_with_bmi_pct_per_sd, r.average_glucose_with_bmi_ci_low, r.average_glucose_with_bmi_ci_high),
                       f'{r.n_participants_with_bmi:,}') for r in d.itertuples()],
                     columns=['Marker', 'Participants', 'R² average glucose', 'R² HbA1c', 'Difference (95% CI)',
                              'Average glucose, % per SD (95% CI)', 'HbA1c, % per SD (95% CI)', 'Average glucose + BMI, % per SD (95% CI)',
                              'Participants with BMI'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
