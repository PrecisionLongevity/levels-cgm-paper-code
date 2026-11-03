"""Supplementary Table 3: HbA1c comparison by anchor year and with the sample restricted on average glucose.

Variance in log fasting insulin explained by average glucose and by HbA1c, and discrimination (AUC) of fasting insulin
>=10 µIU/mL, by year of the anchor HbA1c draw and among participants with average glucose <110 mg/dL (paired bootstrap
95% CIs).

Inputs
  outputs/clinical/hba1c_sensitivity.json    (analysis/clinical/clinical_summaries.py)
Source data
  source_data/stable3_hba1c_sensitivity.csv
Outputs
  stable3_hba1c_sensitivity.md / .csv

Usage
  python tables/stable3_hba1c_sensitivity.py source-data
  python tables/stable3_hba1c_sensitivity.py draw
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'stable3_hba1c_sensitivity'
TITLE = 'Supplementary Table 3 | HbA1c comparison by anchor year and with the sample restricted on average glucose'


def make_source_data():
    hs = json.loads((OUTPUTS / 'clinical/hba1c_sensitivity.json').read_text())
    rows = []
    for y, b in hs['by_year'].items():
        rows.append(dict(sample=f'Anchor year {y}', n_participants=b['n'], r2_average_glucose=b['r2_avg'], r2_hba1c=b['r2_a1c'],
                         r2_difference=b['diff'], r2_difference_ci_low=b['ci'][0], r2_difference_ci_high=b['ci'][1], auc_average_glucose=b['auc_avg'],
                         auc_hba1c=b['auc_a1c'], auc_difference_ci_low=b['auc_diff_ci'][0], auc_difference_ci_high=b['auc_diff_ci'][1]))
    b = hs['avg_lt_110']
    rows.append(dict(sample='Average glucose <110 mg/dL', n_participants=b['n'], r2_average_glucose=b['r2_avg'], r2_hba1c=b['r2_a1c'],
                     r2_difference=b['diff'], r2_difference_ci_low=b['ci'][0], r2_difference_ci_high=b['ci'][1]))
    return [source_data.write(NAME, pd.DataFrame(rows))]


def draw(out_dir):
    d = source_data.read(NAME)
    rows = []
    for r in d.itertuples():
        auc = pd.notna(r.auc_average_glucose)
        rows.append((r.sample, int(r.n_participants), f'{r.r2_average_glucose:.3f}', f'{r.r2_hba1c:.3f}',
                     f'{r.r2_difference:.3f} ({r.r2_difference_ci_low:.3f} to {r.r2_difference_ci_high:.3f})',
                     f'{r.auc_average_glucose:.2f}' if auc else '', f'{r.auc_hba1c:.2f}' if auc else '',
                     f'{r.auc_average_glucose - r.auc_hba1c:.2f} ({r.auc_difference_ci_low:.2f} to {r.auc_difference_ci_high:.2f})' if auc else ''))
    t = pd.DataFrame(rows, columns=['Sample', 'Participants', 'R² average glucose', 'R² HbA1c', 'Difference (95% CI)', 'AUC average glucose',
                                    'AUC HbA1c', 'AUC difference (95% CI)'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
