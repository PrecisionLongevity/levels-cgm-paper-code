"""Supplementary Table 4: test-retest reliability of laboratory measures.

First pair of draws 14 to 90 days apart per participant, among participants never in the diabetes range; one-way
ICC(1) and Pearson r.

Inputs
  outputs/clinical/lab_retest.json    (analysis/clinical/clinical_summaries.py)
Source data
  source_data/stable4_lab_retest.csv
Outputs
  stable4_lab_retest.md / .csv

Usage
  python tables/stable4_lab_retest.py source-data
  python tables/stable4_lab_retest.py draw
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'stable4_lab_retest'
TITLE = 'Supplementary Table 4 | Test-retest reliability of laboratory measures'
MEASURES = [('HbA1c', 'a1c'), ('Fasting insulin (log)', 'fasting_insulin_log'), ('HOMA-IR (log)', 'homa_ir_log'), ('Laboratory fasting glucose', 'fasting_glucose')]


def make_source_data():
    rt = json.loads((OUTPUTS / 'clinical/lab_retest.json').read_text())
    df = pd.DataFrame([dict(measure=lab, n_pairs=rt[k]['n_pairs'], median_days_between_draws=rt[k]['median_gap_days'], icc1=rt[k]['icc1'],
                            pearson_r=rt[k]['r']) for lab, k in MEASURES])
    return [source_data.write(NAME, df)]


def draw(out_dir):
    d = source_data.read(NAME)
    t = pd.DataFrame([(r.measure, int(r.n_pairs), f'{r.median_days_between_draws:.0f}', f'{r.icc1:.2f}', f'{r.pearson_r:.2f}') for r in d.itertuples()],
                     columns=['Measure', 'Pairs', 'Median days between draws', 'ICC(1)', 'Pearson r'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
