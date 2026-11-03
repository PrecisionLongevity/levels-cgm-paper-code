"""Supplementary Table 2: spike width replicated in the width of responses to logged meals.

Participants with both a CGM spike width and a width of responses to logged meals. % difference in fasting insulin and
HOMA-IR per SD of each width at the same average glucose (log outcome, HC3 95% CI), and with both widths in one model.

Inputs
  outputs/clinical/spike_width/width_meal_anchored.csv    (analysis/clinical/spike_width.py)
Source data
  source_data/stable2_meal_width.csv
Outputs
  stable2_meal_width.md / .csv

Usage
  python tables/stable2_meal_width.py source-data
  python tables/stable2_meal_width.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'stable2_meal_width'
TITLE = 'Supplementary Table 2 | Spike width replicated in responses to logged meals, in participants with both widths'
MEASURES = [('spike_width', 'CGM spike width'), ('meal_width', 'Width of responses to logged meals'),
            ('meal_width_adj', 'Width of responses to logged meals, adjusted for meal content and context')]
ROWS = [(m, 'both widths available', 'same average glucose') for m, _ in MEASURES] + \
       [(m, 'both widths in one model, average glucose', 'same average glucose, both widths in one model') for m, _ in MEASURES[:2]]


def make_source_data():
    w = pd.read_csv(OUTPUTS / 'clinical/spike_width/width_meal_anchored.csv')
    rows = []
    for m, smp, comp in ROWS:
        for mk in ('fasting_insulin', 'homa_ir'):
            r = w.loc[(w.marker == mk) & (w.measure == m) & (w['sample'] == smp)].iloc[0]
            rows.append(dict(width_measure=dict(MEASURES)[m], compared_at_the=comp, marker=mk, n_participants=int(r.n), pct_per_sd=r.pct,
                             ci_low=r.lo, ci_high=r.hi))
    return [source_data.write(NAME, pd.DataFrame(rows))]


def draw(out_dir):
    d = source_data.read(NAME)
    pc = lambda r: f'{r.pct_per_sd:.1f} ({r.ci_low:.1f} to {r.ci_high:.1f})'  # noqa: E731
    rows = []
    for (wm, comp), x in d.groupby(['width_measure', 'compared_at_the'], sort=False):
        f, h = x[x.marker == 'fasting_insulin'].iloc[0], x[x.marker == 'homa_ir'].iloc[0]
        rows.append((wm, comp, f'{int(f.n_participants):,}', pc(f), f'{int(h.n_participants):,}', pc(h)))
    t = pd.DataFrame(rows, columns=['Width measure', 'Compared at the', 'Participants', 'Fasting insulin, % per SD (95% CI)',
                                    'Participants with HOMA-IR', 'HOMA-IR, % per SD (95% CI)'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
