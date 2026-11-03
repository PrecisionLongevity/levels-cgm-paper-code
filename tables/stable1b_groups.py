"""Supplementary Table 1b: laboratory markers in the HbA1c and average-glucose groups.

Geometric mean (n) of each marker in the four groups (CGM elevated: average glucose >=110 mg/dL; HbA1c elevated: >=5.7%),
the CGM-elevated-only vs HbA1c-elevated-only difference (OLS of the log marker on group, unadjusted, HC3 95% CI), and the
share above the fasting-insulin and HOMA-IR thresholds (Wilson 95% CI).

Inputs
  outputs/clinical/markers/discordance_markers.csv, discordance_high_share.csv   (analysis/clinical/markers.py)
Source data
  source_data/stable1b_groups.csv
Outputs
  stable1b_groups.md / .csv

Usage
  python tables/stable1b_groups.py source-data
  python tables/stable1b_groups.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'stable1b_groups'
TITLE = 'Supplementary Table 1b | Laboratory markers in the HbA1c and average glucose groups'
GROUPS = [('both normal', 'Both normal'), ('CGM elevated only', 'CGM elevated only'), ('HbA1c elevated only', 'HbA1c elevated only'),
          ('both high', 'Both elevated')]
LABEL = {'fasting_insulin': 'Fasting insulin', 'homa_ir': 'HOMA-IR', 'triglycerides': 'Triglycerides', 'hscrp': 'hsCRP'}
THRESHOLDS = [('fasting_insulin', 'Fasting insulin >=10 µIU/mL, %'), ('homa_ir', 'HOMA-IR >=2.5, %')]


def make_source_data():
    ds = pd.read_csv(OUTPUTS / 'clinical/markers/discordance_markers.csv')
    rows = []
    for _, r in ds.iterrows():
        for g, lab in GROUPS:
            rows.append(dict(marker=LABEL[r['marker']], quantity='geometric mean', group=lab, value=r[f'gm_{g}'], n_participants=int(r[f'n_{g}'])))
        rows.append(dict(marker=LABEL[r['marker']], quantity='% difference, CGM elevated only vs HbA1c elevated only', value=r['cgm_vs_a1c_only_pct'],
                         ci_low=r['lo'], ci_high=r['hi']))
    hs = pd.read_csv(OUTPUTS / 'clinical/markers/discordance_high_share.csv')
    for mk, lab in THRESHOLDS:
        h = hs.query('marker == @mk').set_index('group')
        for g, glab in GROUPS:
            rows.append(dict(marker=lab, quantity='percent above threshold', group=glab, value=h.loc[g, 'pct'], ci_low=h.loc[g, 'lo'], ci_high=h.loc[g, 'hi']))
    df = pd.DataFrame(rows)[['marker', 'quantity', 'group', 'value', 'ci_low', 'ci_high', 'n_participants']]
    df['n_participants'] = df.n_participants.astype('Int64')
    return [source_data.write(NAME, df)]


def draw(out_dir):
    d = source_data.read(NAME)
    rows = []
    for mk in d.marker.unique():
        x = d[d.marker == mk]
        if (x.quantity == 'geometric mean').any():
            cells = [f'{r.value:.2f} ({int(r.n_participants):,})' for r in x[x.quantity == 'geometric mean'].itertuples()]
            c = x[x.quantity.str.startswith('% difference')].iloc[0]
            rows.append([mk, *cells, f'{c.value:.1f} ({c.ci_low:.1f} to {c.ci_high:.1f})'])
        else:
            rows.append([mk, *[f'{r.value:.1f} ({r.ci_low:.1f} to {r.ci_high:.1f})' for r in x.itertuples()], ''])
    t = pd.DataFrame(rows, columns=['Marker', 'Both normal, GM (n)', 'CGM elevated only, GM (n)', 'HbA1c elevated only, GM (n)',
                                    'Both elevated, GM (n)', 'CGM-only vs HbA1c-only, % (95% CI)'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
