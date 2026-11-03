"""Table 1: characteristics of the primary laboratory cohort.

Primary cohort: participants with an anchor HbA1c and at least 14 dense CGM days within ±90 days, without an observed
diabetes-range laboratory value. Laboratory values are those nearest the anchor HbA1c draw.

Inputs
  outputs/canonical_panel/canonical_lab_cgm_panel.csv    one row per participant (analysis/canonical_panel/build_panel.py)
Source data
  source_data/table1_cohort.csv    for each characteristic: statistic, value, quartiles and n
Outputs
  table1_cohort.md / .csv

Usage
  python tables/table1_cohort.py source-data
  python tables/table1_cohort.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data, tables  # noqa: E402
from cgmpaper.paths import OUTPUTS, TABLES_OUT  # noqa: E402

NAME = 'table1_cohort'
TITLE = 'Table 1 | Characteristics of the primary laboratory cohort'
MEDIANS = [('BMI, kg/m²', 'bmi', '{:.1f}'), ('HbA1c, %', 'a1c', '{:.1f}'), ('CGM average glucose, mg/dL', 'mean_glucose', '{:.0f}'),
           ('Dense CGM days within ±90 d', 'n_dense_days', '{:.0f}'), ('Fasting insulin, µIU/mL', 'fasting_insulin', '{:.1f}'),
           ('HOMA-IR', 'homa_ir', '{:.2f}'), ('Fasting glucose, mg/dL', 'fasting_glucose', '{:.0f}'), ('Triglycerides, mg/dL', 'triglycerides', '{:.0f}')]
AGE_LABEL = {20: '≤20s', 30: '30s', 40: '40s', 50: '50s', 60: '60s', 70: '≥70s'}


def make_source_data():
    p = pd.read_csv(OUTPUTS / 'canonical_panel/canonical_lab_cgm_panel.csv', low_memory=False)
    s = p.loc[p.no_diabetes_range.astype(bool)]
    rows = [dict(characteristic='Participants', statistic='count', value=len(s))]
    ag = s.age_bucket.dropna()
    for k, v in ag.clip(lower=20, upper=70).astype(int).value_counts(normalize=True).sort_index().items():
        rows.append(dict(characteristic=f'Age decade: {AGE_LABEL[k]}', statistic='percent', value=100 * v, n=len(ag)))
    sx = s.sex.dropna()
    rows.append(dict(characteristic='Female', statistic='percent', value=100 * (sx == 'female').mean(), n=len(sx)))
    for lab, col, _ in MEDIANS:
        x = s[col].dropna()
        rows.append(dict(characteristic=lab, statistic='median', value=x.median(), q25=x.quantile(.25), q75=x.quantile(.75), n=len(x)))
    rows.append(dict(characteristic='HbA1c <5.7%', statistic='percent', value=100 * (s.a1c < 5.7).mean(), n=int(s.a1c.notna().sum())))
    rows.append(dict(characteristic='Average glucose ≥110 mg/dL', statistic='percent', value=100 * (s.mean_glucose >= 110).mean(),
                     n=int(s.mean_glucose.notna().sum())))
    df = pd.DataFrame(rows)[['characteristic', 'statistic', 'value', 'q25', 'q75', 'n']]
    df['n'] = df.n.astype('Int64')
    return [source_data.write(NAME, df)]


def draw(out_dir):
    d = source_data.read(NAME).set_index('characteristic')
    n = lambda k: f'{int(d.loc[k, "n"]):,}'  # noqa: E731
    rows = [('Participants', f"{int(d.loc['Participants', 'value']):,}", '')]
    ages = [k for k in d.index if k.startswith('Age decade: ')]
    rows.append(('Age decade, %', '', n(ages[0])))
    rows += [(k.replace('Age decade: ', ''), f"{d.loc[k, 'value']:.1f}", '') for k in ages]
    rows.append(('Female, %', f"{d.loc['Female', 'value']:.1f}", n('Female')))
    for lab, _, fmt in MEDIANS:
        r = d.loc[lab]
        rows.append((f'{lab}, median (IQR)', f'{fmt.format(r.value)} ({fmt.format(r.q25)}–{fmt.format(r.q75)})', n(lab)))
    rows.append(('HbA1c <5.7%, %', f"{d.loc['HbA1c <5.7%', 'value']:.1f}", n('HbA1c <5.7%')))
    rows.append(('Average glucose ≥110 mg/dL, %', f"{d.loc['Average glucose ≥110 mg/dL', 'value']:.1f}", n('Average glucose ≥110 mg/dL')))
    return tables.write(pd.DataFrame(rows, columns=['Characteristic', 'Value', 'n']), out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
