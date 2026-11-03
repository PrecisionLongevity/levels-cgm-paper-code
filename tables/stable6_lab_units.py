"""Supplementary Table 6: laboratory units, conversions and accepted ranges.

The values come from the laboratory normalization used by every analysis (cgmpaper/lab_normalization.py): the canonical
unit of each analyte (CANONICAL_UNITS), the accepted range after conversion (BOUNDS) and the accepted alternate units with
their conversion (UNIT_TRANSFORMS: normalized value = reported value x multiplier + offset). No analysis output and no
participant data are read.

Source data
  source_data/stable6_lab_units.csv     one row per analyte and accepted alternate unit
Outputs
  stable6_lab_units.md / .csv

Usage
  python tables/stable6_lab_units.py source-data
  python tables/stable6_lab_units.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, lab_normalization as labs, source_data, tables  # noqa: E402
from cgmpaper.paths import TABLES_OUT  # noqa: E402

NAME = 'stable6_lab_units'
TITLE = 'Supplementary Table 6 | Laboratory units, conversions, and accepted ranges'
# analyte key, printed name, printed canonical unit
ANALYTES = [('a1c', 'HbA1c', '%'), ('fasting_glucose', 'Fasting glucose', 'mg/dL'), ('fasting_insulin', 'Fasting insulin', 'µIU/mL'),
            ('triglycerides', 'Triglycerides', 'mg/dL'), ('hdl_cholesterol', 'HDL cholesterol', 'mg/dL'),
            ('ldl_cholesterol', 'LDL cholesterol', 'mg/dL'), ('apo_b', 'ApoB', 'mg/dL'), ('hscrp', 'hsCRP', 'mg/L'),
            ('uric_acid', 'Uric acid', 'mg/dL')]
# printed form (and printing order) of the normalized unit keys (case, spacing and the micro sign are normalized
# before matching)
UNIT_LABEL = {'mmol/mol': 'IFCC mmol/mol', 'mmol/l': 'mmol/L', 'g/l': 'g/L', 'mg/dl': 'mg/dL', 'uu/ml': 'µU/mL',
              'u[iu]/ml': 'u[IU]/mL', 'mciu/ml': 'mcIU/mL', 'mu/l': 'mU/L'}
# spellings of the canonical unit itself (not printed as alternates)
SAME_AS_CANONICAL = {'%', '%oftotalhgb', 'mg/dl(calc)', 'uiu/ml'}


def make_source_data():
    rows = []
    for key, name, unit in ANALYTES:
        lo, hi = labs.BOUNDS[key]
        alternates = [(u, m, o) for u, (m, o) in labs.UNIT_TRANSFORMS[key].items()
                      if u not in SAME_AS_CANONICAL and u != labs.CANONICAL_UNITS[key].lower()]
        alternates.sort(key=lambda a: list(UNIT_LABEL).index(a[0]))
        for u, m, o in alternates or [(None, None, None)]:
            rows.append(dict(analyte=name, unit=unit, range_low=lo, range_high=hi, alternate_unit=UNIT_LABEL.get(u, u) if u else None,
                             multiplier=m, offset=o))
    return [source_data.write(NAME, pd.DataFrame(rows))]


def num(x):
    return f'{x:,g}'


def draw(out_dir):
    d = source_data.read(NAME)
    out = []
    for name, g in d.groupby('analyte', sort=False):
        r = g.iloc[0]
        alt = g.dropna(subset=['alternate_unit'])
        if alt.empty:
            conv = 'None'
        elif (alt.multiplier == 1).all() and (alt.offset == 0).all():
            units = list(alt.alternate_unit)
            conv = ', '.join(units[:-1]) + f', and {units[-1]} are equivalent'
        else:
            conv = '; '.join(f'{a.alternate_unit} ×{a.multiplier:g}' + (f' +{a.offset:g}' if a.offset else '') for a in alt.itertuples())
        out.append((name, r.unit, f'{num(r.range_low)}–{num(r.range_high)}', conv))
    t = pd.DataFrame(out, columns=['Analyte', 'Unit', 'Accepted range', 'Accepted alternate units and conversion'])
    return tables.write(t, out_dir, NAME, TITLE)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, TABLES_OUT)
