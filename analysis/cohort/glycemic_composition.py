#!/usr/bin/env python3
"""Glycemic composition of the consenting roster by laboratory range.

Denominator: the consenting noninternal roster (cohort_participants.parquet) with at least one
accepted HbA1c or fasting glucose after the shared laboratory normalization.
Ranges (ADA laboratory thresholds, highest accepted value per person; single measurements, not
diagnoses): diabetes range HbA1c >=6.5% or fasting glucose >=126 mg/dL; prediabetes range
HbA1c 5.7-6.4% or fasting glucose 100-125 mg/dL; otherwise normal. HbA1c rounded decimal half-up.
Also reported: HbA1c-only classification, and composition by age decade, sex and CGM availability.

Inputs: labs table via LEVELS_DB; OUTPUTS/cohort/cohort_participants.parquet (analysis/cohort/cohort_reference.py);
OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py).

Outputs: OUTPUTS/cohort/glycemic_composition/glycemic_composition.csv, summary.json,
glycemic_composition.metadata.json.

Usage: python analysis/cohort/glycemic_composition.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from cgmpaper.lab_normalization import normalize_labs, round_a1c  # noqa: E402

VERSION = 'glycemic-composition'
PUB = OUTPUTS
ROSTER = PUB / 'cohort/cohort_participants.parquet'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
OUT = PUB / 'cohort/glycemic_composition'
DB = database_path()
ORDER = ['normal', 'prediabetes range', 'diabetes range']


def tab(frame, col, by=None):
    if by is None:
        c = frame[col].value_counts().reindex(ORDER, fill_value=0)
        return [dict(group='all', range=r, n=int(c[r]), denominator=int(c.sum()), pct=float(100 * c[r] / c.sum())) for r in ORDER]
    rows = []
    for g, s in frame.groupby(by, observed=True):
        c = s[col].value_counts().reindex(ORDER, fill_value=0)
        rows += [dict(group=f'{by}={g}', range=r, n=int(c[r]), denominator=int(c.sum()), pct=float(100 * c[r] / c.sum())) for r in ORDER]
    return rows


def main():
    t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(ROSTER, columns=['USER_ID', 'sex', 'age_bucket', 'glucose_rows'])
    with duckdb.connect(str(DB), read_only=True) as con:
        raw = con.execute("SELECT USER_ID, TYPE, DRAW_DATE, VALUE, UNIT FROM labs WHERE TYPE IN ('a1c','fasting_glucose')").fetchdf()
    daily, _, _ = normalize_labs(raw)
    daily = daily.loc[daily.USER_ID.isin(R.USER_ID)]
    a = daily.loc[daily.TYPE == 'a1c'].assign(r=lambda d: round_a1c(d.value.to_numpy())).groupby('USER_ID').r.max()
    g = daily.loc[daily.TYPE == 'fasting_glucose'].groupby('USER_ID').value.max()
    L = pd.DataFrame({'a1c_max': a, 'fpg_max': g})
    dm = (L.a1c_max >= 6.5) | (L.fpg_max >= 126)
    pre = (L.a1c_max >= 5.7) | ((L.fpg_max >= 100) & (L.fpg_max < 126))
    L['ada_range'] = np.where(dm, 'diabetes range', np.where(pre, 'prediabetes range', 'normal'))
    L['hba1c_range'] = np.select([L.a1c_max >= 6.5, L.a1c_max >= 5.7, L.a1c_max.notna()], ORDER[::-1], None)
    L = L.join(R.set_index('USER_ID'), how='left')
    L['has_cgm'] = L.glucose_rows.fillna(0) > 0
    panel = pd.read_csv(PANEL, usecols=['USER_ID', 'no_diabetes_range'], low_memory=False)
    flagged = set(panel.loc[panel.no_diabetes_range == False, 'USER_ID'])  # noqa: E712
    captured = int(L.loc[L.index.isin(flagged), 'ada_range'].eq('diabetes range').sum())
    guards = dict(roster=len(R), with_glycemic_lab=int(len(L)), panel_flagged=len(flagged), panel_flagged_captured=captured)
    rows = tab(L, 'ada_range') + tab(L, 'ada_range', 'age_bucket') + tab(L, 'ada_range', 'sex') + tab(L, 'ada_range', 'has_cgm')
    T = pd.DataFrame(rows).assign(classification='ada_hba1c_or_fpg')
    H = pd.DataFrame(tab(L.loc[L.hba1c_range.notna()], 'hba1c_range')).assign(classification='hba1c_only')
    out = pd.concat([T, H], ignore_index=True); out.to_csv(OUT / 'glycemic_composition.csv', index=False)
    summary = dict(guards=guards, n_hba1c=int(L.a1c_max.notna().sum()), n_fpg=int(L.fpg_max.notna().sum()),
                   n_with_cgm=int(L.has_cgm.sum()),
                   headline={r['range']: dict(n=r['n'], pct=r['pct']) for r in tab(L, 'ada_range')},
                   headline_hba1c_only={r['range']: dict(n=r['n'], pct=r['pct']) for r in tab(L.loc[L.hba1c_range.notna()], 'hba1c_range')},
                   headline_with_cgm={r['range']: dict(n=r['n'], pct=r['pct']) for r in tab(L.loc[L.has_cgm], 'ada_range')})
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    meta = dict(analysis='glycemic_composition', version=VERSION, guards=guards,
                definition='highest accepted HbA1c (half-up) and fasting glucose per participant; ADA laboratory ranges; not diagnoses',
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'glycemic_composition.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
