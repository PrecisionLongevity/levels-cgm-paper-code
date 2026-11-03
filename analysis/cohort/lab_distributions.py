#!/usr/bin/env python3
"""Aggregate-only LDL, ApoB and creatinine distributions in the primary cohort.

This producer adds LDL, ApoB and creatinine distributions to the set already covered by
analysis/cohort/multimodal_distributions.py (HbA1c, fasting insulin, triglycerides, BMI, steps, sleep and exercise
days), using the same population, output schema and aggregate-only rule.

Population. Primary canonical lab-CGM panel (cgmpaper.publication_panels.load_panel, cohort 'no_diabetes_range',
symmetric window): first HbA1c with >= 14 dense CGM days within +/-90 d and no observed diabetes-range laboratory
value. Same panel and call as analysis/cohort/multimodal_distributions.py.

Analytes
  LDL cholesterol, ApoB   the canonical-draw values already in the panel (canonical-labs normalization; nearest
                          normalized daily value within +/-7 d of the HbA1c anchor, ties -> earlier date).
  Creatinine              not a canonical analyte, so normalized here with the canonical rules made explicit:
                          non-internal accounts; unit whitelist after the canonical unit cleaning (lower case, no
                          whitespace, micro sign -> u): 'mg/dl' x 1, 'umol/l' and 'micromol/l' / 88.42; malformed vendor
                          unit text is NOT repaired (excluded and counted); validity 0.2-15.0 mg/dL; daily median per
                          participant-date; nearest daily value within +/-7 d of the HbA1c anchor, ties -> earlier
                          (the canonical builder's rule, re-implemented and checked internally for consistency).
Display ranges and bin widths are fixed a priori (not data-derived).

Internal consistency checks (run before anything is written)
  - the panel's LDL and ApoB offsets lie within +/-7 d of the anchor for every non-missing value.
  - the nearest-within-7-d matcher used for creatinine, applied to canonically normalized LDL and ApoB daily values
    (cgmpaper.lab_reference.load_normalized_labs), reproduces the panel's ldl_cholesterol and apo_b values and
    offsets exactly for every participant.
  - every creatinine row is accounted for (accepted + unknown unit + invalid/out of range = input rows).
  - histogram counts + below + above = n for every measure.

Inputs
  Database tables (via LEVELS_DB): labs, users.
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py).
  OUTPUTS/cohort/stream_overlap_regions/summary.json (analysis/cohort/stream_overlap.py).
  OUTPUTS/cohort/multimodal_distributions/multimodal_distributions.metadata.json (analysis/cohort/multimodal_distributions.py).

Outputs (OUTPUTS/cohort/lab_distributions_extended/, aggregate only; no participant identifier or participant-level
row is written)
  distribution_summary.csv, distribution_histograms.csv, summary.json, lab_distributions_extended.metadata.json.

Usage
  python analysis/cohort/lab_distributions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from cgmpaper.publication_panels import load_panel  # noqa: E402
from cgmpaper.lab_reference import load_normalized_labs  # noqa: E402

VERSION = 'lab-distributions-extended'
PUB = OUTPUTS
CC = PUB / 'cohort'
REF_META = CC / 'reference_analysis.metadata.json'
S_STREAM_OVERLAP = CC / 'stream_overlap_regions'
S_MULTIMODAL = CC / 'multimodal_distributions'
S_MULTIMODAL_SCRIPT = ROOT / 'analysis/cohort/multimodal_distributions.py'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
DB = database_path()
OUT = CC / 'lab_distributions_extended'
BAND_DAYS = 7
CREAT_UNITS = {'mg/dl': 1.0, 'umol/l': 1.0 / 88.42, 'micromol/l': 1.0 / 88.42}
CREAT_BOUNDS = (0.2, 15.0)
# measure key -> (label, unit, population, display lo, display hi, bin width); fixed a priori
MEASURES = {
    'ldl_cholesterol': ('LDL cholesterol', 'mg/dL', 'primary', 0.0, 250.0, 5.0),
    'apo_b': ('ApoB', 'mg/dL', 'primary', 20.0, 200.0, 5.0),
    'creatinine': ('Creatinine', 'mg/dL', 'primary', 0.4, 1.6, 0.05),
}
POPULATIONS = {'primary': 'primary canonical lab-CGM panel (no observed diabetes-range laboratory value), nearest draw within +/-7 d of the HbA1c anchor'}
EPOCH = pd.Timestamp('1970-01-01')


def fail(msg):
    raise SystemExit('GUARD FAILED: ' + msg)


def same(a, b):
    a = pd.to_numeric(a, errors='coerce').to_numpy(float); b = pd.to_numeric(b, errors='coerce').to_numpy(float)
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


def day_number(values):
    return ((pd.to_datetime(values) - EPOCH) / pd.Timedelta(days=1)).round().astype('int64')


def nearest(anchors, daily, column, band=BAND_DAYS):
    """Non-null daily value nearest the anchor within +/-band days, ties -> earlier date (canonical builder rule)."""
    d = daily[['USER_ID', 'DRAW_DATE', column]].dropna(subset=[column]).copy()
    d['DRAW_DATE'] = pd.to_datetime(d.DRAW_DATE).dt.normalize()
    if d.duplicated(['USER_ID', 'DRAW_DATE']).any():
        fail('daily input is not unique per participant-date: ' + column)
    m = anchors[['USER_ID', 'anchor_date']].merge(d, on='USER_ID', how='inner', validate='one_to_many')
    m['offset_days'] = day_number(m.DRAW_DATE) - day_number(m.anchor_date)
    m = m.loc[m.offset_days.abs() <= band]
    m = m.assign(abs_offset=m.offset_days.abs()).sort_values(['USER_ID', 'abs_offset', 'DRAW_DATE'], kind='stable')
    pick = m.drop_duplicates('USER_ID')[['USER_ID', column, 'offset_days']]
    return anchors[['USER_ID']].merge(pick, on='USER_ID', how='left', validate='one_to_one')


def normalized_creatinine(con):
    raw = con.execute("""SELECT l.USER_ID, l.DRAW_DATE, l.VALUE, l.UNIT FROM labs l JOIN users u ON u.USER_ID=l.USER_ID
                         WHERE u.IS_INTERNAL_USER=FALSE AND l.TYPE='creatinine'""").fetchdf()
    d = raw.copy()
    d['DRAW_DATE'] = pd.to_datetime(d.DRAW_DATE, errors='coerce').dt.normalize()
    d['VALUE'] = pd.to_numeric(d.VALUE, errors='coerce').astype(float)
    unit = (d.UNIT.fillna('').astype(str).str.strip().str.lower().str.replace('µ', 'u', regex=False)
            .str.replace('μ', 'u', regex=False).str.replace(r'\s+', '', regex=True))
    mult = unit.map(CREAT_UNITS)
    unit_ok = mult.notna()
    d['VALUE'] = d.VALUE * mult
    valid = (d.USER_ID.notna() & d.USER_ID.astype(str).str.strip().ne('') & d.DRAW_DATE.notna() & np.isfinite(d.VALUE)
             & d.VALUE.between(*CREAT_BOUNDS))
    audit = dict(input_rows=int(len(d)), unknown_or_malformed_unit_rows=int((~unit_ok).sum()),
                 invalid_or_out_of_range_rows=int((unit_ok & ~valid).sum()), accepted_rows=int((unit_ok & valid).sum()),
                 converted_accepted_rows=int((unit_ok & valid & mult.ne(1.0)).sum()),
                 unit_whitelist=sorted(CREAT_UNITS), bounds_mg_dl=list(CREAT_BOUNDS), canonical_unit='mg/dL')
    if audit['accepted_rows'] + audit['unknown_or_malformed_unit_rows'] + audit['invalid_or_out_of_range_rows'] != audit['input_rows']:
        fail('creatinine rows not accounted for')
    daily = d.loc[unit_ok & valid].groupby(['USER_ID', 'DRAW_DATE'], as_index=False).VALUE.median().rename(columns={'VALUE': 'creatinine'})
    audit['participant_days'] = int(len(daily)); audit['participants_any'] = int(daily.USER_ID.nunique())
    return daily, audit


def summarize(key, values):
    label, unit, pop, lo, hi, width = MEASURES[key]
    v = pd.to_numeric(values, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        fail('no values for ' + key)
    edges = np.round(np.arange(lo, hi + width / 2, width), 6)
    counts, _ = np.histogram(v.loc[v.between(lo, hi)], bins=edges)
    below, above = int(v.lt(lo).sum()), int(v.gt(hi).sum())
    if int(counts.sum()) + below + above != len(v):
        fail('histogram does not partition ' + key)
    stats = dict(measure=key, label=label, unit=unit, population=pop, n=int(len(v)), mean=float(v.mean()),
                 sd=float(v.std(ddof=1)), median=float(v.median()), q25=float(v.quantile(.25)), q75=float(v.quantile(.75)),
                 display_lo=lo, display_hi=hi, bin_width=width, n_below_display=below, n_above_display=above)
    return stats, pd.DataFrame(dict(measure=key, bin_lo=edges[:-1], bin_hi=edges[1:], count=counts.astype(int)))


def main():
    guards = {}
    P = load_panel(PANEL, cohort='no_diabetes_range', required=['ldl_cholesterol', 'apo_b', 'anchor_date'], expected_window='symmetric')
    guards['G1_dependencies'] = dict(primary_n=len(P))
    P['anchor_date'] = pd.to_datetime(P.anchor_date).dt.normalize()
    # G2 offsets
    for c in ('ldl_cholesterol', 'apo_b'):
        off = pd.to_numeric(P.loc[P[c].notna(), f'{c}_offset_days'], errors='coerce')
        if off.isna().any() or off.abs().gt(BAND_DAYS).any():
            fail(f'{c} offsets outside +/-{BAND_DAYS} d')
    guards['G2_panel_offsets_within_7d'] = True
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute("SET threads=4; SET memory_limit='4GB'")
        daily, _, _ = load_normalized_labs(con, types=['ldl_cholesterol', 'apo_b'])
        creat_daily, creat_audit = normalized_creatinine(con)
    # G3 matcher reproduces panel LDL and ApoB
    wide = daily.loc[daily.TYPE.isin(['ldl_cholesterol', 'apo_b'])].pivot(index=['USER_ID', 'DRAW_DATE'], columns='TYPE', values='value').reset_index()
    wide.columns.name = None
    g3 = {}
    for c in ('ldl_cholesterol', 'apo_b'):
        r = nearest(P, wide, c)
        if not (same(r[c], P[c]) and same(r.offset_days, pd.to_numeric(P[f'{c}_offset_days'], errors='coerce'))):
            fail('nearest-within-7-d matcher does not reproduce the panel ' + c)
        g3[c] = int(r[c].notna().sum())
    guards['G3_matcher_reproduces_panel'] = g3
    guards['G4_creatinine_rows_accounted'] = creat_audit
    cr = nearest(P, creat_daily, 'creatinine')
    src = dict(ldl_cholesterol=P.ldl_cholesterol, apo_b=P.apo_b, creatinine=cr.creatinine)
    stats, hists = [], []
    for k in MEASURES:
        s, h = summarize(k, src[k]); stats.append(s); hists.append(h)
    guards['G5_histograms_partition_n'] = True
    S = pd.DataFrame(stats); H = pd.concat(hists, ignore_index=True)
    offs = cr.offset_days.dropna()
    context = dict(primary_panel_n=len(P),
                   creatinine=dict(primary_with_value_within_7d=int(cr.creatinine.notna().sum()),
                                   primary_with_any_accepted_creatinine=int(P.USER_ID.isin(creat_daily.USER_ID).sum()),
                                   same_day_as_anchor=int(offs.eq(0).sum()), median_abs_offset_days=float(offs.abs().median())),
                   ldl_apob_same_participants=int((P.ldl_cholesterol.notna() & P.apo_b.notna()).sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / 'distribution_summary.csv', index=False)
    H.to_csv(OUT / 'distribution_histograms.csv', index=False)
    summ = dict(version=VERSION, populations=POPULATIONS, measures={s['measure']: s for s in stats}, context=context,
                definitions=dict(ldl_cholesterol='canonical panel value (canonical-labs), nearest draw within +/-7 d of the HbA1c anchor',
                                 apo_b='canonical panel value (canonical-labs), nearest draw within +/-7 d of the HbA1c anchor',
                                 creatinine=('serum creatinine, unit whitelist mg/dL or umol/L (/88.42), 0.2-15.0 mg/dL, daily median, '
                                             'nearest draw within +/-7 d of the HbA1c anchor (ties earlier)')))
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    md = dict(analysis='lab_distributions_extended', version=VERSION, status='passed',
              design='aggregate binned counts and summary statistics only; descriptive', guards=guards,
              participant_identifiers_written=False)
    (OUT / 'lab_distributions_extended.metadata.json').write_text(json.dumps(md, indent=1) + '\n')
    print(S[['measure', 'n', 'mean', 'median', 'q25', 'q75', 'n_below_display', 'n_above_display']].to_string(index=False))
    print(json.dumps(dict(context=context, creatinine_audit=creat_audit), indent=1))


if __name__ == '__main__':
    main()
