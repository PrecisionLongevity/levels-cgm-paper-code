#!/usr/bin/env python3
"""Three-set overlap of CGM, logged behavior and laboratory results across all consenting participants.

Computes an area-proportional three-set Euler diagram (CGM, logged behavior, laboratory results) over the
consenting cohort, plus a cohort enrollment funnel.

Definitions (identical to analysis/cohort/cohort_reference.py)
  Roster     cohort_participants.parquet (analysis/cohort/cohort_reference.py): one row per consenting
             noninternal participant.
  Streams    flags from cohort_reference.py: cgm = glucose_rows > 0, food = food_log_rows > 0,
             exercise = log_rows > 0, steps = step_rows > 0, weight = weight_measurements > 0,
             labs = total_lab_records > 0, sleep = sleep_log_count > 0 (presence of any record, not accepted
             reference values).
  Sets       CGM = cgm; LOGGED BEHAVIOR = food OR exercise OR steps OR sleep (task definition; weight is a body
             measurement and is not part of the behavior set; its marginal is reported separately);
             LABS = labs.
  Regions    the 7 Venn regions of (CGM, BEHAVIOR, LABS) plus 'none'; they partition the roster.
  Funnel     consented (roster) -> with CGM -> canonical lab-CGM sample (first HbA1c with >= 14 dense CGM days
             within +/-90 d; canonical_lab_cgm_panel all_lab_linked, symmetric window) -> primary cohort without
             diabetes-range labs (no_diabetes_range). Counts only.

Inputs
  OUTPUTS/cohort/cohort_participants.parquet, OUTPUTS/cohort/cohort_summary.json
    (analysis/cohort/cohort_reference.py)
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)

Outputs (OUTPUTS/cohort/stream_overlap_regions/, aggregate counts only; no participant identifiers written)
  stream_overlap_regions.csv  region-level counts for the Euler diagram
  cohort_funnel.csv           enrollment funnel counts
  summary.json                set marginals, pairwise intersections, regions, funnel, definitions

Usage
  python analysis/cohort/stream_overlap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.publication_panels import load_panel  # noqa: E402

VERSION = 'stream-overlap-regions'
PUB = OUTPUTS
CC = PUB / 'cohort'
S_COHORT_REFERENCE = ROOT / 'analysis/cohort/cohort_reference.py'
OUT = CC / 'stream_overlap_regions'

# flags: key -> cohort_participants.parquet column (as in analysis/cohort/cohort_reference.py)
STREAM_COLS = dict(cgm='glucose_rows', food='food_log_rows', exercise='log_rows', steps='step_rows',
                   weight='weight_measurements', labs='total_lab_records', sleep='sleep_log_count')
BEHAVIOR_STREAMS = ('food', 'exercise', 'steps', 'sleep')
REGIONS = [  # region key, (in CGM, in BEHAVIOR, in LABS)
    ('cgm_only', (True, False, False)), ('behavior_only', (False, True, False)), ('labs_only', (False, False, True)),
    ('cgm_behavior', (True, True, False)), ('cgm_labs', (True, False, True)), ('behavior_labs', (False, True, True)),
    ('all_three', (True, True, True)), ('none', (False, False, False))]


def fail(msg):
    raise SystemExit('GUARD FAILED: ' + msg)


def main():
    guards = {}
    canonical = load_panel(PUB / 'canonical_panel/canonical_lab_cgm_panel.csv', cohort='all_lab_linked', expected_window='symmetric')
    summary_cohort_reference = json.loads((CC / 'cohort_summary.json').read_text())
    data = pd.read_parquet(CC / 'cohort_participants.parquet', columns=['USER_ID', *STREAM_COLS.values()])
    if data.USER_ID.isna().any() or data.USER_ID.duplicated().any():
        fail('roster not one row per participant')
    flags = {k: data[col].gt(0) for k, col in STREAM_COLS.items()}
    coverage = {k + '_any': int(v.sum()) for k, v in flags.items()}
    pair = data.USER_ID.isin(canonical.USER_ID)
    primary = canonical.no_diabetes_range.astype(str).str.lower().isin(['true', '1'])
    funnel = [('consented', len(data)), ('with_cgm', coverage['cgm_any']), ('lab_cgm_sample', int(pair.sum())),
              ('primary_no_diabetes_range_labs', int(primary.sum()))]
    if (pair & ~flags['cgm']).any() or (pair & ~flags['labs']).any():
        fail('lab-CGM sample members without CGM or laboratory records')
    # three sets and regions
    S = dict(cgm=flags['cgm'], behavior=flags['food'] | flags['exercise'] | flags['steps'] | flags['sleep'], labs=flags['labs'])
    rows = []
    for key, (c, b, l) in REGIONS:
        m = (S['cgm'] == c) & (S['behavior'] == b) & (S['labs'] == l)
        rows.append(dict(region=key, in_cgm=c, in_behavior=b, in_labs=l, n=int(m.sum()), n_lab_cgm_sample=int((m & pair).sum()),
                         n_primary=int((m & pair & data.USER_ID.isin(canonical.USER_ID[primary])).sum())))
    R = pd.DataFrame(rows)
    marg = {k: int(v.sum()) for k, v in S.items()}
    if R.n.sum() != len(data):
        fail('regions do not partition the roster')
    for k, (i, _) in (('cgm', (0, None)), ('behavior', (1, None)), ('labs', (2, None))):
        col = ['in_cgm', 'in_behavior', 'in_labs'][i]
        if int(R.loc[R[col], 'n'].sum()) != marg[k]:
            fail('region sums do not equal the set marginal: ' + k)
    if marg['cgm'] != summary_cohort_reference['coverage']['cgm_any'] or marg['labs'] != summary_cohort_reference['coverage']['labs_any']:
        fail('CGM / labs marginals do not equal the published per-stream counts')
    if any(marg['behavior'] < summary_cohort_reference['coverage'][s + '_any'] for s in BEHAVIOR_STREAMS):
        fail('behavior union smaller than a constituent stream')
    if R.n_lab_cgm_sample.sum() != summary_cohort_reference['aligned'] or R.n_primary.sum() != summary_cohort_reference['primary_aligned']:
        fail('sample membership by region does not sum')
    guards['marginals_equal_published'] = dict(cgm=marg['cgm'], labs=marg['labs'],
                                                behavior_constituents={s: summary_cohort_reference['coverage'][s + '_any'] for s in BEHAVIOR_STREAMS})
    guards['regions_partition_roster'] = True
    # pairwise intersections (for the Euler solver)
    pairs = dict(cgm_behavior=int((S['cgm'] & S['behavior']).sum()), cgm_labs=int((S['cgm'] & S['labs']).sum()),
                 behavior_labs=int((S['behavior'] & S['labs']).sum()), all_three=int((S['cgm'] & S['behavior'] & S['labs']).sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / 'stream_overlap_regions.csv', index=False)
    pd.DataFrame(funnel, columns=['stage', 'n']).to_csv(OUT / 'cohort_funnel.csv', index=False)
    summ = dict(version=VERSION, total=len(data), set_marginals=marg, pairwise_intersections=pairs,
                regions={r['region']: r['n'] for r in rows}, funnel=dict(funnel),
                stream_coverage=coverage,
                set_definitions=dict(cgm='glucose_rows > 0 (accepted CGM readings, analysis/cohort/cohort_reference.py)',
                                     behavior='any food log, exercise log, step count or sleep record (cohort_reference.py flags food|exercise|steps|sleep)',
                                     labs='any laboratory record (total_lab_records > 0, analysis/cohort/cohort_reference.py)',
                                     weight_not_in_behavior=coverage['weight_any']),
                lab_cgm_sample_definition='canonical lab-CGM panel (all_lab_linked, symmetric window): first HbA1c with >= 14 dense CGM days within +/-90 d',
                primary_definition='lab-CGM sample without observed diabetes-range laboratory values (no_diabetes_range)')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    md = dict(analysis='stream_overlap', version=VERSION, design='aggregate participant counts only; descriptive',
              guards=guards, participant_identifiers_written=False)
    (OUT / 'stream_overlap_regions.metadata.json').write_text(json.dumps(md, indent=1) + '\n')
    print(json.dumps(dict(marginals=marg, regions=summ['regions'], funnel=summ['funnel']), indent=1))


if __name__ == '__main__':
    main()
