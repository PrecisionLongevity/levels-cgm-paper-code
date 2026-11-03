#!/usr/bin/env python3
"""Observed same-food averaging curve (k = 1-7) in the preventive population.

The paper's meal/CGM-day convention excludes participants with any accepted diabetes-range laboratory value
(analysis/validation/diabetes_range_exclusion.py diabetes_range_users(con)). This producer applies that exclusion
and issues the full observed averaging curve k = 1-7 for the remaining ('preventive') population, using the same
estimator as analysis/meals/same_food_averaging.py.

Estimator: analysis/meals/split_half_reliability.py's halves(), within_person_r, cluster_boot; evaluate() with the
same configuration as analysis/meals/same_food_averaging.py (K_MAX = 8, N_BOOT = 200, SEED = 20260903, so the seeded
bootstrap stream is identical), on isolated single-food meals (_food_analysis.load_responses(): single log, one
macro row, 2-h pre/post isolation, coverage OK), outcome raw 2-h iAUC. r = within-person (participant-demeaned)
Pearson correlation between the means of two disjoint sets of k meals of the same food; 'interleaved' = alternating
meals of the first 2k, 'time-separated' = the 'consecutive' split (first k vs next k in time). 95% CI:
participant-clustered percentile bootstrap. k = 8 is computed only to keep the bootstrap stream identical and is not
reported. Observed values only; no Spearman-Brown projection. Also reported (descriptive, not an estimator change):
median days between the centroid timestamps of the two sets per k and split.

Inputs
  - database tables via LEVELS_DB (through diabetes_range_users(con) in analysis/validation/diabetes_range_exclusion.py)
  - OUTPUTS/meals/same_food_averaging_observed/observed_averaging_curve.csv (analysis/meals/same_food_averaging.py)
  - OUTPUTS/validation/diabetes_range_exclusion_sensitivity/exclusion_sensitivity.csv
    (analysis/validation/diabetes_range_exclusion.py)
  - OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)

Outputs (OUTPUTS/meals/same_food_averaging_preventive/) -- aggregate rows only; no participant identifiers
  - preventive_averaging_curve.csv, averaging_curve_by_population.csv, summary.json

Usage
  python analysis/meals/same_food_averaging_preventive.py
"""
from __future__ import annotations

import importlib.util
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
from analysis.meals import _food_analysis as shared# noqa: E402

VERSION = 'same-food-averaging-preventive'
PUB = OUTPUTS
PP = PUB / 'meals'
OUT = PP / 'same_food_averaging_preventive'
D_SAME_FOOD_AVERAGING = PP / 'same_food_averaging_observed'
CURVE_SAME_FOOD_AVERAGING = D_SAME_FOOD_AVERAGING / 'observed_averaging_curve.csv'
S_SPLIT_HALF = ROOT / 'analysis/meals/split_half_reliability.py'
S_DIABETES_EXCLUSION = ROOT / 'analysis/validation/diabetes_range_exclusion.py'
D_DIABETES_EXCLUSION = PUB / 'validation/diabetes_range_exclusion_sensitivity'
T_DIABETES_EXCLUSION = D_DIABETES_EXCLUSION / 'exclusion_sensitivity.csv'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
DB = database_path()
K_MAX_SAME_FOOD_AVERAGING = 8          # configuration of analysis/meals/same_food_averaging.py (keeps the bootstrap stream identical)
K_REPORT = 7
SPLIT_LABEL = {'interleaved': 'interleaved', 'consecutive': 'time-separated'}


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def separation_days(frame, k, split):
    """Median days between the centroid timestamps of the two sets (same selection rule as split_half_reliability.halves())."""
    f = frame.sort_values(['uid', 'food', 'ts', 'meal_id'], kind='stable').copy()
    f['pos'] = f.groupby(['uid', 'food']).cumcount()
    f['n_cell'] = f.groupby(['uid', 'food']).iauc.transform('size')
    d = f.loc[f.n_cell.ge(2 * k) & f.pos.lt(2 * k)].copy()
    d['side'] = np.where(d.pos.lt(k) if split == 'consecutive' else d.pos.mod(2).eq(0), 'a', 'b')
    d['t'] = (pd.to_datetime(d.ts) - pd.Timestamp('1970-01-01')) / pd.Timedelta(days=1)   # unit-safe (substrate is datetime64[us])
    c = d.groupby(['uid', 'food', 'side']).t.mean().unstack('side').dropna().reset_index()
    c = c.loc[c.groupby('uid').a.transform('size').ge(2)]      # within_person_r eligibility (>=2 cells per person)
    return float((c.b - c.a).abs().median()), int(len(c))


def curve(s_split_half, frame):
    ev = s_split_half.evaluate(frame, K_MAX_SAME_FOOD_AVERAGING, s_split_half.N_BOOT, s_split_half.SEED)
    return ev[['split', 'k', 'n_cells', 'n_users', 'r_empirical', 'ci_low', 'ci_high']]


def main():
    t0 = time.monotonic(); guards = {}
    s_split_half = load('split_half_reliability', S_SPLIT_HALF)
    # all-participant run: must reproduce the saved curve of analysis/meals/same_food_averaging.py (same estimator, same meals)
    frame = shared.load_responses()
    full = curve(s_split_half, frame)
    saved = pd.read_csv(CURVE_SAME_FOOD_AVERAGING)
    g = saved.merge(full, on=['split', 'k'], suffixes=('_saved', '_new'), validate='one_to_one')
    g2 = dict(rows_saved=int(len(saved)), rows_matched=int(len(g)),
              n_equal=bool((g.n_cells_saved == g.n_cells_new).all() and (g.n_users_saved == g.n_users_new).all()),
              max_abs_diff=float(max((g[f'{c}_saved'] - g[f'{c}_new']).abs().max() for c in ('r_empirical', 'ci_low', 'ci_high'))))
    g2['passed'] = bool(len(g) == len(saved) and g2['n_equal'] and g2['max_abs_diff'] < 1e-12)
    guards['all_participant_run_reproduces_same_food_averaging'] = g2
    if not g2['passed']:
        fail(f'all-participant run does not reproduce analysis/meals/same_food_averaging.py {g2}')
    # exclusion set: participants with any diabetes-range laboratory value
    s_diabetes_exclusion = load('diabetes_range_exclusion', S_DIABETES_EXCLUSION)
    with duckdb.connect(str(DB), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    panel = pd.read_csv(PANEL, usecols=['USER_ID', 'no_diabetes_range'], low_memory=False)
    pflag = set(panel.loc[panel.no_diabetes_range == False, 'USER_ID'])  # noqa: E712
    g3 = dict(excluded=len(excl), panel_flagged=len(pflag), panel_flagged_captured=len(pflag & excl))
    g3['passed'] = bool(len(pflag & excl) == len(pflag))
    guards['exclusion_set_contains_panel_flags'] = g3
    if not g3['passed']:
        fail(f'diabetes-range exclusion set does not contain every flagged panel participant {g3}')
    # ---- preventive run
    keep = ~frame.uid.isin(excl)
    prev_frame = frame.loc[keep].reset_index(drop=True)
    removed = dict(meals=int((~keep).sum()), participants_with_meals=int(frame.loc[~keep, 'uid'].nunique()),
                   participants_with_meals_all=int(frame.uid.nunique()), participants_with_meals_preventive=int(prev_frame.uid.nunique()))
    prev = curve(s_split_half, prev_frame)
    # cross-check with the k = 1 and k = 7 rows of analysis/validation/diabetes_range_exclusion.py
    t_diabetes_exclusion = pd.read_csv(T_DIABETES_EXCLUSION).set_index(['analysis', 'sample'])
    r_diabetes_exclusion = t_diabetes_exclusion.loc[('averaging', 'excluding_diabetes_range')]
    pr = prev.set_index(['split', 'k'])
    g4 = dict(k7_r=float(pr.loc[('interleaved', 7), 'r_empirical']), k7_r_diabetes_exclusion=float(r_diabetes_exclusion.estimate),
              k7_n=int(pr.loc[('interleaved', 7), 'n_cells']), k7_n_diabetes_exclusion=int(r_diabetes_exclusion.n),
              k7_users=int(pr.loc[('interleaved', 7), 'n_users']), k7_users_diabetes_exclusion=int(r_diabetes_exclusion.n_users),
              k1_r=float(pr.loc[('interleaved', 1), 'r_empirical']), k1_r_diabetes_exclusion=float(r_diabetes_exclusion.r_k1))
    g4['passed'] = bool(abs(g4['k7_r'] - g4['k7_r_diabetes_range_exclusion']) < 1e-12 and abs(g4['k1_r'] - g4['k1_r_diabetes_range_exclusion']) < 1e-12
                        and g4['k7_n'] == g4['k7_n_diabetes_range_exclusion'] and g4['k7_users'] == g4['k7_users_diabetes_range_exclusion'])
    guards['preventive_matches_exclusion_sensitivity_k1_k7'] = g4
    if not g4['passed']:
        fail(f'preventive k=1/k=7 differ from analysis/validation/diabetes_range_exclusion.py {g4}')
    guards['all_passed'] = True
    # ---- outputs
    rows = []
    for pop, cur, fr_ in (('preventive_excluding_diabetes_range', prev, prev_frame), ('all_participants', full, frame)):
        for r in cur.loc[cur.k.le(K_REPORT)].itertuples():
            sep, nsep = separation_days(fr_, r.k, r.split)
            if nsep != r.n_cells:
                fail(f'separation cells {nsep} != estimator cells {r.n_cells} ({pop}, {r.split}, k={r.k})')
            rows.append(dict(population=pop, split=SPLIT_LABEL[r.split], split_estimator=r.split, k=int(r.k),
                             n_pairs=int(r.n_cells), n_participants=int(r.n_users), r=float(r.r_empirical),
                             ci_low=float(r.ci_low), ci_high=float(r.ci_high), median_days_between_sets=sep))
    T = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.loc[T.population.eq('preventive_excluding_diabetes_range')].drop(columns='population').to_csv(OUT / 'preventive_averaging_curve.csv', index=False)
    T.to_csv(OUT / 'averaging_curve_by_population.csv', index=False)
    P = T.loc[T.population.eq('preventive_excluding_diabetes_range')].set_index(['split', 'k'])
    first070 = {s: (int(c.loc[c.r >= .70, 'k'].min()) if (c.r >= .70).any() else None)
                for s, c in T.loc[T.population.eq('preventive_excluding_diabetes_range')].groupby('split')}
    summary = dict(version=VERSION, population='isolated single-food meals; participants with any diabetes-range laboratory value excluded '
                                               f'({len(excl)}; diabetes_range_users in analysis/validation/diabetes_range_exclusion.py)', removed=removed,
                   headline={f'{s}_k{k}': dict(r=round(float(P.loc[(s, k), 'r']), 4), ci=[round(float(P.loc[(s, k), 'ci_low']), 4),
                                                round(float(P.loc[(s, k), 'ci_high']), 4)], n_pairs=int(P.loc[(s, k), 'n_pairs']),
                                                n_participants=int(P.loc[(s, k), 'n_participants']))
                             for s in ('interleaved', 'time-separated') for k in (1, 7)},
                   first_k_reaching_0_70=first070,
                   note='Observed within-person correlation between averages of two disjoint sets of k meals of the same food; '
                        'not projected. 95% CI: participant-clustered percentile bootstrap, 200 resamples.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    outs = ['preventive_averaging_curve.csv', 'averaging_curve_by_population.csv', 'summary.json']
    meta = dict(analysis='same_food_averaging_preventive', version=VERSION, status='passed', guards=guards,
                estimator='analysis/meals/split_half_reliability.py evaluate(), configuration of analysis/meals/same_food_averaging.py',
                k_max_computed=K_MAX_SAME_FOOD_AVERAGING, k_reported=list(range(1, K_REPORT + 1)),
                seed=int(s_split_half.SEED), n_boot=int(s_split_half.N_BOOT), outcome='IAUC_2H (raw 2-h iAUC)', outputs=outs,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'same_food_averaging_preventive.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(guards=guards, summary=summary), indent=1))


if __name__ == '__main__':
    main()
