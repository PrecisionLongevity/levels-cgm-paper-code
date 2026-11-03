#!/usr/bin/env python3
"""Same-food agreement as the direct correlation between two set averages of the same food.

Reports the plain Pearson r between the averages of two disjoint sets of k meals of the same food, across ALL
person-food pairs, without subtracting each person's average over their foods and without a >= 2-foods-per-person
restriction. This is what a user sees when the same food is eaten again, and it is the convention of replicate-meal
studies. It is reported alongside the within-person (participant-demeaned) variant computed in
analysis/meals/same_food_averaging_preventive.py, which restricts to participants with >= 2 foods.

Population and selection (same frame as analysis/meals/same_food_averaging_preventive.py): isolated single-food meals
from _food_analysis.load_responses() (raw 2-h iAUC); participants identified by
analysis/validation/diabetes_range_exclusion.py diabetes_range_users() are excluded; the two sets of k meals are formed
by analysis/meals/split_half_reliability.py halves() (interleaved = alternate meals of the first 2k; 'consecutive' =
first k vs next k in time, reported here as 'time-separated').

For k = 1-7 and both splits:
  r                 Pearson r across all person-food pairs between the two set averages (no demeaning, no >= 2-foods filter)
  ci_low, ci_high   participant-clustered percentile bootstrap 95% CI: participants resampled with replacement, r recomputed
                    from the summed per-participant moments; 2,000 resamples; numpy default_rng(SeedSequence([SEED, k])) with
                    SEED = 20261029, so each k has its own reproducible stream and the two splits at the same k (which use
                    the same person-food pairs and participants) share the participant draws; quantiles 0.025 / 0.975
                    (numpy linear interpolation, as split_half_reliability.py cluster_boot)
  n_pairs, n_participants, median_days_between_sets (median absolute difference between the centroid timestamps of the
                    two sets, as same_food_averaging_preventive.py separation_days without its >= 2-foods filter)
Also written for comparison: the within-person r of same_food_averaging_preventive.py (same frame) next to the direct r.

Inputs:
  - LEVELS_DB database, via analysis/validation/diabetes_range_exclusion.py diabetes_range_users()
  - OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  - OUTPUTS/meals/same_food_averaging_preventive/preventive_averaging_curve.csv and
    same_food_averaging_preventive.metadata.json (analysis/meals/same_food_averaging_preventive.py)

Outputs (aggregate rows only, no participant identifiers), under OUTPUTS/meals/same_food_direct/:
  same_food_direct_curve.csv, direct_vs_within_person.csv, summary.json, same_food_direct.metadata.json

Usage:  python analysis/meals/same_food_direct.py
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

VERSION = 'same-food-direct'
PUB = OUTPUTS
PP = PUB / 'meals'
OUT = PP / 'same_food_direct'
D_SAME_FOOD_PREVENTIVE = PP / 'same_food_averaging_preventive'
CURVE_SAME_FOOD_PREVENTIVE = D_SAME_FOOD_PREVENTIVE / 'preventive_averaging_curve.csv'
S_SAME_FOOD_PREVENTIVE = ROOT / 'analysis/meals/same_food_averaging_preventive.py'
S_SPLIT_HALF = ROOT / 'analysis/meals/split_half_reliability.py'
S_DIABETES_EXCLUSION = ROOT / 'analysis/validation/diabetes_range_exclusion.py'
CEXP = PP / 'same_food_raw_r/same_food_r_variants.csv'   # analysis/meals/same_food_raw_vs_within.py
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
DB = database_path()
K_MAX, N_BOOT, SEED = 7, 2000, 20261029
SPLITS = ('interleaved', 'consecutive')
SPLIT_LABEL = {'interleaved': 'interleaved', 'consecutive': 'time-separated'}
TOL_R, TOL_DAYS = 1e-12, 1e-9


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def direct_r(h):
    """Pearson r across all person-food pairs between the two set averages (no demeaning, no >= 2-foods filter)."""
    return shared.pearson(h.a.to_numpy(float), h.b.to_numpy(float)), int(len(h)), int(h.uid.nunique())


def moments(h):
    """Per-participant sums (n, sx, sy, sxx, syy, sxy) of the raw set averages; rows sorted by uid (groupby order)."""
    g = h.assign(n=1.0, x=h.a.astype(float), y=h.b.astype(float))
    g = g.assign(xx=g.x * g.x, yy=g.y * g.y, xy=g.x * g.y)
    return g.groupby('uid')[['n', 'x', 'y', 'xx', 'yy', 'xy']].sum().to_numpy()


def r_from_moments(t):
    n, sx, sy, sxx, syy, sxy = t
    cxy, cxx, cyy = sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
    return cxy / np.sqrt(cxx * cyy)


def cluster_boot_direct(mom, k):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(k)]))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = r_from_moments(mom[rng.integers(0, len(mom), size=len(mom))].sum(axis=0))
    return draws


def separation_days_direct(frame, k, split):
    """Median days between the centroid timestamps of the two sets; separation_days of same_food_averaging_preventive.py
    without the >= 2-foods filter."""
    f = frame.sort_values(['uid', 'food', 'ts', 'meal_id'], kind='stable').copy()
    f['pos'] = f.groupby(['uid', 'food']).cumcount()
    f['n_cell'] = f.groupby(['uid', 'food']).iauc.transform('size')
    d = f.loc[f.n_cell.ge(2 * k) & f.pos.lt(2 * k)].copy()
    d['side'] = np.where(d.pos.lt(k) if split == 'consecutive' else d.pos.mod(2).eq(0), 'a', 'b')
    d['t'] = (pd.to_datetime(d.ts) - pd.Timestamp('1970-01-01')) / pd.Timedelta(days=1)
    c = d.groupby(['uid', 'food', 'side']).t.mean().unstack('side').dropna().reset_index()
    return float((c.b - c.a).abs().median()), int(len(c))


def main():
    t0 = time.monotonic()
    guards = {}
    # exclusion set: participants with any diabetes-range laboratory value
    s_split_half, s_diabetes_exclusion, s_same_food_preventive = load('split_half_reliability', S_SPLIT_HALF), load('diabetes_range_exclusion', S_DIABETES_EXCLUSION), load('same_food_averaging_preventive', S_SAME_FOOD_PREVENTIVE)
    with duckdb.connect(str(DB), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    panel = pd.read_csv(PANEL, usecols=['USER_ID', 'no_diabetes_range'], low_memory=False)
    pflag = set(panel.loc[panel.no_diabetes_range == False, 'USER_ID'])  # noqa: E712
    g2 = dict(excluded=len(excl), panel_flagged=len(pflag), panel_flagged_captured=len(pflag & excl))
    g2['passed'] = bool(g2['panel_flagged_captured'] == g2['panel_flagged'])
    guards['exclusion_set_contains_panel_flags'] = g2
    if not g2['passed']:
        fail(f'exclusion set does not contain every flagged panel participant {g2}')
    frame = shared.load_responses()
    frame = frame.loc[~frame.uid.isin(excl)].reset_index(drop=True)
    # ---- estimates (held in memory until every guard has passed)
    ref_same_food_preventive = pd.read_csv(CURVE_SAME_FOOD_PREVENTIVE).set_index(['split_estimator', 'k'])
    rows, within, g3_rows, g5_rows = [], [], [], []
    for split in SPLITS:
        for k in range(1, K_MAX + 1):
            h = s_split_half.halves(frame, k, split)
            # within-person variant (analysis/meals/same_food_averaging_preventive.py) from the same frame
            rw, nw, uw = shared.within_person_r(h)
            dw, ndw = s_same_food_preventive.separation_days(frame, k, split)
            ref = ref_same_food_preventive.loc[(split, k)]
            g3_rows.append(dict(split=split, k=k, r_diff=abs(rw - ref.r), n_pairs_equal=int(nw) == int(ref.n_pairs),
                                n_participants_equal=int(uw) == int(ref.n_participants), n_days_cells_equal=ndw == nw,
                                days_diff=abs(dw - ref.median_days_between_sets)))
            # direct variant
            r, n, u = direct_r(h)
            mom = moments(h)
            draws = cluster_boot_direct(mom, k)
            lo, hi = (float(q) for q in np.quantile(draws, [.025, .975]))
            days, nd = separation_days_direct(frame, k, split)
            g5_rows.append(dict(split=split, k=k, moment_r_diff=abs(float(r_from_moments(mom.sum(axis=0))) - r),
                                n_mom_equal=int(mom[:, 0].sum()) == n and len(mom) == u, days_cells_equal=nd == n,
                                draws_finite=int(np.isfinite(draws).sum()), ci_brackets_r=lo <= r <= hi))
            rows.append(dict(split=SPLIT_LABEL[split], split_estimator=split, k=k, n_pairs=n, n_participants=u, r=r,
                             ci_low=lo, ci_high=hi, median_days_between_sets=days, n_boot=N_BOOT))
            within.append(dict(split=SPLIT_LABEL[split], k=k, r_direct=r, n_pairs_direct=n, n_participants_direct=u,
                               r_within_person=rw, n_pairs_within_person=int(nw), n_participants_within_person=int(uw),
                               r_difference=r - rw))
    # within-person variant reproduces the saved curve of same_food_averaging_preventive.py
    g3t = pd.DataFrame(g3_rows)
    ev = s_split_half.evaluate(frame, s_same_food_preventive.K_MAX_SAME_FOOD_AVERAGING, s_split_half.N_BOOT, s_split_half.SEED).set_index(['split', 'k'])
    ci_diff = max(max(abs(ev.loc[(s, k), 'ci_low'] - ref_same_food_preventive.loc[(s, k), 'ci_low']), abs(ev.loc[(s, k), 'ci_high'] - ref_same_food_preventive.loc[(s, k), 'ci_high']))
                  for s in SPLITS for k in range(1, K_MAX + 1))
    g3 = dict(rows=int(len(g3t)), rows_expected=int(len(ref_same_food_preventive)), max_r_diff=float(g3t.r_diff.max()),
              n_pairs_equal=bool(g3t.n_pairs_equal.all()), n_participants_equal=bool(g3t.n_participants_equal.all()),
              days_cells_equal=bool(g3t.n_days_cells_equal.all()), max_days_diff=float(g3t.days_diff.max()),
              ci_max_diff_evaluate=float(ci_diff))
    g3['passed'] = bool(g3['rows'] == g3['rows_expected'] == 2 * K_MAX and g3['max_r_diff'] < TOL_R and g3['n_pairs_equal']
                        and g3['n_participants_equal'] and g3['days_cells_equal'] and g3['max_days_diff'] < TOL_DAYS
                        and g3['ci_max_diff_evaluate'] < TOL_R)
    guards['within_person_reproduces_preventive_curve'] = g3
    if not g3['passed']:
        fail(f'within-person variant does not reproduce same_food_averaging_preventive.py {g3}')
    # direct r equals the raw_all_pairs values of analysis/meals/same_food_raw_vs_within.py at k = 1 and 7
    T = pd.DataFrame(rows)
    ex = pd.read_csv(CEXP).query("variant == 'raw_all_pairs'").set_index(['split', 'k'])
    g4_rows = []
    for s in ('interleaved', 'time-separated'):
        for k in (1, K_MAX):
            a = T.set_index(['split', 'k']).loc[(s, k)]
            e = ex.loc[(s, k)]
            g4_rows.append(dict(split=s, k=k, r=float(a.r), r_exploratory=float(e.r), r_diff=abs(float(a.r) - float(e.r)),
                                n_pairs=int(a.n_pairs), n_pairs_exploratory=int(e.n_pairs), n_participants=int(a.n_participants),
                                n_participants_exploratory=int(e.n_participants)))
    g4 = dict(rows=g4_rows)
    g4['passed'] = bool(all(r_['r_diff'] < TOL_R and r_['n_pairs'] == r_['n_pairs_exploratory']
                                and r_['n_participants'] == r_['n_participants_exploratory'] for r_ in g4_rows))
    guards['direct_matches_raw_all_pairs_k1_k7'] = g4
    if not g4['passed']:
        fail(f'direct r differs from the raw_all_pairs values {g4}')
    # internal consistency
    g5t = pd.DataFrame(g5_rows)
    k1 = T.loc[T.k.eq(1)].set_index('split')
    g5 = dict(max_moment_r_diff=float(g5t.moment_r_diff.max()), n_moments_equal=bool(g5t.n_mom_equal.all()),
              days_cells_equal=bool(g5t.days_cells_equal.all()), min_finite_draws=int(g5t.draws_finite.min()),
              ci_brackets_r=bool(g5t.ci_brackets_r.all()),
              k1_equal_across_splits=bool(abs(k1.loc['interleaved', 'r'] - k1.loc['time-separated', 'r']) < TOL_R
                                          and k1.loc['interleaved', 'n_pairs'] == k1.loc['time-separated', 'n_pairs']
                                          and k1.loc['interleaved', 'ci_low'] == k1.loc['time-separated', 'ci_low']),
              n_pairs_nonincreasing=bool(all(T.loc[T.split.eq(s)].sort_values('k').n_pairs.diff().dropna().le(0).all()
                                             for s in ('interleaved', 'time-separated'))))
    g5['passed'] = bool(g5['max_moment_r_diff'] < TOL_R and g5['n_moments_equal'] and g5['days_cells_equal']
                        and g5['min_finite_draws'] == N_BOOT and g5['ci_brackets_r'] and g5['k1_equal_across_splits']
                        and g5['n_pairs_nonincreasing'])
    guards['internal_consistency'] = g5
    if not g5['passed']:
        fail(f'internal consistency {g5}')
    guards['all_passed'] = True
    # ---- outputs (aggregate only)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / 'same_food_direct_curve.csv', index=False)
    pd.DataFrame(within).to_csv(OUT / 'direct_vs_within_person.csv', index=False)
    P = T.set_index(['split', 'k'])
    first070 = {s: (int(c.loc[c.r >= .70, 'k'].min()) if (c.r >= .70).any() else None) for s, c in T.groupby('split')}
    cell = lambda s, k: dict(r=round(float(P.loc[(s, k), 'r']), 4),  # noqa: E731
                             ci=[round(float(P.loc[(s, k), 'ci_low']), 4), round(float(P.loc[(s, k), 'ci_high']), 4)],
                             n_pairs=int(P.loc[(s, k), 'n_pairs']), n_participants=int(P.loc[(s, k), 'n_participants']),
                             median_days_between_sets=round(float(P.loc[(s, k), 'median_days_between_sets']), 2))
    summary = dict(version=VERSION,
                   population=f'isolated single-food meals; participants with any diabetes-range laboratory value excluded ({len(excl)}; '
                              'diabetes_range_users in analysis/validation/diabetes_range_exclusion.py)',
                   estimator='direct Pearson r across all person-food pairs between the averages of two disjoint sets of k meals of the '
                             'same food (no subtraction of the person average, no >= 2-foods-per-person restriction)',
                   headline={f'{s}_k{k}': cell(s, k) for s in ('interleaved', 'time-separated') for k in range(1, K_MAX + 1)},
                   first_k_reaching_0_70=first070,
                   at_first_k_reaching_0_70={s: (cell(s, k) if k else None) for s, k in first070.items()},
                   note=f'Observed values; not projected. 95% CI: participant-clustered percentile bootstrap, {N_BOOT:,} resamples, '
                        f'seed SeedSequence([{SEED}, k]).')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    outs = ['same_food_direct_curve.csv', 'direct_vs_within_person.csv', 'summary.json']
    meta = dict(analysis='same_food_direct', version=VERSION, status='passed', guards=guards,
                estimator=summary['estimator'],
                selection='split_half_reliability.halves() on the frame of same_food_averaging_preventive.py',
                bootstrap=dict(n_boot=N_BOOT, seed=SEED, stream='numpy default_rng(SeedSequence([seed, k])) per k; both splits '
                               'at a given k share the participant draws', cluster='participant', interval='percentile 2.5/97.5 '
                               '(numpy.quantile, linear)'),
                k_reported=list(range(1, K_MAX + 1)), outcome='IAUC_2H (raw 2-h iAUC)', outputs=outs,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'same_food_direct.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(T.round(4).to_string(index=False))
    print(json.dumps(dict(first_k_reaching_0_70=first070, all_passed=True)))


if __name__ == '__main__':
    main()
