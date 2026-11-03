#!/usr/bin/env python3
"""Week-to-week glucose reliability stratified by the gap between the two weeks, plus within-week
day-level projection versus observed across-week reliability.

Population and definitions
  Sample: paired seven-day windows from analysis/reliability/repeatability.py (first two non-overlapping
  7-day dense blocks per participant; cached as seven_day_windows.parquet / seven_day_cgm_features.parquet),
  excluding the diabetes-range users identified by analysis/validation/diabetes_range_exclusion.py
  diabetes_range_users(con) (the day-level substrate already excludes them, so the exclusion is expected to
  remove nobody).

Part 1 (reliability_by_gap.csv)
  ICC(A,1) (as defined in repeatability.py icc_a1) of average glucose, SD, MODD and CV between the two weeks,
  by gap between window STARTS: 7 days (back-to-back), 8-35, 36-90, >90 days, plus pooled; n, median and IQR
  of the start gap per bin. reliability_by_gap_weekly_subbins.csv splits 8-35 days by week.
  gap_contrasts.csv: back-to-back minus each later bin (independent bins; independent resample sets).

Part 2 (day_level_icc.csv, projection_vs_observed.csv)
  One-way ICC(1) of daily average glucose (gluc_local_day MEAN, 7 dense days) within week 1 and within week 2;
  Spearman-Brown projection k*r/(1+(k-1)r), k = 1..7; days needed for an ICC of 0.80, m = 0.8(1-r)/(0.2 r).
  projection_vs_observed.csv (tidy): per k = 1..7 the within-week projections (week 1, week 2) and the observed
  across-week ICC(A,1) of the mean of the first k days of week 1 vs the first k days of week 2 (k = 1: one day
  vs one day; k = 7: one week vs one week from daily means), all gaps pooled and by gap bin. The canonical
  one-week vs one-week value is the Part 1 pooled row (window features from the 15-min grid); the daily-mean
  version is given alongside.

95% CI: percentile, N_BOOT seeded participant resamples. Within a gap bin, one set of resample indices is shared
by all metrics and Part 2 quantities (common random numbers), so projection-minus-observed differences have CIs.

Inputs
  - OUTPUTS/reliability/cache/seven_day_windows.parquet, seven_day_cgm_features.parquet,
    OUTPUTS/reliability/axis_reliability.csv, gap_stratified_reliability.csv,
    OUTPUTS/reliability/repeatability_analysis.metadata.json, cache/repeatability_features.metadata.json
    (analysis/reliability/repeatability.py)
  - OUTPUTS/daily_glucose/_cache/gluc_local_day.parquet (analysis/daily_glucose/build_day_panel.py)
  - OUTPUTS/validation/diabetes_range_exclusion_sensitivity/diabetes_range_exclusion_sensitivity.metadata.json
    (analysis/validation/diabetes_range_exclusion.py)

Outputs (OUTPUTS/reliability/reliability_by_gap/), aggregate rows only (no participant identifiers)
  - reliability_by_gap.csv, reliability_by_gap_weekly_subbins.csv, gap_contrasts.csv, day_level_icc.csv,
    projection_vs_observed.csv, summary.json, reliability_by_gap.metadata.json

Usage
  python analysis/reliability/reliability_by_gap.py [--out PATH]
"""
from __future__ import annotations

import argparse
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

VERSION = 'reliability-by-gap'
PUB = OUTPUTS
REP = PUB / 'reliability'
CACHE = REP / 'cache'
WINDOWS = CACHE / 'seven_day_windows.parquet'
FEATURES = CACHE / 'seven_day_cgm_features.parquet'
GLUC = PUB / 'daily_glucose/_cache/gluc_local_day.parquet'
V_DIABETES_EXCLUSION = PUB / 'validation/diabetes_range_exclusion_sensitivity'
DB = database_path()
OUT = REP / 'reliability_by_gap'
S_REPEATABILITY = 'analysis/reliability/repeatability.py'
S_DIABETES_EXCLUSION = 'analysis/validation/diabetes_range_exclusion.py'
SEED = 20260930
N_BOOT = 2000
CHUNK = 100
METRICS = ['mean_glucose', 'sd_glucose', 'modd', 'cv_glucose']
LABEL = {'mean_glucose': 'Average glucose', 'sd_glucose': 'SD', 'modd': 'MODD', 'cv_glucose': 'CV'}
BINS = [('back_to_back_7d', '7 days (back-to-back)', 7, 7), ('8_35d', '8-35 days', 8, 35),
        ('36_90d', '36-90 days', 36, 90), ('over_90d', '>90 days', 91, 10 ** 9)]
SUBBINS = [('8_14d', '8-14 days', 8, 14), ('15_21d', '15-21 days', 15, 21), ('22_28d', '22-28 days', 22, 28),
           ('29_35d', '29-35 days', 29, 35)]
TARGET = 0.80
K = 7


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def rel(p):
    return str(Path(p).relative_to(ROOT))


# ---- estimators ---------------------------------------------------------------------------------------------
def icc_a1_vec(x, y, idx):
    """ICC(A,1) of analysis/reliability/repeatability.py for every row of an index matrix (B x n), via s = x+y, d = x-y.

    msr = var(s)/2, mse = var(d)/2 (ddof=1), msc = n*mean(d)^2/2; ICC = (msr-mse)/(msr+mse+2(msc-mse)/n).
    """
    s = (x + y)[idx]; d = (x - y)[idx]; n = idx.shape[-1]
    msr = s.var(axis=-1, ddof=1) / 2; mse = d.var(axis=-1, ddof=1) / 2; msc = n * d.mean(axis=-1) ** 2 / 2
    den = msr + mse + 2 * (msc - mse) / n
    return np.where(den > 0, (msr - mse) / np.where(den > 0, den, 1), np.nan)


def icc1_vec(M, idx):
    """One-way random ICC(1) (single day) for every row of an index matrix; M is n x k complete."""
    X = M[idx]; n, k = idx.shape[-1], M.shape[1]
    rm = X.mean(axis=-1); gm = rm.mean(axis=-1, keepdims=True)
    msb = k * ((rm - gm) ** 2).sum(axis=-1) / (n - 1)
    msw = ((X - rm[..., None]) ** 2).sum(axis=(-1, -2)) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def spearman_brown(r, k):
    return k * r / (1 + (k - 1) * r)


def days_needed(r, target=TARGET):
    return target * (1 - r) / ((1 - target) * r)


def boot_indices(n, seed):
    rng = np.random.default_rng(seed)
    for start in range(0, N_BOOT, CHUNK):
        yield rng.integers(0, n, (min(CHUNK, N_BOOT - start), n))


def ci(b):
    b = np.asarray(b, float)
    if not np.isfinite(b).all():
        fail('non-finite bootstrap replicate')
    lo, hi = np.quantile(b, [.025, .975])
    return float(lo), float(hi)


def analyse_group(P, M1, M2, seed):
    """Point estimates and bootstrap replicates for one participant set (shared resamples)."""
    n = len(P)
    X = {m: (P[m + '_1'].to_numpy(float), P[m + '_2'].to_numpy(float)) for m in METRICS}
    cum1 = np.cumsum(M1, axis=1) / np.arange(1, K + 1); cum2 = np.cumsum(M2, axis=1) / np.arange(1, K + 1)
    full = np.arange(n)[None, :]

    def stats(idx):
        out = {f'icc_{m}': icc_a1_vec(*X[m], idx) for m in METRICS}
        out['icc1_week1'] = icc1_vec(M1, idx); out['icc1_week2'] = icc1_vec(M2, idx)
        for k in range(1, K + 1):
            out[f'obs_k{k}'] = icc_a1_vec(cum1[:, k - 1], cum2[:, k - 1], idx)
        return out

    point = {key: float(v[0]) for key, v in stats(full).items()}
    reps = {key: [] for key in point}
    for idx in boot_indices(n, seed):
        for key, v in stats(idx).items():
            reps[key].append(v)
    reps = {key: np.concatenate(v) for key, v in reps.items()}
    return point, reps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = time.monotonic(); guards = {}
    out = args.out; out.mkdir(parents=True, exist_ok=True)

    m_repeatability = json.loads((REP / 'repeatability_analysis.metadata.json').read_text())
    m_diabetes_exclusion = json.loads((V_DIABETES_EXCLUSION / 'diabetes_range_exclusion_sensitivity.metadata.json').read_text())
    s_repeatability = load('repeatability_for_reliability_by_gap', S_REPEATABILITY)
    s_diabetes_exclusion = load('diabetes_exclusion_for_reliability_by_gap', S_DIABETES_EXCLUSION)

    # ---- sample (analysis/reliability/repeatability.py reliability_outputs pairing)
    W = pd.read_parquet(WINDOWS); Fe = pd.read_parquet(FEATURES)
    J = W.merge(Fe, on='window_id', validate='one_to_one')
    P = J[J.window_number.eq(1)].merge(J[J.window_number.eq(2)], on='USER_ID', suffixes=('_1', '_2'),
                                       validate='one_to_one')
    P['edge_gap'] = (P.window_start_2 - P.window_end_1).dt.days
    P['start_gap'] = (P.window_start_2 - P.window_start_1).dt.days
    # ---- G2 exclusion
    with duckdb.connect(str(DB), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    n_before = len(P); n_removed = int(P.USER_ID.isin(excl).sum())
    P = P.loc[~P.USER_ID.isin(excl)].sort_values('USER_ID').reset_index(drop=True)
    g2 = dict(excluded_users=len(excl), excluded_users_diabetes_exclusion=m_diabetes_exclusion['guards']['excluded_users'], pairs_before=n_before,
              removed=n_removed, primary_n=len(P), n_second_week_repeatability=m_repeatability['n_second_week'])
    guards['G2_exclusion_and_n'] = g2
    if not (len(excl) == m_diabetes_exclusion['guards']['excluded_users'] and len(P) == m_repeatability['n_second_week']):
        fail(f'G2 {g2}')

    # ---- G3 pooled ICC and CI reproduce 06 axis_reliability.csv (structural consistency check)
    AR = pd.read_csv(REP / 'axis_reliability.csv').set_index('metric')
    g3 = {}
    for i, m in enumerate(METRICS):
        r = s_repeatability.summarize_pair(P[m + '_1'].to_numpy(), P[m + '_2'].to_numpy(), len(P), m_repeatability['bootstrap_replicates'], s_repeatability.SEED + i)
        d = max(abs(r[c] - AR.loc[m, c]) for c in ['icc_a1', 'ci_low', 'ci_high'])
        g3[m] = dict(icc=r['icc_a1'], ci_low=r['ci_low'], ci_high=r['ci_high'], n=r['n_users'], max_abs_diff=float(d),
                     rounded=round(r['icc_a1'], 3))
        if not (d < 1e-12 and r['n_users'] == AR.loc[m, 'n_users']):
            fail(f'G3 {m} {g3[m]}')
    guards['G3_pooled_reproduces_repeatability'] = g3

    # ---- G4 edge-gap strata reproduced from this gap definition (structural consistency check)
    GS = pd.read_csv(REP / 'gap_stratified_reliability.csv').set_index(['metric', 'stratum'])
    if not (P.start_gap == P.edge_gap + 7).all():
        fail('start gap != edge gap + 7')
    g4 = {}
    for i, m in enumerate(METRICS):
        for label, lo, hi in [('adjacent_0d', 0, 0), ('gap_1_30d', 1, 30), ('gap_over_30d', 31, 10 ** 9)]:
            sub = P[(P.start_gap - 7).between(lo, hi)]
            r = s_repeatability.summarize_pair(sub[m + '_1'].to_numpy(), sub[m + '_2'].to_numpy(), len(sub), m_repeatability['bootstrap_replicates'], s_repeatability.SEED + i)
            row = GS.loc[(m, label)]
            d = max(abs(r[c] - row[c]) for c in ['icc_a1', 'ci_low', 'ci_high'])
            g4[f'{m}:{label}'] = dict(n=r['n_users'], max_abs_diff=float(d))
            if not (d < 1e-12 and r['n_users'] == row['n']):
                fail(f'G4 {m} {label} {r} vs {row.to_dict()}')
    guards['G4_edge_gap_strata_reproduce_repeatability'] = g4

    # ---- G6 day-level substrate
    dcon = duckdb.connect(); dcon.register('w', W.loc[W.USER_ID.isin(P.USER_ID), ['USER_ID', 'window_number', 'window_start']])
    D = dcon.execute(f"""SELECT w.USER_ID, w.window_number,
          datediff('day', CAST(w.window_start AS DATE), CAST(g.LOCAL_DAY AS DATE)) AS pos, g.MEAN, g.N_READINGS, g.N_CELLS_15M
        FROM w JOIN read_parquet('{GLUC}') g ON g.USER_ID = w.USER_ID
          AND CAST(g.LOCAL_DAY AS DATE) BETWEEN CAST(w.window_start AS DATE) AND CAST(w.window_start AS DATE) + 6""").df()
    dcon.close()
    cnt = D.groupby(['USER_ID', 'window_number']).pos.agg(['size', 'nunique', 'min', 'max'])
    mats = {}
    for wn in (1, 2):
        piv = D[D.window_number.eq(wn)].pivot(index='USER_ID', columns='pos', values='MEAN').reindex(P.USER_ID)
        mats[wn] = piv.reindex(columns=range(K)).to_numpy(float)
    M1, M2 = mats[1], mats[2]
    corr1 = float(np.corrcoef(M1.mean(1), P.mean_glucose_1)[0, 1]); corr2 = float(np.corrcoef(M2.mean(1), P.mean_glucose_2)[0, 1])
    g6 = dict(windows=int(len(cnt)), all_seven_distinct_days=bool((cnt['size'].eq(7) & cnt['nunique'].eq(7) & cnt['min'].eq(0) & cnt['max'].eq(6)).all()),
              complete_matrices=bool(np.isfinite(M1).all() and np.isfinite(M2).all()),
              min_readings=int(D.N_READINGS.min()), min_cells=int(D.N_CELLS_15M.min()),
              corr_daily_mean_vs_window_mean=[corr1, corr2],
              max_abs_diff_daily_mean_vs_window_mean=float(max(np.abs(M1.mean(1) - P.mean_glucose_1).max(), np.abs(M2.mean(1) - P.mean_glucose_2).max())))
    guards['G6_day_level_substrate'] = g6
    if not (g6['windows'] == 2 * len(P) and g6['all_seven_distinct_days'] and g6['complete_matrices']
            and g6['min_readings'] >= 230 and g6['min_cells'] >= 77 and min(corr1, corr2) > 0.9999):
        fail(f'G6 {g6}')

    # ---- G5 vectorized estimator and bins
    rng = np.random.default_rng(SEED - 1); test = rng.integers(0, len(P), (20, len(P)))
    d5 = 0.0
    for m in METRICS:
        x, y = P[m + '_1'].to_numpy(float), P[m + '_2'].to_numpy(float)
        d5 = max(d5, abs(icc_a1_vec(x, y, np.arange(len(P))[None])[0] - s_repeatability.icc_a1(x, y)))
        v = icc_a1_vec(x, y, test)
        d5 = max(d5, max(abs(v[b] - s_repeatability.icc_a1(x[test[b]], y[test[b]])) for b in range(20)))
    bin_n = {b: int(P.start_gap.between(lo, hi).sum()) for b, _, lo, hi in BINS}
    g5 = dict(max_abs_diff_vectorized_vs_repeatability=float(d5), bin_n=bin_n, bins_partition=sum(bin_n.values()) == len(P),
              min_start_gap=int(P.start_gap.min()))
    guards['G5_estimator_and_bins'] = g5
    if not (d5 < 1e-10 and g5['bins_partition'] and g5['min_start_gap'] == 7):
        fail(f'G5 {g5}')
    guards['all_passed'] = True

    # ---- estimates
    groups = [('pooled', 'All gaps', 7, 10 ** 9)] + BINS + SUBBINS
    res = {}
    for gi, (g, glabel, lo, hi) in enumerate(groups):
        mask = P.start_gap.between(lo, hi).to_numpy()
        point, reps = analyse_group(P.loc[mask], M1[mask], M2[mask], SEED + 1000 * gi)
        sg = P.loc[mask, 'start_gap']
        res[g] = dict(label=glabel, lo=lo, hi=hi, n=int(mask.sum()), median_gap=float(sg.median()),
                      q25_gap=float(sg.quantile(.25)), q75_gap=float(sg.quantile(.75)), min_gap=int(sg.min()),
                      max_gap=int(sg.max()), point=point, reps=reps, sub=P.loc[mask])
        print(f'{g}: n={mask.sum():,}', flush=True)
    if any(abs(res['pooled']['point'][f'icc_{m}'] - g3[m]['icc']) > 1e-10 for m in METRICS):
        fail('pooled point estimate changed between guard and estimate stage')

    def gap_rows(names):
        rows = []
        for g in names:
            r = res[g]
            for m in METRICS:
                lo_, hi_ = ci(r['reps'][f'icc_{m}']); x, y = r['sub'][m + '_1'].to_numpy(), r['sub'][m + '_2'].to_numpy()
                rows.append(dict(metric=m, metric_label=LABEL[m], gap_bin=g, gap_label=r['label'],
                                 start_gap_min_def=r['lo'], start_gap_max_def=None if r['hi'] >= 10 ** 9 else r['hi'],
                                 n=r['n'], median_start_gap_days=r['median_gap'], q25_start_gap_days=r['q25_gap'],
                                 q75_start_gap_days=r['q75_gap'], icc_a1=r['point'][f'icc_{m}'], ci_low=lo_, ci_high=hi_,
                                 pearson_r=float(np.corrcoef(x, y)[0, 1]), median_abs_diff=float(np.median(np.abs(x - y))),
                                 n_boot=N_BOOT))
        return pd.DataFrame(rows)

    by_gap = gap_rows(['pooled'] + [b[0] for b in BINS])
    subbins = gap_rows([b[0] for b in SUBBINS])
    contrasts = []
    for g in [b[0] for b in BINS[1:]]:
        for m in METRICS:
            key = f'icc_{m}'; diff = res['back_to_back_7d']['point'][key] - res[g]['point'][key]
            lo_, hi_ = ci(res['back_to_back_7d']['reps'][key] - res[g]['reps'][key])
            contrasts.append(dict(metric=m, contrast=f'back_to_back_7d minus {g}', n_back_to_back=res['back_to_back_7d']['n'],
                                  n_other=res[g]['n'], difference=diff, ci_low=lo_, ci_high=hi_,
                                  ci_method='independent resample sets per bin'))
    contrasts = pd.DataFrame(contrasts)

    day_rows = []; proj_rows = []
    pr = res['pooled']
    for wk in (1, 2):
        r = pr['point'][f'icc1_week{wk}']; b = pr['reps'][f'icc1_week{wk}']
        row = dict(week=wk, n=pr['n'], days_per_person=K, icc1_day=r)
        row['icc1_ci_low'], row['icc1_ci_high'] = ci(b)
        row['days_for_0_80'] = days_needed(r); row['days_for_0_80_ci_low'], row['days_for_0_80_ci_high'] = ci(days_needed(b))
        row['projected_7_days'] = spearman_brown(r, K)
        row['projected_7_days_ci_low'], row['projected_7_days_ci_high'] = ci(spearman_brown(b, K))
        day_rows.append(row)
    day_level = pd.DataFrame(day_rows)
    for g in ['pooled'] + [b[0] for b in BINS]:
        r = res[g]
        for k in range(1, K + 1):
            if g == 'pooled':
                for wk in (1, 2):
                    b = spearman_brown(r['reps'][f'icc1_week{wk}'], k); lo_, hi_ = ci(b)
                    proj_rows.append(dict(curve=f'projection_within_week{wk}', gap_bin=g, days=k, n=r['n'],
                                          estimate=spearman_brown(r['point'][f'icc1_week{wk}'], k), ci_low=lo_, ci_high=hi_,
                                          estimand='Spearman-Brown projection of within-week day-level ICC(1)'))
            lo_, hi_ = ci(r['reps'][f'obs_k{k}'])
            proj_rows.append(dict(curve='observed_across_weeks', gap_bin=g, days=k, n=r['n'], estimate=r['point'][f'obs_k{k}'],
                                  ci_low=lo_, ci_high=hi_,
                                  estimand='ICC(A,1), mean of first k daily means of week 1 vs of week 2'))
            if g == 'pooled':
                b = spearman_brown(r['reps']['icc1_week1'], k) - r['reps'][f'obs_k{k}']; lo_, hi_ = ci(b)
                proj_rows.append(dict(curve='projection_week1_minus_observed', gap_bin=g, days=k, n=r['n'],
                                      estimate=spearman_brown(r['point']['icc1_week1'], k) - r['point'][f'obs_k{k}'],
                                      ci_low=lo_, ci_high=hi_, estimand='difference, shared resamples'))
    proj = pd.DataFrame(proj_rows)

    files = {'reliability_by_gap.csv': by_gap, 'reliability_by_gap_weekly_subbins.csv': subbins,
             'gap_contrasts.csv': contrasts, 'day_level_icc.csv': day_level, 'projection_vs_observed.csv': proj}
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
    pick = lambda df, **kw: df.loc[np.logical_and.reduce([df[c].eq(v) for c, v in kw.items()])].iloc[0].to_dict()
    obs = lambda k: pick(proj, curve='observed_across_weeks', gap_bin='pooled', days=k)
    summary = dict(
        n=int(len(P)),
        pooled={m: {k: pick(by_gap, metric=m, gap_bin='pooled')[k] for k in ['icc_a1', 'ci_low', 'ci_high', 'n']} for m in METRICS},
        by_gap_average_glucose={g: {k: pick(by_gap, metric='mean_glucose', gap_bin=g)[k] for k in
                                    ['n', 'median_start_gap_days', 'icc_a1', 'ci_low', 'ci_high']} for g, *_ in BINS},
        within_week=day_level.to_dict('records'),
        observed_one_day_vs_one_day={k: obs(1)[k] for k in ['estimate', 'ci_low', 'ci_high', 'n']},
        observed_one_week_vs_one_week_daily_means={k: obs(7)[k] for k in ['estimate', 'ci_low', 'ci_high', 'n']},
        observed_one_week_vs_one_week_window_features={k: pick(by_gap, metric='mean_glucose', gap_bin='pooled')[k] for k in ['icc_a1', 'ci_low', 'ci_high', 'n']})
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='reliability_by_gap', version=VERSION, status='passed',
                design='descriptive; ICC(A,1) (repeatability.icc_a1) by gap between window starts; one-way ICC(1) and Spearman-Brown',
                seed=SEED, n_boot=N_BOOT,
                definitions=dict(population='analysis/reliability/repeatability.py paired seven-day windows minus analysis/validation/diabetes_range_exclusion.py diabetes_range_users',
                                 gap='days between window starts (window_start_2 - window_start_1); 7 = back-to-back',
                                 bins={b: [lo, None if hi >= 10 ** 9 else hi] for b, _, lo, hi in BINS},
                                 subbins={b: [lo, hi] for b, _, lo, hi in SUBBINS},
                                 daily_average_glucose='gluc_local_day MEAN (raw readings), 7 dense days per window',
                                 days_needed='m = 0.8(1-r)/(0.2 r) from week-specific ICC(1)',
                                 ci='percentile, participant bootstrap; resamples shared within a bin; contrasts across bins use independent sets'),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'reliability_by_gap.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
