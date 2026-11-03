#!/usr/bin/env python3
"""Associations of CGM level, day-part, excursion and meal-response measures with insulin markers.

Cross-sectional associations only; no projection. Population: canonical primary cohort
(outputs/canonical_panel, no observed diabetes-range labs). HOMA-IR is the
primary outcome; fasting insulin, triglycerides, TG:HDL and TyG are companion markers.
BMI models are complete-case (no imputation).

Sections
  A  CGM measures vs log(HOMA-IR): age+sex; + mean glucose; + BMI; + mean glucose + BMI.
  B  Logged-meal responses (paper meal tier, meals within +/-90 local days of the HbA1c anchor,
     >=10 meals per person) vs HOMA-IR and companion markers.
  C  HbA1c/CGM discordance groups vs log(HOMA-IR), with and without BMI.
  D  Carbohydrate sensitivity (analysis/meals/meal_populations.py window_mixed adjusted slope)
     high vs low half: adjusted geometric means of each marker (age+sex; + BMI; + BMI + mean glucose).
  E  Pre-specified day-part measure night_minus_day = mean_night (00-05 local) - mean_midday
     (11-15 local), chosen from the section-A screen in the HOMA-IR sample. Confirmed in
     participants with fasting insulin but no HOMA-IR (disjoint from discovery) and, for a
     glucose-free marker, triglycerides in participants without HOMA-IR; plus 200 seeded
     random participant halves of the HOMA-IR sample.
  F  Average CGM glucose by rounded HbA1c value (decimal half-up), primary cohort.

Inputs
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/meals/spike_facets/spike_facets_participants.parquet and its metadata
    (analysis/meals/spike_facets.py)
  OUTPUTS/meals/meal_shape_substrate.parquet (analysis/meals/build_meals.py)
  OUTPUTS/meals/meal_fpca_per_user.csv (analysis/meals/meal_fpca.py)
  OUTPUTS/meal_populations/_cache/window_mixed_participants.parquet and
    window_mixed_correlations.csv (analysis/meals/meal_populations.py)

Outputs (OUTPUTS/metabolic/insulin_markers/)
  cgm_measures_by_marker.csv, meal_responses_by_marker.csv, discordance_homa_bmi.csv,
  discordance_group_counts.csv, carb_sensitivity_high_low.csv, carb_sensitivity_groups.csv,
  night_minus_day_confirmation.csv, night_minus_day_split_halves.csv, night_minus_day_quarters.csv,
  hba1c_mean_cgm_table.csv, hba1c_mean_cgm_steps.csv, summary.json,
  insulin_markers.metadata.json

Usage
  python analysis/metabolic/insulin_markers.py [--out OUTPUT_DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402

VERSION = 'insulin-markers'
PUB = OUTPUTS
OUT = PUB / 'metabolic/insulin_markers'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
FACETS_META = PUB / 'meals/spike_facets/spike_facets.metadata.json'
MEALS = PUB / 'meals/meal_shape_substrate.parquet'
FPCA = PUB / 'meals/meal_fpca_per_user.csv'
CARB = PUB / 'meal_populations/_cache/window_mixed_participants.parquet'
CARB_CORR = PUB / 'meal_populations/window_mixed_correlations.csv'
SEED = 20260926
N_SPLITS = 200
MIN_MEALS = 10
WINDOW_DAYS = 90
MARKERS = {'homa_ir': True, 'fasting_insulin': True, 'triglycerides': True, 'tg_hdl_ratio': True, 'tyg': False}
CGM_MEASURES = ['mean_glucose', 'night_minus_day', 'mean_night', 'mean_midday', 'diurnal_amp', 'sd_glucose', 'cv_glucose',
                'modd', 'mage', 'all_frequency', 'all_size', 'all_clearance']
MEAL_MEASURES = ['premeal_baseline', 'peak_rise', 'iauc_2h', 'rise_beyond_content', 'share_rise_ge30', 'time_to_peak',
                 'return_fraction', 'return_minutes', 'fpc1_mean', 'fpc2_mean', 'fpc3_mean']
ADJ = {'age_sex': ['age', 'male'], 'mean_glucose': ['age', 'male', 'mean_glucose'], 'bmi': ['age', 'male', 'bmi'],
       'mean_glucose_bmi': ['age', 'male', 'mean_glucose', 'bmi']}


def fit(s, x, adj, y='y', log=True):
    """OLS (HC3) of y on standardized x and covariates; returns % per SD (log) or units per SD."""
    need = [x, y] + [c for c in adj if c != 'age']
    s = s.dropna(subset=need).reset_index(drop=True)
    X = s[[x] + [c for c in adj if c != 'age']].astype(float)
    X = (X - X.mean()) / X.std()
    if 'age' in adj:
        ages = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
        X = X.join(ages.loc[:, ages.sum() >= 10])
    m = sm.OLS(s[y].to_numpy(), sm.add_constant(X)).fit(cov_type='HC3')
    b, (lo, hi) = m.params[x], m.conf_int().loc[x]
    f = (lambda v: 100 * (np.exp(v) - 1)) if log else (lambda v: v)
    return dict(n=int(len(s)), estimate=float(f(b)), ci_low=float(f(lo)), ci_high=float(f(hi)), p=float(m.pvalues[x]),
                scale='percent_per_sd' if log else 'units_per_sd')


def family(frame, measures, marker, log, adj_sets, min_n=150):
    rows = []
    d = frame.copy()
    d = d.loc[d[marker] > 0] if log else d
    d['y'] = np.log(d[marker]) if log else d[marker]
    for adj_name, adj in adj_sets.items():
        block = []
        for x in measures:
            a = [c for c in adj if c != x]
            if x == 'mean_glucose' and 'mean_glucose' in adj:
                continue
            s = d.dropna(subset=[x])
            if len(s) < min_n or s[x].std() == 0:
                continue
            block.append(dict(marker=marker, adjustment=adj_name, measure=x, **fit(s, x, a, log=log)))
        if block:
            q = multipletests([r['p'] for r in block], method='fdr_bh')[1]
            for r, qq in zip(block, q):
                r['q_fdr_family'] = float(qq)
            rows += block
    return rows


def load(args):
    guards = {}
    panel = pd.read_csv(PANEL, low_memory=False)
    P = panel.loc[panel.no_diabetes_range == True].copy()  # noqa: E712
    P['male'] = (P.sex == 'male').astype(float)
    P['night_minus_day'] = P.mean_night - P.mean_midday
    P = P.merge(pd.read_parquet(FACETS, columns=['USER_ID', 'all_frequency', 'all_size', 'all_clearance']), on='USER_ID', how='left')
    return panel, P, guards


def meal_responses(P):
    cols = ['USER_ID', 'MEAL_TS', 'tz_offset', 'CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G', 'BASELINE_PRE', 'IAUC_2H',
            'PEAK_RISE_2H', 'TIME_TO_PEAK_MIN', 'RETURN_10_MIN', 'RETURN_10_REACHED']
    M = pd.read_parquet(MEALS, columns=cols).merge(P[['USER_ID', 'anchor_date']].assign(anchor_date=lambda d: pd.to_datetime(d.anchor_date)),
                                                   on='USER_ID')
    local = pd.to_datetime(M.MEAL_TS, utc=True).dt.tz_localize(None) + pd.to_timedelta(M.tz_offset.fillna(0), unit='h')
    M = M.loc[(local.dt.normalize() - M.anchor_date).dt.days.abs() <= WINDOW_DAYS].copy()
    X = sm.add_constant(M[['CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G']].fillna(0))
    M['rise_resid'] = M.PEAK_RISE_2H - sm.OLS(M.PEAK_RISE_2H, X).fit().predict(X)
    M['ret_min'] = np.where(M.RETURN_10_REACHED == True, M.RETURN_10_MIN, np.nan)  # noqa: E712
    M['rise30'] = (M.PEAK_RISE_2H >= 30).astype(float)
    g = M.groupby('USER_ID')
    R = pd.DataFrame({'n_meals': g.size(), 'premeal_baseline': g.BASELINE_PRE.mean(), 'peak_rise': g.PEAK_RISE_2H.mean(),
                      'iauc_2h': g.IAUC_2H.mean(), 'rise_beyond_content': g.rise_resid.mean(), 'time_to_peak': g.TIME_TO_PEAK_MIN.mean(),
                      'return_fraction': g.RETURN_10_REACHED.mean(), 'return_minutes': g.ret_min.mean(),
                      'share_rise_ge30': g.rise30.mean()}).reset_index()
    R = R.loc[R.n_meals >= MIN_MEALS]
    fp = pd.read_csv(FPCA, usecols=['USER_ID', 'fpc1_mean', 'fpc2_mean', 'fpc3_mean'])
    return R.merge(fp, on='USER_ID', how='left'), dict(meals_in_window=int(len(M)), participants_ge_min_meals=int(len(R)))


def discordance(P, guards):
    H = P.loc[P.homa_ir.notna() & (P.homa_ir > 0)].copy()
    H['y'] = np.log(H.homa_ir)
    a1c_hi, cgm_hi = round_a1c(H.a1c.to_numpy()) >= 5.7, H.mean_glucose.to_numpy() >= 110
    H['group'] = np.select([~a1c_hi & ~cgm_hi, ~a1c_hi & cgm_hi, a1c_hi & ~cgm_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    counts = H.groupby('group').agg(n=('y', 'size'), ir=('homa_ir', lambda v: int((v >= 2.5).sum())))
    guards['discordance_counts'] = {k: [int(v.ir), int(v.n)] for k, v in counts.iterrows()}
    rows = []
    for name, sub, adj in [('age_sex', H, ['age', 'male']), ('age_sex_bmi_subset', H.loc[H.bmi.notna()], ['age', 'male']),
                           ('bmi', H.loc[H.bmi.notna()], ['age', 'male', 'bmi'])]:
        s = sub.reset_index(drop=True)
        X = pd.get_dummies(s.group, dtype=float).drop(columns='both_normal')
        ages = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
        X = X.join(ages.loc[:, ages.sum() >= 10]); X['male'] = s.male
        if 'bmi' in adj:
            X['bmi'] = (s.bmi - s.bmi.mean()) / s.bmi.std()
        m = sm.OLS(s.y.to_numpy(), sm.add_constant(X)).fit(cov_type='HC3'); ci = m.conf_int()
        for grp in ['cgm_only', 'a1c_only', 'both_elevated']:
            rows.append(dict(adjustment=name, group=grp, n=int(len(s)), n_group=int((s.group == grp).sum()),
                             pct_vs_both_normal=float(100 * (np.exp(m.params[grp]) - 1)),
                             ci_low=float(100 * (np.exp(ci.loc[grp, 0]) - 1)), ci_high=float(100 * (np.exp(ci.loc[grp, 1]) - 1)),
                             p=float(m.pvalues[grp])))
    bmi = H.groupby('group').bmi.agg(['mean', 'count']).reset_index().rename(columns={'mean': 'bmi_mean', 'count': 'n_bmi'})
    return pd.DataFrame(rows), counts.reset_index().merge(bmi, on='group')


def carbohydrate(guards):
    W = pd.read_parquet(CARB)
    s = W.dropna(subset=['homa_ir'])
    r = float(np.corrcoef(s.acs, s.homa_ir)[0, 1])
    guards['carb_homa_pearson'] = r
    W['high'] = (W.acs > W.acs.median()).astype(int)
    W['male'] = (W.sex == 'male').astype(float)
    rows = []
    for marker, log in [('homa_ir', True), ('fasting_insulin', True), ('fasting_glucose', True), ('a1c', False),
                        ('triglycerides', True), ('tg_hdl_ratio', True), ('hdl_cholesterol', True), ('apo_b', True)]:
        for name, adj in [('age_sex', []), ('bmi', ['bmi']), ('bmi_mean_glucose', ['bmi', 'mean_glucose'])]:
            d = W.dropna(subset=[marker] + adj)
            d = d.loc[d[marker] > 0] if log else d
            d = d.loc[d.age_bucket.map(d.age_bucket.value_counts()) >= 10].reset_index(drop=True)
            y = np.log(d[marker]) if log else d[marker]
            X = pd.get_dummies(d.age_bucket, prefix='age', drop_first=True, dtype=float)
            X['male'] = d.male; X['high'] = d.high.astype(float)
            for c in adj:
                X[c] = d[c]
            m = sm.OLS(y.to_numpy(), sm.add_constant(X)).fit(cov_type='HC3')
            pred = {g: float(np.mean(m.predict(sm.add_constant(X.assign(high=float(g)), has_constant='add')))) for g in (0, 1)}
            f = np.exp if log else (lambda v: v)
            lo, hi = m.conf_int().loc['high']
            rows.append(dict(marker=marker, adjustment=name, n=int(len(d)), adjusted_mean_low=float(f(pred[0])),
                             adjusted_mean_high=float(f(pred[1])), effect=float(f(m.params['high'])), ci_low=float(f(lo)), ci_high=float(f(hi)),
                             effect_scale='ratio_high_low' if log else 'difference_high_low', p=float(m.pvalues['high']),
                             spearman_rho=float(stats.spearmanr(d.acs, d[marker])[0])))
    desc = W.groupby('high').agg(n=('acs', 'size'), slope_median=('acs', 'median'), male_pct=('male', lambda v: 100 * v.mean()),
                                 carbs_median_g=('carbs', 'median'), mean_glucose=('mean_glucose', 'mean'), bmi=('bmi', 'mean')).reset_index()
    return pd.DataFrame(rows), desc


def day_part_confirmation(P):
    rng = np.random.default_rng(SEED)
    H = P.loc[P.homa_ir.notna() & (P.homa_ir > 0)].copy(); H['y'] = np.log(H.homa_ir)
    rows = []
    # disjoint confirmation samples
    I = P.loc[P.homa_ir.isna() & P.fasting_insulin.notna() & (P.fasting_insulin > 0)].copy(); I['y'] = np.log(I.fasting_insulin)
    T = P.loc[P.homa_ir.isna() & P.triglycerides.notna() & (P.triglycerides > 0)].copy(); T['y'] = np.log(T.triglycerides)
    for sample, d, marker in [('discovery_homa', H, 'homa_ir'), ('confirm_insulin_without_homa', I, 'fasting_insulin'),
                              ('confirm_triglycerides_without_homa', T, 'triglycerides')]:
        for name in ['mean_glucose', 'mean_glucose_bmi']:
            rows.append(dict(sample=sample, marker=marker, adjustment=name, **fit(d, 'night_minus_day', ADJ[name])))
    # random halves of the discovery sample
    halves = []
    ids = H.USER_ID.to_numpy()
    for i in range(N_SPLITS):
        a = set(rng.choice(ids, len(ids) // 2, replace=False))
        ra = fit(H.loc[H.USER_ID.isin(a)], 'night_minus_day', ADJ['mean_glucose'])
        rb = fit(H.loc[~H.USER_ID.isin(a)], 'night_minus_day', ADJ['mean_glucose'])
        halves.append(dict(split=i, a_estimate=ra['estimate'], a_p=ra['p'], b_estimate=rb['estimate'], b_p=rb['p'], n_a=ra['n'], n_b=rb['n']))
    hv = pd.DataFrame(halves)
    split_summary = dict(n_splits=N_SPLITS, both_halves_positive=float((hv[['a_estimate', 'b_estimate']] > 0).all(axis=1).mean()),
                         both_halves_p_lt_05=float(((hv.a_p < .05) & (hv.b_p < .05)).mean()),
                         half_estimate_median=float(pd.concat([hv.a_estimate, hv.b_estimate]).median()),
                         half_estimate_q025=float(pd.concat([hv.a_estimate, hv.b_estimate]).quantile(.025)),
                         half_estimate_q975=float(pd.concat([hv.a_estimate, hv.b_estimate]).quantile(.975)))
    # plain-language quarters among same-average peers
    q = []
    for sample, d, marker, thr in [('discovery_homa', H, 'homa_ir', 2.5), ('confirm_insulin_without_homa', I, 'fasting_insulin', None)]:
        d = d.dropna(subset=['night_minus_day']).copy()
        d['bin20'] = pd.qcut(d.mean_glucose, 20, labels=False)
        d['rel'] = d.night_minus_day - d.groupby('bin20').night_minus_day.transform('median')
        d['quarter'] = pd.qcut(d.rel, 4, labels=[1, 2, 3, 4]).astype(int)
        for k, s in d.groupby('quarter'):
            q.append(dict(sample=sample, marker=marker, quarter=int(k), n=int(len(s)), night_minus_day_median=float(s.night_minus_day.median()),
                          mean_glucose_mean=float(s.mean_glucose.mean()), marker_geometric_mean=float(np.exp(s.y.mean())),
                          pct_homa_ir_ge_2_5=(float(100 * (s.homa_ir >= thr).mean()) if thr else None), bmi_mean=float(s.bmi.mean()),
                          n_bmi=int(s.bmi.notna().sum())))
    return pd.DataFrame(rows), hv, split_summary, pd.DataFrame(q)


def hba1c_table(P):
    d = P.copy(); d['a1c_r'] = round_a1c(d.a1c.to_numpy())
    rng = np.random.default_rng(SEED)
    rows = []
    for v, s in d.groupby('a1c_r'):
        if len(s) < 20:
            continue
        bs = [rng.choice(s.mean_glucose.to_numpy(), len(s)).mean() for _ in range(1000)]
        rows.append(dict(hba1c=float(v), n=int(len(s)), mean_cgm=float(s.mean_glucose.mean()), ci_low=float(np.percentile(bs, 2.5)),
                         ci_high=float(np.percentile(bs, 97.5)), median_cgm=float(s.mean_glucose.median())))
    t = pd.DataFrame(rows).set_index('hba1c')
    steps = []
    for a, b in [(5.0, 5.5), (5.5, 6.0), (6.0, 6.4)]:
        if a in t.index and b in t.index:
            steps.append(dict(from_hba1c=a, to_hba1c=b, mean_cgm_from=float(t.loc[a, 'mean_cgm']), mean_cgm_to=float(t.loc[b, 'mean_cgm']),
                              difference_mg_dl=float(t.loc[b, 'mean_cgm'] - t.loc[a, 'mean_cgm'])))
    return t.reset_index(), pd.DataFrame(steps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    panel, P, guards = load(args)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    # A
    rowsA = []
    for marker, log in MARKERS.items():
        rowsA += family(P, CGM_MEASURES, marker, log, ADJ)
    A = pd.DataFrame(rowsA)
    # B
    R, meal_counts = meal_responses(P)
    PB = P.merge(R, on='USER_ID')
    rowsB = []
    for marker, log in MARKERS.items():
        rowsB += family(PB, MEAL_MEASURES, marker, log, {k: ADJ[k] for k in ('age_sex', 'mean_glucose', 'mean_glucose_bmi')})
    B = pd.DataFrame(rowsB)
    C, Cc = discordance(P, guards)
    D, Dd = carbohydrate(guards)
    E, Eh, Es, Eq = day_part_confirmation(P)
    F, Fs = hba1c_table(P)
    H = P.loc[P.homa_ir.notna()]
    samples = dict(primary_n=int(len(P)), homa_n=int(len(H)), homa_with_bmi=int(H.bmi.notna().sum()),
                   fasting_insulin_n=int(P.fasting_insulin.notna().sum()), triglycerides_n=int(P.triglycerides.notna().sum()),
                   tyg_n=int(P.tyg.notna().sum()), tg_hdl_n=int(P.tg_hdl_ratio.notna().sum()),
                   tyg_vs_log_homa_r=float(np.corrcoef(P.dropna(subset=['tyg', 'homa_ir']).tyg, np.log(P.dropna(subset=['tyg', 'homa_ir']).homa_ir))[0, 1]),
                   **meal_counts)
    files = {'cgm_measures_by_marker.csv': A, 'meal_responses_by_marker.csv': B, 'discordance_homa_bmi.csv': C,
             'discordance_group_counts.csv': Cc, 'carb_sensitivity_high_low.csv': D, 'carb_sensitivity_groups.csv': Dd,
             'night_minus_day_confirmation.csv': E, 'night_minus_day_split_halves.csv': Eh, 'night_minus_day_quarters.csv': Eq,
             'hba1c_mean_cgm_table.csv': F, 'hba1c_mean_cgm_steps.csv': Fs}
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
    summary = dict(samples=samples, night_minus_day_split_halves=Es)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    meta = dict(analysis='insulin_markers', version=VERSION,
                design='cross-sectional associations; no projection', seed=SEED, guards=guards,
                definitions=dict(night='00:00-05:00 local', midday='11:00-15:00 local', meal_window_days=WINDOW_DAYS, min_meals=MIN_MEALS,
                                 homa_threshold=2.5, discordance_cutoffs=dict(hba1c=5.7, mean_cgm=110)),
                runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'insulin_markers.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(guards=guards, samples=samples, split=Es), indent=1, default=float))


if __name__ == '__main__':
    main()
