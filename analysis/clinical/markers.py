#!/usr/bin/env python3
"""Compares average glucose with HbA1c as predictors of other laboratory markers of metabolic health.

Primary cohort: canonical panel participants with no diabetes-range laboratory value. Marker values are the canonical
panel values (value nearest the anchor HbA1c draw), log-transformed.
Per marker, in the same participants (marker, average glucose and HbA1c all present):
  R2 of log marker on average glucose alone and on HbA1c alone; difference with a paired participant bootstrap 95% CI;
  % difference per SD of average glucose and, separately, per SD of HbA1c, adjusted for age decade and sex (OLS, HC3);
  % per SD of average glucose additionally adjusted for BMI (BMI-complete participants).
Fasting insulin and HOMA-IR rows are included as the paper's reference comparison.
hsCRP alongside fasting insulin in the paper's other comparisons (same estimators as for fasting insulin):
  spikes      % per SD of spike clearance time, spike size and spike frequency (spike_facets, the paper's measures) at the
              same average glucose (OLS, HC3), for fasting insulin, triglycerides and hsCRP -> spike_markers.csv
  discordance geometric means by average glucose >=110 mg/dL and HbA1c >=5.7% (HbA1c rounded half-up to 0.1); ratio of
              CGM-elevated-only to HbA1c-elevated-only (OLS of the log marker on group, HC3) -> discordance_markers.csv
  carb        higher vs lower half of carbohydrate sensitivity (window slopes, median split over all participants),
              unadjusted ratio of geometric means (OLS, HC3) -> carb_markers.csv

Inputs: OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py);
OUTPUTS/meals/spike_facets/spike_facets_participants.parquet (analysis/meals/spike_facets.py);
OUTPUTS/clinical/_cache/spike_width_participants.parquet and OUTPUTS/clinical/spike_width/width_associations.csv
(analysis/clinical/spike_width.py); OUTPUTS/metabolic/unadjusted_insulin_markers/carb_sensitivity_unadjusted.csv
(analysis/metabolic/unadjusted_insulin_markers.py); OUTPUTS/metabolic/carb_sensitivity_continuous/* (analysis/metabolic/carb_sensitivity_continuous.py);
OUTPUTS/meals/meal_shape_macro_adjusted/person_features.parquet (analysis/meals/meal_shape_macro_adjusted.py).
Outputs: OUTPUTS/clinical/markers/markers_avg_vs_a1c.csv, spike_markers.csv, discordance_markers.csv,
discordance_high_share.csv, carb_markers.csv, carb_markers_continuous.csv, carb_markers_quintiles.csv,
carb_spike_rho.json, metadata.json. Aggregate only.
Usage: python analysis/clinical/markers.py
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings('ignore')
PUB = OUTPUTS
OUT = PUB / 'clinical/markers'
DB = database_path()
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
CARB = OUTPUTS / 'meals/meal_shape_macro_adjusted/person_features.parquet'
R_UNADJUSTED = PUB / 'metabolic/unadjusted_insulin_markers/carb_sensitivity_unadjusted.csv'
WIDTH = PUB / 'clinical/_cache/spike_width_participants.parquet'
WASSOC = PUB / 'clinical/spike_width/width_associations.csv'
R_CARB_CONTINUOUS = PUB / 'metabolic/carb_sensitivity_continuous'
SEED, NB = 20261026, 2000
MARKERS = [('fasting_insulin', 'Fasting insulin'), ('homa_ir', 'HOMA-IR'), ('triglycerides', 'Triglycerides'),
           ('hdl_cholesterol', 'HDL cholesterol'), ('tg_hdl_ratio', 'Triglyceride-to-HDL ratio'), ('uric_acid', 'Uric acid'),
           ('hscrp', 'hsCRP'), ('apo_b', 'ApoB'), ('ldl_cholesterol', 'LDL cholesterol')]
GUARDS: dict = {}


def guard(name, ok, detail=None):
    GUARDS[name] = dict(status='passed' if ok else 'failed', detail=detail)
    if not ok:
        raise AssertionError(f'guard failed: {name}: {detail}')


def r2(y, x):
    X = np.column_stack([np.ones(len(y)), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    return 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


def pct(d, x, extra=''):
    d = d.copy(); d['zx'] = (d[x] - d[x].mean()) / d[x].std(ddof=1)
    f = smf.ols(f'ly ~ zx + C(age_bucket) + C(sex){extra}', d).fit(cov_type='HC3'); lo, hi = f.conf_int().loc['zx']
    return [float(100 * np.expm1(v)) for v in (f.params.zx, lo, hi)] + [int(f.nobs)]


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(PANEL, low_memory=False)
    S = P.loc[P.no_diabetes_range.astype(bool)].copy()
    rng = np.random.default_rng(SEED); rows = []
    for mk, label in MARKERS:
        d = S.loc[S[mk] > 0, [mk, 'mean_glucose', 'a1c', 'age_bucket', 'sex', 'bmi']].dropna(subset=[mk, 'mean_glucose', 'a1c']).copy()
        guard(f'n_{mk}', len(d) >= 150, len(d))
        d['ly'] = np.log(d[mk]); y, g, a = d.ly.to_numpy(), d.mean_glucose.to_numpy(), d.a1c.to_numpy()
        ra, rh = r2(y, g), r2(y, a)
        bt = np.empty(NB)
        for b in range(NB):
            i = rng.integers(0, len(y), len(y)); bt[b] = r2(y[i], g[i]) - r2(y[i], a[i])
        dc = d.dropna(subset=['age_bucket', 'sex'])
        av, hb = pct(dc, 'mean_glucose'), pct(dc, 'a1c')
        db_ = dc.dropna(subset=['bmi']); db_ = db_.assign(zb=(db_.bmi - db_.bmi.mean()) / db_.bmi.std(ddof=1))
        ab = pct(db_, 'mean_glucose', ' + zb')
        rows.append(dict(marker=mk, label=label, n=len(d), r2_avg=ra, r2_a1c=rh, r2_diff=ra - rh,
                         r2_diff_lo=float(np.percentile(bt, 2.5)), r2_diff_hi=float(np.percentile(bt, 97.5)),
                         avg_pct=av[0], avg_lo=av[1], avg_hi=av[2], n_adj=av[3], a1c_pct=hb[0], a1c_lo=hb[1], a1c_hi=hb[2],
                         avg_bmi_pct=ab[0], avg_bmi_lo=ab[1], avg_bmi_hi=ab[2], n_bmi=ab[3]))
    T = pd.DataFrame(rows); T.to_csv(OUT / 'markers_avg_vs_a1c.csv', index=False)

    def ols_term(d, form, term):
        f = smf.ols(form, d).fit(cov_type='HC3'); lo, hi = f.conf_int().loc[term]
        return float(np.exp(f.params[term])), float(np.exp(lo)), float(np.exp(hi)), int(f.nobs), float(f.pvalues[term])
    # spikes at the same average glucose
    F = S.merge(pd.read_parquet(FACETS, columns=['USER_ID', 'all_clearance', 'all_size', 'all_frequency']), on='USER_ID', how='left')
    F = F.merge(pd.read_parquet(WIDTH, columns=['USER_ID', 'spike_width']), on='USER_ID', how='left')
    srows = []
    for mk in ('fasting_insulin', 'homa_ir', 'triglycerides', 'hscrp'):
        for x, lab in (('spike_width', 'spike width'), ('all_clearance', 'spike clearance time'), ('all_size', 'spike size'), ('all_frequency', 'spike frequency')):
            d = F.loc[F[mk] > 0, [mk, x, 'mean_glucose']].dropna().copy(); d['ly'] = np.log(d[mk])
            for c in (x, 'mean_glucose'):
                d['z_' + c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
            e, lo, hi, n, pv = ols_term(d, f'ly ~ z_{x} + z_mean_glucose', f'z_{x}')
            srows.append(dict(marker=mk, measure=lab, n=n, pct=100 * (e - 1), lo=100 * (lo - 1), hi=100 * (hi - 1), p=pv))
    SP = pd.DataFrame(srows); SP.to_csv(OUT / 'spike_markers.csv', index=False)
    # discordance groups
    a = np.floor(S.a1c * 10 + 0.5) / 10
    S['grp'] = np.select([(a < 5.7) & (S.mean_glucose < 110), (a >= 5.7) & (S.mean_glucose >= 110), (a < 5.7) & (S.mean_glucose >= 110)],
                         ['both normal', 'both high', 'CGM elevated only'], 'HbA1c elevated only')
    keymap = {'both normal': 'both_normal', 'both high': 'both_high',
                                                                        'CGM elevated only': 'high_avg_normal_a1c', 'HbA1c elevated only': 'normal_avg_high_a1c'}
    drows = []
    for mk in ('fasting_insulin', 'homa_ir', 'triglycerides', 'hscrp'):
        d = S.loc[S[mk] > 0, [mk, 'grp']].dropna().copy(); d['ly'] = np.log(d[mk])
        gm = d.groupby('grp').ly.agg(['mean', 'size'])
        dd = d[d.grp.isin(['CGM elevated only', 'HbA1c elevated only'])].copy(); dd['cgm_only'] = (dd.grp == 'CGM elevated only').astype(float)
        e, lo, hi, n, pv = ols_term(dd, 'ly ~ cgm_only', 'cgm_only')
        drows.append(dict(marker=mk, **{f'gm_{g}': float(np.exp(gm.loc[g, 'mean'])) for g in keymap}, **{f'n_{g}': int(gm.loc[g, 'size']) for g in keymap},
                          cgm_vs_a1c_only_pct=100 * (e - 1), lo=100 * (lo - 1), hi=100 * (hi - 1), p=pv))
    DS = pd.DataFrame(drows); DS.to_csv(OUT / 'discordance_markers.csv', index=False)
    from statsmodels.stats.proportion import proportion_confint
    hrows = []
    for mk, cut in (('fasting_insulin', 10.0), ('homa_ir', 2.5)):
        d = S.loc[S[mk] > 0, [mk, 'grp']].dropna()
        for g in keymap:
            v = d.loc[d.grp == g, mk]; k = int((v >= cut).sum()); lo, hi = proportion_confint(k, len(v), method='wilson')
            hrows.append(dict(marker=mk, threshold=cut, group=g, n=len(v), k=k, pct=100 * k / len(v), lo=100 * lo, hi=100 * hi))
    HS = pd.DataFrame(hrows); HS.to_csv(OUT / 'discordance_high_share.csv', index=False)
    # carbohydrate sensitivity halves
    C = pd.read_parquet(CARB, columns=['USER_ID', 'PEAK_RISE_2H_slope']).dropna()
    C['high'] = (C.PEAK_RISE_2H_slope > C.PEAK_RISE_2H_slope.median()).astype(float)
    CM = C.merge(S[['USER_ID', 'fasting_insulin', 'homa_ir', 'triglycerides', 'hscrp']], on='USER_ID')
    crows = []
    for mk in ('fasting_insulin', 'homa_ir', 'triglycerides', 'hscrp'):
        d = CM.loc[CM[mk] > 0, [mk, 'high']].dropna().copy(); d['ly'] = np.log(d[mk])
        e, lo, hi, n, pv = ols_term(d, 'ly ~ high', 'high')
        crows.append(dict(marker=mk, n=n, n_high=int(d.high.sum()), ratio=e, lo=lo, hi=hi, p=pv,
                          gm_low=float(np.exp(d.ly[d.high == 0].mean())), gm_high=float(np.exp(d.ly[d.high == 1].mean()))))
    CS = pd.DataFrame(crows); CS.to_csv(OUT / 'carb_markers.csv', index=False)
    from scipy import stats
    prow, qrow = [], []
    for mk in ('fasting_insulin', 'homa_ir', 'triglycerides', 'hscrp'):
        d = CM.loc[CM[mk] > 0, [mk, 'PEAK_RISE_2H_slope']].dropna().copy(); d['ly'] = np.log(d[mk])
        d['zs'] = (d.PEAK_RISE_2H_slope - d.PEAK_RISE_2H_slope.mean()) / d.PEAK_RISE_2H_slope.std(ddof=1)
        e, lo, hi, n, pv = ols_term(d, 'ly ~ zs', 'zs')
        q = pd.qcut(d.PEAK_RISE_2H_slope.rank(method='first'), 5, labels=False) + 1; d['q'] = q
        et, lot, hit, _, pvt = ols_term(d, 'ly ~ q', 'q')
        prow.append(dict(marker=mk, n=n, pct_per_sd=100 * (e - 1), lo=100 * (lo - 1), hi=100 * (hi - 1), p=pv,
                         trend_pct_per_quintile=100 * (et - 1), trend_lo=100 * (lot - 1), trend_hi=100 * (hit - 1), trend_p=pvt))
        for k in range(1, 6):
            y = d.ly[d.q == k]; m_, se = y.mean(), y.std(ddof=1) / np.sqrt(len(y)); t = stats.t.ppf(.975, len(y) - 1)
            qrow.append(dict(marker=mk, quintile=k, n=len(y), geometric_mean=float(np.exp(m_)), lo=float(np.exp(m_ - t * se)), hi=float(np.exp(m_ + t * se))))
    CP = pd.DataFrame(prow); CQ = pd.DataFrame(qrow); CP.to_csv(OUT / 'carb_markers_continuous.csv', index=False); CQ.to_csv(OUT / 'carb_markers_quintiles.csv', index=False)
    CW = C.merge(F[['USER_ID', 'spike_width', 'all_clearance']], on='USER_ID').dropna()
    rho = dict(n=len(CW), width=float(stats.spearmanr(CW.PEAK_RISE_2H_slope, CW.spike_width)[0]),
               clearance=float(stats.spearmanr(CW.PEAK_RISE_2H_slope, CW.all_clearance)[0]))
    (OUT / 'carb_spike_rho.json').write_text(json.dumps(rho, indent=1) + '\n')
    meta = dict(analysis='markers', version='markers', status='passed', guards=GUARDS, seed=SEED, bootstrap=NB,
                runtime_seconds=round(time.time() - t0, 1))
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: v['status'] for k, v in GUARDS.items()}), meta['runtime_seconds'])
    print(json.dumps(rho))
    for X in (T, SP, DS, HS, CS, CP, CQ):
        print(X.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
