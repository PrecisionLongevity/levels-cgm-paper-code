#!/usr/bin/env python3
"""Computes spike width (spike duration) and clearance time, and their associations with metabolic markers.

Definitions (per resolved spike; primary spike definition, detector from analysis/meals/spike_exposure.py via
analysis/metabolic/spike_width_check.py):
  spike         local maximum >= 30 mg/dL above the minimum of the preceding 90 observed minutes, 5-min local-day
                cells, no interpolation; peaks < 30 min apart keep the larger rise; day eligible with >= 192 search centres
  half level    preceding minimum + rise/2
  clearance     minutes from the peak to the first cell at or below the half level (censored spikes excluded)
  upslope       minutes from the first cell at or above the half level after the onset minimum to the peak
  width         upslope + clearance = minutes above half the rise ("spike duration")
  participant   mean over resolved spikes on eligible days; >= 5 eligible days and >= 5 resolved spikes
Meal-anchored width: isolated logged meals with macronutrients and CGM coverage within +/-90 days of the anchor, 2-h peak
rise >= 20 mg/dL; half level = 15-min pre-meal mean + rise/2; width = first reading back at or below the half level
after the peak (within 180 min) minus first reading at or above it up to the peak; >= 10 meals per participant.

Inputs:
  database (LEVELS_DB): glucose_readings, lifestyle_events
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/meals/spike_facets/spike_facets_participants.parquet and spike_facets_repeatability.csv
    (analysis/meals/spike_facets.py)
  OUTPUTS/meals/meal_food_substrate.parquet (analysis/meals/build_meals.py)
  OUTPUTS/metabolic/spike_width_check (analysis/metabolic/spike_width_check.py), imported as a module

Outputs (OUTPUTS/clinical/spike_width):
  width_associations.csv     log fasting insulin / log HOMA-IR per SD of width, clearance, upslope: unadjusted; at the
                             same average glucose; + BMI; + spike size and frequency; >= 20 spikes; at the same overnight
                             (00:00-05:00) or early-morning (04:00-08:00) CGM glucose; at the same laboratory fasting glucose
  width_variance.csv         R2 alone and gain beyond average or overnight glucose (paired bootstrap 95% CI)
  width_markers.csv          triglycerides, HDL, TG/HDL, uric acid, hsCRP, apoB, LDL (canonical panel values): per SD of
                             width and clearance at the same average glucose (+ age decade, sex), and + BMI
  width_meal_anchored.csv    meal-anchored width (raw and adjusted for meal content and context) and CGM width in the
                             same participants
  width_summary.json         repeatability, correlations, descriptives, meal-anchored counts
  metadata.json              guards, parameters
  _cache/spike_width_participants.parquet  participant-level; local only, not released
Estimates: OLS, HC3, % per SD (exp(b) - 1). Aggregate outputs only.

Usage: python analysis/clinical/spike_width.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.meals import _carbohydrate_analysis as carb# noqa: E402
from analysis.meals import _shape_analysis as shape# noqa: E402

warnings.filterwarnings('ignore')
PUB = OUTPUTS
OUT = PUB / 'clinical/spike_width'
DB = database_path()
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
S_SPIKE_WIDTH_CHECK = ROOT / 'analysis/metabolic/spike_width_check.py'
R_SPIKE_WIDTH_CHECK = PUB / 'metabolic/spike_width_check'
SUBSTRATE = PUB / 'meals/meal_food_substrate.parquet'
SEED, NB = 20261022, 2000
MEAL_MIN_RISE, MEAL_MIN_N, MEAL_WIN = 20.0, 10, 180
MARKERS = ['triglycerides', 'hdl_cholesterol', 'tg_hdl_ratio', 'uric_acid', 'hscrp', 'apo_b', 'ldl_cholesterol']
SHAPE = ['spike_width', 'all_clearance', 'upslope_half_time']
GUARDS: dict = {}


def guard(name, ok, detail=None):
    GUARDS[name] = dict(status='passed' if ok else 'failed', detail=detail)
    if not ok:
        raise AssertionError(f'guard failed: {name}: {detail}')


def z(s):
    return (s - s.mean()) / s.std(ddof=1)


def fit(d, y, x, cov, cats=()):
    """% per SD of x on log y; continuous covariates standardized; categorical covariates as fixed effects."""
    d = d.loc[d[y] > 0, [y, x] + list(cov) + list(cats)].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d['ly'] = np.log(d[y]); terms = ['zx'] + [f'z{i}' for i in range(len(cov))]
    d['zx'] = z(d[x])
    for i, c in enumerate(cov):
        d[f'z{i}'] = z(d[c])
    form = 'ly ~ ' + ' + '.join(terms + [f'C({c})' for c in cats])
    f = smf.ols(form, d).fit(cov_type='HC3'); lo, hi = f.conf_int().loc['zx']
    return dict(n=int(len(d)), pct=float(100 * (np.exp(f.params.zx) - 1)), lo=float(100 * (np.exp(lo) - 1)),
                hi=float(100 * (np.exp(hi) - 1)), p=float(f.pvalues.zx))


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    return 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


def load_spike_width_check():
    spec = importlib.util.spec_from_file_location('spike_width_check', S_SPIKE_WIDTH_CHECK)
    m = importlib.util.module_from_spec(spec); sys.modules['spike_width_check'] = m; spec.loader.exec_module(m)
    return m


def cgm_shape(m):
    """Per-participant spike shape measures, extracted as in analysis/metabolic/spike_width_check.py (its event and identity checks re-run)."""
    m_spike_facets = json.loads((m.F_SPIKE_FACETS / 'spike_facets.metadata.json').read_text())
    daily = pd.read_parquet(m.s_spike_facets.CACHE / 'spike_daily.parquet', columns=['USER_ID', 'DAY'])
    panel = pd.read_csv(m.s_spike_facets.PANEL, low_memory=False); panel = panel[panel.no_diabetes_range == True].copy()  # noqa: E712
    days = daily.merge(panel[['USER_ID', 'tz_offset_hours']], on='USER_ID'); days['DAY'] = pd.to_datetime(days.DAY)
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute("SET temp_directory='/tmp/duckdb_tmp'"); con.execute('SET threads=6')
        E, D, bad = m.extract(con, days, panel.USER_ID.to_numpy())
    guard('events_reproduce_spike_facets', bad == 0 and len(D) == m_spike_facets['guard']['person_days'] and len(E) == m_spike_facets['guard']['spikes'],
          dict(day_mismatch=int(bad), person_days=int(len(D)), spikes=int(len(E))))
    E = m.add_shape(E); res = E[E.half_recovery.notna()]
    guard('width_identity', bool(np.allclose(res.spike_width, res.upslope_half_time + res.all_clearance)))
    Pn = m.aggregate(E, D, m.MEASURES)
    saved = pd.read_parquet(FACETS, columns=['USER_ID', 'all_clearance'])
    chk = Pn[['USER_ID', 'all_clearance']].merge(saved, on='USER_ID', suffixes=('', '_spike_facets'))
    guard('clearance_reproduces_facets', len(chk) == len(saved) and float((chk.all_clearance - chk.all_clearance_spike_facets).abs().max()) < 1e-9
          and bool((chk.all_clearance.isna() == chk.all_clearance_spike_facets.isna()).all()), int(len(chk)))
    REP = m.halves(E, D, ['all_clearance', 'spike_width', 'upslope_half_time'])
    return Pn, REP


def meal_width(panel):
    meals = carb.load_macro_meals(SUBSTRATE)
    M = shape.window_meals(meals, panel)
    M = M.loc[M.PEAK_RISE_2H >= MEAL_MIN_RISE].copy()
    M['MEAL_TS'] = pd.to_datetime(M.MEAL_TS, utc=True).dt.tz_convert(None); M['mid'] = np.arange(len(M))
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute("SET temp_directory='/tmp/duckdb_tmp'"); con.execute('SET threads=8')
        con.register('m', M[['mid', 'USER_ID', 'MEAL_TS']])
        R = con.execute(f"""select m.mid, epoch(g.TS - m.MEAL_TS)/60.0 as t, g.VALUE as v from m join glucose_readings g
            on g.USER_ID = m.USER_ID and g.TS >= m.MEAL_TS and g.TS <= m.MEAL_TS + interval {MEAL_WIN} minute""").df()
        EX = con.execute("""select m.mid from m join lifestyle_events e on e.USER_ID = m.USER_ID and e.TYPE = 'exercise'
            and e.START_TS >= m.MEAL_TS and e.START_TS <= m.MEAL_TS + interval 120 minute group by 1""").df()
    R = R.merge(M[['mid', 'BASELINE_PRE', 'PEAK_RISE_2H']], on='mid')
    w2 = R.loc[R.t <= 120]; pk = w2.loc[w2.groupby('mid').v.idxmax(), ['mid', 't', 'v']].rename(columns={'t': 't_peak', 'v': 'v_peak'})
    chk = pk.merge(M[['mid', 'BASELINE_PRE', 'PEAK_RISE_2H']], on='mid')
    agree = float(np.isclose(chk.v_peak - chk.BASELINE_PRE, chk.PEAK_RISE_2H, atol=0.51).mean())
    guard('meal_peak_rise_reproduces_substrate', agree >= 0.99, agree)
    R = R.merge(pk, on='mid'); R['half'] = R.BASELINE_PRE + (R.v_peak - R.BASELINE_PRE) / 2
    up = R.loc[(R.t <= R.t_peak) & (R.v >= R.half)].groupby('mid').t.min().rename('t_up')
    down = R.loc[(R.t > R.t_peak) & (R.v <= R.half)].groupby('mid').t.min().rename('t_down')
    S = M.set_index('mid').join([up, down]); S['width'] = S.t_down - S.t_up; S['ex2h'] = S.index.isin(EX.mid).astype(float)
    cens = float(S.t_down.isna().mean()); S = S.dropna(subset=['width'])
    guard('meal_width_nonnegative', bool((S.width > 0).all()))
    S['hour_band'] = pd.cut(S.MEAL_HOUR, [-1, 5, 10, 14, 18, 21, 24], labels=['night', 'morning', 'midday', 'afternoon', 'evening', 'late']).astype(str)
    S['fasted4'] = S.IS_FASTED_4H.fillna(False).astype(float); S['log_rise'] = np.log(S.PEAK_RISE_2H)
    cov = list(carb.MACROS) + [c + '_sq' for c in carb.MACROS] + ['log_rise', 'BASELINE_PRE', 'fasted4', 'ex2h']
    for c in carb.MACROS:
        S[c + '_sq'] = S[c] ** 2
    X = sm.add_constant(pd.concat([S[cov].astype(float), pd.get_dummies(S.hour_band, drop_first=True, dtype=float)], axis=1))
    r = sm.OLS(S.width.astype(float), X).fit(); S['res'] = S.width - r.fittedvalues
    g = S.groupby('USER_ID'); n = g.size()
    person = pd.DataFrame({'meal_width': g.width.mean(), 'meal_width_adj': g.res.mean()}).loc[n >= MEAL_MIN_N].reset_index()
    info = dict(meals_rise20=int(len(M)), censored_share=cens, uncensored_meals=int(len(S)), meal_level_r2_of_covariates=float(r.rsquared),
                participants=int(len(person)), median_meal_width_min=float(S.width.median()))
    return person, info


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(PANEL); P['anchor_date'] = pd.to_datetime(P.anchor_date)
    PRI = P.loc[P.no_diabetes_range.astype(bool)].copy()
    m = load_spike_width_check(); Pn, REP = cgm_shape(m)
    CACHE = PUB / 'clinical/_cache'; CACHE.mkdir(parents=True, exist_ok=True)  # participant-level, local only (not released)
    Pn[['USER_ID'] + SHAPE].to_parquet(CACHE / 'spike_width_participants.parquet', index=False)
    F = PRI.merge(Pn[['USER_ID'] + SHAPE], on='USER_ID', how='left')
    sz = pd.read_parquet(FACETS, columns=['USER_ID', 'n_eligible_days', 'all_frequency', 'all_size'])
    F = F.merge(sz, on='USER_ID', how='left'); F['n_spikes'] = F.all_frequency * F.n_eligible_days

    # ---------------- associations
    rows = []
    for mk in ('fasting_insulin', 'homa_ir'):
        for x in SHAPE:
            spec = [('unadjusted', [], ()), ('average glucose', ['mean_glucose'], ()),
                    ('average glucose + BMI', ['mean_glucose', 'bmi'], ()),
                    ('average glucose + spike size and frequency', ['mean_glucose', 'all_size', 'all_frequency'], ()),
                    ('overnight glucose (00:00-05:00)', ['mean_night'], ()), ('early-morning glucose (04:00-08:00)', ['mean_fasting_glucose'], ()),
                    ('average + overnight glucose', ['mean_glucose', 'mean_night'], ()), ('laboratory fasting glucose', ['fasting_glucose'], ()),
                    ('average glucose (laboratory fasting glucose sample)', ['mean_glucose'], ())]
            for name, cov, cats in spec:
                d = F.dropna(subset=['fasting_glucose']) if 'laboratory fasting glucose sample' in name else F
                rows.append(dict(marker=mk, measure=x, compared_at_same=name, **fit(d, mk, x, cov, cats)))
            d = F.loc[F.n_spikes >= 20]
            rows.append(dict(marker=mk, measure=x, compared_at_same='average glucose + size and frequency, >= 20 spikes',
                             **fit(d, mk, x, ['mean_glucose', 'all_size', 'all_frequency'])))
    A = pd.DataFrame(rows)
    get = lambda mk, x, c: A.loc[(A.marker == mk) & (A.measure == x) & (A.compared_at_same == c)].iloc[0]  # noqa: E731
    sv = pd.read_csv(R_SPIKE_WIDTH_CHECK / 'spike_shape_insulin_associations.csv').query("adjustment == 'mean_glucose_only'").set_index(['marker', 'measure'])
    for mk in ('fasting_insulin', 'homa_ir'):
        for x in SHAPE:
            a, s = get(mk, x, 'average glucose'), sv.loc[(mk, x)]
            guard(f'{mk}_{x}_matches_spike_width_check', abs(a.pct - s.estimate) < 0.05 and a.n == s.n, dict(here=a.pct, saved=float(s.estimate), n=int(a.n)))
    A.to_csv(OUT / 'width_associations.csv', index=False)

    # ---------------- variance explained
    rng = np.random.default_rng([SEED, 1]); V = []
    for mk in ('fasting_insulin', 'homa_ir'):
        for base in ('mean_glucose', 'mean_night'):
            d = F.loc[F[mk] > 0].dropna(subset=[base] + SHAPE); y = np.log(d[mk].to_numpy())
            for x in SHAPE:
                B, Xa, Xx = d[[base]].to_numpy(float), d[[base, x]].to_numpy(float), d[[x]].to_numpy(float)
                bt = []
                for _ in range(NB):
                    i = rng.integers(0, len(y), len(y)); bt.append(r2(y[i], Xa[i]) - r2(y[i], B[i]))
                V.append(dict(marker=mk, base=base, measure=x, n=int(len(y)), r2_alone=r2(y, Xx), r2_base=r2(y, B), r2_with=r2(y, Xa),
                              gain=r2(y, Xa) - r2(y, B), gain_lo=float(np.percentile(bt, 2.5)), gain_hi=float(np.percentile(bt, 97.5))))
    V = pd.DataFrame(V); V.to_csv(OUT / 'width_variance.csv', index=False)

    # ---------------- other metabolic markers (canonical panel values)
    K = []
    for mk in MARKERS:
        for x in ('spike_width', 'all_clearance', 'all_size'):
            for name, cov in (('average glucose, age, sex', ['mean_glucose']), ('average glucose, BMI, age, sex', ['mean_glucose', 'bmi'])):
                d = F.dropna(subset=['sex', 'age_bucket'])
                if d[mk].notna().sum() < 150:
                    continue
                K.append(dict(marker=mk, measure=x, adjustment=name, **fit(d, mk, x, cov, cats=('sex', 'age_bucket'))))
    K = pd.DataFrame(K); K.to_csv(OUT / 'width_markers.csv', index=False)

    # ---------------- meal-anchored width
    person, minfo = meal_width(shape.load_panel(PANEL))
    W = F.merge(person, on='USER_ID', how='inner')
    Wc = W.dropna(subset=['spike_width', 'meal_width'])
    MR = []
    for mk in ('fasting_insulin', 'homa_ir'):
        for x in ('spike_width', 'meal_width', 'meal_width_adj'):
            MR.append(dict(marker=mk, measure=x, sample='both widths available', **fit(Wc, mk, x, ['mean_glucose'])))
            MR.append(dict(marker=mk, measure=x, sample='both widths available, overnight glucose', **fit(Wc, mk, x, ['mean_night'])))
        d = Wc.loc[Wc[mk] > 0].copy(); d['ly'] = np.log(d[mk])
        for c in ('spike_width', 'meal_width', 'mean_glucose'):
            d['z_' + c] = z(d[c])
        f = smf.ols('ly ~ z_spike_width + z_meal_width + z_mean_glucose', d).fit(cov_type='HC3')
        for x in ('spike_width', 'meal_width'):
            lo, hi = f.conf_int().loc['z_' + x]
            MR.append(dict(marker=mk, measure=x, sample='both widths in one model, average glucose', n=int(len(d)),
                           pct=float(100 * (np.exp(f.params['z_' + x]) - 1)), lo=float(100 * (np.exp(lo) - 1)), hi=float(100 * (np.exp(hi) - 1)),
                           p=float(f.pvalues['z_' + x])))
    MR = pd.DataFrame(MR); MR.to_csv(OUT / 'width_meal_anchored.csv', index=False)

    # ---------------- summary
    rep = REP.set_index('measure')
    r_spike_facets = pd.read_csv(m.F_SPIKE_FACETS / 'spike_facets_repeatability.csv').set_index('facet').loc['clearance']
    guard('clearance_split_half_reproduces_spike_facets', abs(rep.loc['all_clearance', 'pearson_r'] - r_spike_facets.pearson_r) < 1e-12, float(rep.loc['all_clearance', 'pearson_r']))
    Q = F.dropna(subset=['spike_width', 'all_clearance', 'all_size', 'mean_glucose'])
    summ = dict(split_half_r={k: float(rep.loc[k, 'pearson_r']) for k in SHAPE},
                spearman_brown={k: float(rep.loc[k, 'spearman_brown']) for k in SHAPE if 'spearman_brown' in rep.columns},
                n_participants_with_width=int(F.spike_width.notna().sum()),
                corr={'width_clearance': float(Q[['spike_width', 'all_clearance']].corr().iloc[0, 1]),
                      'width_mean_glucose': float(Q[['spike_width', 'mean_glucose']].corr().iloc[0, 1]),
                      'width_size': float(Q[['spike_width', 'all_size']].corr().iloc[0, 1]),
                      'meal_width_cgm_width': float(Wc[['meal_width', 'spike_width']].corr().iloc[0, 1])},
                descriptives={k: dict(median=float(F[k].median()), q1=float(F[k].quantile(.25)), q3=float(F[k].quantile(.75)), sd=float(F[k].std()))
                              for k in SHAPE},
                meal_anchored=minfo, n_common_meal_cgm=int(len(Wc)))
    (OUT / 'width_summary.json').write_text(json.dumps(summ, indent=1) + '\n')

    meta = dict(analysis='spike_width', guards=GUARDS, seed=SEED, bootstrap=NB,
                runtime_seconds=round(time.time() - t0, 1))
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: v['status'] for k, v in GUARDS.items()}), meta['runtime_seconds'])


if __name__ == '__main__':
    main()
