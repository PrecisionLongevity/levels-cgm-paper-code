#!/usr/bin/env python3
"""Computes clinical-frame summary statistics relating CGM-derived average glucose to HbA1c and fasting insulin.

Population: the primary cohort from the canonical lab/CGM panel, restricted to participants never observed in
the diabetes range (no_diabetes_range). Lab rule: canonical-labs insulin exclude_period rule.

Analyses:
  - Fasting insulin (log) regressed on CGM average glucose vs. on HbA1c, including within the HbA1c-normal
    (<5.7%) subgroup; nested R2 decomposition and a bootstrap prevalence ratio for elevated fasting insulin
    at average glucose >= 110 mg/dL among HbA1c-normal participants.
  - Sensitivity of the R2/AUC comparison between average glucose and HbA1c by anchor year, and restricted to
    average glucose < 110 mg/dL.
  - Test-retest reliability (14-90 day same-user repeats) of HbA1c, fasting insulin, HOMA-IR and fasting glucose
    among non-internal users never in the diabetes range.
  - R2 between fasting insulin and local-hour average glucose, and overnight vs. 24-h R2 comparison.
  - Sensitivity of the clearance-time/fasting-insulin association to the spike definition (average glucose only;
    + spike size and frequency; restricted to participants with >= 20 spikes; alternative recovery-based definition).
  - HbA1c at matched CGM average glucose, and at matched lab fasting glucose, by age decade (vs. the 30s).

Inputs:
  - LEVELS_DB tables: labs, users, glucose_readings.
  - OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py).
  - OUTPUTS/meals/spike_facets/spike_facets_participants.parquet (analysis/meals/spike_facets.py).

Outputs (aggregate only), under OUTPUTS/clinical:
  hba1c_resolution.json        spread of average glucose at each normal HbA1c value, nested R2, prevalence ratio, deciles
  avg_glucose_spread_by_a1c.csv
  insulin_by_decile.csv
  hba1c_sensitivity.json       by anchor year (R2 and AUC in the same participants), restriction on average glucose
  lab_retest.json              test-retest reliability of HbA1c, fasting insulin, HOMA-IR, lab fasting glucose
  time_of_day.json             overnight vs 24-h R2 (paired bootstrap); R2 by local hour
  r2_by_hour.csv
  clearance_sensitivity.json   primary spike definition: avg only; + size and frequency; >= 20 spikes; alternative definition
  age_offset.json              HbA1c at the same CGM average glucose and at the same lab fasting glucose, by age decade
  metadata.json                guards and run parameters

Usage: python analysis/clinical/clinical_summaries.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from cgmpaper import lab_normalization as LN  # noqa: E402
from cgmpaper.lab_reference import same_day_indices  # noqa: E402

warnings.filterwarnings('ignore')
PUB = OUTPUTS
OUT = PUB / 'clinical'
DB = database_path()
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
SEED, NB = 20261021, 2000
FI_CUT, AVG_CUT, A1C_NORMAL = 10.0, 110.0, 5.7
GUARDS: dict = {}


def guard(name, ok, detail=None):
    GUARDS[name] = dict(status='passed' if ok else 'failed', detail=detail)
    if not ok:
        raise AssertionError(f'guard failed: {name}: {detail}')


def z(s):
    return (s - s.mean()) / s.std(ddof=1)


def r2(y, x):
    X = np.column_stack([np.ones(len(y)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2_multi(y, xs):
    X = np.column_stack([np.ones(len(y))] + list(xs)); b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def auc(y, s):
    y = np.asarray(y, int); r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pct_ci(f, k):
    ci = f.conf_int().loc[k]
    return [float(100 * np.expm1(f.params[k])), float(100 * np.expm1(ci[0])), float(100 * np.expm1(ci[1]))]


def est_ci(f, k):
    ci = f.conf_int().loc[k]
    return [float(f.params[k]), float(ci[0]), float(ci[1])]


def boot(rng, n, fn, B=NB):
    idx = rng.integers(0, n, (B, n)); vals = np.array([fn(i) for i in idx])
    return np.percentile(vals, [2.5, 97.5], axis=0)


def retest(df, col, logv):
    a = df.dropna(subset=[col])[['USER_ID', 'DRAW_DATE', col]].sort_values(['USER_ID', 'DRAW_DATE']).copy()
    a['v'] = np.log(a[col]) if logv else a[col]
    a['nxt'] = a.groupby('USER_ID').v.shift(-1); a['gap'] = (a.groupby('USER_ID').DRAW_DATE.shift(-1) - a.DRAW_DATE).dt.days
    p = a.loc[a.gap.between(14, 90)].drop_duplicates('USER_ID'); x, y = p.v.to_numpy(), p.nxt.to_numpy()
    msb = np.var((x + y) / 2, ddof=1) * 2; msw = np.sum((x - y) ** 2) / (2 * len(x))
    out = dict(n_pairs=int(len(p)), median_gap_days=float(p.gap.median()), r=float(np.corrcoef(x, y)[0, 1]), icc1=float((msb - msw) / (msb + msw)))
    if logv:
        out['within_person_cv_pct'] = float(100 * np.sqrt(np.expm1(msw)))
    return out


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    guard('lab_rule', LN.LAB_DEFINITION_VERSION == 'canonical-labs' and LN.INSULIN_RULE == 'exclude_period', LN.LAB_DEFINITION_VERSION)
    P = pd.read_csv(PANEL); P['anchor_date'] = pd.to_datetime(P.anchor_date)
    PRI = P.loc[P.no_diabetes_range.astype(bool)].copy()
    S = PRI.loc[PRI.fasting_insulin.notna() & PRI.mean_glucose.notna() & PRI.a1c.notna()].copy()
    S['ly'] = np.log(S.fasting_insulin); S['z_avg'] = z(S.mean_glucose)

    rng = np.random.default_rng([SEED, 1])  # independent stream per section: changes in one section do not move CIs elsewhere
    # ---------------- HbA1c resolution (Fig. 5c)
    N = S.loc[S.a1c < A1C_NORMAL].copy()
    sp = PRI.loc[PRI.a1c < A1C_NORMAL].groupby(PRI.a1c.round(1)).mean_glucose.describe(percentiles=[.05, .25, .5, .75, .95])
    sp = sp.loc[sp['count'] >= 100]; sp.index.name = 'a1c'; sp.to_csv(OUT / 'avg_glucose_spread_by_a1c.csv')

    def nested(idx, d):
        yy, gg, aa = d.ly.to_numpy()[idx], d.mean_glucose.to_numpy(float)[idx], d.a1c.to_numpy(float)[idx]
        ra, rg, rb = r2(yy, aa), r2(yy, gg), r2_multi(yy, [aa, gg])
        return [ra, rg, rb - ra, rb - rg]
    res = dict(n_fi=len(S), n_normal=len(N))
    for nm, d in (('all', S), ('normal', N)):
        pt = nested(np.arange(len(d)), d); lo, hi = boot(rng, len(d), lambda i: nested(i, d))
        res[f'nested_{nm}'] = {k: [float(pt[j]), float(lo[j]), float(hi[j])] for j, k in enumerate(['r2_a1c', 'r2_avg', 'avg_beyond_a1c', 'a1c_beyond_avg'])}
    flag = (N.mean_glucose >= AVG_CUT).to_numpy(); e = (N.fasting_insulin >= FI_CUT).to_numpy()
    rr = lambda i: e[i][flag[i]].mean() / e[i][~flag[i]].mean()  # noqa: E731
    lo, hi = boot(rng, len(N), rr)
    res['prevalence_ratio'] = dict(n=int(len(N)), n_flagged=int(flag.sum()), share_flagged=float(flag.mean()), prev_flagged=float(e[flag].mean()),
                                   prev_not=float(e[~flag].mean()), rr=float(rr(np.arange(len(N)))), ci=[float(lo), float(hi)])
    res['share_normal_primary_ge110'] = float((PRI.loc[PRI.a1c < A1C_NORMAL].mean_glucose >= AVG_CUT).mean())
    rows = []
    for sub, d in (('all', S), ('a1c<5.7', N)):
        for m in ('a1c', 'mean_glucose'):
            # HbA1c is reported to 0.1%: group by reported value (no random tie-breaking); CGM in bins of the same sizes
            # HbA1c values with <100 participants are merged into the nearest end group (e.g. <=4.9%), so every participant is
            # in a group; CGM groups have the same sizes, filled from the lowest average glucose upward.
            vc = d.a1c.round(1).value_counts().sort_index(); ok = vc[vc >= 100].index
            a1g = d.a1c.round(1).clip(lower=ok.min(), upper=ok.max()); sizes = a1g.value_counts().sort_index()
            if m == 'a1c':
                grp = a1g
            else:
                order = d.mean_glucose.rank(method='first').astype(int) - 1; edges = np.cumsum(sizes.to_numpy())
                grp = pd.Series(np.searchsorted(edges, order.to_numpy(), side='right'), index=d.index)
            guard(f'groups_cover_{sub}_{m}', grp.notna().sum() == len(d), int(grp.notna().sum()))
            for k, gdf in d.groupby(grp):
                gm = np.exp(gdf.ly.mean()); lo, hi = boot(rng, len(gdf), lambda i: np.exp(gdf.ly.to_numpy()[i].mean()), 500)
                rows.append(dict(subset=sub, measure=m, group=float(k), n=len(gdf), value_min=float(gdf[m].min()), value_max=float(gdf[m].max()),
                                 value_median=float(gdf[m].median()), gm_insulin=float(gm), ci_low=float(lo), ci_high=float(hi)))
    pd.DataFrame(rows).to_csv(OUT / 'insulin_by_decile.csv', index=False)
    disc = PRI.dropna(subset=['fasting_insulin', 'a1c', 'mean_glucose']).copy()
    disc['grp'] = np.select([(disc.mean_glucose >= AVG_CUT) & (disc.a1c < A1C_NORMAL), (disc.mean_glucose < AVG_CUT) & (disc.a1c >= A1C_NORMAL),
                             (disc.mean_glucose < AVG_CUT) & (disc.a1c < A1C_NORMAL)], ['high_avg_normal_a1c', 'normal_avg_high_a1c', 'both_normal'], 'both_high')
    res['discordant_gm_insulin'] = {}
    for k, v in disc.groupby('grp'):
        lv = np.log(v.fasting_insulin.to_numpy()); lo, hi = boot(rng, len(lv), lambda i: np.exp(lv[i].mean()), 1000)
        res['discordant_gm_insulin'][k] = dict(n=int(len(v)), gm=float(np.exp(lv.mean())), ci=[float(lo), float(hi)])
    (OUT / 'hba1c_resolution.json').write_text(json.dumps(res, indent=1) + '\n')

    rng = np.random.default_rng([SEED, 2])  # independent stream per section: changes in one section do not move CIs elsewhere
    # ---------------- HbA1c sensitivity: by anchor year (R2 and AUC in the same participants), restriction on average glucose
    sens = {}
    def r2diff(d):
        yy, gg, aa = d.ly.to_numpy(), d.mean_glucose.to_numpy(float), d.a1c.to_numpy(float)
        lo, hi = boot(rng, len(d), lambda i: r2(yy[i], gg[i]) - r2(yy[i], aa[i]))
        return dict(n=int(len(d)), r2_avg=float(r2(yy, gg)), r2_a1c=float(r2(yy, aa)), diff=float(r2(yy, gg) - r2(yy, aa)), ci=[float(lo), float(hi)])
    sens['by_year'] = {}
    for yr, d in S.groupby('anchor_year'):
        if len(d) < 400:
            sens.setdefault('years_not_shown', {})[str(int(yr))] = int(len(d)); continue
        out = r2diff(d); e = (d.fasting_insulin >= FI_CUT).to_numpy(int); gg, aa = d.mean_glucose.to_numpy(float), d.a1c.to_numpy(float)
        lo, hi = boot(rng, len(d), lambda i: auc(e[i], gg[i]) - auc(e[i], aa[i]))
        out.update(auc_avg=float(auc(e, gg)), auc_a1c=float(auc(e, aa)), auc_diff_ci=[float(lo), float(hi)], prevalence=float(e.mean()))
        sens['by_year'][str(int(yr))] = out
    guard('years_account_for_cohort', sum(v['n'] for v in sens['by_year'].values()) + sum(sens.get('years_not_shown', {}).values()) == len(S))
    sens['avg_lt_110'] = r2diff(S.loc[S.mean_glucose < AVG_CUT])
    f = smf.ols('ly ~ z_avg + C(anchor_year)', S).fit(cov_type='HC3'); sens['fi_pct_avg_adj_year'] = pct_ci(f, 'z_avg')
    sens['sd_avg_glucose'] = float(S.mean_glucose.std(ddof=1)); sens['median_raw_interval_minutes'] = S.median_raw_interval_minutes.round(1).value_counts().to_dict()
    (OUT / 'hba1c_sensitivity.json').write_text(json.dumps(sens, indent=1, default=float) + '\n')

    rng = np.random.default_rng([SEED, 3])  # independent stream per section: changes in one section do not move CIs elsewhere
    # ---------------- lab test-retest (all non-internal users never in the diabetes range)
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute("SET threads=8; SET memory_limit='10GB'; SET temp_directory='/tmp/duckdb_tmp'")
        raw = con.execute("""SELECT l.USER_ID, l.TYPE, l.DRAW_DATE, l.VALUE, l.UNIT FROM labs l JOIN users u ON u.USER_ID = l.USER_ID
                             WHERE u.IS_INTERNAL_USER = FALSE AND l.TYPE IN ('fasting_insulin', 'a1c', 'fasting_glucose', 'triglycerides', 'hdl_cholesterol')""").df()
        daily, flags, _ = LN.normalize_labs(raw); daily['DRAW_DATE'] = pd.to_datetime(daily.DRAW_DATE)
        dia = set(flags.loc[flags.diabetes_range_ever, 'USER_ID'])
        W = same_day_indices(daily[['USER_ID', 'DRAW_DATE', 'TYPE', 'value']]); W['DRAW_DATE'] = pd.to_datetime(W.DRAW_DATE); W = W.loc[~W.USER_ID.isin(dia)]
        rt = {'a1c': retest(W, 'a1c', False), 'fasting_insulin_log': retest(W, 'fasting_insulin', True),
              'homa_ir_log': retest(W, 'homa_ir', True), 'fasting_glucose': retest(W, 'fasting_glucose', False)}
        (OUT / 'lab_retest.json').write_text(json.dumps(rt, indent=1) + '\n')

        rng = np.random.default_rng([SEED, 4])  # independent stream per section: changes in one section do not move CIs elsewhere
        # ---------------- time of day: R2 by local hour (all accepted readings within +/-90 days of the anchor)
        con.register('a', S[['USER_ID', 'anchor_date', 'tz_offset_hours']].rename(columns={'tz_offset_hours': 'tz'}))
        H = con.execute("""
            SELECT a.USER_ID, hour(g.TS + a.tz * INTERVAL '1 hour') hr, avg(g.VALUE) v, count(*) k
            FROM a JOIN glucose_readings g ON g.USER_ID = a.USER_ID
             AND g.TS >= CAST(a.anchor_date AS TIMESTAMP) - INTERVAL '90 days' - a.tz * INTERVAL '1 hour'
             AND g.TS <  CAST(a.anchor_date AS TIMESTAMP) + INTERVAL '91 days' - a.tz * INTERVAL '1 hour'
            WHERE g.VALUE BETWEEN 20 AND 500 GROUP BY 1, 2""").df()
    Hw = H.pivot(index='USER_ID', columns='hr', values='v'); D = S.set_index('USER_ID').join(Hw)
    rows = []
    for hr in range(24):
        d = D.dropna(subset=[hr]); yy, xx = d.ly.to_numpy(), d[hr].to_numpy(float)
        lo, hi = boot(rng, len(d), lambda i: r2(yy[i], xx[i]), 500)
        rows.append(dict(local_hour=hr, n=len(d), r2=float(r2(yy, xx)), ci_low=float(lo), ci_high=float(hi), median_glucose=float(np.median(xx))))
    HR = pd.DataFrame(rows); HR.to_csv(OUT / 'r2_by_hour.csv', index=False)
    d = S.dropna(subset=['mean_night']); yy, nn, gg = d.ly.to_numpy(), d.mean_night.to_numpy(float), d.mean_glucose.to_numpy(float)
    lo, hi = boot(rng, len(d), lambda i: r2(yy[i], nn[i]) - r2(yy[i], gg[i]))
    tod = dict(n=int(len(d)), r2_night=float(r2(yy, nn)), r2_avg=float(r2(yy, gg)), diff=float(r2(yy, nn) - r2(yy, gg)), ci=[float(lo), float(hi)],
               peak_hours=[int(h) for h in HR.sort_values('r2').tail(3).local_hour.sort_values()], peak_r2=float(HR.r2.max()),
               min_hour=int(HR.loc[HR.r2.idxmin(), 'local_hour']), min_r2=float(HR.r2.min()))
    (OUT / 'time_of_day.json').write_text(json.dumps(tod, indent=1) + '\n')

    rng = np.random.default_rng([SEED, 5])  # independent stream per section: changes in one section do not move CIs elsewhere
    # ---------------- clearance time (primary spike definition, from analysis/meals/spike_facets.py facets)
    F = pd.read_parquet(FACETS, columns=['USER_ID', 'n_eligible_days', 'all_frequency', 'all_size', 'all_clearance'])
    C1 = S.merge(F, on='USER_ID').dropna(subset=['all_frequency', 'all_size', 'all_clearance']).copy()
    for c in ('all_frequency', 'all_size', 'all_clearance'):
        C1['z_' + c] = z(C1[c])
    C1['n_spikes'] = C1.all_frequency * C1.n_eligible_days
    cl = {}
    f = smf.ols('ly ~ z_all_clearance + z_avg', C1).fit(cov_type='HC3'); cl['primary_avg_only'] = dict(n=int(len(C1)), pct=pct_ci(f, 'z_all_clearance'))
    f = smf.ols('ly ~ z_all_clearance + z_avg + z_all_frequency + z_all_size', C1).fit(cov_type='HC3'); cl['primary_avg_size_freq'] = dict(n=int(len(C1)), pct=pct_ci(f, 'z_all_clearance'))
    C2 = C1.loc[C1.n_spikes >= 20]
    f = smf.ols('ly ~ z_all_clearance + z_avg + z_all_frequency + z_all_size', C2).fit(cov_type='HC3'); cl['primary_ge20_spikes'] = dict(n=int(len(C2)), pct=pct_ci(f, 'z_all_clearance'))
    cl['sd_clearance_minutes'] = float(C1.all_clearance.std(ddof=1)); cl['rho_clearance_size'] = float(C1[['all_clearance', 'all_size']].corr('spearman').iloc[0, 1])
    A2 = S.dropna(subset=['recovery_mean', 'peaks_per_day', 'peak_prom_mean']).copy()
    for c in ('recovery_mean', 'peaks_per_day', 'peak_prom_mean'):
        A2['z_' + c] = z(A2[c])
    f = smf.ols('ly ~ z_recovery_mean + z_avg', A2).fit(cov_type='HC3'); cl['alt_avg_only'] = dict(n=int(len(A2)), pct=pct_ci(f, 'z_recovery_mean'))
    f = smf.ols('ly ~ z_recovery_mean + z_avg + z_peaks_per_day + z_peak_prom_mean', A2).fit(cov_type='HC3'); cl['alt_avg_size_freq'] = dict(n=int(len(A2)), pct=pct_ci(f, 'z_recovery_mean'))
    (OUT / 'clearance_sensitivity.json').write_text(json.dumps(cl, indent=1) + '\n')

    rng = np.random.default_rng([SEED, 6])  # independent stream per section: changes in one section do not move CIs elsewhere
    # ---------------- age offset in HbA1c (ages 20s-70s), vs 30s
    G = PRI.loc[PRI.mean_glucose.notna() & PRI.a1c.notna() & PRI.age_bucket.between(20, 70)].copy(); G['age'] = G.age_bucket.astype(int).astype(str)
    f = smf.ols('a1c ~ mean_glucose + C(age, Treatment("30")) + C(sex) + C(anchor_year)', G).fit(cov_type='HC3')
    ag = dict(n=int(len(G)), age_counts=G.age.value_counts().sort_index().to_dict(),
              vs30_at_cgm={x: est_ci(f, f'C(age, Treatment("30"))[T.{x}]') for x in ('20', '40', '50', '60', '70')})
    L = PRI.loc[PRI.fasting_glucose.notna() & PRI.a1c.notna() & PRI.age_bucket.between(20, 70)].copy(); L['age'] = L.age_bucket.astype(int).astype(str)
    f = smf.ols('a1c ~ fasting_glucose + C(age, Treatment("30")) + C(sex) + C(anchor_year)', L).fit(cov_type='HC3')
    ag.update(n_labfg=int(len(L)), vs30_at_lab_fg={x: est_ci(f, f'C(age, Treatment("30"))[T.{x}]') for x in ('20', '40', '50', '60', '70')})
    G['grp'] = np.select([(G.mean_glucose >= AVG_CUT) & (G.a1c < A1C_NORMAL), (G.mean_glucose < AVG_CUT) & (G.a1c >= A1C_NORMAL)], ['high_avg_normal_a1c', 'normal_avg_high_a1c'], 'other')
    mix = pd.crosstab(G.grp, G.age, normalize='index') * 100
    ag['share_under50'] = {k: float(mix.loc[k, ['20', '30', '40']].sum()) for k in ('high_avg_normal_a1c', 'normal_avg_high_a1c')}
    ag['share_60plus'] = {k: float(mix.loc[k, ['60', '70']].sum()) for k in ('high_avg_normal_a1c', 'normal_avg_high_a1c')}
    cnt = G.age.value_counts(); ag['sample_under50'] = float(100 * cnt[['20', '30', '40']].sum() / len(G)); ag['sample_60plus'] = float(100 * cnt[['60', '70']].sum() / len(G))
    (OUT / 'age_offset.json').write_text(json.dumps(ag, indent=1) + '\n')

    meta = dict(analysis='clinical', version='clinical', status='passed', guards=GUARDS, lab_rule=LN.LAB_DEFINITION_VERSION,
                seed=SEED, bootstrap=NB, runtime_seconds=round(time.time() - t0, 1))
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: v['status'] for k, v in GUARDS.items()}), meta['runtime_seconds'])


if __name__ == '__main__':
    main()
