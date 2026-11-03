#!/usr/bin/env python3
"""Rise-based spike facets, meal anchoring and biomarker direction.

Spike definition: a local maximum at least 30 mg/dL above the minimum of the preceding 90
observed minutes, detected by analysis/meals/_spike_analysis.daily_metrics on observed
five-minute local-day cells with no interpolation. Peaks <30 min apart keep the larger rise.

Participant facets
  frequency  mean spikes per eligible day (>=192 searchable centers), >=5 eligible days
  size       mean rise (mg/dL) of spikes on eligible days
  clearance  mean half-recovery minutes over resolved spikes on eligible days, >=5 resolved
Each facet is also computed separately for meal-linked spikes (a food log between `pre`
minutes before and `post` minutes after the preceding-minimum onset) and unlinked spikes.
Windows: primary (15, 60); narrow (10, 45); wide (30, 90).

Samples: canonical primary panel (no observed diabetes-range laboratory records). Logging
groups use analysis/meals/spike_exposure.py per-date food-log counts: high loggers (>=3 logs
on >=80% of dense dates, >=14 dates); loose (>=2 on >=60%); strict (>=3 on >=90%).

Direction models: logistic HbA1c >=5.7% (decimal half-up rounding) on each standardized facet
with mean, median, 10th/25th/75th/90th percentile glucose, time >140 mg/dL, age bucket,
recorded sex and BMI; Wald 95% CI. Peer-matched contrasts: above vs below the median of each
facet within twentieths of mean glucose, 1,000 participant resamples. Joint models enter
meal-linked and unlinked facets together and Wald-test equality.

Inputs: OUTPUTS/meals/_cache/spike_daily.parquet, spike_participants.parquet and
spike_exposure_daily.parquet (analysis/meals/spike_exposure.py);
OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py).

Outputs (OUTPUTS/meals/spike_facets/, participant-level files stay private):
  spike_facets_participants.parquet   per-participant facets (private)
  spike_facets_associations.csv       direction-model results
  spike_facets_repeatability.csv      split-half reliability of facets
  spike_facets_descriptives.csv       sample descriptive statistics
  spike_facets.metadata.json          run metadata

Usage: python analysis/meals/spike_facets.py [--db PATH]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy import signal
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.meals import _spike_analysis as core# noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('spike_exposure_module', ROOT / 'analysis/meals/spike_exposure.py')
s_spike_exposure = importlib.util.module_from_spec(_spec); sys.modules['spike_exposure_module'] = s_spike_exposure; _spec.loader.exec_module(s_spike_exposure)

PUBLICATION = OUTPUTS
OUT = PUBLICATION / 'meals/spike_facets'
CACHE = PUBLICATION / 'meals/_cache'
PANEL = PUBLICATION / 'canonical_panel/canonical_lab_cgm_panel.csv'
DEFINITION_VERSION = 'rise30-facets-meal-anchored'
WINDOWS = {'primary': (15, 60), 'narrow': (10, 45), 'wide': (30, 90)}
LEVEL = ['mean_glucose', 'median_glucose', 'p10_glucose', 'p25_glucose', 'p75_glucose', 'p90_glucose', 'tar_140']
SEED, BOOT, BINS, MIN_DAYS, MIN_RES = 20260924, 1000, 20, 5, 5
FACETS = ('frequency', 'size', 'clearance')


def day_events(bins, g):
    """Per-event (onset_cell, rise, half_recovery_min) mirroring core.daily_metrics exactly."""
    runs = np.split(np.arange(len(bins)), np.flatnonzero(np.diff(bins) != 1) + 1)
    out, n_search = [], 0
    lb, dist, rec = core.LOOKBACK_CELLS, core.PEAK_DISTANCE_CELLS, core.RECOVERY_CELLS
    for run in runs:
        x = g[run]; n_search += max(0, len(x) - lb - 1)
        peaks, _ = signal.find_peaks(x)
        cand = []
        for p in peaks:
            if p >= lb:
                w = x[p - lb:p]; a = float(x[p] - w.min())
                if a >= core.RISE_MG_DL:
                    cand.append((int(p), a, int(p - lb + np.argmin(w))))
        kept = []
        for p, a, o in sorted(cand, key=lambda it: (-it[1], it[0])):
            if all(abs(p - q) >= dist for q, _, _ in kept):
                kept.append((p, a, o))
        kept.sort()
        for i, (p, a, o) in enumerate(kept):
            nxt = kept[i + 1][0] if i + 1 < len(kept) else len(x)
            stop = min(p + rec + 1, nxt, len(x))
            r = np.flatnonzero(x[p + 1:stop] <= x[p] - a / 2)
            out.append((int(bins[run[o]]), a, float((r[0] + 1) * 5) if len(r) else np.nan))
    return out, n_search >= core.MIN_SEARCH_CELLS


def extract(con, days, people):
    ev, dd, meals, mismatch = [], [], [], 0
    for i in range(0, len(people), 300):
        con.register('batch_days', days[days.USER_ID.isin(people[i:i + 300])])
        raw = con.execute(s_spike_exposure.raw_day_query()).df()
        logs = con.execute('''SELECT f.USER_ID, f.TIMESTAMP + p.tz_offset_hours*INTERVAL '1 hour' AS LOCAL_TS
            FROM food_logs f JOIN (SELECT DISTINCT USER_ID, tz_offset_hours FROM batch_days) p USING (USER_ID)
            WHERE f.TYPE IN ('food','food_with_macros') AND f.TIMESTAMP IS NOT NULL''').df()
        con.unregister('batch_days')
        minutes = (logs.LOCAL_TS.astype('datetime64[ns]').astype('int64') // 60_000_000_000).astype(np.int64)
        by_user = {u: np.sort(v) for u, v in minutes.groupby(logs.USER_ID)}
        for r in raw.itertuples(index=False):
            b = np.asarray(r.bins, np.int64); x = np.asarray(r.glucose, float)
            evs, eligible = day_events(b, x)
            ref = core.daily_metrics(b, x)
            resolved = [e[2] for e in evs if np.isfinite(e[2])]
            ok = eligible == bool(ref['peak_day_eligible'])
            if eligible:  # spike_exposure.py counts spikes and recoveries only on eligible days
                ok = (ok and len(evs) == ref['n_spikes'] == ref['daily_n_spikes'] and len(resolved) == ref['n_resolved']
                      and np.isclose(sum(resolved), ref['recovery_sum_minutes'])
                      and np.isclose(max([e[1] for e in evs], default=0.), ref['max_spike_rel']))
            mismatch += not ok
            day0 = int(pd.Timestamp(r.DAY).value // 60_000_000_000)
            L = by_user.get(r.USER_ID, np.array([], np.int64))
            near = L[(L >= day0 - 120) & (L < day0 + 1560)]
            dd.append((r.USER_ID, r.DAY, eligible))
            for onset, rise, half in evs:
                t = day0 + 5 * onset
                ev.append((r.USER_ID, r.DAY, eligible, rise, half,
                           *[bool(((near >= t - pre) & (near <= t + post)).any()) for pre, post in WINDOWS.values()]))
            today = near[(near >= day0) & (near < day0 + 1440)]
            if len(today):
                cells = dict(zip(b.tolist(), x.tolist()))
                for s in today[np.r_[True, np.diff(today) > 30]]:
                    c0 = int((s - day0) // 5)
                    obs = [cells[c] for c in range(c0 + 1, c0 + 25) if c in cells]
                    if len(obs) >= 18:
                        meals.append((r.USER_ID, max(obs)))
        print(f'  {min(i + 300, len(people)):,}/{len(people):,} participants', flush=True)
    E = pd.DataFrame(ev, columns=['USER_ID', 'DAY', 'eligible', 'rise', 'half_recovery', *[f'meal_{w}' for w in WINDOWS]])
    D = pd.DataFrame(dd, columns=['USER_ID', 'DAY', 'eligible'])
    M = pd.DataFrame(meals, columns=['USER_ID', 'peak_2h'])
    return E, D, M, mismatch


def participant_facets(E, D):
    n_elig = D[D.eligible].groupby('USER_ID').size()
    out = pd.DataFrame(index=pd.Index(D.USER_ID.unique(), name='USER_ID'))
    out['n_eligible_days'] = n_elig.reindex(out.index).fillna(0).astype(int)
    groups = {'all': np.ones(len(E), bool)}
    for w in WINDOWS:
        groups[f'meal_{w}'] = E[f'meal_{w}'].to_numpy(); groups[f'unlinked_{w}'] = ~E[f'meal_{w}'].to_numpy()
    for name, mask in groups.items():
        e = E[mask]; el = e[e.eligible]
        out[f'{name}_frequency'] = el.groupby('USER_ID').size().reindex(out.index).fillna(0) / out.n_eligible_days.replace(0, np.nan)
        out[f'{name}_size'] = el.groupby('USER_ID').rise.mean().reindex(out.index)
        res = el[el.half_recovery.notna()].groupby('USER_ID').half_recovery
        out[f'{name}_clearance'] = res.mean().reindex(out.index).where(res.size().reindex(out.index).fillna(0) >= MIN_RES)
        out.loc[out.n_eligible_days < MIN_DAYS, [f'{name}_{f}' for f in FACETS]] = np.nan
    out['meal_share_primary'] = out.meal_primary_frequency / out.all_frequency.replace(0, np.nan)
    return out.reset_index()


def design(frame, cols):
    X = frame[LEVEL + ['bmi']].astype(float).copy()
    X['male'] = (frame.sex == 'male').astype(float)
    X = X.join(pd.get_dummies(frame.age_bucket, prefix='age', drop_first=True, dtype=float))
    X = X.loc[:, (X.std() > 0) & (X.abs().sum() >= 10)]
    X = pd.concat([X, frame[cols].astype(float)], axis=1)
    X = X.fillna(X.median()); X = (X - X.mean()) / X.std()
    return sm.add_constant(X)


def adjusted(frame, cols, outcome='hba1c_ge_5_7'):
    s = frame[frame[outcome].notna() & frame[cols].notna().all(axis=1)].reset_index(drop=True)
    fit = sm.Logit(s[outcome].astype(int).to_numpy(), design(s, cols)).fit(disp=0, maxiter=300)
    ci = fit.conf_int()
    rows = [{'term': c, 'n': len(s), 'estimate': float(np.exp(fit.params[c])), 'ci_low': float(np.exp(ci.loc[c, 0])),
             'ci_high': float(np.exp(ci.loc[c, 1])), 'p': float(fit.pvalues[c]), 'converged': bool(fit.mle_retvals['converged'])} for c in cols]
    if len(cols) == 2:
        r = np.zeros(len(fit.params)); names = list(fit.params.index)
        r[names.index(cols[0])], r[names.index(cols[1])] = 1, -1
        for row in rows: row['equality_p'] = float(fit.t_test(r).pvalue)
    return rows


def peer(frame, col, outcome, rng):
    s = frame[frame[col].notna() & frame[outcome].notna()].reset_index(drop=True)
    bins = pd.qcut(s.mean_glucose, BINS, labels=False).to_numpy()
    x, y = s[col].to_numpy(float), s[outcome].to_numpy(float)
    def diff(idx):
        xb, yb, gb = x[idx], y[idx], bins[idx]
        med = pd.Series(xb).groupby(gb).transform('median').to_numpy(); hi = xb > med
        return 100 * (yb[hi].mean() - yb[~hi].mean())
    est = diff(np.arange(len(s)))
    boots = [diff(rng.integers(0, len(s), len(s))) for _ in range(BOOT)]
    return {'term': col, 'n': len(s), 'estimate': float(est), 'ci_low': float(np.percentile(boots, 2.5)),
            'ci_high': float(np.percentile(boots, 97.5)), 'p': np.nan}


def repeatability(E, D):
    D = D.sort_values(['USER_ID', 'DAY']).copy()
    D['half'] = np.where(D.groupby('USER_ID').cumcount() < D.groupby('USER_ID').DAY.transform('size') / 2, 1, 2)
    halves = []
    for h in (1, 2):
        keep = D[D.half == h][['USER_ID', 'DAY']]
        f = participant_facets(E.merge(keep, on=['USER_ID', 'DAY']), D.merge(keep, on=['USER_ID', 'DAY']))
        halves.append(f.set_index('USER_ID')[[f'all_{x}' for x in FACETS]])
    rows = []
    for x in FACETS:
        a, b = halves[0][f'all_{x}'], halves[1][f'all_{x}']; k = a.notna() & b.notna()
        rows.append({'facet': x, 'n': int(k.sum()), 'pearson_r': float(np.corrcoef(a[k], b[k])[0, 1]),
                     'spearman_rho': float(a[k].corr(b[k], method='spearman')),
                     'design': 'first vs second half of each participant dense dates; facets recomputed within halves'})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', type=Path, default=database_path())
    a = ap.parse_args(argv); started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(CACHE / 'spike_daily.parquet', columns=['USER_ID', 'DAY'])
    cached = pd.read_parquet(CACHE / 'spike_participants.parquet')
    exposure = pd.read_parquet(CACHE / 'spike_exposure_daily.parquet', columns=['USER_ID', 'DAY', 'n_logs'])
    panel = pd.read_csv(PANEL, low_memory=False)
    panel = panel[panel.no_diabetes_range == True].copy()
    people = panel.USER_ID.to_numpy()
    days = daily.merge(panel[['USER_ID', 'tz_offset_hours']], on='USER_ID'); days['DAY'] = pd.to_datetime(days.DAY)
    with duckdb.connect(str(a.db), read_only=True) as con:
        con.execute('SET threads=6')
        E, D, M, mismatch = extract(con, days, people)
    guard = {'person_days': len(D), 'spikes': len(E), 'meals_evaluated': len(M), 'daily_metric_mismatch_days': mismatch}
    if mismatch:
        raise SystemExit(f'GUARD FAILED: event extraction differs from daily_metrics on {mismatch} days')
    P = participant_facets(E, D)
    chk = P.merge(cached[['USER_ID', 'daily_n_spikes', 'spike_resolution', 'high_logger']], on='USER_ID', validate='one_to_one')
    guard['frequency_max_abs_diff'] = float((chk.all_frequency - chk.daily_n_spikes).abs().max())
    guard['clearance_max_abs_diff'] = float((chk.all_clearance - chk.spike_resolution).abs().max())
    guard['frequency_missing_agree'] = bool((chk.all_frequency.isna() == chk.daily_n_spikes.isna()).all())
    guard['clearance_missing_agree'] = bool((chk.all_clearance.isna() == chk.spike_resolution.isna()).all())
    ex = exposure.assign(DAY=pd.to_datetime(exposure.DAY)).groupby('USER_ID').n_logs
    logging = pd.DataFrame({'n_dates': ex.size(), 'f3': ex.apply(lambda v: (v >= 3).mean()), 'f2': ex.apply(lambda v: (v >= 2).mean())})
    logging['high'] = (logging.f3 >= 0.8) & (logging.n_dates >= 14)
    logging['loose'] = (logging.f2 >= 0.6) & (logging.n_dates >= 14)
    logging['strict'] = (logging.f3 >= 0.9) & (logging.n_dates >= 14)
    lg = chk[['USER_ID', 'high_logger']].merge(logging.reset_index(), on='USER_ID', how='left')
    guard['high_logger_rule_reproduced'] = bool((lg.high.fillna(False) == lg.high_logger.astype(bool)).all())
    meal140 = M.groupby('USER_ID').peak_2h.agg(n_meals='size', meal140_fraction=lambda v: float((v >= 140).mean()))
    meal140.loc[meal140.n_meals < 10, 'meal140_fraction'] = np.nan
    P = P.merge(meal140.reset_index(), on='USER_ID', how='left').merge(logging[['high', 'loose', 'strict']].rename(
        columns=lambda c: f'logger_{c}').reset_index(), on='USER_ID', how='left')
    for c in ('logger_high', 'logger_loose', 'logger_strict'):
        P[c] = P[c].fillna(False).astype(bool)
    A = P.merge(panel[['USER_ID', *LEVEL, 'a1c', 'homa_ir', 'age_bucket', 'sex', 'bmi']], on='USER_ID', validate='one_to_one')
    A['hba1c_ge_5_7'] = np.where(A.a1c.notna(), (round_a1c(A.a1c) >= 5.7).astype(float), np.nan)
    A['homa_ir_ge_2_5'] = np.where(A.homa_ir.notna(), (A.homa_ir >= 2.5).astype(float), np.nan)
    rng = np.random.default_rng(SEED); rows = []
    def add(sample, method, outcome, result):
        for r in (result if isinstance(result, list) else [result]):
            rows.append({'sample': sample, 'method': method, 'outcome': outcome, **r})
    samples = {'primary': A, 'high_loggers': A[A.logger_high], 'other_participants': A[~A.logger_high],
               'loose_loggers': A[A.logger_loose], 'strict_loggers': A[A.logger_strict]}
    for x in FACETS:
        for s in ('primary', 'high_loggers', 'other_participants'):
            add(s, 'adjusted_or_per_sd', 'hba1c_ge_5_7', adjusted(samples[s], [f'all_{x}']))
        add('primary', 'peer_matched_pp', 'hba1c_ge_5_7', peer(A, f'all_{x}', 'hba1c_ge_5_7', rng))
        add('primary', 'peer_matched_pp', 'homa_ir_ge_2_5', peer(A, f'all_{x}', 'homa_ir_ge_2_5', rng))
        add('primary', 'adjusted_or_per_sd', 'homa_ir_ge_2_5', adjusted(A, [f'all_{x}'], 'homa_ir_ge_2_5'))
    for s in ('high_loggers', 'loose_loggers', 'strict_loggers'):
        for w in (WINDOWS if s == 'high_loggers' else ['primary']):
            for x in FACETS:
                for t in ('meal', 'unlinked'):
                    add(s, f'adjusted_or_per_sd|window={w}', 'hba1c_ge_5_7', adjusted(samples[s], [f'{t}_{w}_{x}']))
                add(s, f'joint_meal_vs_unlinked|window={w}', 'hba1c_ge_5_7', adjusted(samples[s], [f'meal_{w}_{x}', f'unlinked_{w}_{x}']))
    add('high_loggers', 'adjusted_or_per_sd', 'hba1c_ge_5_7', adjusted(samples['high_loggers'], ['meal140_fraction']))
    for x in ('meal_primary_clearance', 'unlinked_primary_clearance'):
        add('high_loggers', 'peer_matched_pp', 'homa_ir_ge_2_5', peer(samples['high_loggers'], x, 'homa_ir_ge_2_5', rng))
    assoc = pd.DataFrame(rows)
    rep = repeatability(E, D)
    desc = []
    for s, f in samples.items():
        desc.append({'sample': s, 'n': len(f), 'mean_glucose_mean': float(f.mean_glucose.mean()),
                     'hba1c_ge_5_7_pct': float(100 * f.hba1c_ge_5_7.mean()), 'bmi_mean': float(f.bmi.mean()),
                     'frequency_median': float(f.all_frequency.median()), 'size_median': float(f.all_size.median()),
                     'clearance_median': float(f.all_clearance.median()),
                     **{f'meal_linked_share_median_{w}': float((f[f'meal_{w}_frequency'] / f.all_frequency.replace(0, np.nan)).median()) for w in WINDOWS},
                     'meal140_fraction_median': float(f.meal140_fraction.median())})
    desc = pd.DataFrame(desc)
    files = {'spike_facets_participants.parquet': P, 'spike_facets_associations.csv': assoc,
             'spike_facets_repeatability.csv': rep, 'spike_facets_descriptives.csv': desc}
    for name, frame in files.items():
        (frame.to_parquet if name.endswith('.parquet') else frame.to_csv)(OUT / name, index=False)
    meta = {'definition_version': DEFINITION_VERSION, 'windows_minutes': WINDOWS,
            'level_covariates': LEVEL, 'seed': SEED, 'bootstrap_replicates': BOOT, 'mean_glucose_bins': BINS,
            'min_eligible_days': MIN_DAYS, 'min_resolved_spikes': MIN_RES, 'guard': guard,
            'runtime_seconds': round(time.monotonic() - started, 1)}
    (OUT / 'spike_facets.metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    print(json.dumps(guard))


if __name__ == '__main__':
    main()
