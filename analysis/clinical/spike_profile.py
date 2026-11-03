#!/usr/bin/env python3
"""Computes the cohort-aggregate CGM spike profile (peak-aligned glucose excursion shape).

Spikes are exactly those of the spike-width producer (analysis/clinical/spike_width.py, which runs the
spike-width detector re-checked against the spike-facets module): a local maximum >= 30 mg/dL above the minimum of the preceding 90 observed
minutes on 5-min local-day cells of the anchor-window days (dense days, spike_daily.parquet), no interpolation; peaks
< 30 min apart keep the larger rise; a day is eligible with >= 192 searchable centres. This module imports that producer's
module and calls its detector (load_spike_width_check().extract / day_events_shape and aggregate) unchanged; nothing is re-implemented.

Profile. For every resolved spike (falls at least halfway back) on an eligible day, glucose minus the spike's preceding 90-min
minimum is read on a common grid of -120 to +180 min around the peak (5-min cells, observed contiguous CGM only, no
interpolation). Two variants:
  contiguous  every observed cell of the same uninterrupted CGM run;
  isolated    additionally only cells strictly between the previous and the next detected peak of the same run (neighbouring
              spikes do not enter the tails).
Participant profile = mean over that participant's spikes at each grid time; cohort profile = median and IQR over participants
(participants with spike width, i.e. >= 5 eligible days and >= 5 resolved spikes). Averaging peak-aligned traces narrows the
curve (spikes of different widths only coincide at the peak), so its width at half height is shorter than the cohort median
of participant spike widths. The level-registered profile is therefore also issued and is the one used for display: for each
resolved spike and each fraction f of its rise (0, 0.1, ..., 0.9), the time of the first cell after the onset minimum at or
above minimum + f * rise (f = 0: the minimum itself) and of the first cell after the peak at or below it (the detector's half-level
rules with 1/2 replaced by f; downslope searched to the next peak or 72 cells; below half the rise some spikes never get there
and are excluded at that level); participant mean per level, cohort median (IQR) over participants. At f = 0.5 these are exactly
the participant upslope half-time and clearance time, so the drawn half-rise crossings are the cohort medians. Profile-derived features (peak height, half
level, half-rise crossing times, width and clearance of the median curve; linear interpolation between 5-min cells) are reported
next to the paper's measures, the cohort medians (IQR) of per-participant spike height (all_size), spike width, clearance time,
upslope half-time and spike frequency (spikes per eligible day).

Structural checks (fail closed): detected events and person-days equal the producer's guard counts; every
spike's traced peak minus onset equals its detected rise exactly; recomputed participant width, clearance and upslope equal the
cached participant values (|d| < 1e-9, same missingness); recomputed participant height and frequency equal spike_facets (same);
per spike the level-0.5 crossings equal the detector's upslope half-time and clearance and the level-0 onset equals its rise
time; participant means at level 0.5 equal the participant upslope half-time and clearance; cohort medians monotone in the level.

Inputs:
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/clinical/spike_width (analysis/clinical/spike_width.py): metadata.json, width_summary.json,
    _cache/spike_width_participants.parquet
  OUTPUTS/meals/spike_facets/spike_facets_participants.parquet (analysis/meals/spike_facets.py)
  CGM data via LEVELS_DB (spike_daily.parquet cache used by the spike-width/spike-facets modules)
Outputs (aggregate only, no identifiers): OUTPUTS/clinical/spike_profile
  spike_profile.csv         variant, t_min, n_participants, median, q1, q3 (mg/dL above the 90-min minimum)
  spike_profile_levels.csv  side (up/down), level (fraction of the rise), n_participants, t_median, t_q1, t_q3 (min from peak),
                            pooled share of spikes reaching that level on the downslope
  spike_profile_summary.json cohort medians (IQR, n) and profile-derived features
  metadata.json             structural checks and run metadata
Usage: python analysis/clinical/spike_profile.py
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
PUB = OUTPUTS
OUT = PUB / 'clinical/spike_profile'
DB = database_path()
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
PW_SCRIPT = ROOT / 'analysis/clinical/spike_width.py'
W_DIR = PUB / 'clinical/spike_width'
W_CACHE = PUB / 'clinical/_cache/spike_width_participants.parquet'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
GRID = np.arange(-120, 181, 5)          # minutes relative to the peak
LEVELS = np.round(np.arange(0, 1.0, 0.1), 1)   # fractions of the rise for the level-registered profile
TOL = 1e-9
GUARDS: dict = {}


def guard(name, ok, detail=None):
    GUARDS[name] = dict(status='passed' if ok else 'failed', detail=detail)
    if not ok:
        raise AssertionError(f'guard failed: {name}: {detail}')


def load_producer():
    spec = importlib.util.spec_from_file_location('spike_width_for_profile', PW_SCRIPT)
    mod = importlib.util.module_from_spec(spec); sys.modules['spike_width_for_profile'] = mod; spec.loader.exec_module(mod)
    return mod


def traces_for_day(m, bins, g):
    """Detector call (unchanged) plus, per spike, the peak-aligned trace and the level-crossing times.

    m.day_events_shape returns per spike (onset bin, rise, half-recovery min, rise time, upslope, upslope contiguous, drop);
    the peak cell is onset + rise_time / 5 cells inside the same uninterrupted run (bins are unique and contiguous in a run).
    Level crossings use the detector's own rules with the half level generalised to level f (threshold = peak - (1 - f) * rise):
    up = first cell after the onset minimum at or above the threshold (f = 0: the onset minimum itself); down = first cell after
    the peak at or below it, searched to the next peak of the same run or 72 cells (the detector's recovery window)."""
    evs, eligible = m.day_events_shape(bins, g)
    run_id = np.r_[0, np.cumsum(np.diff(bins) != 1)]
    run_end = {int(r): int(np.flatnonzero(run_id == r)[-1]) for r in np.unique(run_id)}
    pos = {int(b): i for i, b in enumerate(bins)}
    peaks = []
    for e in evs:
        io = pos[e[0]]; ip = io + int(round(e[3] / 5))
        peaks.append((io, ip))
    out, ok = [], dict(rise=True, half=True)
    for k, (e, (io, ip)) in enumerate(zip(evs, peaks)):
        a = e[1]
        if run_id[ip] != run_id[io] or g[ip] - g[io] != a:
            ok['rise'] = False
        idx = ip + GRID // 5
        inside = (idx >= 0) & (idx < len(g))
        idc = np.clip(idx, 0, len(g) - 1)
        same_run = inside & (run_id[idc] == run_id[ip])
        v = np.where(same_run, g[idc] - g[io], np.nan)
        prev_p = max([q for (_, q) in peaks[:k] if run_id[q] == run_id[ip]], default=-10 ** 9)
        next_p = min([q for (_, q) in peaks[k + 1:] if run_id[q] == run_id[ip]], default=10 ** 9)
        iso = same_run & (idc > prev_p) & (idc < next_p)
        seg = g[io + 1:ip + 1]
        up = np.array([(io - ip) * 5.0] + [(io + 1 + int(np.flatnonzero(seg >= g[ip] - (1 - f) * a)[0]) - ip) * 5.0 for f in LEVELS[1:]])
        stop = min(ip + m.core.RECOVERY_CELLS + 1, next_p, run_end[int(run_id[ip])] + 1)
        tail = g[ip + 1:stop]
        down = np.full(len(LEVELS), np.nan)
        for j, f in enumerate(LEVELS):
            r = np.flatnonzero(tail <= g[ip] - (1 - f) * a)
            if len(r):
                down[j] = (r[0] + 1) * 5.0
        j5 = int(np.flatnonzero(LEVELS == 0.5)[0])
        same_half = ((np.isnan(down[j5]) and np.isnan(e[2])) or down[j5] == e[2]) and -up[j5] == e[4] and -up[0] == e[3]
        if not same_half:
            ok['half'] = False
        out.append((v, np.where(iso, v, np.nan), up, down))
    return evs, eligible, out, ok


def extract_profiles(m, con, days, people):
    """Same batching and inputs as the spike-width producer's extract(); returns events, days, per-participant mean profiles and crossings."""
    ev, dd, bad = [], [], dict(rise=0, half=0)
    prof = {'contiguous': [], 'isolated': []}; land = []
    nG, nL = len(GRID), len(LEVELS)
    for i in range(0, len(people), 300):
        con.register('batch_days', days[days.USER_ID.isin(people[i:i + 300])])
        raw = con.execute(m.s_spike_facets.s_spike_exposure.raw_day_query()).df()
        con.unregister('batch_days')
        acc = {}
        for r in raw.itertuples(index=False):
            b = np.asarray(r.bins, np.int64); x = np.asarray(r.glucose, float)
            evs, eligible, tr, ok = traces_for_day(m, b, x)
            bad['rise'] += not ok['rise']; bad['half'] += not ok['half']
            dd.append((r.USER_ID, r.DAY, eligible))
            for e, (vc, vi, up, down) in zip(evs, tr):
                ev.append((r.USER_ID, r.DAY, eligible, *e[1:]))
                if eligible and np.isfinite(e[2]):          # resolved spikes on eligible days (the width/clearance spikes)
                    a = acc.setdefault(r.USER_ID, dict(cs=np.zeros(nG), cn=np.zeros(nG), is_=np.zeros(nG), in_=np.zeros(nG),
                                                       us=np.zeros(nL), ds=np.zeros(nL), dn=np.zeros(nL), n=0))
                    fc, fi, fd = np.isfinite(vc), np.isfinite(vi), np.isfinite(down)
                    a['cs'][fc] += vc[fc]; a['cn'] += fc; a['is_'][fi] += vi[fi]; a['in_'] += fi
                    a['us'] += up; a['ds'][fd] += down[fd]; a['dn'] += fd; a['n'] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            for u, a in acc.items():
                prof['contiguous'].append((u, a['cs'] / a['cn'], a['cn'].copy()))
                prof['isolated'].append((u, a['is_'] / a['in_'], a['in_'].copy()))
                land.append((u, a['n'], a['us'] / a['n'], a['ds'] / a['dn'], a['dn'] / a['n']))
        print(f'  {min(i + 300, len(people)):,}/{len(people):,} participants', flush=True)
    E = pd.DataFrame(ev, columns=['USER_ID', 'DAY', 'eligible', 'rise', 'half_recovery', 'rise_time',
                                  'upslope_half_time', 'upslope_half_time_contiguous', 'drop'])
    D = pd.DataFrame(dd, columns=['USER_ID', 'DAY', 'eligible'])
    return E, D, prof, land, bad


def size_frequency(E, D, min_days):
    """spike-facets module participant_facets() 'all' frequency and size (eligible days; NaN below the minimum eligible days)."""
    out = pd.DataFrame(index=pd.Index(D.USER_ID.unique(), name='USER_ID'))
    out['n_eligible_days'] = D[D.eligible].groupby('USER_ID').size().reindex(out.index).fillna(0).astype(int)
    el = E[E.eligible]
    out['all_frequency'] = el.groupby('USER_ID').size().reindex(out.index).fillna(0) / out.n_eligible_days.replace(0, np.nan)
    out['all_size'] = el.groupby('USER_ID').rise.mean().reindex(out.index)
    out.loc[out.n_eligible_days < min_days, ['all_frequency', 'all_size']] = np.nan
    return out.reset_index()


def crossing(t, y, level, side):
    """Interpolated time where the curve crosses `level`, searching outward from the peak (t = 0)."""
    i0 = int(np.flatnonzero(t == 0)[0])
    rng = range(i0, 0, -1) if side == 'up' else range(i0, len(t) - 1)
    for i in rng:
        j = i - 1 if side == 'up' else i + 1
        if y[j] <= level <= y[i] or y[i] <= level <= y[j]:
            if y[j] == y[i]:
                return float(t[j])
            return float(t[i] + (level - y[i]) * (t[j] - t[i]) / (y[j] - y[i]))
    return float('nan')


def describe(s):
    s = s.dropna()
    return dict(n=int(len(s)), median=float(s.median()), q1=float(s.quantile(.25)), q3=float(s.quantile(.75)), sd=float(s.std()))


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(PANEL, low_memory=False)
    PRI = P.loc[P.no_diabetes_range.astype(bool)].copy()
    wmeta = json.loads((W_DIR / 'metadata.json').read_text())

    pw = load_producer(); m = pw.load_spike_width_check()
    # inputs exactly as analysis/clinical/spike_width.py cgm_shape()
    daily = pd.read_parquet(m.s_spike_facets.CACHE / 'spike_daily.parquet', columns=['USER_ID', 'DAY'])
    panel = pd.read_csv(m.s_spike_facets.PANEL, low_memory=False); panel = panel[panel.no_diabetes_range == True].copy()  # noqa: E712
    days = daily.merge(panel[['USER_ID', 'tz_offset_hours']], on='USER_ID'); days['DAY'] = pd.to_datetime(days.DAY)
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute("SET temp_directory='/tmp/duckdb_tmp'"); con.execute('SET threads=6')
        E, D, prof, land, bad = extract_profiles(m, con, days, panel.USER_ID.to_numpy())
    pg = wmeta['guards']['events_reproduce_spike_facets']['detail']
    guard('events_equal_producer', len(D) == pg['person_days'] and len(E) == pg['spikes'],
          dict(person_days=int(len(D)), spikes=int(len(E))))
    guard('traced_peak_equals_detected_rise', bad['rise'] == 0, int(bad['rise']))
    guard('level_crossings_at_half_equal_detector', bad['half'] == 0, int(bad['half']))

    # participant values with the spike-width producer's aggregation, compared with the cache
    E2 = m.add_shape(E); Pn = m.aggregate(E2, D, m.MEASURES)
    cache = pd.read_parquet(W_CACHE)
    chk = cache.merge(Pn[['USER_ID'] + pw.SHAPE], on='USER_ID', how='left', suffixes=('_cache', ''), validate='one_to_one')
    diffs = {c: float(np.nanmax(np.abs(chk[c] - chk[c + '_cache']))) for c in pw.SHAPE}
    miss = all(bool((chk[c].isna() == chk[c + '_cache'].isna()).all()) for c in pw.SHAPE)
    guard('participant_width_clearance_upslope_reproduce_cache', miss and max(diffs.values()) < TOL,
          dict(max_abs_diff=diffs, missingness_agree=miss))
    SF = size_frequency(E, D, m.s_spike_facets.MIN_DAYS)
    fac = pd.read_parquet(FACETS, columns=['USER_ID', 'all_frequency', 'all_size'])
    chk2 = fac.merge(SF, on='USER_ID', how='left', suffixes=('_facets', ''), validate='one_to_one')
    d2 = {c: float(np.nanmax(np.abs(chk2[c] - chk2[c + '_facets']))) for c in ('all_frequency', 'all_size')}
    miss2 = all(bool((chk2[c].isna() == chk2[c + '_facets'].isna()).all()) for c in ('all_frequency', 'all_size'))
    guard('participant_height_frequency_reproduce_facets', miss2 and max(d2.values()) < TOL,
          dict(max_abs_diff=d2, missingness_agree=miss2))

    # cohort medians of participant values (primary cohort)
    V = PRI[['USER_ID']].merge(Pn[['USER_ID'] + pw.SHAPE], on='USER_ID', how='left').merge(SF[['USER_ID', 'all_size', 'all_frequency']],
                                                                                          on='USER_ID', how='left')
    V = V.merge(Pn[['USER_ID', 'rise_time']], on='USER_ID', how='left')
    med = {k: describe(V[k]) for k in ('all_size', 'spike_width', 'all_clearance', 'upslope_half_time', 'rise_time', 'all_frequency')}
    ws = json.loads((W_DIR / 'width_summary.json').read_text())['descriptives']
    dd_ = {k: max(abs(med[k][s] - ws[k][s]) for s in ('median', 'q1', 'q3', 'sd')) for k in pw.SHAPE}
    guard('cohort_width_clearance_upslope_equal_width_summary', max(dd_.values()) < TOL, dd_)

    # cohort profile over the width participants
    width_ids = set(V.loc[V.spike_width.notna(), 'USER_ID'])
    rows, feats = [], {}
    for var, lst in prof.items():
        M = np.vstack([p for u, p, _ in lst if u in width_ids]); N = np.vstack([c for u, _, c in lst if u in width_ids])
        for j, t in enumerate(GRID):
            col = M[:, j][np.isfinite(M[:, j])]
            rows.append(dict(variant=var, t_min=int(t), n_participants=int(len(col)), median=float(np.median(col)),
                             q1=float(np.quantile(col, .25)), q3=float(np.quantile(col, .75)),
                             spikes_contributing=int(N[:, j].sum())))
        prf = pd.DataFrame([r for r in rows if r['variant'] == var])
        t, y = prf.t_min.to_numpy(float), prf['median'].to_numpy(float)
        h = float(y[t == 0][0]); half = h / 2
        up, down = crossing(t, y, half, 'up'), crossing(t, y, half, 'down')
        feats[var] = dict(peak_height=h, half_level=half, t_half_up=up, t_half_down=down, width=down - up, clearance=down,
                          upslope=-up, minimum_before_peak=float(y[t < 0].min()), t_minimum_before_peak=float(t[t < 0][np.argmin(y[t < 0])]),
                          value_at_plus_180=float(y[-1]), value_at_minus_120=float(y[0]),
                          min_participants_any_t=int(prf.n_participants.min()))
    PR = pd.DataFrame(rows)
    guard('profile_peak_positive_and_max_at_0', all(PR.loc[(PR.variant == v)].set_index('t_min')['median'].idxmax() == 0 for v in prof))
    PR.to_csv(OUT / 'spike_profile.csv', index=False)

    # level-registered profile: cohort median (IQR) of participant mean crossing times at each fraction of the rise
    L = pd.DataFrame({'USER_ID': [u for u, *_ in land], 'n_res': [n for _, n, *_ in land]})
    UP = np.vstack([x for *_, x, _, _ in land]); DN = np.vstack([x for *_, x, _ in land]); SH = np.vstack([x for *_, x in land])
    keep = L.USER_ID.isin(width_ids).to_numpy()
    guard('landmark_participants', int(keep.sum()) == len(width_ids), int(keep.sum()))
    j5 = int(np.flatnonzero(LEVELS == 0.5)[0])
    lp = L.assign(up5=-UP[:, j5], dn5=DN[:, j5], up0=-UP[:, 0]).merge(Pn[['USER_ID', 'upslope_half_time', 'all_clearance', 'rise_time']],
                                                                       on='USER_ID', how='left')
    lk = lp.loc[keep]
    dl = dict(upslope=float((lk.up5 - lk.upslope_half_time).abs().max()), clearance=float((lk.dn5 - lk.all_clearance).abs().max()),
              rise_time=float((lk.up0 - lk.rise_time).abs().max()))
    guard('landmarks_at_half_reproduce_participant_upslope_clearance_rise_time', max(dl.values()) < TOL, dl)
    guard('all_resolved_spikes_reach_half_and_above_on_downslope', bool((SH[keep][:, LEVELS >= 0.5] == 1).all()))
    LR = []
    for j, f in enumerate(LEVELS):
        for side, A in (('up', UP), ('down', DN)):
            col = A[keep, j]; col = col[np.isfinite(col)]
            LR.append(dict(side=side, level=float(f), n_participants=int(len(col)), median_min=float(np.median(col)),
                           q1_min=float(np.quantile(col, .25)), q3_min=float(np.quantile(col, .75)),
                           pooled_share_resolved=float(1.0 if side == 'up' else (SH[keep, j] * L.n_res[keep]).sum() / L.n_res[keep].sum())))
    LR = pd.DataFrame(LR)
    upm = LR.loc[LR.side == 'up'].sort_values('level').median_min.to_numpy(); dnm = LR.loc[LR.side == 'down'].sort_values('level').median_min.to_numpy()
    guard('landmark_medians_monotone', bool((np.diff(upm) > 0).all() and (np.diff(dnm) < 0).all()), dict(up=upm.tolist(), down=dnm.tolist()))
    r5 = LR.loc[LR.level == 0.5].set_index('side')
    guard('landmark_half_medians_equal_cohort_medians', abs(r5.loc['up', 'median_min'] + med['upslope_half_time']['median']) < TOL
          and abs(r5.loc['down', 'median_min'] - med['all_clearance']['median']) < TOL,
          dict(up=float(r5.loc['up', 'median_min']), down=float(r5.loc['down', 'median_min'])))
    LR = LR.rename(columns={'median_min': 't_median', 'q1_min': 't_q1', 'q3_min': 't_q3'})   # minutes relative to the peak
    guard('landmark_sides_signed', bool((LR.loc[LR.side == 'up', 't_median'] < 0).all() and (LR.loc[LR.side == 'down', 't_median'] > 0).all()))
    LR.to_csv(OUT / 'spike_profile_levels.csv', index=False)
    feats['level_registered'] = dict(height_scale='cohort median spike height (all_size)', t_half_up=float(LR.loc[(LR.side == 'up') & (LR.level == 0.5), 't_median'].iloc[0]),
                                     t_half_down=float(LR.loc[(LR.side == 'down') & (LR.level == 0.5), 't_median'].iloc[0]),
                                     t_onset=float(LR.loc[(LR.side == 'up') & (LR.level == 0.0), 't_median'].iloc[0]))
    feats['level_registered']['width_between_half_crossings'] = feats['level_registered']['t_half_down'] - feats['level_registered']['t_half_up']
    summ = dict(definition='glucose minus the preceding 90-min minimum, resolved spikes on eligible days, aligned at the peak; '
                           'participant mean per grid time, cohort median and IQR over participants with spike width',
                grid_min=[int(GRID[0]), int(GRID[-1]), 5], n_participants=int(len(width_ids)),
                n_spikes_detected=int(len(E)), n_person_days=int(len(D)),
                n_resolved_spikes_eligible_days_width_participants=int(E[E.eligible & E.half_recovery.notna() & E.USER_ID.isin(width_ids)].shape[0]),
                cohort_medians=med, profile_features=feats,
                labels={'all_size': 'spike height (mg/dL)', 'spike_width': 'spike width (min)', 'all_clearance': 'clearance time (min)',
                        'upslope_half_time': 'upslope half-time (min)', 'rise_time': 'rise time, 90-min minimum to peak (min)',
                        'all_frequency': 'spike frequency (per day)'})
    (OUT / 'spike_profile_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    meta = dict(analysis='spike_profile', version='spike-profile', status='passed', guards=GUARDS,
                runtime_seconds=round(time.time() - t0, 1))
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: v['status'] for k, v in GUARDS.items()}), meta['runtime_seconds'])
    print(json.dumps(dict(medians={k: round(v['median'], 2) for k, v in med.items()},
                          features={v: {k: (round(x, 2) if isinstance(x, float) else x) for k, x in f.items()} for v, f in feats.items()})))


if __name__ == '__main__':
    main()
