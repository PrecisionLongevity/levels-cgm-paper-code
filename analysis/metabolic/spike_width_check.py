#!/usr/bin/env python3
"""Spike recovery time compared with spike width, rise time and rate of decline.

Spike recovery time (facet all_clearance of analysis/meals/spike_facets.py) = minutes from the spike peak until glucose
first falls to or below peak - rise/2 (= preceding 90-minute minimum + rise/2), per resolved spike, averaged over
resolved spikes on eligible days (>=192 searchable centres), >=5 eligible days and >=5 resolved spikes.

Per resolved spike (same spikes, same eligibility, same aggregation) this module adds:
  spike_width         minutes from the first observed cell after the preceding minimum at or above min + rise/2
                      (way up) to the half-recovery cell (way down); = upslope_half_time + recovery time.
  upslope_half_time   minutes from that upward half-rise crossing to the peak.
  rise_time           minutes from the preceding 90-minute minimum (spike onset) to the peak.
  decline_rate        observed fall (peak glucose - glucose at the half-recovery cell, mg/dL) / recovery minutes.
  spike_width_contiguous  sensitivity: upward crossing taken as the start of the last uninterrupted run at or above
                      min + rise/2 that ends at the peak (full width at half rise of the peak's own lobe).
  decline_rate_nominal    sensitivity: (rise/2) / recovery minutes.
Participant value = mean over resolved spikes on eligible days; NaN unless >=5 eligible days and >=5 resolved spikes.

Estimates use the frame and estimator of analysis/metabolic/unadjusted_insulin_markers.py: canonical primary cohort,
OLS of log fasting insulin / log HOMA-IR on the standardized measure (HC3), adjustment 'none' and 'mean_glucose_only'
(average glucose as the single covariate); % per SD. Extra rows enter recovery time together with each other measure
at the same average glucose. Split-half repeatability: first vs second half of each participant's dense dates with
measures recomputed within halves. Correlations among the measures: participant level (Pearson, Spearman) and pooled
over spikes (Spearman).

Consistency checks (stop on failure): event extraction reproduces spike_facets.py (person-days, spikes, per-day
events); aggregated recovery time reproduces the saved all_clearance; split-half recovery-time rows reproduce
spike_facets_repeatability.csv; the insulin-marker rows reproduce unadjusted_insulin_markers.py; spike_width equals
upslope_half_time + recovery time for every spike.

Inputs: glucose readings via LEVELS_DB; OUTPUTS/meals/spike_facets and its cache (analysis/meals/spike_facets.py);
OUTPUTS/metabolic/unadjusted_insulin_markers (analysis/metabolic/unadjusted_insulin_markers.py).
Outputs: OUTPUTS/metabolic/spike_width_check/ (aggregate tables, summary and a short Markdown report).
The functions of this module are also imported by analysis/clinical/spike_width.py.

Usage: python analysis/metabolic/spike_width_check.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
VERSION = 'spike-width-check'
PUB = OUTPUTS
MP = PUB / 'metabolic'
OUT = MP / 'spike_width_check'
S_SPIKE_FACETS = ROOT / 'analysis/meals/spike_facets.py'
S_UNADJUSTED = ROOT / 'analysis/metabolic/unadjusted_insulin_markers.py'
R_UNADJUSTED = MP / 'unadjusted_insulin_markers'
F_SPIKE_FACETS = PUB / 'meals/spike_facets'
TOL = 1e-9
INS = {'fasting_insulin': True, 'homa_ir': True}  # fasting insulin listed first (paper convention)
MEASURES = ['all_clearance', 'spike_width', 'upslope_half_time', 'rise_time', 'decline_rate',
            'spike_width_contiguous', 'decline_rate_nominal']
CORE4 = ['all_clearance', 'spike_width', 'rise_time', 'decline_rate']
LABEL = {'all_clearance': 'Spike recovery time (min, peak to halfway back)',
         'spike_width': 'Spike width (min above half the rise, up and down)',
         'upslope_half_time': 'Upslope half-time (min, halfway up to peak)',
         'rise_time': 'Rise time (min, pre-spike minimum to peak)',
         'decline_rate': 'Rate of decline (mg/dL per min, peak to halfway back)',
         'spike_width_contiguous': 'Spike width, peak lobe only (sensitivity)',
         'decline_rate_nominal': 'Rate of decline, half rise / recovery time (sensitivity)',
         'all_size': 'Spike size (mg/dL rise)', 'all_frequency': 'Spike frequency (per day)',
         'mean_glucose': 'Average glucose (mg/dL)'}


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_spike_facets = _load('spike_facets_for_spike_width_check', S_SPIKE_FACETS)
core = s_spike_facets.core


def day_events_shape(bins, g):
    """day_events() of spike_facets.py plus shape fields per spike (the detector is the same; checked against spike_facets.py)."""
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
            thr = x[p] - a / 2
            r = np.flatnonzero(x[p + 1:stop] <= thr)
            half = float((r[0] + 1) * 5) if len(r) else np.nan
            seg = x[o + 1:p + 1]                      # cells after the minimum up to and including the peak
            up_first = o + 1 + int(np.flatnonzero(seg >= thr)[0])
            below = np.flatnonzero(seg < thr)
            up_contig = o + 1 + (int(below[-1]) + 1 if len(below) else 0)
            if len(r):
                rc = p + 1 + int(r[0])
                drop = float(x[p] - x[rc])
            else:
                drop = np.nan
            out.append((int(bins[run[o]]), a, half, float((p - o) * 5), float((p - up_first) * 5),
                        float((p - up_contig) * 5), drop))
    return out, n_search >= core.MIN_SEARCH_CELLS


def extract(con, days, people):
    ev, dd, bad = [], [], 0
    for i in range(0, len(people), 300):
        con.register('batch_days', days[days.USER_ID.isin(people[i:i + 300])])
        raw = con.execute(s_spike_facets.s_spike_exposure.raw_day_query()).df()
        con.unregister('batch_days')
        for r in raw.itertuples(index=False):
            b = np.asarray(r.bins, np.int64); x = np.asarray(r.glucose, float)
            evs, eligible = day_events_shape(b, x)
            ref, ref_elig = s_spike_facets.day_events(b, x)
            same = eligible == ref_elig and len(ref) == len(evs) and all(
                e[0] == q[0] and e[1] == q[1] and ((np.isnan(e[2]) and np.isnan(q[2])) or e[2] == q[2])
                for e, q in zip(evs, ref))
            bad += not same
            dd.append((r.USER_ID, r.DAY, eligible))
            for e in evs:
                ev.append((r.USER_ID, r.DAY, eligible, *e[1:]))
        print(f'  {min(i + 300, len(people)):,}/{len(people):,} participants', flush=True)
    E = pd.DataFrame(ev, columns=['USER_ID', 'DAY', 'eligible', 'rise', 'half_recovery', 'rise_time',
                                  'upslope_half_time', 'upslope_half_time_contiguous', 'drop'])
    D = pd.DataFrame(dd, columns=['USER_ID', 'DAY', 'eligible'])
    return E, D, bad


def add_shape(E):
    E = E.copy()
    E['all_clearance'] = E.half_recovery
    E['spike_width'] = E.upslope_half_time + E.half_recovery
    E['spike_width_contiguous'] = E.upslope_half_time_contiguous + E.half_recovery
    E['decline_rate'] = E['drop'] / E.half_recovery
    E['decline_rate_nominal'] = (E.rise / 2) / E.half_recovery
    return E


def aggregate(E, D, cols):
    """Clearance as in participant_facets() of spike_facets.py: resolved spikes on eligible days, >=5 days, >=5 resolved."""
    out = pd.DataFrame(index=pd.Index(D.USER_ID.unique(), name='USER_ID'))
    out['n_eligible_days'] = D[D.eligible].groupby('USER_ID').size().reindex(out.index).fillna(0).astype(int)
    el = E[E.eligible & E.half_recovery.notna()]
    g = el.groupby('USER_ID')
    n_res = g.size().reindex(out.index).fillna(0)
    out['n_resolved'] = n_res.astype(int)
    for c in cols:
        out[c] = g[c].mean().reindex(out.index).where(n_res >= s_spike_facets.MIN_RES)
    out.loc[out.n_eligible_days < s_spike_facets.MIN_DAYS, cols] = np.nan
    return out.reset_index()


def halves(E, D, cols):
    D = D.sort_values(['USER_ID', 'DAY']).copy()
    D['half'] = np.where(D.groupby('USER_ID').cumcount() < D.groupby('USER_ID').DAY.transform('size') / 2, 1, 2)
    res = []
    for h in (1, 2):
        keep = D[D.half == h][['USER_ID', 'DAY']]
        res.append(aggregate(E.merge(keep, on=['USER_ID', 'DAY']), D.merge(keep, on=['USER_ID', 'DAY']), cols).set_index('USER_ID'))
    rows = []
    for c in cols:
        a, b = res[0][c], res[1][c].reindex(res[0].index); k = a.notna() & b.notna()
        r = float(np.corrcoef(a[k], b[k])[0, 1]); n = int(k.sum())
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        rows.append(dict(measure=c, label=LABEL[c], n=n, pearson_r=r, r_ci_low=float(np.tanh(z - 1.959963984540054 * se)),
                         r_ci_high=float(np.tanh(z + 1.959963984540054 * se)),
                         spearman_rho=float(a[k].corr(b[k], method='spearman')),
                         spearman_brown=float(2 * r / (1 + r)),
                         design='first vs second half of each participant dense dates; measures recomputed within halves'))
    return pd.DataFrame(rows)


def fit_rows(u, frame, measures, adj_sets, markers=INS, extra=None):
    rows = []
    for marker, log in markers.items():
        d = frame.loc[frame[marker] > 0].copy(); d['y'] = np.log(d[marker])
        for adj_name, adj in adj_sets.items():
            for x in measures:
                a = [c for c in adj if c != x]
                s = d.dropna(subset=[x])
                rows.append(dict(marker=marker, adjustment=adj_name, measure=x, label=LABEL.get(x, x),
                                 **u.s_insulin_markers.fit(s, x, a, log=log)))
    return pd.DataFrame(rows)


def _ci(r):
    return f'{r.estimate:+.1f}% ({r.ci_low:+.1f} to {r.ci_high:+.1f}; n {int(r.n):,})'


def summary_table_and_note(A, J, REP, C, DESC, share_up0):
    """One row per measure."""
    get = lambda mk, adj, ms: A.loc[(A.marker == mk) & (A.adjustment == adj) & (A.measure == ms)].iloc[0]  # noqa: E731
    jget = lambda mk, oth, ms: J.loc[(J.marker == mk) & (J.model == f'mean_glucose + all_clearance + {oth}') & (J.measure == ms)].iloc[0]  # noqa: E731
    pc = C.loc[C.level == 'participant']; sc = C.loc[C.level == 'spike_pooled']
    corr = lambda tab, a, b: tab.loc[((tab.a == a) & (tab.b == b)) | ((tab.a == b) & (tab.b == a))].iloc[0]  # noqa: E731
    rep = REP.set_index('measure'); desc = DESC.set_index('measure')
    rows = []
    for ms in ['all_clearance', 'spike_width', 'upslope_half_time', 'rise_time', 'decline_rate']:
        row = dict(measure=ms, label=LABEL[ms], participant_median=desc.loc[ms, 'participant_median'],
                   participant_q1=desc.loc[ms, 'participant_q1'], participant_q3=desc.loc[ms, 'participant_q3'],
                   split_half_r=rep.loc[ms, 'pearson_r'], split_half_r_ci_low=rep.loc[ms, 'r_ci_low'],
                   split_half_r_ci_high=rep.loc[ms, 'r_ci_high'], split_half_n=int(rep.loc[ms, 'n']))
        if ms != 'all_clearance':
            c = corr(pc, 'all_clearance', ms); s = corr(sc, 'all_clearance', ms)
            row.update(r_with_recovery_participant=c.pearson_r, rho_with_recovery_participant=c.spearman_rho,
                       rho_with_recovery_spike=s.spearman_rho)
        for mk in INS:
            for adj in ('none', 'mean_glucose_only'):
                r = get(mk, adj, ms)
                row.update({f'{mk}_{adj}_n': int(r.n), f'{mk}_{adj}_pct_per_sd': r.estimate,
                            f'{mk}_{adj}_ci_low': r.ci_low, f'{mk}_{adj}_ci_high': r.ci_high,
                            f'{mk}_{adj}_detected': bool(not r.ci_includes_zero)})
        rows.append(row)
    T = pd.DataFrame(rows)
    fi = {ms: get('fasting_insulin', 'mean_glucose_only', ms) for ms in T.measure}
    hi = {ms: get('homa_ir', 'mean_glucose_only', ms) for ms in T.measure}
    ju = {k: jget('fasting_insulin', 'upslope_half_time', k) for k in ('all_clearance', 'upslope_half_time')}
    jd = {k: jget('fasting_insulin', 'decline_rate', k) for k in ('all_clearance', 'decline_rate')}
    jr = {k: jget('fasting_insulin', 'rise_time', k) for k in ('all_clearance', 'rise_time')}
    js = {k: jget('fasting_insulin', 'all_size', k) for k in ('all_clearance', 'all_size')}
    rw = corr(pc, 'all_clearance', 'spike_width'); rd = corr(pc, 'all_clearance', 'decline_rate'); rr = corr(pc, 'all_clearance', 'rise_time')
    ru = corr(pc, 'all_clearance', 'upslope_half_time')
    sw = corr(sc, 'all_clearance', 'spike_width'); sd = corr(sc, 'all_clearance', 'decline_rate')
    md = lambda ms, d=0: f"{desc.loc[ms, 'participant_median']:.{d}f} ({desc.loc[ms, 'participant_q1']:.{d}f} to {desc.loc[ms, 'participant_q3']:.{d}f})"  # noqa: E731
    rp = lambda ms: f"{rep.loc[ms, 'pearson_r']:.2f} ({rep.loc[ms, 'r_ci_low']:.2f} to {rep.loc[ms, 'r_ci_high']:.2f})"  # noqa: E731
    lines = [
        '# Spike recovery time versus spike width', '',
        'Same spikes as the paper (rise of at least 30 mg/dL above the preceding 90-minute minimum), same eligible days,',
        'same participant rule (>=5 eligible days, >=5 resolved spikes): '
        f"{int(desc.loc['all_clearance', 'n_participants']):,} participants, "
        f"{int(desc.loc['all_clearance', 'n_spikes']):,} resolved spikes on eligible days. "
        'Estimates are % difference in the marker per SD of the measure (OLS on the log marker, HC3); '
        "'at the same average glucose' = average glucose as the single covariate. Associations between people only.", '',
        '## How the measures relate', '',
        '- Spike width = upslope half-time (from halfway up to the peak) + spike recovery time (from the peak to halfway back).',
        '  Recovery time is therefore the second half of width, not the same measure.',
        f"- Participant medians (IQR), minutes: recovery time {md('all_clearance', 1)}; width {md('spike_width', 1)}; "
        f"upslope half-time {md('upslope_half_time', 1)}; rise time {md('rise_time', 1)}; "
        f"rate of decline {md('decline_rate', 2)} mg/dL per min.",
        f'- Between people, recovery time and width are nearly interchangeable (Pearson r {rw.pearson_r:.2f}, Spearman {rw.spearman_rho:.2f}; '
        f'n {int(rw.n):,}) because most between-person variation in width comes from the downslope; across individual spikes the '
        f'agreement is weaker (Spearman {sw.spearman_rho:.2f}).',
        f'- Rate of decline is the inverse face of recovery time (r {rd.pearson_r:.2f} between people; Spearman {sd.spearman_rho:.2f} '
        f'across spikes); it also scales with spike size, because a larger rise has further to fall.',
        f'- Rise time (minimum to peak) and upslope half-time are only moderately related to recovery time (r {rr.pearson_r:.2f} and '
        f'{ru.pearson_r:.2f}).', '',
        '## Which measure carries the beyond-average-glucose signal', '',
        '| Measure | Split-half r | Fasting insulin, unadjusted | Fasting insulin, same average glucose | HOMA-IR, unadjusted | HOMA-IR, same average glucose |',
        '|---|---|---|---|---|---|']
    for ms in T.measure:
        lines.append(f"| {LABEL[ms]} | {rp(ms)} | {_ci(get('fasting_insulin', 'none', ms))} | {_ci(fi[ms])} | "
                     f"{_ci(get('homa_ir', 'none', ms))} | {_ci(hi[ms])} |")
    lines += ['',
        f"- Recovery time: fasting insulin {_ci(fi['all_clearance'])} at the same average glucose; HOMA-IR not detected "
        f"({_ci(hi['all_clearance'])}).",
        f"- Width carries the same signal, of similar or slightly larger size (not formally compared) and detected for both markers "
        f"(HOMA-IR only just): fasting insulin {_ci(fi['spike_width'])}, HOMA-IR {_ci(hi['spike_width'])}; it is also slightly more repeatable (split-half r {rp('spike_width')} vs {rp('all_clearance')}).",
        f"- Rate of decline: slower decline goes with higher fasting insulin ({_ci(fi['decline_rate'])}).",
        f"- Rise time: not detected at the same average glucose (fasting insulin {_ci(fi['rise_time'])}; HOMA-IR {_ci(hi['rise_time'])}).",
        f"- The association is strongest for the approach to the peak: upslope half-time {_ci(fi['upslope_half_time'])} (fasting insulin) and "
        f"{_ci(hi['upslope_half_time'])} (HOMA-IR). With upslope half-time also held fixed, recovery time is not detected "
        f"({_ci(ju['all_clearance'])}) while upslope half-time remains ({_ci(ju['upslope_half_time'])}). With rate of decline held "
        f"fixed, recovery time is not detected ({_ci(jd['all_clearance'])}); rate of decline combines recovery time and spike size, and "
        f"recovery time and size are each detected when entered together ({_ci(js['all_clearance'])} and {_ci(js['all_size'])}). "
        f"Holding rise time fixed leaves recovery time detected ({_ci(jr['all_clearance'])}).",
        '- Reading: the beyond-average-glucose association is with how broad a spike is around its peak, not specifically with the',
        '  downslope; the slow approach to the peak carries at least as much as the fall from it. Recovery time captures much of it',
        '  between people because it carries most of the between-person variation in width (participant SD ' +
        f"{desc.loc['all_clearance', 'participant_sd']:.1f} vs {desc.loc['upslope_half_time', 'participant_sd']:.1f} min for upslope half-time)." +
        f'- Caveats: five-minute sampling ({100 * share_up0:.1f}% of resolved spikes reach half the rise only at the peak cell, upslope half-time 0);',
        '  sensor smoothing broadens peaks; recovery is defined only for spikes that fall halfway back (censored spikes excluded);',
        '  rise time is bounded by the 90-minute look-back. Joint models with recovery time and width together are not interpretable',
        f'  (r {rw.pearson_r:.2f}; width = upslope + recovery) and are kept in spike_shape_joint_models.csv for completeness only.', '']
    return T, '\n'.join(lines)


def main():
    t0 = time.monotonic(); guards = {}
    cache = s_spike_facets.CACHE
    m_spike_facets = json.loads((F_SPIKE_FACETS / 'spike_facets.metadata.json').read_text())

    # event extraction (same inputs as spike_facets.py main)
    daily = pd.read_parquet(cache / 'spike_daily.parquet', columns=['USER_ID', 'DAY'])
    cached = pd.read_parquet(cache / 'spike_participants.parquet', columns=['USER_ID', 'spike_resolution'])
    panel = pd.read_csv(s_spike_facets.PANEL, low_memory=False)
    panel = panel[panel.no_diabetes_range == True].copy()  # noqa: E712
    people = panel.USER_ID.to_numpy()
    days = daily.merge(panel[['USER_ID', 'tz_offset_hours']], on='USER_ID'); days['DAY'] = pd.to_datetime(days.DAY)
    with duckdb.connect(str(database_path()), read_only=True) as con:
        con.execute('SET threads=6')
        E, D, bad = extract(con, days, people)
    guards['events'] = dict(person_days=int(len(D)), spikes=int(len(E)), day_mismatch=int(bad))
    if bad or len(D) != m_spike_facets['guard']['person_days'] or len(E) != m_spike_facets['guard']['spikes']:
        fail(f'event extraction differs from spike_facets.py: {guards["events"]}')
    E = add_shape(E)
    res = E[E.half_recovery.notna()]
    # upslope half-time can be 0 when the peak cell is the first cell at or above half the rise (5-minute sampling)
    g6 = dict(width_identity=bool(np.allclose(res.spike_width, res.upslope_half_time + res.all_clearance)),
              rise_ge_upslope=bool((res.rise_time >= res.upslope_half_time).all()),
              upslope_nonnegative=bool((res.upslope_half_time >= 0).all()),
              contiguous_le_first=bool((res.upslope_half_time_contiguous <= res.upslope_half_time).all()),
              drop_ge_half_rise=bool((res['drop'] >= res.rise / 2 - 1e-9).all()),
              share_resolved_upslope_zero=float((res.upslope_half_time == 0).mean()),
              share_resolved_contiguous_differs=float((res.upslope_half_time_contiguous != res.upslope_half_time).mean()))
    guards['internal_consistency'] = g6
    if not all(v for v in g6.values() if isinstance(v, bool)):
        fail(f'internal shape identities violated {g6}')

    # aggregation
    Pn = aggregate(E, D, MEASURES)
    saved = pd.read_parquet(F_SPIKE_FACETS / 'spike_facets_participants.parquet', columns=['USER_ID', 'all_clearance'])
    chk = Pn.merge(saved, on='USER_ID', suffixes=('', '_spike_facets'), validate='one_to_one').merge(cached, on='USER_ID', validate='one_to_one')
    d_spike_facets = float((chk.all_clearance - chk.all_clearance_spike_facets).abs().max()); d_spike_exposure = float((chk.all_clearance - chk.spike_resolution).abs().max())
    miss = bool((chk.all_clearance.isna() == chk.all_clearance_spike_facets.isna()).all() and (chk.all_clearance.isna() == chk.spike_resolution.isna()).all())
    guards['all_clearance_reproduced'] = dict(n_participants=int(len(chk)), n_nonmissing=int(chk.all_clearance.notna().sum()),
                                                 max_abs_diff_spike_facets=d_spike_facets, max_abs_diff_spike_exposure=d_spike_exposure, missingness_agree=miss)
    if not (len(chk) == len(Pn) == len(saved) and d_spike_facets < TOL and d_spike_exposure < TOL and miss):
        fail(f'all_clearance not reproduced {guards["all_clearance_reproduced"]}')
    for c in MEASURES:
        if not (Pn[c].isna() == Pn.all_clearance.isna()).all():
            fail(f'{c} participant set differs from recovery time')

    # split-half
    REP = halves(E, D, MEASURES)
    r_spike_facets = pd.read_csv(F_SPIKE_FACETS / 'spike_facets_repeatability.csv').set_index('facet').loc['clearance']
    rc = REP.set_index('measure').loc['all_clearance']
    g4 = dict(n=int(rc.n), pearson_r=float(rc.pearson_r), saved_n=int(r_spike_facets.n), saved_r=float(r_spike_facets.pearson_r),
              d_r=float(abs(rc.pearson_r - r_spike_facets.pearson_r)), d_rho=float(abs(rc.spearman_rho - r_spike_facets.spearman_rho)))
    guards['split_half_reproduced'] = g4
    if g4['n'] != g4['saved_n'] or g4['d_r'] > 1e-12 or g4['d_rho'] > 1e-12:
        fail(f'split-half not reproduced {g4}')

    # unadjusted_insulin_markers.py frame
    u = _load('unadjusted_for_spike_width_check', S_UNADJUSTED)
    _panel, P, _g = u.s_insulin_markers.load(None)
    new_unadjusted = u.run_families(P, u.s_insulin_markers.CGM_MEASURES, u.NEW_ADJ, markers=INS)
    sv = pd.read_csv(R_UNADJUSTED / 'cgm_measures_unadjusted.csv')
    k = ['marker', 'adjustment', 'measure']
    m = new_unadjusted.merge(sv.loc[sv.adjustment.isin(['none', 'mean_glucose_only'])], on=k, suffixes=('', '_saved'))
    worst = float(max((m[c] - m[c + '_saved']).abs().max() for c in ('estimate', 'ci_low', 'ci_high', 'p')))
    if len(m) != len(new_unadjusted) or len(m) != 2 * (12 + 11) or worst >= TOL or not (m.n == m.n_saved).all():
        fail(f'unadjusted_insulin_markers.py CGM rows not reproduced ({len(m)} rows, max |d| {worst})')
    PF = P.drop(columns=['all_clearance']).merge(Pn[['USER_ID', 'n_resolved'] + MEASURES], on='USER_ID', how='left', validate='one_to_one')
    if not np.allclose(PF.all_clearance.fillna(-1), P.all_clearance.fillna(-1), atol=TOL, rtol=0):
        fail('frame all_clearance differs from unadjusted_insulin_markers.py load()')
    A = fit_rows(u, PF, MEASURES + ['all_size', 'all_frequency'], u.NEW_ADJ)
    mm = A.merge(sv, on=k, suffixes=('', '_saved'))
    d5 = float(max((mm[c] - mm[c + '_saved']).abs().max() for c in ('estimate', 'ci_low', 'ci_high')))
    guards['markers_rows_reproduced'] = dict(rows=int(len(m)), max_abs_diff=worst, issued_rows_matching=int(len(mm)), issued_max_abs_diff=d5)
    if len(mm) != 2 * 2 * 3 or d5 >= TOL or not (mm.n == mm.n_saved).all():
        fail(f'issued recovery/size/frequency rows differ from unadjusted_insulin_markers.py {guards["markers_rows_reproduced"]}')

    # ---------- new numbers
    A['ci_includes_zero'] = (A.ci_low <= 0) & (A.ci_high >= 0)
    joint = []
    for other in ['spike_width', 'upslope_half_time', 'rise_time', 'decline_rate', 'all_size']:
        for marker, log in INS.items():
            d = PF.loc[PF[marker] > 0].copy(); d['y'] = np.log(d[marker])
            s = d.dropna(subset=['mean_glucose', 'all_clearance', other])
            for focus, cov in [('all_clearance', [other, 'mean_glucose']), (other, ['all_clearance', 'mean_glucose'])]:
                joint.append(dict(marker=marker, model=f'mean_glucose + all_clearance + {other}', measure=focus, label=LABEL[focus],
                                  **u.s_insulin_markers.fit(s, focus, cov, log=log)))
    J = pd.DataFrame(joint); J['ci_includes_zero'] = (J.ci_low <= 0) & (J.ci_high >= 0)

    cmeas = CORE4 + ['upslope_half_time', 'all_size', 'all_frequency', 'mean_glucose']
    Q = PF.dropna(subset=CORE4)
    C = []
    for i, a in enumerate(cmeas):
        for b in cmeas[i + 1:]:
            s = Q.dropna(subset=[a, b])
            C.append(dict(level='participant', a=a, b=b, n=int(len(s)), pearson_r=float(s[a].corr(s[b])),
                          spearman_rho=float(s[a].corr(s[b], method='spearman'))))
    sp = res[res.eligible][['all_clearance', 'spike_width', 'rise_time', 'decline_rate', 'upslope_half_time', 'rise']].rename(columns={'rise': 'all_size'})
    for i, a in enumerate(sp.columns):
        for b in sp.columns[i + 1:]:
            C.append(dict(level='spike_pooled', a=a, b=b, n=int(len(sp)), pearson_r=float(sp[a].corr(sp[b])),
                          spearman_rho=float(stats.spearmanr(sp[a], sp[b]).statistic)))
    C = pd.DataFrame(C)
    C['label_a'] = C.a.map(LABEL); C['label_b'] = C.b.map(LABEL)

    DESC = []
    for c in MEASURES:
        v = Q[c]; s = sp[c] if c in sp else None
        DESC.append(dict(measure=c, label=LABEL[c], n_participants=int(v.notna().sum()), participant_median=float(v.median()),
                         participant_q1=float(v.quantile(.25)), participant_q3=float(v.quantile(.75)), participant_sd=float(v.std()),
                         n_spikes=int(len(sp)) if s is not None else np.nan,
                         spike_median=float(s.median()) if s is not None else np.nan))
    DESC = pd.DataFrame(DESC)
    DESC.loc[len(DESC)] = dict(measure='resolved_spikes_per_participant', label='Resolved spikes per participant (eligible days)',
                               n_participants=int(Q.n_resolved.notna().sum()), participant_median=float(Q.n_resolved.median()),
                               participant_q1=float(Q.n_resolved.quantile(.25)), participant_q3=float(Q.n_resolved.quantile(.75)),
                               participant_sd=float(Q.n_resolved.std()), n_spikes=int(len(sp)), spike_median=np.nan)

    T, note = summary_table_and_note(A, J, REP, C, DESC, g6['share_resolved_upslope_zero'])
    OUT.mkdir(parents=True, exist_ok=True)
    files = {'spike_shape_insulin_associations.csv': A, 'spike_shape_joint_models.csv': J,
             'spike_shape_repeatability.csv': REP, 'spike_shape_correlations.csv': C, 'spike_shape_descriptives.csv': DESC,
             'spike_width_check_table.csv': T}
    (OUT / 'spike_width_check_note.md').write_text(note)
    for n_, f in files.items():
        f.to_csv(OUT / n_, index=False)
    meta = dict(analysis='spike_width_check', version=VERSION, status='passed',
                design='cross-sectional associations between people; OLS (HC3) of log marker on standardized measure; % per SD',
                definitions={k_: LABEL[k_] for k_ in MEASURES} | dict(
                    aggregation='mean over resolved spikes on eligible days; >=5 eligible days and >=5 resolved spikes (spike_facets.py clearance rule)',
                    adjustment_sets=u.NEW_ADJ, population='canonical primary cohort (unadjusted_insulin_markers.py load() frame)'),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'spike_width_check.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    pd.set_option('display.width', 220)
    print(json.dumps(guards, indent=1, default=float))
    print(A[['marker', 'adjustment', 'measure', 'n', 'estimate', 'ci_low', 'ci_high', 'p']].round(3).to_string())
    print(J[['marker', 'model', 'measure', 'n', 'estimate', 'ci_low', 'ci_high']].round(3).to_string())
    print(REP[['measure', 'n', 'pearson_r', 'r_ci_low', 'r_ci_high', 'spearman_rho']].round(3).to_string())
    print(C[['level', 'a', 'b', 'n', 'pearson_r', 'spearman_rho']].round(3).to_string())
    print(DESC.round(3).to_string())


if __name__ == '__main__':
    main()
