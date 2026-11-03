#!/usr/bin/env python3
"""Compute standard MAGE (moving-average algorithm) and compare it with the panel's turning-point proxy.

The panel column 'mage' is an operational proxy (per day, mean absolute change between successive strict turning
points of raw readings, within runs without a >10-min gap, that exceed that day's SD; averaged over dense days;
see analysis/canonical_panel/_canonical_grid.py). It is not standard MAGE. This module computes the moving-average MAGE
of Fernandes et al., J Diabetes Sci Technol 2022;16(2):576-577 (doi:10.1177/19322968211061165), as implemented in the R
package iglu 4.2.2, mage(version = 'ma') at the package defaults (short_ma = 5, long_ma = 32, direction = 'avg',
inter_gap = 45, max_gap = 180; read from the installed function's formals, not from this text), and substitutes it for
the turning-point proxy in the analysis/metabolic/unadjusted_insulin_markers.py estimator and tables. The canonical panel and
that module's own released outputs are not modified.

Input trace (one MAGE per participant, every panel participant): accepted raw readings (20-500 mg/dL) on exactly the
dense participant-local days that define the panel's other CGM measures: days of the canonical dense-day grid cache with
>= 230 accepted readings and >= 77 observed 15-minute cells, within -90..+90 days of the anchor HbA1c draw, local day =
UTC + users.TZ_OFFSET_HOURS (the panel's offset). iglu interpolates the readings to a 5-minute grid (no interpolation across
gaps > 45 min) and splits the trace at gaps > 180 min; MAGE is the segment-length-weighted mean over segments of
mean(MAGE+, MAGE-).

Stages (default: all, in order; each stage validates the previous stage's cache and is skipped when that cache verifies):
  traces    -> _cache/mage_standard/traces/chunk_NNN.parquet (participant level; restricted, local only, never released)
  iglu      -> _cache/mage_standard/iglu/chunk_NNN.parquet via analysis/clinical/mage_standard_iglu.R (Rscript of an R environment
               with iglu 4.2.2 installed in _cache/mage_standard/r_lib; pass --rscript)
  validate  -> an independent Python port of iglu's mage_ma_single (direction 'avg', defaults) on every participant (or
               --validate-n random participants, >= 200); max |port - iglu| must be < 1e-6 mg/dL with identical missingness
  estimates -> aggregate outputs in outputs/clinical/mage_standard
Structural checks performed throughout: window-day selection reproduces each panel participant's n_dense_days, n_readings,
first/last dense day, timezone offset and the proxy 'mage'; every extracted trace has exactly the expected reading/day counts;
iglu output covers every participant exactly once; the independent Python port agrees with iglu to within 1e-6 mg/dL with
identical missingness on >= 200 participants; after substituting standard MAGE into the unadjusted_insulin_markers.py
frame, every non-MAGE row keeps its n, estimate and CI exactly (only BH q values of the 12-measure family can change).

Inputs: database tables via LEVELS_DB (glucose_readings); OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv and its
dense-day grid cache (analysis/canonical_panel/build_panel.py, analysis/canonical_panel/_canonical_grid.py); the outputs
of analysis/metabolic/unadjusted_insulin_markers.py (OUTPUTS/metabolic/unadjusted_insulin_markers).
Outputs (aggregate only; no participant identifiers; under OUTPUTS/clinical/mage_standard): mage_rows_all.csv (every
MAGE row, turning-point proxy vs standard MAGE),
headline_homa_insulin_mage_standard.csv and age_sex_sensitivity_mage_standard.csv (same formats as
unadjusted_insulin_markers.py, with MAGE substituted), fdr_status_changes.csv, mage_distribution.csv,
validation_summary.json, summary.json, metadata.json. Participant-level trace caches under
OUTPUTS/clinical/_cache/mage_standard stay local and are never released.
Usage:
  python analysis/clinical/mage_standard.py --rscript <R env>/bin/Rscript [--workers 12]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
sys.dont_write_bytecode = True

VERSION = 'mage-standard'
PUB = OUTPUTS
OUT = PUB / 'clinical/mage_standard'
CACHE = PUB / 'clinical/_cache/mage_standard'          # participant level, local only (not released)
DB = database_path()
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
PANEL_META = PUB / 'canonical_panel/canonical_lab_cgm_panel.metadata.json'
GRID = PUB / 'canonical_panel/_cache/dense_day_grid.parquet'
GRID_META = GRID.with_suffix('.metadata.json')
GRID_SRC = ROOT / 'analysis/canonical_panel/_canonical_grid.py'
R_SCRIPT = ROOT / 'analysis/clinical/mage_standard_iglu.R'
IGLU_TARBALL = CACHE / 'iglu_4.2.2.tar.gz'
D_UNADJUSTED = PUB / 'metabolic/unadjusted_insulin_markers'
WINDOW_DAYS, RAW_DAILY_GATE, MIN_15M_CELLS = 90, 230, 77
CHUNK_USERS = 1000
TOL = 1e-9
PORT_TOL = 1e-6
SEED = 20261029
MAGE_DEFAULTS = dict(version='ma', short_ma=5, long_ma=32, direction='avg', inter_gap=45, max_gap=180)
GUARDS: dict = {}
EPOCH = pd.Timestamp('1970-01-01')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)) if str(Path(p).resolve()).startswith(str(ROOT)) else str(p)


def guard(name, ok, detail=None):
    """Record a structural data-integrity check and stop if it fails."""
    GUARDS[name] = dict(status='passed' if ok else 'failed', detail=detail)
    if not ok:
        raise SystemExit(f'GUARD FAILED: {name}: {detail}')


def day_no(s):
    return ((pd.to_datetime(s) - EPOCH) / pd.Timedelta(days=1)).round().astype('int64')


def write_json(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float) + '\n')
    tmp.replace(p)


# ============================================================================= stage 1: traces
def input_guards(check_db=True):
    pm = json.loads(PANEL_META.read_text())
    ph = sha(PANEL)
    gm = json.loads(GRID_META.read_text())
    gh = sha(GRID)
    guard('definition_versions_consistent', gm['daily_mage_version'] == pm['daily_mage_version'] and gm['dense_day_count_version'] == pm['dense_day_count_version'],
          dict(daily_mage_version=gm['daily_mage_version']))
    return ph, gh


def window_days():
    """Dense local days of each panel participant (the panel's own selection), with reproduction checks."""
    panel = pd.read_csv(PANEL, usecols=['USER_ID', 'anchor_date', 'tz_offset_hours', 'n_dense_days', 'n_readings',
                                        'first_dense_day', 'last_dense_day', 'mage', 'no_diabetes_range'])
    grid = pd.read_parquet(GRID, columns=['USER_ID', 'DAY', 'tz_offset_hours', 'n_raw_readings', 'n_cells', 'daily_mage'])
    dense = grid[(grid.n_cells >= MIN_15M_CELLS) & (grid.n_raw_readings >= RAW_DAILY_GATE)]
    d = dense.merge(panel[['USER_ID', 'anchor_date', 'tz_offset_hours']].rename(columns={'tz_offset_hours': 'panel_tz'}), on='USER_ID')
    off = day_no(d.DAY) - day_no(d.anchor_date)
    d = d.loc[(off >= -WINDOW_DAYS) & (off <= WINDOW_DAYS)].sort_values(['USER_ID', 'DAY']).reset_index(drop=True)
    a = d.groupby('USER_ID').agg(n=('DAY', 'size'), nr=('n_raw_readings', 'sum'), first=('DAY', 'min'), last=('DAY', 'max'),
                                 proxy=('daily_mage', lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan))
    c = panel.set_index('USER_ID').join(a, how='left')
    leg_d = (c.proxy - c.mage).abs()
    ok = dict(all_participants=bool(c.n.notna().all()), n_dense_days=bool((c.n == c.n_dense_days).all()),
              n_readings=bool((c.nr == c.n_readings).all()),
              first_last=bool(((c['first'].dt.strftime('%Y-%m-%d') == c.first_dense_day) & (c['last'].dt.strftime('%Y-%m-%d') == c.last_dense_day)).all()),
              tz=bool((d.tz_offset_hours == d.panel_tz).all()),
              proxy_mage_nan_pattern=bool((c.proxy.isna() == c.mage.isna()).all()), proxy_mage_max_abs_diff=float(np.nanmax(leg_d)))
    guard('window_days_reproduce_panel', all(v for k, v in ok.items() if k != 'proxy_mage_max_abs_diff') and ok['proxy_mage_max_abs_diff'] < TOL,
          dict(ok, participants=int(len(c)), days=int(len(d))))
    return panel, d[['USER_ID', 'DAY', 'tz_offset_hours']].rename(columns={'tz_offset_hours': 'tz'})


TRACE_QUERY = """
SELECT g.USER_ID, g.TS, g.VALUE, CAST(c.DAY AS DATE) AS DAY
FROM glucose_readings g JOIN window_days c ON g.USER_ID = c.USER_ID
 AND g.TS >= CAST(c.DAY AS TIMESTAMP) - c.tz * INTERVAL '1 hour'
 AND g.TS <  CAST(c.DAY AS TIMESTAMP) + INTERVAL '1 day' - c.tz * INTERVAL '1 hour'
WHERE g.USER_ID >= ? AND g.USER_ID <= ? AND g.VALUE BETWEEN 20 AND 500
"""


def trace_fingerprint(panel_sha, grid_sha):
    """Cache key of the extracted traces (panel, dense-day grid, extraction code)."""
    src = inspect.getsource(window_days) + TRACE_QUERY + inspect.getsource(extract_traces)
    return hashlib.sha256(json.dumps(dict(panel=panel_sha, grid=grid_sha, src=hashlib.sha256(src.encode()).hexdigest(),
                                          chunk_users=CHUNK_USERS), sort_keys=True).encode()).hexdigest()


def extract_traces(panel, days, fingerprint, threads):
    tdir = CACHE / 'traces'; tdir.mkdir(parents=True, exist_ok=True)
    man_p = tdir / 'manifest.json'
    if man_p.exists():
        man = json.loads(man_p.read_text())
        if man.get('fingerprint') == fingerprint and all((tdir / k).exists() and sha(tdir / k) == v['sha256'] for k, v in man['chunks'].items()):
            GUARDS.update(man['guards'])
            print(f"traces: verified cache ({len(man['chunks'])} chunks)", flush=True)
            return man
    exp = panel.set_index('USER_ID')[['n_dense_days', 'n_readings']]
    users = np.sort(days.USER_ID.unique())
    con = duckdb.connect(str(DB), read_only=True)
    con.execute(f'SET threads={int(threads)}'); con.execute("SET memory_limit='6GB'"); con.execute("SET TimeZone='UTC'")
    con.register('window_days', days)
    chunks, t0, n_dup, n_rows, bad, n_conf, conf_users = {}, time.time(), 0, 0, [], 0, set()
    for k, start in enumerate(range(0, len(users), CHUNK_USERS)):
        ch = users[start:start + CHUNK_USERS]
        raw = con.execute(TRACE_QUERY, [str(ch[0]), str(ch[-1])]).df()
        raw = raw.sort_values(['USER_ID', 'TS', 'VALUE'], kind='stable').reset_index(drop=True)
        cnt = raw.groupby('USER_ID').size()
        nd = raw.groupby('USER_ID').DAY.nunique()
        sel = days.loc[days.USER_ID.isin(ch)].groupby('USER_ID').size()
        e = exp.loc[ch]
        ok = (cnt.reindex(ch).fillna(-1).astype(int).to_numpy() == e.n_readings.to_numpy()) & \
             (nd.reindex(ch).fillna(-1).astype(int).to_numpy() == e.n_dense_days.to_numpy()) & (sel.reindex(ch).to_numpy() == e.n_dense_days.to_numpy())
        bad += [str(u) for u in np.asarray(ch)[~ok]]
        dup = raw.duplicated(['USER_ID', 'TS'], keep=False)
        dvals = raw.loc[dup].groupby(['USER_ID', 'TS']).VALUE.nunique()
        conflict = dvals[dvals > 1]
        n_conf += int(len(conflict)); conf_users |= set(conflict.index.get_level_values(0))
        n_dup += int(raw.duplicated(['USER_ID', 'TS']).sum()); n_rows += len(raw)
        tbl = pa.table({'USER_ID': pa.array(raw.USER_ID.astype(str).to_numpy(), pa.string()),
                        'TS': pa.array(raw.TS.dt.tz_localize('UTC'), pa.timestamp('us', tz='UTC')),
                        'VALUE': pa.array(raw.VALUE.to_numpy(float), pa.float64())})
        name = f'chunk_{k:03d}.parquet'
        pq.write_table(tbl, tdir / (name + '.tmp')); (tdir / (name + '.tmp')).replace(tdir / name)
        chunks[name] = dict(participants=int(len(ch)), rows=int(len(raw)), sha256=sha(tdir / name))
        print(f'traces: {name} {len(ch)} participants {len(raw):,} readings ({time.time() - t0:.0f} s)', flush=True)
    con.close()
    guard('traces_match_panel_counts', not bad and n_rows == int(exp.n_readings.sum()),
          dict(participants=int(len(users)), readings=int(n_rows), mismatched=len(bad), repeated_timestamp_rows=int(n_dup)))
    # iglu (CGMS2DayByDay) keeps the last of repeated timestamps; rows are written in (TS, VALUE) order, so where repeated
    # timestamps carry different values the larger value is kept, identically in iglu and in the Python port
    GUARDS['repeated_timestamps_recorded'] = dict(status='passed', detail=dict(
        repeated_timestamp_rows=int(n_dup), timestamps_with_differing_values=int(n_conf), participants_with_differing_values=len(conf_users),
        rule='rows ordered by (TS, VALUE); iglu keeps the last of repeated timestamps, i.e. the larger value'))
    man = dict(fingerprint=fingerprint, chunks=chunks, guards={k: GUARDS[k] for k in GUARDS if k in ('traces_match_panel_counts', 'repeated_timestamps_recorded')},
               definition='accepted readings 20-500 mg/dL on the panel dense local days (-90..+90 d of the anchor); TS UTC; sorted by USER_ID, TS, VALUE')
    write_json(man_p, man)
    return man


# ============================================================================= stage 2: iglu
def run_iglu(man, rscript, workers):
    idir = CACHE / 'iglu'; idir.mkdir(parents=True, exist_ok=True)
    run_p = idir / 'run.json'
    run = json.loads(run_p.read_text()) if run_p.exists() else {}
    rsha = sha(R_SCRIPT)
    todo = [k for k, v in man['chunks'].items()
            if not ((idir / k).exists() and run.get('chunks', {}).get(k, {}).get('trace_sha256') == v['sha256']
                    and run['chunks'][k].get('sha256') == sha(idir / k) and run.get('r_script_sha256') == rsha)]
    if todo:
        if not rscript:
            raise SystemExit('iglu results missing or stale; pass --rscript <Rscript of an R environment with arrow, zoo, dplyr>')
        print(f'iglu: computing {len(todo)} chunks with {workers} workers', flush=True)
        t0 = time.time()
        subprocess.run([rscript, str(R_SCRIPT), str(CACHE), str(workers)] + todo, check=True, cwd=ROOT)
        run.setdefault('chunks', {})
        run['runtime_seconds_last'] = round(time.time() - t0, 1)
    session = json.loads((idir / 'session.json').read_text())
    guard('iglu_version_and_defaults', session['iglu'] == '4.2.2' and all(session['defaults'][k] == v for k, v in MAGE_DEFAULTS.items()),
          dict(iglu=session['iglu'], R=session['R'], defaults=session['defaults']))
    frames = []
    for k, v in man['chunks'].items():
        guard_ok = sha(CACHE / 'traces' / k) == v['sha256']
        if not guard_ok:
            guard('trace_chunks_unchanged', False, k)
        f = pd.read_parquet(idir / k)
        run.setdefault('chunks', {})[k] = dict(trace_sha256=v['sha256'], sha256=sha(idir / k), participants=int(len(f)))
        frames.append(f)
    run.update(r_script_sha256=rsha, session=session)
    write_json(run_p, run)
    R = pd.concat(frames, ignore_index=True)
    GUARDS['trace_chunks_unchanged'] = dict(status='passed', detail=len(man['chunks']))
    return R, session


# ============================================================================= stage 3: Python port of iglu mage_ma_single
def _seqsum(x):
    """Sequential double summation (R's LDOUBLE accumulator is a double on this arm64 build)."""
    return float(np.cumsum(x)[-1]) if len(x) else 0.0


def r_mean(x):
    """R mean.default: sum/n, then one correction pass (summary.c)."""
    n = len(x)
    if n == 0:
        return math.nan
    s = _seqsum(x) / n
    if math.isfinite(s):
        s = s + _seqsum(x - s) / n
    return s


def r_sd(x):
    """R sd(): cov.c two-pass with the corrected mean, denominator n - 1."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return math.nan
    m = r_mean(x)
    d = x - m
    return math.sqrt(_seqsum(d * d) / (n - 1))


def r_rollmean_right(g, w):
    """zoo::rollapply(g, w, mean, align='right', fill=NA, na.rm=TRUE): R mean of the non-NA values of each window."""
    n = len(g)
    out = np.full(n, np.nan)
    if n < w:
        return out
    ok = ~np.isnan(g)
    x0 = np.where(ok, g, 0.0)
    m = n - w + 1
    s = x0[0:m].copy(); c = ok[0:m].astype(np.int64)
    for k in range(1, w):
        s = s + x0[k:k + m]; c = c + ok[k:k + m]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s / c
        t = np.zeros(m)
        for k in range(w):
            t = t + np.where(ok[k:k + m], g[k:k + m] - mean, 0.0)   # adding 0.0 leaves a double sum unchanged
        res = np.where(np.isfinite(mean), mean + t / c, mean)
    out[w - 1:] = res
    return out


def cgms_grid(t, g, dt0=5, inter_gap=45):
    """iglu CGMS2DayByDay(dt0 = 5): linear interpolation (R approx) to a UTC grid from 00:05 of the first day; NA across gaps."""
    o = np.argsort(t, kind='stable'); t, g = t[o], g[o]
    td = np.diff(t) / 60.0
    rep = np.flatnonzero(td == 0)
    if len(rep):                                            # repeated times: keep the last one
        t, g = np.delete(t, rep), np.delete(g, rep); td = np.diff(t) / 60.0
    ndays = math.ceil((t[-1] - t[0]) / 86400.0 + 1)
    minD = math.floor(t[0] / 86400.0) * 86400.0
    xo = minD + 60.0 * dt0 * np.arange(1, ndays * 1440 // dt0 + 1)
    y = np.full(len(xo), np.nan)
    inr = (xo >= t[0]) & (xo <= t[-1])
    v = xo[inr]
    i = np.searchsorted(t, v, side='right') - 1
    i = np.clip(i, 0, len(t) - 2) if len(t) > 1 else np.zeros_like(i)
    if len(t) > 1:
        j = i + 1
        yi = g[i] + (g[j] - g[i]) * ((v - t[i]) / (t[j] - t[i]))
        yi = np.where(v == t[j], g[j], np.where(v == t[i], g[i], yi))
    else:
        yi = np.where(v == t[0], g[0], np.nan)
    y[inr] = yi
    for gs in np.flatnonzero(td > inter_gap):
        y[(xo > t[gs]) & (xo < t[gs + 1])] = np.nan
    return xo, y


def _mage_atomic(gl, short_ma, long_ma):
    """mage_atomic() of iglu mage_ma_single.R on one segment; returns (mage_plus or nan, mage_minus or nan) or None."""
    n = len(gl)
    if np.isnan(gl).all() or n < 7 or n < long_ma:
        return None
    sdv = r_sd(gl)
    if not sdv >= 1:          # also NA (fewer than two values): iglu's 'if (sd < 1)' cannot be reached with NA here
        if math.isnan(sdv):
            raise ValueError('segment with < 2 values reached the sd check')
        return None
    ms = r_rollmean_right(gl, short_ma); ml = r_rollmean_right(gl, long_ma)
    ms[:short_ma] = ms[short_ma - 1]; ml[:long_ma] = ml[long_ma - 1]
    D = ms - ml
    G = gl
    if math.isnan(D[0]) or math.isnan(D[-1]):
        raise ValueError('undefined moving-average difference at a segment end')
    ids, types = [0], [1 if D[0] > 0 else 0]
    Dl, Gn = D.tolist(), np.isnan(G).tolist(); Dn = np.isnan(D).tolist()
    for i in range(1, n):
        if not Gn[i] and not Gn[i - 1] and not Dn[i] and not Dn[i - 1]:
            if Dl[i] * Dl[i - 1] < 0:
                ids.append(i); types.append(0 if Dl[i] < Dl[i - 1] else 1)
            elif Dl[i] * Dl[ids[-1]] < 0:
                prev = Dl[ids[-1]]
                ids.append(i); types.append(0 if Dl[i] < prev else 1)
    ids.append(n - 1); types.append(1 if Dl[n - 1] > 0 else 0)
    ne = len(ids) - 1
    minmax, indexes = np.empty(ne), np.empty(ne, dtype=np.int64)
    for i in range(ne):
        s1 = ids[0] if i == 0 else int(indexes[i - 1]); s2 = ids[i + 1]
        w = G[s1:s2 + 1]
        if types[i] == 0:
            k = int(np.nanargmin(w))
        else:
            k = int(np.nanargmax(w))
        minmax[i] = w[k]; indexes[i] = k + s1
    N = ne
    mm = minmax
    plus, minus, plus_pairs, minus_pairs = [], [], [], []
    j = prev_j = 0
    while j <= N - 1:
        delta = mm[j] - mm[prev_j:j + 1]
        i = int(np.argmax(delta)) + prev_j; max_v = delta[i - prev_j]
        if max_v >= sdv:
            for k in range(j, N):
                if mm[k] > mm[j]:
                    j = k
                if (mm[k] - mm[j]) < -sdv or k == N - 1:
                    plus.append(mm[j] - mm[i]); plus_pairs.append((i, j))
                    prev_j = k; j = k
                    break
        else:
            j += 1
    j = prev_j = 0
    while j <= N - 1:
        delta = mm[j] - mm[prev_j:j + 1]
        i = int(np.argmin(delta)) + prev_j; min_v = delta[i - prev_j]
        if min_v <= -sdv:
            for k in range(j, N):
                if mm[k] < mm[j]:
                    j = k
                if (mm[k] - mm[j]) > sdv or k == N - 1:
                    minus.append(mm[j] - mm[i]); minus_pairs.append((i, j, k))
                    prev_j = j; j = k
                    break
        else:
            j += 1
    if not plus and not minus:
        return None
    mp_ = r_mean(np.array(plus)) if plus else math.nan
    mm_ = abs(r_mean(np.array(minus))) if minus else math.nan
    return mp_, mm_


def mage_ma_port(ts_seconds, gl, short_ma=5, long_ma=32, inter_gap=45, max_gap=180, dt0=5):
    """iglu 4.2.2 mage(version='ma', direction='avg', return_type='num') for one participant."""
    xo, y = cgms_grid(np.asarray(ts_seconds, float), np.asarray(gl, float), dt0, inter_gap)
    nn = np.flatnonzero(~np.isnan(y))
    y = y[nn[0]:nn[-1] + 1]
    n = len(y)
    if n < 7 or n < long_ma:
        return math.nan
    gap = np.isnan(y).astype(np.int8)
    b = np.flatnonzero(np.diff(gap)) + 1
    starts = np.r_[0, b]; ends = np.r_[b - 1, n - 1]; vals = gap[starts]
    q = [(s, e) for s, e, v in zip(starts, ends, vals) if v == 1 and (e - s + 1) * dt0 > max_gap]
    if not q:
        segs = [(0, n - 1)]
    else:
        segs = []
        for i, (gs, ge) in enumerate(q):
            if i == 0 and gs > 0:
                segs.append((0, gs - 1))
            segs.append((gs, ge))
            if ge < n - 1:
                segs.append((ge + 1, n - 1 if i == len(q) - 1 else q[i + 1][0] - 1))
    rows = []
    for s, e in segs:
        r = _mage_atomic(y[s:e + 1], short_ma, long_ma)
        if r is None:
            continue
        dur = (e - s) * dt0 * 60.0
        for v in r:
            if not math.isnan(v):
                rows.append((v, dur))
    if not rows:
        return math.nan
    z = np.array([d for _, d in rows]); m = np.array([v for v, _ in rows])
    zz = z.min()
    u = 1.0 if zz < 60 else 60.0 if zz < 3600 else 3600.0 if zz < 86400 else 86400.0   # difftime(units = 'auto')
    h = z / u
    w = h / _seqsum(h)
    return _seqsum(m * w)


def _port_chunk(args):
    path, users = args
    t = pd.read_parquet(path)
    t = t.loc[t.USER_ID.isin(users)]
    out = []
    for u, s in t.groupby('USER_ID', sort=False):
        # arrow (R) converts timestamp[us] to POSIXct as double(us) / 1e6; the same exact operation here
        ts = s.TS.dt.tz_convert('UTC').dt.as_unit('us').astype('int64').to_numpy() / 1e6
        out.append((u, mage_ma_port(ts, s.VALUE.to_numpy(float), **{k: MAGE_DEFAULTS[k] for k in ('short_ma', 'long_ma', 'inter_gap', 'max_gap')})))
    return out


def validate_port(man, R, n_validate, workers):
    users = R.USER_ID.astype(str).to_numpy()
    if n_validate and n_validate < len(users):
        users = np.sort(np.random.default_rng(SEED).choice(users, size=n_validate, replace=False))
    sel = set(users)
    tasks = []
    for k in man['chunks']:
        part = pd.read_parquet(CACHE / 'traces' / k, columns=['USER_ID']).USER_ID.unique()
        us = [u for u in part if u in sel]
        for i in range(0, len(us), 100):
            tasks.append((str(CACHE / 'traces' / k), us[i:i + 100]))
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        res = [x for chunk in pool.imap_unordered(_port_chunk, tasks) for x in chunk]
    P = pd.DataFrame(res, columns=['USER_ID', 'mage_port']).merge(R[['USER_ID', 'mage_iglu']], on='USER_ID', how='left', validate='one_to_one')
    both = P.mage_port.notna() & P.mage_iglu.notna()
    d = (P.mage_port - P.mage_iglu).abs()
    det = dict(participants_compared=int(len(P)), both_defined=int(both.sum()), na_pattern_equal=bool((P.mage_port.isna() == P.mage_iglu.isna()).all()),
               n_na_iglu=int(P.mage_iglu.isna().sum()), max_abs_diff_mg_dl=float(d[both].max()), mean_abs_diff_mg_dl=float(d[both].mean()),
               n_exactly_equal=int((d[both] == 0).sum()), tolerance_mg_dl=PORT_TOL, sampled=bool(n_validate and n_validate < len(R)),
               seed=SEED if n_validate else None, runtime_seconds=round(time.time() - t0, 1))
    guard('python_port_equals_iglu', len(P) >= 200 and det['na_pattern_equal'] and det['max_abs_diff_mg_dl'] < PORT_TOL, det)
    return det


# ============================================================================= stage 4: estimates (estimator and frame of unadjusted_insulin_markers.py)
def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


NUM = ['estimate', 'ci_low', 'ci_high', 'p', 'q_fdr_family']


def producer_tables(m_unadjusted, s_insulin_markers, P):
    """CGM rows of cgm_measures_unadjusted.csv, age_sex_sensitivity.csv and headline_homa_insulin.csv, built with producer
    174's own functions exactly as its main() builds them (meal families are not affected by MAGE and are not rebuilt)."""
    CGM = m_unadjusted.issue(P, s_insulin_markers.CGM_MEASURES)
    CGM.insert(0, 'family', 'cgm')
    A_INSULIN_MARKERS = m_unadjusted.run_families(P, s_insulin_markers.CGM_MEASURES, s_insulin_markers.ADJ)
    T = pd.concat([CGM, A_INSULIN_MARKERS.assign(family='cgm').loc[lambda x: x.adjustment.isin(['age_sex', 'mean_glucose'])]],
                  ignore_index=True).drop(columns='family')
    S = []
    for a, b, lab in [('none', 'age_sex', 'unadjusted vs age+sex'), ('mean_glucose_only', 'mean_glucose', 'mean glucose vs mean glucose+age+sex')]:
        m = m_unadjusted.pair(T, a, b); m.insert(0, 'comparison', lab); m.insert(0, 'family', 'cgm'); m_unadjusted.same_n(m, f'age/sex pair {a}'); S.append(m)
    AS = pd.concat(S, ignore_index=True)
    AS['abs_change_points'] = AS.change_points.abs()
    AS['in_paper_set'] = AS.measure.isin(m_unadjusted.PAPER_CGM)
    hl = CGM.loc[CGM.adjustment.isin(['none', 'mean_glucose_only']) & CGM.marker.isin(['homa_ir', 'fasting_insulin']) & CGM.measure.isin(m_unadjusted.PAPER_CGM)]
    cols = ['n', 'estimate', 'ci_low', 'ci_high', 'p', 'q_fdr_family']
    HL = hl.loc[hl.marker == 'homa_ir', ['family', 'adjustment', 'measure'] + cols].merge(
        hl.loc[hl.marker == 'fasting_insulin', ['family', 'adjustment', 'measure'] + cols], on=['family', 'adjustment', 'measure'],
        suffixes=('_homa_ir', '_fasting_insulin'), validate='one_to_one')
    order = {m: i for i, m in enumerate(m_unadjusted.PAPER_CGM + m_unadjusted.PAPER_MEAL)}
    HL = HL.sort_values(['adjustment', 'family', 'measure'], key=lambda s: s.map(order) if s.name == 'measure' else s).reset_index(drop=True)
    HL['ci_includes_zero_homa_ir'] = (HL.ci_low_homa_ir <= 0) & (HL.ci_high_homa_ir >= 0)
    HL['ci_includes_zero_fasting_insulin'] = (HL.ci_low_fasting_insulin <= 0) & (HL.ci_high_fasting_insulin >= 0)
    return CGM, AS, HL, A_INSULIN_MARKERS


def compare(new, saved, keys, label):
    """Row sets equal; integer columns exact; numeric columns within TOL; string/bool columns equal. Returns max |diff|."""
    if len(new) != len(saved):
        guard(f'{label}_rows_match', False, f'{len(new)} rows vs {len(saved)}')
    m = new.merge(saved, on=keys, suffixes=('_new', '_saved'), validate='one_to_one')
    if len(m) != len(saved):
        guard(f'{label}_rows_match', False, 'keys do not align')
    worst = 0.0
    for c in [c for c in saved.columns if c not in keys]:
        a, b = m[c + '_new'], m[c + '_saved']
        if c.startswith('n_') or c == 'n':
            if not (a.astype(int) == b.astype(int)).all():
                guard(f'{label}_rows_match', False, f'{c} differs')
        elif pd.api.types.is_bool_dtype(b) or b.dtype == object or pd.api.types.is_string_dtype(b):
            if not (a.astype(str) == b.astype(str)).all():
                guard(f'{label}_rows_match', False, f'{c} differs')
        else:
            a, b = a.astype(float), b.astype(float)
            if not (a.isna() == b.isna()).all():
                guard(f'{label}_rows_match', False, f'{c} missingness differs')
            worst = max(worst, float(np.nanmax(np.abs(a - b))) if a.notna().any() else 0.0)
    return worst


def pick(df, **kw):
    r = df
    for k, v in kw.items():
        r = r.loc[r[k] == v]
    if len(r) != 1:
        raise SystemExit(f'expected one row for {kw}, got {len(r)}')
    return r.iloc[0]


def estimates(panel, R, session, val, man, t0):
    m_unadjusted = _load('unadjusted_for_mage_standard', 'analysis/metabolic/unadjusted_insulin_markers.py')
    s_insulin_markers = m_unadjusted.s_insulin_markers
    _panel, P, _ = s_insulin_markers.load(None)
    guard('markers_load_consistent', len(P) == int((panel.no_diabetes_range == True).sum()), len(P))  # noqa: E712
    # ---- the turning-point MAGE proxy reproduces every CGM row saved by analysis/metabolic/unadjusted_insulin_markers.py
    CGM0, AS0, HL0, A0 = producer_tables(m_unadjusted, s_insulin_markers, P)
    savedC = pd.read_csv(D_UNADJUSTED / 'cgm_measures_unadjusted.csv')
    savedA = pd.read_csv(D_UNADJUSTED / 'age_sex_sensitivity.csv'); savedA = savedA.loc[savedA.family == 'cgm'].reset_index(drop=True)
    savedH = pd.read_csv(D_UNADJUSTED / 'headline_homa_insulin.csv'); savedHc = savedH.loc[savedH.family == 'cgm'].reset_index(drop=True)
    g6 = dict(cgm_measures_unadjusted=compare(CGM0, savedC, ['family', 'marker', 'adjustment', 'measure'], 'cgm_measures_unadjusted'),
              age_sex_sensitivity_cgm=compare(AS0, savedA, ['family', 'comparison', 'marker', 'measure'], 'age_sex_sensitivity'),
              headline_cgm=compare(HL0, savedHc, ['family', 'adjustment', 'measure'], 'headline_homa_insulin'),
              cgm_measures_by_marker_insulin_markers=compare(A0, pd.read_csv(m_unadjusted.D_INSULIN_MARKERS / 'cgm_measures_by_marker.csv'), ['marker', 'adjustment', 'measure'], 'insulin_markers_cgm_measures_by_marker'))
    guard('proxy_reproduces_markers_rows', all(v < TOL for v in g6.values()), dict(max_abs_diff=g6, rows=dict(cgm=len(CGM0), age_sex=len(AS0), headline=len(HL0), a_insulin_markers=len(A0))))
    n_ad = {mk: int(pick(HL0, adjustment='none', measure='mean_glucose')[f'n_{mk}']) for mk in ('fasting_insulin', 'homa_ir')}
    # ---- substitute the standard MAGE (same participants, same frame)
    new = R.set_index('USER_ID').mage_iglu
    P2 = P.copy()
    P2['mage_proxy'] = P2['mage']
    P2['mage'] = P2.USER_ID.map(new).astype(float)
    guard('standard_mage_for_every_primary_participant', P2.mage.notna().sum() == P.mage.notna().sum() and P2.USER_ID.isin(new.index).all(),
          dict(primary=int(len(P2)), standard_defined=int(P2.mage.notna().sum()), proxy_defined=int(P.mage.notna().sum())))
    CGM1, AS1, HL1, A1 = producer_tables(m_unadjusted, s_insulin_markers, P2.drop(columns='mage_proxy'))
    k = ['marker', 'adjustment', 'measure']
    allold = pd.concat([CGM0.drop(columns='family'), A0], ignore_index=True)
    allnew = pd.concat([CGM1.drop(columns='family'), A1], ignore_index=True)
    mm = allold.merge(allnew, on=k, suffixes=('_old', '_new'), validate='one_to_one')
    nm = mm.loc[mm.measure != 'mage']
    g7 = dict(non_mage_rows=int(len(nm)), max_abs_diff_estimate_ci_p=float(max((nm[f'{c}_old'] - nm[f'{c}_new']).abs().max() for c in ('estimate', 'ci_low', 'ci_high', 'p'))),
              n_equal=bool((nm.n_old == nm.n_new).all()), mage_n_equal=bool((mm.loc[mm.measure == 'mage', 'n_old'] == mm.loc[mm.measure == 'mage', 'n_new']).all()))
    guard('only_mage_rows_change', g7['max_abs_diff_estimate_ci_p'] == 0 and g7['n_equal'] and g7['mage_n_equal'], g7)
    # FDR (BH within marker x adjustment set over the 12 CGM measures, as in insulin_markers.py): status changes of other rows
    mm['sig_old'], mm['sig_new'] = mm.q_fdr_family_old < 0.05, mm.q_fdr_family_new < 0.05
    fdr = mm.loc[:, k + ['n_old', 'q_fdr_family_old', 'q_fdr_family_new', 'sig_old', 'sig_new']].copy()
    fdr['status_changed'] = fdr.sig_old != fdr.sig_new
    fdr['ci_excludes_0_old'] = (mm.ci_low_old > 0) | (mm.ci_high_old < 0)
    fdr['ci_excludes_0_new'] = (mm.ci_low_new > 0) | (mm.ci_high_new < 0)
    # ---- outputs (aggregate)
    allrows = mm.loc[mm.measure == 'mage', k + [f'{c}_{t}' for t in ('old', 'new') for c in ['n'] + NUM]].copy()
    allrows.insert(0, 'source', np.where(allrows.adjustment.isin(list(m_unadjusted.NEW_ADJ) + list(m_unadjusted.BMI_ADJ)), 'cgm_measures_unadjusted.csv (unadjusted_insulin_markers)', 'cgm_measures_by_marker.csv (insulin_markers sets, unadjusted_insulin_markers age_sex rows)'))
    # unadjusted_insulin_markers.py-format files with MAGE substituted (meal rows taken unchanged from that module's outputs)
    H_out = pd.concat([HL1, savedH.loc[savedH.family == 'meal']], ignore_index=True)[savedH.columns]
    A_full = pd.read_csv(D_UNADJUSTED / 'age_sex_sensitivity.csv')
    A_out = pd.concat([AS1, A_full.loc[A_full.family == 'meal']], ignore_index=True)[A_full.columns]
    # distribution and agreement (participant values stay in memory)
    full = panel[['USER_ID', 'no_diabetes_range', 'mage']].rename(columns={'mage': 'proxy'}).merge(R[['USER_ID', 'mage_iglu']], on='USER_ID', validate='one_to_one')
    full = full.merge(pd.read_csv(PANEL, usecols=['USER_ID', 'sd_glucose', 'mean_glucose']), on='USER_ID', validate='one_to_one')
    dist, corr = [], {}
    for pop, d in (('primary cohort (no observed diabetes-range laboratory value; = Supplementary Table 5 lab-linked panel)', full.loc[full.no_diabetes_range == True]),  # noqa: E712
                   ('all canonical panel participants', full)):
        for col, lab in (('mage_iglu', 'standard MAGE (iglu ma, defaults)'), ('proxy', 'turning-point proxy')):
            s = d[col].dropna()
            dist.append(dict(population=pop, measure=lab, n=int(len(s)), median=float(s.median()), q1=float(s.quantile(.25)), q3=float(s.quantile(.75)),
                             mean=float(s.mean()), sd=float(s.std()), p05=float(s.quantile(.05)), p95=float(s.quantile(.95)),
                             table1_format=f"{s.median():.1f} ({s.quantile(.25):.1f} to {s.quantile(.75):.1f})"))
        b = d.dropna(subset=['mage_iglu', 'proxy'])
        corr[pop] = dict(n=int(len(b)), spearman=float(stats.spearmanr(b.mage_iglu, b.proxy)[0]), pearson=float(stats.pearsonr(b.mage_iglu, b.proxy)[0]),
                         spearman_new_vs_sd=float(stats.spearmanr(b.mage_iglu, b.sd_glucose)[0]), spearman_proxy_vs_sd=float(stats.spearmanr(b.proxy, b.sd_glucose)[0]),
                         spearman_new_vs_mean_glucose=float(stats.spearmanr(b.mage_iglu, b.mean_glucose)[0]),
                         spearman_proxy_vs_mean_glucose=float(stats.spearmanr(b.proxy, b.mean_glucose)[0]),
                         median_ratio_standard_to_proxy=float((b.mage_iglu / b.proxy).median()))
    DIST = pd.DataFrame(dist)
    other_fdr = fdr.loc[fdr.measure != 'mage']
    OUT.mkdir(parents=True, exist_ok=True)
    files = {'mage_rows_all.csv': allrows, 'headline_homa_insulin_mage_standard.csv': H_out,
             'age_sex_sensitivity_mage_standard.csv': A_out, 'fdr_status_changes.csv': fdr.loc[fdr.status_changed | (fdr.measure == 'mage')],
             'mage_distribution.csv': DIST}
    for nme, f in files.items():
        f.to_csv(OUT / nme, index=False)
    write_json(OUT / 'validation_summary.json', dict(
        reference=dict(package='iglu', version=session['iglu'], R=session['R'], call=session['call'], defaults=session['defaults'],
                       source_tarball=rel(IGLU_TARBALL),
                       citation='Fernandes NJ, Nguyen N, Chun E, Punjabi NM, Gaynanova I. Open-source algorithm to calculate mean amplitude of glycemic '
                                'excursions using short and long moving averages. J Diabetes Sci Technol. 2022;16(2):576-577. doi:10.1177/19322968211061165'),
        iglu_run=dict(participants=int(len(R)), na=int(R.mage_iglu.isna().sum()), cpu_seconds_total=round(float(R.seconds.sum()), 1)),
        python_port=dict(val, implementation='mage_ma_port(): line-by-line port of iglu 4.2.2 CGMS2DayByDay(dt0=5) + '
                                               'mage_ma_single(direction="avg"), with R mean/sd/sum summation order'),
        input_trace=dict(definition=man['definition'], chunks=len(man['chunks']), readings=int(sum(v['rows'] for v in man['chunks'].values())))))
    summary = dict(design='cross-sectional associations; estimators and frames of analysis/metabolic/unadjusted_insulin_markers.py; standard MAGE substituted for the turning-point proxy',
                   algorithm=dict(name='moving-average MAGE (Fernandes et al. 2022)', implementation=f"iglu {session['iglu']} mage(version='ma')",
                                  parameters=session['defaults']),
                   samples=dict(panel=int(len(panel)), primary=int(len(P)), fasting_insulin=n_ad['fasting_insulin'], homa_ir=n_ad['homa_ir']),
                   distribution=dist, old_new_correlation=corr,
                   fdr=dict(family='BH within marker x adjustment set over the 12 CGM measures (family() of analysis/metabolic/insulin_markers.py)',
                            other_rows_status_changed=other_fdr.loc[other_fdr.status_changed, k + ['q_fdr_family_old', 'q_fdr_family_new']].to_dict('records'),
                            other_rows_compared=int(len(other_fdr)), figures_use_fdr_for_marker_fill=False,
                            note='Figs. 3d and 4b fill markers by whether the 95% CI excludes 0, not by q.'),
                   guards_passed=True)
    write_json(OUT / 'summary.json', summary)
    meta = dict(analysis='mage_standard', version=VERSION, status='passed', guards=GUARDS, seed=SEED,
                definitions=dict(mage=f"iglu {session['iglu']} mage(version='ma') defaults {session['defaults']}; one value per participant over the dense-day trace",
                                 trace=man['definition'], estimator='analysis/metabolic/unadjusted_insulin_markers.py fit()/family() via issue()/pair(); OLS (HC3) of log marker on standardized measure',
                                 populations='canonical primary cohort (no_diabetes_range == True) as loaded by that module\'s load()'),
                individual_participant_data=False,
                runtime_seconds=round(time.time() - t0, 1))
    write_json(OUT / 'metadata.json', meta)
    print(DIST.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all', help="comma list of traces, iglu, validate, estimates (default: all)")
    ap.add_argument('--rscript', default=os.environ.get('MAGE_RSCRIPT') or shutil.which('Rscript'),
                    help='Rscript of an R environment with arrow, zoo, dplyr (iglu itself is loaded from _cache/mage_standard/r_lib)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--threads', type=int, default=6, help='DuckDB threads for the trace extraction')
    ap.add_argument('--validate-n', type=int, default=0, help='participants for the port validation (0 = all; >= 200)')
    a = ap.parse_args(argv)
    stages = {s.strip() for s in a.stage.split(',')} if a.stage != 'all' else {'traces', 'iglu', 'validate', 'estimates'}
    if a.validate_n and a.validate_n < 200:
        ap.error('--validate-n must be 0 (all) or >= 200')
    t0 = time.time()
    panel_sha, grid_sha = input_guards(check_db=True)
    panel, days = window_days()
    man = extract_traces(panel, days, trace_fingerprint(panel_sha, grid_sha), a.threads)
    if not stages & {'iglu', 'validate', 'estimates'}:
        return
    R, session = run_iglu(man, a.rscript, a.workers)
    guard('iglu_covers_every_participant', len(R) == len(panel) and R.USER_ID.is_unique and set(R.USER_ID) == set(panel.USER_ID)
          and bool((R.set_index('USER_ID').n_input_rows.reindex(panel.USER_ID).to_numpy() == panel.n_readings.to_numpy()).all()),
          dict(participants=int(len(R)), na=int(R.mage_iglu.isna().sum())))
    if not stages & {'validate', 'estimates'}:
        return
    val = validate_port(man, R, a.validate_n, a.workers)
    if 'estimates' in stages:
        estimates(panel, R, session, val, man, t0)


if __name__ == '__main__':
    main()
