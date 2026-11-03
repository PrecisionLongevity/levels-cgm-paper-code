"""Operational relative-rise spike detector and rank-association helpers.

Defines a declared, deterministic relative-rise spike definition (not the
unavailable PROGRESS detector) applied to observed, deduplicated five-minute
local-day glucose cells, plus the daily/user aggregation and Spearman
rank-correlation (simple and age/sex-partial) machinery used to relate spike
metrics to exposures such as logged carbohydrate and steps.

Eligibility and definitions:
  - Eligible search centers require 18 preceding observed cells and an
    observed following cell; days need >=192 eligible search centers to
    count zero-spike days.
  - A spike is a local maximum with a rise of >=30 mg/dL over the preceding
    90-minute minimum; peaks <30 minutes apart are resolved by larger
    amplitude, then earlier time.
  - Recovery is the first observed time the glucose returns to <= half the
    rise amplitude, before the next retained peak, a gap, local-day end, or
    6 hours; otherwise the event is censored (never imputed as a 0-minute
    recovery).
  - User-level daily_n_spikes, max_spike_rel, and spike_resolution require
    >=5 eligible/resolved days; nocturnal_min requires >=5 nights with
    >=68 cells before midpoint.
  - rank_association computes Spearman rho both unadjusted and partialled
    on age (rank) and sex via OLS-residualized ranks, with a t-test on
    n-rank(Z)-1 degrees of freedom; pair_result requires >=100 complete
    pairs and adds bootstrap (500 draws) confidence intervals.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats

VERSION = 'observed-relative-rise-spikes'
LOOKBACK_CELLS = 18
RISE_MG_DL = 30.
PEAK_DISTANCE_CELLS = 6
RECOVERY_CELLS = 72
MIN_SEARCH_CELLS = 192
MIN_NIGHT_CELLS = 68
MIN_DAYS = 5
MIN_RECOVERIES = 5
MIN_PAIR_N = 100


def daily_metrics(bins, glucose):
    """Five-minute local-day bins; no interpolation or cross-day/gap events.

    Eligible search centers have 18 preceding observed cells and an observed
    following cell. Retain local maxima with a >=30 rise over the preceding
    90-minute minimum. Resolve peaks <30 minutes apart by larger amplitude,
    then earlier time. Flat-topped maxima use scipy's middle-cell convention.
    Count zero events only on days with >=192 eligible search centers. Recovery
    is first observed <=half rise before the next retained peak, gap, local-day
    end, or after 6 hours. Censored events never become zero-minute recoveries.
    """
    bins = np.asarray(bins, dtype=np.int64)
    g = np.asarray(glucose, dtype=float)
    if (bins.ndim != 1 or g.shape != bins.shape or not len(bins)
            or (np.diff(bins) <= 0).any() or (bins < 0).any() or (bins >= 288).any()
            or not np.isfinite(g).all() or not ((g >= 20) & (g <= 500)).all()):
        raise ValueError('Expected sorted unique observed local-day cells in 20–500 mg/dL')
    runs = np.split(np.arange(len(bins)), np.flatnonzero(np.diff(bins) != 1)+1)
    amplitudes, recoveries = [], []
    n_search = 0
    for run in runs:
        x = g[run]
        n_search += max(0, len(x)-LOOKBACK_CELLS-1)
        peaks, _ = signal.find_peaks(x)
        candidates = [(int(p), float(x[p]-min(x[p-LOOKBACK_CELLS:p])))
                      for p in peaks if p >= LOOKBACK_CELLS]
        candidates = [(p, a) for p, a in candidates if a >= RISE_MG_DL]
        kept = []
        for p, a in sorted(candidates, key=lambda item: (-item[1], item[0])):
            if all(abs(p-q) >= PEAK_DISTANCE_CELLS for q, _ in kept):
                kept.append((p, a))
        kept.sort()
        for i, (p, a) in enumerate(kept):
            amplitudes.append(a)
            next_peak = kept[i+1][0] if i+1 < len(kept) else len(x)
            stop = min(p+RECOVERY_CELLS+1, next_peak, len(x))
            reached = np.flatnonzero(x[p+1:stop] <= x[p]-a/2)
            if len(reached):
                recoveries.append(float((reached[0]+1)*5))
    eligible = n_search >= MIN_SEARCH_CELLS
    night = g[bins < 84]
    return {
        'n_cells': len(g), 'n_gt150': int((g > 150).sum()),
        'n_search_cells': n_search, 'peak_day_eligible': eligible,
        'daily_n_spikes': float(len(amplitudes)) if eligible else np.nan,
        'max_spike_rel': float(max(amplitudes, default=0)) if eligible else np.nan,
        'n_spikes': len(amplitudes) if eligible else 0,
        'n_resolved': len(recoveries) if eligible else 0,
        'recovery_sum_minutes': float(sum(recoveries)) if eligible else 0.,
        'night_cells': len(night),
        'nocturnal_min': float(min(night)) if len(night) >= MIN_NIGHT_CELLS else np.nan,
    }


def summarize_days(frame):
    """Every dense day is retained; only declared metric-specific gates differ."""
    out = frame.groupby('USER_ID').agg(
        raw_metric_days=('DAY', 'size'), n_cells=('n_cells', 'sum'),
        n_gt150=('n_gt150', 'sum'), n_peak_days=('daily_n_spikes', 'count'),
        daily_n_spikes=('daily_n_spikes', 'mean'), max_spike_rel=('max_spike_rel', 'mean'),
        n_spikes=('n_spikes', 'sum'), n_resolved=('n_resolved', 'sum'),
        recovery_sum_minutes=('recovery_sum_minutes', 'sum'),
        n_nights=('nocturnal_min', 'count'), nocturnal_min=('nocturnal_min', 'mean'))
    out['pct_gt150'] = 100*out.n_gt150/out.n_cells
    out['spike_resolution'] = out.recovery_sum_minutes/out.n_resolved.replace(0, np.nan)
    out.loc[out.n_peak_days < MIN_DAYS, ['daily_n_spikes', 'max_spike_rel', 'spike_resolution']] = np.nan
    out.loc[out.n_resolved < MIN_RECOVERIES, 'spike_resolution'] = np.nan
    out.loc[out.n_nights < MIN_DAYS, 'nocturnal_min'] = np.nan
    out['resolved_fraction'] = out.n_resolved/out.n_spikes.replace(0, np.nan)
    return out.reset_index()


def summarize_exposures(days):
    """Missing logging/steps are not zero; logged true zero carbohydrate is valid."""
    out = days.groupby('USER_ID').agg(
        n_dense_days=('DAY', 'size'), n_logged_days=('n_logs', lambda x: int(x.gt(0).sum())),
        n_high_log_days=('n_logs', lambda x: int(x.ge(3).sum())),
        n_complete_carb_days=('daily_carbs', 'count'), daily_carbs=('daily_carbs', 'mean'),
        n_steps_days=('steps', 'count'), steps=('steps', 'mean'))
    out['logging_day_fraction'] = out.n_logged_days/out.n_dense_days
    out['complete_carb_day_fraction'] = out.n_complete_carb_days/out.n_dense_days
    out['high_logger'] = out.n_dense_days.ge(14) & (out.n_high_log_days/out.n_dense_days).ge(.8)
    out.loc[out.n_complete_carb_days < MIN_DAYS, 'daily_carbs'] = np.nan
    out.loc[out.n_steps_days < MIN_DAYS, 'steps'] = np.nan
    return out.reset_index()


def rank_association(values, *, age_exposure=False):
    """Columns x,y,age,sex; partial-rank t test uses n-rank(Z)-1 df."""
    a = np.asarray(values, float)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError('Expected finite complete-case x,y,age,sex')
    n = len(a)
    if n < 4 or min(np.ptp(a[:, 0]), np.ptp(a[:, 1])) == 0:
        return dict(rho_simple=np.nan, p_simple=np.nan, rho_partial=np.nan, p_partial=np.nan, partial_df=0)
    ranks = stats.rankdata(a[:, :2], axis=0)
    simple, p_simple = stats.spearmanr(a[:, 0], a[:, 1])
    z = np.column_stack([np.ones(n), a[:, 3]] if age_exposure else
                        [np.ones(n), stats.rankdata(a[:, 2]), a[:, 3]])
    residual = ranks-z@np.linalg.lstsq(z, ranks, rcond=None)[0]
    df = n-np.linalg.matrix_rank(z)-1
    if min(np.linalg.norm(residual, axis=0)) < 1e-8 or df < 1:
        partial, p_partial = np.nan, np.nan
    else:
        partial = float(np.clip(np.corrcoef(residual.T)[0, 1], -1, 1))
        p_partial = float(2*stats.t.sf(abs(partial)*np.sqrt(df/max(1-partial**2, 1e-30)), df))
    return dict(rho_simple=float(simple), p_simple=float(p_simple),
                rho_partial=partial, p_partial=p_partial, partial_df=int(df))


def pair_result(frame, x, y, *, age_exposure=False, bootstrap=500, seed=20260916):
    cols = [x, y, 'age_bucket', 'sex_male']
    a = frame[cols].to_numpy(float)
    a = a[np.isfinite(a).all(axis=1)]
    result = dict(n=len(a), status='insufficient_pairs')
    for key in ['rho_simple', 'p_simple', 'rho_partial', 'p_partial', 'partial_df',
                'rho_simple_ci_low', 'rho_simple_ci_high', 'rho_partial_ci_low', 'rho_partial_ci_high']:
        result[key] = np.nan
    result['bootstrap_finite_simple'] = result['bootstrap_finite_partial'] = 0
    if len(a) < MIN_PAIR_N:
        return result
    result.update(rank_association(a, age_exposure=age_exposure))
    result['status'] = 'estimated' if np.isfinite(result['rho_partial']) else 'constant_or_rank_deficient'
    rng = np.random.default_rng(seed)
    draws = [rank_association(a[rng.integers(0, len(a), len(a))], age_exposure=age_exposure)
             for _ in range(bootstrap)]
    for kind in ['simple', 'partial']:
        values = np.array([r['rho_'+kind] for r in draws])
        values = values[np.isfinite(values)]
        result['bootstrap_finite_'+kind] = len(values)
        if len(values) >= .95*bootstrap and bootstrap > 0:
            result['rho_'+kind+'_ci_low'], result['rho_'+kind+'_ci_high'] = np.quantile(values, [.025, .975])
    return result


def bh_fdr(values):
    p = np.asarray(values, float)
    q = np.full(p.shape, np.nan)
    indices = np.flatnonzero(np.isfinite(p))
    order = indices[np.argsort(p[indices])]
    if len(order):
        q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(order)/np.arange(1, len(order)+1))[::-1])[::-1])
    return q
