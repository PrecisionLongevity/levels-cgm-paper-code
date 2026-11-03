#!/usr/bin/env python3
"""Day-part week-to-week reliability, and one table of each CGM signal's reliability and HOMA-IR association.

Cross-sectional associations and observed reliability only; no projection or extrapolation.

Part A (estimated here). The overnight-minus-midday day-part measure used in
analysis/metabolic/insulin_markers.py (night_minus_day = mean of observed 15-minute cells
00:00-05:00 local minus 11:00-15:00 local) is not covered by the reliability analysis;
its week-to-week reliability is computed here in the same two non-overlapping seven-dense-day windows (w1, w2) and with
the same ICC(A,1) estimator and bootstrap (analysis/reliability/repeatability.py icc_a1 /
summarize_pair, 500 participant resamples) used to produce reliability/axis_reliability.csv.
Window cells are the repeatability.py local-day 96-cell grids (cell 0 = local midnight); only
observed cells enter the day-part means (no interpolation), matching cgmpaper/cgm_trace_features.py
local_mean().

Part B (assembly only). One row per CGM signal with its reliability (x) and its HOMA-IR
association (y) read verbatim from the other producers' outputs, with source file and column.
No value in Part B is estimated here.

Consistency checks performed before writing outputs: mean glucose recomputed from the grids with
repeatability.py's metrics_for_days matches seven_day_cgm_features.parquet; the mean-glucose row of
axis_reliability.csv is reproduced from the recomputed window values; day-part cell bounds equal the
trace-feature hour rule applied to cell midpoints.

Inputs (OUTPUTS/...):
  reliability/axis_reliability.csv, reliability/repeatability_analysis.metadata.json,
  reliability/cache/seven_day_windows.parquet, reliability/cache/seven_day_cgm_features.parquet,
  reliability/cache/daily_grid_manifest.json, reliability/cache/repeatability_daily_grids/
    (analysis/reliability/repeatability.py)
  metabolic/insulin_markers/cgm_measures_by_marker.csv,
  metabolic/insulin_markers/meal_responses_by_marker.csv,
  metabolic/insulin_markers/carb_sensitivity_high_low.csv
    (analysis/metabolic/insulin_markers.py)
  metabolic/hba1c_stratified_spike_recovery/stratified_associations.csv
    (analysis/metabolic/hba1c_stratified_spikes.py)
  meals/spike_facets/spike_facets_repeatability.csv (analysis/meals/spike_facets.py)
  meals/same_food_averaging_observed/observed_averaging_curve.csv
    (analysis/meals/same_food_averaging.py)
  meal_populations/window_mixed_reliability.csv (analysis/meals/meal_populations.py)

Outputs (OUTPUTS/metabolic/observability_inputs/):
  day_part_reliability.csv, observability_points.csv, observability_inputs.metadata.json

Usage: python analysis/metabolic/observability_inputs.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402

VERSION = 'observability-inputs'
PUB = OUTPUTS
SUG = PUB / 'reliability'
CACHE = SUG / 'cache'
OUT = PUB / 'metabolic/observability_inputs'
S_REPEATABILITY = ROOT / 'analysis/reliability/repeatability.py'
IMA = PUB / 'metabolic/insulin_markers'
REC = PUB / 'metabolic/hba1c_stratified_spike_recovery'
FAC = PUB / 'meals/spike_facets'
AVG = PUB / 'meals/same_food_averaging_observed'
QSP = PUB / 'meal_populations'
NIGHT_HOURS, MIDDAY_HOURS = (0, 5), (11, 15)
BOOT_SEED_OFFSET = 50  # distinct from the repeatability analysis offsets 0..6 used for its own metrics


def fail(msg):
    raise SystemExit('GUARD FAILED: ' + msg)


def load_s_repeatability():
    spec = importlib.util.spec_from_file_location('repeatability_for_observability', S_REPEATABILITY)
    m = importlib.util.module_from_spec(spec); sys.modules['repeatability_for_observability'] = m; spec.loader.exec_module(m)
    return m


def cells_for(hours):
    """Cells whose midpoint hour h satisfies lo <= h < hi (trace-feature rule, offset already applied)."""
    mid = (np.arange(96) + .5) * 15 / 60
    return np.flatnonzero((mid >= hours[0]) & (mid < hours[1]))


def load_metadata(guards):
    rep = json.loads((SUG / 'repeatability_analysis.metadata.json').read_text())
    if rep['analysis_version'] != 'publication-repeatability' or rep['primary_design'] != 'fixed_gap':
        fail('repeatability analysis definitions changed')
    fm = json.loads((CACHE / 'repeatability_features.metadata.json').read_text())
    gm = json.loads((CACHE / 'daily_grid_manifest.json').read_text())
    qa = json.loads((QSP / 'population_analysis.json').read_text())
    return rep, fm, gm, qa


def window_values(s_repeatability, gm):
    windows = pd.read_parquet(CACHE / 'seven_day_windows.parquet')
    by_user = {u: p for u, p in windows.groupby('USER_ID', sort=False)}
    night_c, mid_c = cells_for(NIGHT_HOURS), cells_for(MIDDAY_HOURS)
    rows = []
    for i, name in enumerate(gm['parts']):
        daily = pd.read_parquet(CACHE / 'repeatability_daily_grids' / name)
        s_repeatability.validate_grid_arrays(daily)
        for uid, part in daily.groupby('USER_ID', sort=False):
            if uid not in by_user:
                continue
            days = pd.to_datetime(part.DAY).to_numpy(dtype='datetime64[D]').astype(np.int64)
            grid = np.stack(part.glucose.to_numpy())
            for w in by_user[uid].itertuples():
                first = np.datetime64(w.window_start, 'D').astype(np.int64); wanted = first + np.arange(7)
                idx = np.searchsorted(days, wanted)
                if not np.array_equal(days[np.minimum(idx, len(days) - 1)], wanted):
                    fail('selected week lost a dense calendar day')
                g = grid[idx]
                mg = s_repeatability.metrics_for_days(wanted, g)['mean_glucose']
                night, mid = g[:, night_c], g[:, mid_c]
                mn = float(np.nanmean(night)) if np.isfinite(night).any() else np.nan
                mm = float(np.nanmean(mid)) if np.isfinite(mid).any() else np.nan
                rows.append(dict(window_id=w.window_id, USER_ID=uid, window_number=int(w.window_number), mean_glucose_recomputed=mg,
                                 mean_night=mn, mean_midday=mm, night_minus_day=mn - mm,
                                 night_cells=int(np.isfinite(night).sum()), midday_cells=int(np.isfinite(mid).sum())))
        if (i + 1) % 10 == 0:
            print(time.strftime('%H:%M:%S'), f'grid part {i + 1}/{len(gm["parts"])}; windows {len(rows):,}', flush=True)
    V = pd.DataFrame(rows)
    if len(V) != len(windows) or V.window_id.duplicated().any():
        fail(f'window grain: {len(V)} vs {len(windows)}')
    return V


def pick(frame, **kw):
    s = frame
    for k, v in kw.items():
        s = s.loc[s[k] == v]
    if len(s) != 1:
        fail(f'expected one row for {kw}, got {len(s)}')
    return s.iloc[0]


def assemble(rel_axis, dp, qa):
    A = pd.read_csv(IMA / 'cgm_measures_by_marker.csv'); B = pd.read_csv(IMA / 'meal_responses_by_marker.csv')
    C = pd.read_csv(IMA / 'carb_sensitivity_high_low.csv'); F = pd.read_csv(FAC / 'spike_facets_repeatability.csv')
    G = pd.read_csv(AVG / 'observed_averaging_curve.csv'); W = pd.read_csv(QSP / 'window_mixed_reliability.csv')
    R = pd.read_csv(REC / 'stratified_associations.csv')
    ax = rel_axis.set_index('metric')
    rows = []

    def add(signal, label, category, x, x_lo, x_hi, x_design, x_n, x_src, y_row, y_src, y0_row=None, y_kind='percent_per_sd', note=''):
        rows.append(dict(signal=signal, label=label, category=category, reliability=x, reliability_ci_low=x_lo,
                         reliability_ci_high=x_hi, reliability_design=x_design, reliability_n=x_n, reliability_source=x_src,
                         homa_estimate=y_row['estimate'] if y_kind == 'percent_per_sd' else 100 * (y_row['effect'] - 1),
                         homa_ci_low=y_row['ci_low'] if y_kind == 'percent_per_sd' else 100 * (y_row['ci_low'] - 1),
                         homa_ci_high=y_row['ci_high'] if y_kind == 'percent_per_sd' else 100 * (y_row['ci_high'] - 1),
                         homa_n=int(y_row['n']), homa_adjustment=y_row['adjustment'], homa_scale=y_kind, homa_source=y_src,
                         homa_no_bmi_estimate=(np.nan if y0_row is None else (y0_row['estimate'] if y_kind == 'percent_per_sd' else 100 * (y0_row['effect'] - 1))),
                         homa_no_bmi_adjustment=(None if y0_row is None else y0_row['adjustment']),
                         homa_no_bmi_n=(np.nan if y0_row is None else int(y0_row['n'])), note=note))
    src_a = 'insulin_markers/cgm_measures_by_marker.csv'
    src_ax = 'reliability/axis_reliability.csv'
    wk = 'ICC(A,1) week 1 vs week 2 (7 dense days each)'
    # mean glucose: its own association (age+sex+BMI; without BMI: age+sex)
    r = ax.loc['mean_glucose']
    add('mean_glucose', 'Mean glucose', 'reliable_health_marker', r.icc_a1, r.ci_low, r.ci_high, wk, int(r.n_users), src_ax,
        pick(A, marker='homa_ir', adjustment='bmi', measure='mean_glucose'), src_a,
        pick(A, marker='homa_ir', adjustment='age_sex', measure='mean_glucose'),
        note='y is the association of mean glucose itself (age, sex, BMI); open marker: age and sex only')
    d = dp.set_index('metric').loc['night_minus_day']
    add('night_minus_day', 'Overnight − midday glucose', 'reliable_health_marker', d.icc_a1, d.ci_low, d.ci_high, wk, int(d.n_users),
        'metabolic/observability_inputs/day_part_reliability.csv',
        pick(A, marker='homa_ir', adjustment='mean_glucose_bmi', measure='night_minus_day'), src_a,
        pick(A, marker='homa_ir', adjustment='mean_glucose', measure='night_minus_day'))
    for m, lab in [('sd_glucose', 'SD'), ('cv_glucose', 'CV'), ('modd', 'MODD')]:
        r = ax.loc[m]
        add(m, lab, 'variability_summary', r.icc_a1, r.ci_low, r.ci_high, wk, int(r.n_users), src_ax,
            pick(A, marker='homa_ir', adjustment='mean_glucose_bmi', measure=m), src_a,
            pick(A, marker='homa_ir', adjustment='mean_glucose', measure=m))
    sh = 'Pearson r, first vs second half of dense dates (facets recomputed within halves)'
    for facet, measure, lab, cat in [('frequency', 'all_frequency', 'Spike frequency', 'reliable_personal_trait'),
                                     ('size', 'all_size', 'Spike size', 'reliable_personal_trait'),
                                     ('clearance', 'all_clearance', 'Recovery after rises', 'range_specific_signal')]:
        f = pick(F, facet=facet)
        add(measure, lab, cat, f.pearson_r, np.nan, np.nan, sh, int(f.n), 'meals/spike_facets/spike_facets_repeatability.csv',
            pick(A, marker='homa_ir', adjustment='mean_glucose_bmi', measure=measure), src_a,
            pick(A, marker='homa_ir', adjustment='mean_glucose', measure=measure),
            note=('range-specific: HbA1c association within HbA1c 5.7-6.4% from analysis/metabolic/hba1c_stratified_spikes.py' if facet == 'clearance' else ''))
    w = W.iloc[0]
    add('carb_sensitivity', 'Carbohydrate sensitivity', 'reliable_personal_trait', w.pearson_r, w.pearson_ci_low, w.pearson_ci_high,
        'Pearson r, chronological odd vs even meals (window_mixed population, >=30 meals per half)', int(w.n_users),
        'meal_populations/window_mixed_reliability.csv',
        pick(C, marker='homa_ir', adjustment='bmi_mean_glucose'), 'insulin_markers/carb_sensitivity_high_low.csv',
        pick(C, marker='homa_ir', adjustment='age_sex'), y_kind='percent_high_vs_low_half',
        note='y is % difference in HOMA-IR, high vs low half of the slope (ratio - 1), not per SD; no-BMI comparator is age+sex only')
    for k in (1, 7):
        g = pick(G, split='interleaved', k=k)
        add(f'same_food_k{k}', f'Same-food response, {k} meal{"s" if k > 1 else ""}', 'needs_repetition', g.r_empirical, g.ci_low, g.ci_high,
            f'within-person r between means of two disjoint sets of {k} same-food responses (interleaved)', int(g.n_cells),
            'meals/same_food_averaging_observed/observed_averaging_curve.csv',
            pick(B, marker='homa_ir', adjustment='mean_glucose_bmi', measure='peak_rise'), 'insulin_markers/meal_responses_by_marker.csv',
            pick(B, marker='homa_ir', adjustment='mean_glucose', measure='peak_rise'),
            note='x n is participant-food cells; y is the participant mean peak rise of logged meals (>=10 meals)')
    M = pd.DataFrame(rows)
    rr = pick(R, stratum='57_64', outcome='a1c', model='M2', measure='all_clearance')
    return M, rr


def main():
    t0 = time.monotonic(); guards = {}
    rep, fm, gm, qa = load_metadata(guards)
    s_repeatability = load_s_repeatability()
    nc, mc = cells_for(NIGHT_HOURS), cells_for(MIDDAY_HOURS)
    guards['night_cells'] = [int(nc[0]), int(nc[-1])]; guards['midday_cells'] = [int(mc[0]), int(mc[-1])]
    if list(nc) != list(range(0, 20)) or list(mc) != list(range(44, 60)):
        fail('day-part cell bounds')
    V = window_values(s_repeatability, gm)
    feat = pd.read_parquet(CACHE / 'seven_day_cgm_features.parquet', columns=['window_id', 'mean_glucose'])
    J = V.merge(feat, on='window_id', validate='one_to_one')
    diff = float((J.mean_glucose_recomputed - J.mean_glucose).abs().max())
    guards['windows'] = int(len(J)); guards['mean_glucose_max_abs_diff'] = diff
    if diff > 1e-9:
        fail(f'mean glucose recomputation diff {diff}')
    P = J.loc[J.window_number.eq(1)].merge(J.loc[J.window_number.eq(2)], on='USER_ID', suffixes=('_1', '_2'), validate='one_to_one')
    axis = pd.read_csv(SUG / 'axis_reliability.csv')
    pub = axis.set_index('metric').loc['mean_glucose']
    i_mg = s_repeatability.METRICS.index('mean_glucose')
    rep_mg = s_repeatability.summarize_pair(P.mean_glucose_recomputed_1.to_numpy(), P.mean_glucose_recomputed_2.to_numpy(), len(P), 500, s_repeatability.SEED + i_mg)
    for k in ('icc_a1', 'ci_low', 'ci_high', 'pearson_r', 'median_abs_diff'):
        d = abs(rep_mg[k] - pub[k]); guards[f'mean_glucose_{k}_diff'] = float(d)
        if d > 1e-12:
            fail(f'axis_reliability mean_glucose {k} not reproduced: {rep_mg[k]} vs {pub[k]}')
    if rep_mg['n_users'] != int(pub.n_users):
        fail('pair count')
    guards['mean_glucose_icc_reproduced'] = True
    out_rows = [dict(metric='mean_glucose_reproduction', **rep_mg)]
    for j, m in enumerate(['night_minus_day', 'mean_night', 'mean_midday']):
        r = s_repeatability.summarize_pair(P[m + '_1'].to_numpy(), P[m + '_2'].to_numpy(), len(P), 500, s_repeatability.SEED + BOOT_SEED_OFFSET + j)
        valid = np.isfinite(P[m + '_1']) & np.isfinite(P[m + '_2'])
        from scipy import stats
        r.update(metric=m, spearman_r=float(stats.spearmanr(P.loc[valid, m + '_1'], P.loc[valid, m + '_2']).statistic))
        out_rows.append(r)
    DP = pd.DataFrame(out_rows)
    guards['night_minus_day_icc'] = float(DP.set_index('metric').loc['night_minus_day', 'icc_a1'])
    guards['median_night_cells_per_window'] = float(V.night_cells.median()); guards['median_midday_cells_per_window'] = float(V.midday_cells.median())
    M, rr = assemble(axis, DP, qa)
    guards['recovery_prediabetes_a1c_estimate'] = float(rr.estimate); guards['recovery_prediabetes_a1c_n'] = int(rr.n)
    OUT.mkdir(parents=True, exist_ok=True)
    DP.to_csv(OUT / 'day_part_reliability.csv', index=False)
    M.to_csv(OUT / 'observability_points.csv', index=False)
    meta = dict(analysis='observability_inputs', version=VERSION,
                design='observed week-to-week reliability and cross-sectional associations; no projection',
                estimator='repeatability.py icc_a1 / summarize_pair (unchanged), 500 bootstrap resamples',
                seed=int(s_repeatability.SEED + BOOT_SEED_OFFSET), day_part_definition=dict(night_local_hours=list(NIGHT_HOURS), midday_local_hours=list(MIDDAY_HOURS),
                                                                             cells='observed only, no interpolation'),
                guards=guards,
                daily_grid_parts=len(gm['parts']),
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'observability_inputs.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(guards=guards), indent=1))
    print(DP.to_string())


if __name__ == '__main__':
    main()
