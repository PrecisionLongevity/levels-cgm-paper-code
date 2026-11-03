"""Fig. 4: spike width, not height, tracks insulin resistance at the same average glucose.

  a  left, the cohort-median spike (for each tenth of the rise, the median across participants of their average time to
     reach it; IQR shaded) with its height and width; right, a schematic day built from the same shape at the
     cohort-median spike frequency, with logged-meal responses (2-h iAUC, meal peak rise)
  b  % difference in fasting insulin and HOMA-IR per SD of each CGM and logged-meal measure at the same average glucose
  c  spike frequency, height and width at the same average glucose for four markers
  d  fasting insulin per SD of spike width under other adjustments

Inputs (analysis outputs)
  clinical/spike_profile/spike_profile_levels.csv, spike_profile_summary.json        a
  clinical/mage_standard/headline_homa_insulin_mage_standard.csv                     b (every row except spike width)
  clinical/spike_width/width_associations.csv                                        b (spike width), d
  clinical/markers/spike_markers.csv                                                 c
Source data
  source_data/fig4a_spike_profile.csv, fig4a_cohort_medians.csv, fig4b_same_average_glucose.csv,
  fig4c_spike_measures_by_marker.csv, fig4d_spike_width_adjustments.csv
Outputs
  fig4_spike_width.svg / .png

Usage
  python figures/fig4_spike_width.py source-data
  python figures/fig4_spike_width.py draw
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import svgfig as sd  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402
from cgmpaper.svgfig import F, S  # noqa: E402

NAME = 'fig4_spike_width'
TITLE = 'Fig. 4 | Spike width, not height, tracks insulin resistance and metabolic markers at the same average glucose'
CL = OUTPUTS / 'clinical'
SPIKE_THRESHOLD = 30
MIN_MEALS = 10
C_UN, C_MG = sd.INK_SECONDARY, sd.CATEGORICAL[0]
C_CURVE, C_HEIGHT, C_WIDTH = sd.INK_SECONDARY, sd.INK, sd.CATEGORICAL[0]
C_MEAL, C_MEAL_FILL = sd.CATEGORICAL[3], sd.CATEGORICAL[1]
MARKERS = [('fasting_insulin', 'Fasting insulin'), ('homa_ir', 'HOMA-IR')]
B_ROWS = [('level', 'mean_glucose'), ('var', 'sd_glucose'), ('var', 'modd'), ('var', 'cv_glucose'), ('var', 'mage'),
          ('spike', 'all_frequency'), ('spike', 'all_size'), ('spike', 'spike_width'), ('meal', 'iauc_2h'), ('meal', 'peak_rise')]
B_LABELS = {'mean_glucose': 'Average glucose', 'sd_glucose': 'SD', 'modd': 'MODD', 'cv_glucose': 'CV', 'mage': 'MAGE',
            'all_frequency': 'Spike frequency', 'all_size': 'Spike height', 'spike_width': 'Spike width',
            'iauc_2h': 'Meal 2-h iAUC', 'peak_rise': 'Meal peak rise'}
B_GROUPS = {'level': 'Level', 'var': 'Variability', 'spike': f'Spikes (rises ≥{SPIKE_THRESHOLD} mg/dL)', 'meal': f'Logged meals (≥{MIN_MEALS} each)'}
B_AXIS = (-15.0, 35.0)       # panel b axis (% per SD): covers the unadjusted (Fig. 3d) and the adjusted estimates
C_MARKERS = [('fasting_insulin', 'Fasting insulin'), ('homa_ir', 'HOMA-IR'), ('triglycerides', 'Triglycerides'), ('hscrp', 'hsCRP')]
C_MEASURES = [('spike frequency', 'Spike frequency'), ('spike size', 'Spike height'), ('spike width', 'Spike width')]
D_ROWS = [('unadjusted', 'Unadjusted'), ('average glucose', 'Same average glucose'),
          ('average glucose + BMI', 'Same average glucose + BMI'),
          ('average glucose + spike size and frequency', 'Same average glucose + spike height and frequency')]
D_TITLE = 'Fasting insulin per SD of spike width, under other adjustments'
D_NOTE = 'Similar for the width of responses to logged meals (Supplementary Table 2).'
# schematic day: spike times (h) and logged meals (log hour, minutes to peak, size relative to the median spike, spike number)
DAY_SPIKES = [8.0, 12.5, 18.5, 22.0]
DAY_MEALS = [(7.0, 60.0, 1.0, 1), (15.0, 45.0, 0.5, None), (17.5, 60.0, 1.0, 3)]
UNLOGGED_LABEL = 2


# ============================================================================= source data
def make_source_data():
    lv = pd.read_csv(CL / 'spike_profile/spike_profile_levels.csv')
    a1 = lv[['side', 'level', 't_median', 't_q1', 't_q3']].rename(columns={
        'level': 'fraction_of_rise', 't_median': 'minutes_from_peak_median', 't_q1': 'minutes_from_peak_q1', 't_q3': 'minutes_from_peak_q3'})
    s = json.loads((CL / 'spike_profile/spike_profile_summary.json').read_text())
    med = s['cohort_medians']
    a2 = pd.DataFrame([dict(quantity='participants in the cohort-median spike', value=s['n_participants']),
                       dict(quantity='spike height, median of participant averages (mg/dL)', value=med['all_size']['median']),
                       dict(quantity='spike width, median of participant averages (min)', value=med['spike_width']['median']),
                       dict(quantity='spike frequency, median of participant averages (per day)', value=med['all_frequency']['median'])])
    h = pd.read_csv(CL / 'mage_standard/headline_homa_insulin_mage_standard.csv')
    w = pd.read_csv(CL / 'spike_width/width_associations.csv')
    rows = []
    for g, m in B_ROWS:
        for mk, _ in MARKERS:
            if m == 'mean_glucose':
                r = sd.pick(h, adjustment='none', measure=m)
                rows.append(dict(marker=mk, group=B_GROUPS[g], measure=B_LABELS[m], n_participants=int(r[f'n_{mk}'])))
                continue
            if m == 'spike_width':
                r = w.loc[(w.marker == mk) & (w.measure == 'spike_width') & (w.compared_at_same == 'average glucose')].iloc[0]
                n, e, lo, hi = int(r.n), r.pct, r.lo, r.hi
            else:
                r = sd.pick(h, adjustment='mean_glucose_only', measure=m)
                n, e, lo, hi = int(r[f'n_{mk}']), r[f'estimate_{mk}'], r[f'ci_low_{mk}'], r[f'ci_high_{mk}']
            rows.append(dict(marker=mk, group=B_GROUPS[g], measure=B_LABELS[m], n_participants=n, pct_per_sd=e, ci_low=lo, ci_high=hi))
    b = pd.DataFrame(rows)
    t = pd.read_csv(CL / 'markers/spike_markers.csv')
    c = pd.DataFrame([dict(marker=mk, measure=lab, n_participants=int(r.n), pct_per_sd=r.pct, ci_low=r.lo, ci_high=r.hi)
                      for mk, _ in C_MARKERS for key, lab in C_MEASURES
                      for r in [t.loc[(t.marker == mk) & (t.measure == key)].iloc[0]]])
    wd = w.loc[(w.marker == 'fasting_insulin') & (w.measure == 'spike_width')].set_index('compared_at_same')
    d = pd.DataFrame([dict(adjustment=lab, n_participants=int(wd.loc[key, 'n']), pct_per_sd=wd.loc[key, 'pct'],
                           ci_low=wd.loc[key, 'lo'], ci_high=wd.loc[key, 'hi']) for key, lab in D_ROWS])
    return [source_data.write('fig4a_spike_profile', a1), source_data.write('fig4a_cohort_medians', a2),
            source_data.write('fig4b_same_average_glucose', b), source_data.write('fig4c_spike_measures_by_marker', c),
            source_data.write('fig4d_spike_width_adjustments', d)]


# ============================================================================= drawing helpers
def arrow_v(x, y0, y1, col, w=1.2, head=3.0):
    """Double-headed vertical arrow between y0 (bottom) and y1 (top)."""
    return [sd.line(x, y0 - head, x, y1 + head, stroke=col, stroke_width=w),
            sd.path(f'M {sd.fmt(x)} {sd.fmt(y1)} l -2.4 {sd.fmt(head + 0.6)} l 4.8 0 Z', fill=col),
            sd.path(f'M {sd.fmt(x)} {sd.fmt(y0)} l -2.4 {sd.fmt(-head - 0.6)} l 4.8 0 Z', fill=col)]


def arrow_h2(x0, x1, y, col, w=1.4, head=3.0):
    return [sd.line(x0 + head, y, x1 - head, y, stroke=col, stroke_width=w),
            sd.path(f'M {sd.fmt(x0)} {sd.fmt(y)} l {sd.fmt(head + 0.6)} -2.4 l 0 4.8 Z', fill=col),
            sd.path(f'M {sd.fmt(x1)} {sd.fmt(y)} l {sd.fmt(-head - 0.6)} -2.4 l 0 4.8 Z', fill=col)]


def triangle(x, y, col, s=3.4):
    return sd.path(f'M {sd.fmt(x)} {sd.fmt(y - s)} L {sd.fmt(x + s)} {sd.fmt(y + s * 0.8)} L {sd.fmt(x - s)} {sd.fmt(y + s * 0.8)} Z', fill=col)


def n_range(v):
    v = sorted(set(v))
    return 'n = ' + (f'{v[0]:,}' if len(v) == 1 else f'{v[0]:,}–{v[-1]:,}')


# ============================================================================= panels
def panel_a(x0, y0, prof, med):
    up = prof[prof.side == 'up'].sort_values('fraction_of_rise')
    dn = prof[prof.side == 'down'].sort_values('fraction_of_rise', ascending=False)
    H = med['spike height, median of participant averages (mg/dL)']
    out = [sd.plabel(x0, y0, 'a', 'How spikes and logged-meal responses are measured')]
    px0, px1 = x0 + 40, x0 + 222
    ytop, ybase = y0 + 40, y0 + 130
    tx = sd.Linear(-95, 105, px0, px1)
    yv = sd.Linear(0, 50, ybase, ytop)
    out.append(sd.text(x0 + 14, y0 + 23, f"Cohort-median spike, {int(med['participants in the cohort-median spike']):,} participants; shading, IQR",
                       size=S, fill=sd.INK_SECONDARY))
    under = [sd.rect(tx(-90), ytop - 4, tx(0) - tx(-90), ybase - ytop + 4, fill=sd.WASH)]
    out.append(sd.text(tx(-88), ytop + 3, '90-min look-back', size=S, fill=sd.INK_MUTED))
    for v in (10, 20, 30, 40):
        under.append(sd.line(px0, yv(v), px1, yv(v), stroke=sd.GRID, stroke_width=0.8))
    for v in (0, 10, 20, 30, 40):
        out.append(sd.text(px0 - 5, yv(v) + 2.7, f'{v}', size=S, fill=sd.INK_MUTED, anchor='end'))
    cy = (ytop + ybase) / 2
    out.append(sd.text(x0 + 10, cy, 'mg/dL above the 90-min minimum', size=S, fill=sd.INK_SECONDARY, anchor='middle',
                       transform=f'rotate(-90,{sd.fmt(x0 + 10)},{sd.fmt(cy)})'))
    under.append(sd.line(px0, ybase, px1, ybase, stroke=sd.INK_MUTED, stroke_width=0.9, stroke_dasharray='3 2'))
    outer = ([(tx(r.minutes_from_peak_q1), yv(r.fraction_of_rise * H)) for r in up.itertuples()] + [(tx(0), yv(H))]
             + [(tx(r.minutes_from_peak_q3), yv(r.fraction_of_rise * H)) for r in dn.itertuples()])
    inner = ([(tx(r.minutes_from_peak_q3), yv(r.fraction_of_rise * H)) for r in up.itertuples()] + [(tx(0), yv(H))]
             + [(tx(r.minutes_from_peak_q1), yv(r.fraction_of_rise * H)) for r in dn.itertuples()])
    d = ('M ' + ' L '.join(f'{sd.fmt(a)} {sd.fmt(b)}' for a, b in outer) + ' L '
         + ' L '.join(f'{sd.fmt(a)} {sd.fmt(b)}' for a, b in reversed(inner)) + ' Z')
    under.append(sd.path(d, fill=sd.BLUE_RAMP[1], stroke=None, opacity=0.75))
    pts = ([(float(r.minutes_from_peak_median), float(r.fraction_of_rise)) for r in up.itertuples()] + [(0.0, 1.0)]
           + [(float(r.minutes_from_peak_median), float(r.fraction_of_rise)) for r in dn.itertuples()])
    out.append(sd.polyline([(tx(t), yv(f * H)) for t, f in pts if t <= 0 or f >= 0.5], C_CURVE, width=1.7))
    out.append(sd.path('M ' + ' L '.join(f'{sd.fmt(tx(t))} {sd.fmt(yv(f * H))}' for t, f in pts if t > 0 and f <= 0.5),
                       stroke=C_CURVE, stroke_width=1.5, stroke_dasharray='3 2'))
    t_on = float(up.loc[up.fraction_of_rise == 0.0, 'minutes_from_peak_median'].iloc[0])
    t_up = float(up.loc[up.fraction_of_rise == 0.5, 'minutes_from_peak_median'].iloc[0])
    t_dn = float(dn.loc[dn.fraction_of_rise == 0.5, 'minutes_from_peak_median'].iloc[0])
    out.append(sd.circle(tx(t_on), ybase, 2.2, fill=C_CURVE))
    out += arrow_v(tx(0), ybase, yv(H), C_HEIGHT)
    out += arrow_h2(tx(t_up), tx(t_dn), yv(H / 2), C_WIDTH, w=1.6)
    xa = ybase + 6
    out.append(sd.line(px0, xa, px1, xa, stroke=sd.RULE))
    for t in (-90, -60, -30, 0, 30, 60, 90):
        out.append(sd.line(tx(t), xa, tx(t), xa + 3, stroke=sd.RULE))
        out.append(sd.text(tx(t), xa + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text((px0 + px1) / 2, xa + 21, 'Minutes from the peak', size=S, fill=sd.INK_SECONDARY, anchor='middle'))
    hx, wcol = px1 + 8, 330 - (px1 + 8) + x0 - 6
    entries = [(ytop + 2, C_HEIGHT, f'Height {H:.1f} mg/dL', 'rise from the 90-min minimum to the peak; ≥30 to count'),
               (yv(H / 2) - 6, C_WIDTH, f"Width {med['spike width, median of participant averages (min)']:.0f} min", 'time above half the rise')]
    last = None
    for yy, col, head, body in entries:
        out.append(sd.text(hx, yy, head, size=S, fill=col, weight='700'))
        els, last = sd.wrap_lines(hx, yy + 9.4, body, wcol, size=S, lead=9.4, fill=sd.INK_SECONDARY)
        out += els
    out.append(sd.text(hx, max(last, ybase - 2) + 6, 'cohort medians', size=S, fill=sd.INK_MUTED, font_style='italic'))
    left_bottom = max(xa + 24, last + 8)
    # schematic day
    sx0, sx1 = x0 + 346, 634
    out.append(sd.line(x0 + 338, y0 + 16, x0 + 338, left_bottom, stroke=sd.GRID))
    freq = med['spike frequency, median of participant averages (per day)']
    out.append(sd.text(sx0, y0 + 23, f'One day, schematic: {int(round(freq))} spikes = cohort-median frequency ({freq:.1f} per day)',
                       size=S, fill=sd.INK_SECONDARY))
    th = sd.Linear(0, 24, sx0 + 4, sx1 - 4)
    t_arr, f_arr = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    shape = lambda x: np.interp(x, t_arr, f_arr, left=0.0, right=0.0)  # noqa: E731
    grid = np.arange(0, 24 * 60 + 1, 2.0)
    g = np.zeros_like(grid)
    for h_ in DAY_SPIKES:
        g += H * shape(grid - h_ * 60)
    for log_h, off, scale, num in DAY_MEALS:
        if num is None:
            g += scale * H * shape(grid - (log_h * 60 + off))
    pk = {k + 1: (th(h_), yv(H)) for k, h_ in enumerate(DAY_SPIKES)}
    for log_h, off, scale, num in DAY_MEALS:
        m0 = log_h * 60
        sel = (grid >= m0) & (grid <= m0 + 120)
        poly = [(th(m0 / 60), ybase)] + [(th(t / 60), yv(v)) for t, v in zip(grid[sel], g[sel])] + [(th((m0 + 120) / 60), ybase)]
        under.append(sd.path('M ' + ' L '.join(f'{sd.fmt(a)} {sd.fmt(b)}' for a, b in poly) + ' Z', fill=C_MEAL_FILL, stroke=None, opacity=0.35))
        out.append(triangle(th(m0 / 60), ybase + 5.5, C_MEAL))
    under.append(sd.line(th(0), ybase, th(24), ybase, stroke=sd.INK_MUTED, stroke_width=0.9, stroke_dasharray='3 2'))
    out.append(sd.polyline([(th(t / 60), yv(v)) for t, v in zip(grid, g)], C_CURVE, width=1.5))
    for k, (x, y) in pk.items():
        out.append(sd.text(x, y - 5, str(k), size=F['body'], fill=sd.INK, anchor='middle', weight='700'))
    x2, y2 = pk[UNLOGGED_LABEL]
    out.append(sd.text(x2 + 6, y2 - 5, 'no meal logged', size=S, fill=sd.INK_MUTED))
    sn = [m for m in DAY_MEALS if m[3] is None][0]
    snx = th((sn[0] * 60 + sn[1]) / 60)
    out.append(sd.text(snx, yv(sn[2] * H) - 14.4, 'rise <30:', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text(snx, yv(sn[2] * H) - 5, 'no spike', size=S, fill=sd.INK_MUTED, anchor='middle'))
    m1 = [m for m in DAY_MEALS if m[3] == 1][0]
    ax = th(m1[0]) - 7
    out += arrow_v(ax, ybase, yv(H), C_MEAL)
    out.append(sd.line(ax + 2, yv(H), pk[1][0] - 2, yv(H), stroke=C_MEAL, stroke_width=0.8, stroke_dasharray='1.5 1.5'))
    out.append(sd.text(ax - 4, yv(H * 0.62), 'Meal peak rise', size=S, fill=C_MEAL, anchor='end', weight='600'))
    out.append(sd.text(ax - 4, yv(H * 0.62) + 9.4, '2-h iAUC (shaded)', size=S, fill=C_MEAL, anchor='end', weight='600'))
    out.append(sd.line(th(0), xa + 4, th(24), xa + 4, stroke=sd.RULE))
    for h_ in (0, 6, 12, 18, 24):
        out.append(sd.line(th(h_), xa + 4, th(h_), xa + 7, stroke=sd.RULE))
        out.append(sd.text(th(h_), xa + 15, f'{h_:02d}:00', size=S, fill=sd.INK_MUTED, anchor='middle'))
    yk = xa + 27
    out.append(sd.line(sx0, yk - 2.6, sx0 + 9, yk - 2.6, stroke=C_CURVE, stroke_width=1.5))
    out.append(sd.text(sx0 + 13, yk, 'Spikes (1–4): rises ≥30 mg/dL found from CGM alone, logged or not', size=S, fill=sd.INK_SECONDARY))
    out.append(triangle(sx0 + 4.5, yk + 9.4 - 2.6, C_MEAL))
    out.append(sd.text(sx0 + 13, yk + 9.4, 'Logged meal: 2-h iAUC and meal peak rise, spike or not', size=S, fill=sd.INK_SECONDARY))
    return under + out, max(left_bottom, yk + 9.4 + 4)


def panel_b(x0, y0, b, right):
    lim = B_AXIS
    out = [sd.plabel(x0, y0, 'b', 'At the same average glucose')]
    out.append(sd.text(x0 + 14, y0 + 23, 'Insulin markers per SD of each measure', size=S, fill=sd.INK_SECONDARY))
    lab_x = x0 + 102
    cw = (right - lab_x - 10) / 2
    fw = cw - 38
    cols = [(mk, lab_x + 10 + i * cw) for i, (mk, _) in enumerate(MARKERS)]
    yh = y0 + 23 + 15
    for mk, cx in cols:
        out.append(sd.text(cx + fw / 2 + 8, yh, dict(MARKERS)[mk], size=F['body'], fill=C_MG, anchor='middle', weight='600'))
    out.append(sd.line(x0, yh + 5, right, yh + 5, stroke=sd.RULE))
    y, prev, y_hdr, rows_y = yh + 5, None, None, []
    for g in b.group.unique():
        y += 17 if prev else 12
        out.append(sd.text(x0 + 2, y, g, size=S, fill=sd.INK_MUTED, weight='600'))
        for mk, cx in cols:
            out.append(sd.text(cx + cw - 6, y, n_range(b.loc[(b.group == g) & (b.marker == mk), 'n_participants']), size=S,
                               fill=sd.INK_MUTED, anchor='end'))
        y += 1
        prev, y_hdr = g, y
        for meas in b.loc[b.group == g, 'measure'].unique():
            y += 14 if y_hdr == y else 13
            rows_y.append(y)
            out.append(sd.text(lab_x, y + 3, meas, size=F['body'], fill=sd.INK, anchor='end', weight='600' if meas == 'Average glucose' else None))
            if meas == 'Average glucose':
                continue
            for mk, cx in cols:
                x = sd.Linear(lim[0], lim[1], cx, cx + fw)
                r = sd.pick(b, marker=mk, measure=meas)
                out += sd.ci_mark(x, lim, y, r.pct_per_sd, r.ci_low, r.ci_high, C_MG, r=1.9, caps=True)
                ex = sd.excl0(r.ci_low, r.ci_high)
                out.append(sd.text(cx + cw - 6, y + 3, sd.pct(r.pct_per_sd), size=S, fill=C_MG if ex else sd.INK_MUTED, anchor='end',
                                   weight='700' if ex else None))
    rx_ = (cols[0][1] + cols[1][1] + fw) / 2 + 5
    rtxt = 'held the same in every row below'
    out.append(sd.rect(rx_ - sd.tw(rtxt, S) / 2 - 3, rows_y[0] - 6, sd.tw(rtxt, S) + 6, 12, fill=sd.SURFACE))
    out.append(sd.text(rx_, rows_y[0] + 3, rtxt, size=S, fill=sd.INK_MUTED, anchor='middle', font_style='italic'))
    ybot = y + 9
    ticks = [t for t in range(int(math.ceil(lim[0] / 10) * 10), int(lim[1]) + 1, 10)]
    under = []
    for mk, cx in cols:
        x = sd.Linear(lim[0], lim[1], cx, cx + fw)
        under.append(sd.line(x(0), rows_y[0] - 9, x(0), ybot, stroke=sd.INK_MUTED, stroke_width=0.9))
        for t in ticks:
            if t:
                under.append(sd.line(x(t), rows_y[0] - 9, x(t), ybot, stroke=sd.GRID, stroke_width=0.8))
            out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
            out.append(sd.text(x(t), ybot + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
        out.append(sd.line(x(lim[0]), ybot, x(lim[1]), ybot, stroke=sd.RULE))
    out.append(sd.text((cols[0][1] + cols[1][1] + fw) / 2, ybot + 23, '% difference per SD, at the same average glucose', size=S,
                       fill=C_MG, anchor='middle'))
    return under + out, ybot + 27


def panel_c(x0, y0, c, right, y_end_target):
    lim = (float(math.floor(c.ci_low.min() / 5) * 5), float(math.ceil(c.ci_high.max() / 5) * 5))
    out = [sd.plabel(x0, y0, 'c', 'Width, not height, across markers')]
    out.append(sd.text(x0 + 14, y0 + 23, 'At the same average glucose; % per SD of each spike measure', size=S, fill=sd.INK_SECONDARY))
    lab_x = x0 + 72
    fx0, fx1 = lab_x + 8, right - 40
    x = sd.Linear(lim[0], lim[1], fx0, fx1)
    yh = y0 + 23 + 15
    out.append(sd.line(x0, yh + 5, right, yh + 5, stroke=sd.RULE))
    y = yh + 5
    rows_y = []
    step_g = 15.0
    avail = y_end_target - 27 - 9 - (yh + 5)        # row pitch stretched so that c ends where b ends
    step_r = max(12.6, (avail - len(C_MARKERS) * step_g - 2) / (len(C_MARKERS) * len(C_MEASURES)))
    for mk, mlab in C_MARKERS:
        y += step_g
        out.append(sd.text(x0 + 2, y, mlab, size=F['body'], fill=sd.INK, weight='700'))
        out.append(sd.text(right, y, n_range(c.loc[c.marker == mk, 'n_participants']), size=S, fill=sd.INK_MUTED, anchor='end'))
        for _, lab in C_MEASURES:
            y += step_r
            rows_y.append(y)
            r = sd.pick(c, marker=mk, measure=lab)
            ex = sd.excl0(r.ci_low, r.ci_high)
            out.append(sd.text(lab_x, y + 3, lab, size=S, fill=sd.INK, anchor='end', weight='600' if lab == 'Spike width' else None))
            out += sd.ci_mark(x, lim, y, r.pct_per_sd, r.ci_low, r.ci_high, C_MG, r=1.9, caps=True)
            out.append(sd.text(right, y + 3, sd.pct(r.pct_per_sd), size=S, fill=C_MG if ex else sd.INK_MUTED, anchor='end', weight='700' if ex else None))
        y += 2
    ybot = y + 7
    under = [sd.line(x(0), rows_y[0] - 9, x(0), ybot, stroke=sd.INK_MUTED, stroke_width=0.9)]
    for t in range(int(math.ceil(lim[0] / 10) * 10), int(lim[1]) + 1, 10):
        if t:
            under.append(sd.line(x(t), rows_y[0] - 9, x(t), ybot, stroke=sd.GRID, stroke_width=0.8))
        out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
        out.append(sd.text(x(t), ybot + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.line(fx0, ybot, fx1, ybot, stroke=sd.RULE))
    out.append(sd.text((fx0 + fx1) / 2, ybot + 23, '% difference per SD, at the same average glucose', size=S, fill=C_MG, anchor='middle'))
    return under + out, ybot + 27


def panel_d(x0, y0, d):
    lim = (min(0.0, float(math.floor(d.ci_low.min() / 2) * 2)), float(math.ceil(d.ci_high.max() / 2) * 2))
    n0 = int(d.n_participants.iloc[0])
    out = [sd.plabel(x0, y0, 'd', D_TITLE)]
    out.append(sd.text(x0 + 14, y0 + 23, f'% difference per SD of spike width; {n0:,} participants unless shown', size=S, fill=sd.INK_SECONDARY))
    lab_x = x0 + 236
    fx0, fx1 = lab_x + 12, lab_x + 12 + 230
    vx = fx1 + 14
    x = sd.Linear(lim[0], lim[1], fx0, fx1)
    yh = y0 + 23 + 15
    out.append(sd.text((fx0 + fx1) / 2, yh, 'Fasting insulin', size=F['body'], fill=C_MG, anchor='middle', weight='600'))
    out.append(sd.line(x0, yh + 5, 634, yh + 5, stroke=sd.RULE))
    y, rows_y = yh + 5, []
    for r in d.itertuples():
        y += 14
        rows_y.append(y)
        col = C_UN if r.adjustment == 'Unadjusted' else C_MG
        ex = sd.excl0(r.ci_low, r.ci_high)
        out.append(sd.text(lab_x, y + 3, r.adjustment, size=F['body'], fill=sd.INK, anchor='end',
                           weight='600' if r.adjustment == 'Same average glucose' else None))
        out += sd.ci_mark(x, lim, y, r.pct_per_sd, r.ci_low, r.ci_high, col, r=2.2, caps=True)
        out.append(sd.text(vx, y + 3, sd.ci_txt(r.pct_per_sd, r.ci_low, r.ci_high), size=S, fill=col if ex else sd.INK_MUTED,
                           weight='700' if ex else None))
        if int(r.n_participants) != n0:
            out.append(sd.text(634, y + 3, f'n = {int(r.n_participants):,}', size=S, fill=sd.INK_MUTED, anchor='end'))
    ybot = y + 10
    step = 2 if lim[1] - lim[0] <= 12 else 5
    under = [sd.line(x(0), rows_y[0] - 9, x(0), ybot, stroke=sd.INK_MUTED, stroke_width=0.9)]
    for t in range(int(lim[0]), int(lim[1]) + 1, step):
        if t:
            under.append(sd.line(x(t), rows_y[0] - 9, x(t), ybot, stroke=sd.GRID, stroke_width=0.8))
        out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
        out.append(sd.text(x(t), ybot + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.line(fx0, ybot, fx1, ybot, stroke=sd.RULE))
    out.append(sd.text((fx0 + fx1) / 2, ybot + 23, '% difference in fasting insulin per SD of spike width', size=S,
                       fill=sd.INK_SECONDARY, anchor='middle'))
    out.append(sd.text(x0 + 14, ybot + 38, D_NOTE, size=S, fill=sd.INK_SECONDARY))
    return under + out, ybot + 45


# ============================================================================= figure
def draw(out_dir):
    prof = source_data.read('fig4a_spike_profile')
    med = source_data.values('fig4a_cohort_medians')
    b, c, d = (source_data.read('fig4b_same_average_glucose'), source_data.read('fig4c_spike_measures_by_marker'),
               source_data.read('fig4d_spike_width_adjustments'))
    body = []
    ya = sd.TOP + 2
    g_a, ya_end = panel_a(6, ya, prof, med)
    body += g_a + [sd.line(6, ya_end + 3, 634, ya_end + 3, stroke=sd.GRID)]
    yb = ya_end + 11
    split = 344
    g_b, yb_end = panel_b(6, yb, b, right=split - 8)
    g_c, yc_end = panel_c(split + 8, yb, c, right=634, y_end_target=yb_end)
    body += g_b + g_c + [sd.line(split, yb + 2, split, max(yb_end, yc_end) - 2, stroke=sd.GRID)]
    yd = max(yb_end, yc_end) + 6
    body.append(sd.line(6, yd, 634, yd, stroke=sd.GRID))
    g_d, yd_end = panel_d(6, yd + 8, d)
    body += g_d + sd.key_filled(20, yd_end + 4)
    return sd.write(sd.render_svg(yd_end + 10, [], body, TITLE), out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
