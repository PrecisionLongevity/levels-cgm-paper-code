"""Fig. 6: reliability of food responses and carbohydrate sensitivity.

  a  one participant's repeated meals of the same food (glucose change from the pre-meal level) and their average
  b  agreement (Pearson r, 95% CI) between two averages of k meals of the same food in the same person, first k vs next
     k meals in time order
  c  schematic of carbohydrate sensitivity (peak rise per gram of carbohydrate) and three participants nearest the 10th,
     50th and 90th percentile slopes
  d  distribution of slopes across participants, and agreement between slopes from odd- and even-numbered meals
  e  fasting insulin across quintiles of carbohydrate sensitivity, and the per-SD associations

Panels a, c and the scatter in d show individual participants. Their participant-level inputs are never distributed;
drawing this figure therefore needs the analysis outputs. The source data hold only the summary values printed on
those panels.

Inputs (analysis outputs)
  exemplars/traces/fig6_trace_exemplars.json                 a  (restricted; analysis/exemplars/exemplar_traces.py)
  exemplars/exemplar_manifest.json                            c  (restricted; analysis/exemplars/exemplar_candidates.py)
  meals/same_food_direct/same_food_direct_curve.csv           b
  meal_populations/_cache/broad_mixed_participants.parquet    d, distribution (restricted)
  meal_populations/_cache/broad_mixed_halves.parquet          d, odd/even scatter (restricted)
  meal_populations/broad_mixed_reliability.csv                d
  meal_populations/population_analysis.json                   e, participants with slopes near the laboratory draw
  clinical/markers/carb_markers_quintiles.csv, carb_markers_continuous.csv   e
Source data
  source_data/fig6a_example.csv, fig6b_same_food_agreement.csv, fig6c_participants.csv, fig6d_slope_histogram.csv,
  fig6d_summary.csv, fig6e_insulin_by_quintile.csv, fig6e_per_sd.csv
  In fig6d_slope_histogram.csv, bins holding 1-10 participants are merged with a neighbouring bin.
Outputs
  fig6_meals.svg / .png

Usage
  python figures/fig6_meals.py source-data
  python figures/fig6_meals.py draw
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
from cgmpaper.svgfig import F, S, wrap  # noqa: E402

NAME = 'fig6_meals'
TITLE = 'Fig. 6 | Reliability of food responses and carbohydrate sensitivity'
EXEMPLARS = OUTPUTS / 'exemplars'
QSP = OUTPUTS / 'meal_populations'
SAME_FOOD = OUTPUTS / 'meals/same_food_direct/same_food_direct_curve.csv'
MARKERS = OUTPUTS / 'clinical/markers'
K_MAX = 7
NOTE_SHIFT = 20              # panel b notes moved up to clear the 'one meal' label
SLOPE_BIN = 0.04             # histogram bin width (mg/dL per g)
EXAMPLES = [('p10', 3, '10th percentile'), ('p50', 1, 'median'), ('p90', 4, '90th percentile')]   # (group, rank, label)
C_I = sd.CATEGORICAL[0]
C_THIN = '#A9B0BC'
C_LOWS, C_HIGHS = sd.CATEGORICAL[0], sd.ACCENT
C_IR = sd.CATEGORICAL[0]


# ============================================================================= inputs
def load_examples():
    """The three participants of panel c (restricted)."""
    man = json.loads((EXEMPLARS / 'exemplar_manifest.json').read_text())['panels']['carb_slope_panel']
    out = []
    for g, rank, lab in EXAMPLES:
        c = [c for c in man['groups'][g] if c['rank'] == rank][0]
        t = c['trace']
        out.append(dict(label=lab, acs=c['metrics']['acs'], n_meals=c['metrics']['n_meals'], carbs=np.asarray(t['carbs_g'], float),
                        rise=np.asarray(t['peak_rise'], float), intercept=t['intercept_at_mean_macros']))
    return out


def slope_histogram(acs):
    lo_q, hi_q = np.quantile(acs, [0.005, 0.995])
    lo, hi = SLOPE_BIN * math.floor(lo_q / SLOPE_BIN), SLOPE_BIN * math.ceil(hi_q / SLOPE_BIN)
    edges = np.round(np.arange(lo, hi + SLOPE_BIN / 2, SLOPE_BIN), 6)
    counts, _ = np.histogram(acs, edges)
    return edges, counts, int((acs < lo).sum()) + int((acs >= edges[-1]).sum())


def odd_even_axes(a, b):
    lo_q, hi_q = np.quantile(np.concatenate([a, b]), [0.005, 0.995])
    lo, hi = 0.5 * math.floor(lo_q / 0.5), 0.5 * math.ceil(hi_q / 0.5)
    return lo, hi, (a >= lo) & (a <= hi) & (b >= lo) & (b <= hi)


def make_source_data():
    ex = json.loads((EXEMPLARS / 'traces/fig6_trace_exemplars.json').read_text())['selected']['meal_to_meal']
    a = pd.DataFrame([dict(food=ex['food'], n_meals_shown=int(ex['n_shown']), n_meals=int(ex['n_meals']))])
    cur = pd.read_csv(SAME_FOOD)
    b = cur.loc[(cur.split == 'time-separated') & cur.k.le(K_MAX), ['k', 'n_pairs', 'n_participants', 'r', 'ci_low', 'ci_high',
                                                                   'median_days_between_sets']]
    # slopes rounded to the printed precision (individual participants)
    c = pd.DataFrame([dict(participant=f"{e['label']} slope", slope_mg_dl_per_g=round(float(e['acs']), 2), n_meals=e['n_meals'])
                      for e in load_examples()])
    acs = pd.read_parquet(QSP / '_cache/broad_mixed_participants.parquet', columns=['acs']).acs.to_numpy(float)
    edges, counts, n_off = slope_histogram(acs)
    d_hist = source_data.merge_small_bins(pd.DataFrame(dict(bin_lo=edges[:-1], bin_hi=edges[1:], count=counts)))
    d_hist = d_hist.rename(columns={'bin_lo': 'slope_from', 'bin_hi': 'slope_to', 'count': 'n_participants'})
    q10, q50, q90 = (float(v) for v in np.quantile(acs, [0.10, 0.50, 0.90]))
    halves = pd.read_parquet(QSP / '_cache/broad_mixed_halves.parquet', columns=['acs_first', 'acs_second'])
    *_, keep = odd_even_axes(halves.acs_first.to_numpy(float), halves.acs_second.to_numpy(float))
    rel = pd.read_csv(QSP / 'broad_mixed_reliability.csv').iloc[0]
    d_summ = pd.DataFrame([('participants with a slope', len(acs)), ('10th percentile slope', q10), ('median slope', q50),
                           ('90th percentile slope', q90), ('participants outside the 0.5th-99.5th percentile axis', n_off),
                           ('odd/even agreement, Pearson r', rel.pearson_r), ('odd/even r, 95% CI low', rel.pearson_ci_low),
                           ('odd/even r, 95% CI high', rel.pearson_ci_high), ('participants in the odd/even comparison', int(rel.n_users)),
                           ('minimum meals per half', int(rel.minimum_meals_per_half)),
                           ('odd/even pairs outside the axes', int((~keep).sum()))], columns=['quantity', 'value'])
    q = pd.read_csv(MARKERS / 'carb_markers_quintiles.csv').query("marker == 'fasting_insulin'").sort_values('quintile')
    e1 = q[['quintile', 'n', 'geometric_mean', 'lo', 'hi']].rename(columns={'n': 'n_participants', 'geometric_mean': 'fasting_insulin_geometric_mean',
                                                                             'lo': 'ci_low', 'hi': 'ci_high'})
    cn = pd.read_csv(MARKERS / 'carb_markers_continuous.csv').set_index('marker')
    pa = json.loads((QSP / 'population_analysis.json').read_text())['summary']['window_mixed']
    e2 = pd.DataFrame([dict(marker=mk, n_participants=int(cn.loc[mk, 'n']), pct_per_sd=cn.loc[mk, 'pct_per_sd'], ci_low=cn.loc[mk, 'lo'],
                            ci_high=cn.loc[mk, 'hi'], n_participants_with_slopes_near_lab_draw=int(pa['n_users']))
                       for mk in ('fasting_insulin', 'homa_ir')])
    return [source_data.write('fig6a_example', a), source_data.write('fig6b_same_food_agreement', b),
            source_data.write('fig6c_participants', c), source_data.write('fig6d_slope_histogram', d_hist),
            source_data.write('fig6d_summary', d_summ), source_data.write('fig6e_insulin_by_quintile', e1),
            source_data.write('fig6e_per_sd', e2)]


# ============================================================================= panels
def trace(xs, ys, minutes, vals, col, width=1.5):
    """Polyline over observed cells only; a missing cell breaks the line (no interpolation)."""
    segs, cur = [], []
    for m, v in zip(minutes, vals):
        if v is None or not np.isfinite(v):
            cur = []
        else:
            cur.append((xs(m), ys(v)))
    if len(cur) > 1:
        segs.append(cur)
    return [sd.path('M ' + ' L '.join(f'{sd.fmt(x)} {sd.fmt(y)}' for x, y in g), stroke=col, stroke_width=width, stroke_linejoin='round')
            for g in segs]


def panel_example(p, ex, a):
    minutes = ex['minutes']
    mm = ex['selected']['meal_to_meal']
    allv = [v for m in mm['meals'] for v in m if v is not None] + [v for v in mm['average'] if v is not None]
    lo, hi = 20 * math.floor(min(allv) / 20), 20 * math.ceil(max(allv) / 20)
    xs = sd.Linear(-30, 120, p.px0, p.px1)
    ys = sd.Linear(lo, hi, p.py1, p.py0)
    out = [sd.axis_left(p, ys, list(range(int(lo), int(hi) + 1, 20)), 'Glucose change (mg/dL)',
                        fmt_fn=lambda v: f'{v:+.0f}'.replace('-', '−') if v else '0'),
           sd.axis_bottom(p, xs, [-30, 0, 30, 60, 90, 120], 'Minutes after the meal', fmt_fn=lambda v: f'{v:.0f}'.replace('-', '−')),
           sd.line(p.px0, ys(0), p.px1, ys(0), stroke=sd.INK_MUTED, stroke_width=0.9),
           sd.line(xs(0), p.py0, xs(0), p.py1, stroke=sd.RULE, stroke_width=0.8, stroke_dasharray='3 2')]
    for m in mm['meals']:
        out += trace(xs, ys, minutes, m, C_THIN, 0.8)
    out += trace(xs, ys, minutes, mm['average'], sd.INK, 1.9)
    r = a.iloc[0]
    nm = f'{int(r.n_meals)} meals' if r.n_meals_shown == r.n_meals else f'{int(r.n_meals_shown)} of {int(r.n_meals)} meals'
    out += [sd.text(p.x0 + 14, p.y0 + 23, f'Example: one participant, {r.food}', size=F['body'], fill=sd.INK, weight='700'),
            sd.text(p.x0 + 14, p.y0 + 33, f'thin: each of {nm} · bold: their average', size=S, fill=sd.INK_SECONDARY)]
    return sd.group(out)


def panel_agreement(p, b):
    y0, y1 = math.floor(b.ci_low.min() * 10) / 10, math.ceil(b.ci_high.max() * 10) / 10
    y0 = round(y0 - 0.1, 1)          # room for the 'one meal' label under the k = 1 point
    x = sd.Linear(0.6, K_MAX + 0.4, p.px0, p.px1)
    y = sd.Linear(y0, y1, p.py1, p.py0)
    yt = [round(y0 + 0.1 * i, 1) for i in range(int(round((y1 - y0) * 10)) + 1)]
    out = [sd.axis_left(p, y, yt, 'Agreement between two averages (r)', fmt_fn=lambda v: f'{v:.1f}'),
           sd.axis_bottom(p, x, list(range(1, K_MAX + 1)), 'Meals of the same food averaged in each set (k)', fmt_fn=lambda v: f'{v:.0f}'),
           sd.line(p.px0, y(0.7), p.px1, y(0.7), stroke=sd.INK_MUTED, stroke_width=0.8, stroke_dasharray='3 2'),
           sd.text(p.px0 + 4, y(0.7) + 10, 'r = 0.70', size=S, fill=sd.INK_MUTED)]
    s = b.sort_values('k')
    out.append(sd.polyline([(x(r.k), y(r.r)) for r in s.itertuples()], C_I, 1.7))
    for r in s.itertuples():
        xx = x(r.k)
        out.append(sd.line(xx, y(r.ci_low), xx, y(r.ci_high), stroke=C_I, stroke_width=1.2))
        for yy in (y(r.ci_low), y(r.ci_high)):
            out.append(sd.line(xx - 2.5, yy, xx + 2.5, yy, stroke=C_I, stroke_width=1.2))
        out.append(sd.circle(xx, y(r.r), 2.9, fill=C_I, stroke=sd.SURFACE, **{'stroke-width': 0.8}))
    r1, rk = s.iloc[0], s.iloc[-1]
    out.append(sd.text(x(1) + 8, y(r1.r) + 13, f'one meal: r = {r1.r:.2f} ({r1.ci_low:.2f}–{r1.ci_high:.2f})', size=F['body'], fill=sd.INK, weight='600'))
    out.append(sd.text(x(K_MAX) + 6, y(rk.r) + 3, f'{rk.r:.2f}', size=F['body'], fill=C_I, weight='700'))
    out.append(sd.text(p.px0 + 8, y(y1) + 10, 'first k vs next k meals of the same food, in time order', size=S, fill=sd.INK_SECONDARY))
    out.append(sd.text(p.px1 - 2, p.py1 - 32 - NOTE_SHIFT, f'k = 1: {int(r1.n_pairs):,} person–food pairs, {int(r1.n_participants):,} participants',
                       size=S, fill=sd.INK_MUTED, anchor='end'))
    out.append(sd.text(p.px1 - 2, p.py1 - 22 - NOTE_SHIFT, f'k = {K_MAX}: {int(rk.n_pairs):,} pairs, {int(rk.n_participants):,} participants',
                       size=S, fill=sd.INK_MUTED, anchor='end'))
    out.append(sd.text(p.px1 - 2, p.py1 - 12 - NOTE_SHIFT, f'sets at k = {K_MAX}: median {float(rk.median_days_between_sets):.0f} days apart',
                       size=S, fill=sd.INK_MUTED, anchor='end'))
    return sd.group(out)


def panel_slope_schematic(x0, y0, w, h):
    """Idealized slope lines (arbitrary units; not data)."""
    out = [sd.text(x0, y0 + 2, 'Schematic, not data', size=S, fill=sd.INK_MUTED, weight='700'),
           sd.text(x0, y0 + 13, 'peak rise vs carbohydrate:', size=S, fill=sd.INK_SECONDARY),
           sd.text(x0, y0 + 23, 'the slope is carbohydrate sensitivity', size=S, fill=sd.INK_SECONDARY)]
    q = sd.Panel(x0, y0 + 26, w, h - 26, ml=16, mr=6, mt=10, mb=30)
    xg = sd.Linear(0, 110, q.px0, q.px1)
    yg = sd.Linear(0, 1.0, q.py1, q.py0)
    out += [sd.line(q.px0, q.py1, q.px1, q.py1, stroke=sd.INK_MUTED, stroke_width=0.9),
            sd.line(q.px0, q.py0, q.px0, q.py1, stroke=sd.INK_MUTED, stroke_width=0.9),
            sd.text(q.px0 - 5, (q.py0 + q.py1) / 2, 'peak rise', size=S, fill=sd.INK_SECONDARY, anchor='middle',
                    transform=f'rotate(-90,{sd.fmt(q.px0 - 5)},{sd.fmt((q.py0 + q.py1) / 2)})'),
            sd.text((q.px0 + q.px1) / 2, q.py1 + 22, 'carbohydrate (g)', size=S, fill=sd.INK_SECONDARY, anchor='middle')]
    for g in (30, 90):
        out.append(sd.line(xg(g), q.py1, xg(g), q.py1 + 3, stroke=sd.INK_MUTED))
        out.append(sd.text(xg(g), q.py1 + 11, f'{g}', size=S, fill=sd.INK_MUTED, anchor='middle'))
    for a1, a2, lab, col, dy, dx in ((0.34, 0.92, 'steep', C_HIGHS, 3, 16), (0.20, 0.30, 'shallow', C_LOWS, 6, 0)):  # peak rise at 30 g and 90 g
        b = (a2 - a1) / 60
        g0, g1 = 12, min(104, 30 + (0.98 - a1) / b)
        out.append(sd.line(xg(g0), yg(a1 + b * (g0 - 30)), xg(g1), yg(a1 + b * (g1 - 30)), stroke=col, stroke_width=1.6))
        for g, a in ((30, a1), (90, a2)):
            out.append(sd.circle(xg(g), yg(a), 2.6, fill=col, stroke=sd.SURFACE, **{'stroke-width': 0.8}))
        out.append(sd.text(xg(g1) - dx, yg(a1 + b * (g1 - 30)) - dy, lab, size=S, fill=col, weight='700', anchor='end'))
    return sd.group(out)


def panel_participants(p, ex, c):
    gap = 12
    w = (p.w - 2 * gap) / 3
    xmax = max(float(e['carbs'].max()) for e in ex)
    x_hi = 50 * math.ceil(xmax / 50)
    y_lo = 20 * math.floor(min(float(e['rise'].min()) for e in ex) / 20)
    y_hi = 20 * math.ceil(max(float(e['rise'].max()) for e in ex) / 20)
    out = []
    for i, e in enumerate(ex):
        r = c.iloc[i]
        q = sd.Panel(p.x0 + i * (w + gap), p.y0, w, p.h, ml=36 if i == 0 else 12, mr=6, mt=p.mt, mb=p.mb)
        x = sd.Linear(0, x_hi, q.px0, q.px1)
        y = sd.Linear(y_lo, y_hi, q.py1, q.py0)
        out.append(sd.axis_left(q, y, list(range(int(y_lo), int(y_hi) + 1, 40)), 'Peak rise (mg/dL)' if i == 0 else '',
                                fmt_fn=(lambda v: f'{v:.0f}'.replace('-', '−')) if i == 0 else (lambda v: '')))
        out.append(sd.axis_bottom(q, x, list(range(0, int(x_hi) + 1, 100)), 'Carbohydrate (g)', fmt_fn=lambda v: f'{v:.0f}'))
        out.append(sd.line(q.px0, y(0), q.px1, y(0), stroke=sd.RULE, stroke_width=0.8))
        for cx, ry in zip(e['carbs'], e['rise']):
            out.append(sd.circle(x(cx), y(ry), 1.5, fill=sd.BLUE_RAMP[3], opacity=0.55))
        xs = np.array([0, float(e['carbs'].max())])
        ys = e['intercept'] + e['acs'] * xs
        out.append(sd.line(x(xs[0]), y(ys[0]), x(xs[1]), y(ys[1]), stroke=sd.INK, stroke_width=1.8))
        out.append(sd.text(q.px0 + 2, q.py0 - 15, f"{r.participant.replace(' slope', '')} participant", size=F['body'], fill=sd.INK, weight='700'))
        out.append(sd.text(q.px0 + 2, q.py0 - 5, f'{r.slope_mg_dl_per_g:.2f} mg/dL per g · {int(r.n_meals)} meals', size=S, fill=sd.INK_SECONDARY))
    return sd.group(out)


def panel_distribution(p, acs, example_slopes, summ):
    edges, counts, _ = slope_histogram(acs)
    lo, hi = float(edges[0]), float(edges[-1])
    q10, q50, q90 = summ['10th percentile slope'], summ['median slope'], summ['90th percentile slope']
    x = sd.Linear(lo, hi, p.px0, p.px1)
    ymax = 100 * math.ceil(counts.max() * 1.5 / 100)
    y = sd.Linear(0, ymax, p.py1, p.py0)
    xt = [round(v, 1) for v in np.arange(math.ceil(lo * 2) / 2, hi + 1e-9, 0.5)]
    out = [sd.axis_left(p, y, sd.nice_ticks(0, ymax, 4), 'Participants', fmt_fn=lambda v: f'{v:,.0f}'),
           sd.axis_bottom(p, x, xt, 'Carbohydrate sensitivity (mg/dL per g)', fmt_fn=lambda v: f'{v:.1f}'.replace('-', '−')),
           sd.line(x(0), p.py0, x(0), p.py1, stroke=sd.INK_MUTED, stroke_width=1, stroke_dasharray='3 2')]
    for a, bb, n in zip(edges[:-1], edges[1:], counts):
        out.append(sd.rect(x(a) + 0.3, y(n), max(0.1, x(bb) - x(a) - 0.6), p.py1 - y(n), fill=sd.BLUE_RAMP[3]))
    for v, lab in ((q10, '10th'), (q50, '50th'), (q90, '90th')):
        j = min(len(counts) - 1, max(0, int(np.searchsorted(edges, v, side='right')) - 1))
        cbin = counts[max(0, j - 1):j + 2].max()                     # clear the neighbouring bars too
        ytop = y(counts.max() if lab == '50th' else cbin) - 6
        out.append(sd.line(x(v), ytop, x(v), p.py1, stroke=sd.INK, stroke_width=1.1, stroke_dasharray=None if lab == '50th' else '2 1.6'))
        out.append(sd.text(x(v), ytop - 3, lab, size=S, fill=sd.INK, anchor='middle', weight='600'))
    for v in example_slopes:
        xv = x(v)
        out.append(sd.path(f'M {sd.fmt(xv)} {sd.fmt(p.py1 + 1)} L {sd.fmt(xv - 3.5)} {sd.fmt(p.py1 + 7)} L {sd.fmt(xv + 3.5)} {sd.fmt(p.py1 + 7)} Z',
                           fill=sd.ACCENT, stroke=None))
    lines = [('10th · 50th · 90th percentile:', sd.INK, '700', F['body']),
             (f'{q10:.2f} · {q50:.2f} · {q90:.2f} mg/dL per g', sd.INK, '700', F['body']),
             (f"{int(summ['participants with a slope']):,} participants", sd.INK_SECONDARY, None, S),
             (f"axis: 0.5th–99.5th percentile ({int(summ['participants outside the 0.5th-99.5th percentile axis'])} not drawn)", sd.INK_SECONDARY, None, S),
             ('orange marks: the three participants in c', sd.ACCENT, None, S)]
    ty = p.py0 - 4
    for s_, col, wt, sz in lines:
        out.append(sd.text(p.px1, ty, s_, size=sz, fill=col, anchor='end', weight=wt))
        ty += 9.6
    return sd.group(out)


def panel_repeat(p, halves, summ):
    a, b = halves.acs_first.to_numpy(float), halves.acs_second.to_numpy(float)
    lo, hi, keep = odd_even_axes(a, b)
    x = sd.Linear(lo, hi, p.px0, p.px1)
    y = sd.Linear(lo, hi, p.py1, p.py0)
    t = [round(v, 1) for v in np.arange(lo, hi + 1e-9, 0.5)]
    fm = lambda v: f'{v:.1f}'.replace('-', '−')  # noqa: E731
    out = [sd.axis_left(p, y, t, 'Even-numbered meals', fmt_fn=fm),
           sd.axis_bottom(p, x, t, 'Odd-numbered meals (mg/dL per g)', fmt_fn=fm),
           sd.line(x(lo), y(lo), x(hi), y(hi), stroke=sd.INK_MUTED, stroke_width=1, stroke_dasharray='4 3')]
    for u, v in zip(a[keep], b[keep]):
        out.append(sd.circle(x(u), y(v), 0.9, fill=sd.CATEGORICAL[0], opacity=0.22))
    out.append(sd.rect(p.px0 + 2, p.py0 + 1, 116, 33, fill=sd.SURFACE, opacity=0.88))
    out.append(sd.text(p.px0 + 4, p.py0 + 9, f"r = {summ['odd/even agreement, Pearson r']:.2f} ({summ['odd/even r, 95% CI low']:.2f}–"
                                             f"{summ['odd/even r, 95% CI high']:.2f})", size=F['body'], fill=sd.INK, weight='600'))
    out.append(sd.text(p.px0 + 4, p.py0 + 19, f"{int(summ['participants in the odd/even comparison']):,} participants,", size=S, fill=sd.INK_SECONDARY))
    out.append(sd.text(p.px0 + 4, p.py0 + 29, f"≥{int(summ['minimum meals per half'])} meals per half", size=S, fill=sd.INK_SECONDARY))
    out.append(sd.text(p.px1 - 2, p.py1 - 4, f"{int(summ['odd/even pairs outside the axes'])} outside axes", size=S, fill=sd.INK_MUTED, anchor='end'))
    return sd.group(out)


def panel_quintiles(p, q5):
    q = sd.Panel(p.x0, p.y0, p.w, p.h, ml=48, mr=10, mt=34, mb=p.mb)
    y = sd.Linear(5.0, 6.5, q.py1, q.py0)
    xb = sd.Band([1, 2, 3, 4, 5], q.px0, q.px1, pad=0.3)
    out = [sd.axis_left(q, y, [5.0, 5.5, 6.0, 6.5], 'Fasting insulin', fmt_fn=lambda v: f'{v:.1f}'),
           sd.line(q.px0, q.py1, q.px1, q.py1, stroke=sd.RULE),
           sd.text(q.px0, q.py0 - 7, 'Geometric mean, µIU/mL (95% CI)', size=S, fill=sd.INK_MUTED)]
    for r in q5.itertuples():
        cx = xb.center(int(r.quintile))
        out.append(sd.line(cx, y(r.ci_low), cx, y(r.ci_high), stroke=C_IR, stroke_width=1.3))
        for yy in (y(r.ci_low), y(r.ci_high)):
            out.append(sd.line(cx - 3, yy, cx + 3, yy, stroke=C_IR, stroke_width=1.3))
        out.append(sd.circle(cx, y(r.fasting_insulin_geometric_mean), 3.0, fill=C_IR, stroke=C_IR, **{'stroke-width': 1.4}))
        lab = {1: 'Q1 (flattest)', 5: 'Q5 (steepest)'}.get(int(r.quintile), f'Q{int(r.quintile)}')
        out.append(sd.text(cx, q.py1 + 11, lab, size=S, fill=sd.INK_SECONDARY, anchor='middle'))
        out.append(sd.text(cx, q.py1 + 21, f'n = {int(r.n_participants):,}', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text((q.px0 + q.px1) / 2, q.py1 + 32, 'Quintile of carbohydrate sensitivity', size=S, fill=sd.INK_SECONDARY, anchor='middle'))
    return sd.group(out)


def panel_per_sd(x0, y0, w, e2):
    fmt = lambda r: f'{r.pct_per_sd:+.1f}% ({r.ci_low:+.1f} to {r.ci_high:+.1f})'.replace('-', '−')  # noqa: E731
    fi, hm = sd.pick(e2, marker='fasting_insulin'), sd.pick(e2, marker='homa_ir')
    items = [('Per SD of the slope (regression)', sd.INK, '700', F['head']),
             (f'Fasting insulin: {fmt(fi)}', sd.INK, '600', S), (f'n = {int(fi.n_participants):,}', sd.INK_MUTED, None, S),
             (f'HOMA-IR: {fmt(hm)}', sd.INK, '600', S), (f'n = {int(hm.n_participants):,}', sd.INK_MUTED, None, S),
             ('Both not detected (95% CI includes 0).', sd.INK_SECONDARY, None, S),
             (f'Unadjusted; slopes from meals within ±90 days of the laboratory draw ({int(fi.n_participants_with_slopes_near_lab_draw):,} participants).',
              sd.INK_SECONDARY, None, S)]
    out, ty = [], y0
    for i, (s_, col, wt, sz) in enumerate(items):
        for part in wrap(s_, w, sz, wt):
            out.append(sd.text(x0, ty, part, size=sz, fill=col, weight=wt))
            ty += 9.4
        ty += 0 if (i + 1 < len(items) and items[i + 1][0].startswith('n = ')) else 3
    return sd.group(out), ty


# ============================================================================= figure
def draw(out_dir):
    try:
        ex_traces = json.loads((EXEMPLARS / 'traces/fig6_trace_exemplars.json').read_text())
        examples = load_examples()
        acs = pd.read_parquet(QSP / '_cache/broad_mixed_participants.parquet', columns=['acs']).acs.to_numpy(float)
        halves = pd.read_parquet(QSP / '_cache/broad_mixed_halves.parquet', columns=['acs_first', 'acs_second'])
    except FileNotFoundError as err:
        cli.restricted_outputs_missing(f'{err.filename} not found: Fig. 6 shows individual participants (panels a, c and the scatter in d), whose '
                                       'data are not distributed; drawing it needs the analysis outputs.')
    a, b, c = source_data.read('fig6a_example'), source_data.read('fig6b_same_food_agreement'), source_data.read('fig6c_participants')
    summ = source_data.values('fig6d_summary')
    e1, e2 = source_data.read('fig6e_insulin_by_quintile'), source_data.read('fig6e_per_sd')
    ya = sd.TOP + 4
    pa_ = sd.Panel(6, ya, 238, 206, ml=44, mr=8, mt=42, mb=38)
    pb_ = sd.Panel(258, ya, 376, 206, ml=46, mr=34, mt=24, mb=38)
    yh = ya + 206 + 10
    yc = yh + 8
    pc_ = sd.Panel(170, yc, 464, 154, mt=42, mb=30)
    yd = yc + 154 + 12
    pd1 = sd.Panel(6, yd, 384, 178, ml=46, mr=8, mt=30, mb=36)
    pd2 = sd.Panel(404, yd, 230, 178, ml=46, mr=8, mt=30, mb=36)
    ye = yd + 178 + 12
    pe_ = sd.Panel(6, ye, 420, 124, ml=112, mr=4, mt=40, mb=34)
    g_er, yr_end = panel_per_sd(440, ye + 26, 194, e2)
    body = [sd.plabel(6, ya, 'a', 'The same food, meal after meal'), panel_example(pa_, ex_traces, a),
            sd.plabel(258, ya, 'b', 'How many meals of the same food agree'), panel_agreement(pb_, b),
            sd.line(6, yh, 634, yh, stroke=sd.GRID),
            sd.plabel(6, yc, 'c', 'Carbohydrate sensitivity, and three participants'), panel_slope_schematic(20, yc + 18, 132, 136),
            panel_participants(pc_, examples, c),
            sd.plabel(6, yd, 'd', 'Slopes differ between people and repeat across a person’s meals'), panel_distribution(pd1, acs, [e['acs'] for e in examples], summ),
            panel_repeat(pd2, halves, summ),
            sd.plabel(6, ye, 'e', 'Fasting insulin across quintiles of carbohydrate sensitivity: no gradient'), panel_quintiles(pe_, e1), g_er]
    return sd.write(sd.render_svg(max(ye + 124, yr_end) + 10, [], body, TITLE), out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
