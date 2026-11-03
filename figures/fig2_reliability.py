"""Fig. 2: how reliably CGM measures agree from one week to another.

  a  schematic (idealized curves, not data): one person in two separate weeks, two weekly averages per person, agreement
     across people (intraclass correlation, ICC), and the gap between week starts
  b  week-to-week ICC(A,1) with 95% participant-bootstrap CIs for average glucose, SD, MODD and CV, pooled and by gap
     between week starts
  c  ICC(A,1) by days of wear in two windows 14 days apart

Inputs (analysis outputs)
  reliability/reliability_by_gap/reliability_by_gap.csv     b   (analysis/reliability/reliability_by_gap.py)
  reliability/reliability_by_window_length.csv               c   (analysis/reliability/repeatability.py)
Source data
  source_data/fig2b_reliability_by_gap.csv, fig2c_reliability_by_days_of_wear.csv
Outputs
  fig2_reliability.svg / .png

Usage
  python figures/fig2_reliability.py source-data
  python figures/fig2_reliability.py draw
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import svgfig as sd  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402
from cgmpaper.svgfig import F, S  # noqa: E402

NAME = 'fig2_reliability'
TITLE = 'Fig. 2 | Week-to-week agreement of CGM measures'
REL = OUTPUTS / 'reliability'
# (key, label, colour)
MEASURES = [('mean_glucose', 'Average glucose', sd.CATEGORICAL[0]), ('sd_glucose', 'SD', sd.CATEGORICAL[2]),
            ('modd', 'MODD', sd.CATEGORICAL[4]), ('cv_glucose', 'CV', sd.CATEGORICAL[3])]
KEYS = [m for m, *_ in MEASURES]
BINS = [('pooled', 'All gaps', 'pooled'), ('back_to_back_7d', 'Back-to-back', '(7 days)'), ('8_35d', '8–35', 'days'),
        ('36_90d', '36–90', 'days'), ('over_90d', '>90', 'days')]
C_W1, C_W2 = sd.CATEGORICAL[0], sd.CATEGORICAL[2]


def make_source_data():
    r = pd.read_csv(REL / 'reliability_by_gap/reliability_by_gap.csv')
    r = r.loc[r.metric.isin(KEYS) & r.gap_bin.isin([b for b, *_ in BINS])]
    b = r[['metric', 'gap_bin', 'gap_label', 'start_gap_min_def', 'start_gap_max_def', 'n', 'icc_a1', 'ci_low', 'ci_high']].rename(
        columns={'start_gap_min_def': 'gap_min_days', 'start_gap_max_def': 'gap_max_days', 'n': 'n_participants'})
    length = pd.read_csv(REL / 'reliability_by_window_length.csv')
    fg = length[(length.design == 'fixed_gap') & length.icc_a1.notna() & length.metric.isin(KEYS)]
    c = fg[['metric', 'days', 'n_users', 'icc_a1', 'ci_low', 'ci_high']].rename(columns={'days': 'days_of_wear', 'n_users': 'n_participants'})
    return [source_data.write('fig2b_reliability_by_gap', b), source_data.write('fig2c_reliability_by_days_of_wear', c)]


# ============================================================================= panel a: schematic
def ideal_trace(day):
    """Idealized glucose-like curve (not data): a daily rhythm plus three meal bumps; week 2 slightly higher."""
    h = (day % 1) * 24
    v = 100 + 3 * math.cos(2 * math.pi * (h - 4) / 24) + (3 if day >= 7 else 0)
    for c, a in ((8, 11), (13, 8), (19, 13)):
        v += a * (0.75 + 0.25 * math.sin(day * 1.7)) * math.exp(-0.5 * ((h - c) / 1.7) ** 2)
    return v


def panel_schematic(x0, y0, w):
    out = []
    top, bot = y0 + 44, y0 + 102
    # 1  one person, two weeks
    a0, a1 = x0 + 8, x0 + 212
    days = [14 * i / 700 for i in range(701)]
    vals = [ideal_trace(d) for d in days]
    y = sd.Linear(92, 124, bot, top)
    x = sd.Linear(0, 14, a0, a1)
    out.append(sd.text(a0, y0 + 28, '1  One person, two separate weeks', size=F['head'], fill=sd.INK, weight='700'))
    out.append(sd.rect(x(0), top, x(7) - x(0), bot - top, fill='#E3EEF8'))
    out.append(sd.rect(x(7), top, x(14) - x(7), bot - top, fill='#E1F3EC'))
    out.append(sd.polyline([(x(d), y(v)) for d, v in zip(days, vals)], sd.INK_SECONDARY, 0.8))
    for (a, b), col, lab in (((0, 7), C_W1, 'week 1 average'), ((7, 14), C_W2, 'week 2 average')):
        m = sum(v for d, v in zip(days, vals) if a <= d < b) / sum(1 for d in days if a <= d < b)
        out.append(sd.line(x(a), y(m), x(b), y(m), stroke=col, stroke_width=2.2))
        out.append(sd.text((x(a) + x(b)) / 2, bot + 11, lab, size=S, fill=col, anchor='middle', weight='600'))
    out.append(sd.line(a0, bot, a1, bot, stroke=sd.RULE))
    out.append(sd.arrow_h(a1 + 6, (top + bot) / 2, 20, stroke=sd.INK_MUTED, width=1.4, head=5))
    # 2  two averages per person
    b0, b1 = a1 + 34, a1 + 190
    out.append(sd.text(b0, y0 + 28, '2  Two averages per person', size=F['head'], fill=sd.INK, weight='700'))
    people = [(104, 106), (97, 96), (118, 115), (109, 112), (92, 94), (125, 127)]
    yb = sd.Linear(86, 132, bot, top)
    step = (b1 - b0) / len(people)
    out.append(sd.line(b0, bot, b1, bot, stroke=sd.RULE))
    for i, (w1, w2) in enumerate(people):
        cx = b0 + step * (i + .5)
        out.append(sd.line(cx - 5, yb(w1), cx + 5, yb(w2), stroke=sd.RULE, stroke_width=1.1))
        out.append(sd.circle(cx - 5, yb(w1), 3.2, fill=C_W1, stroke=sd.SURFACE, stroke_width=0.8))
        out.append(sd.circle(cx + 5, yb(w2), 3.2, fill=C_W2, stroke=sd.SURFACE, stroke_width=0.8))
    out.append(sd.text((b0 + b1) / 2, bot + 11, 'six people (week 1, week 2)', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.arrow_h(b1 + 6, (top + bot) / 2, 20, stroke=sd.INK_MUTED, width=1.4, head=5))
    # 3  agreement across people
    c0 = b1 + 34
    out.append(sd.text(c0, y0 + 28, '3  Agreement across people (ICC)', size=F['head'], fill=sd.INK, weight='700'))
    sx0 = c0 + 16
    sx1 = sx0 + (bot - top)
    ys, xs = sd.Linear(86, 132, bot, top), sd.Linear(86, 132, sx0, sx1)
    out.append(sd.line(sx0, bot, sx1, bot, stroke=sd.RULE))
    out.append(sd.line(sx0, top, sx0, bot, stroke=sd.RULE))
    out.append(sd.line(xs(86), ys(86), xs(132), ys(132), stroke=sd.INK_MUTED, stroke_dasharray='3 2'))
    for w1, w2 in people:
        out.append(sd.circle(xs(w1), ys(w2), 3.0, fill=sd.CATEGORICAL[0], stroke=sd.SURFACE, stroke_width=0.8))
    out.append(sd.text((sx0 + sx1) / 2, bot + 11, 'week 1', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text(sx0 - 5, (top + bot) / 2, 'week 2', size=S, fill=sd.INK_MUTED, anchor='middle',
                       transform=f'rotate(-90,{sd.fmt(sx0 - 5)},{sd.fmt((top + bot) / 2)})'))
    tx = sx1 + 12
    for i, (s, col, wt) in enumerate([('Near 1: points lie', sd.INK, '600'), ('close to the identity', sd.INK, '600'),
                                      ('line; people keep', sd.INK, '600'), ('their level.', sd.INK, '600'),
                                      ('Lower: week-to-week', sd.INK_SECONDARY, None), ('change is large', sd.INK_SECONDARY, None),
                                      ('relative to differences', sd.INK_SECONDARY, None), ('between people.', sd.INK_SECONDARY, None)]):
        out.append(sd.text(tx, top + 3 + 9.2 * i, s, size=S, fill=col, weight=wt))
    ex = 'ICC (intraclass correlation) = share of all variation in the weekly averages that reflects stable differences between people.'
    ye = bot + 27
    for s in sd.wrap(ex, w - 8, F['body'], '600'):
        out.append(sd.text(x0 + 8, ye, s, size=F['body'], fill=sd.INK, weight='600'))
        ye += 10
    return sd.group(out), ye - 4


def gap_strip(x0, y0):
    """Definition of the gap between week starts used in panel b."""
    out = [sd.text(x0 + 8, y0 + 10, '4  Gap between week starts (panel b)', size=F['head'], fill=sd.INK, weight='700')]
    txt = 'Days from the start of week 1 to the start of week 2: 7 days = back-to-back weeks; longer gaps leave days between the two weeks.'
    yy = y0 + 23
    for s in sd.wrap(txt, 300, S):
        out.append(sd.text(x0 + 8, yy, s, size=S, fill=sd.INK_SECONDARY))
        yy += 9.4
    d0, unit, yb = x0 + 340, 7.0, y0 + 22   # 7 units per day (diagram only)
    out.append(sd.rect(d0, yb, 7 * unit, 10, fill='#E3EEF8', stroke=C_W1, stroke_width=1.0, rx=2))
    out.append(sd.text(d0 + 3.5 * unit, yb + 20, 'week 1', size=S, fill=C_W1, anchor='middle', weight='600'))
    g0 = d0 + 7 * unit
    out.append(sd.line(g0 + 3, yb + 5, g0 + 12 * unit - 3, yb + 5, stroke=sd.RULE, stroke_width=1.0, stroke_dasharray='2 2'))
    out.append(sd.text(g0 + 6 * unit, yb + 20, 'days between', size=S, fill=sd.INK_MUTED, anchor='middle'))
    w2 = g0 + 12 * unit
    out.append(sd.rect(w2, yb, 7 * unit, 10, fill='#E1F3EC', stroke=C_W2, stroke_width=1.0, rx=2))
    out.append(sd.text(w2 + 3.5 * unit, yb + 20, 'week 2', size=S, fill=C_W2, anchor='middle', weight='600'))
    ya = yb - 6
    out.append(sd.line(d0, yb - 1, d0, ya - 3, stroke=sd.INK_MUTED, stroke_width=0.9))
    out.append(sd.line(w2, yb - 1, w2, ya - 3, stroke=sd.INK_MUTED, stroke_width=0.9))
    out.append(sd.arrow_h(d0 + 1, ya, w2 - d0 - 2, stroke=sd.INK_MUTED, width=1.0, head=3.0))
    out.append(sd.text((d0 + w2) / 2, ya - 4, 'gap', size=S, fill=sd.INK, anchor='middle', weight='700'))
    return sd.group(out), max(yy, yb + 24)


# ============================================================================= panels b and c
def panel_gap(p, b):
    t = {(r.gap_bin, r.metric): r for r in b.itertuples()}
    lo = math.floor(min(t[k, m].ci_low for k, *_ in BINS for m in KEYS) * 10) / 10
    hi = math.ceil(max(t[k, m].ci_high for k, *_ in BINS for m in KEYS) * 10) / 10
    y = sd.Linear(lo, hi, p.py1, p.py0)
    yt = [round(lo + 0.1 * i, 1) for i in range(int(round((hi - lo) * 10)) + 1)]
    out = [sd.axis_left(p, y, yt, 'Week-to-week ICC(A,1), 95% CI', fmt_fn=lambda v: f'{v:.1f}')]
    x = sd.Band([k for k, *_ in BINS], p.px0, p.px1, pad=0.12)
    xd = (x.center('pooled') + x.center('back_to_back_7d')) / 2
    out.append(sd.line(xd, p.py0, xd, p.py1 + 30, stroke=sd.RULE, stroke_width=0.8, stroke_dasharray='3 2'))
    dodge = min(8.5, x.bandwidth / 5.0)
    offs = {m: (i - 1.5) * dodge for i, m in enumerate(KEYS)}
    for m, _, col in MEASURES:
        pts = []
        for k, *_ in BINS:
            r = t[k, m]
            cx = x.center(k) + offs[m]
            filled = not (r.ci_low <= 0 <= r.ci_high)
            out.append(sd.line(cx, y(r.ci_low), cx, y(r.ci_high), stroke=col, stroke_width=1.4))
            out.append(sd.circle(cx, y(r.icc_a1), 3.2, fill=col if filled else sd.SURFACE, stroke=col if not filled else sd.SURFACE,
                                 stroke_width=0.8 if filled else 1.3))
            if k != 'pooled':
                pts.append((cx, y(r.icc_a1)))
        out.append(sd.polyline(pts, col, 0.9))
    for k, *_ in BINS:
        r = t[k, 'mean_glucose']
        out.append(sd.text(x.center(k) + offs['mean_glucose'] + 2.5, y(r.ci_low) + 10, f'{r.icc_a1:.2f}', size=S,
                           fill=MEASURES[0][2], anchor='end', weight='700'))
    yb = p.py1 + 11
    for k, l1, l2 in BINS:
        cx = x.center(k)
        out.append(sd.text(cx, yb, l1, size=S, fill=sd.INK, anchor='middle', weight='600'))
        out.append(sd.text(cx, yb + 9.4, l2, size=S, fill=sd.INK_SECONDARY, anchor='middle'))
        out.append(sd.text(cx, yb + 20.4, f"n {int(t[k, 'mean_glucose'].n_participants):,}", size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text((x.center('back_to_back_7d') + x.center('over_90d')) / 2, yb + 33, 'Gap between week starts',
                       size=F['body'], fill=sd.INK_SECONDARY, anchor='middle'))
    cx = p.px0 + 2
    for _, short, col in MEASURES:
        out.append(sd.circle(cx + 4, p.py0 - 15, 3.2, fill=col))
        out.append(sd.text(cx + 11, p.py0 - 12, short, size=S, fill=sd.INK_SECONDARY))
        cx += 11 + sd.tw(short, S) + 13
    return sd.group(out)


def panel_length(p, c):
    vals = [v for m in KEYS for v in c.loc[c.metric == m, ['ci_low', 'ci_high']].to_numpy().ravel()]
    ylo, yhi = math.floor(min(vals) * 10) / 10, math.ceil(max(vals) * 10) / 10
    dmax = int(c.days_of_wear.max())
    x = sd.Linear(0.5, dmax + 0.5, p.px0, p.px1)
    y = sd.Linear(ylo, yhi, p.py1, p.py0)
    yt = [round(ylo + 0.1 * i, 1) for i in range(int(round((yhi - ylo) * 10)) + 1)]
    out = [sd.axis_left(p, y, yt, 'ICC(A,1), 95% CI', fmt_fn=lambda v: f'{v:.1f}'),
           sd.axis_bottom(p, x, [1, 2, 4, 7, 10, 14], 'Days of wear in each window', fmt_fn=lambda v: f'{v:.0f}')]
    ends = []
    for m, _, col in MEASURES:
        s = c[c.metric == m].sort_values('days_of_wear')
        out.append(sd.band([(x(d), y(v)) for d, v in zip(s.days_of_wear, s.ci_low)], [(x(d), y(v)) for d, v in zip(s.days_of_wear, s.ci_high)], col, .13))
        out.append(sd.polyline([(x(d), y(v)) for d, v in zip(s.days_of_wear, s.icc_a1)], col, 1.6))
        for d, v in zip(s.days_of_wear, s.icc_a1):
            out.append(sd.circle(x(d), y(v), 2.0, fill=col, stroke=sd.SURFACE, stroke_width=0.6))
        last = s.iloc[-1]
        if m == 'mean_glucose':
            first = s.iloc[0]
            ly = y(first.icc_a1) - 30
            out.append(sd.line(x(first.days_of_wear), y(first.icc_a1) - 4, x(first.days_of_wear), ly + 3, stroke=col, stroke_width=0.8))
            out.append(sd.text(x(first.days_of_wear) - 3, ly, f'{first.icc_a1:.3f} at {int(first.days_of_wear)} day', size=S, fill=col, weight='700'))
        ends.append([y(last.icc_a1), f'{last.icc_a1:.3f}', col])
    ends.sort()
    for i in range(1, len(ends)):     # end labels kept at least 9.5 units apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 9.5)
    for yy, s_, col in ends:
        out.append(sd.text(x(dmax) + 7, yy + 3, s_, size=S, fill=col, weight='700'))
    cx = p.px0 - 30
    for _, short, col in MEASURES:
        out.append(sd.rect(cx, p.py0 - 19, 8, 8, fill=col, rx=1.6))
        out.append(sd.text(cx + 12, p.py0 - 12, short, size=S, fill=sd.INK_SECONDARY))
        cx += 12 + sd.tw(short, S) + 14
    return sd.group(out)


def draw(out_dir):
    b, c = source_data.read('fig2b_reliability_by_gap'), source_data.read('fig2c_reliability_by_days_of_wear')
    ya = sd.TOP + 4
    body = [sd.plabel(6, ya, 'a', 'What week-to-week agreement means: the same person measured in two separate weeks')]
    g_a, ya_end = panel_schematic(6, ya, 628)
    body.append(g_a)
    g_g, yg_end = gap_strip(6, ya_end + 4)
    body.append(g_g)
    body.append(sd.line(6, yg_end + 8, 634, yg_end + 8, stroke=sd.GRID))
    ybc, hbc = yg_end + 18, 300
    n_pool = int(sd.pick(b, gap_bin='pooled', metric='mean_glucose').n_participants)
    n_c = int(c.n_participants.max())
    pb = sd.Panel(6, ybc, 360, hbc, ml=40, mr=6, mt=52, mb=58)
    pc = sd.Panel(378, ybc, 256, hbc, ml=40, mr=34, mt=52, mb=58)
    body += [sd.plabel(6, ybc, 'b', 'Week-to-week agreement by gap between weeks'), panel_gap(pb, b),
             sd.text(20, ybc + 24, f'{n_pool:,} participants; two non-overlapping 7-day windows each', size=S, fill=sd.INK_SECONDARY),
             sd.plabel(378, ybc, 'c', 'Agreement by days of wear'), panel_length(pc, c),
             sd.text(392, ybc + 24, f'{n_c:,} participants with ≥42 dense days; windows 14 days apart', size=S, fill=sd.INK_SECONDARY)]
    yy = ybc + hbc + 4
    for s_ in sd.wrap('MODD needs ≥2 consecutive days.', 240, S):
        body.append(sd.text(392, yy, s_, size=S, fill=sd.INK_MUTED))
        yy += 9.4
    return sd.write(sd.render_svg(yy + 2 + 6, [], body, TITLE), out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
