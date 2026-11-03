"""Supplementary Fig. 2: how the HbA1c and average-glucose groups differ in body weight, diet and lifestyle.

For each measure, the difference of each group from the both-normal group (age- and sex-adjusted OLS, HC3 95% CI), and
the share of each group aged under 50 years (difference in percentage points, not age adjusted). Groups: CGM elevated
only (average glucose >=110 mg/dL, HbA1c <5.7%), HbA1c elevated only, both elevated; reference both normal.

Inputs
  outputs/metabolic/discordance_behavior_no_zero_fill/   differences, descriptives and coverage per measure
                                                         (analysis/metabolic/discordance_behavior_no_zero_fill.py)
  outputs/clinical/discordance_groups/age_under50.csv    share aged under 50 by group (analysis/clinical/discordance_groups.py)
Source data
  source_data/sfig2_discordant_groups.csv
Outputs
  sfig2_discordant_groups.svg / .png

Usage
  python figures/sfig2_discordant_groups.py source-data
  python figures/sfig2_discordant_groups.py draw
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import svgfig as sd  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402

NAME = 'sfig2_discordant_groups'
TITLE = 'Supplementary Fig. 2 | HbA1c–average glucose discordant groups differ in body weight, diet, and lifestyle'
PRINT_MM = 180.0
BEHAVIOR = OUTPUTS / 'metabolic/discordance_behavior_no_zero_fill'
AGE = OUTPUTS / 'clinical/discordance_groups/age_under50.csv'
GROUPS = {'both_normal': 'Both normal', 'cgm_only': 'CGM elevated only', 'a1c_only': 'HbA1c elevated only', 'both_elevated': 'Both elevated'}
COMPARED = ['cgm_only', 'a1c_only', 'both_elevated']
COLOR = {'cgm_only': sd.CATEGORICAL[0], 'a1c_only': sd.INK_MUTED, 'both_elevated': sd.INK_MUTED}
# (variable, unit, scale applied to the producer value, format of the both-normal median)
MEASURES = [('bmi', 'kg/m²', 1.0, '{:.1f}'), ('steps', 'steps/day', 1.0, '{:,.0f}'),
            ('ex_share', 'percentage points', 100.0, '{:.0f}%'), ('sleep_h', 'h/night', 1.0, '{:.2f} h'),
            ('sleep_mid', 'h', 1.0, '{:.2f} h'), ('alc_share', 'percentage points', 100.0, '{:.1f}%'),
            ('carbs_meal', 'g', 1.0, '{:.0f} g'), ('carb_share', 'percentage points', 100.0, '{:.1f}%'),
            ('fiber_meal', 'g', 1.0, '{:.1f} g'), ('late_share', 'percentage points', 100.0, '{:.1f}%')]
LABELS = {'BMI (kg/m²)': 'BMI', 'Sleep (h/night)': 'Sleep duration', 'Sleep midpoint (h)': 'Sleep midpoint',
          'Carbohydrate per meal (g)': 'Carbohydrate per meal', 'Fiber per meal (g)': 'Fiber per meal'}
AGE_MEASURE = 'Aged under 50 years'
AGE_TITLE = 'Aged under 50 years (percentage points; not age adjusted)'


def make_source_data():
    diff = pd.read_csv(BEHAVIOR / 'differences_vs_both_normal_no_zero_fill.csv')
    desc = pd.read_csv(BEHAVIOR / 'descriptives_no_zero_fill.csv')
    cov = pd.read_csv(BEHAVIOR / 'coverage_no_zero_fill.csv')
    rows = []
    for var, unit, scale, _ in MEASURES:
        d = diff.loc[diff.variable == var].set_index('group')
        label = d.label.iloc[0]
        label = LABELS.get(label, label)
        rows.append(dict(measure=label, unit=unit, group='all', n_participants=int(sd.pick(cov, variable=var, group='all').n)))
        rows.append(dict(measure=label, unit=unit, group='both_normal', median=sd.pick(desc, variable=var, group='both_normal')['median'] * scale))
        for g in COMPARED:
            r = d.loc[g]
            rows.append(dict(measure=label, unit=unit, group=g, difference_vs_both_normal=r['diff'] * scale,
                             ci_low=r.ci_low * scale, ci_high=r.ci_high * scale))
    a = pd.read_csv(AGE).set_index('group')
    unit = 'percentage points'
    rows.append(dict(measure=AGE_MEASURE, unit=unit, group='all', n_participants=int(a.loc['all', 'n']), share_pct=a.loc['all', 'pct_under50']))
    rows.append(dict(measure=AGE_MEASURE, unit=unit, group='both_normal', share_pct=a.loc['both_normal', 'pct_under50']))
    for g in COMPARED:
        r = a.loc[g]
        rows.append(dict(measure=AGE_MEASURE, unit=unit, group=g, difference_vs_both_normal=r.diff_pp_vs_both_normal,
                         ci_low=r.diff_lo, ci_high=r.diff_hi, share_pct=r.pct_under50))
    cols = ['measure', 'unit', 'group', 'n_participants', 'median', 'difference_vs_both_normal', 'ci_low', 'ci_high', 'share_pct']
    df = pd.DataFrame(rows)[cols]
    df['n_participants'] = df.n_participants.astype('Int64')
    return [source_data.write(NAME, df)]


def mini_forest(p, rows, title, subtitle):
    """One measure: difference of each group from both normal with its 95% CI (open marker when the CI includes 0)."""
    d = rows.set_index('group')
    vals = np.r_[d.loc[COMPARED, 'ci_low'].to_numpy(), d.loc[COMPARED, 'ci_high'].to_numpy(), 0.0]
    lo, hi = sd.nice_bounds(vals.min(), vals.max(), 0.12)
    ticks = [t for t in sd.nice_ticks(lo, hi, 3) if lo <= t <= hi]
    x = sd.Linear(lo, hi, p.px0, p.px1)
    big = max(abs(lo), abs(hi)) >= 10
    fmt_t = (lambda v: sd.sg(v, 0) if v else '0') if big else (lambda v: sd.sg(v, 1) if v else '0')
    out = [sd.line(p.px0, p.py1, p.px1, p.py1, stroke=sd.RULE), sd.line(x(0), p.py0 - 2, x(0), p.py1, stroke=sd.INK_MUTED, stroke_width=1)]
    for t in ticks:
        out.append(sd.line(x(t), p.py1, x(t), p.py1 + 3, stroke=sd.RULE))
        out.append(sd.text(x(t), p.py1 + 11, fmt_t(t), size=sd.S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text(p.x0 + 2, p.y0 + 8, title, size=8.0, fill=sd.INK, weight='600'))
    out.append(sd.text(p.x0 + 2, p.y0 + 18, subtitle, size=sd.S, fill=sd.INK_MUTED))
    step = p.ph / 3
    for i, g in enumerate(COMPARED):
        r = d.loc[g]
        yy = p.py0 + step * (i + 0.5)
        out.append(sd.text(p.px0 - 6, yy + 3, GROUPS[g], size=sd.S, fill=sd.INK if g == 'cgm_only' else sd.INK_SECONDARY, anchor='end'))
        det = sd.excl0(r.ci_low, r.ci_high)
        col = COLOR[g]
        out.append(sd.line(x(r.ci_low), yy, x(r.ci_high), yy, stroke=col, stroke_width=1.4))
        out.append(sd.circle(x(r.difference_vs_both_normal), yy, 2.9, fill=col if det else sd.SURFACE, stroke=col, **{'stroke-width': 1.2}))
        if pd.notna(r.share_pct):
            out.append(sd.text(x(r.ci_high) + 5, yy + 3, f'{r.share_pct:.0f}% under 50', size=sd.S, fill=col))
    return sd.group(out)


def draw(out_dir):
    df = source_data.read(NAME)
    head, top = sd.header(TITLE)
    cols = [(6, 314, 96), (324, 310, 96)]
    ch, y0 = 72, top + 2
    body = []
    for k, (var, unit, scale, fmt_ref) in enumerate(MEASURES):
        label = df.measure.unique()[k]
        rows = df[df.measure == label]
        g = rows.set_index('group')
        r_, cc = divmod(k, 2)
        cx, w, ml = cols[cc]
        mp = sd.Panel(cx, y0 + r_ * (ch + 6), w, ch, ml=ml, mr=8, mt=24, mb=15)
        subtitle = f"n={int(g.loc['all', 'n_participants']):,}; both normal median {fmt_ref.format(g.loc['both_normal', 'median'])}"
        body.append(mini_forest(mp, rows, f'{label} ({unit})', subtitle))
    ya_end = y0 + 5 * (ch + 6)
    age = df[df.measure == AGE_MEASURE].set_index('group')
    subtitle = (f"n={int(age.loc['all', 'n_participants']):,} aged 20–79 years; under 50: both normal "
                f"{age.loc['both_normal', 'share_pct']:.0f}%, all {age.loc['all', 'share_pct']:.0f}%")
    body.append(mini_forest(sd.Panel(cols[0][0], ya_end, cols[0][1], ch, ml=cols[0][2], mr=8, mt=24, mb=15),
                            df[df.measure == AGE_MEASURE], AGE_TITLE, subtitle))
    ya_end += ch + 6
    body.append(sd.legend_row(12, ya_end + 4, [('CGM elevated only', COLOR['cgm_only']), ('HbA1c elevated only / both elevated', sd.INK_MUTED)], size=sd.S))
    body.append(sd.text(634, ya_end + 5, 'filled: 95% CI excludes 0; open: not detected', size=sd.S, fill=sd.INK_MUTED, anchor='end'))
    height = round(ya_end + 18 + 4.0, 1)
    return sd.write(sd.render_svg(height, head, body, TITLE, print_mm=PRINT_MM), out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
