"""Fig. 3: average glucose tracks insulin resistance and other metabolic markers.

  a  share with HOMA-IR >=2.5 by quartile of average glucose (Wilson 95% CI), and the geometric-mean HOMA-IR ratio of
     the top vs the bottom quartile
  b  % difference in fasting insulin and HOMA-IR per SD of average glucose, without and with BMI, same participants
  c  % difference per SD of average glucose and per SD of HbA1c across metabolic markers, age- and sex-adjusted,
     same participants for both
  d  % difference in fasting insulin and HOMA-IR per SD of each CGM and logged-meal measure, unadjusted

Inputs (analysis outputs)
  metabolic/unadjusted_insulin_markers/  homa_by_glucose_quarter_unadjusted.csv, summary.json (a); bmi_same_participant.csv (b)
  clinical/markers/markers_avg_vs_a1c.csv                                   c
  clinical/mage_standard/headline_homa_insulin_mage_standard.csv             d (every row except spike width)
  clinical/spike_width/width_associations.csv                               d (spike width)
Source data
  source_data/fig3a_homa_by_quartile.csv, fig3b_bmi.csv, fig3c_average_glucose_vs_hba1c.csv, fig3d_unadjusted.csv
Outputs
  fig3_average_glucose.svg / .png

Usage
  python figures/fig3_average_glucose.py source-data
  python figures/fig3_average_glucose.py draw
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import svgfig as sd  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402
from cgmpaper.svgfig import F, S  # noqa: E402

NAME = 'fig3_average_glucose'
TITLE = 'Fig. 3 | Average glucose tracks insulin resistance and metabolic markers'
UIM = OUTPUTS / 'metabolic/unadjusted_insulin_markers'
CL = OUTPUTS / 'clinical'
HOMA_THRESHOLD = 2.5
SPIKE_THRESHOLD = 30      # mg/dL rise that defines a spike
MIN_MEALS = 10            # isolated logged meals per participant for the meal measures
MARKERS = [('fasting_insulin', 'Fasting insulin'), ('homa_ir', 'HOMA-IR')]
ROWS = [('level', 'mean_glucose'), ('var', 'sd_glucose'), ('var', 'modd'), ('var', 'cv_glucose'), ('var', 'mage'),
        ('spike', 'all_frequency'), ('spike', 'all_size'), ('spike', 'spike_width'), ('meal', 'iauc_2h'), ('meal', 'peak_rise')]
LABELS = {'mean_glucose': 'Average glucose', 'sd_glucose': 'SD', 'modd': 'MODD', 'cv_glucose': 'CV', 'mage': 'MAGE',
          'all_frequency': 'Spike frequency', 'all_size': 'Spike height', 'spike_width': 'Spike width',
          'iauc_2h': 'Meal 2-h iAUC', 'peak_rise': 'Meal peak rise'}
GROUPS = {'level': 'Level', 'var': 'Variability', 'spike': f'Spikes (CGM rises ≥{SPIKE_THRESHOLD} mg/dL, no meal log needed)',
          'meal': f'Logged meals (≥{MIN_MEALS} isolated meals)'}
C_AVG, C_HBA, C_BMI = sd.CATEGORICAL[0], sd.CATEGORICAL[2], sd.CATEGORICAL[3]
LAB_MARKERS = [('fasting_insulin', 'Fasting insulin'), ('homa_ir', 'HOMA-IR'), ('triglycerides', 'Triglycerides'),
               ('tg_hdl_ratio', 'Triglyceride/HDL ratio'), ('hdl_cholesterol', 'HDL cholesterol'), ('uric_acid', 'Uric acid'),
               ('hscrp', 'hsCRP'), ('apo_b', 'ApoB'), ('ldl_cholesterol', 'LDL cholesterol')]


# ============================================================================= source data
def make_source_data():
    q = pd.read_csv(UIM / 'homa_by_glucose_quarter_unadjusted.csv')
    q = q.loc[q.adjustment == 'none'].sort_values('quarter').reset_index(drop=True)
    s = json.loads((UIM / 'summary.json').read_text())
    r41 = s['homa_q4_vs_q1_unadjusted']
    a = pd.DataFrame(dict(quartile=q.quarter, average_glucose_min=q.mean_glucose_min, average_glucose_max=q.mean_glucose_max,
                          n_participants=q.n_quarter, n_homa_ir_ge_2_5=q.n_homa_ge_2_5, pct_homa_ir_ge_2_5=q.pct_homa_ge_2_5,
                          wilson_ci_low=q.pct_ci_low, wilson_ci_high=q.pct_ci_high))
    a['geometric_mean_ratio_vs_q1'] = [None, None, None, q.loc[3, 'ratio_vs_q1']]
    a['ratio_ci_low'] = [None, None, None, r41['ci'][0]]
    a['ratio_ci_high'] = [None, None, None, r41['ci'][1]]
    bm = pd.read_csv(UIM / 'bmi_same_participant.csv')
    rows = []
    for mk, _ in MARKERS:
        r = sd.pick(bm, family='cgm', comparison='without vs with BMI', marker=mk, measure='mean_glucose')
        for adj, sfx in (('without BMI', 'without'), ('with BMI', 'with')):
            rows.append(dict(marker=mk, n_participants=int(r.n_with), adjustment=adj, pct_per_sd=r[f'estimate_{sfx}'],
                             ci_low=r[f'ci_low_{sfx}'], ci_high=r[f'ci_high_{sfx}'], attenuation_pct=r.attenuation_pct))
    b = pd.DataFrame(rows)
    t = pd.read_csv(CL / 'markers/markers_avg_vs_a1c.csv').set_index('marker')
    c = pd.DataFrame([dict(marker=mk, n_participants=int(t.loc[mk, 'n_adj']), measure=meas, pct_per_sd=t.loc[mk, f'{p}_pct'],
                           ci_low=t.loc[mk, f'{p}_lo'], ci_high=t.loc[mk, f'{p}_hi'])
                      for mk, _ in LAB_MARKERS for meas, p in (('Average glucose', 'avg'), ('HbA1c', 'a1c'))])
    h = pd.read_csv(CL / 'mage_standard/headline_homa_insulin_mage_standard.csv')
    w = pd.read_csv(CL / 'spike_width/width_associations.csv')
    rows = []
    for g, m in ROWS:
        for mk, _ in MARKERS:
            if m == 'spike_width':
                r = w.loc[(w.marker == mk) & (w.measure == 'spike_width') & (w.compared_at_same == 'unadjusted')].iloc[0]
                n, e, lo, hi = int(r.n), r.pct, r.lo, r.hi
            else:
                r = sd.pick(h, adjustment='none', measure=m)
                n, e, lo, hi = int(r[f'n_{mk}']), r[f'estimate_{mk}'], r[f'ci_low_{mk}'], r[f'ci_high_{mk}']
            rows.append(dict(marker=mk, group=GROUPS[g], measure=LABELS[m], n_participants=n, pct_per_sd=e, ci_low=lo, ci_high=hi))
    d = pd.DataFrame(rows)
    return [source_data.write('fig3a_homa_by_quartile', a), source_data.write('fig3b_bmi', b),
            source_data.write('fig3c_average_glucose_vs_hba1c', c), source_data.write('fig3d_unadjusted', d)]


# ============================================================================= panels
def panel_a(pl, q):
    top = float(math.ceil(q.wilson_ci_high.max() / 10) * 10)
    y = sd.Linear(0, top, pl.py1, pl.py0)
    xb = sd.Band([1, 2, 3, 4], pl.px0, pl.px1, pad=0.40)
    out = [sd.axis_left(pl, y, list(range(0, int(top) + 1, 10)), f'% with HOMA-IR ≥{HOMA_THRESHOLD:g}', fmt_fn=lambda v: f'{v:.0f}'),
           sd.line(pl.px0, pl.py1, pl.px1, pl.py1, stroke=sd.RULE)]
    for i, r in enumerate(q.itertuples()):
        k = int(r.quartile)
        cx = xb.center(k)
        out.append(sd.rect(xb(k), y(r.pct_homa_ir_ge_2_5), xb.bandwidth, pl.py1 - y(r.pct_homa_ir_ge_2_5), fill=sd.BLUE_RAMP[3 + i], rx=1.5))
        out += sd.vbar_ci(cx, y, r.wilson_ci_low, r.wilson_ci_high)
        out.append(sd.text(cx, y(r.wilson_ci_high) - 5, f'{r.pct_homa_ir_ge_2_5:.1f}%', size=F['body'], fill=sd.INK, anchor='middle', weight='600'))
        lab = (f'<{r.average_glucose_max:.1f}' if k == 1 else
               (f'≥{r.average_glucose_min:.1f}' if k == 4 else f'{r.average_glucose_min:.1f}–{r.average_glucose_max:.1f}'))
        out.append(sd.text(cx, pl.py1 + 11, f'Q{k}', size=S, fill=sd.INK_SECONDARY, anchor='middle', weight='600'))
        out.append(sd.text(cx, pl.py1 + 21, lab, size=S, fill=sd.INK_MUTED, anchor='middle'))
        out.append(sd.text(cx, pl.py1 + 31, f'{int(r.n_homa_ir_ge_2_5)}/{int(r.n_participants)}', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text((pl.px0 + pl.px1) / 2, pl.py1 + 43, 'Quartile of average glucose (mg/dL)', size=F['body'], fill=sd.INK_SECONDARY, anchor='middle'))
    return out


def panel_b(x0, y0, w, b):
    lim = (0.0, float(math.ceil(b.ci_high.max() / 10) * 10))
    out = [sd.plabel(x0, y0, 'b', 'Partly shared with BMI')]
    out.append(sd.text(x0 + 14, y0 + 23, 'Average glucose, the same participants without and with BMI', size=S, fill=sd.INK_SECONDARY))
    kx = x0 + 14
    for lab, col in (('without BMI (unadjusted)', C_AVG), ('with BMI', C_BMI)):
        out.append(sd.dot(kx + 3, y0 + 33.4, col, True))
        out.append(sd.text(kx + 10, y0 + 36, lab, size=S, fill=sd.INK_SECONDARY))
        kx += 10 + sd.tw(lab, S) + 14
    p = sd.Panel(x0, y0 + 42, w, 118, ml=84, mr=40, mt=8, mb=34)
    x = sd.Linear(lim[0], lim[1], p.px0, p.px1)
    ticks = list(range(int(lim[0]), int(lim[1]) + 1, 10))
    out += [sd.line(x(t), p.py0, x(t), p.py1, stroke=sd.GRID, stroke_width=0.8) for t in ticks if t]
    out += [sd.axis_bottom(p, x, ticks, '', fmt_fn=lambda v: sd.sg(v, 0) if v else '0'),
            sd.line(x(0), p.py0, x(0), p.py1, stroke=sd.INK_MUTED, stroke_width=1),
            sd.text((p.px0 + p.px1) / 2, p.py1 + 25, '% difference per SD of average glucose', size=S, fill=sd.INK_SECONDARY, anchor='middle')]
    step = p.ph / len(MARKERS)
    for i, (mk, lab) in enumerate(MARKERS):
        yc = p.py0 + step * (i + 0.5)
        without, with_ = sd.pick(b, marker=mk, adjustment='without BMI'), sd.pick(b, marker=mk, adjustment='with BMI')
        out.append(sd.text(p.px0 - 8, yc, lab, size=F['body'], fill=sd.INK, anchor='end', weight='600'))
        out.append(sd.text(p.px0 - 8, yc + 10, f'n = {int(with_.n_participants):,}', size=S, fill=sd.INK_MUTED, anchor='end'))
        for dy, r, col in ((-7, without, C_AVG), (7, with_, C_BMI)):
            out += sd.ci_mark(x, lim, yc + dy, r.pct_per_sd, r.ci_low, r.ci_high, col)
            out.append(sd.text(x(r.ci_high) + 5, yc + dy + 3, sd.pct(r.pct_per_sd), size=S, fill=col, weight='700'))
    att = {mk: sd.pick(b, marker=mk, adjustment='with BMI').attenuation_pct for mk, _ in MARKERS}
    s = (f"With BMI, the estimate was {att['homa_ir']:.0f}% lower for HOMA-IR and {att['fasting_insulin']:.0f}% lower for fasting insulin "
         '(log scale) and stayed above 0.')
    els, yy = sd.wrap_lines(x0 + 14, p.y0 + p.h + 2, s, w - 18)
    return out + els, yy


def panel_c(x0, y0, c):
    out, under = [sd.plabel(x0, y0, 'c', 'Average glucose vs HbA1c across metabolic markers')], []
    out.append(sd.text(x0 + 14, y0 + 23, 'Per SD of each, adjusted for age and sex; the same participants for both', size=S,
                       fill=sd.INK_SECONDARY))
    kx = x0 + 14
    for lab, col in (('Average glucose', C_AVG), ('HbA1c', C_HBA)):
        out.append(sd.dot(kx + 3, y0 + 33.4, col, True))
        out.append(sd.text(kx + 10, y0 + 36, lab, size=S, fill=sd.INK_SECONDARY))
        kx += 10 + sd.tw(lab, S) + 14
    lab_r, n_r, ax0, ax1, v1, v2 = x0 + 118, x0 + 166, x0 + 180, x0 + 500, x0 + 566, x0 + 628
    lim = (float(math.floor(min(0.0, min(c.ci_low.min(), c.ci_high.min())) / 10) * 10), float(math.ceil(max(c.ci_low.max(), c.ci_high.max()) / 10) * 10))
    x = sd.Linear(lim[0], lim[1], ax0, ax1)
    yh = y0 + 50
    out.append(sd.text(n_r, yh, 'n', size=S, fill=sd.INK_MUTED, anchor='end', weight='600'))
    out.append(sd.text(v1, yh, 'Average glucose', size=S, fill=C_AVG, anchor='end', weight='600'))
    out.append(sd.text(v2, yh, 'HbA1c', size=S, fill=C_HBA, anchor='end', weight='600'))
    out.append(sd.line(x0, yh + 4, 634, yh + 4, stroke=sd.RULE))
    y, step, ytop = yh + 4, 16.0, yh + 6
    for mk, lab in LAB_MARKERS:
        y += step
        avg, a1c = sd.pick(c, marker=mk, measure='Average glucose'), sd.pick(c, marker=mk, measure='HbA1c')
        out.append(sd.text(lab_r, y + 3, lab, size=F['body'], fill=sd.INK, anchor='end', weight='600'))
        out.append(sd.text(n_r, y + 3, f'{int(avg.n_participants):,}', size=S, fill=sd.INK_MUTED, anchor='end'))
        marks = []
        for dy, r, col in ((-3.2, avg, C_AVG), (3.2, a1c, C_HBA)):
            marks += sd.ci_mark(x, lim, y + dy, r.pct_per_sd, r.ci_low, r.ci_high, col, r=2.5, caps=True)
        out.append(sd.group(marks))
        for vx, r, col in ((v1, avg, C_AVG), (v2, a1c, C_HBA)):
            ex = sd.excl0(r.ci_low, r.ci_high)
            out.append(sd.text(vx, y + 3, sd.pct(r.pct_per_sd), size=S, fill=col if ex else sd.INK_MUTED, anchor='end', weight='700' if ex else None))
    ybot = y + 10
    for t in range(int(lim[0]), int(lim[1]) + 1, 10):
        under.append(sd.line(x(t), ytop, x(t), ybot, stroke=sd.INK_MUTED if t == 0 else sd.GRID, stroke_width=0.9 if t == 0 else 0.8))
        out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
        out.append(sd.text(x(t), ybot + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.line(x(lim[0]), ybot, x(lim[1]), ybot, stroke=sd.RULE))
    out.append(sd.text((ax0 + ax1) / 2, ybot + 23, '% difference per SD (95% CI), adjusted for age and sex', size=S, fill=sd.INK_SECONDARY,
                       anchor='middle'))
    return under + out, ybot + 28


def panel_d(x0, y0, d):
    out, under = [], []
    col = C_AVG
    lim = (float(math.floor(d.ci_low.min() / 5) * 5), float(math.ceil(d.ci_high.max() / 5) * 5))
    lab_x, cw, fw = x0 + 150, 232, 176
    cols = [('fasting_insulin', lab_x + 12), ('homa_ir', lab_x + 12 + cw)]
    out.append(sd.plabel(x0, y0, 'd', 'Average glucose and other CGM and meal measures, unadjusted'))
    out.append(sd.text(x0 + 14, y0 + 23, 'Per SD of each measure, no adjustment', size=S, fill=sd.INK_SECONDARY))
    yh = y0 + 23 + 15
    for mk, cx in cols:
        out.append(sd.text(cx + fw / 2, yh, dict(MARKERS)[mk], size=F['body'], fill=col, anchor='middle', weight='600'))
    out.append(sd.line(x0, yh + 5, 634, yh + 5, stroke=sd.RULE))
    y, y_hdr, prev, spans = yh + 5, None, None, {}
    for g in d.group.unique():
        y += 17 if prev else 12
        out.append(sd.text(x0 + 2, y, g, size=S, fill=sd.INK_MUTED, weight='600'))
        for mk, cx in cols:
            v = sorted(set(d.loc[(d.group == g) & (d.marker == mk), 'n_participants']))
            out.append(sd.text(cx + cw - 8, y, 'n = ' + (f'{v[0]:,}' if len(v) == 1 else f'{v[0]:,}–{v[-1]:,}'), size=S,
                               fill=sd.INK_MUTED, anchor='end'))
        y += 1
        prev, y_hdr = g, y
        for meas in d.loc[d.group == g, 'measure'].unique():
            y += 14 if y_hdr == y else 13
            spans.setdefault(g, [y, y])[1] = y
            out.append(sd.text(lab_x, y + 3, meas, size=F['body'], fill=sd.INK, anchor='end', weight='600' if meas == 'Average glucose' else None))
            for mk, cx in cols:
                x = sd.Linear(lim[0], lim[1], cx, cx + fw)
                r = sd.pick(d, marker=mk, measure=meas)
                out += sd.ci_mark(x, lim, y, r.pct_per_sd, r.ci_low, r.ci_high, col, r=1.9, caps=True)
                ex = sd.excl0(r.ci_low, r.ci_high)
                out.append(sd.text(cx + cw - 8, y + 3, sd.pct(r.pct_per_sd), size=S, fill=col if ex else sd.INK_MUTED, anchor='end',
                                   weight='700' if ex else None))
    ybot = y + 9
    ticks = [t for t in range(int(math.ceil(lim[0] / 10) * 10), int(lim[1]) + 1, 10)]
    for mk, cx in cols:
        x = sd.Linear(lim[0], lim[1], cx, cx + fw)
        for gi, (y_a, y_b) in enumerate(spans.values()):   # vertical rules per row group, clear of the group labels
            y_end = ybot if gi == len(spans) - 1 else y_b + 7
            under.append(sd.line(x(0), y_a - 8, x(0), y_end, stroke=sd.INK_MUTED, stroke_width=0.9))
            under += [sd.line(x(t), y_a - 8, x(t), y_end, stroke=sd.GRID, stroke_width=0.8) for t in ticks if t]
        for t in ticks:
            out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
            out.append(sd.text(x(t), ybot + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
        out.append(sd.line(x(lim[0]), ybot, x(lim[1]), ybot, stroke=sd.RULE))
    out.append(sd.text((cols[0][1] + cols[1][1] + fw) / 2, ybot + 23, '% difference per SD of the measure (95% CI)', size=S,
                       fill=col, anchor='middle'))
    return under + out, ybot + 30


# ============================================================================= figure
def draw(out_dir):
    a, b = source_data.read('fig3a_homa_by_quartile'), source_data.read('fig3b_bmi')
    c, d = source_data.read('fig3c_average_glucose_vs_hba1c'), source_data.read('fig3d_unadjusted')
    yb = sd.TOP + 2
    q4 = a.iloc[3]
    body = [sd.plabel(6, yb, 'a', f'HOMA-IR ≥{HOMA_THRESHOLD:g} by quartile of average glucose'),
            sd.text(20, yb + 23, f'Unadjusted; {int(a.n_participants.sum()):,} participants with HOMA-IR; 95% CI: Wilson', size=S, fill=sd.INK_SECONDARY),
            sd.text(20, yb + 33, f'Geometric-mean HOMA-IR, Q4 vs Q1: {q4.geometric_mean_ratio_vs_q1:.2f} times '
                                 f'({q4.ratio_ci_low:.2f}–{q4.ratio_ci_high:.2f})', size=S, fill=sd.INK, weight='600')]
    pl = sd.Panel(6, yb + 36, 330, 198 - 36 + 12, ml=46, mr=4, mt=14, mb=50)
    body += panel_a(pl, a)
    g_b, yb_b = panel_b(352, yb, 282, b)
    body += g_b + [sd.line(344, yb + 2, 344, max(pl.y0 + pl.h, yb_b) - 2, stroke=sd.GRID)]
    yb_end = max(pl.y0 + pl.h, yb_b)
    body.append(sd.line(6, yb_end + 4, 634, yb_end + 4, stroke=sd.GRID))
    yc = yb_end + 12
    g_c, yc_end = panel_c(6, yc, c)
    body += g_c + [sd.line(6, yc_end + 2, 634, yc_end + 2, stroke=sd.GRID)]
    g_d, yd_end = panel_d(6, yc_end + 10, d)
    body += g_d
    body += sd.key_filled(20, yd_end + 4, col=C_AVG)
    return sd.write(sd.render_svg(yd_end + 14, [], body, TITLE), out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
