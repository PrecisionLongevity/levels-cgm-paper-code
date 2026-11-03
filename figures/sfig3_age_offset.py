"""Supplementary Fig. 3: HbA1c by age decade at the same average CGM glucose.

Difference in HbA1c (percentage points, 95% CI) of each age decade relative to participants in their 30s, from a model
of HbA1c on average glucose and age decade.

Inputs
  outputs/clinical/age_offset.json    (analysis/clinical/clinical_summaries.py)
Source data
  source_data/sfig3_age_offset.csv    one row per age decade: participants, difference vs 30s and its 95% CI
Outputs
  sfig3_age_offset.svg / .png

Usage
  python figures/sfig3_age_offset.py source-data
  python figures/sfig3_age_offset.py draw
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import mpl_style as S  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

NAME = 'sfig3_age_offset'
TITLE = 'Supplementary Fig. 3 | HbA1c by age at the same average glucose'
DECADES = ['20', '30', '40', '50', '60', '70']


def make_source_data():
    a = json.loads((OUTPUTS / 'clinical/age_offset.json').read_text())
    rows = []
    for d in DECADES:
        est, lo, hi = a['vs30_at_cgm'][d] if d != '30' else (0.0, None, None)
        rows.append(dict(age_decade=f'{d}s', n_participants=a['age_counts'][d], hba1c_difference_vs_30s_pct_points=est,
                         ci_low=lo, ci_high=hi, reference=(d == '30')))
    return [source_data.write(NAME, pd.DataFrame(rows))]


def draw(out_dir):
    df = source_data.read(NAME)
    S.apply_style()
    fig = plt.figure(figsize=(5.0, 3.55))
    y_top = S.header(fig, TITLE)
    ax = fig.add_axes([0.14, 0.19, 0.82, y_top - 0.035 - 0.19])
    ax.axhline(0, color=S.INK, lw=0.6, zorder=1)
    ref_x = int(df.index[df.reference].tolist()[0])
    pts = df[~df.reference]
    xs, ys = list(pts.index), list(pts.hba1c_difference_vs_30s_pct_points)
    ax.errorbar(xs, ys, yerr=[pts.hba1c_difference_vs_30s_pct_points - pts.ci_low, pts.ci_high - pts.hba1c_difference_vs_30s_pct_points],
                fmt='o', ms=4.8, color=S.BLUE, capsize=0, elinewidth=1.0, mec='white', mew=0.5, zorder=3)
    ax.plot(xs[:1] + [ref_x] + xs[1:], ys[:1] + [0.0] + ys[1:], color=S.BLUE, lw=0.8, zorder=2)
    ax.plot([ref_x], [0], marker='o', ms=4.8, mfc='white', mec=S.INK, mew=0.9, ls='none', zorder=4)
    ax.text(ref_x, 0.025, 'Reference', ha='center', va='bottom', fontsize=S.SEC, color=S.INK)
    sixties = df[df.age_decade == '60s'].iloc[0]
    j60 = int(df.index[df.age_decade == '60s'][0])
    ax.text(j60 - 0.22, sixties.hba1c_difference_vs_30s_pct_points, f'{sixties.hba1c_difference_vs_30s_pct_points:.2f}',
            ha='right', va='center', fontsize=S.SEC, color=S.BLUE)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f'{r.age_decade}\nn = {S.fmt_n(r.n_participants)}' for r in df.itertuples()])
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(-0.2, 0.36)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(['−0.2', '−0.1', '0', '0.1', '0.2', '0.3'])
    ax.set_xlabel('Age decade (n)')
    ax.set_ylabel('Difference in HbA1c vs 30s at the same\naverage glucose, percentage points')
    return S.save(fig, out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
