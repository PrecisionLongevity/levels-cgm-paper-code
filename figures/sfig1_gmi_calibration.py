"""Supplementary Fig. 1: HbA1c against average CGM glucose, with the published glucose management indicator (GMI).

Draws mean HbA1c per 5 mg/dL bin of average glucose (95% CI), this cohort's least-squares fit of HbA1c on average
glucose, and the published GMI line (Bergenstal et al., Diabetes Care 2018).

Inputs
  outputs/clinical/gmi_calibration/gmi_fits.json            fitted lines (analysis/clinical/gmi_calibration.py)
  outputs/clinical/gmi_calibration/gmi_bins_by_glucose.csv  binned means; not distributed (bins can hold as few as
                                                            10 participants), so drawing needs the analysis outputs
Source data
  source_data/sfig1_gmi_calibration.csv                     the two fitted lines (intercept, slope, 95% CI, n)
Outputs
  sfig1_gmi_calibration.svg / .png

Usage
  python figures/sfig1_gmi_calibration.py source-data
  python figures/sfig1_gmi_calibration.py draw
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import mpl_style as S  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

GMI_DIR = OUTPUTS / 'clinical/gmi_calibration'
NAME = 'sfig1_gmi_calibration'
TITLE = 'Supplementary Fig. 1 | HbA1c and average glucose: the cohort reproduces the published GMI slope'
MIN_BIN = 10  # bins with fewer participants are not drawn


def make_source_data():
    fits = json.loads((GMI_DIR / 'gmi_fits.json').read_text())
    fit, gmi = fits['hba1c_on_glucose'], fits['gmi_published']
    lo, hi = fits['drawn_ranges']['glucose_bins']
    rows = [dict(line='This cohort, least-squares fit of HbA1c on average glucose', intercept_pct=fit['intercept_pct'],
                 slope_pct_per_mg_dl=fit['slope_pct_per_mg_dl'], slope_ci_low=fit['slope_ci'][0], slope_ci_high=fit['slope_ci'][1],
                 n_participants=fits['n'], glucose_from_mg_dl=lo, glucose_to_mg_dl=hi),
            dict(line='Published GMI (Bergenstal et al. 2018)', intercept_pct=gmi['intercept_pct'],
                 slope_pct_per_mg_dl=gmi['slope_pct_per_mg_dl'], slope_ci_low=None, slope_ci_high=None,
                 n_participants=None, glucose_from_mg_dl=lo, glucose_to_mg_dl=hi)]
    return [source_data.write(NAME, pd.DataFrame(rows))]


def load_bins():
    p = GMI_DIR / 'gmi_bins_by_glucose.csv'
    if not p.is_file():
        cli.restricted_outputs_missing(f'{p} not found: the binned points of Supplementary Fig. 1 are not distributed; '
                                       'drawing this figure needs the analysis outputs.')
    return pd.read_csv(p)


def draw(out_dir):
    lines = source_data.read(NAME)
    fit, gmi = lines.iloc[0], lines.iloc[1]
    bins = load_bins()
    S.apply_style()
    fig = plt.figure(figsize=(6.0, 4.6))
    y_top = S.header(fig, TITLE)
    ax = fig.add_axes([0.11, 0.115, 0.86, y_top - 0.03 - 0.115])
    xr = (float(fit.glucose_from_mg_dl), float(fit.glucose_to_mg_dl))
    xlim, ylim = (xr[0] - 5, xr[1] + 5), (4.5, 8.5)
    ax.axhspan(ylim[0], 5.7, color='#EEF1F5', zorder=0, lw=0)
    ax.text(xlim[0] + 1.0, 5.66, 'HbA1c <5.7%', ha='left', va='top', fontsize=S.TICK, color=S.MUTED)
    xs = np.linspace(*xr, 100)
    ax.plot(xs, gmi.intercept_pct + gmi.slope_pct_per_mg_dl * xs, color=S.INK, lw=1.2, ls=(0, (4, 2.5)), zorder=2)
    ax.plot(xs, fit.intercept_pct + fit.slope_pct_per_mg_dl * xs, color=S.ORANGE, lw=1.4, zorder=3)
    ax.errorbar(bins.mean_glucose, bins.mean_hba1c, yerr=[bins.mean_hba1c - bins.ci_lo, bins.ci_hi - bins.mean_hba1c], fmt='s',
                ms=3.6, color=S.ORANGE, elinewidth=0.8, capsize=0, mec='white', mew=0.4, zorder=4)
    f4 = lambda v: f'{v:.4f}'  # noqa: E731
    lab_fit = ['This cohort, least-squares fit',
               f'{f4(fit.slope_pct_per_mg_dl)} points per mg/dL (95% CI {f4(fit.slope_ci_low)}–{f4(fit.slope_ci_high)})']
    lab_gmi = [f'Published GMI = {gmi.intercept_pct:g} + {gmi.slope_pct_per_mg_dl:g} × glucose',
               f'{f4(gmi.slope_pct_per_mg_dl)} points per mg/dL']
    ax.text(xr[1], 5.30, '\n'.join(lab_fit), ha='right', va='top', fontsize=S.TICK, color=S.ORANGE, linespacing=1.3)
    ax.text(xr[0] + 1.0, 6.75, '\n'.join(lab_gmi), ha='left', va='bottom', fontsize=S.TICK, color=S.INK, linespacing=1.3)
    note = (f'Points: mean HbA1c per 5 mg/dL of average glucose (95% CI), bins with ≥{MIN_BIN} participants; '
            f'n = {int(fit.n_participants):,}')
    ax.text(xlim[0] + 1.0, ylim[1] - 0.08, note, ha='left', va='top', fontsize=S.TICK, color=S.MUTED)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(80, int(xr[1]) + 1, 20))
    ax.set_yticks([5, 6, 7, 8])
    ax.set_xlabel('Average CGM glucose (mg/dL)')
    ax.set_ylabel('HbA1c (%)')
    return S.save(fig, out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
