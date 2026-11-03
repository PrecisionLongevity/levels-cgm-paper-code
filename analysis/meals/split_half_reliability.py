#!/usr/bin/env python3
"""Empirical split-half reliability of same-food meal response averaging.

For each number of same-food meals averaged per half (k = 1..6), computes the
within-person correlation between two half-means of meal incremental AUC
(iAUC), using either consecutive or interleaved assignment of repeated
participant-food meals to the two halves. Each k's Spearman-Brown projection
uses the first-versus-second-response correlation on the same eligible
participant-food cells (cells with at least 2k observations), so the
projection is re-derived from the current data rather than assumed.
Interleaving reduces chronological separation between the two halves but does
not by itself rule out temporal drift. Confidence intervals come from a
participant-clustered percentile bootstrap.

Inputs: meal-level response substrate produced by
analysis/meals/build_meals.py (shared.DEFAULT_SUBSTRATE).

Outputs (OUTPUTS/meal_populations/, participant-level data aggregated to
summary statistics only):
  - empirical_split_half_reliability.csv
  - empirical_split_half_reliability_summary.md
  - empirical_split_half_reliability.metadata.json
  - fig3d_empirical_overlay.png

Usage: python analysis/meals/split_half_reliability.py
  [--meal-substrate PATH] [--output-dir PATH] [--bootstrap-reps N] [--k-max N]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.meals import _food_analysis as shared
K_MAX=6
N_BOOT=200
SEED=20260903
TARGET=.70


def spearman_brown(r1,k):
    k=np.asarray(k,dtype=float)
    # Nonpositive correlations do not define a nonnegative reliability curve.
    return k*r1/(1+(k-1)*r1) if np.isfinite(r1) and 0<=r1<=1 else np.full(k.shape,np.nan)


def halves(frame,k,split):
    if split not in ('consecutive','interleaved') or k<1:
        raise ValueError('Invalid split or number of meals per half')
    frame=frame.sort_values(['uid','food','ts','meal_id'],kind='stable').copy()
    frame['pos']=frame.groupby(['uid','food']).cumcount()
    frame['n_cell']=frame.groupby(['uid','food']).iauc.transform('size')
    data=frame.loc[frame.n_cell.ge(2*k)&frame.pos.lt(2*k)].copy()
    if data.empty:
        return pd.DataFrame(columns=['uid','food','a','b'])
    data['side']=np.where(data.pos.lt(k) if split=='consecutive' else data.pos.mod(2).eq(0),'a','b')
    return data.groupby(['uid','food','side']).iauc.mean().unstack('side').dropna().reset_index()


within_person_r=shared.within_person_r


def cluster_boot(h,rng,n_boot=N_BOOT):
    h=h.loc[h.groupby('uid').a.transform('size').ge(2)].copy()
    if h.uid.nunique()<2:
        return np.nan,np.nan
    h['a']=h.a-h.groupby('uid').a.transform('mean')
    h['b']=h.b-h.groupby('uid').b.transform('mean')
    h['aa'],h['bb'],h['ab']=h.a**2,h.b**2,h.a*h.b
    moments=h.groupby('uid')[['aa','bb','ab']].sum().to_numpy()
    draws=[]
    for _ in range(n_boot):
        total=moments[rng.integers(0,len(moments),size=len(moments))].sum(axis=0)
        if total[0]>0 and total[1]>0:
            draws.append(total[2]/np.sqrt(total[0]*total[1]))
    return tuple(np.quantile(draws,[.025,.975])) if draws else (np.nan,np.nan)


def evaluate(frame,k_max=K_MAX,n_boot=N_BOOT,seed=SEED):
    rng=np.random.default_rng(seed)
    base=halves(frame,1,'consecutive')
    reference=shared.within_person_r(base)[0]
    rows=[]
    for split in ('consecutive','interleaved'):
        for k in range(1,k_max+1):
            h=halves(frame,k,split)
            r,ncells,nusers=shared.within_person_r(h)
            low,high=cluster_boot(h,rng,n_boot) if ncells>=3 else (np.nan,np.nan)
            own=base.merge(h[['uid','food']],on=['uid','food'],validate='one_to_one')
            own_r=shared.within_person_r(own)[0]
            rows.append({'split':split,'k':k,'n_cells':ncells,'n_users':nusers,'r_empirical':r,
                         'ci_low':low,'ci_high':high,'r1_reference':reference,
                         'sb_from_current_r1':float(spearman_brown(reference,k)),
                         'r1_own_cells':own_r,'sb_from_own_r1':float(spearman_brown(own_r,k))})
    return pd.DataFrame(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meal-substrate',type=Path,default=shared.DEFAULT_SUBSTRATE)
    parser.add_argument('--output-dir',type=Path,default=shared.DEFAULT_OUT)
    parser.add_argument('--bootstrap-reps',type=int,default=N_BOOT)
    parser.add_argument('--k-max',type=int,default=K_MAX)
    args=parser.parse_args(argv)
    if args.bootstrap_reps<1 or args.k_max<1:
        raise ValueError('Positive bootstrap count and k-max required')
    frame=shared.load_responses(args.meal_substrate)
    result=evaluate(frame,args.k_max,args.bootstrap_reps)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output_dir/'empirical_split_half_reliability.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.4,4.2),dpi=200)
    reference=float(result.r1_reference.iloc[0])
    ks=np.arange(1,13)
    ax.plot(ks,spearman_brown(reference,ks),color='0.6',label=f'Spearman–Brown from current r₁ = {reference:.3f}')
    ax.axhline(TARGET,color='#d9822b',ls='--',lw=1)
    for split,color,marker in [('consecutive','#1f6fb2','o'),('interleaved','#2a9d8f','s')]:
        rows=result.loc[result.split.eq(split)&result.r_empirical.notna()]
        ax.vlines(rows.k,rows.ci_low,rows.ci_high,color=color,lw=1)
        ax.plot(rows.k,rows.r_empirical,marker=marker,color=color,label=f'Empirical {split}')
    ax.set(xlabel='Same-food meals averaged per half (k)',ylabel='Within-person correlation',xlim=(.5,12.5),ylim=(-1,1))
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir/'fig3d_empirical_overlay.png')
    plt.close(fig)
    lines=['# Empirical same-food averaging reliability','',
           f'{args.bootstrap_reps} participant-clustered percentile bootstrap resamples, seed{SEED}.',
           'Each k compares its two half-means and the first-versus-second reference on the same eligible cells; participants require at least two cells.',
           'The all-cohort Spearman–Brown curve is descriptive. The matched-cell curve tests that extrapolation under changing heavy-repeater selection.',
           '',result.round(5).to_string(index=False)]
    (args.output_dir/'empirical_split_half_reliability_summary.md').write_text('\n'.join(lines)+'\n')
    shared.write_metadata(args.output_dir/'empirical_split_half_reliability.metadata.json',substrate=args.meal_substrate,
                          bootstrap_reps=args.bootstrap_reps,
                          seed=SEED,k_max=args.k_max,r1_reference=reference,
                          bootstrap='participant-clustered; duplicated participants remain distinct; exact centered cluster moment sums')
    print(f'Empirical averaging rows {len(result)}; current k1 r {reference:.6f}')


if __name__=='__main__':
    main()
