#!/usr/bin/env python3
"""Crossed variance-components analysis of repeated single-food glucose responses.

Response = person + named food + person×food + occasion. EMS coefficients use
seeded Monte Carlo; signed method-of-moments diagnostics are retained, with
negative components truncated at zero for displayed shares. Disconnected designs
use their actual graph rank; unidentifiable four-component designs fail explicitly.

Within-participant permutation is a sensitivity: it destroys food-general as
well as person×food associations. The nonnegative observed-minus-null contrast
is not an identified heteroscedasticity correction or biological personalization.

Inputs
------
- Meal-response substrate produced by analysis/meals/build_meals.py (default
  path given by shared.DEFAULT_SUBSTRATE), containing per-occasion two-hour iAUC
  and peak-rise responses linked to participant id, normalized food name, and
  barcode.

Outputs
-------
- OUTPUTS/meal_populations/food_general_vs_personal_variance_components.csv
  (per-specification variance-component estimates and diagnostics)
- OUTPUTS/meal_populations/food_general_vs_personal_variance.metadata.json
- OUTPUTS/meal_populations/food_general_vs_personal_variance_summary.md

Usage
-----
python analysis/meals/food_variance.py [--meal-substrate PATH]
    [--output-dir PATH] [--traces N] [--null-reps N]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.meals import _food_analysis as shared
from analysis.meals._crossed_variance import (
    prepare_design,ems_coefficients,fit_from_coefficients,corrected_statistics,
)

SEED=20260615
N_TRACE=500
N_NULL=100


def pull(path=shared.DEFAULT_SUBSTRATE,identity='name'):
    return shared.load_responses(path,identity)


def restrict(frame,min_users,value_col):
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=[value_col]).copy()
    counts=frame.groupby('food').uid.nunique()
    return frame.loc[frame.food.isin(counts.index[counts.ge(min_users)])].copy()


def henderson3(data,col,n_trace=N_TRACE,seed=SEED):
    design=prepare_design(data.uid.to_numpy(),data.food.to_numpy())
    coefficients=ems_coefficients(design,n_trace,seed)
    fit=fit_from_coefficients(design,data[col].to_numpy(float),coefficients)
    result=variance_summary_columns(fit)
    result.update(N=design.n,nP=design.nperson,nF=design.nfood,nPF=design.ncell,
                  connected_components=design.connected_components,design_df=design.df.tolist())
    return result


def variance_summary_columns(fit):
    mapping={'P':'person','F':'food','PF':'person_food','E':'occasion'}
    result={f's2_{short}':fit[f's2_{long}'] for short,long in mapping.items()}
    result|={f'pct_{short}':fit[f'pct_{long}_raw'] for short,long in mapping.items()}
    result|={f's2_{short}_signed':fit[f's2_{long}_signed'] for short,long in mapping.items()}
    total=sum(result[f's2_{short}'] for short in mapping)
    pf,food,error=result['s2_PF'],result['s2_F'],result['s2_E']
    result.update(total=total,rel_personal=pf/(pf+error) if pf+error>0 else np.nan,
                  rel_food_raw=(food+pf)/(food+pf+error) if food+pf+error>0 else np.nan,
                  within_cell_check=0.,ems_condition_number=fit['ems_condition_number'],
                  n_boundary_components=fit['n_boundary_components'])
    return result


def permutation_estimates(design,y,coefficients,n_null=N_NULL,seed=SEED+7):
    order=np.argsort(design.person,kind='stable')
    groups=np.split(order,np.cumsum(design.person_count.astype(int))[:-1])
    values=[]
    for rep in range(n_null):
        rng=np.random.default_rng(seed+rep)
        shuffled=y.copy()
        for indices in groups:
            shuffled[indices]=rng.permutation(y[indices])
        values.append(fit_from_coefficients(design,shuffled,coefficients))
    return values


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meal-substrate',type=Path,default=shared.DEFAULT_SUBSTRATE)
    parser.add_argument('--output-dir',type=Path,default=shared.DEFAULT_OUT)
    parser.add_argument('--traces',type=int,default=N_TRACE)
    parser.add_argument('--null-reps',type=int,default=N_NULL)
    args=parser.parse_args(argv)
    if args.traces<1 or args.null_reps<1:
        raise ValueError('Positive Monte Carlo and permutation counts required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frames={identity:pull(args.meal_substrate,identity) for identity in ['name','barcode']}
    for frame in frames.values():
        frame['iauc_per_carb']=np.divide(frame.iauc,frame.carbs,
            out=np.full(len(frame),np.nan),where=frame.carbs.ge(1).to_numpy())
    specs=[('name',50,'iauc','IAUC_2H'),('name',20,'iauc','IAUC_2H'),
           ('name',50,'peak','PEAK_RISE_2H'),('name',20,'peak','PEAK_RISE_2H'),
           ('barcode',50,'iauc','IAUC_2H'),('barcode',20,'iauc','IAUC_2H'),
           ('name',50,'iauc_per_carb','IAUC_per_g_carb'),('name',20,'iauc_per_carb','IAUC_per_g_carb')]
    rows,skipped=[],[]
    for identity,threshold,col,label in specs:
        frame=restrict(frames[identity],threshold,col)
        if frame.empty:
            skipped.append({'identity':identity,'min_users':threshold,'response':label,'reason':'no eligible foods'})
            continue
        try:
            design=prepare_design(frame.uid.to_numpy(),frame.food.to_numpy())
        except ValueError as error:
            skipped.append({'identity':identity,'min_users':threshold,'response':label,'reason':str(error)})
            continue
        y=frame[col].to_numpy(float)
        coefficients=ems_coefficients(design,args.traces,SEED)
        fit=fit_from_coefficients(design,y,coefficients)
        null=permutation_estimates(design,y,coefficients,args.null_reps)
        null_values=np.array([item['s2_person_food'] for item in null])
        contrast=corrected_statistics(fit,null_values)
        sizes=frame.groupby(['uid','food']).size()
        row={'identity':identity,'min_users':threshold,'response':label,'method':'monte_carlo_ems_moments',
             'n_resp':len(frame),'n_users':design.nperson,'n_foods':design.nfood,'n_cells':design.ncell,
             'n_cells_ge2':int(sizes.ge(2).sum()),'within_df':int(design.df[3]),
             'connected_components':design.connected_components,'df_person':design.df[0],
             'df_food':design.df[1],'df_interaction':design.df[2],**variance_summary_columns(fit),
             'icc_user_food_raw':shared.icc1(frame.loc[frame.groupby(['uid','food'])[col].transform('size').ge(2)],col),
             's2_PF_null':float(null_values.mean()),'s2_PF_null_sd':float(null_values.std()),
             'pct_PF_null':float(np.nanmean([item['pct_person_food_raw'] for item in null])),
             'pct_PF_corrected':contrast['pct_person_food_corrected'],
             'rel_personal_corrected':contrast['single_meal_personal_reliability'],
             'null_excess_at_zero_boundary':contrast['null_excess_at_zero_boundary'],
             'null_contrast_interpretation':'exploratory observed-minus-permuted excess; not identified correction',
             'ems_traces':args.traces,'null_reps':args.null_reps}
        rows.append(row)
        print(f'{identity}, >= {threshold} users, {label}: {len(frame):,} responses; {design.nperson:,} participants',flush=True)
    result=pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No specification identifies all four crossed variance components')
    result.to_csv(args.output_dir/'food_general_vs_personal_variance_components.csv',index=False)
    shared.write_metadata(args.output_dir/'food_general_vs_personal_variance.metadata.json',substrate=args.meal_substrate,
                          seed=SEED,ems_traces=args.traces,null_reps=args.null_reps,skipped_specifications=skipped,
                          negative_component_policy='signed estimates retained; zero truncation for displayed shares',
                          null_interpretation='exploratory null-excess contrast, not an identified biological or heteroscedasticity correction')
    lines=['# Observational crossed food-response variance','',
           'Primary specification: normalized food names observed in at least50 participants; corrected positive two-hour iAUC. Sensitivities use at least20 participants, exact barcodes, peak rise, or iAUC per gram carbohydrate (carbohydrate>=1g).',
           'Monte Carlo EMS expectations are approximate. Signed estimates, zero-boundary diagnostics, connected design counts and condition numbers accompany the conventional nonnegative shares.',
           'Permutation destroys food-general and person×food associations. Its subtraction is an exploratory null-excess contrast; it does not identify a true personalization fraction or isolate heteroscedasticity.',
           'Portion, preparation, recording choices and unlogged intake remain uncontrolled. All results condition on observed repeated meals and eligible named foods.',
           '',result.round(5).to_string(index=False)]
    (args.output_dir/'food_general_vs_personal_variance_summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
