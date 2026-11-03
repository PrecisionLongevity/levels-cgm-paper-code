#!/usr/bin/env python3
"""Per-participant carbohydrate-response slopes and their biomarker correlates.

Computes an adjusted carbohydrate slope (mg/dL glucose response per gram of
carbohydrate) for each eligible participant from corrected single-food meals,
requiring at least 30 such meals, complete nonnegative macros, carbohydrate
SD >=5 g, and a full-rank per-participant OLS. Primary meals must lie within
inclusive +/-90 local days of the canonical lab anchor, and the cohort excludes
participants with accepted diabetes-range labs ever. The slope is correlated
(Pearson/Spearman, FDR-corrected) against diet, glycemic-variability, glucose-level
and metabolic-panel measures, compared against HOMA-IR via a median-split
discordance table, assessed for odd/even split-half reliability with bootstrap
and leave-one-out diagnostics, and benchmarked in out-of-fold linear models
(participant folds, training-only predictor preprocessing) against glycemic
outcomes. Correlations are descriptive; a nonsignificant association does not
establish equivalence or that glucose responses do not matter.

Inputs
    - Meal substrate file (food.DEFAULT_SUBSTRATE), produced by
      analysis/meals/build_meals.py.
    - Canonical lab/CGM panel (shape.DEFAULT_PANEL), produced by
      analysis/canonical_panel/build_panel.py.

Outputs (written under OUTPUTS/meals, participant-level files stay private)
    - carb_sensitivity_correlations.csv: slope correlations with biomarkers.
    - carb_sensitivity_cv_models.csv: out-of-fold model comparisons.
    - carb_sensitivity_discordance.csv: carb-slope vs HOMA-IR median-split cells.
    - carb_sensitivity_reliability.csv: split-half reliability diagnostics.
    - carb_sensitivity_whole_tenure_sensitivity.csv (optional, with
      --whole-tenure-sensitivity): window vs whole-tenure slope comparison.
    - _cache/carb_sensitivity_participants.parquet: participant-level slopes
      merged with the canonical panel (private).
    - carb_sensitivity.metadata.json, carb_sensitivity_index_summary.md.

Usage
    python analysis/meals/carb_sensitivity_index.py [--meal-substrate PATH]
        [--canonical-panel PATH] [--output-dir PATH] [--whole-tenure-sensitivity]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.meals import _food_analysis as food
from analysis.meals import _shape_analysis as shape
from analysis.meals import _carbohydrate_analysis as carb
from cgmpaper.cross_validation import participant_folds,linear_oof_predictions,CV_DEFINITION_VERSION

CORRELATES=[('carbs','Habitual carb intake','Diet'),('fiber','Habitual fiber intake','Diet'),
            ('cv_glucose','Glucose CV','Glycemic variability'),('sd_glucose','Glucose SD','Glycemic variability'),
            ('skewness','Distribution skewness','Glycemic variability'),('kurtosis','Distribution kurtosis','Glycemic variability'),
            ('mean_glucose','Mean glucose','Glucose level'),('tar_140','Time above 140 mg/dL','Glucose level'),
            ('homa_ir','HOMA-IR','Metabolic panel'),('a1c','HbA1c','Metabolic panel'),
            ('tg_hdl_ratio','TG/HDL ratio','Metabolic panel'),('bmi','BMI','Metabolic panel'),('hscrp','hs-CRP','Metabolic panel')]


def correlate(frame):
    rows=[]
    for column,label,domain in CORRELATES:
        if column not in frame:
            continue
        paired=frame[['acs',column]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(paired)<3 or paired.acs.std()==0 or paired[column].std()==0:
            r,p,rho,ps=np.nan,np.nan,np.nan,np.nan
        else:
            r,p=stats.pearsonr(paired.acs,paired[column])
            rho,ps=stats.spearmanr(paired.acs,paired[column])
        rows.append({'variable':column,'label':label,'domain':domain,'n':len(paired),
                     'pearson_r':r,'pearson_p':p,'spearman_rho':rho,'spearman_p':ps})
    result=pd.DataFrame(rows)
    valid=result.pearson_p.notna()
    result['q_fdr']=np.nan
    if valid.any():
        result.loc[valid,'q_fdr']=stats.false_discovery_control(result.loc[valid,'pearson_p'])
    return result


def evaluate_models(frame,min_n=100):
    basic={'mean':['mean_glucose'],'mean_sd':['mean_glucose','sd_glucose'],
           'mean_slope':['mean_glucose','acs'],'mean_sd_slope':['mean_glucose','sd_glucose','acs']}
    cov=['age_bucket','sex_male','bmi']
    rows=[]
    for family in ['cgm','covariate_complete']:
        specs=basic.copy()
        if family=='covariate_complete':
            specs|={name+'_demographic':features+cov for name,features in basic.items()}
        required=sorted({column for values in specs.values() for column in values})
        for outcome,log in shape.OUTCOMES:
            sample=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=required+[outcome]).copy()
            if log:
                sample=sample.loc[sample[outcome].gt(0)]
            if len(sample)<min_n:
                continue
            folds=participant_folds(sample.USER_ID,5,42)
            y=np.log(sample[outcome].to_numpy()) if log else sample[outcome].to_numpy()
            for name,features in specs.items():
                prediction=linear_oof_predictions(sample[features],y,folds)
                rows.append({'family':family,'outcome':outcome,'model':name,'n':len(sample),
                             'cv_r2':r2_score(y,prediction),'cv_mae':mean_absolute_error(y,prediction),
                             'target_scale':'natural_log' if log else 'original',
                             'cv_definition_version':CV_DEFINITION_VERSION})
    return pd.DataFrame(rows,columns=['family','outcome','model','n','cv_r2','cv_mae','target_scale','cv_definition_version'])


def discordance(frame):
    sample=frame.loc[np.isfinite(frame.homa_ir)&frame.homa_ir.gt(0)].copy()
    if sample.empty:
        return pd.DataFrame(columns=['carb_group','homa_group','n']),{}
    cs,ir=sample.acs.median(),sample.homa_ir.median()
    hi_cs,hi_ir=sample.acs.gt(cs),sample.homa_ir.gt(ir)
    rows=[]
    for c in [True,False]:
        for h in [True,False]:
            rows.append({'carb_group':'high' if c else 'low','homa_group':'high' if h else 'low',
                         'n':int(((hi_cs==c)&(hi_ir==h)).sum())})
    return pd.DataFrame(rows),{'n':len(sample),'carb_median':float(cs),'homa_median':float(ir),
                              'pearson_r':food.pearson(sample.acs,sample.homa_ir)}


def reliability_summary(halves,n_boot=2000,seed=20260916):
    """Report raw and rank repeatability without choosing an outlier cutoff."""
    paired=halves[['acs_first','acs_second']].replace([np.inf,-np.inf],np.nan).dropna()
    x,y=paired.to_numpy(float).T
    result={'split':'chronological_odd_even','n_users':len(paired),'minimum_meals_per_half':carb.MIN_MEALS,
            'pearson_r':food.pearson(x,y),'spearman_rho':food.pearson(stats.rankdata(x),stats.rankdata(y)),
            'bootstrap_replicates':n_boot,'bootstrap_seed':seed}
    if len(paired)<4:
        return result
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(n_boot):
        index=rng.integers(len(x),size=len(x));a,b=x[index],y[index]
        samples.append([food.pearson(a,b),food.pearson(stats.rankdata(a),stats.rankdata(b))])
    boot=np.asarray(samples)
    for j,label in enumerate(['pearson','spearman']):
        finite=boot[:,j][np.isfinite(boot[:,j])]
        result[label+'_valid_bootstrap']=len(finite)
        if len(finite)==n_boot:
            result[label+'_ci_low'],result[label+'_ci_high']=map(float,np.quantile(finite,[.025,.975]))
        else:
            result[label+'_ci_low']=result[label+'_ci_high']=np.nan
    loo=[food.pearson(np.delete(x,i),np.delete(y,i)) for i in range(len(x))]
    result['pearson_leave_one_out_min']=float(np.nanmin(loo))
    result['pearson_leave_one_out_max']=float(np.nanmax(loo))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meal-substrate',type=Path,default=food.DEFAULT_SUBSTRATE)
    parser.add_argument('--canonical-panel',type=Path,default=shape.DEFAULT_PANEL)
    parser.add_argument('--output-dir',type=Path,default=food.DEFAULT_OUT)
    parser.add_argument('--whole-tenure-sensitivity',action='store_true')
    args=parser.parse_args(argv)
    panel=shape.load_panel(args.canonical_panel)
    meals=carb.load_slope_meals(args.meal_substrate,panel)
    traits=carb.participant_slopes(meals)
    if traits.empty:
        raise ValueError('No identifiable carbohydrate slopes in the canonical window')
    merged=traits.merge(panel,on='USER_ID',validate='one_to_one')
    halves=carb.split_half(meals)
    reliability_diagnostics=reliability_summary(halves)
    reliability=reliability_diagnostics['pearson_r']
    correlations=correlate(merged)
    comparisons=evaluate_models(merged)
    cells,discordance_summary=discordance(merged)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cache=args.output_dir/'_cache'
    cache.mkdir(exist_ok=True)
    merged.to_parquet(cache/'carb_sensitivity_participants.parquet',index=False)
    correlations.to_csv(args.output_dir/'carb_sensitivity_correlations.csv',index=False)
    comparisons.to_csv(args.output_dir/'carb_sensitivity_cv_models.csv',index=False)
    cells.to_csv(args.output_dir/'carb_sensitivity_discordance.csv',index=False)
    reliability_frame=pd.DataFrame([reliability_diagnostics])
    reliability_frame.to_csv(args.output_dir/'carb_sensitivity_reliability.csv',index=False)
    if args.whole_tenure_sensitivity:
        whole=carb.participant_slopes(carb.load_slope_meals(args.meal_substrate,panel,whole_tenure=True))
        paired=traits[['USER_ID','acs']].merge(whole[['USER_ID','acs']],on='USER_ID',suffixes=('_window','_whole'))
        pd.DataFrame([{'n':len(paired),'pearson_r':food.pearson(paired.acs_window,paired.acs_whole),
                       'trait_source':'whole_tenure_same_canonical_window_eligible_users_sensitivity'}]).to_csv(
                           args.output_dir/'carb_sensitivity_whole_tenure_sensitivity.csv',index=False)
    panel_metadata=shape.validate_publication_panel(args.canonical_panel)
    food.write_metadata(args.output_dir/'carb_sensitivity.metadata.json',substrate=args.meal_substrate,
                        cohort='no_diabetes_range',
                        carbohydrate_definition_version=carb.VERSION,
                        panels={'symmetric':{'filename':args.canonical_panel.name}},
                        canonical_panel_metadata=panel_metadata,n_users=len(traits),median_slope=float(traits.acs.median()),
                        n_window_meals=len(meals),minimum_meals=carb.MIN_MEALS,minimum_carb_sd=carb.MIN_CARB_SD,
                        full_rank_columns=5,reliability_n_users=len(halves),reliability_pearson=reliability,
                        reliability_diagnostics=reliability_diagnostics,
                        discordance=discordance_summary,trait_window='inclusive ±90 participant-local days of canonical anchor',
                        cv='five participant folds, seed42; training-only 0.5/99.5 predictor clipping/imputation/scaling; unclipped original/log endpoints',
                        longitudinal_scope='Historical future-inclusive branch excluded from the publication default.')
    lines=['# Canonical-window carbohydrate-response slopes','',
           f'Participants with an adjusted index: {len(traits):,}; median {traits.acs.median():.6f} mg/dL per gram carbohydrate.',
           f'Odd/even trait Pearson correlation: {reliability:.6f} across {len(halves):,} participants with at least 30 eligible meals per half.',
           'Both raw and rank correlations, participant-bootstrap intervals and leave-one-participant-out diagnostics are reported; no slope/outlier exclusion is applied based on these results.',
           str(reliability_diagnostics),
           'Corrected single-food responses, complete nonnegative macros, at least 30 meals, carbohydrate SD >=5 g and full-rank OLS. No missing macro is treated as zero.',
           'All primary meals and CGM covariates match the canonical ±90-day lab window; participants with accepted diabetes-range labs ever are excluded.',
           'Associations and OOF comparisons are observational. Slopes can depend on portions, chosen foods, recording patterns and estimation error. A null estimate does not establish equivalence or absence of biological importance.',
           '',correlations.round(5).to_string(index=False),'',comparisons.round(5).to_string(index=False),
           '', 'The historical future-inclusive longitudinal null is not reproduced as a publication claim.']
    (args.output_dir/'carb_sensitivity_index_summary.md').write_text('\n'.join(lines)+'\n')
    print(f'Canonical-window slope participants {len(traits):,}; reliability participants {len(halves):,}; OOF comparison rows {len(comparisons)}')


if __name__=='__main__':
    main()
