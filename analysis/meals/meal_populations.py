#!/usr/bin/env python3
"""Fit carbohydrate-sensitivity slopes and meal-shape reliability in population
definitions matched to each scientific question.

Populations compared: observed-record meals with no lab restriction
(broad_mixed/broad_single), meals restricted to the canonical lab-anchored
window using only participants with a lab draw (window_mixed/window_single),
and the same window computed across all participants with any lab draw
(all_labs_window_mixed/all_labs_window_single). '_single' variants restrict
to single-food, single-log meals as a sensitivity check; '_mixed' variants use
all qualifying meals. Within each population, participants must have at least
carb.MIN_MEALS meals with carbohydrate-intake standard deviation of at least
carb.MIN_CARB_SD before a per-participant carbohydrate-sensitivity slope (acs)
is fit; split-half reliability is computed by chronological odd/even halves.
Meal-shape reliability is assessed the same way (broad vs. canonical-window,
odd/even halves, >=30 meals, common learned basis, no prospective
validation). Context associations (context_maximal) regress each
glycemic-variability axis on each context exposure, adjusting for mean
glucose except when mean glucose is itself the axis, over the maximal
available sample per axis/exposure pair, with Benjamini-Hochberg FDR control
within each axis/adjustment group.

Inputs:
  - OUTPUTS/meals/meal_food_substrate.parquet (analysis/meals/build_meals.py)
  - OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv
    (analysis/canonical_panel/build_panel.py)
  - OUTPUTS/meals/meal_shape_substrate.parquet (analysis/meals/build_meals.py)
  - OUTPUTS/reliability/cache/context_primary.parquet
    (analysis/reliability/context.py)
  - helper modules analysis/meals/carb_sensitivity_index.py and
    analysis/meals/meal_shape_qualification.py (loaded dynamically)

Outputs (written to OUTPUTS/meal_populations/, participant-level files kept
private under _cache/):
  - carbohydrate_attrition.csv, shape_broad_reliability.csv,
    shape_window_reliability.csv, context_maximal.csv
  - per-scope *_participants.parquet, *_halves.parquet (private, _cache/),
    *_reliability.csv, *_correlations.csv, *_cv_models.csv
  - population_analysis.json (aggregate summary)

Usage: python analysis/meals/meal_populations.py [--output-dir DIR]
[--results-root DIR]
"""

from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis.meals import _carbohydrate_analysis as carb
from analysis.meals import _food_analysis as food
from analysis.meals import _shape_analysis as shape
from analysis.reliability import _context_analysis as context

VERSION = 'question-specific-populations'
DEFAULT_OUT = OUTPUTS / 'meal_populations'


def module(name):
    spec = importlib.util.spec_from_file_location('population_'+name[:3], Path(__file__).with_name(name))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def single_food(frame):
    return frame.loc[frame.N_LOGS_IN_CLUSTER.eq(1) & frame.N_MACRO_ROWS.eq(1) & frame.food_key.notna()].copy()


def attrition(frame, scope, stage):
    return dict(scope=scope, stage=stage, n_meals=len(frame), n_users=int(frame.USER_ID.nunique()))


def slope_attrition(meals, scope):
    """Sequential gates before fitting; no exclusions chosen from fitted slopes."""
    rows=[attrition(meals,scope,'eligible_meals')]
    counted=meals.loc[meals.groupby('USER_ID').USER_ID.transform('size').ge(carb.MIN_MEALS)]
    rows.append(attrition(counted,scope,'at_least_30_meals'))
    spread=counted.loc[counted.groupby('USER_ID').CARBS_G.transform('std').ge(carb.MIN_CARB_SD)]
    rows.append(attrition(spread,scope,'carbohydrate_sd_at_least_5g'))
    return rows


def context_maximal(data):
    """Keep reference scaling; select on this model's observed variables only."""
    rows=[]
    for axis in context.AXES:
        for exposure in context.CONTEXTS:
            cov=[] if axis=='mean_glucose' else ['mean_glucose_z']
            terms=[exposure+'_z']+cov
            sample=data.replace([np.inf,-np.inf],np.nan).dropna(subset=[axis+'_z']+terms)
            row=dict(axis=axis, context=exposure, adjustment='none' if not cov else 'mean_glucose_only',
                     population='maximal_per_axis_exposure', n_adjusted=len(sample),
                     status='insufficient_or_constant',adjusted_standardized_beta=np.nan,
                     ci_low=np.nan,ci_high=np.nan,p_value=np.nan)
            if len(sample)>=300 and sample[exposure+'_z'].nunique()>=3:
                x=sm.add_constant(sample[terms].astype(float),has_constant='add')
                if np.linalg.matrix_rank(x.to_numpy())==len(x.columns):
                    fit=sm.OLS(sample[axis+'_z'],x).fit(cov_type='HC3')
                    row.update(status='estimated',adjusted_standardized_beta=fit.params[exposure+'_z'],
                               ci_low=fit.conf_int().loc[exposure+'_z',0],ci_high=fit.conf_int().loc[exposure+'_z',1],
                               p_value=fit.pvalues[exposure+'_z'])
                else: row['status']='rank_deficient'
            rows.append(row)
    result=pd.DataFrame(rows);result['q_value']=np.nan
    for _,part in result.groupby(['axis','adjustment']):
        result.loc[part.index,'q_value']=context.bh(part.p_value)
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--results-root',type=Path,default=OUTPUTS)
    args=ap.parse_args(argv)
    out=args.output_dir
    source=args.results_root/'meals'
    panel_path=args.results_root/'canonical_panel/canonical_lab_cgm_panel.csv'
    substrate=source/'meal_food_substrate.parquet'
    all_meals=carb.load_macro_meals(substrate)
    primary=shape.load_panel(panel_path)
    all_panel=shape.load_panel(panel_path,cohort='all')
    slope=module('95_carb_sensitivity_index.py')
    qualification=module('118_meal_shape_qualification.py')
    out.mkdir(parents=True);(out/'_cache').mkdir()
    summary={}; flows=[]; outputs=[]
    def csv(frame,name):
        frame.to_csv(out/name,index=False);outputs.append(name)
    def parquet(frame,name):
        frame.to_parquet(out/'_cache'/name,index=False);outputs.append('_cache/'+name)
    scopes=[('broad_mixed',all_meals,None),('broad_single',single_food(all_meals),None),
            ('window_mixed',shape.window_meals(all_meals,primary),primary),
            ('window_single',shape.window_meals(single_food(all_meals),primary),primary),
            ('all_labs_window_mixed',shape.window_meals(all_meals,all_panel),all_panel),
            ('all_labs_window_single',shape.window_meals(single_food(all_meals),all_panel),all_panel)]
    for scope,meals,panel in scopes:
        print('Fitting '+scope,flush=True)
        flows+=slope_attrition(meals,scope)
        traits=carb.participant_slopes(meals)
        flows.append(attrition(meals.loc[meals.USER_ID.isin(traits.USER_ID)],scope,'full_rank_slope'))
        halves=carb.split_half(meals)
        reliability=slope.reliability_summary(halves)
        merged=traits if panel is None else traits.merge(panel,on='USER_ID',validate='one_to_one')
        parquet(merged,scope+'_participants.parquet')
        parquet(halves,scope+'_halves.parquet')
        csv(pd.DataFrame([reliability]),scope+'_reliability.csv')
        correlations=slope.correlate(merged)
        csv(correlations,scope+'_correlations.csv')
        if panel is not None: csv(slope.evaluate_models(merged),scope+'_cv_models.csv')
        summary[scope]=dict(n_users=len(traits),n_meals=len(meals),n_fitted_meals=int(traits.n_meals.sum()),
                            median_slope=float(traits.acs.median()),reliability=reliability,
                            scope='observed record; no lab restriction' if panel is None else 'inclusive +/-90 local days of lab anchor',
                            correlations=json.loads(correlations.to_json(orient='records',double_precision=15)))
        counts,edges=np.histogram(traits.acs,bins=40)
        summary[scope]['histogram']=dict(counts=counts.tolist(),edges=edges.tolist(),tail_clipping=False)
        print(json.dumps({k:summary[scope][k] for k in ['n_users','n_meals','median_slope']})+
              '; repeatability n='+str(reliability['n_users']),flush=True)
    csv(pd.DataFrame(flows),'carbohydrate_attrition.csv')
    del scopes,all_meals
    print('Broad shape repeatability',flush=True)
    scored=shape.load_scored_meals(source,source/'meal_shape_substrate.parquet')
    broad_shape=qualification.trait_split_half(scored)
    window_shape=qualification.trait_split_half(shape.window_meals(scored,primary))
    csv(broad_shape,'shape_broad_reliability.csv');csv(window_shape,'shape_window_reliability.csv')
    summary['shape']=dict(n_scored_meals=len(scored),n_scored_users=int(scored.USER_ID.nunique()),
                          broad=json.loads(broad_shape.to_json(orient='records',double_precision=15)),
                          window=json.loads(window_shape.to_json(orient='records',double_precision=15)),
                          split='chronological odd/even; >=30 meals total; common learned basis; no prospective validation')
    del scored
    folder=args.results_root/'reliability'
    data=pd.read_parquet(folder/'cache/context_primary.parquet')
    maximal=context_maximal(data)
    csv(maximal,'context_maximal.csv')
    summary['context']=dict(n_first_weeks=len(data),models=len(maximal),
                           n_min=int(maximal.n_adjusted.min()),n_max=int(maximal.n_adjusted.max()),
                           population='per-axis/exposure available cases; existing demographic-matched sensitivity retained')
    document=dict(version=VERSION,summary=summary)
    (out/'population_analysis.json').write_text(json.dumps(food.json_safe(document),indent=2,allow_nan=False)+'\n')
    print('Completed population analysis; aggregate summary written.',flush=True)


if __name__=='__main__': main()
