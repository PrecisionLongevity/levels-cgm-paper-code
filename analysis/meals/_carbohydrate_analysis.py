"""Helper functions for canonical-window observational carbohydrate-response slope models.

Loads single-food meal rows from the macro-annotated meal substrate, fits an
ordinary-least-squares model of peak post-meal glucose rise (PEAK_RISE_2H) on
carbohydrate, fiber, fat, and protein grams per meal, and returns per-participant
adjusted (acs) and unadjusted (ucs) carbohydrate-sensitivity slopes together with
fit diagnostics. Also provides a split-half helper for reliability analyses.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from analysis.meals import _food_analysis as food
from analysis.meals import _shape_analysis as shape

VERSION='canonical-window-carbohydrate-slope'
MIN_MEALS=30
MIN_CARB_SD=5.
MACROS=['CARBS_G','FIBER_G','FAT_G','PROTEIN_G']


def load_macro_meals(path=food.DEFAULT_SUBSTRATE):
    food.validate_substrate(path)
    frame=pd.read_parquet(path)
    if frame.duplicated(['USER_ID','MEAL_EVENT_ID']).any():
        raise ValueError('Duplicate corrected meal keys')
    required=MACROS+['PEAK_RISE_2H','IAUC_2H']
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=required).copy()
    frame=frame.loc[frame[MACROS].ge(0).all(axis=1)]
    if not (frame.IS_ISOLATED_2_2.fillna(False)&frame.CGM_COVERAGE_OK.fillna(False)).all():
        raise ValueError('Macro meal rows contradict isolation/coverage metadata')
    return frame.sort_values(['USER_ID','MEAL_TS','MEAL_EVENT_ID'],kind='stable').reset_index(drop=True)


def load_slope_meals(path=food.DEFAULT_SUBSTRATE,panel=None,whole_tenure=False):
    frame=load_macro_meals(path)
    # Match the stated single-food definition, not merely a single macro event.
    frame=frame.loc[frame.N_LOGS_IN_CLUSTER.eq(1)&frame.N_MACRO_ROWS.eq(1)&frame.food_key.notna()].copy()
    if panel is None:
        panel=shape.load_panel()
    return shape.window_meals(frame,panel,whole_tenure=whole_tenure)


def fit_slope(frame,min_meals=MIN_MEALS,min_carb_sd=MIN_CARB_SD):
    if len(frame)<min_meals or frame.CARBS_G.std()<min_carb_sd:
        return None
    x=np.column_stack([np.ones(len(frame)),frame[MACROS].to_numpy(float)])
    y=frame.PEAK_RISE_2H.to_numpy(float)
    beta,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    if rank!=5:
        return None
    residual=y-x@beta
    total=np.sum((y-y.mean())**2)
    unadjusted=np.polyfit(frame.CARBS_G,y,1)
    unadjusted_residual=y-np.polyval(unadjusted,frame.CARBS_G)
    covariance=np.linalg.pinv(x.T@x)*(residual@residual)/(len(y)-5)
    return {'acs':float(beta[1]),'ucs':float(unadjusted[0]),'n_meals':len(frame),
            'carb_sd':float(frame.CARBS_G.std()),'slope_se_ols':float(np.sqrt(max(covariance[1,1],0))),
            'design_condition_number':float(np.linalg.cond(x)),
            'r2_in_sample':1-float(residual@residual)/total if total>0 else np.nan,
            'r2_univariate':1-float(unadjusted_residual@unadjusted_residual)/total if total>0 else np.nan,
            'intercept_at_mean_macros':float(beta[0]+beta[2:]@frame[MACROS[1:]].mean().to_numpy()),
            'carbs':float(frame.CARBS_G.mean()),'fiber':float(frame.FIBER_G.mean()),
            'carb_p10':float(frame.CARBS_G.quantile(.1)),'carb_p90':float(frame.CARBS_G.quantile(.9)),
            'carb_max':float(frame.CARBS_G.max())}


def participant_slopes(meals):
    rows=[]
    for user,frame in meals.groupby('USER_ID',sort=True):
        result=fit_slope(frame)
        if result is not None:
            rows.append({'USER_ID':user,**result})
    return pd.DataFrame(rows,columns=['USER_ID','acs','ucs','n_meals','carb_sd','slope_se_ols',
                                     'design_condition_number','r2_in_sample','r2_univariate','intercept_at_mean_macros',
                                     'carbs','fiber','carb_p10','carb_p90','carb_max'])


def split_half(meals):
    meals=meals.sort_values(['USER_ID','MEAL_TS','MEAL_EVENT_ID'],kind='stable').copy()
    meals['half']=meals.groupby('USER_ID').cumcount()%2
    first=participant_slopes(meals.loc[meals.half.eq(0)])[['USER_ID','acs']]
    second=participant_slopes(meals.loc[meals.half.eq(1)])[['USER_ID','acs']]
    return first.merge(second,on='USER_ID',suffixes=('_first','_second'),validate='one_to_one')
