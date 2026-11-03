"""Defines observed-week behavioral context variables and fits matched descriptive regressions
against glycemic axes.

Daily context (food logging, carbohydrate/fiber intake, steps, exercise, sleep, alcohol) is
computed per local calendar day over each participant's seven-day primary window and then
summarized per window. Context variables require a minimum number of valid observed days
(e.g. at least three days for carbohydrate/fiber, steps, or sleep hours) before a window-level
value is reported; otherwise the value is missing. Nearest paired BMI is taken from weight
measurements within 90 days of the window midpoint, combined with reported height, and
retained only within a plausible physiological range.

For each (glycemic axis, context variable) pair, an OLS regression (HC3 robust standard
errors) of the winsorized/standardized axis on the winsorized/standardized context variable is
fit, both unadjusted (controlling only for mean glucose when the axis is not mean glucose
itself) and adjusted for age, sex, BMI, and year. Models require at least 300 complete
participant-windows and at least three distinct context values, and are skipped if the design
matrix is rank-deficient. Benjamini-Hochberg q-values are computed within each
(axis, adjustment) group. Unadjusted Spearman correlations between each axis and each context
variable are also reported.

Inputs: primary_windows, macro_nutrients, food_logs, steps, lifestyle_events, weight, and users
tables via LEVELS_DB.

Outputs: a results DataFrame of regression estimates, a raw DataFrame of Spearman correlations,
a scales DataFrame of winsorization summaries, and the merged participant-window DataFrame,
returned to the calling module for further output.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from cgmpaper.local_exposure_dates import event_day_sql, sleep_midpoint_sql

VERSION='observed-week-context-fixed-local'
AXES=['mean_glucose','sd_glucose','modd','ac_var']
CONTEXTS=['food_logs','food_log_days','carbs_g_day','fiber_g_day','fiber_carb_ratio',
          'mean_steps','exercise_days','exercise_minutes_exercise_day',
          'mean_sleep_hours','sleep_midpoint_sd','alcohol_days','alcohol_drinks']


def daily_context(con):
    """primary_windows contains one seven-local-day window per participant.

    Event dates are recomputed from UTC timestamps on the CGM fixed-offset clock.
    Sleep requires a wake timestamp; source EVENT_DATE/DOSE_NUMERIC are not used
    for sleep day or midpoint. Food uses the same fixed participant offset.
    """
    return con.execute(f'''
      WITH dates AS (
        SELECT w.USER_ID,w.window_id,d.DAY,w.TZ_OFFSET_HOURS
        FROM primary_windows w CROSS JOIN LATERAL
          generate_series(w.window_start,w.window_end-INTERVAL '1 day',INTERVAL '1 day') d(DAY)
      ), items AS (
        SELECT m.USER_ID,m.LOG_ID,
          CASE WHEN m.CARBS BETWEEN 0 AND 1500 THEN m.CARBS END AS carbs,
          CASE WHEN m.FIBER BETWEEN 0 AND 200 THEN m.FIBER END AS fiber
        FROM macro_nutrients m JOIN primary_windows w USING(USER_ID)
      ), macros AS (
        SELECT USER_ID,LOG_ID,
          CASE WHEN count(carbs)=count(*) AND count(fiber)=count(*) THEN sum(carbs) END AS carbs,
          CASE WHEN count(carbs)=count(*) AND count(fiber)=count(*) THEN sum(fiber) END AS fiber
        FROM items GROUP BY USER_ID,LOG_ID
      ), food AS (
        SELECT f.USER_ID,cast(f.TIMESTAMP+w.TZ_OFFSET_HOURS*INTERVAL '1 hour' AS DATE) AS DAY,
          count(*) AS food_logs,
          CASE WHEN count(m.carbs)=count(*) THEN sum(m.carbs) END AS carbs,
          CASE WHEN count(m.fiber)=count(*) THEN sum(m.fiber) END AS fiber
        FROM food_logs f JOIN primary_windows w USING(USER_ID)
        LEFT JOIN macros m ON m.USER_ID=f.USER_ID AND m.LOG_ID=f.ID
        WHERE f.TYPE IN ('food','food_with_macros') AND f.TIMESTAMP IS NOT NULL
        GROUP BY f.USER_ID,DAY
      ), step_days AS (
        SELECT s.USER_ID,s.DAY,
          CASE WHEN count(TOTAL_STEPS)=count(*) AND min(TOTAL_STEPS)>=0
            AND isfinite(max(TOTAL_STEPS)) AND min(TOTAL_STEPS)=max(TOTAL_STEPS)
            THEN max(TOTAL_STEPS) END AS steps
        FROM steps s JOIN primary_windows w USING(USER_ID) GROUP BY s.USER_ID,s.DAY
      ), events AS (
        SELECT DISTINCT e.USER_ID,e.TYPE,{event_day_sql('e','w.TZ_OFFSET_HOURS')} AS DAY,e.START_TS,e.END_TS,
          e.DURATION_MIN,CASE WHEN e.TYPE='sleep' THEN {sleep_midpoint_sql('e','w.TZ_OFFSET_HOURS')}
            ELSE e.DOSE_NUMERIC END AS DOSE_NUMERIC,e.DOSE_UNIT,e.SOURCE_APP
        FROM lifestyle_events e JOIN primary_windows w USING(USER_ID)
        WHERE e.TYPE IN ('exercise','sleep','alcohol') AND
          {event_day_sql('e','w.TZ_OFFSET_HOURS')}>=w.window_start AND
          {event_day_sql('e','w.TZ_OFFSET_HOURS')}<w.window_end
      ), exercise AS (
        SELECT USER_ID,DAY,count(*) AS exercise_logs,
          CASE WHEN count(*) FILTER (WHERE DURATION_MIN>0 AND DURATION_MIN<=720)=count(*)
            THEN sum(DURATION_MIN) END AS exercise_minutes
        FROM events WHERE TYPE='exercise' GROUP BY USER_ID,DAY
      ), sleep_valid AS (
        SELECT *,max(DURATION_MIN)OVER(PARTITION BY USER_ID,DAY) AS longest
        FROM events WHERE TYPE='sleep' AND DURATION_MIN BETWEEN 120 AND 840
      ), sleep AS (
        SELECT USER_ID,DAY,max(DURATION_MIN)/60. AS sleep_hours,
          CASE WHEN count(*)=count(DOSE_NUMERIC) AND min(DOSE_NUMERIC)>=0 AND max(DOSE_NUMERIC)<24
            AND min(DOSE_NUMERIC)=max(DOSE_NUMERIC) THEN max(DOSE_NUMERIC) END AS sleep_midpoint
        FROM sleep_valid WHERE DURATION_MIN=longest GROUP BY USER_ID,DAY
      ), alcohol AS (
        SELECT USER_ID,DAY,count(*) AS alcohol_logs,
          CASE WHEN count(DOSE_NUMERIC)=count(*) AND min(DOSE_NUMERIC)>=0 AND isfinite(max(DOSE_NUMERIC))
            AND count(*) FILTER (WHERE DOSE_UNIT='drinks')=count(*) THEN sum(DOSE_NUMERIC) END AS alcohol_units
        FROM events WHERE TYPE='alcohol' GROUP BY USER_ID,DAY
      ) SELECT d.USER_ID,d.window_id,d.DAY,coalesce(f.food_logs,0) AS food_logs,
          f.carbs,f.fiber,s.steps,coalesce(e.exercise_logs,0) AS exercise_logs,e.exercise_minutes,
          sl.sleep_hours,sl.sleep_midpoint,coalesce(a.alcohol_logs,0) AS alcohol_logs,
          CASE WHEN a.alcohol_logs IS NULL THEN 0 ELSE a.alcohol_units END AS alcohol_units
        FROM dates d LEFT JOIN food f USING(USER_ID,DAY) LEFT JOIN step_days s USING(USER_ID,DAY)
        LEFT JOIN exercise e USING(USER_ID,DAY) LEFT JOIN sleep sl USING(USER_ID,DAY)
        LEFT JOIN alcohol a USING(USER_ID,DAY) ORDER BY d.USER_ID,d.DAY
    ''').df()


def circular_sd(hours):
    a=np.asarray(hours,float);a=a[np.isfinite(a)]
    if len(a)<3:return np.nan
    resultant=abs(np.mean(np.exp(2j*np.pi*a/24)))
    if resultant<1e-12:return np.nan
    return float(np.sqrt(max(0,-2*np.log(min(1,resultant))))*24/(2*np.pi))


def summarize_context(daily):
    if daily.duplicated(['window_id','DAY']).any():raise ValueError('Duplicate context dates')
    if not daily.groupby('window_id').size().eq(7).all():raise ValueError('Context requires all seven dates')
    rows=[]
    for window,days in daily.groupby('window_id',sort=False):
        nutrient=days.loc[days.carbs.notna()&days.fiber.notna()]
        row=dict(window_id=window,food_logs=int(days.food_logs.sum()),food_log_days=int(days.food_logs.gt(0).sum()),
                 nutrient_days=len(nutrient),step_days=int(days.steps.notna().sum()),
                 sleep_days=int(days.sleep_hours.notna().sum()),midpoint_days=int(days.sleep_midpoint.notna().sum()),
                 exercise_days=int(days.exercise_logs.gt(0).sum()),
                 exercise_duration_days=int(days.exercise_minutes.notna().sum()),
                 alcohol_days=int(days.alcohol_logs.gt(0).sum()),
                 carbs_g_day=np.nan,fiber_g_day=np.nan,fiber_carb_ratio=np.nan,
                 mean_steps=float(days.steps.mean()) if days.steps.notna().sum()>=3 else np.nan,
                 exercise_minutes_exercise_day=float(days.exercise_minutes.mean()),
                 mean_sleep_hours=float(days.sleep_hours.mean()) if days.sleep_hours.notna().sum()>=3 else np.nan,
                 sleep_midpoint_sd=circular_sd(days.sleep_midpoint),
                 alcohol_drinks=float(days.alcohol_units.sum()) if days.alcohol_units.notna().all() else np.nan)
        if len(nutrient)>=3:
            row.update(carbs_g_day=float(nutrient.carbs.mean()),fiber_g_day=float(nutrient.fiber.mean()),
                       fiber_carb_ratio=float(nutrient.fiber.sum()/nutrient.carbs.sum()) if nutrient.carbs.sum()>0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def nearest_bmi(con):
    return con.execute('''WITH weights AS (
        SELECT x.USER_ID,cast(x.DAY AS DATE) AS DAY,median(x.WEIGHT) AS weight_lb
        FROM weight x JOIN primary_windows w USING(USER_ID)
        WHERE x.WEIGHT BETWEEN 80 AND 500 GROUP BY x.USER_ID,cast(x.DAY AS DATE)
      ), ranked AS (
        SELECT w.window_id,x.weight_lb,u.HEIGHT_INCHES,
          date_diff('day',w.window_start+INTERVAL '3 days',x.DAY) AS offset_days,
          row_number()OVER(PARTITION BY w.window_id ORDER BY
            abs(date_diff('day',w.window_start+INTERVAL '3 days',x.DAY)),x.DAY) AS rn
        FROM primary_windows w JOIN weights x ON w.USER_ID=x.USER_ID AND
          abs(date_diff('day',w.window_start+INTERVAL '3 days',x.DAY))<=90
        JOIN users u ON u.USER_ID=w.USER_ID
      ), bmi AS (
        SELECT window_id,offset_days,
          CASE WHEN HEIGHT_INCHES BETWEEN 48 AND 90 THEN 703.*weight_lb/pow(HEIGHT_INCHES,2) END AS bmi
        FROM ranked WHERE rn=1
      ) SELECT window_id,offset_days,CASE WHEN bmi BETWEEN 15 AND 60 THEN bmi END AS paired_bmi FROM bmi
    ''').df()


def winsor_z(values):
    x=pd.to_numeric(values,errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
    low,high=x.quantile([.005,.995]);x=x.clip(low,high);sigma=x.std(ddof=1)
    return ((x-x.mean())/sigma if np.isfinite(sigma) and sigma>0 else pd.Series(np.nan,index=x.index)),sigma


def bh(values):
    p=np.asarray(values,float);q=np.full(p.shape,np.nan);idx=np.flatnonzero(np.isfinite(p))
    order=idx[np.argsort(p[idx])]
    if len(order):q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return q


def context_models(primary,context,bmi):
    data=primary.merge(context,on='window_id',validate='one_to_one').merge(bmi,on='window_id',how='left',validate='one_to_one')
    data['sex_male']=data.SEX.map({'male':1.,'female':0.})
    data['age']=pd.to_numeric(data.AGE_BUCKET,errors='coerce').where(lambda x:x.between(10,100))
    data['year']=pd.to_datetime(data.window_start).dt.year
    scales=[]
    for axis in AXES:
        data[axis+'_z'],sigma=winsor_z(data[axis])
        scales.append({'axis':axis,'winsorized_sd':sigma,'unit':'dimensionless' if axis=='ac_var' else 'mg/dL'})
    for c in ['age','year','paired_bmi']:data[c+'_z']=winsor_z(data[c])[0]
    # Global reference scaling is descriptive only, never a CV transformation.
    for c in CONTEXTS:data[c+'_z']=winsor_z(data[c])[0]
    rows=[];raw=[]
    for context in CONTEXTS:
        complete=[a+'_z' for a in AXES]+[context+'_z','age_z','year_z','paired_bmi_z','sex_male']
        subset=data.dropna(subset=complete)
        # Shared person set across all four metrics and both adjustments.
        for axis in AXES:
            for adjusted in [False,True]:
                covars=[] if axis=='mean_glucose' else ['mean_glucose_z']
                adjustment=('age_sex_bmi_year' if adjusted else 'none') if axis=='mean_glucose' else (
                    'mean_glucose_age_sex_bmi_year' if adjusted else 'mean_glucose_only')
                if adjusted:covars+=['age_z','sex_male','paired_bmi_z','year_z']
                row={'axis':axis,'context':context,'adjustment':adjustment,'n_adjusted':len(subset),
                     'status':'insufficient_or_constant',
                     'adjusted_standardized_beta':np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_value':np.nan}
                columns=[context+'_z']+covars
                if len(subset)>=300 and subset[context+'_z'].nunique()>=3:
                    design=sm.add_constant(subset[columns].astype(float),has_constant='add')
                    if np.linalg.matrix_rank(design.to_numpy())==len(design.columns):
                        fit=sm.OLS(subset[axis+'_z'],design).fit(cov_type='HC3')
                        row.update(status='estimated',adjusted_standardized_beta=float(fit.params[context+'_z']),
                                   ci_low=float(fit.conf_int().loc[context+'_z',0]),ci_high=float(fit.conf_int().loc[context+'_z',1]),
                                   p_value=float(fit.pvalues[context+'_z']))
                    else:row['status']='rank_deficient'
                rows.append(row)
        for axis in AXES+['cv_glucose']:
            pair=data[[axis,context]].replace([np.inf,-np.inf],np.nan).dropna()
            rho=stats.spearmanr(pair[axis],pair[context]).statistic if len(pair)>=3 and (pair.nunique()>=2).all() else np.nan
            raw.append({'axis':axis,'context':context,'n':len(pair),'spearman_r':float(rho)})
    results=pd.DataFrame(rows);results['q_value']=np.nan
    for _,part in results.groupby(['axis','adjustment']):results.loc[part.index,'q_value']=bh(part.p_value)
    return results,pd.DataFrame(raw),pd.DataFrame(scales),data
