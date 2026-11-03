# Source data

One tidy CSV per display item (per panel where a single table would be unwieldy). Every file is aggregate: it holds
no participant identifiers, no per-participant, per-meal or per-day rows and no dates, and no count cell between 1 and
10 (histogram bins with 1-10 participants are merged with a neighbouring bin; smaller groups are not released). A few
small counts can still be derived arithmetically from released totals and percentages (participants with sex not
recorded in some age decades of fig1c_age_sex.csv; one percentage of fig1d_glycemic_range.csv); the same values can
be derived from the published Fig. 1.
Values are stored at full precision; the published figures and tables round them.

The files are written by `python <script> source-data` (or `python run_all.py --stage source-data`) from the
analysis outputs, and read by `python <script> draw`. Fig. 6 and Supplementary Fig. 1 also draw elements that
show individual participants or small groups; those elements need the restricted analysis outputs and are not in these
files. Fig. 1 can be drawn from these files alone, with histogram bins of 1-10 participants merged.

| File | Display item | Script | Rows |
|---|---|---|---|
| `fig1ab_streams.csv` | Fig. 1a,b | `figures/fig1_cohort.py` | 21 |
| `fig1b_funnel.csv` | Fig. 1b | `figures/fig1_cohort.py` | 4 |
| `fig1c_age_sex.csv` | Fig. 1c | `figures/fig1_cohort.py` | 9 |
| `fig1d_glycemic_range.csv` | Fig. 1d | `figures/fig1_cohort.py` | 30 |
| `fig1e_distributions.csv` | Fig. 1e | `figures/fig1_cohort.py` | 9 |
| `fig1e_histograms.csv` | Fig. 1e | `figures/fig1_cohort.py` | 264 |
| `fig2b_reliability_by_gap.csv` | Fig. 2b | `figures/fig2_reliability.py` | 20 |
| `fig2c_reliability_by_days_of_wear.csv` | Fig. 2c | `figures/fig2_reliability.py` | 55 |
| `fig3a_homa_by_quartile.csv` | Fig. 3a | `figures/fig3_average_glucose.py` | 4 |
| `fig3b_bmi.csv` | Fig. 3b | `figures/fig3_average_glucose.py` | 4 |
| `fig3c_average_glucose_vs_hba1c.csv` | Fig. 3c | `figures/fig3_average_glucose.py` | 18 |
| `fig3d_unadjusted.csv` | Fig. 3d | `figures/fig3_average_glucose.py` | 20 |
| `fig4a_spike_profile.csv` | Fig. 4a | `figures/fig4_spike_width.py` | 20 |
| `fig4a_cohort_medians.csv` | Fig. 4a | `figures/fig4_spike_width.py` | 4 |
| `fig4b_same_average_glucose.csv` | Fig. 4b | `figures/fig4_spike_width.py` | 20 |
| `fig4c_spike_measures_by_marker.csv` | Fig. 4c | `figures/fig4_spike_width.py` | 12 |
| `fig4d_spike_width_adjustments.csv` | Fig. 4d | `figures/fig4_spike_width.py` | 4 |
| `fig5a_hba1c_glucose.csv` | Fig. 5a | `figures/fig5_hba1c.py` | 20 |
| `fig5b_correlations.csv` | Fig. 5b | `figures/fig5_hba1c.py` | 3 |
| `fig5c_variance_explained.csv` | Fig. 5c | `figures/fig5_hba1c.py` | 14 |
| `fig5d_groups.csv` | Fig. 5d | `figures/fig5_hba1c.py` | 4 |
| `fig5e_group_contrast.csv` | Fig. 5e | `figures/fig5_hba1c.py` | 2 |
| `fig6a_example.csv` | Fig. 6a | `figures/fig6_meals.py` | 1 |
| `fig6b_same_food_agreement.csv` | Fig. 6b | `figures/fig6_meals.py` | 7 |
| `fig6c_participants.csv` | Fig. 6c | `figures/fig6_meals.py` | 3 |
| `fig6d_slope_histogram.csv` | Fig. 6d | `figures/fig6_meals.py` | 37 |
| `fig6d_summary.csv` | Fig. 6d | `figures/fig6_meals.py` | 11 |
| `fig6e_insulin_by_quintile.csv` | Fig. 6e | `figures/fig6_meals.py` | 5 |
| `fig6e_per_sd.csv` | Fig. 6e | `figures/fig6_meals.py` | 2 |
| `sfig1_gmi_calibration.csv` | Supplementary Fig. 1 | `figures/sfig1_gmi_calibration.py` | 2 |
| `sfig2_discordant_groups.csv` | Supplementary Fig. 2 | `figures/sfig2_discordant_groups.py` | 55 |
| `sfig3_age_offset.csv` | Supplementary Fig. 3 | `figures/sfig3_age_offset.py` | 6 |
| `table1_cohort.csv` | Table 1 | `tables/table1_cohort.py` | 18 |
| `stable1a_markers.csv` | Supplementary Table 1a | `tables/stable1a_markers.py` | 9 |
| `stable1b_groups.csv` | Supplementary Table 1b | `tables/stable1b_groups.py` | 28 |
| `stable2_meal_width.csv` | Supplementary Table 2 | `tables/stable2_meal_width.py` | 10 |
| `stable3_hba1c_sensitivity.csv` | Supplementary Table 3 | `tables/stable3_hba1c_sensitivity.py` | 4 |
| `stable4_lab_retest.csv` | Supplementary Table 4 | `tables/stable4_lab_retest.py` | 4 |
| `stable5_cgm_measures.csv` | Supplementary Table 5 | `tables/stable5_cgm_measures.py` | 12 |
| `stable6_lab_units.csv` | Supplementary Table 6 | `tables/stable6_lab_units.py` | 12 |
| `stable7a_age_sex.csv` | Supplementary Table 7A | `tables/stable7_age_sex.py` | 38 |
| `stable7b_variance_explained.csv` | Supplementary Table 7B | `tables/stable7_age_sex.py` | 4 |

## fig1ab_streams.csv

Fig. 1a,b (figures/fig1_cohort.py). Participant counts by data stream, pairwise overlaps and the regions of the CGM / behaviour / laboratory diagram; median consented interval. One row per quantity.

| Column | Description |
|---|---|
| `quantity` | name of the quantity |
| `value` | value of the quantity |

## fig1b_funnel.csv

Fig. 1b (figures/fig1_cohort.py). Participant counts at each step from consent to the primary lab-linked cohort.

| Column | Description |
|---|---|
| `stage` | step of the participant flow |
| `n_participants` | number of participants |

## fig1c_age_sex.csv

Fig. 1c (figures/fig1_cohort.py). Participants by age decade and sex. Participants with sex not recorded are the difference between all_participants and female + male. Groups of 1-10 participants are not released.

| Column | Description |
|---|---|
| `age_decade` | age decade |
| `female` | female participants |
| `male` | male participants |
| `all_participants` | all participants in the row |
| `note` | explanatory note |

## fig1d_glycemic_range.csv

Fig. 1d (figures/fig1_cohort.py). Share of each age decade (and of the whole lab-linked panel) by glycemic range of the anchor HbA1c / fasting glucose.

| Column | Description |
|---|---|
| `group` | group |
| `range` | glycemic range of the anchor laboratory values |
| `pct_of_group` | percentage of the group in the range |
| `n_group` | participants in the group |

## fig1e_distributions.csv

Fig. 1e (figures/fig1_cohort.py). Summary of each distribution shown in panel e (mean, quartiles, displayed range).

| Column | Description |
|---|---|
| `measure` | CGM or laboratory measure |
| `population` | population summarized |
| `n_participants` | number of participants |
| `mean` | mean |
| `q25` | 25th percentile |
| `q75` | 75th percentile |
| `display_lo` | lower end of the displayed axis |
| `display_hi` | upper end of the displayed axis |

## fig1e_histograms.csv

Fig. 1e (figures/fig1_cohort.py). Histogram bin counts of the distributions in panel e. Bins holding 1-10 participants are merged with a neighbouring bin (n_bins_merged > 1).

| Column | Description |
|---|---|
| `measure` | CGM or laboratory measure |
| `bin_lo` | lower edge of the histogram bin |
| `bin_hi` | upper edge of the histogram bin |
| `n_participants` | number of participants |
| `n_bins_merged` | number of original bins merged into this row (1 = not merged) |

## fig2b_reliability_by_gap.csv

Fig. 2b (figures/fig2_reliability.py). Week-to-week reliability (ICC(A,1)) of each CGM metric by the gap between the two seven-day windows.

| Column | Description |
|---|---|
| `metric` | CGM metric |
| `gap_bin` | gap category |
| `gap_label` | gap category as printed |
| `gap_min_days` | shortest gap in the category (days) |
| `gap_max_days` | longest gap in the category (days) |
| `n_participants` | number of participants |
| `icc_a1` | intraclass correlation ICC(A,1) between the two windows |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig2c_reliability_by_days_of_wear.csv

Fig. 2c (figures/fig2_reliability.py). Reliability of each CGM metric by the number of days of wear averaged in each window.

| Column | Description |
|---|---|
| `metric` | CGM metric |
| `days_of_wear` | days of wear averaged in each window |
| `n_participants` | number of participants |
| `icc_a1` | intraclass correlation ICC(A,1) between the two windows |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig3a_homa_by_quartile.csv

Fig. 3a (figures/fig3_average_glucose.py). HOMA-IR >= 2.5 and geometric-mean HOMA-IR ratio by quartile of average glucose.

| Column | Description |
|---|---|
| `quartile` | quartile of average glucose |
| `average_glucose_min` | lowest average glucose in the quartile (mg/dL) |
| `average_glucose_max` | highest average glucose in the quartile (mg/dL) |
| `n_participants` | number of participants |
| `n_homa_ir_ge_2_5` | participants with HOMA-IR >= 2.5 |
| `pct_homa_ir_ge_2_5` | percentage with HOMA-IR >= 2.5 |
| `wilson_ci_low` | lower bound of the Wilson 95% interval |
| `wilson_ci_high` | upper bound of the Wilson 95% interval |
| `geometric_mean_ratio_vs_q1` | ratio of geometric-mean HOMA-IR to the first quartile |
| `ratio_ci_low` | lower bound of the 95% CI of the ratio |
| `ratio_ci_high` | upper bound of the 95% CI of the ratio |

## fig3b_bmi.csv

Fig. 3b (figures/fig3_average_glucose.py). Association of average glucose with fasting insulin and HOMA-IR before and after adjustment for BMI.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_participants` | number of participants |
| `adjustment` | covariates in the model |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `attenuation_pct` | percentage reduction of the estimate after adjustment |

## fig3c_average_glucose_vs_hba1c.csv

Fig. 3c (figures/fig3_average_glucose.py). Associations of average glucose and of HbA1c with each laboratory marker, in the same participants.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_participants` | number of participants |
| `measure` | CGM or laboratory measure |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig3d_unadjusted.csv

Fig. 3d (figures/fig3_average_glucose.py). Unadjusted associations of each CGM measure with fasting insulin and HOMA-IR.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `group` | group |
| `measure` | CGM or laboratory measure |
| `n_participants` | number of participants |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig4a_spike_profile.csv

Fig. 4a (figures/fig4_spike_width.py). Median (IQR) timing of the average spike shape: minutes from the peak at which glucose crosses each fraction of the rise, on the way up and down.

| Column | Description |
|---|---|
| `side` | up = rise to the peak, down = fall after the peak |
| `fraction_of_rise` | fraction of the spike rise |
| `minutes_from_peak_median` | median minutes from the peak |
| `minutes_from_peak_q1` | 25th percentile of minutes from the peak |
| `minutes_from_peak_q3` | 75th percentile of minutes from the peak |

## fig4a_cohort_medians.csv

Fig. 4a (figures/fig4_spike_width.py). Cohort medians printed on panel a.

| Column | Description |
|---|---|
| `quantity` | name of the quantity |
| `value` | value of the quantity |

## fig4b_same_average_glucose.csv

Fig. 4b (figures/fig4_spike_width.py). Associations of each CGM measure with fasting insulin and HOMA-IR at the same average glucose.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `group` | group |
| `measure` | CGM or laboratory measure |
| `n_participants` | number of participants |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig4c_spike_measures_by_marker.csv

Fig. 4c (figures/fig4_spike_width.py). Associations of the spike measures with laboratory markers at the same average glucose.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `measure` | CGM or laboratory measure |
| `n_participants` | number of participants |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig4d_spike_width_adjustments.csv

Fig. 4d (figures/fig4_spike_width.py). Association of spike width with fasting insulin under successive adjustments.

| Column | Description |
|---|---|
| `adjustment` | covariates in the model |
| `n_participants` | number of participants |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig5a_hba1c_glucose.csv

Fig. 5a (figures/fig5_hba1c.py). Average glucose by HbA1c value (points with 95% CI) and the fitted lines within HbA1c ranges.

| Column | Description |
|---|---|
| `element` | drawn element (point or fitted line) |
| `hba1c_pct` | HbA1c value (%) |
| `hba1c_range` | HbA1c range |
| `n_participants` | number of participants |
| `average_glucose_mg_dl` | mean of average glucose (mg/dL) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `hba1c_min` | lower end of the HbA1c range (%) |
| `hba1c_max` | upper end of the HbA1c range (%) |
| `intercept_mg_dl` | intercept of the fitted line (mg/dL) |
| `slope_per_point` | slope of average glucose per HbA1c percentage point (mg/dL) |
| `slope_per_0_1` | slope per 0.1 HbA1c percentage point (mg/dL) |
| `slope_per_0_1_ci_low` | lower bound of the 95% CI of the slope per 0.1 point |
| `slope_per_0_1_ci_high` | upper bound of the 95% CI of the slope per 0.1 point |

## fig5b_correlations.csv

Fig. 5b (figures/fig5_hba1c.py). Correlation between HbA1c and average glucose within HbA1c ranges.

| Column | Description |
|---|---|
| `hba1c_range` | HbA1c range |
| `hba1c_min` | lower end of the HbA1c range (%) |
| `hba1c_max` | upper end of the HbA1c range (%) |
| `n_participants` | number of participants |
| `pearson_r` | Pearson correlation |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `hba1c_sd` | SD of HbA1c in the range (%) |

## fig5c_variance_explained.csv

Fig. 5c (figures/fig5_hba1c.py). Variance in laboratory markers explained by average glucose and by HbA1c.

| Column | Description |
|---|---|
| `scope` | sample |
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_participants` | number of participants |
| `quantity` | name of the quantity |
| `estimate` | estimate |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig5d_groups.csv

Fig. 5d (figures/fig5_hba1c.py). Groups defined by average glucose >= 110 mg/dL and HbA1c >= 5.7%: size and share with fasting insulin >= 10 uIU/mL.

| Column | Description |
|---|---|
| `group` | group |
| `n_with_fasting_insulin` | participants with a fasting insulin value |
| `n_fasting_insulin_ge_10` | participants with fasting insulin >= 10 uIU/mL |
| `pct_fasting_insulin_ge_10` | percentage with fasting insulin >= 10 uIU/mL |
| `wilson_ci_low` | lower bound of the Wilson 95% interval |
| `wilson_ci_high` | upper bound of the Wilson 95% interval |
| `n_primary_cohort` | participants in the primary cohort |
| `pct_of_primary_cohort` | percentage of the primary cohort in the group |

## fig5e_group_contrast.csv

Fig. 5e (figures/fig5_hba1c.py). Difference in fasting insulin and HOMA-IR between the CGM-elevated-only and HbA1c-elevated-only groups.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_cgm_elevated_only` | participants in the CGM-elevated-only group |
| `n_hba1c_elevated_only` | participants in the HbA1c-elevated-only group |
| `pct_difference` | percentage difference (CGM-elevated-only vs HbA1c-elevated-only) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `p_value` | two-sided p value |

## fig6a_example.csv

Fig. 6a (figures/fig6_meals.py). Values printed on panel a (one participant's repeated meals of one food; the traces are not released).

| Column | Description |
|---|---|
| `food` | food |
| `n_meals_shown` | meals drawn |
| `n_meals` | meals |

## fig6b_same_food_agreement.csv

Fig. 6b (figures/fig6_meals.py). Agreement between two averages of k meals of the same food (first k vs next k meals).

| Column | Description |
|---|---|
| `k` | meals in each of the two averages |
| `n_pairs` | pairs compared |
| `n_participants` | number of participants |
| `r` | Pearson correlation |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `median_days_between_sets` | median days between the two sets of meals |

## fig6c_participants.csv

Fig. 6c (figures/fig6_meals.py). Values printed on panel c for the three example participants, slopes rounded to the printed precision (meal-level points are not released).

| Column | Description |
|---|---|
| `participant` | example participant |
| `slope_mg_dl_per_g` | carbohydrate-sensitivity slope (mg/dL per g) |
| `n_meals` | meals |

## fig6d_slope_histogram.csv

Fig. 6d (figures/fig6_meals.py). Histogram of carbohydrate-sensitivity slopes across participants. Bins holding 1-10 participants are merged with a neighbouring bin (n_bins_merged > 1).

| Column | Description |
|---|---|
| `slope_from` | lower edge of the slope bin (mg/dL per g) |
| `slope_to` | upper edge of the slope bin (mg/dL per g) |
| `n_participants` | number of participants |
| `n_bins_merged` | number of original bins merged into this row (1 = not merged) |

## fig6d_summary.csv

Fig. 6d (figures/fig6_meals.py). Percentiles of the slope distribution and odd/even-meal agreement (the participant-level scatter is not released). One row per quantity.

| Column | Description |
|---|---|
| `quantity` | name of the quantity |
| `value` | value of the quantity |

## fig6e_insulin_by_quintile.csv

Fig. 6e (figures/fig6_meals.py). Geometric-mean fasting insulin by quintile of carbohydrate sensitivity.

| Column | Description |
|---|---|
| `quintile` | quintile of carbohydrate sensitivity |
| `n_participants` | number of participants |
| `fasting_insulin_geometric_mean` | geometric-mean fasting insulin (uIU/mL) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## fig6e_per_sd.csv

Fig. 6e (figures/fig6_meals.py). Per-SD association of carbohydrate sensitivity with fasting insulin and HOMA-IR.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_participants` | number of participants |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `n_participants_with_slopes_near_lab_draw` | participants with a slope from meals within +/-90 days of the laboratory draw |

## sfig1_gmi_calibration.csv

Supplementary Fig. 1 (figures/sfig1_gmi_calibration.py). Fitted lines of HbA1c on average glucose: the published GMI equation and the line fitted in this cohort. The binned means drawn in the figure are not released (bins can hold few participants).

| Column | Description |
|---|---|
| `line` | fitted line |
| `intercept_pct` | intercept (% HbA1c) |
| `slope_pct_per_mg_dl` | slope (% HbA1c per mg/dL average glucose) |
| `slope_ci_low` | lower bound of the 95% CI of the slope |
| `slope_ci_high` | upper bound of the 95% CI of the slope |
| `n_participants` | number of participants |
| `glucose_from_mg_dl` | start of the drawn line (mg/dL) |
| `glucose_to_mg_dl` | end of the drawn line (mg/dL) |

## sfig2_discordant_groups.csv

Supplementary Fig. 2 (figures/sfig2_discordant_groups.py). Characteristics of the four groups defined by average glucose and HbA1c: medians or shares, and differences from the both-normal group.

| Column | Description |
|---|---|
| `measure` | CGM or laboratory measure |
| `unit` | unit |
| `group` | group |
| `n_participants` | number of participants |
| `median` | median |
| `difference_vs_both_normal` | difference from the both-normal group |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `share_pct` | percentage (for binary characteristics) |

## sfig3_age_offset.csv

Supplementary Fig. 3 (figures/sfig3_age_offset.py). HbA1c difference from participants in their 30s at the same average glucose, by age decade.

| Column | Description |
|---|---|
| `age_decade` | age decade |
| `n_participants` | number of participants |
| `hba1c_difference_vs_30s_pct_points` | HbA1c difference from participants in their 30s at the same average glucose (percentage points) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `reference` | reference decade (True for the 30s) |

## table1_cohort.csv

Table 1 (tables/table1_cohort.py). Characteristics of the lab-linked primary cohort.

| Column | Description |
|---|---|
| `characteristic` | characteristic |
| `statistic` | statistic reported |
| `value` | value of the quantity |
| `q25` | 25th percentile |
| `q75` | 75th percentile |
| `n` | number of participants with a value (the denominator of percentages) |

## stable1a_markers.csv

Supplementary Table 1a (tables/stable1a_markers.py). Average glucose vs HbA1c as predictors of each laboratory marker.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_participants` | number of participants |
| `r2_average_glucose` | R2 of the log marker on average glucose |
| `r2_hba1c` | R2 of the log marker on HbA1c |
| `r2_difference` | difference in R2 (average glucose minus HbA1c) |
| `r2_difference_ci_low` | lower bound of the 95% CI of the R2 difference |
| `r2_difference_ci_high` | upper bound of the 95% CI of the R2 difference |
| `average_glucose_pct_per_sd` | % per SD of average glucose (age and sex adjusted) |
| `average_glucose_ci_low` | lower bound of the 95% CI |
| `average_glucose_ci_high` | upper bound of the 95% CI |
| `hba1c_pct_per_sd` | % per SD of HbA1c (age and sex adjusted) |
| `hba1c_ci_low` | lower bound of the 95% CI |
| `hba1c_ci_high` | upper bound of the 95% CI |
| `average_glucose_with_bmi_pct_per_sd` | % per SD of average glucose additionally adjusted for BMI |
| `average_glucose_with_bmi_ci_low` | lower bound of the 95% CI |
| `average_glucose_with_bmi_ci_high` | upper bound of the 95% CI |
| `n_participants_with_bmi` | participants with BMI |

## stable1b_groups.csv

Supplementary Table 1b (tables/stable1b_groups.py). Laboratory markers in the four average-glucose x HbA1c groups.

| Column | Description |
|---|---|
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `quantity` | name of the quantity |
| `group` | group |
| `value` | value of the quantity |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |
| `n_participants` | number of participants |

## stable2_meal_width.csv

Supplementary Table 2 (tables/stable2_meal_width.py). Meal-anchored spike width and fasting insulin / HOMA-IR.

| Column | Description |
|---|---|
| `width_measure` | spike-width measure |
| `compared_at_the` | comparison (unadjusted or same average glucose) |
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `n_participants` | number of participants |
| `pct_per_sd` | percentage difference in the marker per SD of the measure (log-linear OLS, HC3 standard errors) |
| `ci_low` | lower bound of the 95% confidence interval |
| `ci_high` | upper bound of the 95% confidence interval |

## stable3_hba1c_sensitivity.csv

Supplementary Table 3 (tables/stable3_hba1c_sensitivity.py). Average glucose vs HbA1c in sensitivity samples.

| Column | Description |
|---|---|
| `sample` | sensitivity sample |
| `n_participants` | number of participants |
| `r2_average_glucose` | R2 of the log marker on average glucose |
| `r2_hba1c` | R2 of the log marker on HbA1c |
| `r2_difference` | difference in R2 (average glucose minus HbA1c) |
| `r2_difference_ci_low` | lower bound of the 95% CI of the R2 difference |
| `r2_difference_ci_high` | upper bound of the 95% CI of the R2 difference |
| `auc_average_glucose` | AUC of average glucose for the marker threshold |
| `auc_hba1c` | AUC of HbA1c for the marker threshold |
| `auc_difference_ci_low` | lower bound of the 95% CI of the AUC difference |
| `auc_difference_ci_high` | upper bound of the 95% CI of the AUC difference |

## stable4_lab_retest.csv

Supplementary Table 4 (tables/stable4_lab_retest.py). Test-retest reliability of laboratory measures.

| Column | Description |
|---|---|
| `measure` | CGM or laboratory measure |
| `n_pairs` | pairs compared |
| `median_days_between_draws` | median days between the two draws |
| `icc1` | one-way ICC(1) |
| `pearson_r` | Pearson correlation |

## stable5_cgm_measures.csv

Supplementary Table 5 (tables/stable5_cgm_measures.py). Definitions and cohort median (IQR) of each CGM measure.

| Column | Description |
|---|---|
| `measure` | CGM or laboratory measure |
| `definition` | definition |
| `unit` | unit |
| `summarized_over` | population summarized |
| `n` | number of participants (or meals, see n_unit) |
| `n_unit` | unit of n (participants or meals) |
| `median` | median |
| `q1` | 25th percentile |
| `q3` | 75th percentile |
| `decimals` | decimals printed |

## stable6_lab_units.csv

Supplementary Table 6 (tables/stable6_lab_units.py). Laboratory units, accepted ranges and accepted alternate units with their conversion. One row per analyte and alternate unit.

| Column | Description |
|---|---|
| `analyte` | laboratory analyte |
| `unit` | unit |
| `range_low` | lowest accepted value (canonical unit) |
| `range_high` | highest accepted value (canonical unit) |
| `alternate_unit` | accepted alternate unit (empty: none) |
| `multiplier` | conversion: canonical value = reported value x multiplier + offset |
| `offset` | conversion offset |

## stable7a_age_sex.csv

Supplementary Table 7A (tables/stable7_age_sex.py). Per-SD associations with and without adjustment for age and sex.

| Column | Description |
|---|---|
| `comparison` | comparison (unadjusted or same average glucose) |
| `marker` | laboratory marker (fasting_insulin, homa_ir, triglycerides, hdl_cholesterol, hscrp, ...) |
| `measure` | CGM or laboratory measure |
| `n_participants` | number of participants |
| `pct_per_sd_without_age_sex` | % per SD without age and sex |
| `ci_low_without` | lower bound of the 95% CI without age and sex |
| `ci_high_without` | upper bound of the 95% CI without age and sex |
| `pct_per_sd_with_age_sex` | % per SD with age and sex |
| `ci_low_with` | lower bound of the 95% CI with age and sex |
| `ci_high_with` | upper bound of the 95% CI with age and sex |
| `change_points` | change in the estimate (percentage points) |

## stable7b_variance_explained.csv

Supplementary Table 7B (tables/stable7_age_sex.py). Variance in fasting insulin and HOMA-IR explained by average glucose and HbA1c beyond age and sex.

| Column | Description |
|---|---|
| `outcome` | outcome |
| `scope` | sample |
| `n_participants` | number of participants |
| `r2_average_glucose_unadjusted` | R2 of average glucose alone |
| `r2_hba1c_unadjusted` | R2 of HbA1c alone |
| `delta_r2_average_glucose_beyond_age_sex` | R2 added by average glucose beyond age and sex |
| `delta_r2_average_glucose_ci_low` | lower bound of the 95% CI |
| `delta_r2_average_glucose_ci_high` | upper bound of the 95% CI |
| `delta_r2_hba1c_beyond_age_sex` | R2 added by HbA1c beyond age and sex |
| `delta_r2_hba1c_ci_low` | lower bound of the 95% CI |
| `delta_r2_hba1c_ci_high` | upper bound of the 95% CI |
| `difference` | difference (average glucose minus HbA1c) |
| `difference_ci_low` | lower bound of the 95% CI of the difference |
| `difference_ci_high` | upper bound of the 95% CI of the difference |
