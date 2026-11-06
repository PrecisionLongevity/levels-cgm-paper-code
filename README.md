# levels-cgm-paper-code

Analysis code and source data for

> Paul Ginart and Michael Loesche. **Consumer continuous glucose monitoring uncovers hidden metabolic risk in free-living
> adults without diabetes.** *medRxiv* (2026). https://doi.org/10.64898/2026.11.03.26367126

The study is a cross-sectional analysis of 62,122 consenting adults on the Levels digital-health platform (IRB protocol
LH-001, Advarra Pro00058897). It links free-living continuous glucose monitoring (CGM), logged meals and commercial
laboratory results, and asks which CGM measures are repeatable and which track insulin resistance in a primary
laboratory cohort of 7,930 adults without diabetes.

This repository holds the code that computes every result from the study database, the code that draws the six main
figures, Table 1, the three Supplementary Figures and Supplementary Tables 1a, 1b and 2-7, and the source data of each of
these display items. It holds no participant-level data.

## Data access

The participant-level data (CGM readings, food logs, activity, sleep and laboratory records) cannot be shared:
participants consented to research use by the study team, not to public release. The data are owned by Levels Health,
Inc. and were analysed by the authors under a data use agreement; requests for access should be addressed to Levels
Health, Inc.

What is shared instead is `source_data/`: one aggregate CSV per display item, with the values each figure and table is
drawn from (described column by column in `source_data/README.md`). The files contain no participant identifiers, no
per-participant, per-meal or per-day rows, no dates and no count cells between 1 and 10 (a few small counts can be
derived from released totals and percentages, as they can from the published Fig. 1; see `source_data/README.md`).

## Repository layout

| Path | Contents |
|---|---|
| `analysis/` | Producers, grouped by analysis: read the study database and write analysis outputs (see below) |
| `figures/` | One script per figure: `fig1_cohort.py` ... `fig6_meals.py`, `sfig1_gmi_calibration.py` ... `sfig3_age_offset.py` |
| `tables/` | One script per table: `table1_cohort.py`, `stable1a_markers.py` ... `stable7_age_sex.py` |
| `cgmpaper/` | Shared package: paths and settings, laboratory normalization, CGM metrics, figure drawing and table helpers |
| `source_data/` | Aggregate source data of every display item, and `README.md` describing every file and column |
| `run_all.py` | Runs everything in order: producers, then source data, then figures and tables |
| `requirements-lock.txt` | Exact Python package versions |

`analysis/` sub-directories: `canonical_panel` (participant-level laboratory-CGM panel around each anchor HbA1c draw),
`cohort` (cohort description, data streams, laboratory distributions), `daily_glucose` (participant-local day panel),
`reliability` (week-to-week reliability), `meals` (meal substrate, meal-response shape, spikes, carbohydrate
sensitivity, same-food agreement), `discordance` (HbA1c-CGM pairs), `metabolic` (associations with insulin markers and
sensitivity analyses), `validation` (exclusion of participants with a diabetes-range laboratory value), `clinical`
(summaries behind Figs 3-5 and the Supplementary Tables, standard MAGE), `exemplars` (example participants shown in
Fig. 6). Modules whose name starts with `_` are helpers imported by the producers. Every module starts with a
short description of what it computes, its inputs, its outputs and how to run it.

## Setup

Python 3.13 with the pinned packages, and the Cairo library (CairoSVG renders the PNG versions of the figures):

```bash
python3.13 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt
# macOS: brew install cairo   (Debian/Ubuntu: apt-get install libcairo2)
```

Standard MAGE (`analysis/clinical/mage_standard.py`) also needs R with the CRAN package iglu 4.2.2 (plus arrow and
jsonlite); pass its `Rscript` with `--rscript`.

## Running

Redrawing the figures and tables from the shared source data needs no database:

```bash
python figures/fig3_average_glucose.py draw          # writes outputs/figures/fig3_average_glucose.svg and .png
python tables/stable4_lab_retest.py draw             # writes outputs/tables/stable4_lab_retest.md and .csv
python run_all.py --stage draw                       # every display item the source data support (see below)
```

Fig. 6 and Supplementary Fig. 1 also contain elements that show individual participants or small groups (the
participant traces and scatter of Fig. 6, the binned means of Supplementary Fig. 1). Their scripts draw these elements
from the restricted analysis outputs and, when those are not available, stop with a message and exit status 3, which
`run_all.py` reports as skipped; the source data hold the summary values printed on those panels. Fig. 1 is drawn from
the source data alone, where histogram bins of 1-10 participants are merged with a neighbouring bin; with the restricted
outputs its script draws the unmerged bins.

The full analysis, from the database, runs on a machine with authorised access to the study database:

```bash
export LEVELS_DB=/secure/location/levels_irb_v2_2026-10-16.duckdb
python run_all.py --list                             # the steps, in order
python run_all.py --rscript /path/to/Rscript         # producers -> source data -> figures and tables
```

Every script reads the database location from `LEVELS_DB` and stops with a message if it is not set. Analysis outputs
are written to `outputs/` (or to the directory in `LEVELS_OUTPUTS`); they include participant-level files and must stay
on the secure machine (`outputs/` is ignored by git). The figure and table scripts take `source-data` (write
`source_data/*.csv` from the analysis outputs), `draw` (draw from `source_data/`) or `all`, and `--out-dir DIR`.

## Display items, scripts and source data

| Display item | Script | Source data (`source_data/`) |
|---|---|---|
| Fig. 1 Cohort and linked data streams | `figures/fig1_cohort.py` | `fig1ab_streams.csv`, `fig1b_funnel.csv`, `fig1c_age_sex.csv`, `fig1d_glycemic_range.csv`, `fig1e_distributions.csv`, `fig1e_histograms.csv` |
| Fig. 2 How long to wear a sensor | `figures/fig2_reliability.py` | `fig2b_reliability_by_gap.csv`, `fig2c_reliability_by_days_of_wear.csv` |
| Fig. 3 Average glucose tracks insulin resistance better than other CGM markers | `figures/fig3_average_glucose.py` | `fig3a_homa_by_quartile.csv`, `fig3b_bmi.csv`, `fig3c_average_glucose_vs_hba1c.csv`, `fig3d_unadjusted.csv` |
| Fig. 4 Spike width, not height, tracks insulin resistance and metabolic markers at the same average glucose | `figures/fig4_spike_width.py` | `fig4a_spike_profile.csv`, `fig4a_cohort_medians.csv`, `fig4b_same_average_glucose.csv`, `fig4c_spike_measures_by_marker.csv`, `fig4d_spike_width_adjustments.csv` |
| Fig. 5 Average CGM glucose tracks insulin resistance more closely than HbA1c | `figures/fig5_hba1c.py` | `fig5a_hba1c_glucose.csv`, `fig5b_correlations.csv`, `fig5c_variance_explained.csv`, `fig5d_groups.csv`, `fig5e_group_contrast.csv` |
| Fig. 6 Reliability of food responses and carbohydrate sensitivity | `figures/fig6_meals.py` | `fig6a_example.csv`, `fig6b_same_food_agreement.csv`, `fig6c_participants.csv`, `fig6d_slope_histogram.csv`, `fig6d_summary.csv`, `fig6e_insulin_by_quintile.csv`, `fig6e_per_sd.csv` |
| Table 1 Cohort characteristics | `tables/table1_cohort.py` | `table1_cohort.csv` |
| Supplementary Fig. 1 GMI calibration | `figures/sfig1_gmi_calibration.py` | `sfig1_gmi_calibration.csv` |
| Supplementary Fig. 2 Discordant groups | `figures/sfig2_discordant_groups.py` | `sfig2_discordant_groups.csv` |
| Supplementary Fig. 3 HbA1c by age | `figures/sfig3_age_offset.py` | `sfig3_age_offset.csv` |
| Supplementary Table 1a | `tables/stable1a_markers.py` | `stable1a_markers.csv` |
| Supplementary Table 1b | `tables/stable1b_groups.py` | `stable1b_groups.csv` |
| Supplementary Table 2 | `tables/stable2_meal_width.py` | `stable2_meal_width.csv` |
| Supplementary Table 3 | `tables/stable3_hba1c_sensitivity.py` | `stable3_hba1c_sensitivity.csv` |
| Supplementary Table 4 | `tables/stable4_lab_retest.py` | `stable4_lab_retest.csv` |
| Supplementary Table 5 | `tables/stable5_cgm_measures.py` | `stable5_cgm_measures.csv` |
| Supplementary Table 6 | `tables/stable6_lab_units.py` | `stable6_lab_units.csv` |
| Supplementary Table 7 | `tables/stable7_age_sex.py` | `stable7a_age_sex.csv`, `stable7b_variance_explained.csv` |

The analysis outputs each script reads, and the producer that writes them, are listed at the top of the script.

## License

See `LICENSE`.
