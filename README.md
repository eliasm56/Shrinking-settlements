# Detecting severe population decline in Arctic settlements from nighttime lights

This repository contains the analysis pipeline for the study titled 
"Implications of Shrinking Populations in the Russian Arctic for Costs 
of Permafrost Degradation", which has been submitted to Environmental Research Letters
for review. The end goal is to estimate how much of Russia's Arctic 
building stock could be damaged by climate-driven loss of permafrost bearing capacity, 
and how much of that damage falls on buildings in **settlements that are either abandoned or shrinking**.

To get there, the pipeline uses annual nighttime lights (NTL) for 1992–2013 to
identify settlements that have experienced **severe population decline** (a
population loss of 50% or more). It then flags the buildings in those
settlements as inactive and runs a damage model that splits the results
between active and inactive buildings.

The pipeline has seven steps:

1. Build a clean NTL time series for each settlement.
2. Summarise each settlement's time series as two change metrics.
3. Train and cross-validate a logistic regression on settlements with known
   population change, and save the model weights.
4. Apply the saved model to every settlement and assess it against an
   independent validation dataset.
5. Mark buildings that fall inside settlements predicted to be in severe
   decline as inactive.
6. Assign each building to its municipality.
7. Model bearing-capacity damage to Russian buildings under future climate
   scenarios, with Monte Carlo uncertainty, split by active and inactive
   buildings.

---

## Contents

- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Configuring paths](#configuring-paths)
- [Running the pipeline](#running-the-pipeline)
  - [Step 1: Build the NTL time series](#step-1-build-the-ntl-time-series)
  - [Step 2: Calculate NTL change metrics](#step-2-calculate-ntl-change-metrics)
  - [Step 3: Train and evaluate the logistic regression](#step-3-train-and-evaluate-the-logistic-regression)
  - [Step 4: Predict all settlements and validate](#step-4-predict-all-settlements-and-validate)
  - [Step 5: Flag buildings in inactive settlements](#step-5-flag-buildings-in-inactive-settlements)
  - [Step 6: Assign buildings to municipalities](#step-6-assign-buildings-to-municipalities)
  - [Step 7: Model building damage](#step-7-model-building-damage)
- [Key settings](#key-settings)
- [Provided outputs](#provided-outputs)
- [Notes and known limitations](#notes-and-known-limitations)

---

## Requirements

- Python 3
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [statsmodels](https://www.statsmodels.org/)
- [scikit-learn](https://scikit-learn.org/)
- [GeoPandas](https://geopandas.org/) (steps 4–7)
- [xarray](https://xarray.dev/) and
  [netCDF4](https://unidata.github.io/netcdf4-python/) (step 7 only)

GeoPandas depends on compiled geospatial libraries (GDAL, PROJ, GEOS), so the
easiest way to install everything is with conda:

```bash
conda create -n ntl -c conda-forge python geopandas statsmodels scikit-learn xarray netcdf4
conda activate ntl
```

---

## Repository structure

```text
inputs/                                        # All input data except the buildings GeoPackage
outputs/                                       # Expected outputs (see "Provided outputs")
weights/
└── two_variable_full_model_weights.json       # Saved logistic regression weights
code/
├── time_series/
│   ├── ALAN_settlement_extract.js             # CCNL extraction in Google Earth Engine (optional)
│   ├── build_ccnl_ntl_time_series.py          # Step 1
│   └── ntl_time_series_analysis.py            # Step 2
├── models/
│   ├── train_and_eval_logistic_regression.py  # Step 3
│   ├── predict_and_validate.py                # Step 4
│   └── damage_model_RUSonly.py                # Step 7
└── gis/
    ├── join_activity_to_bldgs.py              # Step 5
    └── join_admin2_to_bldg.py                 # Step 6
```

Most scripts start with a detailed header that explains their inputs, outputs
and method in more depth than this README.

---

## Data

All input data are in the `inputs/` folder, except for the buildings
GeoPackage, which must be downloaded separately (see
[Downloading the buildings data](#downloading-the-buildings-data) below). The
table below lists what each step reads.

| Data | Used in | Description |
| --- | --- | --- |
| CCNL settlement NTL CSVs | Step 1 | A folder of CSV files of annual CCNL nighttime lights, with one row per settlement per year. Required columns: `sett`, `year`, `ntl_sum`. See [How the CCNL data were extracted](#how-the-ccnl-data-were-extracted). |
| `model_training_dataset.csv` | Step 3 | One row per settlement with known population change. Required columns: `sett`, `pop_decrease` (population change in %, negative = decline), `collapse_loss_pct` and `recent_vs_early_pct` (from step 2). |
| `master_matched_validation_with_all_metrics.csv` | Step 4 | Independent validation settlements. Required columns: `sett`, `percent_decrease` (population change in %), plus the two NTL predictors. |
| Settlement points shapefile (`rus_sett_DCW_ACPR.shp`) | Step 4 | Settlement point locations with a `sett` name field. |
| Settlement buffers shapefile (`named_sett_500m_dissolved_buffers.shp`) | Step 4 | 500 m dissolved settlement buffers with a `sett` name field. |
| Buildings GeoPackage (`HABITAT_OSM_bldg_type_activity_topo_with_stories.gpkg`, layer `buildings`) | Step 5 | Arctic building footprints, available from the [Arctic Data Center](https://arcticdata.io/catalog/view/doi%3A10.18739%2FA28K74Z87). Not included in this repository. |
| Russian municipal boundaries (`geoBoundaries-RUS-ADM2_simplified.shp`) | Step 6 | geoBoundaries ADM2 (municipality) polygons for Russia. The `shapeName` field gives the municipality name. |
| Replacement cost inventory (`ACPR_Adm1_cost_inventory.csv`) | Step 7 | Residential and non-residential replacement cost per m² (2021 USD) for each region. Required columns: `shapeGroup`, `shapeName`, `RES_COST_PER_AREA`, `NONRES_COST_PER_AREA`. |
| Bearing capacity change (`building_bc_change_by_hazard_map.csv`) | Step 7 | Building-level projected change in pile bearing capacity for four climate models under SSP2-4.5 and SSP5-8.5. Provided in a ZIP file in `inputs/`. See [Bearing capacity data](#bearing-capacity-data). |

Settlement names (`sett`) are the key that links the settlement datasets in
steps 1–5, so they must be spelled consistently. Name matching ignores case
and extra whitespace.

### How the CCNL data were extracted

The CCNL CSVs in `inputs/` were exported from Google Earth Engine with
`code/time_series/ALAN_settlement_extract.js`. You don't need to run this
script, because its outputs are already provided. It is included so you can
see how the data were accessed.

The script:

- reads the CCNL annual nighttime lights collection (`BNU/FGS/CCNL/v1`,
  band `b1`)
- summarises the pixels inside each 500 m settlement buffer at 1 km
  resolution, keeping valid zero-valued pixels
- exports one CSV per year to a Google Drive folder, with the fields `sett`,
  `year`, `ntl_sum`, `ntl_mean`, `ntl_median`, `ntl_min`, `ntl_max`,
  `ntl_stdDev` and `ntl_count`

Only `sett`, `year` and `ntl_sum` are used in the rest of the pipeline.

To run it yourself, paste it into the
[Earth Engine Code Editor](https://code.earthengine.google.com/) and make
these changes first:

- upload the settlement buffers to your own Earth Engine assets, and change
  the `settlements` asset path to point at them (the path in the script is a
  private asset)
- set `START_YEAR` and `END_YEAR` to the years you want (the analysis uses
  1992–2013)

Then start the export tasks from the **Tasks** tab. There is one task per
year.

### Downloading the buildings data

The buildings GeoPackage is published separately and isn't included in this
repository. Download
`HABITAT_OSM_bldg_type_activity_topo_with_stories.gpkg` from the Arctic Data
Center:

<https://arcticdata.io/catalog/view/doi%3A10.18739%2FA28K74Z87>

Then set `BUILDINGS_GPKG` at the top of `code/gis/join_activity_to_bldgs.py`
to the downloaded file's location.

### Bearing capacity data

The bearing capacity data used in step 7 are provided as
`building_bc_change_by_hazard_map.csv`, inside a ZIP file in `inputs/`.
Unzip it before running step 7. Full details are in
[`README_bc_maps.txt`](README_bc_maps.txt). In summary:

- The values are building-level changes in pile bearing capacity (0–1)
  between 2015–2024 and 2055–2064. They come from the permafrost geotechnical
  model of Streletskiy et al. (2012).
- The model was forced with air temperature and precipitation from four
  global climate models: AWI-CM-1-1-MR, CESM2-WACCM, MPI-ESM1-2-HR and
  NorESM2-MM.
- There is one column for each combination of climate model and scenario
  (SSP2-4.5 and SSP5-8.5), so eight hazard maps in total. Each column name
  starts with `bc_diff_ssp245_` or `bc_diff_ssp585_`.
- Rows are joined to building footprints using the `bldg_id` field.

> Streletskiy, D. A., Shiklomanov, N. I., & Nelson, F. E. (2012).
> Permafrost, infrastructure, and climate change: A GIS-based landscape
> approach to geotechnical modeling. *Arctic, Antarctic, and Alpine
> Research*, 44(3), 368–380. <https://doi.org/10.1657/1938-4246-44.3.368>

---

## Configuring paths

All file paths are set as constants near the top of each script (in the
`FILES`, `FILE PATHS` or `INPUTS` sections). They are currently absolute
Windows paths, so **edit them before running anything**:

- point input paths at the matching files in `inputs/`
- point `BUILDINGS_GPKG` in `join_activity_to_bldgs.py` at your downloaded
  buildings GeoPackage
- point output paths wherever you want results written
- to use the provided model weights instead of running step 3, set
  `MODEL_WEIGHTS_FILE` in `predict_and_validate.py` to
  `weights/two_variable_full_model_weights.json`

`predict_and_validate.py`, `join_activity_to_bldgs.py` and
`join_admin2_to_bldg.py` also set the
`PROJ_LIB` environment variable to a specific conda environment, so GeoPandas
can find its projection database. Update that path for your machine, or
delete those lines if GeoPandas already works in your environment.

---

## Running the pipeline

Run the scripts in order from the repository root. Each step reads the
outputs of the step before it.

```bash
python code/time_series/build_ccnl_ntl_time_series.py
python code/time_series/ntl_time_series_analysis.py
python code/models/train_and_eval_logistic_regression.py
python code/models/predict_and_validate.py
python code/gis/join_activity_to_bldgs.py
python code/gis/join_admin2_to_bldg.py
python code/models/damage_model_RUSonly.py
```

Every script prints progress messages and summary information to the console.

You don't have to run every step. The repository includes the outputs of
each stage (see [Provided outputs](#provided-outputs)), so you can start
partway through:

| To start at | Use these provided files instead of rerunning earlier steps |
| --- | --- |
| Step 4 | `weights/two_variable_full_model_weights.json` (from step 3) and `outputs/ntl_collapse_recovery_metrics_v2.csv` (from step 2) |
| Step 5 | `outputs/ntl_all_settlements_logistic_predictions_buffers.gpkg` (from step 4), plus the downloaded buildings GeoPackage |
| Results only | `outputs/Damage model outputs.zip` holds the step 7 results, so you can inspect them without running anything |

The building GeoPackages written by steps 5 and 6 are not provided, because
they are built from the separately downloaded buildings data. To rerun step 7,
first run steps 5 and 6.

Compare your own results against the files in `outputs/` to check that the
pipeline is reproducing the published analysis.

### Step 1: Build the NTL time series

```bash
python code/time_series/build_ccnl_ntl_time_series.py
```

This step reads every CSV in the CCNL input folder and combines them into a
single long-format table. It then:

- strips byte-order marks and whitespace from column names
- treats missing or non-numeric `ntl_sum` values as 0 (no detected light)
- keeps only the years 1992–2013
- removes settlements that never have any detectable light (peak `ntl_sum` = 0)

**Output:** `ntl_timeseries_sett_v2.csv`, with one row per settlement per year.

### Step 2: Calculate NTL change metrics

```bash
python code/time_series/ntl_time_series_analysis.py
```

This step reduces each settlement's time series to the two metrics used as
model predictors:

| Metric | Definition | Meaning |
| --- | --- | --- |
| `collapse_loss_pct` | 100 × (early median − trough) / early median | How far NTL fell from the early-1990s baseline to its lowest point. |
| `recent_vs_early_pct` | 100 × (recent median − early median) / early median | How NTL in the early 2010s compares with the early-1990s baseline. |

Where:

- **early median** is the median `ntl_sum` for 1992–1995
- **recent median** is the median `ntl_sum` for 2010–2013
- **trough** is the lowest `ntl_sum` in the full time series

Medians are used so that one anomalous year cannot drive the result. Both
metrics are missing (NaN) when the early baseline is zero, because a
percentage of zero is undefined.

**Output:** `ntl_collapse_recovery_metrics_v2.csv`, with one row per
settlement. It contains the two metrics and the values used to calculate
them.

### Step 3: Train and evaluate the logistic regression

```bash
python code/models/train_and_eval_logistic_regression.py
```

This step labels each training settlement as severe decline (1) if
`pop_decrease` ≤ −50, or not severe decline (0) otherwise. It then fits a
logistic regression using `collapse_loss_pct` and `recent_vs_early_pct` as
predictors.

- **Evaluation:** stratified 5-fold cross-validation, with a fixed random seed
  so results are reproducible. Predictors are standardised within each fold
  using the training data only, so no information leaks from the test fold.
  Results are reported per fold and pooled across all out-of-fold
  predictions.
- **Final model:** refitted on all training settlements. Its coefficients
  and odds ratios are per one standard deviation of each predictor.
- **Saved weights:** the final model's intercept, coefficients and
  standardisation parameters are saved to
  `two_variable_full_model_weights.json`. Step 4 uses this file, so the model
  is only fitted once.

**Outputs** (in `logistic_two_variable_cv/`):

| File | Contents |
| --- | --- |
| `two_variable_class_counts.csv` | Number of training settlements in each class. |
| `two_variable_cv_fold_results.csv` | Metrics and fit statistics for each fold. |
| `two_variable_cv_predictions.csv` | Out-of-fold predicted probability and class for each settlement. |
| `two_variable_oof_cv_classification_report.csv` | Classification report on the pooled out-of-fold predictions. |
| `two_variable_oof_cv_confusion_matrix.csv` | Pooled out-of-fold confusion matrix. |
| `two_variable_full_model_coefficients.csv` | Final-model coefficients, odds ratios, standard errors and p values. |
| `two_variable_full_model_summary.txt` | Full statsmodels summary of the final model. |
| `two_variable_full_model_weights.json` | Saved model weights for step 4. |

### Step 4: Predict all settlements and validate

```bash
python code/models/predict_and_validate.py
```

This step loads the saved weights from step 3; it does not refit the model.
Before predicting, it checks that the saved predictors and severe-decline
threshold match its own settings, and stops with an error if they don't.
It then:

1. Predicts the probability of severe decline for every settlement in
   `ntl_collapse_recovery_metrics_v2.csv`. Settlements with a probability of
   0.45 or more are classed as severe decline.
2. Joins the predictions to the settlement points and buffers and writes
   them as GeoPackages for mapping.
3. Relabels the independent validation settlements using the same −50%
   threshold, predicts them with the same model, and calculates accuracy
   metrics.

The validation data are used once, to evaluate a model and probability
threshold that have already been chosen. They are not used to tune either.

**Outputs** (in `logistic_prediction_all_settlements_v3/`):

| File | Contents |
| --- | --- |
| `ntl_all_settlements_logistic_predictions.csv` | Predicted probability and class for every settlement. |
| `ntl_all_settlements_logistic_predictions_points.gpkg` | Settlement points with predictions attached. |
| `ntl_all_settlements_logistic_predictions_buffers.gpkg` | Settlement buffers with predictions attached (used in step 5). |
| `ntl_all_settlements_logistic_predictions_spatial_join_summary.csv` | How many spatial features were matched to a prediction. |
| `ntl_all_settlements_logistic_scaler_params.csv` | Standardisation parameters and coefficients used, for reference. |

**Validation outputs** (in `accuracy_assessment/logistic_regression_v3/`):

| File | Contents |
| --- | --- |
| `logistic_master_validation_predictions.csv` | Predicted and true class for each validation settlement. |
| `logistic_master_validation_metrics.csv` | Accuracy, balanced accuracy, precision, recall, F1 and ROC AUC. |
| `logistic_master_validation_confusion_matrix.csv` | Validation confusion matrix. |
| `logistic_master_validation_classification_report.csv` | Per-class validation report. |
| `logistic_master_validation_run_summary.csv` | Settings, sample sizes and output paths for the run. |

### Step 5: Flag buildings in inactive settlements

```bash
python code/gis/join_activity_to_bldgs.py
```

This step takes the settlement buffers predicted as severe decline in step 4
and spatially joins them to the building footprints, after reprojecting the
buffers to the buildings' coordinate reference system. Each building gets an
`active` field:

- `active = 0`: the building intersects a settlement buffer predicted to be
  in severe decline (inactive)
- `active = 1`: all other buildings

Any columns whose names start with `ntl` are dropped from the output.

**Output:** `HABITAT_OSM_buildings_v3.gpkg` (layer `buildings`).

### Step 6: Assign buildings to municipalities

```bash
python code/gis/join_admin2_to_bldg.py
```

This step adds a `municipality` field to the step 5 buildings, so that step 7
can report damage by municipality. It reprojects the geoBoundaries ADM2
polygons to the buildings' coordinate reference system, then spatially joins
them to the buildings. Each building takes the name of the municipality it
intersects.

- A building that crosses a municipal boundary keeps only its first matching
  municipality.
- A building that doesn't intersect any municipality has a missing
  `municipality` value. Step 7 labels these as `Unknown`.

**Output:** `HABITAT_OSM_buildings_v3_municipality.gpkg` (layer
`buildings`). This is the building inventory used by step 7.

### Step 7: Model building damage

```bash
python code/models/damage_model_RUSonly.py
```

This is the final step of the analysis. It estimates how much of the Russian
building stock is damaged by projected loss of permafrost bearing capacity
(BC), and reports the results separately for active and inactive buildings.

**Building inventory.** The model reads the step 6 output,
`HABITAT_OSM_buildings_v3_municipality.gpkg`. Only buildings with `shapeGroup == "RUS"` are used.
The buildings layer must contain these fields:

| Field | Meaning |
| --- | --- |
| `shapeGroup` | Country code. |
| `shapeName` | Region. Used to look up replacement costs. |
| `municipality` | Municipality (from step 6). Missing values are labelled `Unknown`. |
| `Source` | Source of the building footprint. |
| `active` | 1 = active, 0 = inactive (from step 5). |
| `Value` | Building type: 1 = residential, anything else = non-residential. |
| `EXTENT` | Permafrost extent class at the building: `C`, `D`, `S` or `I`. |
| `Shape_Area` | Footprint area (m²). |
| `num_stories` | Number of storeys (optional). |

**Method.** For each hazard map, the script:

1. Finds the BC loss at each building's centroid, using the nearest grid
   cell. If the netCDF file has a time dimension, it is averaged. The loss is
   converted to a fraction between 0 and 1: negative values (decreases) are
   flipped to positive, and percentages are divided by 100.
2. Calculates floor area. For residential buildings with a storey count,
   this is footprint × storeys; otherwise it is the footprint area.
3. Calculates replacement cost as floor area × regional cost per m²,
   converted from 2021 to 2024 USD with a GDP deflator.
4. Marks a building as damaged when its BC loss exceeds 1 − 1/FS, where FS is
   the foundation's factor of safety.
5. Calculates damaged area and cost as floor area (or cost) × a damage
   multiplier for the building's permafrost extent class.
6. Repeats steps 2–5 for every Monte Carlo draw, sampling the uncertain
   inputs each time (see below). It then sums the totals and damages by
   region, municipality, footprint source and activity status.

**Uncertainty.** Each draw samples these sources of uncertainty:

| Source | How it is sampled |
| --- | --- |
| Factor of safety | Uniform, 1.05–1.56 for Russia. |
| Permafrost extent multiplier | Uniform within each class: C 0.90–1.00, D 0.50–0.90, S 0.10–0.50. Class I is always 0. |
| Missed buildings | Footprint detection F1 = 0.79. The fraction of buildings missing from the inventory is drawn from a triangular distribution, and areas and costs are scaled up to compensate. |
| Building type errors | Type-label F1 = 0.84 (residential) and 0.82 (non-residential). Each building may be randomly switched between residential and non-residential. |
| Storey underestimation | Residential buildings with a storey count get +0, +1 or +2 storeys, with probabilities 0.2, 0.6 and 0.2. |
| Hazard and exposure scaling | Optional lognormal multipliers. These are off by default (`HAZARD_SIGMA = AREA_SIGMA = 0`). |

**Modes.** By default the script runs the `ALL` mode: every uncertainty
source is active, with 300 draws per hazard map and a fixed random seed (7).
For one-at-a-time sensitivity analysis, uncomment other modes in
`MODES_TO_RUN` (for example `FS_ONLY` or `EXTENT_ONLY`). In those modes only
the named source varies, the others are held at their midpoints or neutral
values, and 75 draws are used per hazard map.

**Outputs** (in `damage_model_w_uncertainty_russia_with_adm2_v2/`, one set per
mode, where `<MODE>` is the mode name):

| File | Contents |
| --- | --- |
| `BC_risk_analysis_RUS_uncert_netcdf__<MODE>_mean.csv` | Mean across hazard maps. |
| `BC_risk_analysis_RUS_uncert_netcdf__<MODE>_min.csv` | Minimum across hazard maps. |
| `BC_risk_analysis_RUS_uncert_netcdf__<MODE>_max.csv` | Maximum across hazard maps. |
| `BC_risk_analysis_RUS_uncert_netcdf__<MODE>_MC_country_draws.csv` | Damage for each draw and hazard map, totalled for the country. |
| `BC_risk_analysis_RUS_uncert_netcdf__<MODE>_MC_region_draws.csv` | Damage for each draw and hazard map, by region. |
| `BC_risk_analysis_RUS_uncert_netcdf__<MODE>_MC_municipality_draws.csv` | Damage for each draw and hazard map, by municipality. |

The three summary files have one row per scenario, region, municipality,
footprint source and activity status. Each row contains:

- total and damaged floor area (m²) and replacement cost (2024 USD), for all
  buildings and split into residential and non-residential
- the mean, 5th, 50th and 95th percentiles of each of these across the Monte
  Carlo draws
- the proportion of area and cost damaged

The per-draw files are the ones to use for plotting full uncertainty
distributions. They are split by activity status and by residential and
non-residential buildings. They are written only when
`WRITE_MC_DRAWS = True`, which is the default, and they are recreated from
scratch on every run.

---

## Key settings

These constants are set at the top of the scripts. If you change one, rerun
every step from the first script that uses it.

| Setting | Value | Script(s) | Purpose |
| --- | --- | --- | --- |
| `START_YEAR`, `END_YEAR` | 1992, 2013 | Step 1 | Years kept in the time series. |
| `EARLY_YEARS` | 1992–1995 | Step 2 | Baseline window. |
| `RECENT_YEARS` | 2010–2013 | Step 2 | Recent window. |
| `SEVERE_DECLINE_THRESHOLD` | −50 | Steps 3 and 4 | Population change (%) at or below which a settlement counts as severe decline. **Must be the same in both scripts**, and step 4 checks this. |
| `PREDICTORS` | `collapse_loss_pct`, `recent_vs_early_pct` | Steps 3 and 4 | Model predictors. **Must be the same in both scripts**, and step 4 checks this. |
| `PROB_THRESHOLD` | 0.50 (step 3), 0.45 (step 4) | Steps 3 and 4 | Probability cut-off for classifying a settlement as severe decline. Step 3 uses it for cross-validation metrics; step 4 uses it for the final predictions and validation. |
| `N_SPLITS`, `RANDOM_STATE` | 5, 42 | Step 3 | Number of cross-validation folds and the random seed. |
| `MODES_TO_RUN` | `["ALL"]` | Step 7 | Which uncertainty modes to run. |
| `MC_DRAWS_BY_MODE` | 300 (`ALL`), 75 (others) | Step 7 | Monte Carlo draws per hazard map. |
| `RANDOM_SEED` | 7 | Step 7 | Random seed for the Monte Carlo draws. |
| `FS_RANGES`, `EXTENT_BOUNDS` | See [step 7](#step-7-model-building-damage) | Step 7 | Ranges for the factor of safety and permafrost extent multipliers. |
| `GDP_DEFLATOR_2021_TO_2024` | 1.1386 | Step 7 | Converts replacement costs from 2021 to 2024 USD. |

---

## Provided outputs

The repository includes the expected outputs of the analysis, so you can
check your results against them or skip steps (see
[Running the pipeline](#running-the-pipeline)).

| File | Produced by | Contents |
| --- | --- | --- |
| `outputs/ntl_collapse_recovery_metrics_v2.csv` | Step 2 | NTL change metrics for every settlement. |
| `weights/two_variable_full_model_weights.json` | Step 3 | Saved logistic regression weights. |
| `outputs/ntl_all_settlements_logistic_predictions.csv` | Step 4 | Predicted probability and class for every settlement. |
| `outputs/ntl_all_settlements_logistic_predictions_points.gpkg` | Step 4 | Settlement points with predictions attached. |
| `outputs/ntl_all_settlements_logistic_predictions_buffers.gpkg` | Step 4 | Settlement buffers with predictions attached. |
| `outputs/Damage model outputs.zip` | Step 7 | All damage model output CSVs (summaries and per-draw results). |

When you run the scripts yourself, each one writes to the output folder set
at the top of the script. Those folders are listed in each step's
**Outputs** section above.

---

## Notes and known limitations

- **Settlements dark in the baseline period.** Settlements with zero light in
  1992–1995 have no defined `recent_vs_early_pct`. They are left out of
  model training and receive no prediction in step 4. Their
  `collapse_loss_pct` is recorded as 0, not missing.
- **Very small baselines.** A very small early baseline can make
  `recent_vs_early_pct` extremely large (thousands of percent or more), and
  these settlements can have a strong influence on the model.
- **Different probability thresholds.** Cross-validation metrics in step 3
  are calculated at a 0.50 cut-off, while the final predictions and
  validation in step 4 use 0.45.
- **Unpredicted settlements count as active.** In step 5, any building that
  doesn't intersect a buffer predicted as severe decline is marked active.
  This includes buildings in settlements that received no prediction.
- **Run time and disk space.** Step 7 runs 300 Monte Carlo draws for every
  hazard map and, by default, writes every draw to disk. The per-draw CSVs
  can be large. Set `DRAW_STRIDE` above 1 to keep only every *n*th draw, or
  set `WRITE_MC_DRAWS = False` to skip them.
- **Hard-coded paths.** File paths are absolute and must be edited before
  running. See [Configuring paths](#configuring-paths).
