# APHREH ADSMap Pipeline

This repository implements APHREH-ADSMap: Acute Public-Health Risks from Environmental Hazards - Automated Data-driven Statistical Mapping. It is a full computational workflow to estimate spatial health vulnerability linked to environmental exposures. The pipeline identifies exposed and non-exposed days, computes incidence differentials, performs bootstrap-based effect estimation, aggregates results across years, and derives summary metrics including MARM and WMARM. It also includes optional uncertainty and sensitivity analyses.

All behaviour is controlled through the configuration file conf.py.

---

## 1. Method Overview

### Exposure and Non-Exposure Days
For each year and parameter combination (exposure percentile and lag), exposure thresholds are computed and days are classified into exposed or non-exposed.

### Incidence and Incidence Differentials
Daily incidence is computed per spatial unit (BSA), considering lagged effects. Incidence differentials compare exposed versus baseline conditions.

### Vulnerability Index
A bootstrap implementation of a weighted Mann-Whitney procedure estimates effect sizes and confidence intervals for each spatial unit.

### MARM and WMARM
Indexes across years are aggregated and standardised. Two summary measures are produced:

- MARM: multi-annual relative measure
- WMARM: population-weighted MARM (used for identifying the optimal parameters)

The maximum WMARM determines the optimal exposure-threshold/lag combination.

---

## 2. Repository Structure

main.py<br>
conf.py<br>
data_import.py<br>
A_expnonexp_days.py<br>
B_incidence_diff.py<br>
C_compute_index.py<br>
D_post_process.py<br>
E_compute_MARM.py<br>
F_prepare_output.py<br>
G_sensitivity_analysis.py<br>
db_filters.py<br>
multifile_iterator.py<br>
README.md<br>
LICENSE<br>


Run the full workflow by executing:

python main.py

---

## 3. Required Input Data

Place input files in the directory specified by conf.dspath.

### Mandatory files

| File | Description |
|------|-------------|
| exposure_data.csv | Exposure grid with DATE_STR and spatial unit columns |
| outcome_data.csv | Outcome (e.g. health events) time series with DATE_STR |
| {reference_geo_level}.csv | Reference grid with population fields POP_YYYY |
| {source_geo_level}{reference_geo_level}.csv | Cross-grid mapping linking exposure grid and analysis grid |

### Required fields

Exposure and outcome datasets:
- DATE_STR in format %y%m%d
- One numeric column per spatial unit

Reference grid:
- Column matching conf.geoid
- One or more POP_YYYY columns

Cross grid:
- Columns matching conf.source_geoid and conf.geoid
- Area field used for proportional allocation

---

## 4. Configuration (conf.py)

Important settings include:

### Paths
dspath<br>
respath<br>
model_version<br>

### Temporal subsets
years<br>
months<br>
zones<br>


### Exposure threshold and lag search
exposure_percentile_params = [start, end, step]<br>
lag_params = [min, max, step]<br>


### Statistical parameters
bootstrap_iterations<br>
baseline_semiwindow<br>
semiwindow_max<br>
random_noise<br>


### Optional analyses
uncertainty_flag<br>
uncertainty_nvalues<br>
uncertainty_iterations<br>

sens_an_iterations<br>
sensitivity_minmax<br>


### Output control
saveout = 1 or 0


---

## 5. Workflow Executed by main.py

### Step 1. Import and harmonise data
- Load exposure, outcome, reference grid, cross grid
- Restrict to selected years and months
- Harmonise datasets using uniform_data()
- Reproject exposure grid onto analysis grid via cross-grid weights

### Step 2. Parameter sweep
For each combination of exposure percentile and lag:
1. Compute yearly exposure thresholds
2. Identify exposed and non-exposed days
3. Compute incidence and baseline incidence
4. Compute incidence differentials
5. Bootstrap effect estimation and confidence intervals for each year
6. Aggregate across years (median, IQR)
7. Compute MARM and WMARM
8. Save outputs if enabled

Each parameter combination results in a directory:<br>
Parametric/PXX_LYY/


### Step 3. Identify optimal parameters
The parameter pair with the highest WMARM value is identified, and its folder is copied to:<br>
MAX_WMARM_PXX_LYY/


### Step 4. Uncertainty analysis (optional)
Recomputes WMARM distributions for the top N parameter combinations.

### Step 5. Parameter surface plot
A 3D surface plot of WMARM is saved in:<br>
PLOT/


### Step 6. Sensitivity analysis (optional)
Modifies incidence during exposed days by controlled percentages and measures changes in WMARM.

---

## 6. Running the Pipeline

### 1. Install dependencies
pip install pandas numpy scipy matplotlib dbfread<br>


### 2. Configure conf.py
Set paths such as:<br>
dspath = "path/to/data/"<br>
respath = "path/to/results/"<br>
model_version = "v1/"<br>


### 3. Run
python main.py


---

## 7. Output Structure

Example folder structure under respath:<br>

v1/<br>
Parametric/<br>
P90_L3/<br>
index_raw.csv<br>
index_formatted.csv<br>
index_cumulated_raw.csv<br>
index_cumulated_formatted.csv<br>
marm_value.txt<br>
wmarm_value.txt<br>

MAX_WMARM_P90_L3/<br>

PLOT/<br>
WMARM.tiff<br>
WMARM.csv<br>

Years/<br>
exposure_thresholds.csv<br>

Uncertainty_analysis/<br>
Uncertainty_analysis.csv<br>

Sensitivity_analysis/<br>
Sensitivity_analysis.tiff<br>
Sensitivity_analysis_interpolated.tiff<br>


### Main outputs

| File | Meaning |
|------|---------|
| index_raw.csv | Multi-year vulnerability values |
| index_formatted.csv | Readable version including confidence intervals |
| index_cumulated_raw.csv | Median and IQR across years |
| marm_value.txt | MARM summary |
| wmarm_value.txt | WMARM summary |
| WMARM.csv | WMARM for all parameter combinations |
| exposure_thresholds.csv | Yearly thresholds |


## 8. License

This repository is provided under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license.

See the LICENSE file for the full terms.
