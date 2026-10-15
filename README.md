# APHREH-ADSMap 2.0.1

APHREH-ADSMap is a desktop application for spatial epidemiological analysis of associations between environmental hazards and health outcomes. It provides threshold-lag analyses, BSA-level risk coefficients, uncertainty-aware pooling, parameter optimization, synthetic calibration, robustness/sensitivity workflows, candidate-confounder adjustment and an interactive results workspace.

This distribution contains the executable Python sources and no datasets. Supply your own study-specific input data.

## Requirements

* Windows, macOS or Linux with a desktop environment and Python 3.11 to 3.13 (64-bit recommended).
* Python packages `numpy`, `scipy`, `psutil` and `matplotlib` from `requirements.txt`.
* Python Tkinter for the desktop interface. Tkinter is commonly included in standard Python installers for Windows and macOS; on some Linux distributions it must be installed separately (for example, `python3-tk`).
* Sufficient free disk space and available RAM for the study size. Larger threshold-lag grids and Monte Carlo counts can require substantial computation.

The application does not require QGIS, PyQGIS, Qt, a GIS server, or a database server. Source geometry for maps is optional; when supplied, the results workspace supports GeoJSON and GeoPackage.

## Install and start

Extract the archive into a directory of your choice. Open a terminal in the directory containing `main.py`, then create a virtual environment and install the dependencies.

Windows (PowerShell, using Python 3.12; substitute 3.11 or 3.13 if appropriate):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

macOS / Linux (check that `python3 --version` reports Python 3.11, 3.12 or 3.13):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Use the **New project** form to select the folder containing your input files and the destination project directory. Select **Data sources and field mapping > Configure** to set the four required CSV files and their exact column names. The values in `configs/default.json` are generic placeholders, not an included example dataset.

A project directory is created when you start an analysis. Do not reuse an output directory for another dataset or for altered reference scientific settings. For an existing project, choose **Open project** and select the directory containing `resolved_config.json` and `results.sqlite`.

## Required input structure

Provide four CSV tables using the column mappings in **Data sources and field mapping**:

1. **Hazard**: one date column and spatial-grid exposure columns, with grid identifiers matching the crosswalk.
2. **Outcome**: one date column and one column per BSA, named exactly as the BSA identifiers in the population table. The observed outcome is **daily integer event counts**, not pre-normalized incidence.
3. **Population**: one row per BSA, including BSA identifier, area and population estimates mapped to one or more years.
4. **Crosswalk**: BSA-to-grid intersections, identifying BSA, grid cell and intersection area.

The hazard grid column names must match the crosswalk grid identifiers, and each BSA must have strictly positive intersection-area coverage. The exact wide-table conventions and source checks are enforced by the application when running. Set column mappings to your own files rather than trying to rename study data to match placeholders. Hazard and outcome dates must overlap exactly. Supported date formats include `YYYY-MM-DD`, `YYYYMMDD` and six-digit `YYMMDD` (interpreted as years 20YY).

Optional inputs are geometry (GeoPackage or GeoJSON for spatial visualization), an external holiday calendar (when holiday matching is enabled), and candidate-confounder hazard/crosswalk CSV sources (when confounder adjustment is requested).

All source paths can be selected in the interface or specified in a JSON configuration. Paths are evaluated on the machine where the project is executed. Avoid copying another machine's absolute paths into your configuration.

## Running an analysis

1. Select **New project**, then set an input directory and a new output directory.
2. Configure source tables and verify their column mappings, analysis thresholds, cumulative lags, Monte Carlo counts and population-year policy.
3. Use the relevant configuration panels to review calibration, robustness/sensitivity and confounder adjustment settings.
4. Choose the workflow and select **Save project and run**. The console reports progress; durable results and checkpoints are written under the project directory.
5. Select **Results workspace** to inspect surfaces, diagnostics, calibration, spatial maps, BSA-specific results and available exports. Completed projects can be reopened without rerunning the model.

The standard reference analysis uses exact common hazard/outcome dates. Missing intervening days are not silently treated as consecutive. If a population estimate is unavailable for an analysis year, the `ANNUAL_NEAREST` population policy chooses the nearest configured population year, with ties resolved toward the more recent year. Check whether this assumption is appropriate for your study.

A production study should select statistically and computationally appropriate values for bootstrap iterations, null searches, calibration realizations, support constraints and robustness checks. Template defaults are starting settings, not a guarantee of inferential adequacy. Robustness/sensitivity Analyses 2–5 are not started by default unless configured or explicitly launched from **Run robustness / sensitivity**.

### Batch execution without the desktop interface

For example, after editing `configs/default.json` or preparing your own configuration:

```bash
python main.py --headless --config configs/default.json --input /path/to/input --output /path/to/project --phase all
```

Replace the two example paths with locations on your machine. On Windows, use Windows paths. Supported phases are `all`, `reference`, `calibration`, `robustness` and `confounder`. `robustness` and `confounder` require a compatible, previously completed reference project.

To review command-line options use `python main.py --headless --help`.

## Output and checkpoint management

Each run maintains a project `manifest.json`, `resolved_config.json`, canonical data, `results.sqlite`, generated results, execution logs and checkpoint state. Treat `results.sqlite` and the `canonical` directory as part of the same project; do not delete or edit either while a project is active. The results workspace can open results created in another session. Resume the same project only with compatible code, inputs and reference configuration; unsupported changes are deliberately blocked to protect reproducibility.

On machines without a graphical desktop, headless execution remains available but the interactive Results workspace does not open. Matplotlib supports saving BSA-specific figures when invoked through the workspace on a desktop system.

## Methodological references

The authoritative mathematical documents are published together at **[APHREH-ADSMap mathematical formulation, Zenodo DOI 10.5281/zenodo.18671472](https://doi.org/10.5281/zenodo.18671472)**. They are not duplicated in this software archive. The internal configuration schema identifier is retained for compatibility with previously saved projects. The filenames below are the canonical document names specified by the accompanying DOI register.

| Module | Exact document in the Zenodo repository | Relevant formulation / section pointers |
| --- | --- | --- |
| MF, core mathematical formulation | `APHREH_ADSMap_P03_MF_mathematical_formulation.pdf` | core formulation, source harmonization, temporal dependence and estimator; MF §3.2 (dependence), MF §4.1 (pooling) |
| PO, parameter optimization | `APHREH_ADSMap_P03_PO_parameter_optimization.pdf` | leave-one-year-out surfaces, synchronized null searches, search-adjusted support, robust-region selection and fingerprint |
| RS, robustness and sensitivity | `APHREH_ADSMap_P03_RS_robustness_sensitivity.pdf` | stability, support, BSA-specific exposure, baseline and dependence sensitivity, fixed-point comparison and reoptimization |
| CA, confounder adjustment | `APHREH_ADSMap_P03_CA_confounder_adjustment.pdf` | cross-fitting and candidate profiling; CA §§5.1, 7.3, 8.1, 9.2, 9.4, 9.6, 16.2 and 18 are cross-referenced in the implementation |
| CM, calibration | `APHREH_ADSMap_P03_CM_calibration.pdf` | RC-magnitude calibration and STEP/LINEAR_RAMP threshold-lag calibration, including synthetic outcome generation and recovery |

Section pointers above are implementation cross-references. Consult the PDFs for their complete formulations and authoritative section numbering. For publication-related reproducibility material, see [publication supplementaries, DOI 10.5281/zenodo.18671558](https://doi.org/10.5281/zenodo.18671558).

## Distribution contents

```text
main.py                    desktop / headless launcher
requirements.txt           Python dependencies
configs/default.json       generic editable configuration template
aphreh_adsmap/             runtime code (estimator, GUI, results, calibration,
                           optimization, robustness and confounder modules)
README.md                  installation, usage and methodology references
```
