import os
import pandas as pd
import conf
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights

# DATA IMPORT
exposure_grid, outcome, refgrid, crossgrid = import_data()

# DATA SLICING
exposure_grid = slice_data(exposure_grid,conf.years,conf.months)
outcome = slice_data(outcome,conf.years,conf.months,conf.zones)

# IDENTIFICATION OF EXPOSURE AND NON-EXPOSURE DAYS
exp_threshold, exposed_days, non_exposed_days = define_exposure_days(exposure_grid,conf.years)

# COMPUTATION OF INCIDENCE DIFFERENTIALS
incidence = compute_zones_incidence(outcome,refgrid)
incidence_baseline = compute_incidence_baseline(incidence,non_exposed_days)
incidence_differentials = compute_incidence_differentials(incidence,incidence_baseline)
exposure = cross_grid_computation(exposure_grid,crossgrid,[conf.source_geoid,conf.geoid],[conf.cross_area_field,conf.area_field])
exposure, outcome = uniform_data(exposure,outcome)

for y in conf.years:
    expth = exp_threshold[y]
    weights = compute_weights(expth,exposure)

    # COMPUTATION OF VULNERABILITY INDEX


    br = 1