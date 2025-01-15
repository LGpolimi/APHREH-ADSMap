import os
import pandas as pd
import conf
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials

# DATA IMPORT
exposure, outcome, refgrid = import_data()

# DATA SLICING
exposure = slice_data(exposure,conf.years,conf.zones)
outcome = slice_data(outcome,conf.years,conf.zones)
exposure, outcome = uniform_data(exposure,outcome)

# IDENTIFICATION OF EXPOSURE AND NON-EXPOSURE DAYS
exp_threshold, exposed_days, non_exposed_days = define_exposure_days(exposure,conf.years)

# COMPUTATION OF INCIDENCE DIFFERENTIALS
incidence = compute_zones_incidence(outcome,refgrid)
incidence_baseline = compute_incidence_baseline(incidence,non_exposed_days)
incidence_differentials = compute_incidence_differentials(incidence,incidence_baseline)

br = 1