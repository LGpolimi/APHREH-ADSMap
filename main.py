import os
import pandas as pd
import conf
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights
from C_compute_index import compute_index_main

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
index_df = pd.DataFrame()

for y in conf.years:
    expth = exp_threshold[y]
    weights = compute_weights(expth,exposure)
    expdays = [ed for ed in exposed_days if pd.to_datetime(ed).year == y]
    nonexpdays = [ned for ned in non_exposed_days if pd.to_datetime(ned).year == y]

    # COMPUTATION OF VULNERABILITY INDEX
    yearly_index_df, yearly_permutations_r, yearly_permutations_p = compute_index_main(incidence_differentials,weights,expdays,nonexpdays,y)
    index_df = pd.concat([index_df, yearly_index_df], axis=1)

    if conf.saveout == 1:
        yearly_permutations_r.to_csv(conf.outpath + conf.output_prefix + 'permutations_r_'+str(y)+'.csv')
        yearly_permutations_p.to_csv(conf.outpath + conf.output_prefix + 'permutations_p_'+str(y)+'.csv')
if conf.saveout == 1:
    index_df.to_csv(conf.outpath + conf.output_prefix + 'index_raw.csv')
    formatted_df = pd.DataFrame(index=index_df.index)
    for y in conf.years:
        proc_index = index_df.copy(deep=True)
        proc_index = proc_index.apply(lambda x: "{:.2f}".format(x))
        formatted_df[y] = proc_index.apply(lambda row: f"{row[str(y) + 'INDEX']} ({row[str(y) + 'CI_LOW']}|{row[str(y) + 'CI_HIGH']})", axis=1)
    formatted_df.to_csv(conf.outpath + conf.output_prefix + 'index_formatted.csv')

br = 1