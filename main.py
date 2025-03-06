import os
import pandas as pd
import shutil
import conf
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights
from C_compute_index import compute_index_main
from D_post_process import cumulate_across_years
from E_compute_MARM import compute_marm, compute_wmarm
from F_prepare_output import merge_relevant_info, generate_chart

# DATA IMPORT
exposure_grid, outcome, refgrid, crossgrid = import_data()

# DATA SLICING
exposure_grid = slice_data(exposure_grid,conf.years,conf.months)
outcome = slice_data(outcome,conf.years,conf.months,conf.zones)

# SET PARAMETERS CYCLE
cyc = 0
totcycs = len(conf.exposure_percentile_list)*len(conf.timelag_list)
marms = dict()
wmarms = dict()
wmarm_db = pd.DataFrame()
for th in conf.exposure_percentile_list:
    for l in conf.timelag_list:
        cyc = cyc + 1
        conf.exposure_percentile = th
        conf.lag = l.days
        conf.timelag = l
        out_prefix = 'Parametric\\P' + str(int(th*100)) + '_L' + str(l.days) + '\\'
        conf.out_prefix = out_prefix
        if not os.path.isdir(conf.outpath + out_prefix):
            os.mkdir(conf.outpath + out_prefix)
        conf.param_string = 'Iteration ' + str(cyc) + '/' + str(totcycs) + ' - EXP: ' + str(th*100) + '° perc - LAG: ' + str(l.days) + '\t'

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
                yearly_permutations_r.to_csv(conf.outpath + out_prefix + 'permutations_r_'+str(y)+'.csv')
                yearly_permutations_p.to_csv(conf.outpath + out_prefix + 'permutations_p_'+str(y)+'.csv')

# POST-PROCESS INDEX ACROSS YEARS
        if len(conf.years) > 1:
            cum_index_df, cum_index_df_formatted = cumulate_across_years(index_df)
            marm_db, marm = compute_marm(index_df)
            key = f"P{int(th * 100)}_L{l.days}"
            marms[key] = marm
            wmarm = compute_wmarm(refgrid,marm_db)
            wmarms[key] = wmarm
            wmarm_db.loc[int(th * 100),l.days] = wmarm

# MERGE OUTPUT INFORMATION
            index_df, index_df_formatted, cum_index_df, cum_index_df_formatted = merge_relevant_info(marm_db,cum_index_df,cum_index_df_formatted)

        # SAVE RESULTS
        if conf.saveout == 1:
            if len(conf.years) > 1:
                index_df.to_csv(conf.outpath + out_prefix + 'index_raw.csv')
                index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_formatted.csv')
                cum_index_df.to_csv(conf.outpath + out_prefix + 'index_cumulated_raw.csv')
                cum_index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_cumulated_formatted.csv',encoding='utf-8-sig')
                with open(conf.outpath + out_prefix + 'marm_value.txt', 'w') as file:
                    file.write(f'MARM: {marm}')
                with open(conf.outpath + out_prefix + 'wmarm_value.txt', 'w') as file:
                    file.write(f'WMARM: {wmarm}')
        else:
            print(f"ERROR: NO PROCESSABLE DATA FOUND FOR {out_prefix}")

# INDENTIFY MOST RELEVANT RESULTS
# Identify the combination of th and l with the max wmarm value
max_wmarm_key = max(wmarms, key=wmarms.get)
max_wmarm_value = wmarms[max_wmarm_key]
# Replicate the folder with the results and rename the subfolder
source_folder = os.path.join(conf.outpath, max_wmarm_key)
destination_folder = os.path.join(conf.outpath, 'MAX_WMARM_' + max_wmarm_key)
if os.path.exists(source_folder):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
        print(f"Deleted existing folder {destination_folder}")
    shutil.copytree(source_folder, destination_folder)
    print(f"Replicated folder {max_wmarm_key} with max WMARM value: {max_wmarm_value}")
else:
    print(f"Source folder {source_folder} does not exist.")

# Plot the 3D chart
generate_chart(wmarm_db)
br = 1