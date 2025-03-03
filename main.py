import os
import pandas as pd
import shutil
import conf
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights
from C_compute_index import compute_index_main
from D_post_process import cumulate_across_years
from E_compute_MARM import compute_marm

# DATA IMPORT
exposure_grid, outcome, refgrid, crossgrid = import_data()

# DATA SLICING
exposure_grid = slice_data(exposure_grid,conf.years,conf.months)
outcome = slice_data(outcome,conf.years,conf.months,conf.zones)

# SET PARAMETERS CYCLE
cyc = 0
totcycs = len(conf.exposure_percentile_list)*len(conf.timelag_list)
marms = dict()
for th in conf.exposure_percentile_list:
    for l in conf.timelag_list:
        cyc = cyc + 1
        conf.exposure_percentile = th
        conf.lag = l.days
        conf.timelag = l
        out_prefix = 'P' + str(int(th*100)) + '_L' + str(l.days) + '//'
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

# SAVE RESULTS PART 1
        if conf.saveout == 1:
            index_df.to_csv(conf.outpath + out_prefix + 'index_raw.csv')
            formatted_df = pd.DataFrame(index=index_df.index)
            for y in conf.years:
                proc_index = index_df.copy(deep=True)
                proc_index = proc_index.applymap(lambda x: "{:.2f}".format(x))
                formatted_df[y] = proc_index.apply(lambda row: f"{row[str(y) + 'INDEX']} ({row[str(y) + 'CI_LOW']}|{row[str(y) + 'CI_HIGH']})", axis=1)
            formatted_df.to_csv(conf.outpath + out_prefix + 'index_formatted.csv')

# POST-PROCESS INDEX ACROSS YEARS
        if len(conf.years) > 1:
            cum_index_df, cum_index_df_formatted = cumulate_across_years(index_df)
            marm_db, marm = compute_marm()
            key = f"P{int(th * 100)}_L{l.days}"
            marms[key] = marm

# SAVE RESULTS PART 2
        if conf.saveout == 1:
            if len(conf.years) > 1:
                cum_index_df.to_csv(conf.outpath + out_prefix + 'index_cumulated_raw.csv')
                cum_index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_cumulated_formatted.csv',encoding='utf-8-sig')
                marm_db.to_csv(conf.outpath + out_prefix + 'standard_effects_raw.csv')
                with open(conf.outpath + out_prefix + 'marm_value.txt', 'w') as file:
                    file.write(f'MARM: {marm}')

# INDENTIFY MOST RELEVANT RESULTS
# Identify the combination of th and l with the max marm value
max_marm_key = max(marms, key=marms.get)
max_marm_value = marms[max_marm_key]
# Replicate the folder with the results and rename the subfolder
source_folder = os.path.join(conf.outpath, max_marm_key)
destination_folder = os.path.join(conf.outpath, 'MAXMARM_' + max_marm_key)
if os.path.exists(source_folder):
    shutil.copytree(source_folder, destination_folder)
    print(f"Replicated folder {source_folder} to {destination_folder} with max MARM value: {max_marm_value}")
else:
    print(f"Source folder {source_folder} does not exist.")
br = 1