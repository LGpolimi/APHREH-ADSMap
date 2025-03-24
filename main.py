import os
import sys
import pandas as pd
import shutil
import copy
import conf
from conf import global_struct, parametric_struct, yearly_struct, global_results
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights
from C_compute_index import compute_index_main
from D_post_process import cumulate_across_years
from E_compute_MARM import compute_marm, compute_wmarm, identify_max_wmarm, run_uncertainty_analysis, save_variables, decode_params
from F_prepare_output import merge_relevant_info, generate_chart
from G_sensitivity_analysis import run_sensitivity_analysis

# DATA IMPORT
setup_data = global_struct()
setup_data.exposure_grid, setup_data.outcome, setup_data.refgrid, setup_data.crossgrid = import_data()

# DATA SLICING
setup_data.exposure_grid = slice_data(setup_data.exposure_grid,conf.years,conf.months)
setup_data.outcome = slice_data(setup_data.outcome,conf.years,conf.months,conf.zones)
setup_data.exposure = cross_grid_computation(setup_data.exposure_grid,setup_data.crossgrid,[conf.source_geoid,conf.geoid],[conf.cross_area_field,conf.area_field])
setup_data.exposure, setup_data.outcome = uniform_data(setup_data.exposure,setup_data.outcome)

# SET PARAMETERS CYCLE
cyc = 0
totcycs = len(conf.exposure_percentile_list)*len(conf.lag_list)
main_res = global_results()
'''marms = dict()
wmarms = dict()
wmarm_db = pd.DataFrame()
max_wmarm = 0'''
main_ds = list()

for th in conf.exposure_percentile_list:
    for l in conf.timelag_list:
        cyc = cyc + 1
        conf.exposure_percentile = th
        conf.lag = l.days
        conf.timelag = l
        out_prefix = 'Parametric\\P' + str(int(th*100)) + '_L' + str(l.days) + '\\'
        conf.out_prefix = out_prefix
        if not os.path.isdir(conf.outpath + 'Parametric\\'):
            os.mkdir(conf.outpath + 'Parametric\\')
        if not os.path.isdir(conf.outpath + out_prefix):
            os.mkdir(conf.outpath + out_prefix)
        conf.process_string = 'MAIN Iteration ' + str(cyc) + '/' + str(totcycs) + ' - '
        conf.param_string = conf.process_string + 'EXP: ' + str(th*100) + '° perc - LAG: ' + str(l.days) + '\t'

        it_ds = parametric_struct()
        it_ds.th=th
        it_ds.lag=l.days

# IDENTIFICATION OF EXPOSURE AND NON-EXPOSURE DAYS
        it_ds.exp_threshold, it_ds.exposed_days, it_ds.non_exposed_days = define_exposure_days(setup_data.exposure_grid,conf.years,1)

# COMPUTATION OF INCIDENCE DIFFERENTIALS
        it_ds.incidence = compute_zones_incidence(setup_data.outcome,setup_data.refgrid)
        it_ds.incidence_baseline = compute_incidence_baseline(it_ds.incidence,it_ds.non_exposed_days)
        it_ds.incidence_differentials = compute_incidence_differentials(it_ds.incidence,it_ds.incidence_baseline)
        it_ds.index_df = pd.DataFrame()

        for y in conf.years:
            y_ds = yearly_struct()
            y_ds.year = y
            y_ds.expth = it_ds.exp_threshold[y]
            y_ds.weights = compute_weights(y_ds.expth,setup_data.exposure)
            y_ds.expdays = [ed for ed in it_ds.exposed_days if pd.to_datetime(ed).year == y]
            y_ds.nonexpdays = [ned for ned in it_ds.non_exposed_days if pd.to_datetime(ned).year == y]

            # COMPUTATION OF VULNERABILITY INDEX
            y_ds.yearly_index_df, y_ds.yearly_permutations_r, y_ds.yearly_permutations_p = compute_index_main(it_ds.incidence_differentials,y_ds.weights,y_ds.expdays,y_ds.nonexpdays,y)
            it_ds.index_df = pd.concat([it_ds.index_df, y_ds.yearly_index_df], axis=1)

            if conf.saveout == 1:
                y_ds.yearly_permutations_r.to_csv(conf.outpath + out_prefix + 'permutations_r_'+str(y)+'.csv')
                y_ds.yearly_permutations_p.to_csv(conf.outpath + out_prefix + 'permutations_p_'+str(y)+'.csv')

            it_ds.yearly_ds.append(copy.deepcopy(y_ds))

        main_ds.append(copy.deepcopy(it_ds))

# POST-PROCESS INDEX ACROSS YEARS
        if len(conf.years) > 1:

            it_ds.results.cum_index_df, it_ds.cum_index_df_formatted = cumulate_across_years(it_ds.index_df)
            it_ds.results.stdeff_db, marm = compute_marm(it_ds.index_df)
            main_res.marm_db.loc[int(th * 100), l.days] = marm
            key = f"P{int(th * 100)}_L{l.days}"
            main_res.marms[key] = marm
            wmarm = compute_wmarm(setup_data.refgrid,it_ds.results.stdeff_db)
            main_res.wmarms[key] = wmarm
            main_res.wmarm_db.loc[int(th * 100),l.days] = wmarm
            if wmarm > main_res.max_wmarm:
                main_res.max_wmarm = wmarm
                conf.max_wmarm_ds = copy.deepcopy(it_ds)
                #save_variables(exp_threshold, exposed_days, non_exposed_days,incidence,incidence_baseline)

# MERGE OUTPUT INFORMATION
            it_ds.results.out_index_df, it_ds.results.out_index_df_formatted, it_ds.results.cum_index_df, it_ds.results.cum_index_df_formatted = merge_relevant_info(it_ds.results.stdeff_db,it_ds.results.cum_index_df,it_ds.results.cum_index_df_formatted)

        # SAVE RESULTS
        if conf.saveout == 1:
            if len(conf.years) > 1:
                it_ds.results.out_index_df.to_csv(conf.outpath + out_prefix + 'index_raw.csv')
                it_ds.results.out_index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_formatted.csv')
                it_ds.results.cum_index_df.to_csv(conf.outpath + out_prefix + 'index_cumulated_raw.csv')
                it_ds.results.cum_index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_cumulated_formatted.csv',encoding='utf-8-sig')
                with open(conf.outpath + out_prefix + 'marm_value.txt', 'w') as file:
                    file.write(f'MARM: {marm}')
                with open(conf.outpath + out_prefix + 'wmarm_value.txt', 'w') as file:
                    file.write(f'WMARM: {wmarm}')
        else:
            print(f"ERROR: NO PROCESSABLE DATA FOUND FOR {out_prefix}")

# INDENTIFY MOST RELEVANT RESULTS
main_res.opt_params, temp_max_wmarm = identify_max_wmarm(main_res.wmarms)
if temp_max_wmarm != main_res.max_wmarm:
    print('ERROR: MAX WMARM VALUE NOT MATCHING')
    sys.exit("Stopping execution")

# RUN UNCERTAINTY ANALYSIS
if conf.uncertainty_flag == 1:
    uncert_res = global_results()
    uncert_res.wmarms, uncert_res.wmarm_db, uncert_res.max_wmarm = run_uncertainty_analysis(main_res.wmarms,main_res.wmarm_db,main_res.max_wmarm,main_ds,setup_data.refgrid)

    # INDENTIFY MOST RELEVANT RESULTS
    uncert_res.opt_params, uncert_res.max_wmarm = identify_max_wmarm(uncert_res.wmarms)
    if uncert_res.max_wmarm != main_res.max_wmarm:
        print('MAX WMARM VALUE UPDATED')
        th,l = decode_params(uncert_res.opt_params)
        conf.exposure_percentile = th
        conf.lag = l
        for ds in main_ds:
            if ds.th == th and ds.lag == l:
                conf.max_wmarm_ds = copy.deepcopy(ds)

    # UPDATE RESULTS WITH UNCERTAINTY ANALYSIS
    updated_main_res = copy.deepcopy(main_res)
    updated_main_res.integrate_results(uncert_res)
else:
    updated_main_res = main_res

# PLOT 3D CHART
generate_chart(updated_main_res.wmarm_db)

# RUN SENSITIVITY ANALYSIS
if conf.sensan_flag == 1 and len(conf.years)>1:
    run_sensitivity_analysis(updated_main_res.opt_params, updated_main_res.max_wmarm,conf.max_wmarm_ds,setup_data)
br = 1