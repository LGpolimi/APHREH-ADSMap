import os
import pandas as pd
import shutil
import datetime as dt
import re
import matplotlib.pyplot as plt
import conf
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights
from C_compute_index import compute_index_main
from D_post_process import cumulate_across_years
from E_compute_MARM import compute_marm, compute_wmarm
from F_prepare_output import merge_relevant_info

def deconde_optparams(key):
    match = re.match(r'P(\d+)_L(\d+)', key)
    if match:
        th = int(match.group(1)) / 100
        l = int(match.group(2))
        return th, l
    else:
        raise ValueError("Invalid key format")

def modify_data(exposed_days,incidence_base,change):
    print(f"Modifying incidence data by {change}%")
    pchange = change/100
    modified_incidence = incidence_base.copy()
    exposed_incidence = incidence_base.loc[incidence_base['DATE'].isin(exposed_days)].copy(deep=True)
    exposed_incidence.drop(columns='DATE',inplace=True)
    cumulated_incidence = exposed_incidence.sum().sum()
    cumulated_change = cumulated_incidence * (pchange)
    if change<0:
        convergence = 0
        min_inc = 0.1
        while convergence == 0:
            print(f"Checking convergence with min incidence = {min_inc}")
            valid_cells = exposed_incidence[exposed_incidence > min_inc]
            cell_count = valid_cells[~valid_cells.isna()].count().sum()
            cellwise_change = cumulated_change / cell_count
            error_cells = valid_cells[valid_cells < abs(cellwise_change)]
            if error_cells.isna().all().all():
                convergence = 1
            else:
                min_inc = min_inc + 0.1
    else:
        cell_count = exposed_incidence.count().sum()
        cellwise_change = cumulated_change / cell_count
        valid_cells = exposed_incidence.copy(deep=True)
    for row in valid_cells.index:
        for col in valid_cells.columns:
            if not pd.isna(valid_cells.loc[row,col]):
                modified_incidence.loc[row,col] = valid_cells.loc[row,col] + cellwise_change

    return modified_incidence

def run_sensitivity_analysis(opt_params,base_wmarm):
    import conf

    # SENSITIVITY DATA
    xvector = []
    yvector = []
    marms = dict()
    wmarms = dict()

    # DATA IMPORT
    exposure_grid, outcome, refgrid, crossgrid = import_data()

    # DATA SLICING
    exposure_grid = slice_data(exposure_grid, conf.years, conf.months)
    outcome = slice_data(outcome, conf.years, conf.months, conf.zones)

    # SET PARAMETERS CYCLE
    totcycs = int((conf.sensitivity_minmax[1]-conf.sensitivity_minmax[0])/conf.sensitivity_minmax[2])
    th, l = deconde_optparams(opt_params)
    timelag = dt.timedelta(days=l)
    datachange = conf.sensitivity_minmax[0] - conf.sensitivity_minmax[2]
    for cyc in range(totcycs):
        datachange = datachange + conf.sensitivity_minmax[2]
        xvector.append(datachange)
        out_prefix = 'Sensitivity_analysis\\P' + str(int(th * 100)) + '_L' + str(l) + '\\'
        conf.sens_outprefix = out_prefix
        if not os.path.isdir(conf.outpath+'Sensitivity_analysis\\'):
            os.mkdir(conf.outpath+'Sensitivity_analysis\\')
        if not os.path.isdir(conf.outpath + out_prefix):
            os.mkdir(conf.outpath + out_prefix)
        conf.param_string = 'SENSITIVITY ANALYSIS: Iteration ' + str(cyc) + '/' + str(totcycs) + ' - Sensitivity change = ' + str(datachange) + ' - EXP: ' + str(
            th * 100) + '° perc - LAG: ' + str(l) + '\t'

        # IDENTIFICATION OF EXPOSURE AND NON-EXPOSURE DAYS
        exp_threshold, exposed_days, non_exposed_days = define_exposure_days(exposure_grid, conf.years)

        # COMPUTATION OF INCIDENCE DIFFERENTIALS
        incidence_base = compute_zones_incidence(outcome, refgrid)

        # MODIFY DATA
        incidence = modify_data(exposed_days, incidence_base, datachange)

        incidence_baseline = compute_incidence_baseline(incidence, non_exposed_days)
        incidence_differentials = compute_incidence_differentials(incidence, incidence_baseline)
        exposure = cross_grid_computation(exposure_grid, crossgrid, [conf.source_geoid, conf.geoid],
                                          [conf.cross_area_field, conf.area_field])
        exposure, outcome = uniform_data(exposure, outcome)
        index_df = pd.DataFrame()

        for y in conf.years:
            expth = exp_threshold[y]
            weights = compute_weights(expth, exposure)
            expdays = [ed for ed in exposed_days if pd.to_datetime(ed).year == y]
            nonexpdays = [ned for ned in non_exposed_days if pd.to_datetime(ned).year == y]

            # COMPUTATION OF VULNERABILITY INDEX
            yearly_index_df, yearly_permutations_r, yearly_permutations_p = compute_index_main(
                incidence_differentials, weights, expdays, nonexpdays, y)
            index_df = pd.concat([index_df, yearly_index_df], axis=1)

            if conf.saveout == 1:
                yearly_permutations_r.to_csv(conf.outpath + out_prefix + 'permutations_r_' + str(y) + '.csv')
                yearly_permutations_p.to_csv(conf.outpath + out_prefix + 'permutations_p_' + str(y) + '.csv')

        # POST-PROCESS INDEX ACROSS YEARS
        if len(conf.years) > 1:
            cum_index_df, cum_index_df_formatted = cumulate_across_years(index_df)
            marm_db, marm = compute_marm(index_df)
            key = f"P{int(th * 100)}_L{l}"
            marms[key] = marm
            wmarm = compute_wmarm(refgrid, marm_db)
            wmarms[key] = wmarm
            #wmarm_db.loc[int(th * 100), l] = wmarm

            yvector.append(((wmarm-base_wmarm)/base_wmarm)*100)

            # MERGE OUTPUT INFORMATION
            index_df, index_df_formatted, cum_index_df, cum_index_df_formatted = merge_relevant_info(marm_db,cum_index_df,cum_index_df_formatted)
        # SAVE RESULTS
        if conf.saveout == 1:
            if len(conf.years) > 1:
                index_df.to_csv(conf.outpath + out_prefix + 'index_raw.csv')
                index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_formatted.csv')
                cum_index_df.to_csv(conf.outpath + out_prefix + 'index_cumulated_raw.csv')
                cum_index_df_formatted.to_csv(conf.outpath + out_prefix + 'index_cumulated_formatted.csv',
                                              encoding='utf-8-sig')
                with open(conf.outpath + out_prefix + 'marm_value.txt', 'w') as file:
                    file.write(f'MARM: {marm}')
                with open(conf.outpath + out_prefix + 'wmarm_value.txt', 'w') as file:
                    file.write(f'WMARM: {wmarm}')
        else:
            print(f"ERROR: NO PROCESSABLE DATA FOUND FOR {out_prefix}")

    # PLOT CHART
    plt.figure(figsize=(10, 6))
    plt.plot(xvector, yvector, marker='o', linestyle='-', color='b')
    plt.xlabel('% change in data during exposure')
    plt.ylabel('% change in WMARM')
    plt.title('SENSITIVITY ANALYSIS')
    plt.grid(True)
    plt.show()
    if conf.saveout == 1:
        plt.savefig(conf.outpath + 'Sensitivity_analysis\\Sensitivity_analysis.tiff')

    return