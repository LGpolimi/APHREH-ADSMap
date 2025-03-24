import os
import pandas as pd
import numpy as np
import shutil
import datetime as dt
import re
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import conf
from conf import sens_non_exposed_days, sens_exposed_days
from data_import import import_data, slice_data, uniform_data
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, cross_grid_computation, compute_weights
from C_compute_index import compute_index_main
from D_post_process import cumulate_across_years
from E_compute_MARM import compute_marm, compute_wmarm, decode_params, save_variables
from F_prepare_output import merge_relevant_info


def modify_data(exposed_days,incidence_base,change):
    print(f"Modifying incidence data by {change}%")
    pchange = change/100
    modified_incidence = incidence_base.copy(deep=True)
    exposed_incidence = incidence_base.loc[incidence_base['DATE'].isin(exposed_days)].copy(deep=True)
    exposed_incidence.drop(columns='DATE',inplace=True)
    for row in exposed_incidence.index:
        for col in exposed_incidence.columns:
            if not pd.isna(exposed_incidence.loc[row, col]):
                baseval = exposed_incidence.loc[row, col]
                modified_incidence.loc[row, col] =  baseval + (baseval*pchange)

    #DEBUG
    print(f"Base cumulated incidence - exposed: {incidence_base.loc[incidence_base['DATE'].isin(exposed_days)].drop(columns=['DATE']).sum().sum()}")
    print(f"Modified cumulated incidence - exposed: {modified_incidence.loc[modified_incidence['DATE'].isin(exposed_days)].drop(columns=['DATE']).sum().sum()}")
    print(f"Base cumulated incidence - non exposed: {incidence_base.loc[~incidence_base['DATE'].isin(exposed_days)].drop(columns=['DATE']).sum().sum()}")
    print(f"Modified cumulated incidence - non exposed: {modified_incidence.loc[~modified_incidence['DATE'].isin(exposed_days)].drop(columns=['DATE']).sum().sum()}")

    br = 1

    return modified_incidence

def run_sensitivity_analysis(opt_params,base_wmarm,param_ds,setup_data):
    import conf
    # IMPORT DATA STRUCTURE
    exp_threshold = param_ds.exp_threshold
    exposed_days = param_ds.exposed_days
    non_exposed_days = param_ds.non_exposed_days
    incidence_base = param_ds.incidence
    incidence_baseline = param_ds.incidence_baseline

    # SENSITIVITY DATA
    yvector_df = pd.DataFrame(columns=['p2.5', 'p25', 'p50', 'p75', 'p97.5'])
    marms = dict()
    wmarms = dict()

    '''# DATA IMPORT
    exposure_grid, outcome, refgrid, crossgrid = import_data()

    # DATA SLICING
    exposure_grid = slice_data(exposure_grid, conf.years, conf.months)
    outcome = slice_data(outcome, conf.years, conf.months, conf.zones)'''

    # SET PARAMETERS CYCLE
    totcycs = int((conf.sensitivity_minmax[1]-conf.sensitivity_minmax[0])/conf.sensitivity_minmax[2])+1
    totiters = totcycs * conf.sens_an_iterations
    th, l = decode_params(opt_params)
    conf.exposure_percentile = th
    conf.lag = l
    timelag = dt.timedelta(days=l)
    conf.process_string = 'INIT SENSITIVITY ANALYSIS ON EXP: '
    conf.param_string =  conf.process_string + str(th*100) + '° perc - LAG: ' + str(l)
    datachange = conf.sensitivity_minmax[0] - conf.sensitivity_minmax[2]
    if not os.path.isdir(conf.outpath + 'Sensitivity_analysis\\'):
        os.mkdir(conf.outpath + 'Sensitivity_analysis\\')


    # RUN CYCLES
    xvector = []
    ncycs = range(totcycs)
    for cyc in ncycs:
        datachange = datachange + conf.sensitivity_minmax[2]
        yvector = []
        xvector.append(datachange)
        # MODIFY DATA
        incidence = modify_data(exposed_days, incidence_base, datachange)
        # COMPUTATION OF INCIDENCE DIFFERENTIALS
        incidence_differentials = compute_incidence_differentials(incidence, incidence_baseline)
        for it in range(conf.sens_an_iterations):
            out_prefix = 'Sensitivity_analysis\\P' + str(int(th * 100)) + '_L' + str(l) + '_'+str(datachange)+'\\'
            conf.sens_outprefix = out_prefix
            conf.param_string = 'SENSITIVITY ANALYSIS: Iteration ' + str(it+(cyc*conf.sens_an_iterations)) + '/' + str(totiters) + ' - Sensitivity change = ' + str(datachange) + ' - EXP: ' + str(
                th * 100) + '° perc - LAG: ' + str(l) + '\t'
            index_df = pd.DataFrame()
            for y in conf.years:
                for yi in param_ds.yearly_ds:
                    if yi.year == y:
                        weights = yi.weights
                        expdays = yi.expdays
                        nonexpdays = yi.nonexpdays

                # COMPUTATION OF VULNERABILITY INDEX
                yearly_index_df, yearly_permutations_r, yearly_permutations_p = compute_index_main(
                    incidence_differentials, weights, expdays, nonexpdays, y)
                index_df = pd.concat([index_df, yearly_index_df], axis=1)

        # POST-PROCESS INDEX ACROSS YEARS
            cum_index_df, cum_index_df_formatted = cumulate_across_years(index_df)
            marm_db, marm = compute_marm(index_df)
            key = f"P{int(th * 100)}_L{l}"
            marms[key] = marm
            wmarm = compute_wmarm(setup_data.refgrid, marm_db)
            wmarms[key] = wmarm

            yvector.append(((wmarm-base_wmarm)/base_wmarm)*100)

        yp2 = np.quantile(yvector,0.025)
        yp25 = np.quantile(yvector,0.25)
        yp50 = np.quantile(yvector,0.5)
        yp75 = np.quantile(yvector,0.75)
        yp97 = np.quantile(yvector,0.975)
        yvector_df.loc[cyc] = [yp2, yp25, yp50, yp75, yp97]

    # PLOT CHART 1
    plt.figure(figsize=(10, 6))
    plt.plot(xvector, yvector_df['p2.5'], marker='o', linestyle='-', color='red', label='2.5%')
    plt.plot(xvector, yvector_df['p25'], marker='o', linestyle='-', color='lightblue', label='25%')
    plt.plot(xvector, yvector_df['p50'], marker='o', linestyle='-', color='darkgreen', label='50%')
    plt.plot(xvector, yvector_df['p75'], marker='o', linestyle='-', color='blue', label='75%')
    plt.plot(xvector, yvector_df['p97.5'], marker='o', linestyle='-', color='darkred', label='97.5%')
    plt.xlabel('% change in outcome during exposure')
    plt.ylabel('% change in WMARM')
    plt.title('SENSITIVITY ANALYSIS')
    plt.grid(True)
    # Set the position of the spines
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Add arrows
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    if conf.saveout == 1:
        plt.savefig(conf.outpath + 'Sensitivity_analysis\\Sensitivity_analysis.tiff')
    plt.show()

    # PLOT CHART 2
    # Compute spline interpolations
    splines_df = pd.DataFrame(columns=yvector_df.columns)
    x_spline = np.linspace(min(xvector), max(xvector), 500)
    for col in yvector_df.columns:
        yvec = yvector_df[col].values
        cs = CubicSpline(xvector, yvec, bc_type='natural')
        y_spline = cs(x_spline)
        splines_df[col] = y_spline
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_spline, splines_df['p2.5'], linestyle='-', color='red', label='2.5%')
    plt.plot(x_spline, splines_df['p25'], linestyle='-', color='lightblue', label='25%')
    plt.plot(x_spline, splines_df['p50'], linestyle='-', color='darkgreen', label='50%')
    plt.plot(x_spline, splines_df['p75'], linestyle='-', color='blue', label='75%')
    plt.plot(x_spline, splines_df['p97.5'], linestyle='-', color='darkred', label='97.5%')
    plt.xlabel('% change in outcome during exposure')
    plt.ylabel('% change in WMARM')
    plt.title('SENSITIVITY ANALYSIS - INTERPOLATED CURVES')
    plt.grid(True)
    # Set the position of the spines
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Add arrows
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    if conf.saveout == 1:
        plt.savefig(conf.outpath + 'Sensitivity_analysis\\Sensitivity_analysis_interpolated.tiff')
    plt.show()

    return