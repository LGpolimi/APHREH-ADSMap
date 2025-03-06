import pandas as pd
import numpy as np
import os
import csv
import conf

def save_dict_to_csv(dictionary, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Year', 'Threshold'])
        for key, value in dictionary.items():
            writer.writerow([key, value])

def define_exposure_days(exposure,years):

    import conf

    expthresholds = dict()
    exposed_days = list()
    non_exposed_days = list()
    for y in years:
        expproc = exposure.loc[exposure['DATE'].dt.year == y].copy(deep=True)
        exptemp = expproc.copy(deep=True)
        exposure_nodate = exptemp.drop(columns='DATE').copy(deep=True)
        exposure_nodate['AVG'] = exposure_nodate.mean(axis=1)
        expproc['AVG'] = exposure_nodate['AVG']
        expavg_ts = np.asarray(expproc['AVG'])
        exp_threshold = np.percentile(expavg_ts,conf.exposure_percentile*100)
        expthresholds[y] = exp_threshold

        exposed_days_db = expproc.loc[expproc['AVG'] >= exp_threshold]
        y_exposed_days = list(exposed_days_db['DATE'])
        non_exposed_days_db = expproc.loc[expproc['AVG'] < exp_threshold]
        y_non_exposed_days = list(non_exposed_days_db['DATE'])
        for ed in y_exposed_days:
            exposed_days.append(ed)
        for ned in y_non_exposed_days:
            non_exposed_days.append(ned)

    if not os.path.isdir(conf.yearly_folder):
        os.mkdir(conf.yearly_folder)
    save_dict_to_csv(expthresholds,conf.yearly_folder+'exposure_thresholds.csv')

    return expthresholds, exposed_days, non_exposed_days