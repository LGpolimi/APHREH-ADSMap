import pandas as pd
import numpy as np
import conf

def define_exposure_days(exposure,years):

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

    return expthresholds, exposed_days, non_exposed_days