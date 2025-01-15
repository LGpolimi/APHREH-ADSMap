import os
from datetime import timedelta

# PARAMETERS SETTING

dspath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\datasource\\analysis_ready\\'
outpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\results\\'
if not os.path.isdir(outpath):
    os.mkdir(outpath)

exposure_db_name = 'exposure_data.csv'
outcome_db_name = 'outcome_data.csv'
reference_geo_level = 'MIL2A'
geoid = reference_geo_level + '_IDcu'
incidence_popmultiplier = 100000

years = [2019]
zones = [1,2,3,10,12,15]

exposure_percentile = 0.9
lag = 2
timelag = timedelta(days=lag)
baseline_semiwindow = 15
dynawindow = 1
