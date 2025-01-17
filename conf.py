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
area_field = reference_geo_level + '_Area'
incidence_popmultiplier = 100000
source_geo_level = 'MIL1B'
source_geoid = source_geo_level + '_IDcu'
cross_area_field = 'Area'

years = [2019,2023]
months = [5,6,7,8,9]
zones = ['ALL']

exposure_nullvalues = [-1,-9999]
outcome_nullvalues = [-1,-9999]
exposure_percentile = 0.95
lag = 2
timelag = timedelta(days=lag)
baseline_semiwindow = 15
dynawindow = 1
semiwindow_max = 60
bootstrap_iterations = 100
random_noise = 0.1