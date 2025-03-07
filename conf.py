import os
from datetime import timedelta

# PARAMETERS SETTING

model_version = 'V3devall'
dspath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\25 03 - V2\\Set_up_I\\datasource\\analysis_ready\\'
respath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\25 03 - V2\\Set_up_I\\results\\'
outpath = respath + model_version + '\\'
if not os.path.isdir(respath):
    os.mkdir(respath)
if not os.path.isdir(outpath):
    os.mkdir(outpath)
out_prefix = ''
yearly_folder = outpath + 'Years\\'
saveout = 1
sensan_flag = 1 # Flag to launch sensitivity analysis

exposure_db_name = 'exposure_data.csv'
outcome_db_name = 'outcome_data.csv'
reference_geo_level = 'MIL2A'
geoid = reference_geo_level + '_IDcu'
area_field = reference_geo_level + '_Area'
incidence_popmultiplier = 100000
source_geo_level = 'MIL1B'
source_geoid = source_geo_level + '_IDcu'
cross_area_field = 'Area'

years = [2017,2018,2019,2020,2021]
months = [5,6,7,8,9]
zones = ['ALL'] # Set to 'ALL' to analyze all BSAs

exposure_nullvalues = [-1,-9999]
outcome_nullvalues = [-1,-9999]
baseline_semiwindow = 15
dynawindow = 1
semiwindow_max = 60
bootstrap_iterations = 100
random_noise = 0.01
scale_exposure_threhsold = [20,50] # min likely value, max likely value

optmode_flag = 1
if optmode_flag == 0:
    exposure_percentile_list = [0.95]
    lag = 2
    timelag_list = [timedelta(days=lag)]
if optmode_flag == 1:
    exposure_percentile_params = [0.84,0.96,0.02]
    lag_params = [1,6,1]
    exposure_percentile_list = [x / 100 for x in range(int(exposure_percentile_params[0]*100), int(exposure_percentile_params[1]*100)+1, int(exposure_percentile_params[2]*100))]
    lag = list(range(lag_params[0], lag_params[1]+1, lag_params[2]))
    timelag_list = [timedelta(days=l) for l in lag]
exposure_percentile = 0
timelag = 0
param_string = ''

sensitivity_minmax = [-40,+140,20] # % Change of exposed days events: min, max, step
sens_an_iterations = 100
sens_outprefix = ''