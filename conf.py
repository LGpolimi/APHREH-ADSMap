import os
import pandas as pd
from datetime import timedelta

# PATHS

model_version = 'V10'
dspath = 'D:\\HEALTH_GEOMATICS\\projects_in_progress\\2501-VulnerabilityIndex\\25 03 - V2\\Set_up_I\\datasource\\analysis_ready\\'
respath = 'D:\\HEALTH_GEOMATICS\\projects_in_progress\\2501-VulnerabilityIndex\\25 03 - V2\\Set_up_I\\results\\'
outpath = respath + model_version + '\\'
if not os.path.isdir(respath):
    os.mkdir(respath)
if not os.path.isdir(outpath):
    os.mkdir(outpath)
yearly_folder = outpath + 'Years\\'

# FLAGS

saveout = 1 # Flag to save output
dynawindow = 1 # Flag to enable dynamic windowing
optmode_flag = 1 # Flag to launch optimization mode
sensan_flag = 1 # Flag to launch sensitivity analysis
uncertainty_flag = 1 # Flag to launch uncertainty analysis on highest values

# FIELDS NAMES

exposure_db_name = 'exposure_data.csv'
outcome_db_name = 'outcome_data.csv'
reference_geo_level = 'MIL2A'
geoid = reference_geo_level + '_IDcu'
area_field = reference_geo_level + '_Area'
incidence_popmultiplier = 100000
source_geo_level = 'MIL1B'
source_geoid = source_geo_level + '_IDcu'
cross_area_field = 'Area'

# PARAMETERS

exposure_nullvalues = [-1,-9999]
outcome_nullvalues = [-1,-9999]
years = [2017,2018,2019]
months = [5,6,7,8,9]
zones = ['ALL'] # Set to 'ALL' to analyze all BSAs
baseline_semiwindow = 15
semiwindow_max = 60
bootstrap_iterations = 1000
random_noise = 0.01
scale_exposure_threhsold = [20,50] # min likely value, max likely value
uncertainty_nvalues = 5 # Number of highest values to analyze
uncertainty_iterations = 100
sensitivity_minmax = [-60,+200,20] # % Change of exposed days events: min, max, step
sens_an_iterations = 100

if optmode_flag == 0:
    exposure_percentile_list = [0.95]
    lag_list = [2]
    timelag_list = [timedelta(days=lag_list[0])]
if optmode_flag == 1:
    exposure_percentile_params = [0.9,0.95,0.01]
    lag_params = [0,7,1]
    exposure_percentile_list = [x / 100 for x in range(int(exposure_percentile_params[0]*100), int(exposure_percentile_params[1]*100)+1, int(exposure_percentile_params[2]*100))]
    lag_list = list(range(lag_params[0], lag_params[1]+1, lag_params[2]))
    timelag_list = [timedelta(days=l) for l in lag_list]


# GLOBAL DATA STRUCTURES

exposure_percentile = 0
lag = 0
timelag = timedelta(days=0)
out_prefix = ''
process_string = ''
param_string = ''

class global_struct():
    def __init__(self):
        self.crossgrid = 0
        self.refgrid = 0
        self.exposure_grid = 0
        self.exposure = 0
        self.outcome = 0

class global_results():
    def __init__(self):
        self.marms = dict()
        self.wmarms = dict()
        self.marm_db = pd.DataFrame()
        self.wmarm_db = pd.DataFrame()
        self.max_wmarm = 0
        self.opt_params = ''
    def integrate_results(self,updated):
        for key in updated.marms.keys():
            self.marms[key] = updated.marms[key]
        for key in updated.wmarms.keys():
            self.wmarms[key] = updated.wmarms[key]
        self.max_wmarm = updated.max_wmarm
        self.opt_params = updated.opt_params
        for m_row in updated.marm_db.iterrows():
            self.marm_db.loc[m_row[0],:] = m_row[1]
        for wm_row in updated.wmarm_db.iterrows():
            self.wmarm_db.loc[wm_row[0],:] = wm_row[1]


class parametric_results():
    def __init__(self):
        self.marm = 0
        self.wmarm = 0
        self.stdeff_db = pd.DataFrame()
        self.out_index_df = pd.DataFrame()
        self.out_index_df_formatted = pd.DataFrame()
        self.cum_index_df = pd.DataFrame()
        self.cum_index_df_formatted = pd.DataFrame()

class parametric_struct():
    def __init__(self):
        self.th = 0
        self.lag = 0
        self.timelag = timedelta(days=0)
        self.exp_threshold = dict()
        self.exposed_days =[]
        self.non_exposed_days = []
        self.incidence = pd.DataFrame()
        self.incidence_baseline = pd.DataFrame()
        self.yearly_ds =list()
        self.results = parametric_results()
        self.param_string = ''
    def set_paramstring(self):
        self.param_string = 'EXP: ' + str(self.th*100) + '° perc - LAG: ' + str(self.lag) + '\t'

class yearly_struct():
    def __init__(self):
        self.year = 0
        self.expth = 0
        self.weights = pd.DataFrame()
        self.expdays = []
        self.nonexpdays = []
        self.yearly_index_df = pd.DataFrame()
        self.yearly_permutations_r = pd.DataFrame()
        self.yearly_permutations_p = pd.DataFrame()

max_wmarm_ds = parametric_struct()

sens_outprefix = ''
sens_exp_threshold = dict()
sens_exposed_days =[]
sens_non_exposed_days = []
sens_incidence_base = pd.DataFrame()
sens_incidence_baseline = pd.DataFrame()