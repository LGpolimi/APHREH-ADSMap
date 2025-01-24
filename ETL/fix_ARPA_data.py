import os
import glob
import pandas as pd

rootpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_II\\datasource\\LMB_T\\'
outpath = rootpath
dbsname = 'LMB_T_ARPA_fulldb'

csv_files = glob.glob(os.path.join(rootpath, dbsname + '*.csv'))
nfiles = len(csv_files)
dataframes = list()
for file in csv_files:
    try:
        df = pd.read_csv(file,encoding='utf-8',low_memory=False)
        print('Fixing ' + file)
        df.set_index('Unnamed: 0',inplace=True,drop=True)
        df.index.rename('DATETIME_STR',inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.to_csv(file,encoding='utf-8')
    except:
        print('Error in ' + file)

br = 1