import pandas as pd
import datetime as dt
import os

rootpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\25 03 - V2\\Set_up_III\\datasource\\'
inpath = rootpath + 'processing\\'
outpath = rootpath + 'analysis_ready\\'
if not os.path.isdir(outpath):
    os.mkdir(outpath)
outname = 'outcome_data'
geolevel = 'LMB3A'
geoid = geolevel + '_IDcu'
saveout = 1
basegrid = pd.read_csv(inpath+geolevel+'.csv',low_memory=False)
indb = pd.read_csv(inpath+'AREU_RSP_'+geolevel+'_2015to2020.csv',low_memory=False)

indb['DATETIME'] = pd.to_datetime(indb['DATA'].str.strip(), format='%d%b%Y:%H:%M:%S.%f')
indb['DATE_STR'] = indb['DATETIME'].dt.strftime('%y%m%d')
unique_dates = indb['DATE_STR'].unique()
unique_geoids = basegrid[geoid].unique()
unique_geoids.sort()
result_df = pd.DataFrame(index=unique_dates, columns=unique_geoids).fillna(0)
totiters = len(unique_dates)*len(unique_geoids)
iti = 0
for date in unique_dates:
    for geoid_value in unique_geoids:
        count = indb[(indb['DATE_STR'] == date) & (indb[geoid] == geoid_value)].shape[0]
        result_df.at[date, geoid_value] = count
        iti = iti + 1
        print('Processed ' + str(iti) + ' out of ' + str(totiters) + ' iterations (' + str(iti/totiters*100) + '%)')

result_df.index.rename('DATE_STR',inplace=True)

if saveout == 1:
    result_df.to_csv(outpath+outname+'.csv')

br = 1