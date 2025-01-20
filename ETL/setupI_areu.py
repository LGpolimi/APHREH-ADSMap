import pandas as pd
import datetime as dt

rootpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\datasource\\'
outpath = rootpath + 'analysis_ready\\'
outname = 'outcome_data'
geolevel = 'MIL2A'
geoid = geolevel + '_IDcu'
saveout = 1
basegrid = pd.read_csv(rootpath+geolevel+'.csv',low_memory=False)
indb = pd.read_csv(rootpath+'AREU_CVD_'+geolevel+'_2016to2023.csv',low_memory=False)

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