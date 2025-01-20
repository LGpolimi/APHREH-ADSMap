import os
import glob
import pandas as pd
from crossgrid import cross_grid_computation

rootpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\datasource\\MI_T\\'
outpath = rootpath
outname = 'exposure_data'
geolevel_source = 'MIL1B'
geolevel_dest = 'MIL2A'
geoid_source = geolevel_source + '_IDcu'
geoid_dest = geolevel_dest + '_IDcu'
dbsname = 'MI_T_ARPA_fulldb'
crossgrid = pd.read_csv(rootpath + geolevel_source + geolevel_dest + '.csv',low_memory=False)
saveout = 1

csv_files = glob.glob(os.path.join(rootpath, dbsname + '*.csv'))
nfiles = len(csv_files)
dataframes = [pd.read_csv(file,encoding='utf-8',low_memory=False) for file in csv_files]
for df in dataframes:
    df.set_index('Unnamed: 0',inplace=True,drop=True)
    df.index.rename('DATETIME_STR',inplace=True)
combined_df = pd.concat(dataframes)
if '0' in combined_df.columns:
    combined_df.drop(columns=['0'],inplace=True)

if 'DATE' not in combined_df.columns:
    combined_df['DATE'] = pd.to_datetime(combined_df.index.str[:4] + '-' + combined_df.index.str[4:6] + '-' + combined_df.index.str[6:8] + ' ' + combined_df.index.str[9:] + ':00', format='%Y-%m-%d %H:%M')
combined_df.reset_index(inplace=True,drop=True)

df_forproc = combined_df.copy(deep=True)
daily_avg_df = df_forproc.resample('D', on='DATE').mean()
daily_avg_df['DATECOL'] = daily_avg_df.index
daily_avg_df.reset_index(inplace=True,drop=True)
daily_avg_df.dropna(how='any', inplace=True)
daily_avg_df.reset_index(inplace=True,drop=True)
daily_avg_df['DATECOL'] = daily_avg_df['DATECOL'].dt.strftime('%y%m%d')
daily_avg_df.set_index('DATECOL', inplace=True, drop=True)
daily_avg_df.index.rename('DATE_STR',inplace=True)
if 'DATE' in daily_avg_df.columns:
    daily_avg_df.drop(columns=['DATE'],inplace=True)

del(df_forproc)
df_forproc = combined_df.copy(deep=True)
daily_max_df = df_forproc.resample('D', on='DATE').max()
daily_max_df['DATECOL'] = daily_max_df.index
daily_max_df.reset_index(inplace=True,drop=True)
daily_max_df.dropna(how='any', inplace=True)
daily_max_df.reset_index(inplace=True,drop=True)
daily_max_df['DATECOL'] = daily_max_df['DATECOL'].dt.strftime('%y%m%d')
daily_max_df.set_index('DATECOL', inplace=True, drop=True)
daily_max_df.index.rename('DATE_STR',inplace=True)
if 'DATE' in daily_max_df.columns:
    daily_max_df.drop(columns=['DATE'],inplace=True)

#remapped_daily_avg_df = cross_grid_computation(daily_avg_df,crossgrid,[geoid_source,geoid_dest],['Area',geolevel_dest+'_Area'])

if saveout == 1:
    daily_avg_df.to_csv(outpath+outname+'.csv')

br = 1