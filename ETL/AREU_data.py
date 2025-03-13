import pandas as pd
from multifile_iterator import import_iterate_time
import datetime as dt
from db_filters import filter_db


data_root = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\25 03 - V2\\Set_up_III\\datasource\\processing\\'
#data_root = 'C:\\Users\\Lorenzo\\Documents\\Work\\Research\\2411_Environmental_RiskModel\\Set_up_I\\datasource\\'
file_root = 'EVT_MISS_PAZ_'
outname = 'AREU_RSP_2015to2020_LMB3A'
wrtiteout = 1
date_begin = dt.datetime(2015,1,1)
date_end = dt.datetime(2020,12,31)

areu_db = import_iterate_time('year',[date_begin,date_end],data_root,file_root,'.csv','_LMB3A',"foo",import_params={'encoding':'ISO-8859-1','on_bad_lines':'skip'})

areu_db_cardio = filter_db(areu_db,1,'MOTIVO_DTL',['INFETTIVA','ORECCHIO NASO GOLA','RESPIRATORIA'])
areu_db_cardio_luogo = filter_db(areu_db_cardio,1,'LUOGO',['CASA','STRADA','UFFICI ED ES. PUBBL.','FERROVIA / METRO','IMPIANTO SPORTIVO','IMPIANTO LAVORATIVO','UFFICIO LUOGO PUBBL','SCUOLA'])

if wrtiteout == 1:
    areu_db_cardio_luogo.to_csv(data_root+outname+'.csv',index=False)

br = 1