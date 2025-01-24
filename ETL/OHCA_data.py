import pandas as pd
import datetime as dt
from db_filters import filter_db


data_root = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_II\\datasource\\'

file_root = 'ohca_1622_LMB3A_raw'
outname = 'OHCA_LMB3A_2016to2022'
wrtiteout = 1
date_begin = dt.datetime(2016,1,1)
date_end = dt.datetime(2022,5,31)

#areu_db = import_iterate_time('year',[date_begin,date_end],data_root,file_root,'.csv','_MI',"foo",import_params={'encoding':'ISO-8859-1','on_bad_lines':'skip'})
areu_db = pd.read_csv(data_root+file_root+'.csv',encoding='ISO-8859-1',low_memory=False)

#areu_db_cardio = filter_db(areu_db,1,'MOTIVO_DTL','CARDIOCIRCOLATORIA')
areu_db_cardio_luogo = filter_db(areu_db,1,'luogo_n',['CASA','STRADA','UFFICI ED ES. PUBBL.','FERROVIA / METRO','IMPIANTO SPORTIVO','IMPIANTO LAVORATIVO','UFFICIO LUOGO PUBBL','SCUOLA'])

if wrtiteout == 1:
    areu_db_cardio_luogo.to_csv(data_root+outname+'.csv',index=False)

br = 1