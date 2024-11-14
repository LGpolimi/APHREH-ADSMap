import pandas as pd
import math
import datetime as dt
import warnings
import numpy as np
import calendar
warnings.filterwarnings("ignore")

root_path = 'C:\\Users\\Lorenzo\\Documents\\Work\\Research\\2411_Environmental_RiskModel\\Set_up_I\\datasource\\'
filenamepre = 'TEMP2m_'
filenamepost = 'UTCplus1.txt'
outfold = 'C:\\Users\\Lorenzo\\Documents\\Work\\Research\\2411_Environmental_RiskModel\\Set_up_I\\datasource\\'
outprefix = 'LMB_T_'
refgridname = 'LMB1B_MM'
refgrid = pd.read_csv(root_path+refgridname+'.csv')
fetchlimits = 1
if fetchlimits == 1:
    limdataroot = 'C:\\Users\\Lorenzo\\Documents\\Work\\Research\\2411_Environmental_RiskModel\\Set_up_I\\QGIS\\'
    limdataname = 'MIL1B_MM'
    lim_outprefix = 'MI_T_'
    limitsdb = pd.read_csv(limdataroot+limdataname+'.csv')
    limit_cells = list(limitsdb['MIL1B_IDcu'].unique())

years = [2023]
months = range(1,13)

writeout = 1
writeall = 1

ncols = 177
nrows = 174
cellsize = 1500
xll = 1436301
yll = 4916704

nullval = -9999
offsetrows = 5

index_data = [[j*nrows + i + 1 for j in range(ncols)] for i in range(nrows)]
index_df = pd.DataFrame(index_data)
index_vector = index_df.values.flatten()

def datetostr(currday):
    curryearstr = str(currday.year)
    currmonthstr = str(currday.month)
    if len(currmonthstr) == 1:
        currmonthstr = '0' + currmonthstr
    currdaystr = str(currday.day)
    if len(currdaystr) == 1:
        currdaystr = '0' + currdaystr
    currhourstr = str(currday.hour)
    if len(currhourstr) == 1:
        currhourstr = '0' + currhourstr
    dtid = curryearstr + currmonthstr + currdaystr + '_' + currhourstr
    filedtid = curryearstr + currmonthstr + currdaystr + currhourstr
    return (dtid, filedtid)

totdays = 0
for y in years:
    for m in months:
        totdays= totdays+ int(calendar.monthrange(y,m)[1])
totsheets = totdays * 24
nsheet = 0

for ityear in years:
    for itmonth in months: #Set start and end date with [year, month, day]
        startdate = [ityear,itmonth,1]
        endate = [ityear,itmonth,calendar.monthrange(ityear,itmonth)[1]]
        outdb = pd.DataFrame()
        outdbrev = pd.DataFrame()
        currday = dt.datetime(startdate[0],startdate[1],startdate[2])
        stopdate = dt.datetime(endate[0],endate[1],endate[2]) + dt.timedelta(days=1)
        stdtid,stdtfn = datetostr(currday)
        eddtid,eddtfn = datetostr(dt.datetime(endate[0],endate[1],endate[2]))
        outdtstr = stdtid + '_to_' + eddtid
        totdays = (stopdate-currday).days
        if fetchlimits == 1:
            lim_outdb = pd.DataFrame()
            lim_outdbrev = pd.DataFrame()
        while currday < stopdate:
            nsheet = nsheet + 1
            dtid,filenamedt = datetostr(currday)
            filenamestr = filenamepre + filenamedt + filenamepost
            print('\n\nWORKING ON NEW SHEET: ',dtid,' (processing = ',str(((nsheet-1)/totsheets)*100),' %)')
            gotfile = 0
            try:
                filetry_raw = pd.read_table(root_path+'T_original\\'+filenamestr,sep='\t',skiprows=offsetrows,on_bad_lines='skip')
                gotfile = 1
            except:
                gotfile = 0
                print('\n\n\t\tMISSING: file for ', dtid, ' not found')
                outdb[dtid] = [np.nan] * outdb.shape[0]
                outdbrev.loc[dtid] = [np.nan] * outdbrev.shape[1]
                if fetchlimits == 1:
                    lim_outdb[dtid] = [np.nan] * lim_outdb.shape[0]
                    lim_outdbrev.loc[dtid] = [np.nan] * lim_outdbrev.shape[1]
            if gotfile == 1:
                filetry_split = filetry_raw.iloc[:, 0].str.split(expand=True)
                filetry = filetry_split.apply(pd.to_numeric, errors='coerce')
                if filetry.shape[0] != nrows or filetry.shape[1] != ncols:
                    print('ERROR ON SHEET: ', dtid, ' - wrong dimensions')
                    br = 2
                else:

                    '''for i in range(filetry.shape[0]):
                        for j in range(filetry.shape[1]):
                            iti = iti + 1
                            print('WORKING ON SHEET: ', dtid, ' (sheet processing = ',str((((i*ncols)+j) / (nrows*ncols)) * 100),' %)\ttotal processing = ',str((iti / totiters) * 100), ' %)')
                            newval = filetry.iloc[i, j]
                            newind = (j * nrows) + i + 1
                            if writeall == 1:
                                outdb.loc[newind, dtid] = newval
                                outdbrev.loc[dtid, newind] = newval
                            if fetchlimits == 1:
                                if newind in limit_cells:
                                    lim_outdb.loc[newind, dtid] = newval
                                    lim_outdbrev.loc[dtid, newind] = newval'''
                    flattened_db = filetry.values.flatten()
                    if writeall == 1:
                        if outdb.shape[0] == 0:
                            outdb = pd.DataFrame({dtid: flattened_db},index=index_vector)
                        else:
                            outdb[dtid] = flattened_db
                        if outdbrev.shape[1] == 0:
                            outdbrev = pd.DataFrame(columns=index_vector)
                        outdbrev.loc[dtid] = flattened_db

                    if fetchlimits == 1:
                        no_lim_outdb = pd.DataFrame({dtid: flattened_db}, index=index_vector)
                        if lim_outdb.shape[0] == 0:
                            lim_outdb = no_lim_outdb.loc[no_lim_outdb.index.isin(limit_cells)].copy(deep=True)
                        else:
                            lim_outdb[dtid] = no_lim_outdb.loc[no_lim_outdb.index.isin(limit_cells),:].copy(deep=True)
                        if lim_outdbrev.shape[1] == 0:
                            lim_outdbrev = pd.DataFrame(columns=limit_cells)
                        lim_series = no_lim_outdb.loc[no_lim_outdb.index.isin(limit_cells), :].copy(deep=True)
                        lim_series.sort_index(inplace=True)
                        lim_series_flat = lim_series.values.flatten()
                        lim_outdbrev.loc[dtid] = lim_series_flat

            currday = currday + dt.timedelta(hours=1)
        if writeout == 1:
            print('SAVING outputs for', outdtstr)
            outdb.index.rename('LMB1B_IDcu', inplace=True)
            outdb.replace(nullval, np.nan, inplace=True)
            outdbrev.replace(nullval, np.nan, inplace=True)
            if writeall == 1:
                outdb.to_csv(outfold + outprefix + 'ARPA_forjoin_' + outdtstr + '.csv')
                outdbrev.to_csv(outfold + outprefix + 'ARPA_fulldb_' + outdtstr + '.csv')
            if fetchlimits == 1:
                lim_outdb.replace(nullval, np.nan, inplace=True)
                lim_outdbrev.replace(nullval, np.nan, inplace=True)
                lim_outdb.index.rename('MIL1B_IDcu', inplace=True)
                lim_outdb.to_csv(outfold + lim_outprefix + 'ARPA_forjoin_' + outdtstr + '.csv')
                lim_outdbrev.to_csv(outfold + lim_outprefix + 'ARPA_fulldb_' + outdtstr + '.csv')


br = 1