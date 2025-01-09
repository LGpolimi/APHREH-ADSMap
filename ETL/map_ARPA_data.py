import pandas as pd
import math
import datetime as dt
import warnings
import numpy as np
import calendar
warnings.filterwarnings("ignore")

root_path = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\datasource\\'
filenamepre = 'TEMP2m_'
filenamepost = 'UTCplus1.txt'
outfold = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\datasource\\'
refgridname = 'LMB1B_MM'
refgrid = pd.read_csv(root_path+refgridname+'.csv')
fetchlimits = 1

writeout = 1

ncols = 177
nrows = 174
cellsize = 1500
xll = 1436301
yll = 4916704

nullval = -9999
offsetrows = 5


def fetch_limits(input_db, xfield, yfield):
    y_limits = list(input_db[yfield].unique())
    x_limits = list(input_db[xfield].unique())
    return x_limits, y_limits

def infer_cell(refgrid, i, j, ncols, nrows, cellsize, xll, yll):
    allcells = refgrid['LMB1B_IDcu'].unique()
    x = xll + (j * cellsize)
    y = yll + (nrows * cellsize) - (i * cellsize)
    cell_id = (j * nrows) + i + 1
    if cell_id in allcells:
        xcell = refgrid.loc[refgrid['LMB1B_IDcu'] == cell_id, 'x_min'].values[0]
        ycell = refgrid.loc[refgrid['LMB1B_IDcu'] == cell_id, 'y_min'].values[0]
    else:
        xcell = 0
        ycell = 0
    return cell_id, xcell, ycell

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

if fetchlimits == 1:
    limitsdb = pd.read_csv(
        'C:\\Users\\Lorenzo\\Documents\\Work\\Research\\2411_Environmental_RiskModel\\Set_up_I\\QGIS\\MIL1B_MM.csv')
    xlimits, ylimits = fetch_limits(limitsdb, 'X_min', 'Y_min')
    limit_cells = list(limitsdb['MIL1B_IDcu'])
    limit_i = list()
    limit_j = list()
    limitdb = pd.DataFrame()
    lrow = 0
    for i in range(nrows):
        for j in range(ncols):
            print('COMPUTING LIMITS: processing = ',str(((i*ncols)+(j+1))/(nrows*ncols)*100),' %')
            newcellid, cellx, celly = infer_cell(refgrid, i, j, ncols, nrows, cellsize, xll, yll)
            if newcellid in limit_cells:
                limit_i.append(i)
                limit_j.append(j)
                limitdb.loc[lrow, 'MIL1B_IDcu'] = newcellid
                limitdb.loc[lrow, 'i'] = i
                limitdb.loc[lrow, 'j'] = j
                limitdb.loc[lrow,'error_x'] = xll+(i*cellsize) - cellx
                limitdb.loc[lrow,'error_y'] = yll+(j*nrows)-(j*cellsize) - celly
                lrow = lrow + 1
    rows_range = [min(limit_i), max(limit_i)]
    cols_range = [min(limit_j), max(limit_j)]
else:
    rows_range = [0, nrows]
    cols_range = [0, ncols]

for ityear in [2018,2023]:
    for itmonth in range(1,13): #Set start and end date with [year, month, day]
        startdate = [ityear,itmonth,1]
        endate = [ityear,itmonth,calendar.monthrange(ityear,itmonth)[1]]

        outdb = pd.DataFrame()
        outdbrev = pd.DataFrame()
        currday = dt.datetime(startdate[0],startdate[1],startdate[2])
        stopdate = dt.datetime(endate[0],endate[1],endate[2]) + dt.timedelta(days=1)
        stdtid,stdtfn = datetostr(currday)
        eddtid,eddtfn = datetostr(stopdate - dt.timedelta(hours=1))
        outdtstr = stdtid + '_to_' + eddtid
        totdays = (stopdate-currday).days
        totsheets = totdays * 24
        sheetiterations = nrows * ncols * 2
        totiterations = sheetiterations * totsheets
        nsheet = 0

        while currday < stopdate:
            nsheet = nsheet + 1
            dtid,filenamedt = datetostr(currday)
            filenamestr = filenamepre + filenamedt + filenamepost
            print('\n\nWORKING ON NEW SHEET: ',dtid,' (processing = ',str((nsheet/totsheets)*100),' %)')

            try:
                filetry = pd.read_table(root_path+'T_original\\'+filenamestr,sep='\t',skiprows=offsetrows)
                newset = pd.DataFrame()

                for i in range(nrows):
                    newline = filetry.iloc[i]
                    fullarr = str(newline.get(0)).split(' ')
                    emp = 1
                    while emp == 1:
                        if ('') in fullarr:
                            empind = fullarr.index('')
                            fullarr.pop(empind)
                        else:
                            emp = 0
                    for j in range(ncols):
                        print('WORKING ON SHEET: ',dtid,' processing = ',str((((i)*(ncols)+(j+1))/sheetiterations)*100),' % (total processing = ',str(((((nsheet-1)*sheetiterations)+((i)*(ncols)+(j+1)))/totiterations)*100),' %)')
                        newset.loc[i,j] = float(fullarr[j])
                if fetchlimits != 1:
                    for cellid in range(nrows*ncols):
                        print('WORKING ON SHEET: ',dtid,' processing = ',str((((sheetiterations/2)+cellid+1)/sheetiterations)*100),' % (total processing = ',str(((((nsheet-1)*sheetiterations)+((sheetiterations/2)+cellid+1))/totiterations)*100),' %)')
                        cellcol = math.floor(cellid/nrows)
                        cellrow = cellid - (cellcol*nrows)
                        cellval = newset.iloc[cellrow,cellcol]
                        if cellval != nullval:
                            outdb.loc[cellid+1,dtid] = cellval
                            outdbrev.loc[dtid,cellid+1] = cellval
                else:
                    for i in range(rows_range[0], rows_range[1]):
                        for j in range(cols_range[0], cols_range[1]):
                            if not limitdb[(limitdb['i'] == i) & (limitdb['j'] == j)].empty:
                                cellid = limitdb.loc[(limitdb['i'] == i) & (limitdb['j'] == j), 'MIL1B_IDcu'].values[0]
                                odr = outdb.shape[0]
                                orr = outdbrev.shape[0]
                                if outdb.shape[1] > 0:
                                    if cellid not in list(outdb['MIL1B_IDcu']):
                                        outdb.loc[odr, 'MIL1B_IDcu'] = cellid
                                else:
                                    outdb.loc[odr, 'MIL1B_IDcu'] = cellid
                                cellval = newset.iloc[i, j]
                                outdb.loc[outdb['MIL1B_IDcu'] == cellid, dtid] = cellval
                                outdbrev.loc[dtid, cellid] = cellval
            except:
                print('\n\n\t\tMISSING: file for ',dtid,' not found')
                outdb.loc[0,dtid] = ''
                outdbrev.loc[dtid,0] = ''

            currday = currday + dt.timedelta(hours=1)

        outdb.set_index('MIL1B_IDcu', inplace=True)
        if writeout == 1:
            outdbrev.to_csv(outfold + 'MI_T_ARPA_fulldb_' + outdtstr + '.csv')
            outdb.to_csv(outfold+'MI_T_ARPA_forjoin_'+outdtstr+'.csv')

br = 1