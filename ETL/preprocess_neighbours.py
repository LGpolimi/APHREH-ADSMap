import pandas as pd

# WORKING ON THE RESULT OF THE INTERSECTION BETWEEN THE SHAPEFILE OF THE GRID AND ITS BUFFERED VERSION

dspath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\QGIS\\'
geoarea = 'MIL0B'
uid = 'MIL0B_IDcu'
buffersize = 2
interarea_field = 'inter_area'
rawdb = pd.read_csv(dspath+geoarea+'_neighbours_raw.csv',encoding='utf-8',low_memory=False)
rawdb.drop(rawdb.loc[rawdb[interarea_field]<=(buffersize**4)].index,inplace=True)
saveout = 1
muns = list(rawdb[uid].unique())
outdb = pd.DataFrame()
totiters = len(muns)
iti = 0
for m in muns:
    sub = rawdb.loc[rawdb[uid]==m].copy(deep=True)
    neighbours = list(sub[uid+'_2'].unique().astype(str))
    neighbours.pop(neighbours.index(str(m)))
    nn = len(neighbours)
    nlist = '|'.join(neighbours)
    outdb.loc[m,'N_NEIGHBOURS'] = int(nn)
    outdb.loc[m, 'NEIGHBOURS'] = nlist
    iti += 1
    print('Preprocessing neighbours of ',geoarea,'processing =',str((iti/totiters)*100),'%')
outdb.index.rename(uid,inplace=True)
if saveout == 1:
    outdb.to_csv(dspath+geoarea+'_neighbours.csv',encoding='utf-8',sep=',')

br = 1