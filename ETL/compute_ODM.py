import pandas as pd
import numpy as np

root_path = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\QGIS\\'
centroids_db_name = 'MIL0B_centroids'
x_field = 'X'
y_field = 'Y'
cellid = 'MIL0B_IDcu'
outgeoname = 'MIL0B'
writeout = 1

centroids_db = pd.read_csv(root_path+centroids_db_name+'.csv')
coordinates = centroids_db[[x_field, y_field]].values

allcells = list(centroids_db[cellid].unique().astype(int))
outdb = pd.DataFrame(columns=allcells)
totiters = centroids_db.shape[0]
iti = 0

for row in centroids_db.iterrows():
    newcell = row[1][cellid]
    x = row[1][x_field]
    y = row[1][y_field]
    distances = np.sqrt((coordinates[:, 0] - x) ** 2 + (coordinates[:, 1] - y) ** 2)
    outdb.loc[newcell] = distances
    iti += 1
    print('Computing OD matrix: processing = ',str((iti/totiters)*100),'%')

max_val = outdb.max().max()
norm_outdb = outdb/max_val
outdb.index.rename(cellid,inplace=True)
norm_outdb.index.rename(cellid,inplace=True)
if writeout==1:
    outdb.to_csv(root_path+outgeoname+'_ODM.csv')
    norm_outdb.to_csv(root_path+outgeoname+'_NORM_ODM.csv')

br = 1