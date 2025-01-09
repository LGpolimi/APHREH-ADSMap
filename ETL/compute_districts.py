import pandas as pd
import numpy as np
import os

####################################################################################################################
target_population = 5000
###################################################################################################################


dspath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\QGIS\\'
geoarea = 'MIL0B'
pop_field = 'POP_2021'
uid = 'MIL0B_IDcu'
saveout = 1
popdb = pd.read_csv(dspath+geoarea+'_population.csv',encoding='utf-8')
popdb.set_index(uid,inplace=True,drop=True)
popdb.sort_values(by=pop_field,inplace=True,ascending=False)
nghdb = pd.read_csv(dspath+geoarea+'_neighbours.csv',encoding='utf-8')
nghdb.set_index(uid,inplace=True,drop=True)
odmatrixtype = 2 # 1 = topological as output from GIS, 2 = euclidean
odmatrix = pd.read_csv(dspath+geoarea+'_ODM.csv',encoding='utf-8',low_memory=False)
odmatrix.set_index(uid,inplace=True,drop=True)
centroids = pd.read_csv(dspath+geoarea+'_centroids.csv',encoding='utf-8',low_memory=False)
muns = list(popdb.index.values)
tp = target_population*0.9
tolerance = 0.1
popfield = pop_field

def find_candidates(neighbours,selected,candidates,refdb):
    candlist = list()
    for n in neighbours:
        if n not in selected and n not in candidates and refdb.loc[int(n),['ASSIGNED']].values[0] == 0:
            candlist.append(n)
    return(candlist)

def sort_candidates(center,candidates,odmatrix,centroids):
    canddb = pd.DataFrame({'CANDIDATE':candidates})
    centerid = centroids.loc[centroids[uid]==center,uid].values[0]
    for c in candidates:
        if odmatrixtype == 1:
            candid = centroids.loc[centroids['GEO2A_IDna'] == c, 'GEO2A_IDis'].values[0]
            dist = odmatrix.loc[(odmatrix['origin_id']==centerid)&(odmatrix['destination_id']==candid),'total_cost'].values[0]
        elif odmatrixtype == 2:
            candind = int(c)
            dist = odmatrix.loc[centerid,c].astype(float)
        canddb.loc[canddb['CANDIDATE']==c,'DIST'] = dist
    canddb.sort_values(by='DIST',ascending=True,inplace=True)
    canddb.reset_index(inplace=True,drop=True)
    sorted_candidates = list(canddb['CANDIDATE'])
    return(sorted_candidates)


refdb = pd.DataFrame(popdb.index)
#refdb['CANDIDATE'] = 0
refdb['ASSIGNED'] = 0
refdb['DISTRICT'] = -1
refdb.set_index(uid,inplace=True,drop=True)
districts = pd.DataFrame()
dist_id = 0
totiters = len(muns)
iti = 0
for m in muns:
    checkm = refdb.loc[m,'ASSIGNED']
    if checkm == 0:
        dist_id = dist_id + 1
        selected = list()
        selected.append(str(m))
        refdb.loc[m,'ASSIGNED'] = 1
        refdb.loc[m,'DISTRICT'] = dist_id
        mpop = int(popdb.loc[m,pop_field])
        distpop = mpop
        distflag = 0
        if distpop >= tp:
            distflag = 1
        while distpop < tp and distflag == 0:
            newadd_flag = 0
            candidates = list()
            for sm in selected:
                neighbours = nghdb.loc[int(sm),'NEIGHBOURS'].split('|')
                partcand = find_candidates(neighbours,selected,candidates,refdb)
                candidates = candidates + partcand
            if len(candidates)>0:
                candidates = sort_candidates(m,candidates,odmatrix,centroids)
            for cms in candidates:
                cm = int(cms)
                cmpop = int(popdb.loc[cm,pop_field])
                if ((distpop + cmpop) < tp) or (abs((distpop+cmpop)-tp)<abs(distpop-tp)):
                    selected.append(cms)
                    refdb.loc[cm, 'ASSIGNED'] = 1
                    refdb.loc[cm, 'DISTRICT'] = dist_id
                    distpop = distpop + cmpop
                    newadd_flag = 1
            if newadd_flag == 0:
                distflag = 1
        districts.loc[dist_id,'DIST_ID'] = dist_id
        districts.loc[dist_id,popfield] = distpop
        distmlist = '|'.join(selected)
        districts.loc[dist_id,'MUN_LIST'] = distmlist
        iti = iti + 1
        print(f"FIRST ITERATION: Assigned municipality {m} to district {dist_id} processing = {str((iti/totiters)*100)}%")
    else:
        iti = iti + 1
        print(f"FIRST ITERATION: Municipality {m} already assigned processing = {str((iti/totiters)*100)}%")

districts['DIFF'] = districts[popfield] - target_population
oversub = districts.loc[districts['DIFF']>0]
if oversub.shape[0]>0:
    tpth = np.quantile(np.asarray(oversub['DIFF']),0.9)
else:
    tpth = target_population*tolerance
undersub = districts.loc[districts['DIFF']<0]
removesub = undersub.loc[abs(districts['DIFF'])>tpth]
splitlist = list()
for row in removesub.iterrows():
    newlist = row[1]['MUN_LIST'].split('|')
    splitlist = splitlist + newlist
for sms in splitlist:
    sm = int(sms)
    refdb.loc[sm, 'ASSIGNED'] = 0
    refdb.loc[sm, 'DISTRICT'] = -1
districts.drop(removesub.index,inplace=True)

unassigned = refdb.loc[refdb['ASSIGNED']==0].shape[0]
unit = 0
while unassigned > 0:
    leftdb = refdb.loc[refdb['ASSIGNED']==0].copy(deep=True)
    leftdb.index = leftdb.index.astype(int)
    unassigned = leftdb.shape[0]
    if unassigned > 0:
        unit = unit + 1
        print('Including',str(unassigned),'left uncovered: iteration = ',str(unit))
        leftdb = leftdb.merge(popdb,left_index=True,right_index=True,how='inner')
        leftdb.sort_values(by=pop_field,ascending=False,inplace=True)
        umlist = list(leftdb.index.values)
        for um in umlist:
            ums = str(um)
            neighbours = [int(n) for n in nghdb.loc[um, 'NEIGHBOURS'].split('|')]
            canddist = list(refdb.loc[neighbours,'DISTRICT'].unique())
            if -1 in canddist:
                canddist.pop(canddist.index(-1))
            if len(canddist)>0:
                canddistdb = districts.loc[districts['DIST_ID'].isin(canddist)].copy(deep=True)
                canddistdb.sort_values(by=popfield,ascending=True,inplace=True)
                seldist = int(canddistdb['DIST_ID'].values[0])
                refdb.loc[um, 'ASSIGNED'] = 1
                refdb.loc[um, 'DISTRICT'] = seldist
                addpop = popdb.loc[um,popfield]
                districts.loc[seldist, popfield] = int(districts.loc[seldist, popfield]) + addpop
                districts.loc[seldist, 'MUN_LIST'] = districts.loc[seldist, 'MUN_LIST'] + '|' + ums

districts.sort_values(by=popfield,ascending=False,inplace=True)
districts.reset_index(inplace=True,drop=True)
districts['DIST_ID'] = districts.index.values
districts['DIST_ID'] = districts['DIST_ID'] + 1
districts['DIFF'] = districts[popfield] - target_population

totfix = districts.shape[0]
fixed = 0
for dist in districts.iterrows():
    distmuns = [int(dm) for dm in dist[1]['MUN_LIST'].split('|')]
    for dm in distmuns:
        refdb.loc[dm, 'DISTRICT'] = dist[1]['DIST_ID']
    fixed = fixed + 1
    print('Setting final districts values: processing = ',str((fixed/totfix)*100),'%')
'''# DEBUGGING
debugiters = len(muns)
debugi = 0
errors = list()
for m in muns:
    mdist = refdb.loc[m,'DISTRICT']
    distmunlist = [int(dm) for dm in districts.loc[districts['DIST_ID']==mdist,'MUN_LIST'].values[0].split('|')]
    if m not in distmunlist:
        print('ERROR on municipality ',m)
        errors.append(m)
    debugi = debugi + 1
    print('Debugging: processing = ',str((debugi/debugiters)*100),'%')'''

if saveout == 1:
    if target_population >= 1000:
        tpstr = str(round(target_population/1000)) + 'k'
    else:
        tpstr = str(target_population)
    outfold = dspath+'Districts_'+tpstr+'//'
    if not os.path.isdir(outfold):
        os.makedirs(outfold)
    refdb.to_csv(outfold+uid+'_districts_'+tpstr+'.csv',encoding='utf-8')
    districts.to_csv(outfold + 'Districts_' + tpstr + '.csv', encoding='utf-8',index=False)
br = 1