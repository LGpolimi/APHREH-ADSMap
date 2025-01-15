import pandas as pd

rootpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\24 10 - V1\\Set_up_I\\datasource\\'

geolvl = 'MIL2A'
geoid = geolvl + '_IDcu'

popdb = pd.read_csv(rootpath+'raw_population_MIL2A.csv',encoding='utf-8',sep=';')
grid = pd.read_csv(rootpath+'MIL2A_nopop.csv',encoding='utf-8')

allareas = grid[geoid].unique()
allareas.sort()
for area in allareas:
    popsub = popdb.loc[popdb['ID_NIL']==area].copy(deep=True)
    allyears = popsub['Anno'].unique()
    for y in range(2015,2024):
        if y in allyears:
            pop = popsub.loc[popsub['Anno'] == y,'Residenti'].values[0]
        else:
            closesty = min(allyears, key=lambda x: (abs(x - y), -x))
            pop = popsub.loc[popsub['Anno'] == closesty, 'Residenti'].values[0]
        grid.loc[grid[geoid] == area,'POP_'+str(y)] = pop
br = 1