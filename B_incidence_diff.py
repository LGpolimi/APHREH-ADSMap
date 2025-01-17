import pandas as pd
import numpy as np
import conf
from datetime import datetime, timedelta


def compute_zones_incidence(outcome,refgrid):

    bsa_list = list(outcome.columns.values)
    if 'DATE' in bsa_list:
        bsa_list.remove('DATE')
    if 'AVG' in bsa_list:
        bsa_list.remove('AVG')
    bsa_list = [int(bsa) for bsa in bsa_list]

    incidence = pd.DataFrame(index=outcome.index,columns=outcome.columns)
    incidence['DATE'] = outcome['DATE']
    if 'AVG' in incidence.columns.values:
        incidence.drop(columns='AVG',inplace=True)

    alldates = outcome['DATE'].unique()
    alldates.sort()
    allyears = outcome['DATE'].dt.year.unique()
    popyears = list()
    for col in refgrid.columns.values:
        if 'POP' in col:
            popystr = col.replace('POP_','')
            popyears.append(int(popystr))
    totiters = len(alldates) * len(bsa_list)
    iti = 0
    for dty in allyears:
        y = int(dty)
        ystr = str(y)
        if y in popyears:
            popy = 'POP_'+ystr
        else:
            popy = 'POP_' + str(min(allyears, key=lambda x: (abs(x - y), -x)))
        y_dates = [dt for dt in alldates if pd.to_datetime(dt).year == y]
        if y+1 in allyears:
            for dp in range(conf.lag+1):
                y_dates.append(pd.to_datetime(max(y_dates)) + timedelta(days=dp))
        min_d = pd.to_datetime(min(y_dates))
        max_d = pd.to_datetime(max(y_dates))
        for bsa in bsa_list:
            refpop = refgrid.loc[refgrid[conf.geoid]==bsa,popy].values[0]
            emptyflag = 0
            if refpop == 0:
                emptyflag = 1
            for td in pd.date_range(start=min_d, end=max_d - conf.timelag):
                if emptyflag == 0:
                    events = outcome.loc[outcome['DATE'].isin([td,td+conf.timelag]),bsa].sum()
                    inc = (events/refpop) * conf.incidence_popmultiplier
                    incidence.loc[incidence['DATE'] == td,bsa] = inc
                else:
                    incidence.loc[incidence['DATE'] == td,bsa] = -1
                iti += 1
                print('Computing lagged incidence: ' + str(iti) + ' out of ' + str(totiters),' (',str(round(iti/totiters*100,2)),'%)')
    incidence.dropna(axis=0,inplace=True)
    incidence = incidence.loc[:, (incidence != -1).any(axis=0)].copy(deep=True)
    return incidence

def compute_incidence_baseline(incidence,non_exp_days):
    bsa_list = list(incidence.columns.values)
    if 'DATE' in bsa_list:
        bsa_list.remove('DATE')
    if 'AVG' in bsa_list:
        bsa_list.remove('AVG')
    bsa_list = [int(bsa) for bsa in bsa_list]

    incbaseline = pd.DataFrame(index=incidence.index, columns=incidence.columns)
    incbaseline['DATE'] = incidence['DATE']
    if 'AVG' in incbaseline.columns.values:
        incbaseline.drop(columns='AVG', inplace=True)
    alldates = incidence['DATE'].unique()
    alldates.sort()
    totiters = len(alldates) * len(bsa_list)
    iti = 0
    for y in conf.years:
        y_dates = [dt for dt in alldates if pd.to_datetime(dt).year == y]
        y_non_exposed = [dt for dt in non_exp_days if pd.to_datetime(dt).year == y]
        min_d = pd.to_datetime(min(y_dates))
        max_d = pd.to_datetime(max(y_dates))
        for td in pd.date_range(start=min_d, end=max_d):
            for bsa in bsa_list:
                nullline = 0
                window = pd.date_range(start=td - timedelta(days=conf.baseline_semiwindow), end=td + timedelta(days=conf.baseline_semiwindow))
                window_nonexp = [ned for ned in window if ned in y_non_exposed]
                if len(window_nonexp) == 0:
                    if conf.dynawindow == 1:
                        extension = 1
                        while len(window_nonexp) == 0:
                            window = pd.date_range(start=td - timedelta(days=(conf.baseline_semiwindow+extension)), end=td + timedelta(days=(conf.baseline_semiwindow+extension)))
                            window_nonexp = [ned for ned in window if ned in y_non_exposed]
                            extension += 1
                            print('Extending baseline window: ' + str(extension))
                            if extension > conf.semiwindow_max:
                                print('No viable values found skipping day ', td)
                                incbaseline.loc[incbaseline['DATE'] == td, bsa] = -1
                                nullline = 1
                                break
                    else:
                        incbaseline.loc[incbaseline['DATE'] == td, bsa] = -1
                        nullline = 1
                if nullline == 0:
                    incidence_subset = incidence.loc[incidence['DATE'].isin(window), bsa].copy(deep=True)
                    incbaseline.loc[incbaseline['DATE'] == td, bsa] = incidence_subset.mean()
                iti += 1
                print('Computing baseline incidence: ' + str(iti) + ' out of ' + str(totiters), ' (', str(round(iti / totiters * 100, 2)), '%)')
    incbaseline.dropna(axis=0, inplace=True)
    incbaseline = incbaseline.loc[:, (incbaseline != -1).any(axis=0)].copy(deep=True)
    return incbaseline


def compute_incidence_differentials(incidence,incidence_baseline):
    incidence_differentials = incidence - incidence_baseline
    incidence_differentials['DATE'] = incidence['DATE']
    return incidence_differentials

def cross_grid_computation(indb,crossgridmap,uid_fields,proportion_fields):
    outdb = pd.DataFrame(index=indb.index)
    dest_grid_uids = crossgridmap[uid_fields[1]].unique()
    dest_grid_uids.sort()
    totiters = len(dest_grid_uids)
    iti = 0
    for id in dest_grid_uids:
        subset = crossgridmap.loc[crossgridmap[uid_fields[1]] == id].copy(deep=True)
        subset['RATIO'] = subset[proportion_fields[0]] / subset[proportion_fields[1]]
        source_uids = subset[uid_fields[0]].unique()
        weights_dict = {}
        for sid in source_uids:
            weight = subset.loc[subset[uid_fields[0]] == sid,'RATIO'].values[0]
            weights_dict[sid] = weight
        for row in indb.iterrows():
            newval = 0
            for key in weights_dict.keys():
                newval = newval + row[1][key] * weights_dict[key]
            outdb.loc[row[0],id] = newval
        iti = iti + 1
        print('Computing crossed values ' + str(iti) + ' out of ' + str(totiters) + ' iterations (' + str(iti/totiters*100) + '%)')
    return outdb

def compute_weights(expth,exposure):
    expproc = exposure.copy(deep=True)
    expnodate = expproc.drop(columns='DATE').copy(deep=True)
    weights_raw = abs(expnodate-expth)

    nan_cells_weights_raw = weights_raw.isna()
    nan_or_null_values = exposure.isna() | exposure.isin(conf.exposure_nullvalues)
    selected_cells = nan_cells_weights_raw | nan_or_null_values
    weights_raw[selected_cells] = 0.01

    min_val = weights_raw.min().min()
    max_val = weights_raw.max().max()
    weights = 0.01 + (weights_raw - min_val) * (0.99 / (max_val - min_val))

    return weights