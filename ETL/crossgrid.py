import pandas as pd

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
                newval = newval + row[1][str(key)] * weights_dict[key]
            outdb.loc[row[0],id] = newval
        iti = iti + 1
        print('Computing crossed values ' + str(iti) + ' out of ' + str(totiters) + ' iterations (' + str(iti/totiters*100) + '%)')
    return outdb