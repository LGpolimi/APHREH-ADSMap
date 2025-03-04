import pandas as pd
import numpy as np
import conf

def compute_marm(index_df):
    import conf
    stdeff_db = index_df.copy(deep=True)

    # Compute observations’ ‘standardized effect’ value
    for y in conf.years:
        rdist_db = pd.read_csv(conf.outpath + conf.out_prefix + 'permutations_r_'+str(y)+'.csv')
        for bsa in rdist_db[conf.geoid].unique():
            rdist = np.asarray(rdist_db[rdist_db[conf.geoid] == bsa].values)
            stdev = np.std(rdist)
            index = stdeff_db.loc[bsa, str(y)+'INDEX']
            std_effect = index / stdev
            stdeff_db.loc[bsa,str(y)+'STDEFF'] = std_effect

    # Compute BSA-specific weighted average
    for bsa in stdeff_db.index.values:
        weights_sum = 0
        average_value = 0
        for y in conf.years:
            stdeff = stdeff_db.loc[bsa, str(y)+'STDEFF']
            ci_low = stdeff_db.loc[bsa, str(y)+'CI_LOW']
            ci_high = stdeff_db.loc[bsa, str(y)+'CI_HIGH']
            w = abs(ci_high - ci_low)
            average_value = average_value + (stdeff * w)
            weights_sum = weights_sum + w
        weighted_average = average_value / weights_sum
        stdeff_db.loc[bsa, 'AVG_STDEFF'] = weighted_average

    # Compute MARM
    stdeff_values = np.abs(stdeff_db['AVG_STDEFF'].values)
    marm = stdeff_values.sum()/len(stdeff_values)

    return stdeff_db, marm

def compute_wmarm(refgrid,stdeff_db):
    import conf
    wmarm_num = 0
    wmarm_den = 0
    for bsa in stdeff_db.index.values:
        cumpop = 0
        for y in conf.years:
            ypop = refgrid.loc[refgrid[conf.geoid] == bsa, 'POP_'+str(y)].values[0]
            cumpop = cumpop + ypop
        avgpop = cumpop / len(conf.years)
        stdeff = abs(stdeff_db.loc[bsa, 'AVG_STDEFF'])
        wmarm_num = wmarm_num + (stdeff * avgpop)
        wmarm_den = wmarm_den + avgpop
    wmarm = wmarm_num / wmarm_den
    return wmarm