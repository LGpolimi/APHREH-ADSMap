import pandas as pd
import numpy as np
import conf

def compute_marm():
    import conf
    index_df = pd.read_csv(conf.outpath + conf.out_prefix + 'index_raw.csv')
    stdeff_db = index_df.copy(deep=True)

    # Compute observations’ ‘standardized effect’ value
    for y in conf.years:
        rdist_db = pd.read_csv(conf.outpath + conf.out_prefix + 'permutations_r_'+str(y)+'.csv')
        for bsa in rdist_db[conf.geoid].unique():
            rdist = np.asarray(rdist_db[rdist_db[conf.geoid] == bsa].values)
            stdev = np.std(rdist)
            index = stdeff_db.loc[stdeff_db[conf.geoid] == bsa, str(y)+'INDEX'].values[0]
            std_effect = index / stdev
            stdeff_db.loc[stdeff_db[conf.geoid] == bsa,str(y)+'STDEFF'] = std_effect

    # Compute BSA-specific weighted average
    for bsa in stdeff_db[conf.geoid].unique():
        weights_sum = 0
        average_value = 0
        for y in conf.years:
            stdeff = stdeff_db.loc[stdeff_db[conf.geoid] == bsa, str(y)+'STDEFF'].values[0]
            ci_low = stdeff_db.loc[stdeff_db[conf.geoid] == bsa, str(y)+'CI_LOW'].values[0]
            ci_high = stdeff_db.loc[stdeff_db[conf.geoid] == bsa, str(y)+'CI_HIGH'].values[0]
            w = abs(ci_high - ci_low)
            average_value = average_value + (stdeff * w)
            weights_sum = weights_sum + w
        weighted_average = average_value / weights_sum
        stdeff_db.loc[stdeff_db[conf.geoid] == bsa, 'AVG_STDEFF'] = weighted_average

    # Compute MARM
    stdeff_values = stdeff_db['AVG_STDEFF'].values
    marm = stdeff_values.sum()/len(stdeff_values)

    return stdeff_db, marm