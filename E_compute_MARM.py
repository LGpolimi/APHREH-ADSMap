import pandas as pd
import numpy as np
import os
import shutil
import re
import conf
from A_expnonexp_days import define_exposure_days
from B_incidence_diff import compute_zones_incidence, compute_incidence_baseline, compute_incidence_differentials, compute_weights
from C_compute_index import compute_index_main
from D_post_process import cumulate_across_years

def decode_params(key):
    match = re.match(r'P(\d+)_L(\d+)', key)
    if match:
        th = int(match.group(1)) / 100
        l = int(match.group(2))
        return th, l
    else:
        raise ValueError("Invalid key format")

def save_variables(exp_threshold, exposed_days, non_exposed_days,incidence_base,incidence_baseline):
    conf.sens_exp_threshold = exp_threshold
    conf.sens_exposed_days =  exposed_days
    conf.sens_non_exposed_days = non_exposed_days
    conf.sens_incidence_base = incidence_base
    conf.sens_incidence_baseline = incidence_baseline

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
    thresholds = pd.read_csv(conf.yearly_folder+'exposure_thresholds.csv')
    thresholds['Scaled_Threshold'] = ((thresholds['Threshold'] - conf.scale_exposure_threhsold[0]) / (conf.scale_exposure_threhsold[1]-conf.scale_exposure_threhsold[0])) * 10
    thresholds['Year'] = thresholds['Year'].astype(int)
    for bsa in stdeff_db.index.values:
        weights_sum = 0
        average_value = 0
        for y in conf.years:
            stdeff = stdeff_db.loc[bsa, str(y)+'STDEFF']
            ci_low = stdeff_db.loc[bsa, str(y)+'CI_LOW']
            ci_high = stdeff_db.loc[bsa, str(y)+'CI_HIGH']
            w1 = 1 / abs(ci_high - ci_low)
            w2 = thresholds.loc[thresholds['Year'] == y, 'Scaled_Threshold'].values[0]
            w = w1*w2
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
            if 'POP_'+str(y) in refgrid.columns:
                ypop = refgrid.loc[refgrid[conf.geoid] == bsa, 'POP_'+str(y)].values[0]
            else:
                min_y_dist = 9999
                closest_y = 0
                for c in refgrid.columns:
                    if 'POP_' in c:
                        cy = int(c.split('_')[1])
                        ydist = abs(y - cy)
                        saveflag = 0
                        if ydist < min_y_dist:
                            saveflag = 1
                        elif ydist == min_y_dist:
                            if cy > closest_y:
                                saveflag = 1
                        if saveflag == 1:
                            min_y_dist = ydist
                            closest_y = cy
                            closest_col = 'POP_' + str(cy)
                ypop = refgrid.loc[refgrid[conf.geoid] == bsa, closest_col].values[0]
            cumpop = cumpop + ypop
        avgpop = cumpop / len(conf.years)
        stdeff = abs(stdeff_db.loc[bsa, 'AVG_STDEFF'])
        wmarm_num = wmarm_num + (stdeff * avgpop)
        wmarm_den = wmarm_den + avgpop
    wmarm = wmarm_num / wmarm_den
    return wmarm

def identify_max_wmarm(wmarms):
    import conf
    # Identify the combination of th and l with the max wmarm value
    max_wmarm_key = max(wmarms, key=wmarms.get)
    max_wmarm_value = wmarms[max_wmarm_key]
    # Replicate the folder with the results and rename the subfolder
    source_folder = os.path.join(conf.outpath, max_wmarm_key)
    destination_folder = os.path.join(conf.outpath, 'MAX_WMARM_' + max_wmarm_key)
    if os.path.exists(source_folder):
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            print(f"Deleted existing folder {destination_folder}")
        shutil.copytree(source_folder, destination_folder)
        print(f"Replicated folder {max_wmarm_key} with max WMARM value: {max_wmarm_value}")
    else:
        print(f"Source folder {source_folder} does not exist.")
    return max_wmarm_key, max_wmarm_value

def run_uncertainty_analysis(wmarms,wmarm_db,max_wmarm,ds,refgrid):
    import conf
    conf.process_string = 'UNCERTAINTY ANALYSIS ON: '
    totiters = conf.uncertainty_nvalues * conf.uncertainty_iterations
    iti = 0
    uncertainty_db = pd.DataFrame(columns=['Original_WMARM', 'Averaged_WMARM', 'STDev','Min','Max','Values'])
    unc_path = os.path.join(conf.outpath, 'Uncertainty_analysis\\')
    if not os.path.exists(unc_path):
        os.mkdir(unc_path)
    flattened_values = wmarm_db.values.flatten()
    N = conf.uncertainty_nvalues
    values_to_check = pd.Series(flattened_values).nlargest(N).values
    assess_values = dict()
    for key in wmarms.keys():
        if wmarms[key] in values_to_check:
            assess_values[key] = wmarms[key]
    ki = 0
    for key in assess_values.keys():
        ki = ki +1
        uncertainty_db.loc[key, 'Original_WMARM'] = assess_values[key]
        th, l = decode_params(key)
        conf.exposure_percentile = th
        conf.lag = l
        conf.param_string = conf.process_string + str(th * 100) + '° perc - LAG: ' + str(l) + ' (' + str(ki) + '/' + str(N) + ') '
        for instance in ds:
            if instance.th == th and instance.lag == l:
                unc_struct = instance
                exp_threshold = unc_struct.exp_threshold
                exposed_days = unc_struct.exposed_days
                non_exposed_days = unc_struct.non_exposed_days
                incidence = unc_struct.incidence
                incidence_baseline = unc_struct.incidence_baseline
                incidence_differentials = unc_struct.incidence_differentials
        wmarms_vector = []
        for it in range(conf.uncertainty_iterations):
            iti = iti + 1
            conf.param_string = conf.process_string + ' - Iteration ' + str(iti) + '/' + str(totiters) + ' - processing = ' + str(iti/totiters*100) + '%\t'
            index_df = pd.DataFrame()
            for y in conf.years:
                #expth = exp_threshold[y]
                for yi in unc_struct.yearly_ds:
                    if yi.year == y:
                        weights = yi.weights
                        expdays = yi.expdays
                        nonexpdays = yi.nonexpdays
                # COMPUTATION OF VULNERABILITY INDEX
                yearly_index_df, yearly_permutations_r, yearly_permutations_p = compute_index_main(
                    incidence_differentials, weights, expdays, nonexpdays, y)
                index_df = pd.concat([index_df, yearly_index_df], axis=1)
            # POST-PROCESS INDEX ACROSS YEARS
            marm_db, marm = compute_marm(index_df)
            key = f"P{int(th * 100)}_L{l}"
            wmarm = compute_wmarm(refgrid, marm_db)
            wmarms_vector.append(wmarm)
        averaged_wmarm = np.mean(np.asarray(wmarms_vector))
        wmarm_db.loc[th * 100, l] = averaged_wmarm
        wmarms[key] = averaged_wmarm
        if averaged_wmarm >= max_wmarm:
            max_wmarm = averaged_wmarm
            #save_variables(exp_threshold, exposed_days, non_exposed_days, incidence, incidence_baseline)
            conf.max_wmarm_ds = unc_struct
        uncertainty_db.loc[key, 'Averaged_WMARM'] = averaged_wmarm
        uncertainty_db.loc[key, 'STDev'] = np.std(wmarms_vector)
        uncertainty_db.loc[key, 'Min'] = min(wmarms_vector)
        uncertainty_db.loc[key, 'Max'] = max(wmarms_vector)
        uncertainty_db.loc[key, 'Values'] = list(wmarms_vector)
    uncertainty_db.to_csv(unc_path + 'Uncertainty_analysis.csv')
    return wmarms, wmarm_db, max_wmarm