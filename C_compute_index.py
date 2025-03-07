import pandas as pd
import numpy as np
import conf
import random
from scipy.stats import rankdata, norm
from data_import import decode_datestr


def weighted_mannwhitneyu(exp_deltas, nonexp_deltas, exp_weights, nonexp_weights):
    import conf
    # Combine the data and weights
    n = len(exp_deltas)
    data = np.concatenate([exp_deltas, nonexp_deltas])
    weights = np.concatenate([exp_weights, nonexp_weights])
    groups = np.concatenate([np.ones(len(exp_deltas)), np.zeros(len(nonexp_deltas))])

    # Compute the ranks
    ranks = rankdata(data)

    # Apply weights to the ranks
    weighted_ranks = ranks * weights

    # Separate the ranks for the two groups
    rank_exp = weighted_ranks[groups == 1]
    rank_nonexp = weighted_ranks[groups == 0]

    # Compute the U statistic
    U = np.sum(rank_exp) - n * (n + 1) / 2

    # Compute the mean and standard deviation of U
    n1 = len(exp_deltas)
    n2 = len(nonexp_deltas)
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    # Compute the z-score
    z = (U - mean_U) / std_U

    # Compute the p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    # Compute the effect size
    r = U/(n**2)

    return U, p_value, r

def extract_sample(deltas,weights,expdays,nonexpdays):
    import conf
    sample_size = len(expdays)*2
    sampled_expdays = random.choices(expdays, k=sample_size)
    sampled_nonexpdays = random.choices(nonexpdays, k=sample_size)
    sampled_deltas_exp = deltas.loc[sampled_expdays].copy(deep=True)
    sampled_weights_exp = weights.loc[sampled_expdays, sampled_deltas_exp.columns].copy(deep=True)
    noise = np.random.uniform(-conf.random_noise, conf.random_noise, sampled_deltas_exp.shape)
    sampled_deltas_exp += sampled_deltas_exp * noise
    sampled_deltas_exp.reset_index(inplace=True,drop=True)
    sampled_weights_exp.reset_index(inplace=True,drop=True)
    sampled_deltas_nonexp = deltas.loc[sampled_nonexpdays].copy(deep=True)
    sampled_weights_nonexp = weights.loc[sampled_nonexpdays, sampled_deltas_nonexp.columns].copy(deep=True)
    noise = np.random.uniform(-conf.random_noise, conf.random_noise, sampled_deltas_nonexp.shape)
    sampled_deltas_nonexp += sampled_deltas_nonexp * noise
    sampled_deltas_nonexp.reset_index(inplace=True,drop=True)
    sampled_weights_nonexp.reset_index(inplace=True,drop=True)
    # DEBUGGING
    if sampled_deltas_nonexp[sampled_deltas_nonexp.isna().any(axis=1)].shape[0] > 0:
        print('Error: NaN values in sampled_deltas_nonexp')
    if sampled_deltas_exp[sampled_deltas_exp.isna().any(axis=1)].shape[0] > 0:
        print('Error: NaN values in sampled_deltas_exp')

    return sampled_deltas_exp, sampled_deltas_nonexp, sampled_weights_exp, sampled_weights_nonexp



def compute_results_arrays(deltas,weights,expdays,nonexpdays,y):
    import conf
    nit = conf.bootstrap_iterations
    bsas = list(deltas.columns.values)
    if 'DATE' in bsas:
        bsas.remove('DATE')
    if 'DATE_STR' in bsas:
        bsas.remove('DATE_STR')
    bsas = [int(bsa) for bsa in bsas]
    rbc_arrays = pd.DataFrame(index=bsas,columns=range(nit))
    pval_arrays = pd.DataFrame(index=bsas,columns=range(nit))
    totiters = len(bsas) * nit
    iti = 0
    for it in range(nit):
        outcol = it
        sampled_deltas_exp, sampled_deltas_nonexp, sampled_weights_exp, sampled_weights_nonexp = extract_sample(deltas,weights,expdays,nonexpdays)
        bi = 0
        for bsa in bsas:
            bi = bi + 1
            exp_deltas = sampled_deltas_exp[bsa]
            nonexp_deltas = sampled_deltas_nonexp[bsa]
            exp_weights = sampled_weights_exp[bsa]
            nonexp_weights = sampled_weights_nonexp[bsa]

            U, p, r = weighted_mannwhitneyu(exp_deltas, nonexp_deltas, exp_weights, nonexp_weights)
            rbc_arrays.loc[bsa,outcol] = r
            pval_arrays.loc[bsa,outcol] = p
            iti = iti + 1

            print(conf.param_string+'Performing Mann-Withney test for year '+str(y)+': working on permutation ' + str(it+1) + ' out of ' +str(nit) + ' for area ' + str(bsa) + ' (' + str(bi) + ' out of ' + str(len(bsas)) + ')\t Total processing = ' + str(iti) + ' out of ' + str(totiters) + ' (' + str(round(iti / totiters * 100, 2)), '%)')

    return rbc_arrays, pval_arrays

def linear_int_ci(df,threshval,distcol):
    import conf
    df[distcol] = pd.to_numeric(df[distcol], errors='coerce')
    if df[df[distcol] > 0].shape[0]  > 0 and df[df[distcol] < 0].shape[0] > 0:
        lowest_pos = df[df[distcol] > 0].nsmallest(1, distcol).index[0]
        lowest_neg = df[df[distcol] < 0].nlargest(1, distcol).index[0]
        ww = df.loc[lowest_neg, 'CUM_WEIGHT']
        ww1 = df.loc[lowest_pos, 'CUM_WEIGHT']
        rww = df.loc[lowest_neg, 'RBC']
        rww1 = df.loc[lowest_pos, 'RBC']
        ci_val = rww + (((threshval-ww)/(ww1-ww))*(rww1-rww))
    elif df[df[distcol] > 0].shape[0]  == 0 and df[df[distcol] < 0].shape[0] > 0:
        lowest_neg = df[df[distcol] < 0].nlargest(1, distcol).index[0]
        ci_val = df.loc[lowest_neg, 'RBC']
    elif df[df[distcol] > 0].shape[0]  > 0 and df[df[distcol] < 0].shape[0] == 0:
        lowest_pos = df[df[distcol] > 0].nsmallest(1, distcol).index[0]
        ci_val = df.loc[lowest_pos, 'RBC']
    return ci_val

def compute_weighted_ci(full_df):
    import conf
    df = full_df[['RBC','WEIGHT']].copy(deep=True)
    df.sort_values(by='RBC', ascending=True, inplace=True)
    df.reset_index(inplace=True,drop=True)
    cumulated_weight = df['WEIGHT'].sum()
    df['CUM_WEIGHT'] = df['WEIGHT'].cumsum()
    weight_low = 0.025 * cumulated_weight
    weight_high = 0.975 * cumulated_weight
    df['DIST_from_LOW'] = df['CUM_WEIGHT'] - weight_low
    df['DIST_from_HIGH'] = df['CUM_WEIGHT'] - weight_high
    gotlower = 0
    gothgiher = 0
    if 0 in df['DIST_from_LOW'].values:
        gotlower = 1
        lower_ci = df.loc[df['DIST_from_LOW'] == 0,'RBC'].values[0]
    if 0 in df['DIST_from_HIGH'].values:
        gothgiher = 1
        higher_ci = df.loc[df['DIST_from_HIGH'] == 0,'RBC'].values[0]
    if gotlower == 0:
        lower_ci = linear_int_ci(df,weight_low,'DIST_from_LOW')
    if gothgiher == 0:
        higher_ci = linear_int_ci(df,weight_high,'DIST_from_HIGH')
    return lower_ci, higher_ci

def compute_results(rbc_arrays,pval_arrays,y):
    import conf
    results = pd.DataFrame(index=rbc_arrays.index)
    weights = 1 - pval_arrays
    iti = 0
    totiters = len(rbc_arrays.index)
    for bsa in rbc_arrays.index:
        bsa_df = pd.DataFrame(index=rbc_arrays.columns)
        bsa_df['RBC'] = rbc_arrays.loc[bsa]
        bsa_df['WEIGHT'] = weights.loc[bsa]
        bsa_df['WEIGHTED_RBC'] = bsa_df['RBC'] * bsa_df['WEIGHT']
        weighted_average = bsa_df['WEIGHTED_RBC'].sum() / bsa_df['WEIGHT'].sum()
        results.loc[bsa, str(y) + 'INDEX'] = weighted_average
        lower_ci, higher_ci = compute_weighted_ci(bsa_df)
        results.loc[bsa, str(y) + 'CI_LOW'] = lower_ci
        results.loc[bsa, str(y) + 'CI_HIGH'] = higher_ci
        iti += 1
        print(conf.param_string+'Averaging values to compute index and confidence intervals for year '+str(y)+': working on area ' + str(bsa) + ' ('+ str(iti) + ' out of ' + str(totiters) + ' - total processing = ' + str(round(iti / totiters * 100, 2)), '%)')

    return results


def compute_index_main(deltas,weights,expdays,nonexpdays,y):
    import conf
    if 'DATE' in deltas.columns:
        deltas.set_index('DATE',inplace=True,drop=True)
    weights['DATE_STR'] = weights.index
    weights = decode_datestr(weights)
    weights.set_index('DATE',inplace=True,drop=True)
    expdays = [day for day in expdays if pd.to_datetime(day) in deltas.index]
    nonexpdays = [day for day in nonexpdays if pd.to_datetime(day) in deltas.index]
    rbc_arrays, pval_arrays = compute_results_arrays(deltas, weights, expdays, nonexpdays,y)
    results = compute_results(rbc_arrays, pval_arrays,y)
    results.index.rename(conf.geoid,inplace=True)
    rbc_arrays.index.rename(conf.geoid, inplace=True)
    pval_arrays.index.rename(conf.geoid, inplace=True)

    return results, rbc_arrays, pval_arrays