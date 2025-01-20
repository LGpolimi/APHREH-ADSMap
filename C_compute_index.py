import pandas as pd
import numpy as np
import conf
import random
from scipy.stats import rankdata, norm
from data_import import decode_datestr


def weighted_mannwhitneyu(exp_deltas, nonexp_deltas, exp_weights, nonexp_weights):
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

            print('Performing Mann-Withney test for year '+str(y)+': working on permutation ' + str(it+1) + ' out of ' +str(nit) + ' for area ' + str(bsa) + ' (' + str(bi) + ' out of ' + str(len(bsas)) + ')\t Total processing = ' + str(iti) + ' out of ' + str(totiters) + ' (' + str(round(iti / totiters * 100, 2)), '%)')

    return rbc_arrays, pval_arrays



def compute_results(rbc_arrays, pval_arrays,y):

    results = pd.DataFrame(index=rbc_arrays.index,columns=['INDEX'])
    weights = 1 - pval_arrays
    iti = 0
    totiters = len(rbc_arrays.columns)
    for bsa in rbc_arrays.index:
        bsa_df = pd.DataFrame(index=rbc_arrays.columns)
        bsa_df['RBC'] = rbc_arrays.loc[bsa]
        bsa_df['WEIGHT'] = weights.loc[bsa]
        bsa_df['WEIGHTED_RBC'] = bsa_df['RBC'] * bsa_df['WEIGHT']
        weighted_average = bsa_df['WEIGHTED_RBC'].sum() / bsa_df['WEIGHT'].sum()
        results.loc[bsa,'INDEX'] = weighted_average
        iti += 1
        print('Averaging values to compute index for year '+str(y)+': ' + str(iti) + ' out of ' + str(totiters), ' (', str(round(iti / totiters * 100, 2)), '%)')

    return results


def compute_index_main(deltas,weights,expdays,nonexpdays,y):
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