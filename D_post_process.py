import pandas as pd
import numpy as np
import conf

def cumulate_across_years(index_df):
    import conf
    suffixes = ['INDEX','CI_LOW','CI_HIGH']
    var_df = pd.DataFrame(index=index_df.index)
    var_df_formatted = pd.DataFrame(index=index_df.index)
    for row in index_df.iterrows():
        formatted_strings = list()
        for suffix in suffixes:
            cols = [col for col in index_df.columns if suffix in col]
            values = row[1][cols].values
            med = np.median(values)
            q25 = np.percentile(values,25)
            q75 = np.percentile(values,75)
            iqr = q75 - q25
            var_df.loc[row[0],suffix+'_MEDIAN'] = med
            var_df.loc[row[0],suffix+'_Q25'] = q25
            var_df.loc[row[0],suffix+'_Q75'] = q75
            var_df.loc[row[0],suffix+'_IQR'] = iqr
            formatted_strings.append("{:.2f}±{:.2f}".format(med, iqr))
        formatted_string = formatted_strings[0] + ' (' + formatted_strings[1] + '|' + formatted_strings[2] + ')'
        var_df_formatted.loc[row[0],'INDEX'] = formatted_string
    return var_df, var_df_formatted