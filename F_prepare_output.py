import pandas as pd
import conf

def merge_relevant_info(marm_db,cum_index_df,cum_index_df_formatted):
    import conf
    # Rearrange columns
    columns = list(marm_db.columns)
    new_columns = []
    added_columns = set()
    for col in columns:
        if col.endswith('CI_HIGH'):
            year = col[:4]
            new_columns.append(col)
            added_columns.add(col)
            steff_col = year + 'STDEFF'
            if steff_col in columns and steff_col not in added_columns:
                new_columns.append(steff_col)
                added_columns.add(steff_col)
        elif not col.endswith('STDEFF') and col not in added_columns:
            new_columns.append(col)
            added_columns.add(col)
    for col in columns:
        if col.endswith('STDEFF') and col not in added_columns:
            new_columns.append(col)
            added_columns.add(col)
    index_df_sorted = marm_db[new_columns].copy(deep=True)
    index_df_sorted.set_index(conf.geoid, inplace=True)

    # Prepare formatted output
    formatted_df = pd.DataFrame(index=index_df_sorted.index)
    for y in conf.years:
        proc_index = index_df_sorted.copy(deep=True)
        proc_index = proc_index.applymap(lambda x: "{:.2f}".format(x))
        formatted_df[str(y) + ' SCORE'] = proc_index.apply(
            lambda row: f"{row[str(y) + 'INDEX']} ({row[str(y) + 'CI_LOW']}|{row[str(y) + 'CI_HIGH']})", axis=1)
        formatted_df[str(y) + ' STD EFFECT'] = proc_index.loc[:, str(y) + 'STDEFF'].astype(float)

    # Add cumulative data
    marm_db_indexed = marm_db.set_index(conf.geoid)
    cum_index_df['AVG_STDEFF'] = marm_db_indexed['AVG_STDEFF']
    cum_index_df_formatted['AVERAGED STD EFFECT'] = marm_db_indexed['AVG_STDEFF']

    return index_df_sorted, formatted_df, cum_index_df, cum_index_df_formatted