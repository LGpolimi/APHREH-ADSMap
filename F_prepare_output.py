import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    # Prepare formatted output
    formatted_df = pd.DataFrame(index=index_df_sorted.index)
    for y in conf.years:
        proc_index = index_df_sorted.copy(deep=True)
        proc_index = proc_index.applymap(lambda x: "{:.2f}".format(x))
        formatted_df[str(y) + ' SCORE'] = proc_index.apply(
            lambda row: f"{row[str(y) + 'INDEX']} ({row[str(y) + 'CI_LOW']}|{row[str(y) + 'CI_HIGH']})", axis=1)
        formatted_df[str(y) + ' STD EFFECT'] = proc_index.loc[:, str(y) + 'STDEFF'].astype(float)

    # Add cumulative data
    cum_index_df['AVG_STDEFF'] = marm_db['AVG_STDEFF']
    cum_index_df_formatted['AVERAGED STD EFFECT'] = marm_db['AVG_STDEFF']

    return index_df_sorted, formatted_df, cum_index_df, cum_index_df_formatted

def generate_chart(wmarm_db):
    import conf

    # Extract data from wmarm_db
    th_values = wmarm_db.index.values
    l_values = wmarm_db.columns.values
    th, l = np.meshgrid(th_values, l_values)
    wmarm = wmarm_db.values.T  # Transpose to match the shape of th and l
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    surf = ax.plot_surface(th, l, wmarm, cmap='coolwarm', edgecolor='none')
    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # Set tick labels
    ax.set_xticks(th_values)
    ax.set_xticklabels([f'{x:.2f}' for x in conf.exposure_percentile_list], rotation=45, ha='right', va='top')
    ax.set_yticks(l_values)
    ax.set_yticklabels([str(x) for x in list(range(conf.lag_params[0], conf.lag_params[1] + 1, conf.lag_params[2]))],rotation=0)
    fig.subplots_adjust(bottom=0.2)
    # Set labels
    ax.set_xlabel('Exposure Percentile Threshold', labelpad=20)
    ax.set_ylabel('Lag Days', labelpad=10)
    ax.set_zlabel('WMARM', labelpad=10)
    # Set title
    ax.set_title('3D Surface Plot of WMARM*', pad=20)
    # Rotate
    ax.view_init(azim=160)
    # Extend the x-axis limits
    ax.set_xlim(th_values[0] - 1, th_values[-1] + 1)
    # Show plot
    plt.show()

    if conf.saveout == 1:
        plot_dir = os.path.join(conf.outpath, 'PLOT')
        os.makedirs(plot_dir, exist_ok=True)

        fig.savefig(
            os.path.join(plot_dir, 'WMARM.tiff'),
            bbox_inches='tight',  # trims extra whitespace, preserves labels
            pad_inches=0.2,  # slight padding to avoid cropping
            dpi=300  # higher resolution for publication-quality output
        )
        wmarm_db.index.rename('Exposure Threshold [%] VS Time Lag [days]', inplace=True)
        for col in wmarm_db.columns:
            wmarm_db.rename(columns={col: f'LAG {col}'}, inplace=True)
        wmarm_db.to_csv(conf.outpath + 'PLOT\\WMARM.csv')