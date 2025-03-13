import pandas as pd
from netCDF4 import Dataset
import os
import calendar
import datetime as dt

rootpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\25 03 - V2\\Set_up_III\\datasource\\'
inpath = rootpath + 'CAMS\\'
outpath = rootpath + 'analysis_ready\\'
pollutant = 'PM25'
prefix = 'daily_'
extension = '.nc'
grid_geoid = 'LMB1D'
grid = pd.read_csv(rootpath + 'processing\\'+grid_geoid + '.csv')
outname = 'exposure_data.csv'
years = [2015,2016,2017,2018,2019,2020]
months = [1,2,3,4,5,6,7,8,9,10,11,12]
start_date = dt.datetime(years[0], months[0], 1)
last_day = calendar.monthrange(years[-1], months[-1])[1]
end_date = dt.datetime(years[-1], months[-1], last_day)
tot_days = (end_date - start_date).days

# Initialize an empty DataFrame
valid_cells = list(grid[grid_geoid+'_IDor'].unique())
indexes = []
currday = start_date - dt.timedelta(days=1)
for d in range(tot_days+1):
    currday = currday + dt.timedelta(days=1)
    y_str = str(currday.year)[-2:]
    m_str = f"{currday.month:02}"
    d_str = f"{currday.day:02}"
    indexes.append(f"{y_str}{m_str}{d_str}")
out_df = pd.DataFrame(index=indexes)

# Loop through each year and month to read the data
iti = 0
for year in years:
    for month in months:
        # Open the dataset
        file_path = f"{inpath}{prefix}{pollutant}_{year}_{month:02}{extension}"
        ds = Dataset(file_path)

        # Extract the PM25 data
        full_poll_data = ds.variables['pollution'][:]

        # Reshape the data to 2D (time, cells)
        num_days, num_rows, num_cols = full_poll_data.shape
        totiters = tot_days * num_rows * num_cols
        for d in range(num_days):
            date_str = [f"{str(year)[-2:]}{month:02}{d+1:02}"]
            polldata = full_poll_data [d, :, :]
            for i in range(num_rows+1):
                for j in range(num_cols+1):
                    iti = iti + 1
                    print(f"Processing iteration {iti} / {totiters} - ({(iti/totiters)*100:.2f}%)")
                    cellid = j + (i * num_cols) + 1
                    if cellid in valid_cells:
                        custom_id = grid.loc[grid[grid_geoid+'_IDor'] == cellid, grid_geoid+'_IDcu'].values[0]
                        poll_value = polldata[i, j]
                        out_df.loc[date_str, custom_id] = poll_value

out_df.index.name = 'DATE_STR'

# Save the DataFrame to a CSV file (optional)
output_file = outpath + outname
out_df.to_csv(output_file)

print("Data extraction complete.")

br = 1