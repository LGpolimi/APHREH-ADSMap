import pandas as pd

import os
import pandas as pd

# Define the path and prefix
path = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2024 07 - EnvironmentalEpidemiology\\24 10 - Vulnerability Model\\25 03 - V2\\Set_up_III\\datasource\\processing\\'
prefix = 'EVT_MISS_PAZ_'
output_name = 'EVT_MISS_PAZ_1520_LMB3A.csv'

# List all files in the directory
all_files = os.listdir(path)

# Filter files that start with the prefix and have a .csv extension
csv_files = [f for f in all_files if f.startswith(prefix) and f.endswith('.csv')]

# Read and concatenate all CSV files
dataframes = [pd.read_csv(os.path.join(path, file)) for file in csv_files]
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(os.path.join(path, output_name), index=False)