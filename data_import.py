import pandas as pd
import conf

def decode_datestr(db):
    db['DATE'] = pd.to_datetime(db['DATE_STR'], format='%y%m%d')
    db.set_index('DATE_STR',inplace=True,drop=True)
    return db

def import_data():
    exposure = pd.read_csv(conf.dspath + conf.exposure_db_name,low_memory=False)
    outcome = pd.read_csv(conf.dspath + conf.outcome_db_name,low_memory=False)
    exposure['DATE_STR'] = exposure['DATE_STR'].astype(str)
    outcome['DATE_STR'] = outcome['DATE_STR'].astype(str)
    exposure.columns = [int(col) if col.isdigit() else col for col in exposure.columns]
    outcome.columns = [int(col) if col.isdigit() else col for col in outcome.columns]
    exposure = decode_datestr(exposure)
    outcome = decode_datestr(outcome)

    refgrid = pd.read_csv(conf.dspath + conf.reference_geo_level+'.csv',low_memory=False)
    refgrid[conf.geoid] = refgrid[conf.geoid].astype(int)
    for col in refgrid.columns.values:
        if 'POP' in col:
            refgrid[col] = refgrid[col].astype(int)

    return exposure, outcome, refgrid

def slice_data(db,years,zones):
    db = db.loc[db['DATE'].dt.year.isin(years)].copy(deep=True)
    if 'ALL' not in zones:
        if 'DATE' not in zones:
            zones.append('DATE')
        db = db[db.columns.intersection(zones)].copy(deep=True)
    return db

def uniform_data(exposure,outcome):
    exp_dates = exposure['DATE'].unique()
    out_dates = outcome['DATE'].unique()
    common_dates = list(set(exp_dates).intersection(out_dates))
    exp_zones = list(exposure.columns.values)
    out_zones = list(outcome.columns.values)
    common_zones = list(set(exp_zones).intersection(out_zones))
    exposure = exposure.loc[exposure['DATE'].isin(common_dates)].copy(deep=True)
    outcome = outcome.loc[outcome['DATE'].isin(common_dates)].copy(deep=True)
    exposure = exposure[exposure.columns.intersection(common_zones)].copy(deep=True)
    outcome = outcome[outcome.columns.intersection(common_zones)].copy(deep=True)

    return exposure, outcome