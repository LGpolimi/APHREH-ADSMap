import pandas as pd

def filter_db(input_db,axis,field_name,values):

    if axis == 'columns':
        axis = 1
    elif axis == 'rows':
        axis = 0
    if type(values)!=list:
        values = [values]
    if axis == 1:
        if field_name in list(input_db.columns.values):
            output_db = input_db.loc[input_db[field_name].isin(values)].copy(deep=True)
        else:
            raise ValueError("Field name not found in database.")
    elif axis == 0:
        if field_name in list(input_db.index.values):
            output_db = input_db.loc[input_db.index.isin(values)].copy(deep=True)
        else:
            raise ValueError("Field name not found in database.")
    return output_db