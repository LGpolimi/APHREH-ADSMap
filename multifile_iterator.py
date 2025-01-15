import pandas as pd
import datetime as dt
from dbfread import DBF

def import_iterate_time(granularity,extremes,folder,file_root,file_extension,file_suffix="foo",dt_format="foo",import_params="foo",chunks_mode="foo"):

    delimiter = ','
    encoding = 'utf-8'
    on_bad_lines = 'skip'
    sep = delimiter
    if import_params != 'foo':
        for key in import_params.keys():
            if key == 'delimiter':
                delimiter = import_params[key]
            if key == 'encoding':
                encoding = import_params[key]
            if key == 'on_bad_lines':
                on_bad_lines = import_params[key]
            if key == 'sep':
                sep = import_params[key]

    if chunks_mode == 'foo':
        chunks_mode = 0

    if granularity == 'year':
        lower_limit = extremes[0].year
        upper_limit = extremes[1].year
        cycler = range(lower_limit,upper_limit+1)
    else:
        lower_limit = extremes[0]
        upper_limit = extremes[1]
        cycler = list()
        clock = dt.datetime(extremes[0].year(),extremes[0].month(),extremes[0].day())
        while clock <= dt.datetime(extremes[1].year(),extremes[1].month(),extremes[1].day()):
            if granularity == 'month':
                curr_month = int(clock.month())
                if curr_month not in cycler:
                    cycler.append(clock)
            else:
                cycler.append(clock)
            if granularity == 'day':
                clock = clock + dt.timedelta(days=1)
            if granularity == 'hour':
                clock = clock + dt.timedelta(hours=1)

    merge_flag = 0

    for i in cycler:
        if granularity != 'year':
            if dt_format == "foo":
                raise ValueError("Please specify a valid datetime format.")
            date_string = i.strftime(dt_format)
        else:
            date_string = str(i)
        if file_suffix != "foo":
            file_name = folder + file_root + date_string + file_suffix + file_extension
            print('Importing ' + file_root + date_string + file_suffix + file_extension)
        else:
            file_name = folder + file_root + date_string + file_extension
            print('Importing ' + file_root + date_string + file_extension)
        if file_extension == '.csv':
            if merge_flag == 0:
                try:
                    if chunks_mode == 0:
                        df = pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding)
                    else:
                        chunklist = list()
                        for chunk in  pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding,
                                                 chunksize=500000):
                            chunklist.append(chunk)
                        df = pd.concat(chunklist)
                    merge_flag = 1
                except:
                    print('Error importing ' + file_name + ' or file not found.')
            else:
                try:
                    if chunks_mode == 0:
                        df = pd.concat([df,pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding)])
                    else:
                        chunklist = list()
                        for chunk in  pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding,
                                                 chunksize=500000):
                            chunklist.append(chunk)
                        df = pd.concat([df,pd.concat(chunklist)])
                except:
                    print('Error importing ' + file_name + ' or file not found.')
        if file_extension == '.tab' or file_extension == '.txt':
            if merge_flag == 0:
                try:
                    if chunks_mode == 0:
                        df = pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding,on_bad_lines=on_bad_lines)
                    else:
                        chunklist = list()
                        for chunk in  pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding,on_bad_lines=on_bad_lines,
                                                 chunksize=500000):
                            chunklist.append(chunk)
                        df = pd.concat(chunklist)
                    merge_flag = 1
                except:
                    print('Error importing ' + file_name + ' or file not found.')
            else:
                try:
                    if chunks_mode == 0:
                        df = pd.concat([df,pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding,on_bad_lines=on_bad_lines)])
                    else:
                        chunklist = list()
                        for chunk in  pd.read_csv(file_name,sep=delimiter,low_memory=False,encoding=encoding,on_bad_lines=on_bad_lines,
                                                 chunksize=500000):
                            chunklist.append(chunk)
                        df = pd.concat([df,pd.concat(chunklist)])
                except:
                    print('Error importing ' + file_name + ' or file not found.')
        elif file_extension == '.xlsx':
            if merge_flag == 0:
                try:
                    df = pd.read_excel(file_name)
                    merge_flag = 1
                except:
                    print('Error importing ' + file_name + ' or file not found.')
            else:
                try:
                    df = pd.concat([df,pd.read_excel(file_name)])
                except:
                    print('Error importing ' + file_name + ' or file not found.')
        elif file_extension == '.dbf':
            if merge_flag == 0:
                try:
                    df = pd.DataFrame(DBF(file_name))
                    merge_flag = 1
                except:
                    print('Error importing ' + file_name + ' or file not found.')
            else:
                try:
                    df =  df = pd.concat([df,pd.DataFrame(DBF(file_name))])
                except:
                    print('Error importing ' + file_name + ' or file not found.')

    return(df)