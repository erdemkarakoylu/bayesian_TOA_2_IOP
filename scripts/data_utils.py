import pandas as pd
import numpy as np

def data_loader(
        fp, na_value=-999, 
        date_time_cols=['year', 'month' ,'day', 'hour', 'minute', 'second']):
    """
    Load data from a file with a NOMAD-type header
    """
    with open(fp, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'fields=' in line:
            columns = line.strip().strip('/fields=').split(',')
        if '/end_header' in line:
            skiprows = i + 1
    df= pd.read_csv(
        fp, names=columns, skiprows=skiprows, na_values=na_value)
    df.insert(0, 'datetime', pd.to_datetime(df[date_time_cols]))
    return df.drop(date_time_cols, axis=1)
