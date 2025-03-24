import pandas as pd

def data_loader(fp, skiprows, na_value=-999):
    with open(fp, 'r') as f:
        for line in f:
            if 'fields=' in line:
                break
    columns = line.strip().strip('/fields=').split(',')

    return pd.read_csv(
        fp, names=columns, skiprows=skiprows, na_values=-999)