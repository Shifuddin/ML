# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:29:50 2018

@author: shifuddin
"""

import pandas as pd
import zipfile
import urllib

#uci_base_uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
def load_zip(uri, csv_file_name, X_start, X_end, y_start, y_end, return_X_y):
    response = urllib.request.urlopen(uri).read()
    with open('data.zip', 'wb') as f:
        f.write(response)

    zf = zipfile.ZipFile('data.zip') 
    df = pd.read_csv(zf.open(csv_file_name))    
    
    if return_X_y == True:
        X = df.iloc[:, X_start:X_end].values
        y = df.iloc[:, y_start:y_end].values
        return X, y
    else:
        return df

def load_excel(uri, sheetname, X_start, X_end, y_start, y_end):
    df = pd.read_excel(uri, sheet_name=sheetname)
    X = df.iloc[:, X_start:X_end].values
    y = df.iloc[:, y_start:y_end].values
    
    return X, y


def load_csv(uri, separator, X_start, X_end, y_start, y_end, return_X_y):
    df = pd.read_csv(uri, sep = separator)
    
    if return_X_y == True:
        X = df.iloc[:, X_start:X_end].values
        y = df.iloc[:, y_start:y_end].values
        return X, y
    else:
        return df

def load_X_y(dataframe, X_start, X_end, y_start, y_end):
    X = dataframe.iloc[:, X_start:X_end].values
    y = dataframe.iloc[:, y_start:y_end].values
    return X, y