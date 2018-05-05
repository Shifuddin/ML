# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:29:50 2018

@author: shifuddin
"""

import pandas as pd
import zipfile
import urllib

#uci_base_uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
def load_zip(uri, csv_file_name, X_start, X_end, y_start, y_end):
    response = urllib.request.urlopen(uri).read()
    with open('data.zip', 'wb') as f:
        f.write(response)

    zf = zipfile.ZipFile('data.zip') 
    df = pd.read_csv(zf.open(csv_file_name))    
    
    X = df.iloc[:, X_start:X_end].values
    y = df.iloc[:, y_start:y_end].values
    
    return X, y

def load_excel(uri, sheetname, X_start, X_end, y_start, y_end):
    df = pd.read_excel(uri, sheet_name=sheetname)
    X = df.iloc[:, X_start:X_end].values
    y = df.iloc[:, y_start:y_end].values
    
    return X, y


def load_csv(uri, separator, X_start, X_end, y_start, y_end):
    df = pd.read_csv(uri, sep = separator)
    X = df.iloc[:, X_start:X_end].values
    y = df.iloc[:, y_start:y_end].values
    
    return X, y
