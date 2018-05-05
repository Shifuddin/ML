# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:51:18 2018
decission tree with banknote
@author: shifuddin
"""

from load_data import load_csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
'''
Load feature values as X and target as Y
here we read day dataset
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt'
X,y = load_csv(uri, ',', 0,9,9,10, True)
y = pd.get_dummies(y.ravel(), drop_first=True)


'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

'''
Feature scaling 
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc_y = StandardScaler()
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

'''
Fit DecisionTreeRegressor with Bike Day data
'''
regressor = MLPRegressor(hidden_layer_sizes=(100,50))
regressor.fit(X_train, y_train)

'''
Predicting result
'''
y_pred = regressor.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
