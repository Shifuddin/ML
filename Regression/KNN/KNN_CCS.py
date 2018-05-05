# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:20:26 2018
knn with CCS dataset from uci
@author: shifuddin
"""
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from load_data import load_excel
'''
Load feature values as X and target as Y
here we read day dataset
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
X,y = load_excel(uri, 'Sheet1', 0,8, 8,9)

'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

knn_regressor = neighbors.KNeighborsRegressor(algorithm='auto', n_neighbors= 30, weights = 'uniform')
knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
