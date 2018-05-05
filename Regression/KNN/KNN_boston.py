# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:34:39 2018
K nearest neighbor with boston dataset
@author: shifuddin
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
'''
Load feature values as X and target as Y
'''
X,y = load_boston(return_X_y=True)

'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

knn_regressor = neighbors.KNeighborsRegressor(algorithm='kd_tree', n_neighbors= 4, weights = 'distance')
knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))









